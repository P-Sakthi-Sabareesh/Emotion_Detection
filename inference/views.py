import ipaddress
import json
import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from django.conf import settings
from django.core.files.base import ContentFile
from django.db import DatabaseError
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST, require_http_methods

from history.models import PredictionRecord
from inference.auth import is_authenticated, require_api_auth
from inference.rate_limit import allow_request
from inference.services import PredictionInputError, inference_service
from inference.validation import validate_image_bytes

logger = logging.getLogger(__name__)

_sample_cache: dict[str, tuple[float, Any]] = {}
_sample_cache_ttl_seconds = 60

_CONSENT_TRUE_VALUES = {"1", "true", "yes", "on", "granted"}


# ---------- helpers ----------------------------------------------------------


def _client_ip(request: HttpRequest) -> str | None:
    forwarded = request.META.get("HTTP_X_FORWARDED_FOR")
    if forwarded and settings.FER_TRUST_X_FORWARDED_FOR:
        return forwarded.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR")


def _anonymize_ip(ip: str | None) -> str | None:
    if not ip or not settings.FER_IP_ANONYMIZE:
        return ip
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return None
    if isinstance(addr, ipaddress.IPv4Address):
        network = ipaddress.ip_network(f"{ip}/24", strict=False)
    else:
        network = ipaddress.ip_network(f"{ip}/48", strict=False)
    return str(network.network_address)


def _consent_granted(request: HttpRequest, body: dict | None = None) -> bool:
    if not settings.FER_REQUIRE_CONSENT:
        return True
    candidates: list[str] = []
    header = request.META.get("HTTP_X_CONSENT")
    if header:
        candidates.append(header)
    for source in (request.POST, request.GET):
        value = source.get("consent") if hasattr(source, "get") else None
        if value:
            candidates.append(str(value))
    if body:
        value = body.get("consent")
        if value is not None:
            candidates.append(str(value))
    return any(str(v).strip().lower() in _CONSENT_TRUE_VALUES for v in candidates)


def _enforce_rate_limit(request: HttpRequest, scope: str, limit: int) -> JsonResponse | None:
    ip = _client_ip(request) or "unknown"
    user_id = None
    user = getattr(request, "user", None)
    if user is not None and getattr(user, "is_authenticated", False):
        user_id = getattr(user, "id", None)
    key = f"{scope}:{user_id or ip}"
    if not allow_request(key, limit):
        return JsonResponse(
            {"error": "rate_limited", "retry_after_seconds": 60},
            status=429,
            headers={"Retry-After": "60"},
        )
    return None


def _preflight_content_length(request: HttpRequest) -> JsonResponse | None:
    raw = request.META.get("CONTENT_LENGTH")
    if not raw:
        return None
    try:
        declared = int(raw)
    except (TypeError, ValueError):
        return JsonResponse({"error": "invalid_content_length"}, status=400)
    max_bytes = settings.FER_MAX_UPLOAD_MB * 1024 * 1024
    if declared > max_bytes:
        return JsonResponse(
            {"error": "payload_too_large", "max_bytes": max_bytes},
            status=413,
            headers={"Connection": "close"},
        )
    return None


def _consent_error() -> JsonResponse:
    return JsonResponse(
        {
            "error": "consent_required",
            "message": (
                "Biometric analysis requires explicit consent. "
                "Resend with consent=true (body field, query string, or X-Consent header)."
            ),
        },
        status=451,
    )


def _persist_prediction(
    request: HttpRequest,
    source_mode: str,
    result: Any,
    uploaded_file_name: str | None = None,
    image_content: bytes | None = None,
    consent_granted: bool = False,
) -> PredictionRecord | None:
    if not settings.FER_STORE_PREDICTIONS:
        return None
    record = PredictionRecord(
        source_mode=source_mode,
        predicted_emotion=result.emotion,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
        model_version=result.model_version,
        model_loaded=getattr(result, "model_loaded", True),
        degraded=getattr(result, "degraded", False),
        raw_scores=result.scores,
        image_sha256=result.image_sha256,
        request_ip_prefix=_anonymize_ip(_client_ip(request)),
        user_agent=(request.META.get("HTTP_USER_AGENT", "") or "")[:255],
        consent_granted=consent_granted,
    )
    if (
        settings.FER_STORE_IMAGES
        and uploaded_file_name
        and image_content
        and consent_granted
    ):
        suffix = Path(uploaded_file_name).suffix.lower()[:10]
        safe_name = f"{uuid4().hex}{suffix if suffix else '.jpg'}"
        record.image.save(safe_name, ContentFile(image_content), save=False)
    try:
        record.save()
    except DatabaseError:
        logger.exception("Failed to persist prediction record")
        return None
    return record


def _result_payload(record: PredictionRecord | None, result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "emotion": result.emotion,
        "confidence": round(result.confidence, 4),
        "scores": result.scores,
        "latency_ms": result.latency_ms,
        "model_version": result.model_version,
        "model_loaded": getattr(result, "model_loaded", True),
        "degraded": getattr(result, "degraded", False),
    }
    if record is not None:
        payload["id"] = record.id
    return payload


# ---------- health ----------------------------------------------------------


@require_GET
def healthz(_: HttpRequest) -> JsonResponse:
    return JsonResponse({"status": "ok"})


@require_GET
def readyz(_: HttpRequest) -> JsonResponse:
    status = inference_service.load_status()
    from django.db import connection  # local import: keeps import graph light

    db_ok = True
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
    except Exception:
        db_ok = False
    ready = db_ok and (
        status["loaded"] or not settings.FER_REQUIRE_SIGNED_ARTIFACT
    )
    body: dict[str, Any] = {
        "status": "ready" if ready else "degraded",
        "model": status,
        "database": "ok" if db_ok else "error",
    }
    if not status.get("loaded"):
        body["hint"] = (
            "Model artifact missing. Run `python scripts/bootstrap_model.py` "
            "with FER_MODEL_SIGNING_KEY set to download + sign the FER ViT."
        )
    return JsonResponse(body, status=200 if ready else 503)


# ---------- page views (session-authenticated) -----------------------------


@require_http_methods(["GET", "POST"])
def upload_prediction_page(request: HttpRequest) -> HttpResponse:
    context: dict[str, Any] = {"consent_required": settings.FER_REQUIRE_CONSENT}
    if request.method == "POST":
        if settings.FER_REQUIRE_AUTH and not is_authenticated(request):
            context["error"] = "Authentication required."
            return render(request, "inference/upload.html", context, status=401)
        limited = _enforce_rate_limit(
            request, "predict-upload-page", settings.FER_API_RATE_LIMIT_IMAGE_PER_MIN
        )
        if limited is not None:
            context["error"] = "Rate limit exceeded. Please retry in a minute."
            return render(request, "inference/upload.html", context, status=429)
        oversized = _preflight_content_length(request)
        if oversized is not None:
            context["error"] = "File is too large."
            return render(request, "inference/upload.html", context, status=413)
        if not _consent_granted(request):
            context["error"] = (
                "You must tick the consent checkbox before running biometric analysis."
            )
            return render(request, "inference/upload.html", context, status=451)

        image = request.FILES.get("image")
        if not image:
            context["error"] = "Please select an image."
        else:
            try:
                image_bytes = image.read()
                validate_image_bytes(image_bytes)
                result = inference_service.predict_from_bytes(image_bytes)
                _persist_prediction(
                    request,
                    PredictionRecord.SourceMode.UPLOAD,
                    result,
                    uploaded_file_name=image.name,
                    image_content=image_bytes,
                    consent_granted=True,
                )
                context["result"] = result
                context["sorted_scores"] = sorted(
                    result.scores.items(), key=lambda item: item[1], reverse=True
                )
                if getattr(result, "degraded", False):
                    context["warning"] = (
                        "Model not loaded — prediction served by heuristic fallback."
                    )
            except PredictionInputError as exc:
                context["error"] = str(exc)
            except Exception:
                logger.exception("Upload prediction failed")
                context["error"] = "Prediction failed. Please try again."
    return render(request, "inference/upload.html", context)


@require_GET
def live_prediction_page(request: HttpRequest) -> HttpResponse:
    return render(
        request,
        "inference/live.html",
        {"consent_required": settings.FER_REQUIRE_CONSENT},
    )


# ---------- JSON APIs ------------------------------------------------------


@csrf_exempt
@require_POST
@require_api_auth
def predict_image_api(request: HttpRequest) -> JsonResponse:
    oversized = _preflight_content_length(request)
    if oversized is not None:
        return oversized
    limited = _enforce_rate_limit(
        request, "predict-image", settings.FER_API_RATE_LIMIT_IMAGE_PER_MIN
    )
    if limited is not None:
        return limited
    if not _consent_granted(request):
        return _consent_error()

    image = request.FILES.get("image")
    if not image:
        return JsonResponse({"error": "image is required"}, status=400)
    try:
        image_bytes = image.read()
        validate_image_bytes(image_bytes)
        result = inference_service.predict_from_bytes(image_bytes)
        record = _persist_prediction(
            request,
            PredictionRecord.SourceMode.UPLOAD,
            result,
            uploaded_file_name=image.name,
            image_content=image_bytes,
            consent_granted=True,
        )
    except PredictionInputError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception:
        logger.exception("predict_image_api failed")
        return JsonResponse({"error": "prediction_failed"}, status=500)
    return JsonResponse(_result_payload(record, result))


@csrf_exempt
@require_POST
@require_api_auth
def predict_live_frame_api(request: HttpRequest) -> JsonResponse:
    if len(request.body or b"") > settings.FER_MAX_BASE64_BYTES:
        return JsonResponse(
            {"error": "payload_too_large"}, status=413, headers={"Connection": "close"}
        )
    limited = _enforce_rate_limit(
        request, "predict-live", settings.FER_API_RATE_LIMIT_LIVE_PER_MIN
    )
    if limited is not None:
        return limited
    try:
        body = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "invalid_json"}, status=400)

    if not _consent_granted(request, body=body):
        return _consent_error()

    frame = body.get("image")
    if not frame:
        return JsonResponse({"error": "image is required"}, status=400)
    try:
        result = inference_service.predict_from_base64(frame)
        record = _persist_prediction(
            request,
            PredictionRecord.SourceMode.LIVE,
            result,
            consent_granted=True,
        )
    except PredictionInputError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception:
        logger.exception("predict_live_frame_api failed")
        return JsonResponse({"error": "prediction_failed"}, status=500)
    return JsonResponse(_result_payload(record, result))


_SAMPLE_MAP = {
    "sample-1": "samples/sample-1.jpg",
    "sample-2": "samples/sample-2.jpg",
    "sample-3": "samples/sample-3.jpg",
    "sample-4": "samples/sample-4.jpg",
    "sample-5": "samples/sample-5.jpg",
}


@csrf_exempt
@require_POST
@require_api_auth
def predict_sample_api(request: HttpRequest, sample_name: str) -> JsonResponse:
    limited = _enforce_rate_limit(
        request, "predict-sample", settings.FER_API_RATE_LIMIT_SAMPLE_PER_MIN
    )
    if limited is not None:
        return limited

    path = _SAMPLE_MAP.get(sample_name)
    if not path:
        return JsonResponse({"error": "unknown_sample"}, status=404)
    sample_root = (Path(settings.BASE_DIR) / "static").resolve()
    sample_path = (sample_root / path).resolve()
    if sample_root not in sample_path.parents and sample_path != sample_root:
        return JsonResponse({"error": "unknown_sample"}, status=404)

    try:
        now = time.time()
        cached = _sample_cache.get(sample_name)
        if cached and now - cached[0] < _sample_cache_ttl_seconds:
            result = cached[1]
        else:
            with sample_path.open("rb") as file_obj:
                content = file_obj.read()
            result = inference_service.predict_from_bytes(content)
            _sample_cache[sample_name] = (now, result)
        _persist_prediction(
            request,
            PredictionRecord.SourceMode.SAMPLE,
            result,
            consent_granted=True,  # curated, non-user sample
        )
    except FileNotFoundError:
        return JsonResponse({"error": "sample_not_available"}, status=404)
    except PredictionInputError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception:
        logger.exception("predict_sample_api failed")
        return JsonResponse({"error": "prediction_failed"}, status=500)
    return JsonResponse(_result_payload(None, result))
