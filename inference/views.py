import json
from typing import Any
from pathlib import Path
from uuid import uuid4
import time

from django.conf import settings
from django.core.files.base import ContentFile
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_http_methods

from history.models import PredictionRecord
from inference.rate_limit import allow_request
from inference.services import PredictionInputError, inference_service
from inference.validation import validate_image_bytes

_sample_cache: dict[str, tuple[float, Any]] = {}
_sample_cache_ttl_seconds = 60


def _client_ip(request: HttpRequest) -> str | None:
    forwarded = request.META.get("HTTP_X_FORWARDED_FOR")
    if forwarded and settings.FER_TRUST_X_FORWARDED_FOR:
        return forwarded.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR")


def _persist_prediction(
    request: HttpRequest,
    source_mode: str,
    result: Any,
    uploaded_file_name: str | None = None,
    image_content: bytes | None = None,
) -> PredictionRecord:
    record = PredictionRecord(
        source_mode=source_mode,
        predicted_emotion=result.emotion,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
        model_version=result.model_version,
        raw_scores=result.scores,
        image_sha256=result.image_sha256,
        request_ip=_client_ip(request),
        user_agent=request.META.get("HTTP_USER_AGENT", "")[:255],
    )
    if uploaded_file_name and image_content:
        suffix = Path(uploaded_file_name).suffix.lower()[:10]
        safe_name = f"{uuid4().hex}{suffix if suffix else '.jpg'}"
        record.image.save(safe_name, ContentFile(image_content), save=False)
    record.save()
    return record


def _enforce_rate_limit(request: HttpRequest, scope: str, limit: int) -> JsonResponse | None:
    ip = _client_ip(request) or "unknown"
    key = f"{scope}:{ip}"
    if not allow_request(key, limit):
        return JsonResponse({"error": "Rate limit exceeded. Please retry in a minute."}, status=429)
    return None


def _enforce_auth(request: HttpRequest) -> JsonResponse | None:
    if settings.FER_REQUIRE_AUTH and not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required."}, status=401)
    return None


@require_GET
def health_api(_: HttpRequest) -> JsonResponse:
    return JsonResponse({"status": "ok"})


@require_http_methods(["GET", "POST"])
def upload_prediction_page(request: HttpRequest) -> HttpResponse:
    context = {}
    if request.method == "POST":
        limited = _enforce_rate_limit(request, "predict-upload-page", settings.FER_API_RATE_LIMIT_IMAGE_PER_MIN)
        if limited is not None:
            context["error"] = "Rate limit exceeded. Please retry in a minute."
            return render(request, "inference/upload.html", context, status=429)
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
                )
                context["result"] = result
                context["sorted_scores"] = sorted(result.scores.items(), key=lambda item: item[1], reverse=True)
            except PredictionInputError as exc:
                context["error"] = str(exc)
            except Exception:
                context["error"] = "Prediction failed. Please try again."
    return render(request, "inference/upload.html", context)


@require_GET
def live_prediction_page(request: HttpRequest) -> HttpResponse:
    return render(request, "inference/live.html")


@require_http_methods(["POST"])
def predict_image_api(request: HttpRequest) -> JsonResponse:
    auth_block = _enforce_auth(request)
    if auth_block is not None:
        return auth_block
    limited = _enforce_rate_limit(request, "predict-image", settings.FER_API_RATE_LIMIT_IMAGE_PER_MIN)
    if limited is not None:
        return limited
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
        )
    except PredictionInputError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception:
        return JsonResponse({"error": "Prediction failed"}, status=500)
    return JsonResponse(
        {
            "id": record.id,
            "emotion": result.emotion,
            "confidence": round(result.confidence, 4),
            "scores": result.scores,
            "latency_ms": result.latency_ms,
            "model_version": result.model_version,
        }
    )


@require_http_methods(["POST"])
def predict_live_frame_api(request: HttpRequest) -> JsonResponse:
    auth_block = _enforce_auth(request)
    if auth_block is not None:
        return auth_block
    limited = _enforce_rate_limit(request, "predict-live", settings.FER_API_RATE_LIMIT_LIVE_PER_MIN)
    if limited is not None:
        return limited
    if len(request.body or b"") > settings.FER_MAX_BASE64_BYTES:
        return JsonResponse({"error": "Payload too large"}, status=413)
    try:
        body = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)
    frame = body.get("image")
    if not frame:
        return JsonResponse({"error": "image is required"}, status=400)
    try:
        result = inference_service.predict_from_base64(frame)
        record = _persist_prediction(request, PredictionRecord.SourceMode.LIVE, result)
    except PredictionInputError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception:
        return JsonResponse({"error": "Prediction failed"}, status=500)
    return JsonResponse(
        {
            "id": record.id,
            "emotion": result.emotion,
            "confidence": round(result.confidence, 4),
            "scores": result.scores,
            "latency_ms": result.latency_ms,
            "model_version": result.model_version,
        }
    )


@require_http_methods(["POST"])
def predict_sample_api(request: HttpRequest, sample_name: str) -> JsonResponse:
    auth_block = _enforce_auth(request)
    if auth_block is not None:
        return auth_block
    limited = _enforce_rate_limit(request, "predict-sample", settings.FER_API_RATE_LIMIT_SAMPLE_PER_MIN)
    if limited is not None:
        return limited
    sample_map = {
        "sample-1": "samples/sample-1.jpg",
        "sample-2": "samples/sample-2.jpg",
        "sample-3": "samples/sample-3.jpg",
        "sample-4": "samples/sample-4.jpg",
        "sample-5": "samples/sample-5.jpg",
    }
    path = sample_map.get(sample_name)
    if not path:
        return JsonResponse({"error": "Unknown sample."}, status=404)
    try:
        now = time.time()
        cached = _sample_cache.get(sample_name)
        if cached and now - cached[0] < _sample_cache_ttl_seconds:
            result = cached[1]
        else:
            with open(Path(settings.BASE_DIR) / "static" / path, "rb") as file_obj:
                content = file_obj.read()
            result = inference_service.predict_from_bytes(content)
            _sample_cache[sample_name] = (now, result)
        _persist_prediction(request, PredictionRecord.SourceMode.SAMPLE, result)
    except FileNotFoundError:
        return JsonResponse({"error": "Sample image not available yet."}, status=404)
    except PredictionInputError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception:
        return JsonResponse({"error": "Prediction failed"}, status=500)
    return JsonResponse(
        {
            "emotion": result.emotion,
            "confidence": round(result.confidence, 4),
            "scores": result.scores,
            "latency_ms": result.latency_ms,
            "model_version": result.model_version,
        }
    )

# Create your views here.
