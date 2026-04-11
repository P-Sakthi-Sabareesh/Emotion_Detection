from django.conf import settings
from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from history.models import PredictionRecord
from inference.auth import require_api_auth


@csrf_exempt
@require_http_methods(["DELETE"])
@require_api_auth
def delete_record(request: HttpRequest, record_id: int) -> JsonResponse:
    deleted, _ = PredictionRecord.objects.filter(pk=record_id).delete()
    return JsonResponse({"deleted": int(deleted), "id": record_id})


@csrf_exempt
@require_http_methods(["DELETE"])
@require_api_auth
def delete_by_sha256(request: HttpRequest, sha256: str) -> JsonResponse:
    if len(sha256) != 64:
        return JsonResponse({"error": "invalid_sha256"}, status=400)
    queryset = PredictionRecord.objects.filter(image_sha256=sha256)
    deleted, _ = queryset.delete()
    return JsonResponse({"deleted": int(deleted), "sha256": sha256})


@require_http_methods(["GET"])
@require_api_auth
def retention_policy(_: HttpRequest) -> JsonResponse:
    return JsonResponse(
        {
            "retention_days": settings.FER_RETENTION_DAYS,
            "store_predictions": settings.FER_STORE_PREDICTIONS,
            "store_images": settings.FER_STORE_IMAGES,
            "ip_anonymize": settings.FER_IP_ANONYMIZE,
        }
    )
