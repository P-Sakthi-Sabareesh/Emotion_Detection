from django.contrib import admin

from history.models import PredictionRecord


@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):
    list_display = (
        "created_at",
        "source_mode",
        "predicted_emotion",
        "confidence",
        "latency_ms",
        "model_version",
    )
    list_filter = ("source_mode", "predicted_emotion", "model_version", "created_at")
    search_fields = ("predicted_emotion", "model_version", "image_sha256")

# Register your models here.
