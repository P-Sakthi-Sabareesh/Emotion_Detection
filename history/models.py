from django.db import models


class PredictionRecord(models.Model):
    class SourceMode(models.TextChoices):
        UPLOAD = "upload", "Upload"
        LIVE = "live", "Live"
        SAMPLE = "sample", "Sample"

    source_mode = models.CharField(max_length=16, choices=SourceMode.choices)
    predicted_emotion = models.CharField(max_length=32)
    confidence = models.FloatField()
    latency_ms = models.PositiveIntegerField()
    model_version = models.CharField(max_length=64)
    image = models.ImageField(upload_to="predictions/%Y/%m/%d/", null=True, blank=True)
    image_sha256 = models.CharField(max_length=64, blank=True, default="")
    raw_scores = models.JSONField(default=dict, blank=True)
    request_ip = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.CharField(max_length=255, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.predicted_emotion} ({self.confidence:.2f}) at {self.created_at:%Y-%m-%d %H:%M:%S}"

# Create your models here.
