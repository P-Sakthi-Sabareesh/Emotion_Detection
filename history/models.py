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
    model_loaded = models.BooleanField(default=True)
    degraded = models.BooleanField(default=False)
    image = models.ImageField(upload_to="predictions/%Y/%m/%d/", null=True, blank=True)
    image_sha256 = models.CharField(max_length=64, blank=True, default="")
    raw_scores = models.JSONField(default=dict, blank=True)
    request_ip_prefix = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.CharField(max_length=255, blank=True, default="")
    consent_granted = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["-created_at"]),
            models.Index(fields=["source_mode", "-created_at"]),
            models.Index(fields=["model_version"]),
            models.Index(fields=["image_sha256"]),
        ]

    def __str__(self) -> str:
        return f"{self.predicted_emotion} ({self.confidence:.2f}) at {self.created_at:%Y-%m-%d %H:%M:%S}"
