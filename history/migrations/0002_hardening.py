"""Consent + hardening migration.

Renames ``request_ip`` to ``request_ip_prefix`` (storing only anonymized
network address) and adds consent + degraded flags plus performance indexes.
"""
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("history", "0001_initial"),
    ]

    operations = [
        migrations.RenameField(
            model_name="predictionrecord",
            old_name="request_ip",
            new_name="request_ip_prefix",
        ),
        migrations.AddField(
            model_name="predictionrecord",
            name="model_loaded",
            field=models.BooleanField(default=True),
        ),
        migrations.AddField(
            model_name="predictionrecord",
            name="degraded",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="predictionrecord",
            name="consent_granted",
            field=models.BooleanField(default=False),
        ),
        migrations.AddIndex(
            model_name="predictionrecord",
            index=models.Index(fields=["-created_at"], name="history_pre_created_idx"),
        ),
        migrations.AddIndex(
            model_name="predictionrecord",
            index=models.Index(
                fields=["source_mode", "-created_at"], name="history_pre_source_created_idx"
            ),
        ),
        migrations.AddIndex(
            model_name="predictionrecord",
            index=models.Index(
                fields=["model_version"], name="history_pre_modelver_idx"
            ),
        ),
        migrations.AddIndex(
            model_name="predictionrecord",
            index=models.Index(
                fields=["image_sha256"], name="history_pre_sha256_idx"
            ),
        ),
    ]
