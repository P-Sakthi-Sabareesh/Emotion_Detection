"""Retention cleanup for PredictionRecord.

Run on a cron / Celery beat / Kubernetes CronJob:

    python manage.py purge_predictions --days 30

Or rely on ``FER_RETENTION_DAYS`` from settings.
"""
from datetime import timedelta

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from history.models import PredictionRecord


class Command(BaseCommand):
    help = "Delete PredictionRecord rows older than the configured retention window."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=None)
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **options):
        days = options["days"] if options["days"] is not None else settings.FER_RETENTION_DAYS
        if days <= 0:
            self.stdout.write(self.style.WARNING("Retention disabled (days <= 0)."))
            return
        cutoff = timezone.now() - timedelta(days=days)
        queryset = PredictionRecord.objects.filter(created_at__lt=cutoff)
        count = queryset.count()
        if options["dry_run"]:
            self.stdout.write(f"Would delete {count} records older than {cutoff.isoformat()}")
            return
        deleted, _ = queryset.delete()
        self.stdout.write(
            self.style.SUCCESS(
                f"Deleted {deleted} prediction records older than {cutoff.isoformat()}"
            )
        )
