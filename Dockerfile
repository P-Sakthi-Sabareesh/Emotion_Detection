FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libjpeg62-turbo \
        libpng16-16 \
        libwebp7 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a non-root runtime user
RUN groupadd --system app && useradd --system --gid app --home /app --shell /sbin/nologin app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt && pip install gunicorn==22.0.0

COPY . .

# Collect static assets once at build time so whitenoise can serve them
# with compressed, hash-manifest filenames. Needs a dummy SECRET_KEY that
# satisfies the production guard in config/settings.py.
RUN DJANGO_DEBUG=False \
    DJANGO_SECRET_KEY="build-time-placeholder-secret-key-build-time-placeholder-xx" \
    DJANGO_ALLOWED_HOSTS=localhost \
    FER_REQUIRE_SIGNED_ARTIFACT=False \
    python manage.py collectstatic --noinput

RUN mkdir -p /app/media /app/staticfiles \
    && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/readyz || exit 1

CMD ["gunicorn", "config.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "3", \
     "--threads", "4", \
     "--timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
