# syntax=docker/dockerfile:1.6
#
# Multi-stage build for EmotionDetect.
#
# Stage 1 (builder): installs heavy ML dependencies (torch + transformers)
# and runs scripts/bootstrap_model.py to download the FER ViT from
# HuggingFace, export it to ONNX, and sign the manifest.  The transient
# ~3 GB of torch + transformers + HF cache stays in this stage.
#
# Stage 2 (runtime): slim Python image with only the Django runtime deps
# and the tiny onnxruntime CPU provider.  Copies just the signed artifact
# out of the builder stage.  Final image stays small (~600 MB).
#
# Build:
#     docker build \
#       --build-arg FER_MODEL_SIGNING_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(48))')" \
#       -t emotiondetect-web:latest .

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libjpeg62-turbo \
        libpng16-16 \
        libwebp7 \
    && rm -rf /var/lib/apt/lists/*

# Install runtime deps first (layer reused across rebuilds) then the heavy
# bootstrap-only deps (torch, transformers, onnx).  CPU-only torch wheel
# is pulled from the official CPU index to stay under 300 MB instead of
# the 2.5 GB CUDA variant.
COPY requirements.txt requirements-bootstrap.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-bootstrap.txt

# Bring in just what the bootstrap script needs.
COPY scripts/bootstrap_model.py ./scripts/bootstrap_model.py
COPY ml/__init__.py ./ml/__init__.py
COPY ml/scripts/ ./ml/scripts/
COPY ml/features.py ./ml/features.py
COPY ml/model_contract.py ./ml/model_contract.py

# FER_MODEL_SIGNING_KEY is required at build time so the bootstrap can
# sign the manifest.  Pass it via:
#     docker build --build-arg FER_MODEL_SIGNING_KEY=...
#
# NOTE: build args are visible in image history (`docker history`) unless
# you use BuildKit secrets (`--mount=type=secret`).  For real production
# builds, prefer the secret-mount variant below.
ARG FER_MODEL_SIGNING_KEY=""
RUN test -n "$FER_MODEL_SIGNING_KEY" || (echo "ERROR: --build-arg FER_MODEL_SIGNING_KEY is required" && exit 2)

RUN FER_MODEL_SIGNING_KEY="$FER_MODEL_SIGNING_KEY" \
    python scripts/bootstrap_model.py --output-dir /build/artifacts

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        libpng16-16 \
        libwebp7 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Non-root runtime user
RUN groupadd --system app && useradd --system --gid app --home /app --shell /sbin/nologin app

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install gunicorn==22.0.0

# Project source
COPY . .

# Signed artifact from the builder stage (drops in alongside the committed
# code so the runtime loader finds it at FER_MODEL_PATH).
COPY --from=builder /build/artifacts/ /app/ml/artifacts/

# Collect static assets once at build time so whitenoise can serve them
# with compressed, hash-manifest filenames.  Uses a placeholder secret
# that only exists during this single RUN layer.
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
