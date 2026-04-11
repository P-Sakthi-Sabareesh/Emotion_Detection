# EmotionDetect — FER-2013 Full-Stack

Production-grade facial emotion detection web app.

- **Model**: [`dima806/facial_emotions_image_detection`](https://huggingface.co/dima806/facial_emotions_image_detection) — ViT-base fine-tuned on FER-2013, ~90%+ top-1 accuracy. Exported to ONNX at bootstrap time and served via `onnxruntime`.
- **Backend**: Django 5.1 on gunicorn, Postgres-ready, Redis-backed sliding-window rate limiter, signed-artifact loader, biometric consent gating, structured JSON logging.
- **Frontend**: glass/aurora UI with live webcam and upload workflows, CSP-friendly (external JS, static assets via whitenoise), responsive + reduced-motion aware.
- **Infra**: multi-stage Docker build, healthz/readyz split, HMAC-signed model manifest, retention cron, 23-test suite.

The ONNX model is **not** committed to the repo — the bootstrap script downloads the FP32 weights from HuggingFace on first run and signs them locally with your private `FER_MODEL_SIGNING_KEY`. This keeps the repo small, guarantees downstream clones always get the exact FP32 baseline (deterministic export), and lets every deployment rotate its own signing key.

---

## Quick start

**Prerequisites**
- Python 3.12+
- Docker Desktop / Docker Engine 20+ (for the containerized path)
- ~1.5 GB free disk for the one-time model download

### 1. Clone
```bash
git clone https://github.com/P-Sakthi-Sabareesh/Emotion_Detection.git
cd Emotion_Detection
```

### 2. Generate a signing key (once per environment)
```bash
export FER_MODEL_SIGNING_KEY=$(python -c 'import secrets;print(secrets.token_urlsafe(48))')
# keep this value — the runtime verifies every artifact load against it
```

### 3. Bootstrap the model (downloads + signs)
```bash
pip install -r requirements.txt -r requirements-bootstrap.txt
python scripts/bootstrap_model.py
```

First run takes ~2 minutes (downloads the ViT from HuggingFace, exports to ONNX, signs the manifest, verifies load). Subsequent runs are idempotent — if the cached artifact matches the signed manifest, the script exits immediately.

### 4. Run — pick one

**Local (Django runserver)**
```bash
python manage.py migrate
python manage.py runserver
```

**Containerized (Docker Compose, recommended for production parity)**
```bash
docker compose up --build
```
The compose build passes `FER_MODEL_SIGNING_KEY` from your shell into the multi-stage image builder so the container bakes in a signed artifact. No model files are shipped in the git repo.

Open http://localhost:8000 (or 8008 if running the compose profile).

---

## Hitting the API

```bash
# health / readiness
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz

# upload prediction (browser flow, uses session + consent checkbox)
open http://localhost:8000/prediction/upload/

# live webcam
open http://localhost:8000/prediction/live/

# programmatic (bearer-auth, explicit consent)
curl -sS \
  -H "Authorization: Bearer $FER_API_TOKEN" \
  -F "image=@some_face.jpg" \
  -F "consent=true" \
  http://localhost:8000/api/v1/predict/image/
```

---

## Project structure

```
config/          Django settings, URL root, JSON logging formatter
core/            Landing + static pages (home, prediction hub, about, contact)
inference/       Prediction views, JSON APIs, model loader, consent gate,
                 auth decorator, rate limiter, request-id middleware
history/         PredictionRecord model + retention management command
ml/features.py   Shared engineered-features builder (sklearn path)
ml/scripts/      Legacy training pipeline (sklearn baseline, not shipped)
scripts/         bootstrap_model.py — download + export + sign FP32 ONNX
static/          CSS / JS / sample gallery images
templates/       Base + prediction templates (glass/aurora UI)
docs/            Production hardening notes
```

## Endpoints

| Path | Method | Auth | Purpose |
|---|---|---|---|
| `/healthz` | GET | — | liveness (always 200 when process alive) |
| `/readyz` | GET | — | readiness (503 if model unloaded or DB down) |
| `/prediction/upload/` | GET/POST | session | HTML upload form with consent checkbox |
| `/prediction/live/` | GET | session | live webcam UI (JS polls live-frame API) |
| `/api/v1/predict/image/` | POST | Bearer + consent | multipart upload, JSON result |
| `/api/v1/predict/live-frame/` | POST | Bearer + consent | base64 frame, JSON result |
| `/api/v1/predict/sample/<name>/` | POST | Bearer | curated sample gallery (consent auto-granted) |
| `/api/v1/records/<id>/` | DELETE | Bearer | right-to-erasure by record id |
| `/api/v1/records/sha256/<hash>/` | DELETE | Bearer | right-to-erasure by image hash |

All API endpoints return `451 Unavailable For Legal Reasons` if called without consent, `401` with `WWW-Authenticate: Bearer` if auth fails, `429` with `Retry-After: 60` if rate-limited.

## Configuration

All behavior is env-driven. See `.env.example` for the full annotated list. Critical values:

| Env var | Default | Meaning |
|---|---|---|
| `FER_MODEL_SIGNING_KEY` | — | HMAC-SHA256 signing key; **required** for bootstrap and runtime |
| `FER_REQUIRE_SIGNED_ARTIFACT` | `True` (prod) | Refuse to serve if the signature doesn't verify |
| `FER_REQUIRE_AUTH` | `True` (prod) | Require Bearer token on API endpoints |
| `FER_API_TOKENS` | — | Comma-separated bearer tokens |
| `FER_REQUIRE_CONSENT` | `True` | Enforce consent gate on all prediction endpoints |
| `FER_STORE_PREDICTIONS` | `True` | Persist PredictionRecord rows to DB |
| `FER_STORE_IMAGES` | `False` | Persist uploaded image bytes (GDPR Art. 9 — default off) |
| `FER_RETENTION_DAYS` | `30` | Retention window for `purge_predictions` command |
| `FER_RATE_LIMIT_BACKEND` | `redis` (prod) | `redis` or `inprocess` |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Rate-limit store |

## Model artifact lifecycle

The runtime expects `ml/artifacts/emotion_classifier.onnx` + `ml/artifacts/artifact_manifest.json`. These files are **generated, not committed**. Generation flows:

1. **Fresh local dev**: `python scripts/bootstrap_model.py`
2. **Docker build**: multi-stage Dockerfile runs the same script in the builder stage. Build arg `FER_MODEL_SIGNING_KEY` is required.
3. **CI**: GitHub Actions runs the bootstrap before tests so the test suite can exercise the real model.

The script is idempotent: if a cached artifact already matches the signed manifest, it exits without re-downloading. Rotate the signing key and pass `--force` to re-export.

## Retention cron

```bash
python manage.py purge_predictions --days 30             # honor FER_RETENTION_DAYS default
python manage.py purge_predictions --days 7 --dry-run    # preview
```

Run on a schedule via `docker compose --profile cron run worker`, a k8s CronJob, or systemd timer.

## Tests

```bash
python manage.py test inference --verbosity 1
```

23 tests cover: signed-manifest sha256+HMAC accept/reject paths, decompression-bomb guard, format whitelist, multi-frame reject, rule-based post-processing, degraded-fallback sentinel, consent gate (451), Bearer auth, per-IP + per-user rate limiting, liveness vs readiness endpoints, feature engineering round-trip.

## Security controls (summary)

- Image payload validation (size, format, pixel count, multi-frame reject, decompression-bomb guard via `Pillow.MAX_IMAGE_PIXELS`).
- HMAC-SHA256 signed model manifest — runtime refuses to load any artifact whose sha256 or signature doesn't match.
- Biometric consent gate on every prediction endpoint (451 Unavailable For Legal Reasons on refusal).
- Default `FER_STORE_IMAGES=False`; prediction records retain only emotion + confidence + anonymized `/24` IP prefix.
- Right-to-erasure via DELETE endpoints + retention command.
- API endpoints use Bearer token auth (CSRF-exempt) separate from the session-authenticated HTML forms.
- Redis ZSET sliding-window rate limiter (atomic Lua script) — fails closed in production if Redis is unreachable.
- Request-id middleware + structured JSON logging with propagation.
- Whitenoise-compressed static assets with hashed filenames.
- Hardened Django `--deploy` defaults: mandatory 50+ char `SECRET_KEY`, explicit `ALLOWED_HOSTS`, secure cookies, HSTS, SSL redirect, referrer policy, X-Frame DENY.

See `docs/PRODUCTION_HARDENING.md` for the full audit → fix mapping.

## License

See repository for license terms. The underlying ViT model is distributed under its HuggingFace model card terms.
