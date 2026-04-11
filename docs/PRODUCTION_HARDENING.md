# Production Hardening — Change Log

This document maps the hardening work to the audit findings it closes. Read
this alongside `CLAUDE.md` (architecture) and `.env.example` (configuration).

## P0 — Closed

| # | Finding | Where it was fixed |
|---|---|---|
| 1 | Unsafe model deserialization (RCE via joblib) | `inference/model_loader.py` verifies sha256 and HMAC on the artifact manifest *before* any deserialization. Runtime refuses to start (or refuses to serve) if `FER_REQUIRE_SIGNED_ARTIFACT=True` and verification fails. Sign with `ml/scripts/sign_artifact.py`. |
| 2 | `DEBUG=True` default + hardcoded fallback SECRET_KEY | `config/settings.py` defaults to `DEBUG=False`, requires a ≥50-char `DJANGO_SECRET_KEY` and explicit `DJANGO_ALLOWED_HOSTS` in production. |
| 3 | Biometric data without consent or retention | `FER_REQUIRE_CONSENT=True` by default; consent enforced on the HTML form, the live JS, and all JSON endpoints (body `consent=true`, query arg, or `X-Consent` header). `FER_STORE_IMAGES=False` by default. `FER_RETENTION_DAYS` drives `history/management/commands/purge_predictions.py`. `DELETE /api/v1/records/<id>/` and `/api/v1/records/sha256/<hash>/` implement right-to-erasure. |
| 4 | Silent failure: lying `model_version` on fallback | `PredictionResult` now carries `model_loaded` and `degraded`. Fallback predictions are logged at WARNING, tagged with `FER_FALLBACK_MODEL_VERSION` (`fallback-heuristic-v1`), and surface in the JSON response. `/readyz` reports 503 while the model is unloaded. |

## P1 — Closed

| # | Finding | Where it was fixed |
|---|---|---|
| 4 | Decompression bombs and polyglot images | `inference/validation.py` sets `Image.MAX_IMAGE_PIXELS`, disables `LOAD_TRUNCATED_IMAGES`, enforces a format whitelist (`FER_ALLOWED_IMAGE_FORMATS`), rejects multi-frame images, and converts `DecompressionBombError` into `400`. |
| 5 | In-process rate limiter | `inference/rate_limit.py` now provides a Redis ZSET sliding-window backend (`FER_RATE_LIMIT_BACKEND=redis`). Production fails closed when Redis is unavailable; dev falls back to in-process with a warning log. `FER_RATE_LIMIT_STRICT=1` refuses to start if the backend is not Redis. |
| 7 | Silent fallback | See P0-4 above. |
| 8 | Readiness vs liveness | `/healthz` returns 200 unconditionally (liveness). `/readyz` checks model-load status and a `SELECT 1` on the DB and returns 503 otherwise. |
| 9 | Auth inconsistency between HTML and JSON endpoints | `inference/auth.py::require_api_auth` wraps every prediction API uniformly. HTML upload view applies the same session check. |
| 11 | Rate-limit after upload read | Views now run a Content-Length preflight *before* reading the body. `DATA_UPLOAD_MAX_MEMORY_SIZE` / `FILE_UPLOAD_MAX_MEMORY_SIZE` / `DATA_UPLOAD_MAX_NUMBER_FIELDS` are all capped in `settings.py`. |

## P2 — Closed

- Sample traversal: sample paths are now resolved against a fixed root with `Path.resolve()` and a parent-containment check.
- `_fallback_scores` still logs but cannot silently masquerade as the real model.
- `PredictionRecord` now has indexes on `created_at`, `(source_mode, created_at)`, `model_version`, and `image_sha256`.
- IP addresses are anonymized by default (`/24` for IPv4, `/48` for IPv6) before storage — see `FER_IP_ANONYMIZE`.
- CSP-hostile inline `<script>` in `templates/inference/live.html` moved to `static/js/live.js` so a future strict-CSP middleware does not break the page.

## Still deferred (explicit TODOs)

These require a separate PR with their own risk budget; the scaffolding is in place but the change itself is not in this commit.

1. **CNN model + ONNX export.** `inference/model_loader.py` already recognizes `.onnx` and uses `onnxruntime.InferenceSession`. Training a proper CNN (MobileNetV3 / HSEmotion) and exporting it is a separate workstream. Until then, run with the signed sklearn artifact.
2. **Face detection.** Currently the multi-crop heuristic still approximates face alignment. Plug in YuNet/BlazeFace in `inference/services.py::_build_inference_variants`.
3. **Celery async offload.** Inference still runs on the request thread. `Dockerfile` + `docker-compose.yml` already define a `worker` profile for the retention command; extending it for a Celery worker pool is a one-day change.
4. **WebSocket streaming.** Polling still happens every 1.2 s. Add Django Channels + Redis pub/sub, keep the HMAC signing and consent enforcement, move the rate limit to per-connection.
5. **Strict CSP middleware** (django-csp). JS is already external, so the switch is now feasible without breaking the page.
6. **Actually run `makemigrations` against Postgres.** Migration `0002_hardening` covers field renames/additions from the sqlite-era `0001_initial` — apply it in a maintenance window, not mid-traffic.

## Operating the signed artifact flow

```bash
# 1. Train (same as before)
python ml/scripts/train_fer2013.py \
  --dataset-root data/raw \
  --output-dir ml/artifacts \
  --model-version fer2013-sklearn-v2 \
  --max-per-class 1200 \
  --no-calibrate-probabilities \
  --jobs 1

# 2. Sign the artifact using a secret stored in your CI secrets manager
FER_MODEL_SIGNING_KEY=$(op read op://Prod/fer-signing-key) \
  python ml/scripts/sign_artifact.py --manifest ml/artifacts/artifact_manifest.json

# 3. Deploy the artifact + manifest together. Runtime refuses to start if either
#    is missing or the HMAC does not verify.
```

## Operating retention

```bash
python manage.py purge_predictions --days 30          # honor FER_RETENTION_DAYS default
python manage.py purge_predictions --days 7 --dry-run # preview
```

Run this on a schedule (cron, k8s `CronJob`, or Celery beat — see
`docker-compose.yml` `cron` profile).

## Health / readiness probes

```bash
curl -fsS http://127.0.0.1:8000/healthz        # always 200 when process alive
curl -fsS http://127.0.0.1:8000/readyz         # 200 only when model+DB healthy
```

Kubernetes:

```yaml
livenessProbe:
  httpGet: {path: /healthz, port: 8000}
  periodSeconds: 15
readinessProbe:
  httpGet: {path: /readyz, port: 8000}
  periodSeconds: 5
  failureThreshold: 3
```
