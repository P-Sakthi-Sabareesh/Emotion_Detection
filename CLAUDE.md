# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Django-based facial emotion detection web app trained on FER-2013. Ships a Django UI, prediction APIs (upload / live webcam / sample gallery), and an scikit-learn training pipeline. Target labels: `angry, disgust, fear, happy, neutral, sad, surprise`.

## Common Commands

Environment setup and running the server:
```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Tests (Django test runner):
```bash
python manage.py test                      # whole suite
python manage.py test inference             # one app
python manage.py test inference.tests.EmotionInferenceSafeguardsTests.test_rule_correction_reduces_angry_false_positive  # single test
```

FER-2013 training pipeline (order matters):
```bash
python ml/scripts/download_fer2013.py --output-dir data/raw
python ml/scripts/train_fer2013.py --dataset-root data/raw --output-dir ml/artifacts \
  --model-version fer2013-sklearn-v2 --max-per-class 1200 --no-calibrate-probabilities --jobs 1
python ml/scripts/plot_metrics.py --metrics ml/artifacts/metrics.json --out ml/artifacts/confusion_matrix.png
python ml/scripts/prepare_demo_samples.py --dataset-root data/raw --output-dir static/samples
```

Kaggle credentials (`~/.kaggle/kaggle.json`) must be configured before running the downloader.

## Architecture

### Django apps
- `config/` — settings, URL root (`config.urls`), WSGI/ASGI. Settings are env-driven; many `FER_*` knobs live here.
- `core/` — static pages (`home`, `prediction_hub`, `about`, `contact`). No DB models.
- `inference/` — prediction views, JSON APIs, model service, validation, rate limiting.
- `history/` — `PredictionRecord` model that persists every prediction (source mode, emotion, confidence, latency, model version, raw scores, sha256, IP/UA, optional image).
- `ml/` — offline training pipeline and artifact contract (no runtime Django imports).

### Request flow
1. Browser hits a `core` page or one of the `inference` endpoints defined in `inference/urls.py`:
   - Pages: `/prediction/upload/`, `/prediction/live/`
   - APIs: `/api/health/`, `/api/predict/image/`, `/api/predict/live-frame/`, `/api/predict/sample/<sample-name>/`
2. View enforces optional auth (`FER_REQUIRE_AUTH`) and per-IP rate limit (`inference.rate_limit.allow_request`, scope-keyed: `predict-image` / `predict-live` / `predict-sample`).
3. `inference.validation.validate_image_bytes` checks size, format, and pixel count before decoding.
4. `inference.services.inference_service` (a module-level `EmotionInferenceService` singleton) runs prediction.
5. `history.models.PredictionRecord` persists the result; uploaded image bytes go to `media/predictions/YYYY/MM/DD/` with a UUID filename.

### Inference service (`inference/services.py`)
The `EmotionInferenceService` is the hot path and has several subtle behaviors that must be preserved:
- **Lazy model load** from `settings.FER_MODEL_PATH`. Supports `.joblib` (sklearn, via `predict_proba`) and `.keras` (TensorFlow, reshapes features to `(1,48,48,1)` and calls `predict`). If loading fails, `_model` stays `None` and a **deterministic fallback scorer** (`_fallback_scores`) keeps the UI functional without ML dependencies.
- **Multi-crop ensemble**: `_build_inference_variants` produces four weighted 48×48 grayscale crops (full, center, upper-center, top-band). Probabilities are averaged with those weights, not just the raw model output.
- **Rule-based corrections** (`_apply_probability_rules`, gated by `FER_ENABLE_RULE_BASED_CORRECTION`): nudges angry↔happy, disgust→neutral, and fear↔surprise when scores are close. These heuristics exist because the FER-2013 baseline has known confusion pairs — tests in `inference/tests.py` pin this behavior.
- **Low-confidence fallback** (`_select_with_fallback`, gated by `FER_ENABLE_CONFIDENCE_FALLBACK`): if best score < `FER_MIN_CONFIDENCE` or margin < `FER_MIN_MARGIN`, returns `FER_FALLBACK_EMOTION` (default `neutral`). Off by default.
- `predict_from_base64` strips `data:` prefixes and delegates to `predict_from_bytes`.

When modifying service logic, run `python manage.py test inference` — the tests stub `_model` and `_to_feature_vector` to assert rule corrections and fallback selection.

### Training pipeline (`ml/scripts/train_fer2013.py`)
- Loads grayscale 48×48 images per class from `<dataset-root>/{train,test}/<emotion>/`.
- Builds engineered features: raw pixels + gradient magnitude + 2×2 pooled versions + intensity/contrast scalars (`build_features`).
- Runs **model selection** across three candidates (`sgd_log_elasticnet`, `sgd_modified_huber`, `linear_svc_margin`) using a stratified validation split, selecting on `validation_macro_f1`.
- Retrains the winner on the full train split, optionally wraps it in `CalibratedClassifierCV` (sigmoid/isotonic).
- Writes four artifacts to `--output-dir`: `emotion_classifier.joblib`, `model_metadata.json`, `metrics.json`, `artifact_manifest.json` (with sha256s).
- The saved artifact is a `SklearnEmotionModel` dataclass from `ml/model_contract.py` — a thin wrapper exposing `predict_proba` and `model_version`. `EmotionInferenceService` reads `model_version` via `getattr`, so changing the contract requires updating both sides.

### Configuration & security
All behavior is tunable via env vars (see `.env.example`). Notable knobs:
- `FER_MODEL_PATH`, `FER_MODEL_VERSION` — runtime model selection.
- `FER_MAX_UPLOAD_MB`, `FER_MAX_IMAGE_PIXELS`, `FER_MAX_BASE64_BYTES` — input hardening.
- `FER_API_RATE_LIMIT_{IMAGE,LIVE,SAMPLE}_PER_MIN` — per-scope throttles.
- `FER_TRUST_X_FORWARDED_FOR` — only honor `X-Forwarded-For` when behind a trusted proxy.
- `FER_REQUIRE_AUTH` — gate APIs behind `request.user.is_authenticated`.
- `FER_ENABLE_RULE_BASED_CORRECTION`, `FER_ENABLE_CONFIDENCE_FALLBACK`, `FER_MIN_CONFIDENCE`, `FER_MIN_MARGIN`, `FER_FALLBACK_EMOTION`.
- When `DJANGO_DEBUG=False`, `config/settings.py` **raises at import time** unless `DJANGO_SECRET_KEY` is set to something other than the dev default, and forces secure cookies / SSL redirect / HSTS on by default.
- Database auto-selects Postgres when `DATABASE_URL` is set, otherwise falls back to SQLite at `db.sqlite3`.

### Production caveats worth knowing
- `inference/rate_limit.py` is **in-process only** (thread-locked `deque` per key). It does not work across multiple workers or hosts — replace with Redis/API gateway throttling before horizontal scaling.
- Model artifacts are deserialized via `joblib.load`, which can execute arbitrary code from the artifact. Treat `ml/artifacts/` as trusted-only storage; never load a `.joblib` from an untrusted source.
- Prediction records retain IP, user agent, and optionally the uploaded image. Enforce retention/deletion before handling real user data.
