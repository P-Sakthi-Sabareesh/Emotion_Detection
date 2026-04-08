# EmotionDetect (FER-2013 Full-Stack)

Production-oriented facial emotion detection web application with:
- Django web UI + APIs
- Kaggle FER-2013 training pipeline
- Upload and live webcam prediction workflows
- Prediction history persistence with confidence and latency
- Rate-limited prediction APIs with security hardening
- Enterprise-style UI cards for sample/upload/live workflows

## Stack
- Python, Django
- NumPy, scikit-learn, Pillow, Matplotlib
- Kaggle CLI for FER-2013 data ingestion
- SQLite (local) with PostgreSQL-ready config

## Project Structure
- `config/` Django settings and URL routing
- `core/` landing and static pages
- `inference/` prediction views, APIs, model service
- `history/` persistence for prediction records
- `ml/scripts/` dataset download, training, metrics plotting
- `ml/artifacts/` model and evaluation outputs

## Quick Start
1. Create and activate venv.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Apply migrations:
   - `python manage.py migrate`
4. Run server:
   - `python manage.py runserver`

## FER-2013 Training (Kaggle)
1. Configure Kaggle credentials (`~/.kaggle/kaggle.json`).
2. Download dataset:
   - `python ml/scripts/download_fer2013.py --output-dir data/raw`
3. Train production baseline model:
   - `python ml/scripts/train_fer2013.py --dataset-root data/raw --output-dir ml/artifacts --model-version fer2013-sklearn-v2 --max-per-class 1200 --no-calibrate-probabilities --jobs 1`
4. Plot confusion matrix:
   - `python ml/scripts/plot_metrics.py --metrics ml/artifacts/metrics.json --out ml/artifacts/confusion_matrix.png`
5. Prepare stable sample gallery from FER-2013:
   - `python ml/scripts/prepare_demo_samples.py --dataset-root data/raw --output-dir static/samples`

## Runtime Endpoints
- `GET /api/health/`
- `POST /api/predict/image/`
- `POST /api/predict/live-frame/`
- `POST /api/predict/sample/<sample-name>/`

## Demo Workflow
1. Open home page.
2. Navigate to `Prediction`.
3. Try one of:
   - sample gallery prediction
   - upload image prediction
   - live webcam prediction

## Notes
- Default model path is `ml/artifacts/emotion_classifier.joblib`.
- If model file is missing, fallback scoring is used so UI remains functional.
- For stronger FER performance, replace baseline with a trained CNN artifact.

## Security Controls
- Image payload validation (size, format, max pixels).
- API rate limiting per source mode (`image`, `live`, `sample`).
- Optional auth gate for prediction APIs (`FER_REQUIRE_AUTH`).
- Safer production defaults: security headers, SSL redirect toggle, non-default secret requirement when debug is off.

## Production Caveats
- FER model artifact is still loaded via `joblib`; treat artifact storage as trusted-only.
- In-memory rate limiting should be replaced by centralized throttling (Redis/API gateway) for multi-instance deployment.
- For enterprise compliance, enforce retention/deletion policy for prediction records and uploaded images.
