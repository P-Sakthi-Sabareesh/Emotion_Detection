import base64
import hashlib
import io
import logging
import time
from dataclasses import dataclass
from typing import Final

import joblib
import numpy as np
from django.conf import settings
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABEL_TO_INDEX: Final[dict[str, int]] = {label: idx for idx, label in enumerate(EMOTION_LABELS)}


class PredictionInputError(ValueError):
    pass


@dataclass
class PredictionResult:
    emotion: str
    confidence: float
    scores: dict[str, float]
    model_version: str
    latency_ms: int
    image_sha256: str


class EmotionInferenceService:
    def __init__(self) -> None:
        self._model = None
        self._loaded = False
        self.model_version = settings.FER_MODEL_VERSION

    def _load_model(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        model_path = settings.FER_MODEL_PATH
        try:
            if model_path.endswith(".keras"):
                from tensorflow import keras

                self._model = keras.models.load_model(model_path)
                self.model_version = settings.FER_MODEL_VERSION
            else:
                self._model = joblib.load(model_path)
                self.model_version = getattr(self._model, "model_version", self.model_version)
            logger.info("Loaded FER model from %s", model_path)
        except FileNotFoundError:
            logger.warning("FER model not found at %s; using fallback predictor", model_path)
            self._model = None
        except Exception:
            logger.exception("Failed to load model at %s; using fallback predictor", model_path)
            self._model = None

    @staticmethod
    def _validate_size(content: bytes) -> None:
        max_bytes = settings.FER_MAX_UPLOAD_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise PredictionInputError(f"File is too large. Maximum allowed is {settings.FER_MAX_UPLOAD_MB} MB.")

    @staticmethod
    def _to_feature_vector(content: bytes) -> tuple[np.ndarray, str]:
        image_hash = hashlib.sha256(content).hexdigest()
        with Image.open(io.BytesIO(content)) as image:
            gray = image.convert("L")
            variants = EmotionInferenceService._build_inference_variants(gray)
        return variants, image_hash

    @staticmethod
    def _build_inference_variants(gray_image: Image.Image) -> list[tuple[np.ndarray, float]]:
        width, height = gray_image.size
        if width < 24 or height < 24:
            resized = gray_image.resize((48, 48), Image.Resampling.BILINEAR)
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            return [(arr.flatten().reshape(1, -1), 1.0)]

        def crop_box(x0: float, y0: float, x1: float, y1: float) -> Image.Image:
            left = max(0, min(width - 2, int(width * x0)))
            top = max(0, min(height - 2, int(height * y0)))
            right = max(left + 2, min(width, int(width * x1)))
            bottom = max(top + 2, min(height, int(height * y1)))
            return gray_image.crop((left, top, right, bottom))

        center = crop_box(0.2, 0.08, 0.8, 0.9)
        upper_center = crop_box(0.2, 0.0, 0.8, 0.72)
        top_band = crop_box(0.1, 0.0, 0.9, 0.62)
        full = gray_image

        candidates: list[tuple[Image.Image, float]] = [
            (full, 0.18),
            (center, 0.5),
            (upper_center, 0.22),
            (top_band, 0.1),
        ]

        variants: list[tuple[np.ndarray, float]] = []
        for candidate, weight in candidates:
            arr = np.asarray(candidate.resize((48, 48), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            variants.append((arr.flatten().reshape(1, -1), weight))
        return variants

    @staticmethod
    def _fallback_scores(features: np.ndarray) -> np.ndarray:
        flat = features.reshape(-1)
        brightness = float(np.mean(flat))
        contrast = float(np.std(flat))
        score = np.array(
            [
                max(0.01, (0.55 - brightness) + contrast),  # angry
                max(0.01, (0.45 - contrast) * 0.6),  # disgust
                max(0.01, 0.35 + (0.5 - brightness)),  # fear
                max(0.01, brightness + 0.25),  # happy
                max(0.01, 0.45 + (0.3 - abs(0.5 - brightness))),  # neutral
                max(0.01, 0.4 + (0.52 - brightness)),  # sad
                max(0.01, 0.35 + contrast),  # surprise
            ],
            dtype=np.float32,
        )
        return score / np.sum(score)

    @staticmethod
    def _normalize_probabilities(probabilities: np.ndarray | list[float]) -> np.ndarray:
        normalized = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if normalized.shape[0] != len(EMOTION_LABELS):
            raise PredictionInputError("Model returned invalid prediction scores.")
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        normalized = np.clip(normalized, 0.0, None)
        total = float(np.sum(normalized))
        if total <= 0.0:
            return np.full(len(EMOTION_LABELS), 1.0 / len(EMOTION_LABELS), dtype=np.float32)
        return normalized / total

    @staticmethod
    def _apply_probability_rules(probabilities: np.ndarray) -> np.ndarray:
        if not getattr(settings, "FER_ENABLE_RULE_BASED_CORRECTION", True):
            return probabilities

        adjusted = np.array(probabilities, dtype=np.float32, copy=True)
        angry_idx = LABEL_TO_INDEX["angry"]
        happy_idx = LABEL_TO_INDEX["happy"]
        neutral_idx = LABEL_TO_INDEX["neutral"]
        disgust_idx = LABEL_TO_INDEX["disgust"]
        surprise_idx = LABEL_TO_INDEX["surprise"]
        fear_idx = LABEL_TO_INDEX["fear"]

        # Common FER confusion: smiling faces can be over-predicted as angry.
        if (
            adjusted[angry_idx] >= adjusted[happy_idx]
            and adjusted[happy_idx] >= 0.30
            and (adjusted[angry_idx] - adjusted[happy_idx]) <= 0.18
        ):
            transfer = min(0.12, adjusted[angry_idx] * 0.25)
            adjusted[angry_idx] -= transfer
            adjusted[happy_idx] += transfer

        # "disgust" is often over-confident on weak evidence; prefer neutral when close.
        if adjusted[disgust_idx] >= 0.33 and adjusted[neutral_idx] >= 0.22:
            transfer = min(0.07, adjusted[disgust_idx] * 0.2)
            adjusted[disgust_idx] -= transfer
            adjusted[neutral_idx] += transfer

        # Fear/surprise confusion tends to lean fear; rebalance when very close.
        if adjusted[fear_idx] > adjusted[surprise_idx] and (adjusted[fear_idx] - adjusted[surprise_idx]) <= 0.08:
            transfer = min(0.05, adjusted[fear_idx] * 0.15)
            adjusted[fear_idx] -= transfer
            adjusted[surprise_idx] += transfer

        return EmotionInferenceService._normalize_probabilities(adjusted)

    @staticmethod
    def _select_with_fallback(probabilities: np.ndarray) -> tuple[str, float]:
        best_idx = int(np.argmax(probabilities))
        best_emotion = EMOTION_LABELS[best_idx]
        best_confidence = float(probabilities[best_idx])

        sorted_indices = np.argsort(probabilities)[::-1]
        second_confidence = float(probabilities[int(sorted_indices[1])]) if len(sorted_indices) > 1 else 0.0
        margin = best_confidence - second_confidence

        min_confidence = float(getattr(settings, "FER_MIN_CONFIDENCE", 0.35))
        min_margin = float(getattr(settings, "FER_MIN_MARGIN", 0.05))
        fallback_emotion = getattr(settings, "FER_FALLBACK_EMOTION", "neutral")
        use_fallback = getattr(settings, "FER_ENABLE_CONFIDENCE_FALLBACK", False)

        if (
            use_fallback
            and fallback_emotion in LABEL_TO_INDEX
            and (best_confidence < min_confidence or margin < min_margin)
        ):
            fallback_idx = LABEL_TO_INDEX[fallback_emotion]
            return fallback_emotion, float(probabilities[fallback_idx])
        return best_emotion, best_confidence

    def predict_from_bytes(self, content: bytes) -> PredictionResult:
        self._validate_size(content)
        self._load_model()
        started = time.perf_counter()
        try:
            feature_variants, image_hash = self._to_feature_vector(content)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise PredictionInputError("Unsupported or corrupted image file.") from exc

        weighted_scores = np.zeros(len(EMOTION_LABELS), dtype=np.float32)
        total_weight = 0.0
        for features, weight in feature_variants:
            if self._model is not None and hasattr(self._model, "predict_proba"):
                probabilities = self._model.predict_proba(features)[0]
            elif self._model is not None and hasattr(self._model, "predict"):
                image_features = features.reshape(1, 48, 48, 1)
                probabilities = self._model.predict(image_features, verbose=0)[0]
            else:
                probabilities = self._fallback_scores(features)
            normalized = self._normalize_probabilities(probabilities)
            weighted_scores += normalized * float(weight)
            total_weight += float(weight)

        if total_weight > 0:
            probabilities = weighted_scores / total_weight
        else:
            probabilities = np.full(len(EMOTION_LABELS), 1.0 / len(EMOTION_LABELS), dtype=np.float32)

        probabilities = self._normalize_probabilities(probabilities)
        probabilities = self._apply_probability_rules(probabilities)
        emotion, confidence = self._select_with_fallback(probabilities)
        latency_ms = int((time.perf_counter() - started) * 1000)
        scores = {label: float(probabilities[i]) for i, label in enumerate(EMOTION_LABELS)}
        return PredictionResult(
            emotion=emotion,
            confidence=confidence,
            scores=scores,
            model_version=self.model_version,
            latency_ms=latency_ms,
            image_sha256=image_hash,
        )

    def predict_from_base64(self, payload: str) -> PredictionResult:
        if not payload:
            raise PredictionInputError("Missing base64 image payload.")
        encoded = payload.split(",", 1)[1] if "," in payload else payload
        try:
            content = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise PredictionInputError("Invalid base64 image payload.") from exc
        return self.predict_from_bytes(content)


inference_service = EmotionInferenceService()
