import base64
import binascii
import hashlib
import io
import logging
import threading
import time
from dataclasses import dataclass
from typing import Final

import numpy as np
from django.conf import settings
from PIL import Image, ImageOps, UnidentifiedImageError

from inference.model_loader import LoadedArtifact, ModelSignatureError, load_verified_artifact
from ml.features import build_engineered_features

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
    model_loaded: bool
    degraded: bool


class EmotionInferenceService:
    def __init__(self) -> None:
        self._artifact: LoadedArtifact | None = None
        self._load_lock = threading.Lock()
        self._loaded = False
        self._load_error: str | None = None

    # ---- lifecycle --------------------------------------------------------

    def _load_model(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            try:
                self._artifact = load_verified_artifact()
                if self._artifact is None:
                    logger.warning("FER model artifact unavailable; serving degraded predictions")
                    self._load_error = "artifact_missing"
                else:
                    logger.info(
                        "Loaded FER model artifact",
                        extra={
                            "model_version": self._artifact.version,
                            "model_kind": self._artifact.kind,
                        },
                    )
            except ModelSignatureError as exc:
                logger.error("Refusing to load untrusted model artifact: %s", exc)
                self._artifact = None
                self._load_error = f"signature_error: {exc}"
            except Exception as exc:
                logger.exception("Unexpected error while loading FER model: %s", exc)
                self._artifact = None
                self._load_error = "load_error"
            finally:
                self._loaded = True

    def is_ready(self) -> bool:
        if not self._loaded:
            self._load_model()
        return self._artifact is not None

    def load_status(self) -> dict[str, object]:
        if not self._loaded:
            self._load_model()
        return {
            "loaded": self._artifact is not None,
            "model_version": self.current_model_version(),
            "kind": self._artifact.kind if self._artifact is not None else None,
            "error": self._load_error,
        }

    def current_model_version(self) -> str:
        if self._artifact is not None:
            return self._artifact.version
        return settings.FER_FALLBACK_MODEL_VERSION

    # ---- capability introspection ----------------------------------------

    def _input_spec(self) -> dict:
        if self._artifact is None:
            return {}
        manifest = getattr(self._artifact, "manifest", {}) or {}
        spec = manifest.get("input_spec")
        return spec if isinstance(spec, dict) else {}

    def _class_permutation(self) -> list[int] | None:
        if self._artifact is None:
            return None
        manifest = getattr(self._artifact, "manifest", {}) or {}
        perm = manifest.get("class_permutation")
        if isinstance(perm, list) and len(perm) == len(EMOTION_LABELS):
            return [int(i) for i in perm]
        return None

    def _is_vit_pipeline(self) -> bool:
        if self._artifact is None or self._artifact.kind != "onnx":
            return False
        spec = self._input_spec()
        return str(spec.get("image_mode", "")).upper() == "RGB"

    # ---- preprocessing ----------------------------------------------------

    @staticmethod
    def _validate_size(content: bytes) -> None:
        max_bytes = settings.FER_MAX_UPLOAD_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise PredictionInputError(
                f"File is too large. Maximum allowed is {settings.FER_MAX_UPLOAD_MB} MB."
            )

    @staticmethod
    def _to_feature_vector(content: bytes) -> tuple[list[tuple[np.ndarray, float]], str]:
        image_hash = hashlib.sha256(content).hexdigest()
        with Image.open(io.BytesIO(content)) as image:
            oriented = ImageOps.exif_transpose(image) or image
            gray = oriented.convert("L")
            variants = EmotionInferenceService._build_inference_variants(gray)
        return variants, image_hash

    @staticmethod
    def _build_inference_variants(gray_image: Image.Image) -> list[tuple[np.ndarray, float]]:
        width, height = gray_image.size
        if width < 24 or height < 24:
            resized = gray_image.resize((48, 48), Image.Resampling.BILINEAR)
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            features = build_engineered_features(arr[None, ...])
            return [(features, 1.0)]

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
            arr = np.asarray(
                candidate.resize((48, 48), Image.Resampling.BILINEAR), dtype=np.float32
            ) / 255.0
            features = build_engineered_features(arr[None, ...])
            variants.append((features, weight))
        return variants

    # ---- scoring ----------------------------------------------------------

    @staticmethod
    def _fallback_scores(features: np.ndarray) -> np.ndarray:
        flat = features.reshape(-1)
        brightness = float(np.mean(flat))
        contrast = float(np.std(flat))
        score = np.array(
            [
                max(0.01, (0.55 - brightness) + contrast),
                max(0.01, (0.45 - contrast) * 0.6),
                max(0.01, 0.35 + (0.5 - brightness)),
                max(0.01, brightness + 0.25),
                max(0.01, 0.45 + (0.3 - abs(0.5 - brightness))),
                max(0.01, 0.4 + (0.52 - brightness)),
                max(0.01, 0.35 + contrast),
            ],
            dtype=np.float32,
        )
        return score / np.sum(score)

    @staticmethod
    def _normalize_probabilities(probabilities) -> np.ndarray:
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
        angry = LABEL_TO_INDEX["angry"]
        happy = LABEL_TO_INDEX["happy"]
        neutral = LABEL_TO_INDEX["neutral"]
        disgust = LABEL_TO_INDEX["disgust"]
        surprise = LABEL_TO_INDEX["surprise"]
        fear = LABEL_TO_INDEX["fear"]

        if (
            adjusted[angry] >= adjusted[happy]
            and adjusted[happy] >= 0.30
            and (adjusted[angry] - adjusted[happy]) <= 0.18
        ):
            transfer = min(0.12, adjusted[angry] * 0.25)
            adjusted[angry] -= transfer
            adjusted[happy] += transfer

        if adjusted[disgust] >= 0.33 and adjusted[neutral] >= 0.22:
            transfer = min(0.07, adjusted[disgust] * 0.2)
            adjusted[disgust] -= transfer
            adjusted[neutral] += transfer

        if adjusted[fear] > adjusted[surprise] and (adjusted[fear] - adjusted[surprise]) <= 0.08:
            transfer = min(0.05, adjusted[fear] * 0.15)
            adjusted[fear] -= transfer
            adjusted[surprise] += transfer

        return EmotionInferenceService._normalize_probabilities(adjusted)

    @staticmethod
    def _select_with_fallback(probabilities: np.ndarray) -> tuple[str, float]:
        best_idx = int(np.argmax(probabilities))
        best_emotion = EMOTION_LABELS[best_idx]
        best_confidence = float(probabilities[best_idx])

        sorted_indices = np.argsort(probabilities)[::-1]
        second_confidence = (
            float(probabilities[int(sorted_indices[1])]) if len(sorted_indices) > 1 else 0.0
        )
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

    def _score_features(self, features: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return (normalized-probabilities, model_loaded)."""
        artifact = self._artifact
        if artifact is None:
            return self._normalize_probabilities(self._fallback_scores(features)), False

        model = artifact.model
        if artifact.kind == "sklearn" and hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            return self._normalize_probabilities(probs), True
        if artifact.kind == "onnx":
            input_name = model.get_inputs()[0].name
            reshaped = features.reshape(1, 48, 48, 1).astype(np.float32)
            output = model.run(None, {input_name: reshaped})[0]
            return self._normalize_probabilities(output[0]), True
        if hasattr(model, "predict"):
            reshaped = features.reshape(1, 48, 48, 1).astype(np.float32)
            probs = model.predict(reshaped, verbose=0)[0]
            return self._normalize_probabilities(probs), True
        # Unknown wrapper; degrade safely.
        return self._normalize_probabilities(self._fallback_scores(features)), False

    # ---- public API -------------------------------------------------------

    def _predict_vit(self, content: bytes) -> tuple[np.ndarray, bool, str]:
        """Run the ONNX ViT path. Returns (probabilities, loaded, image_hash)."""
        image_hash = hashlib.sha256(content).hexdigest()
        spec = self._input_spec()
        width = int(spec.get("width", 224))
        height = int(spec.get("height", 224))
        mean = np.asarray(spec.get("image_mean") or [0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray(spec.get("image_std") or [0.229, 0.224, 0.225], dtype=np.float32)

        try:
            with Image.open(io.BytesIO(content)) as image:
                oriented = ImageOps.exif_transpose(image) or image
                rgb = oriented.convert("RGB")
                resized = rgb.resize((width, height), Image.Resampling.BILINEAR)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise PredictionInputError("Unsupported or corrupted image file.") from exc

        arr = np.asarray(resized, dtype=np.float32) / 255.0  # HWC, 0..1
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        tensor = arr[None, ...].astype(np.float32)  # NCHW

        session = self._artifact.model  # type: ignore[union-attr]
        input_name = session.get_inputs()[0].name
        logits = session.run(None, {input_name: tensor})[0][0]

        perm = self._class_permutation()
        if perm is not None:
            logits = logits[perm]
        # softmax
        shifted = logits - float(np.max(logits))
        exps = np.exp(shifted)
        probabilities = (exps / float(np.sum(exps))).astype(np.float32)
        return self._normalize_probabilities(probabilities), True, image_hash

    def predict_from_bytes(self, content: bytes) -> PredictionResult:
        self._validate_size(content)
        self._load_model()
        started = time.perf_counter()

        if self._is_vit_pipeline():
            probabilities, model_loaded_any, image_hash = self._predict_vit(content)
        else:
            try:
                feature_variants, image_hash = self._to_feature_vector(content)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise PredictionInputError("Unsupported or corrupted image file.") from exc

            weighted = np.zeros(len(EMOTION_LABELS), dtype=np.float32)
            total_weight = 0.0
            model_loaded_any = False
            for features, weight in feature_variants:
                normalized, model_loaded = self._score_features(features)
                weighted += normalized * float(weight)
                total_weight += float(weight)
                model_loaded_any = model_loaded_any or model_loaded

            if total_weight > 0:
                probabilities = weighted / total_weight
            else:
                probabilities = np.full(
                    len(EMOTION_LABELS), 1.0 / len(EMOTION_LABELS), dtype=np.float32
                )

            probabilities = self._normalize_probabilities(probabilities)
            if model_loaded_any:
                # Rule-based nudges are tuned for the sklearn baseline's
                # confusion matrix; skip them for a trained ViT.
                probabilities = self._apply_probability_rules(probabilities)

        emotion, confidence = self._select_with_fallback(probabilities)
        latency_ms = int((time.perf_counter() - started) * 1000)
        scores = {label: float(probabilities[i]) for i, label in enumerate(EMOTION_LABELS)}

        degraded = not model_loaded_any
        version = (
            self._artifact.version
            if (self._artifact is not None and model_loaded_any)
            else settings.FER_FALLBACK_MODEL_VERSION
        )
        if degraded:
            logger.warning(
                "Served degraded prediction",
                extra={"image_sha256": image_hash, "latency_ms": latency_ms},
            )
        return PredictionResult(
            emotion=emotion,
            confidence=confidence,
            scores=scores,
            model_version=version,
            latency_ms=latency_ms,
            image_sha256=image_hash,
            model_loaded=model_loaded_any,
            degraded=degraded,
        )

    def predict_from_base64(self, payload: str) -> PredictionResult:
        if not payload:
            raise PredictionInputError("Missing base64 image payload.")
        if len(payload) > settings.FER_MAX_BASE64_BYTES:
            raise PredictionInputError("Base64 payload too large.")
        encoded = payload.split(",", 1)[1] if "," in payload else payload
        try:
            content = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise PredictionInputError("Invalid base64 image payload.") from exc
        return self.predict_from_bytes(content)


inference_service = EmotionInferenceService()
