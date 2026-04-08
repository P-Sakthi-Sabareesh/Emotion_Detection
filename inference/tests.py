import numpy as np
from django.test import TestCase, override_settings

from inference.services import EmotionInferenceService


class _StubModel:
    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = np.array([probabilities], dtype=np.float32)

    def predict_proba(self, _: np.ndarray) -> np.ndarray:
        return self._probabilities


class EmotionInferenceSafeguardsTests(TestCase):
    def test_rule_correction_reduces_angry_false_positive(self) -> None:
        service = EmotionInferenceService()
        service._loaded = True
        service._model = _StubModel([0.46, 0.03, 0.02, 0.38, 0.07, 0.02, 0.02])
        service._to_feature_vector = lambda _: (  # type: ignore[method-assign]
            [(np.zeros((1, 48 * 48), dtype=np.float32), 1.0)],
            "hash",
        )

        result = service.predict_from_bytes(b"image")

        self.assertEqual(result.emotion, "happy")
        self.assertGreater(result.scores["happy"], result.scores["angry"])

    @override_settings(
        FER_ENABLE_CONFIDENCE_FALLBACK=True,
        FER_MIN_CONFIDENCE=0.5,
        FER_MIN_MARGIN=0.12,
        FER_FALLBACK_EMOTION="neutral",
    )
    def test_low_confidence_prediction_falls_back_to_neutral(self) -> None:
        service = EmotionInferenceService()
        service._loaded = True
        service._model = _StubModel([0.20, 0.13, 0.12, 0.18, 0.19, 0.10, 0.08])
        service._to_feature_vector = lambda _: (  # type: ignore[method-assign]
            [(np.zeros((1, 48 * 48), dtype=np.float32), 1.0)],
            "hash",
        )

        result = service.predict_from_bytes(b"image")

        self.assertEqual(result.emotion, "neutral")
        self.assertAlmostEqual(result.confidence, result.scores["neutral"])
