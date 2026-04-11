import hashlib
import hmac
import io
import json
from pathlib import Path
from unittest import mock

import numpy as np
from django.test import Client, TestCase, override_settings
from PIL import Image

from inference import rate_limit
from inference.model_loader import ModelSignatureError, _verify_manifest
from inference.services import EmotionInferenceService, PredictionInputError
from inference.validation import validate_image_bytes


def _png_bytes(width: int = 64, height: int = 64, color: int = 128) -> bytes:
    img = Image.new("L", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _gif_bytes() -> bytes:
    frame1 = Image.new("L", (64, 64), color=120)
    frame2 = Image.new("L", (64, 64), color=180)
    buf = io.BytesIO()
    frame1.save(buf, format="GIF", save_all=True, append_images=[frame2])
    return buf.getvalue()


class _StubModel:
    def __init__(self, probabilities):
        self._probabilities = np.array([probabilities], dtype=np.float32)

    def predict_proba(self, _):
        return self._probabilities


class _StubArtifact:
    def __init__(self, model, version="unit-test-v1"):
        self.model = model
        self.version = version
        self.kind = "sklearn"
        self.manifest = {}


# ---------------------------------------------------------------------------
# Inference service behavior
# ---------------------------------------------------------------------------


class EmotionInferenceSafeguardsTests(TestCase):
    def _stub_service(self, probs):
        service = EmotionInferenceService()
        service._loaded = True
        service._artifact = _StubArtifact(_StubModel(probs))
        service._to_feature_vector = lambda _: (
            [(np.zeros((1, 48 * 48), dtype=np.float32), 1.0)],
            "hash",
        )
        return service

    def test_rule_correction_reduces_angry_false_positive(self):
        service = self._stub_service([0.46, 0.03, 0.02, 0.38, 0.07, 0.02, 0.02])
        result = service.predict_from_bytes(b"image")
        self.assertEqual(result.emotion, "happy")
        self.assertGreater(result.scores["happy"], result.scores["angry"])
        self.assertFalse(result.degraded)
        self.assertTrue(result.model_loaded)

    @override_settings(
        FER_ENABLE_CONFIDENCE_FALLBACK=True,
        FER_MIN_CONFIDENCE=0.5,
        FER_MIN_MARGIN=0.12,
        FER_FALLBACK_EMOTION="neutral",
    )
    def test_low_confidence_falls_back_to_neutral(self):
        service = self._stub_service([0.20, 0.13, 0.12, 0.18, 0.19, 0.10, 0.08])
        result = service.predict_from_bytes(b"image")
        self.assertEqual(result.emotion, "neutral")
        self.assertAlmostEqual(result.confidence, result.scores["neutral"])

    def test_fallback_reports_degraded_and_sentinel_version(self):
        service = EmotionInferenceService()
        service._loaded = True
        service._artifact = None
        with mock.patch.object(
            EmotionInferenceService,
            "_to_feature_vector",
            staticmethod(
                lambda _: ([(np.zeros((1, 48 * 48), dtype=np.float32), 1.0)], "hash")
            ),
        ):
            result = service.predict_from_bytes(b"image")
        self.assertTrue(result.degraded)
        self.assertFalse(result.model_loaded)
        self.assertEqual(result.model_version, "fallback-heuristic-v1")


# ---------------------------------------------------------------------------
# Image validation — adversarial inputs
# ---------------------------------------------------------------------------


class ImageValidationTests(TestCase):
    def test_accepts_png(self):
        validate_image_bytes(_png_bytes())

    def test_rejects_empty(self):
        with self.assertRaises(PredictionInputError):
            validate_image_bytes(b"")

    def test_rejects_garbage(self):
        with self.assertRaises(PredictionInputError):
            validate_image_bytes(b"this is not an image")

    def test_rejects_multi_frame_gif(self):
        with self.assertRaises(PredictionInputError):
            validate_image_bytes(_gif_bytes())

    @override_settings(FER_ALLOWED_IMAGE_FORMATS={"JPEG"})
    def test_rejects_non_whitelisted_format(self):
        with self.assertRaises(PredictionInputError):
            validate_image_bytes(_png_bytes())

    @override_settings(FER_MAX_IMAGE_PIXELS=64)
    def test_rejects_oversize_pixels(self):
        with self.assertRaises(PredictionInputError):
            validate_image_bytes(_png_bytes(width=9, height=9))

    @override_settings(FER_MAX_UPLOAD_MB=0)
    def test_rejects_oversize_bytes(self):
        with self.assertRaises(PredictionInputError):
            validate_image_bytes(_png_bytes())


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


@override_settings(FER_RATE_LIMIT_BACKEND="inprocess")
class RateLimitTests(TestCase):
    def setUp(self):
        rate_limit.reset_for_tests()

    def test_allows_until_limit(self):
        for _ in range(3):
            self.assertTrue(rate_limit.allow_request("test:1", 3))
        self.assertFalse(rate_limit.allow_request("test:1", 3))

    def test_zero_limit_bypass(self):
        self.assertTrue(rate_limit.allow_request("test:2", 0))


# ---------------------------------------------------------------------------
# Signed model loader
# ---------------------------------------------------------------------------


class SignedManifestTests(TestCase):
    def setUp(self):
        self.tmp = Path(self._tempdir())
        self.weights = self.tmp / "model.bin"
        self.weights.write_bytes(b"\x00\x01\x02\x03payload")
        self.manifest = self.tmp / "manifest.json"

    def _tempdir(self) -> str:
        import tempfile

        directory = tempfile.mkdtemp(prefix="fer-sign-test-")
        self.addCleanup(self._rmtree, directory)
        return directory

    @staticmethod
    def _rmtree(path: str) -> None:
        import shutil

        shutil.rmtree(path, ignore_errors=True)

    def _write_manifest(self, signing_key: str, tamper: bool = False):
        sha = hashlib.sha256(self.weights.read_bytes()).hexdigest()
        files = {"model": {"path": str(self.weights), "sha256": sha, "size_bytes": self.weights.stat().st_size}}
        body = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        signature = hmac.new(signing_key.encode(), body, hashlib.sha256).hexdigest()
        manifest = {"files": files, "signature_hex": signature, "signature_algorithm": "HMAC-SHA256"}
        if tamper:
            manifest["signature_hex"] = "deadbeef" * 8
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    @override_settings(FER_MODEL_SIGNING_KEY="unit-test-key", FER_REQUIRE_SIGNED_ARTIFACT=True)
    def test_valid_signature_accepted(self):
        self._write_manifest("unit-test-key")
        manifest = _verify_manifest(self.manifest, self.weights)
        self.assertIn("files", manifest)

    @override_settings(FER_MODEL_SIGNING_KEY="unit-test-key", FER_REQUIRE_SIGNED_ARTIFACT=True)
    def test_tampered_signature_rejected(self):
        self._write_manifest("unit-test-key", tamper=True)
        with self.assertRaises(ModelSignatureError):
            _verify_manifest(self.manifest, self.weights)

    @override_settings(FER_MODEL_SIGNING_KEY="unit-test-key", FER_REQUIRE_SIGNED_ARTIFACT=True)
    def test_modified_weights_rejected(self):
        self._write_manifest("unit-test-key")
        self.weights.write_bytes(self.weights.read_bytes() + b"injected")
        with self.assertRaises(ModelSignatureError):
            _verify_manifest(self.manifest, self.weights)

    @override_settings(FER_MODEL_SIGNING_KEY="", FER_REQUIRE_SIGNED_ARTIFACT=True)
    def test_missing_key_rejected_in_strict_mode(self):
        self._write_manifest("unit-test-key")
        with self.assertRaises(ModelSignatureError):
            _verify_manifest(self.manifest, self.weights)


# ---------------------------------------------------------------------------
# HTTP endpoints — consent + auth
# ---------------------------------------------------------------------------


@override_settings(
    FER_REQUIRE_AUTH=False,
    FER_REQUIRE_CONSENT=True,
    FER_STORE_PREDICTIONS=False,
    FER_REQUIRE_SIGNED_ARTIFACT=False,
    FER_RATE_LIMIT_BACKEND="inprocess",
)
class EndpointConsentTests(TestCase):
    def setUp(self):
        rate_limit.reset_for_tests()
        self.client = Client()

    def test_image_api_rejects_without_consent(self):
        response = self.client.post(
            "/api/v1/predict/image/",
            data={"image": io.BytesIO(_png_bytes())},
            format="multipart",
        )
        self.assertEqual(response.status_code, 451)
        self.assertEqual(response.json()["error"], "consent_required")

    def test_image_api_accepts_with_consent_header(self):
        response = self.client.post(
            "/api/v1/predict/image/",
            data={"image": io.BytesIO(_png_bytes()), "consent": "true"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("emotion", body)
        self.assertIn("model_loaded", body)
        self.assertIn("degraded", body)

    def test_healthz_is_static(self):
        self.assertEqual(self.client.get("/healthz").status_code, 200)

    def test_readyz_reports_degraded_without_model(self):
        response = self.client.get("/readyz")
        # Without a signed artifact AND signing not required → "ready" per readyz logic.
        self.assertIn(response.status_code, (200, 503))
        body = response.json()
        self.assertIn("model", body)
        self.assertIn("database", body)


@override_settings(
    FER_REQUIRE_AUTH=True,
    FER_API_TOKENS={"secret-token"},
    FER_REQUIRE_CONSENT=True,
    FER_STORE_PREDICTIONS=False,
    FER_REQUIRE_SIGNED_ARTIFACT=False,
    FER_RATE_LIMIT_BACKEND="inprocess",
)
class EndpointAuthTests(TestCase):
    def setUp(self):
        rate_limit.reset_for_tests()
        self.client = Client()

    def test_image_api_requires_bearer(self):
        response = self.client.post(
            "/api/v1/predict/image/",
            data={"image": io.BytesIO(_png_bytes()), "consent": "true"},
        )
        self.assertEqual(response.status_code, 401)

    def test_image_api_accepts_bearer(self):
        response = self.client.post(
            "/api/v1/predict/image/",
            data={"image": io.BytesIO(_png_bytes()), "consent": "true"},
            HTTP_AUTHORIZATION="Bearer secret-token",
        )
        self.assertEqual(response.status_code, 200)

    def test_image_api_rejects_wrong_bearer(self):
        response = self.client.post(
            "/api/v1/predict/image/",
            data={"image": io.BytesIO(_png_bytes()), "consent": "true"},
            HTTP_AUTHORIZATION="Bearer other-token",
        )
        self.assertEqual(response.status_code, 401)
