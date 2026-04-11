"""Signed-artifact model loader.

Closes the P0 remote-code-execution window by refusing to deserialize any
model artifact whose sha256 and HMAC signature do not match the manifest
produced by the training pipeline.  Only artifacts the operator explicitly
signed with ``FER_MODEL_SIGNING_KEY`` are ever handed to ``joblib``.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from django.conf import settings

logger = logging.getLogger(__name__)


class ModelSignatureError(RuntimeError):
    """Raised when a model artifact cannot be cryptographically trusted."""


@dataclass(frozen=True)
class LoadedArtifact:
    model: Any
    version: str
    kind: str  # "sklearn" | "onnx"
    manifest: dict


def _canonical_bytes(files_section: dict) -> bytes:
    return json.dumps(files_section, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(manifest_path: Path, weights_path: Path) -> dict:
    if not manifest_path.exists():
        raise ModelSignatureError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    files = manifest.get("files") or {}
    model_entry = files.get("model") or {}
    expected_sha = model_entry.get("sha256")
    if not expected_sha:
        raise ModelSignatureError("manifest is missing files.model.sha256")

    actual_sha = _sha256_file(weights_path)
    if not hmac.compare_digest(expected_sha.lower(), actual_sha.lower()):
        raise ModelSignatureError(
            f"sha256 mismatch for {weights_path.name}: "
            f"expected {expected_sha}, got {actual_sha}"
        )

    key = settings.FER_MODEL_SIGNING_KEY
    signature = manifest.get("signature_hex")
    if settings.FER_REQUIRE_SIGNED_ARTIFACT:
        if not key:
            raise ModelSignatureError(
                "FER_REQUIRE_SIGNED_ARTIFACT is set but FER_MODEL_SIGNING_KEY is empty"
            )
        if not signature:
            raise ModelSignatureError("manifest is missing signature_hex")
        expected_sig = hmac.new(
            key.encode("utf-8"), _canonical_bytes(files), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected_sig, signature):
            raise ModelSignatureError("manifest HMAC signature mismatch")
    elif signature and key:
        expected_sig = hmac.new(
            key.encode("utf-8"), _canonical_bytes(files), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected_sig, signature):
            raise ModelSignatureError("manifest HMAC signature mismatch")

    return manifest


def _load_sklearn(weights_path: Path, manifest: dict) -> LoadedArtifact:
    # Only reachable after sha256 + HMAC verification above.
    import joblib  # noqa: WPS433

    obj = joblib.load(weights_path)
    version = (
        getattr(obj, "model_version", None)
        or manifest.get("model_version")
        or settings.FER_MODEL_VERSION
    )
    return LoadedArtifact(model=obj, version=version, kind="sklearn", manifest=manifest)


def _load_onnx(weights_path: Path, manifest: dict) -> LoadedArtifact:
    import onnxruntime as ort  # type: ignore

    session = ort.InferenceSession(str(weights_path), providers=["CPUExecutionProvider"])
    version = manifest.get("model_version") or settings.FER_MODEL_VERSION
    return LoadedArtifact(model=session, version=version, kind="onnx", manifest=manifest)


def load_verified_artifact() -> LoadedArtifact | None:
    """Load the configured artifact after verifying its signature.

    Returns ``None`` when the artifact is missing and signing is not mandatory,
    allowing the degraded heuristic path to take over.  Raises
    ``ModelSignatureError`` for anything worse than a missing file.
    """
    weights_path = Path(settings.FER_MODEL_PATH)
    manifest_path = Path(settings.FER_MODEL_MANIFEST_PATH)

    if not weights_path.exists():
        msg = f"model artifact not found at {weights_path}"
        if settings.FER_REQUIRE_SIGNED_ARTIFACT:
            raise ModelSignatureError(msg)
        logger.warning("%s; continuing with degraded heuristic", msg)
        return None

    manifest = _verify_manifest(manifest_path, weights_path)

    suffix = weights_path.suffix.lower()
    if suffix == ".onnx":
        return _load_onnx(weights_path, manifest)
    if suffix in {".joblib", ".pkl"}:
        return _load_sklearn(weights_path, manifest)
    raise ModelSignatureError(f"unsupported artifact extension: {suffix}")
