"""Shared feature engineering for FER-2013 sklearn baseline.

Both the training pipeline (``ml/scripts/train_fer2013.py``) and the runtime
inference service (``inference/services.py``) must build the exact same
feature vector — otherwise a model trained on 5762-dim engineered features
will fail at inference on 2304-dim raw pixels.  Keep one authoritative copy
here.
"""
from __future__ import annotations

import numpy as np

ENGINEERED_FEATURE_DIM = 2304 + 2304 + 576 + 576 + 1 + 1  # 5762
INPUT_SHAPE = (48, 48)


def build_engineered_features(images: np.ndarray) -> np.ndarray:
    """Compute per-sample engineered features.

    Parameters
    ----------
    images : np.ndarray
        Shape ``(n_samples, 48, 48)``, dtype float32, values in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Shape ``(n_samples, 5762)``, dtype float32.  Concatenation of:
        raw pixels, gradient magnitude, 2x2-pooled raw, 2x2-pooled gradient,
        per-sample mean intensity, per-sample contrast.
    """
    if images.ndim != 3 or images.shape[1:] != INPUT_SHAPE:
        raise ValueError(
            f"Expected shape (n, 48, 48), got {images.shape!r}"
        )
    arr = images.astype(np.float32, copy=False)
    n = arr.shape[0]

    raw = arr.reshape(n, -1)
    grad_x = np.diff(arr, axis=2, append=arr[:, :, -1:])
    grad_y = np.diff(arr, axis=1, append=arr[:, -1:, :])
    grad_mag = np.sqrt((grad_x * grad_x) + (grad_y * grad_y)).astype(np.float32, copy=False)
    pooled = arr.reshape(n, 24, 2, 24, 2).mean(axis=(2, 4))
    pooled_grad = grad_mag.reshape(n, 24, 2, 24, 2).mean(axis=(2, 4))
    intensity = raw.mean(axis=1, keepdims=True)
    contrast = raw.std(axis=1, keepdims=True)

    features = np.concatenate(
        [
            raw,
            grad_mag.reshape(n, -1),
            pooled.reshape(n, -1),
            pooled_grad.reshape(n, -1),
            intensity,
            contrast,
        ],
        axis=1,
    )
    return features.astype(np.float32, copy=False)
