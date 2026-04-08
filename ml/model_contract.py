from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

CONTRACT_SCHEMA_VERSION = "1.1.0"


@dataclass
class ModelInputSpec:
    image_mode: str = "L"
    width: int = 48
    height: int = 48
    flatten: bool = True
    value_range: tuple[float, float] = (0.0, 1.0)


@dataclass
class ModelCalibrationSpec:
    enabled: bool = False
    method: str = "none"
    cv: int = 0
    validation_ece: float | None = None
    validation_brier: float | None = None
    validation_nll: float | None = None


@dataclass
class ModelTrainingSpec:
    algorithm: str
    random_state: int
    train_samples: int
    validation_samples: int
    test_samples: int
    selected_on_metric: str = "validation_macro_f1"
    selected_score: float | None = None
    class_weight: str | None = None
    feature_spec: str = "raw_48x48_grayscale"


@dataclass
class SklearnEmotionModel:
    model: Any
    labels: list[str]
    model_version: str
    contract_schema_version: str = CONTRACT_SCHEMA_VERSION
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()
    )
    input_spec: ModelInputSpec = field(default_factory=ModelInputSpec)
    training_spec: ModelTrainingSpec | None = None
    calibration_spec: ModelCalibrationSpec = field(default_factory=ModelCalibrationSpec)
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix; got shape={features.shape!r}")
        return self.model.predict_proba(features)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "contract_schema_version": self.contract_schema_version,
            "model_version": self.model_version,
            "labels": list(self.labels),
            "created_at_utc": self.created_at_utc,
            "input_spec": asdict(self.input_spec),
            "training_spec": asdict(self.training_spec) if self.training_spec else None,
            "calibration_spec": asdict(self.calibration_spec),
            "extra_metadata": dict(self.extra_metadata),
        }
