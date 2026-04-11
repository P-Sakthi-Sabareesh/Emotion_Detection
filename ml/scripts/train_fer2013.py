#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from PIL import Image
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The sys.path manipulation above is required because this file is run as a
# script (python ml/scripts/train_fer2013.py), not imported as a module — so
# the ml/ package isn't on sys.path until we put it there.
from ml.features import build_engineered_features  # noqa: E402
from ml.model_contract import (  # noqa: E402
    ModelCalibrationSpec,
    ModelTrainingSpec,
    SklearnEmotionModel,
)

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def load_split(split_dir: Path, max_per_class: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    images = []
    targets = []
    for idx, emotion in enumerate(EMOTIONS):
        class_dir = split_dir / emotion
        if not class_dir.exists():
            continue
        files = sorted([p for p in class_dir.iterdir() if p.is_file()])
        if max_per_class:
            files = files[:max_per_class]
        for file_path in files:
            with Image.open(file_path) as image:
                gray = image.convert("L").resize((48, 48), Image.Resampling.BILINEAR)
                arr = np.asarray(gray, dtype=np.float32) / 255.0
                images.append(arr)
                targets.append(idx)
    if not images:
        return np.empty((0, 48, 48), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.asarray(images, dtype=np.float32), np.asarray(targets, dtype=np.int32)


def build_features(images: np.ndarray) -> np.ndarray:
    # Delegated to ml.features so the runtime and the trainer produce
    # byte-identical feature vectors.
    return build_engineered_features(images)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def probs_from_estimator(estimator: object, features: np.ndarray) -> np.ndarray:
    def _stabilize_probs(raw_probs: np.ndarray) -> np.ndarray:
        probs = np.nan_to_num(raw_probs, nan=1e-8, posinf=1.0, neginf=1e-8)
        probs = np.clip(probs, 1e-8, 1.0)
        denom = np.sum(probs, axis=1, keepdims=True)
        denom[denom <= 0.0] = 1.0
        return probs / denom

    if hasattr(estimator, "predict_proba"):
        probs = np.asarray(estimator.predict_proba(features), dtype=np.float64)
        return _stabilize_probs(probs)
    if hasattr(estimator, "decision_function"):
        decision = np.asarray(estimator.decision_function(features), dtype=np.float64)
        if decision.ndim == 1:
            decision = np.stack([-decision, decision], axis=1)
        return _stabilize_probs(softmax(decision))
    raise TypeError(f"Estimator {type(estimator).__name__} does not expose probabilities or decision_function")


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correctness = (predictions == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        low = edges[idx]
        high = edges[idx + 1]
        mask = (confidences > low) & (confidences <= high)
        if not np.any(mask):
            continue
        bin_acc = np.mean(correctness[mask])
        bin_conf = np.mean(confidences[mask])
        ece += abs(bin_acc - bin_conf) * (np.sum(mask) / len(y_true))
    return float(ece)


def multiclass_brier_score(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    y_one_hot = np.eye(num_classes, dtype=np.float64)[y_true]
    return float(np.mean(np.sum((y_one_hot - probs) ** 2, axis=1)))


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, labels: list[str]) -> dict[str, object]:
    preds = np.argmax(probs, axis=1)
    top2 = np.argsort(probs, axis=1)[:, -2:]
    top2_correct = np.any(top2 == y_true[:, None], axis=1)
    report = classification_report(
        y_true,
        preds,
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted")),
        "top2_accuracy": float(np.mean(top2_correct)),
        "mean_max_confidence": float(np.mean(np.max(probs, axis=1))),
        "ece_15": expected_calibration_error(y_true, probs, bins=15),
        "brier_score": multiclass_brier_score(y_true, probs, len(labels)),
        "nll": float(log_loss(y_true, probs, labels=np.arange(len(labels)))),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        "classification_report": report,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_candidate_estimators(random_state: int) -> dict[str, object]:
    return {
        "sgd_log_elasticnet": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        penalty="elasticnet",
                        alpha=3e-5,
                        l1_ratio=0.15,
                        max_iter=2200,
                        tol=1e-4,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=12,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "sgd_modified_huber": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SGDClassifier(
                        loss="modified_huber",
                        penalty="elasticnet",
                        alpha=5e-5,
                        l1_ratio=0.2,
                        max_iter=2200,
                        tol=1e-4,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=12,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "linear_svc_margin": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        dual="auto",
                        max_iter=1500,
                        tol=1e-3,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    # Flush stdout line-by-line so background/CI log tails update in real time.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:  # nosec B110
        # sys.stdout.reconfigure is available on Python 3.7+ but may fail if
        # stdout is wrapped by the harness. Default buffered output is fine
        # as a fallback; we silently proceed.
        pass

    parser = argparse.ArgumentParser(description="Train FER-2013 model with validation and calibration.")
    parser.add_argument("--dataset-root", default="data/raw")
    parser.add_argument("--output-dir", default="ml/artifacts")
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--model-version", default="fer2013-sklearn-v2")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--calibration-method", choices=["sigmoid", "isotonic"], default="sigmoid")
    parser.add_argument("--calibration-cv", type=int, default=3)
    parser.add_argument("--calibrate-probabilities", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument(
        "--candidates",
        default="sgd_log_elasticnet,sgd_modified_huber",
        help=(
            "Comma-separated subset of candidate estimators to train. "
            "Known names: sgd_log_elasticnet, sgd_modified_huber, linear_svc_margin. "
            "LinearSVC is off by default because it lacks early stopping and can run "
            "for many minutes on high-dim features without converging."
        ),
    )
    args = parser.parse_args()

    wanted = {name.strip() for name in args.candidates.split(",") if name.strip()}

    dataset_root = Path(args.dataset_root).resolve()
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise SystemExit(
            f"Expected dataset folders not found at {dataset_root}. "
            "Run download_fer2013.py first."
        )

    max_per_class = args.max_per_class if args.max_per_class > 0 else None

    print("Loading train split...")
    train_images, y_train = load_split(train_dir, max_per_class=max_per_class)
    print(f"Train samples: {len(train_images)}")

    print("Loading test split...")
    test_images, y_test = load_split(test_dir, max_per_class=max_per_class)
    print(f"Test samples: {len(test_images)}")

    if len(train_images) == 0 or len(test_images) == 0:
        raise SystemExit("Dataset appears empty. Verify FER-2013 extraction and class folders.")

    train_idx = np.arange(len(train_images))
    try:
        train_sub_idx, val_idx = train_test_split(
            train_idx,
            test_size=args.validation_ratio,
            random_state=args.random_state,
            stratify=y_train,
        )
    except ValueError:
        train_sub_idx, val_idx = train_test_split(
            train_idx,
            test_size=args.validation_ratio,
            random_state=args.random_state,
            stratify=None,
        )
    train_sub_images = train_images[train_sub_idx]
    y_train_sub = y_train[train_sub_idx]
    val_images = train_images[val_idx]
    y_val = y_train[val_idx]

    print("Building engineered feature vectors...")
    x_train_sub = build_features(train_sub_images)
    x_val = build_features(val_images)
    x_train_full = build_features(train_images)
    x_test = build_features(test_images)

    print(f"Feature dimension: {x_train_sub.shape[1]}")
    print("Running validation model selection...")
    all_candidates = build_candidate_estimators(random_state=args.random_state)
    unknown = wanted - set(all_candidates)
    if unknown:
        raise SystemExit(f"Unknown --candidates entries: {sorted(unknown)}. Known: {sorted(all_candidates)}")
    candidates = {name: estimator for name, estimator in all_candidates.items() if name in wanted}
    if not candidates:
        raise SystemExit("No candidates selected; pass --candidates with at least one name.")
    candidate_rows: list[dict[str, object]] = []
    selected_name: str | None = None
    selected_model: object | None = None
    selected_score = -1.0
    for name, estimator in candidates.items():
        model = clone(estimator)
        model.fit(x_train_sub, y_train_sub)
        probs_val = probs_from_estimator(model, x_val)
        metrics_val = compute_metrics(y_val, probs_val, EMOTIONS)
        macro_f1 = float(metrics_val["macro_f1"])
        candidate_rows.append(
            {
                "name": name,
                "validation_accuracy": float(metrics_val["accuracy"]),
                "validation_macro_f1": macro_f1,
                "validation_weighted_f1": float(metrics_val["weighted_f1"]),
                "validation_ece_15": float(metrics_val["ece_15"]),
                "validation_nll": float(metrics_val["nll"]),
            }
        )
        print(f"  {name}: val_macro_f1={macro_f1:.4f} val_acc={metrics_val['accuracy']:.4f}")
        if macro_f1 > selected_score:
            selected_score = macro_f1
            selected_name = name
            selected_model = estimator

    if selected_model is None or selected_name is None:
        raise SystemExit("No valid candidate model was trained.")

    print(f"Selected candidate: {selected_name} (validation_macro_f1={selected_score:.4f})")

    print("Training final model on full train split...")
    final_model = clone(selected_model)
    final_model.fit(x_train_full, y_train)

    deployed_model: object = final_model
    calibration_spec = ModelCalibrationSpec(enabled=False)
    if args.calibrate_probabilities:
        print(f"Applying probability calibration ({args.calibration_method}, cv={args.calibration_cv})...")
        calibrator = CalibratedClassifierCV(
            estimator=clone(selected_model),
            method=args.calibration_method,
            cv=args.calibration_cv,
            n_jobs=args.jobs,
        )
        calibrator.fit(x_train_full, y_train)
        deployed_model = calibrator
        val_probs_calibrated = probs_from_estimator(calibrator, x_val)
        val_cal_metrics = compute_metrics(y_val, val_probs_calibrated, EMOTIONS)
        calibration_spec = ModelCalibrationSpec(
            enabled=True,
            method=args.calibration_method,
            cv=args.calibration_cv,
            validation_ece=float(val_cal_metrics["ece_15"]),
            validation_brier=float(val_cal_metrics["brier_score"]),
            validation_nll=float(val_cal_metrics["nll"]),
        )

    val_probs = probs_from_estimator(deployed_model, x_val)
    test_probs = probs_from_estimator(deployed_model, x_test)
    val_metrics = compute_metrics(y_val, val_probs, EMOTIONS)
    test_metrics = compute_metrics(y_test, test_probs, EMOTIONS)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    wrapped = SklearnEmotionModel(
        model=deployed_model,
        labels=EMOTIONS,
        model_version=args.model_version,
        training_spec=ModelTrainingSpec(
            algorithm=selected_name,
            random_state=args.random_state,
            train_samples=int(len(y_train_sub)),
            validation_samples=int(len(y_val)),
            test_samples=int(len(y_test)),
            selected_score=selected_score,
            class_weight="balanced",
            feature_spec="raw+gradient+pooled(48x48 grayscale)",
        ),
        calibration_spec=calibration_spec,
        extra_metadata={
            "dataset_root": str(dataset_root),
            "calibration_requested": bool(args.calibrate_probabilities),
            "candidate_count": len(candidate_rows),
        },
    )
    model_path = output_dir / "emotion_classifier.joblib"
    joblib.dump(wrapped, model_path)

    metadata_path = output_dir / "model_metadata.json"
    metadata = wrapped.to_metadata()
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics = {
        "model_version": args.model_version,
        "contract_schema_version": wrapped.contract_schema_version,
        "dataset_root": str(dataset_root),
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "class_labels": EMOTIONS,
        "model_selection": {
            "selected_candidate": selected_name,
            "selected_metric": "validation_macro_f1",
            "selected_score": selected_score,
            "candidates": candidate_rows,
        },
        "calibration": metadata["calibration_spec"],
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "accuracy": float(test_metrics["accuracy"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "confusion_matrix": test_metrics["confusion_matrix"],
        "classification_report": test_metrics["classification_report"],
        "mean_max_confidence": float(test_metrics["mean_max_confidence"]),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    manifest = {
        "artifact_contract_version": "1",
        "model_version": args.model_version,
        "files": {
            "model": {
                "path": str(model_path),
                "sha256": sha256_file(model_path),
                "size_bytes": model_path.stat().st_size,
            },
            "metadata": {
                "path": str(metadata_path),
                "sha256": sha256_file(metadata_path),
                "size_bytes": metadata_path.stat().st_size,
            },
            "metrics": {
                "path": str(metrics_path),
                "sha256": sha256_file(metrics_path),
                "size_bytes": metrics_path.stat().st_size,
            },
        },
    }
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Validation Macro-F1: {val_metrics['macro_f1']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f} | Test Macro-F1: {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
