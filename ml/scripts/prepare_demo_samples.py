#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

TARGET_SAMPLES = ["happy", "sad", "angry", "surprise", "neutral"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
from inference.services import inference_service  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare stable demo samples from FER-2013 test split.")
    parser.add_argument("--dataset-root", default="data/raw")
    parser.add_argument("--output-dir", default="static/samples")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for idx, emotion in enumerate(TARGET_SAMPLES, start=1):
        class_dir = dataset_root / "test" / emotion
        if not class_dir.exists():
            raise SystemExit(f"Missing class directory: {class_dir}")
        best_path = None
        best_conf = -1.0
        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file():
                continue
            content = image_path.read_bytes()
            result = inference_service.predict_from_bytes(content)
            pred_label = result.emotion
            conf = float(result.confidence)
            if pred_label == emotion and conf > best_conf:
                best_conf = conf
                best_path = image_path
        if best_path is None:
            raise SystemExit(f"Could not find stable sample for class '{emotion}'.")

        out_path = output_dir / f"sample-{idx}.jpg"
        shutil.copyfile(best_path, out_path)

        selected.append(
            {
                "sample_file": f"sample-{idx}.jpg",
                "expected_emotion": emotion,
                "source_dataset_path": str(best_path.relative_to(dataset_root)),
                "model_confidence": round(best_conf, 4),
            }
        )

    manifest_path = output_dir / "sample_manifest.json"
    manifest_path.write_text(json.dumps({"samples": selected}, indent=2), encoding="utf-8")
    print(f"Saved demo samples and manifest to {output_dir}")


if __name__ == "__main__":
    main()
