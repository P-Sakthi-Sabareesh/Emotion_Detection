# Sample Face Sources

Sample gallery images are now generated from FER-2013 test split to keep demo inputs consistent with model training data.

- Dataset: `Kaggle msambare/fer2013`
- Selection method:
  - For each target class (`happy`, `sad`, `angry`, `surprise`, `neutral`), choose a test image that the current deployed model predicts correctly with the highest confidence.
  - Upscale from `48x48` to `240x240` for UI display.

Generated artifact:
- `static/samples/sample_manifest.json` (includes expected class, source path, confidence)

Regeneration command:
- `python ml/scripts/prepare_demo_samples.py --dataset-root data/raw --output-dir static/samples`
