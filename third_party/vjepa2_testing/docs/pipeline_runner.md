# End-to-End Pipeline Runner

The new runner (`src/run_pipeline.py`) lets you start training, save checkpoints, run inference, and upload the results with a single command driven by a JSON file. Each terminal session can point to a different config so multiple runs execute in parallel without interfering with one another.

## Quick start

1. Copy the sample config and edit it:
   ```bash
   cp configs/pipeline_run.sample.json my_run.json
   ```
2. Update the fields you care about (classes/S3 clip paths, checkpoint folder, ROI, inference segment, upload locations, etc.).
3. Launch the pipeline:
   ```bash
   python -m src.run_pipeline --config my_run.json
   ```
4. Training logs and checkpoints live under `artifacts/classifiers/<checkpoint_subdir>`. Annotated inference videos output to `artifacts/inference/<checkpoint_subdir>` before being uploaded to the configured S3 URI.
5. The runner automatically uploads the same JSON config to the `uploads.config_s3_path` location when the job finishes.

## Config reference

- `run_name` (string, optional): Friendly name for the run. Used as a fallback when `training.checkpoint_subdir` is omitted.
- `training`
  - `checkpoint_subdir` (string, required): Subfolder under `artifacts/classifiers/` for checkpoints/logs so independent runs never collide.
  - `freeze_pooler` (bool, optional): Mirrors the attention pooler flag from `constants.py`. `true` = freeze pooler and train only the head; `false` = end-to-end classifier training.
  - `enable_movement_augmentations` (bool, optional): Defaults to `true`. Set to `false` to disable the motion-related RandAugment ops (rotate, shear, vertical translate) while keeping all other augmentations active.
  - `class_s3_paths` (dict, required): Map of class label → one or more S3 prefixes that contain clips for that class.
- `inference`
  - `roi` (array[4], required): `[x, y, width, height]` ROI in normalized coordinates.
  - `video_s3_path` (string, required): Source video used for inference.
  - `start_time` / `duration` (floats, required): Segment boundaries in seconds.
  - `results_s3_path` (string, required): Destination prefix for the annotated video + JSON predictions.
  - `checkpoint_path` (string, optional): Manually override which checkpoint to load; defaults to the one just trained.
- `uploads`
  - `config_s3_path` (string, optional but recommended): Full S3 path where the runner uploads the exact JSON config used for traceability.

Use different config files (or at least different `checkpoint_subdir` values) per experiment to keep outputs isolated and to run jobs concurrently in separate terminals.
