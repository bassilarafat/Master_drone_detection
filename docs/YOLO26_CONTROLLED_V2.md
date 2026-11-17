# Controlled YOLO26 v2 — prospective execution, 16 November 2026

The seed0 historical-settings campaign was stopped at the researcher's request. Its completed epochs, log files, `best.pt`, `last.pt`, and `results.csv` remain in place; [the stop record](../04_experiments/clean_v2_standard_protocol/EXPLORATORY_STOP_RECORD.json) records checkpoint hashes and the classification **Exploratory E0 / seed0**. The old campaign is excluded from the approved three-seed controlled aggregates, and its launcher script now refuses to restart a retired campaign.

This new version resolves execution controls prospectively. It preserves the [approved analysis plan](../configs/paper/analysis_plan_yolo26_v1.json), all-seed reporting, primary E3–E2 comparison, and Validation-only development. Older protocols and the existing manuscript remain historical snapshots; no saved result is rewritten as a controlled result.

## Fixed E0/E1 controls

| Control | Value |
| --- | --- |
| Initialization | Public hash-pinned YOLO26s COCO backbone modules 0–10; seed-initialized neck/heads |
| Dataset | Immutable Clean_v2 Train 6,299 / Val 1,827 images |
| Seeds | 17, 42, 123, completed irrespective of score direction |
| Image size / physical batch / nominal batch | 960 / 1 / 16 |
| Optimizer | Explicit MuSGD, lr0 0.01, momentum 0.937, weight decay 0.0005 |
| Schedule | 100 epochs, cosine decay, lrf 0.01, warmup disabled |
| Stopping / resume | No early stopping, no resume, no automatic OOM rebuild or NaN checkpoint recovery |
| Precision | Explicit FP32, AMP disabled, TF32 disabled |
| Augmentation | Explicit common settings; no mosaic/mixup/cutmix; pinned HSV/scale/translation/horizontal flip |
| Inference | Explicit `nms=True`: one-to-many prediction and standard NMS in the installed version |
| Final controlled checkpoint | Unstripped `last.pt` after completed epoch 100 in every arm/seed |
| Evaluator | Common project one-to-one evaluator; AP floor .001; operating confidence .25; matching IoU .50; NMS IoU .70; maxDet300 |

Native per-epoch validation is monitoring. It does not change the prespecified epoch100 checkpoint rule. The common exported evaluator reports final Validation measurements and the project-defined original-pixel size bins. A separately selected future external checkpoint must follow its own frozen protocol; Test scores never select it.

The installed MuSGD implementation derives some parameter-group rates from the configured `lr0`; the startup record reports both `lr0=0.01` and actual initial group rates (0.03/0.01). This is the pinned optimizer's behavior, not `optimizer=auto` selecting a different optimizer. These group rates are retained in `resolved_training.json` for both arms.

FP32, disabled warmup/mosaic, the cosine schedule and checkpoint rule are explicitly resolved before this run. They are not inferred from the exploratory seed0 results or imposed on earlier runs retrospectively. E2/E3 remain blocked until verified, matched Train-only negative bundles and their executable integration exist.

## Reproducibility and safeguards

`configs/paper/protocol_yolo26_v2.json` and the E0/E1 v2 YAMLs contain the exact shared settings. `configs/paper/YOLO26_V2_EXECUTION_LOCK.json` binds them, public input identities, installed implementation identities and the project runner. An existing lock cannot be overwritten. Any later implementation change requires a preserved, prospective new version.

The runner uses a runtime dataset YAML containing Train and Val only. Full before/after release verification includes hashing frozen Test payloads for structural integrity; no Test image is passed to the detector or used for selection. Frozen label-cache writes are suppressed. Dataset failures are persisted as failures rather than reported with an assumed `modified:false` flag. Actual optimizer updates and resolved runtime settings are recorded.

Outputs use `04_experiments/controlled_yolo26_v2/`, separate from every existing run. The launcher starts one explicitly selected arm/seed and does not automatically start E1 or another seed. Future execution remains governed by the approved sequence, not by whether Seed 17 improves a score.

## Validation performed before first launch

- Regression checks for opt-in execution, arm/seed restrictions, common E0/E1 settings, and truthful failure/integrity recording.
- Existing frozen-cache and evaluator checks, without dataset training.
- Synthetic GPU forward/backward, MuSGD update and EMA check for both architectures using the same public backbone. This uses no real dataset images and is not a paper result. Evidence: `06_results/paper_yolo26_v2/synthetic_preflight.json`.
- Full frozen snapshot verification and a dry-run before first training.

To prepare or launch a later approved arm, inspect the exact controls first:

```powershell
$env:PYTHONPATH = "$PWD/src"
.venv\Scripts\python.exe -m drone_detection.yolo26_controlled --arm E0 --seed 17
```

The explicit launcher is `scripts/start_controlled_yolo26.ps1 -Arm E0 -Seed 17`. Its logs and active process ID are saved under `04_experiments/controlled_yolo26_v2/logs/`.
