n# Controlled YOLO26 laptop v3 ? PASS and SEALED

Validated and sealed on 16 November 2026. **No official 100-epoch run started.** E0/E1 dry-run identity checks passed after sealing. The official v3 E0 Seed17 directory does not exist yet.

## Exact settings and design

Shared E0/E1 controls: 100 epochs, 960 pixels, physical batch4, nominal batch16, accumulation4, AMP enabled, workers2, MuSGD lr0=.01/momentum=.937/decay=.0005, cosine schedule/lrf=.01, no warmup or early stopping, and save_period1. All requested augmentation controls remain explicit. Seeds17/42/123 are fixed irrespective of performance. E0 is baseline, E1 is the whole declared P2 variant, E2 adds source-matched random backgrounds, and E3 adds source-matched hard negatives. E2/E3 remain blocked pending verified bundles/integration; no fusion intervention is substituted.

Relative to superseded v2: batch1?4, AMP false?true, workers0?2, and every completed epoch is retained. Physical batch and precision changes are prospective and shared by both arms. Older files/results are preserved and classified separately; official v3 starts at epoch1 from public backbone initialization.

## Preflight results

| Required check | E0 | E1 |
| --- | --- | --- |
| Construction and expected strides | PASS: 8/16/32 | PASS: 4/8/16/32 |
| Public backbone initialization, 240 tensors, source hash | PASS | PASS |
| Batch4 + AMP forward/backward and actual optimizer steps | PASS | PASS |
| Finite saved live model, EMA and optimizer state | PASS | PASS |
| Checkpoint writing/reloading | PASS | PASS |
| Deliberately corrupted canonical checkpoint fallback | PASS | PASS |
| Actual native resume, first epoch1 then epoch2 only | PASS | PASS |
| Optimizer/scaler/scheduler restoration | PASS | PASS |
| Same v3 Validation exporter invocation | PASS | PASS |
| Peak allocated GPU memory | 2.64 GiB | 3.9 GiB |

The successful execution-only fixture uses the first512 Train images and first4 Validation images sorted by current frozen manifest path. Their provenance is saved in each attempt. No scores selected settings; these short results are not official study or external-benchmark measurements. Test images never entered the detector. Full frozen-release integrity checks, including structural hashing of Test payloads, passed before and after the fixture.

All12 CPU regression checks passed, including protocol mismatch, finished-run refusal, corrupt-checkpoint fallback, atomic-write interruption, append-only events, Decimal-to-float coordinate export and the early Test guard. `--split test` aborts before configuration/model reads.

## Numerical and failure history

GradScaler needed initial calibration: E0 executed24 actual optimizer updates in the first test epoch and56 cumulatively after resume; E1 executed22 then54. Both executed32 additional actual updates in the resumed epoch. Final scaler values were256 and64. Saved live weights, EMA and optimizer tensors are finite. **This does not claim zero initial scaled-gradient overflows or identical actual update counts across arms.** Native scaler skips are recorded; nominal schedules are matched, and E2/E3 actual-budget parity remains a later verification requirement.

Earlier short fixtures lacked enough update attempts and failed the actual-update assertion. A later execution test found Decimal GT versus float prediction arithmetic in the exporter. V3 now uses an explicitly typed Validation module with a regression test; the old sealed v2 implementation was preserved. Failed reports, logs and checkpoints remain in their attempt folders. The previous approval-service usage-limit failure was resolved before this successful retry; it is not the current blocker.

## Crash-safe continuation

Each checkpoint embeds arm, seed, configuration SHA, protocol SHA, dataset release, architecture identity and public initialization identity. Native optimizer/scaler/EMA/epoch state, a full-precision live-model state and scheduler state are retained. Writes are staged and flushed; immutable epoch checkpoints are published before atomic canonical last replacement. Recovery scans epochs and canonical last, chooses the newest intact matching checkpoint, and repairs canonical last if needed. A process file lock prevents concurrent writers. Original start provenance is preserved with append-only event files.

A completed epoch means a durable checkpoint commit; an incomplete epoch may repeat. Native continuation is not claimed bitwise identical to uninterrupted training, and unfinished gradient accumulation/random progression can differ. If epoch100 is committed, resume refuses retraining. See HOW_TO_RESUME_CONTROLLED_TRAINING.md for final-evaluation recovery limitations and laptop sleep/shutdown behavior.

## Modified/new files

- configs/paper/yolo26_laptop_v3.review.yaml: settings, save_period1, initialization/environment/reproducibility provenance.
- configs/paper/YOLO26_LAPTOP_V3_LOCK.json: canonical sealed artifact with exact settings, hashes, environment, git/dirty state and timestamp.
- src/drone_detection/yolo26_recovery.py: checkpoints, recovery and events.
- src/drone_detection/yolo26_workflow.py: lifecycle, identity/seal guards, Test lock and dataset verification.
- src/drone_detection/yolo26_laptop.py: compatibility interface.
- src/drone_detection/yolo26_validation.py: common v3 evaluator and coordinate correction.
- tools/run_controlled.py and tools/preflight_controlled.py: public commands and execution checks.
- tests/test_controlled_recovery.py, test_yolo26_laptop.py and test_yolo26_validation.py: regression coverage.
- docs/HOW_TO_RESUME_CONTROLLED_TRAINING.md: shutdown/resume instructions.
- scripts/prepare_portable_training.py: archived draft generator disabled to prevent regenerating obsolete settings.

The candidate YAML retains its original drafting status label to preserve its validated bytes. **A matching SEALED lock is the authoritative execution status.** It is not permission to alter the YAML after validation.

## Final identities

Configuration SHA-256:
`183ce28af7b93aaa7c7d4b036050966ece520d83603e2ca3234dad7353bcda8e`

Base protocol SHA-256:
`0842b17cf6a95841f63fe2205ec2a510d1b32848cd4a5d3ef09c888faf382232`

Public donor SHA-256:
`646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`

Transferred initial-backbone SHA-256:
`b928e4ec27534c8a27f038005eb76ae934b7858af32a9f8f60a0192e988823a3`

Seal timestamp UTC: `2026-11-16T21:25:35.367474+00:00`. Exact Python, PyTorch, CUDA, Ultralytics, GPU/VRAM, OS, project git commit, dirty status and seal command are in the canonical lock. Fresh start events also capture the actual launch command/environment.

## Commands for a later explicit launch

```powershell
python tools/run_controlled.py --arm E0 --seed 17 --dry-run
python tools/run_controlled.py --arm E0 --seed 17
python tools/run_controlled.py --arm E0 --seed 17 --pause
python tools/run_controlled.py --arm E0 --seed 17 --resume
```

Only the dry-run was executed after sealing. **No official start command was run.** The package is an overlay for the existing repository and does not bundle datasets, public weights or Python dependencies.
