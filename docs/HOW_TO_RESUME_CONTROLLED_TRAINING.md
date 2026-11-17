# Controlled YOLO26 v3: start, stop and resume

Run commands from `D:\master\Master_Drone_Detection`, using the project environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

No official training is started by preparing these files. Commands below require a validated matching seal; execution fails while preflight is incomplete.

## Fresh start

```powershell
python tools/run_controlled.py --arm E0 --seed 17
```

This starts epoch 1 from the pinned public backbone, not from an older v2 checkpoint. A pre-existing run directory cannot be overwritten. Use the supplied hidden-process launcher if terminal closure is expected; a foreground command can be interrupted by closing its terminal.

## Normal interruption

Request a pause, then wait for the current epoch to finish and its checkpoint to be committed:

```powershell
python tools/run_controlled.py --arm E0 --seed 17 --pause
```

Then resume later:

```powershell
python tools/run_controlled.py --arm E0 --seed 17 --resume
```

The last complete epoch checkpoint contains the live model, EMA, optimizer, AMP scaler, scheduler, epoch number and experimental identity. Original start provenance is retained; interruption and resume events are added as new files under `events/`. A kernel file lock prevents two processes from operating on the same run. The OS releases that lock after a crash or shutdown.

## Sudden PC shutdown / power loss

After restarting, use the same `--resume` command. The launcher examines canonical `last.pt` and retained `epoch*.pt` checkpoints, rejects invalid files, and selects the highest valid completed epoch, preferring `last.pt` when tied. A different arm/seed/config/protocol/dataset/architecture/initialization is a hard error.

If epoch 36's checkpoint was committed and power failed during epoch 37, epoch 37 may repeat; epochs 1–36 are preserved. “Completed” means its durable checkpoint has been committed, not merely that GPU computation or a CSV row finished. No software can preserve work if power fails before it reaches persistent storage. Files are staged, flushed and atomically replaced; partial temporary files are never chosen for resume.

The pipeline does not claim bitwise equivalence with uninterrupted training. Native epoch-boundary resume discards unfinished gradient accumulation and may reset random data augmentation progression. These interruptions are recorded, and actual optimizer steps are saved rather than assumed identical across runs. The same sealed settings still apply; do not use interruption as a way to retune a model.

If completed epoch 100 is found, `--resume` reports FINISHED and refuses to retrain. Final Validation export can require a separate recovery step if power failed after epoch 100 but before evaluation finished; do not retrain completed epochs to regenerate metrics.

## Other arms and seeds

```powershell
python tools/run_controlled.py --arm E1 --seed 17 --resume
python tools/run_controlled.py --arm E0 --seed 42 --resume
python tools/run_controlled.py --arm E1 --seed 123 --resume
python tools/run_controlled.py --arm E2 --seed 42 --resume
python tools/run_controlled.py --arm E3 --seed 123 --resume
```

**E2/E3 intentionally remain blocked** until their matched, independently verified negative bundles and execution integration are available. Their scientific definitions remain E1 plus random versus hard background crops. These example commands show the intended interface; they do not claim those arms are executable now. E0/E1 support all fixed seeds 17, 42 and 123.

## Check before launch

```powershell
python tools/run_controlled.py --arm E0 --seed 17 --dry-run
python tools/run_controlled.py --arm E0 --seed 17 --resume --dry-run
```

`--split test` is rejected before any dataset/model operation. Full dataset integrity checks may hash Test payloads to verify the frozen manifest; they do not decode Test images for the detector, perform Test inference, or expose Test scores.

## Laptop and disk behavior

Training does not execute while Windows is shut down or sleeping. No lid/power policy or automatic boot-start service is changed. Resume after opening the laptop is explicit. Keep the project, dataset, public weights and matching software environment available.

`save_period: 1` retains each completed epoch checkpoint. Additional full-precision live-model state makes these larger than inference-only weights; leave sufficient free disk space for the study. Do not delete epoch checkpoints until the run is complete and backed up. Local persistence is not protection against disk failure; maintain a separate backup.

## Validation and sealing status

Current status: **PASS and SEALED**, with no official training launched. Both arms passed batch4/AMP and native checkpoint recovery on fixed Train/Val execution subsets. The common exporter and post-seal dry-runs passed. The initial failed attempts are preserved; see YOLO26_V3_VALIDATION_REPORT.md and the hardware/numerical audits. The matching execution lock is authoritative even though the unchanged candidate YAML retains its old drafting label.
