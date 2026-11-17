# Local YOLO26s performance tuning

Measured on 2026-11-15 with RTX 5050 Laptop 8 GB, 16 GB RAM,
PyTorch 2.11.0+cu128, Ultralytics 8.4.157, Clean_v2, image size 960,
AMP enabled and image caching disabled.

| Batch | Workers | Images/second | Peak CUDA reserved GiB |
| ---: | ---: | ---: | ---: |
| 4 | 2 | 24.35 | 2.92 |
| 8 | 2 | 27.47 | 5.66 |
| 8 | 4 | 27.14 | 5.66 |
| 12 | 4 | Aborted: memory pressure and severe slowdown | About 8.3 |

These are short actual training benchmarks: 20 warmup batches, followed by
80 timed batches, with CUDA synchronization at timing boundaries. They exclude
startup, validation and checkpoint saving. Two versus four workers is effectively
a tie in these measurements; two uses fewer processes. Batch 12 pushed beyond
dedicated memory and ran at several seconds per batch, so it was stopped early.
Reserved CUDA memory differs from total device memory shown by NVIDIA tools.

Selected batch 8, workers 2. This is the fastest measured configuration among
these candidates, not an exhaustive hardware maximum. Dataset integrity is
checked before and after benchmarking; candidate weights are discarded.

The original run was interrupted during epoch 4. Resume uses last.pt from the
end of epoch 3, retaining optimizer state and repeating the incomplete epoch.
Before resuming, last.pt is copied to before_performance_tuning.pt. The batch
change is recorded in performance_resume.json; this local exploratory run is
separate from the locked paper protocol.

Resume from PowerShell:

```powershell
./scripts/start_yolo26_local.ps1 -ResumeCheckpoint 04_experiments/local_clean_v2/YOLO26s_Clean_v2_local_seed17-2/weights/last.pt
```

Logs are under 04_experiments/local_clean_v2/logs. Avoid launching duplicate runs.
