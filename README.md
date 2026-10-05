# Master Drone Detection

A New Computer Vision and Deep Learning Approach for Real-Time Drone Detection.

Research code for training and evaluating drone detectors, with an emphasis on
small UAVs and real-time performance. Shared implementation lives in the
`drone_detection` package; model definitions and experiment settings live in
`configs/`.

## Layout

```text
src/drone_detection/     Reusable training, evaluation, metrics and custom models
configs/experiments/     Reproducible experiment inputs
configs/models/         Model topology, historical definitions and labelled drafts
tests/                  Workflow regression and CPU model checks
docs/architecture.md    Boundaries, migration notes and experiment workflow
03_code/                Compatible entry points for existing commands
notebooks/              Exploration and historical analyses
01_literature/          Research planning and literature notes
02_datasets/            Local datasets and their data.yaml files
04_experiments/         Training runs, checkpoints and existing experiment evidence
05_models/              Local pretrained weights
06_results/             Evaluation CSVs and analysis outputs
07_figures/             Thesis figures
08_thesis/              Thesis files
09_papers/              Publications
```

## Setup

Use Python 3.11 or later and activate the project environment. Install a PyTorch
build appropriate for your machine before installing research dependencies.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m drone_detection environment
```

If the existing environment already has all research dependencies, use
`python -m pip install --no-deps -e .`. Editable installation also makes package
imports available to notebooks using that environment. Without installation,
set `$env:PYTHONPATH = "$PWD/src"` from the project root for the commands below.

Datasets and weights are local assets and are not downloaded by these workflows.
After relocating the dataset, update `path:` inside its existing `data.yaml`.
Ultralytics is pinned to the version used for this model integration (8.4.157).

## Run experiments

Inspect resolved paths and settings before starting training:

```powershell
python -m drone_detection train --config configs/experiments/exp002_high_resolution.yaml --dry-run
python -m drone_detection train --config configs/experiments/yolo26s_p2.yaml --dry-run
```

Start a run explicitly and export predictions from a chosen checkpoint:

```powershell
python -m drone_detection train --config configs/experiments/exp002_high_resolution.yaml
python -m drone_detection evaluate --model 04_experiments/EXP002_YOLOv8s_high_resolution/YOLOv8s_960/weights/best.pt --output-dir 06_results/EXP002_new_run
```

Use the actual checkpoint path printed by training: repeated run names get an
incremented directory. Relative CLI and YAML paths resolve from the repository
root. `drone-detection` is an equivalent installed command.

The evaluation command exports prediction boxes and their best IoU. Object-level
matching, false-negative analysis and mAP are separate evaluation protocols.
The original commands remain available:

```powershell
python 03_code/environment_check.py
python 03_code/train_exp002_high_resolution.py --help
python 03_code/evaluate_exp002.py --help
```

## Custom models

```python
from drone_detection.models import load_model

model = load_model("configs/models/yolo26s_p2.yaml")
```

Use the shared loader for custom models and historical GAM checkpoints. It
registers project-owned layers in memory without editing installed library files.

The original GAM configuration bypasses the attention output. It is preserved
under `configs/models/legacy/`; the corrected definition has a separate
`YOLO26s_GAM_connected` experiment and has not been trained. Mamba and HEPAN are
incomplete drafts. See [model definitions](configs/models/README.md) before
selecting a model.

## Baseline vs P2 heatmaps

Compare trained checkpoints on the same image with a shared color scale:

```powershell
python -m drone_detection heatmap --source 02_datasets/DUT_Anti_UAV/images/val/00180.jpg --device cpu
```

This exports PNG/PDF comparisons, matched object close-ups, and per-scale UAV
score maps under `06_results/heatmaps/`. See [heatmap usage and interpretation](docs/heatmaps.md).

The [four-image reference report](06_results/heatmaps/baseline_vs_p2/README.md)
includes comparisons for tiny, small, medium and large UAVs, with PNG/PDF figures.

## Checks and documentation

```powershell
python -m unittest discover -s tests -v
```

Tests use temporary fixtures and synthetic CPU inputs. They do not train on the
dataset. Existing data, notebooks, checkpoints and research results are retained.

[Architecture and migration notes](docs/architecture.md) explain module boundaries,
path handling, experiment configuration and historical compatibility. Existing
[experiment](EXPERIMENT_REGISTRY.md) and [notebook](NOTEBOOK_REGISTRY.md) registries
remain dated research snapshots.
