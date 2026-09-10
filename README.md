# Master Drone Detection



\## Title

A New Computer Vision and Deep Learning Approach for Real-Time Drone Detection



\## Objective

Develop and evaluate a deep learning based drone detection system with focus on accuracy and real-time performance.



\## Environment

\- Python 3.11

\- PyTorch

\- NVIDIA RTX 5050 Laptop GPU



## Project Structure



01_literature  - Research papers
02_datasets    - Dataset files
03_code        - Reproducible training and evaluation scripts
04_experiments - Experiment configurations and logs
05_models      - Saved weights
06_results     - Generated metrics and prediction tables
07_figures     - Thesis figures
08_thesis      - Thesis files
09_papers      - Publications
notebooks      - Exploration, visualization, and interactive analysis

## Script Workflow

Run commands from the repository root with the project virtual environment active.

```powershell
python 03_code/environment_check.py
python 03_code/train_exp002_high_resolution.py
python 03_code/evaluate_exp002.py
```

Training writes the EXP002 run under `04_experiments`. Evaluation writes prediction boxes and IoU values to `06_results/EXP002_high_resolution/`.

Notebooks are retained for exploratory work and figures. They should load saved models or CSV results rather than contain the only copy of a repeatable experiment.

