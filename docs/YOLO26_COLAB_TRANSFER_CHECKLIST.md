# YOLO26 Colab transfer checklist

GitHub is the canonical source. Colab clones:

`https://github.com/bassilarafat/Master_drone_detection.git`

and checks out commit
`504fa9dae6f1e4fc3f156a6ea93a6d9121b53515`.

Google Drive is **not mounted on the local machine**, so upload these items
manually under `MyDrive/Master_Drone_Detection_Assets/`:

| Drive destination | Local source | Approximate size |
|---|---|---:|
| `datasets/DUT_Anti_UAV_Clean_v2/` | `D:\master\Master_Drone_Detection\02_datasets\DUT_Anti_UAV_Clean_v2\` | 2.62 GB |
| `models/yolo26s.pt` | `D:\master\Master_Drone_Detection\05_models\pretrained\paper\yolo26s.pt` | 19.5 MB |
| `experiments/paper_yolo26/` | `D:\master\Master_Drone_Detection\04_experiments\paper_yolo26\` | currently empty/incomplete; optional before E0 |

The dataset is frozen; upload it without renaming or editing files. The model
must remain named `yolo26s.pt`. The notebook verifies the Clean_v2 release
record hash and model SHA-256 before any dry-run or preflight.

Do **not** upload `.venv`, `.git`, Python caches, source code, or old generated
results. The Colab notebook is in GitHub at
`notebooks/YOLO26_E0_Seed17_Colab.ipynb`.
