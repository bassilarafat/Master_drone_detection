from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
model = YOLO(str(ROOT / "04_experiments" / "YOLO26s_baseline-5" / "weights" / "best.pt"))
results = model.val(
    data=str(ROOT / "02_datasets" / "DUT_Anti_UAV" / "data.yaml"),
    split="test",
    imgsz=960,
    batch=4,
    conf=0.001,
    iou=0.7,
    max_det=300,
    rect=False,
    device="0",
    workers=0,
    plots=True,
    project=str(ROOT / "YOLO26s_AUDIT" / "Reproduced_Evaluation"),
    name="ultralytics_test",
    exist_ok=False,
    verbose=True,
)
print("RESULTS_DIR", results.save_dir)
print("PRECISION", results.box.mp)
print("RECALL", results.box.mr)
print("MAP50", results.box.map50)
print("MAP50_95", results.box.map)
