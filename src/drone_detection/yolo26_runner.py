"""Controlled YOLO26 E0 training and validation on the frozen Clean_v2 release."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .data_audit import digest
from .dataset_snapshot import _annotations
from .paper_common import PROJECT_ROOT, project_path, read_csv, size_bin, write_json
from .paper_metrics import export_metrics
from .yolo26_preflight import _load_protocol


def _config() -> dict:
    config = project_path("configs/paper/E0_Y26.yaml")
    values = json.loads(json.dumps(__import__("yaml").safe_load(config.read_text(encoding="utf-8"))))
    if values["experiment_id"] != "E0_Y26" or values["seeds"] != [17, 42, 123]:
        raise ValueError("E0 configuration does not match the preregistered seed plan")
    protocol = _load_protocol()
    if protocol["status"] not in ("preflight_only", "E0_training_authorized"):
        raise ValueError("YOLO26 protocol is not authorized for E0 training")
    if digest(project_path(values["model"])) != protocol["models"]["E0"]["sha256"]:
        raise ValueError("E0 topology hash changed")
    if digest(project_path(values["initialization"])) != protocol["initialization"]["sha256"]:
        raise ValueError("YOLO26 checkpoint hash changed")
    return values


def _train(config: dict, output: Path, seed: int) -> Path:
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import init_seeds
    from .models import transfer_backbone_weights
    from .paper_runner import configure_runtime

    configure_runtime()
    model_yaml = project_path(config["model"])
    initialization = project_path(config["initialization"])

    class PaperTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            init_seeds(seed, deterministic=True)
            model = DetectionModel(str(model_yaml), ch=3, nc=1, verbose=verbose)
            donor = YOLO(str(initialization))
            if donor.model.model[-1].nc != 80 or donor.names.get(0) != "person":
                raise ValueError("YOLO26 initialization is not the pinned COCO80 checkpoint")
            keys = transfer_backbone_weights(model, donor.model.float(), last_module_exclusive=11)
            report = {
                "policy": "shared COCO backbone modules 0..10 only",
                "loaded_tensors": len(keys),
                "loaded_keys": keys,
                "pretrained_sha256": digest(initialization),
                "seed": seed,
                "pretrained_classes": 80,
                "target_classes": 1,
                "train_called": False,
            }
            write_json(Path(self.save_dir) / "initialization.json", report)
            return model

    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_yaml))
    model.train(
        trainer=PaperTrainer,
        data=str(project_path(config["dataset"])),
        imgsz=config["imgsz"],
        epochs=config["epochs"],
        batch=config["batch"],
        nbs=config["nbs"],
        optimizer=config["optimizer"],
        lr0=config["lr0"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        seed=seed,
        deterministic=True,
        device=0,
        workers=0,
        cache=False,
        amp=True,
        plots=False,
        val=True,
        project=str(output),
        name="run",
        exist_ok=False,
        verbose=True,
    )
    checkpoint = output / "run" / "weights" / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training completed without {checkpoint}")
    return checkpoint


def _evaluate(config: dict, checkpoint: Path, output: Path) -> dict:
    from ultralytics import YOLO

    dataset = project_path(config["dataset"]).parent
    rows = [row for row in read_csv(dataset / "dataset_manifest.csv") if row["new_split"] == "val"]
    model = YOLO(str(checkpoint))
    records = []
    for row in rows:
        image = dataset / row["new_relative_image"]
        width, height = int(row["width"]), int(row["height"])
        parsed, issues = _annotations(dataset / row["new_relative_label"])
        if issues:
            raise ValueError(issues)
        targets = [
            {
                "box": [(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height],
                "size": size_bin(w * width * h * height),
                "class": int(cls),
            }
            for cls, x, y, w, h in parsed
        ]
        result = model.predict(
            str(image), imgsz=config["imgsz"], conf=0.001, iou=0.7, max_det=300,
            batch=1, rect=False, augment=False, half=False, agnostic_nms=False,
            device=0, save=False, verbose=False,
        )[0]
        predictions = [
            {"box": list(map(float, box)), "confidence": float(score), "class": int(cls)}
            for box, score, cls in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            )
        ]
        records.append({
            "image": image.name,
            "split": "val",
            "width": width,
            "height": height,
            "targets": targets,
            "predictions": predictions,
            "image_sha256": row["image_sha256"],
            "label_sha256": row["label_sha256"],
            "scene_group_id": row["scene_group_id"],
            "negative_verified": False,
        })
    output.mkdir(parents=True, exist_ok=False)
    summary = export_metrics(output, records, confidence=0.25, matching_iou=0.5)
    write_json(output / "summary.json", summary)
    return summary


def train_e0(seed: int = 17) -> dict:
    if seed != 17:
        raise ValueError("This authorized run is restricted to E0 Seed 17")
    config = _config()
    output = project_path(config["output_root"]) / f"seed_{seed}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing paper run: {output}")
    started = time.time()
    checkpoint = _train(config, output, seed)
    summary = _evaluate(config, checkpoint, output / "evaluation" / "val")
    duration = time.time() - started
    record = {
        "status": "COMPLETED",
        "experiment_id": "E0_Y26",
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "validation_output": str(output / "evaluation" / "val"),
        "training_seconds": duration,
        "test_evaluated": False,
        "clean_v1_or_v2_modified": False,
        "summary": summary,
    }
    write_json(output / "summary.json", record)
    return record
