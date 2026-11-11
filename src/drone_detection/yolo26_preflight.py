"""YOLO26 architecture and memory preflight without dataset training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .data_audit import digest
from .models import transfer_backbone_weights
from .paper_common import PROJECT_ROOT, project_path


PROTOCOL_PATH = PROJECT_ROOT / "configs/paper/protocol_yolo26.json"


def _tensor_hash(model: torch.nn.Module, keys: list[str]) -> str:
    hasher = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(keys):
        hasher.update(key.encode("utf-8"))
        hasher.update(state[key].detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def _load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["status"] != "preflight_only":
        raise ValueError("YOLO26 protocol is not in preflight-only status")
    release_record = project_path(protocol["dataset"]["release_record"])
    if digest(release_record) != protocol["dataset"]["release_record_sha256"]:
        raise ValueError("Pinned Clean_v2 release record changed")
    for arm in ("E0", "E1"):
        model = protocol["models"][arm]
        model_path = project_path(model["path"])
        if digest(model_path) != model["sha256"]:
            raise ValueError(f"Pinned {arm} model YAML changed")
    weights = project_path(protocol["initialization"]["path"])
    if digest(weights) != protocol["initialization"]["sha256"]:
        raise ValueError("Pinned YOLO26s checkpoint changed")
    return protocol


def _build_optimizer(model: torch.nn.Module, protocol: dict[str, Any]):
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    controls = protocol["controls"]
    model.args = get_cfg(
        overrides={
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "optimizer": controls["optimizer"],
            "lr0": controls["lr0"],
            "momentum": controls["momentum"],
            "weight_decay": controls["weight_decay"],
        }
    )
    trainer = DetectionTrainer.__new__(DetectionTrainer)
    trainer.args = model.args
    return trainer.build_optimizer(
        model,
        name=controls["optimizer"],
        lr=controls["lr0"],
        momentum=controls["momentum"],
        decay=controls["weight_decay"],
        iterations=100_000,
    )


def _run_update(model: torch.nn.Module, device: torch.device, protocol: dict[str, Any]) -> dict[str, Any]:
    size = protocol["controls"]["imgsz"]
    model.train()
    batch = {
        "img": torch.randn(1, 3, size, size, device=device),
        "cls": torch.tensor([[0.0]], device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=device),
        "batch_idx": torch.tensor([0], dtype=torch.long, device=device),
    }
    optimizer = _build_optimizer(
        model, protocol
    )
    optimizer.zero_grad(set_to_none=True)
    loss_components, _ = model.loss(batch, model(batch["img"]))
    loss = loss_components.sum()
    loss.backward()
    optimizer.step()

    from ultralytics.utils.torch_utils import ModelEMA

    ema = ModelEMA(model)
    ema.update(model)
    return {
        "loss": float(loss.detach().cpu()),
        "optimizer": type(optimizer).__name__,
        "ema_updated": True,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def _run_arm(arm: str, protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import init_seeds

    model_spec = protocol["models"][arm]
    weights_path = project_path(protocol["initialization"]["path"])
    init_seeds(protocol["controls"]["default_seed"], deterministic=True)
    target = DetectionModel(str(project_path(model_spec["path"])), ch=3, nc=1, verbose=False)
    donor = YOLO(str(weights_path))
    if donor.model.model[-1].nc != protocol["initialization"]["classes"]:
        raise ValueError("YOLO26s checkpoint does not contain the pinned COCO class count")
    if donor.names.get(0) != protocol["initialization"]["class_zero"]:
        raise ValueError("YOLO26s checkpoint class 0 is not the pinned COCO class")
    keys = transfer_backbone_weights(
        target,
        donor.model.float(),
        last_module_exclusive=protocol["initialization"]["backbone_last_module_exclusive"],
    )
    backbone_hash = _tensor_hash(target, keys)
    target = target.to(device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    update = _run_update(target, device, protocol)
    memory = {}
    if torch.cuda.is_available():
        memory = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    return {
        "model": arm,
        "strides": [int(stride) for stride in target.stride.tolist()],
        "expected_strides": model_spec["outputs"],
        "strides_match": [int(stride) for stride in target.stride.tolist()] == model_spec["outputs"],
        "parameters": sum(parameter.numel() for parameter in target.parameters()),
        "transferred_module_boundary": "0..10 inclusive",
        "transferred_tensors": len(keys),
        "backbone_tensor_sha256": backbone_hash,
        "update": update,
        "device": str(device),
        "cuda_memory": memory,
    }


def run(output: str | Path) -> dict[str, Any]:
    protocol = _load_protocol()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reports = [_run_arm(arm, protocol, device) for arm in ("E0", "E1")]
    if not all(report["strides_match"] for report in reports):
        raise ValueError("YOLO26 detection strides do not match the pinned protocol")
    if reports[0]["backbone_tensor_sha256"] != reports[1]["backbone_tensor_sha256"]:
        raise ValueError("E0 and E1 do not share identical transferred backbone tensors")
    result = {
        "status": "PASS",
        "protocol": str(PROTOCOL_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "models": reports,
        "identical_pretrained_backbones": True,
        "training_called": False,
        "real_dataset_inference": False,
        "test_evaluated": False,
        "note": "Synthetic one-batch forward/backward, MuSGD update, and EMA update only.",
    }
    output_path = project_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
