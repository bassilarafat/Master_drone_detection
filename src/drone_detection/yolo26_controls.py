"""Versioned YOLO26 controls; imports never train, load a detector, or touch Test images."""
from __future__ import annotations

from copy import deepcopy
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
from datetime import datetime, timezone

import yaml

from .data_audit import digest
from .paper_common import PROJECT_ROOT, project_path, pin_release, outside_dataset, bootstrap

PROTOCOL = "configs/paper/protocol_yolo26_v2.json"
LOCK = "configs/paper/YOLO26_V2_EXECUTION_LOCK.json"
CONFIGS = {"E0": "configs/paper/E0_Y26_v2.yaml", "E1": "configs/paper/E1_Y26_P2_v2.yaml"}
EXECUTION_SOURCES = (
    "src/drone_detection/yolo26_controls.py",
    "src/drone_detection/yolo26_controlled.py",
    "src/drone_detection/paper_runner.py",
    "src/drone_detection/paper_common.py",
    "src/drone_detection/paper_metrics.py",
    "src/drone_detection/training.py",
    "src/drone_detection/dataset_snapshot.py",
    "src/drone_detection/paths.py",
    "src/drone_detection/data_audit.py",
    "src/drone_detection/metrics.py",
    "src/drone_detection/models/__init__.py",
    "src/drone_detection/models/attention.py",
    "src/drone_detection/models/loader.py",
)


def _check_environment(protocol: dict) -> None:
    path = project_path(protocol["environment"]["path"])
    if digest(path) != protocol["environment"]["sha256"]:
        raise ValueError("Pinned environment lock changed")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if platform.python_version() != protocol["environment"]["python_version"]:
        raise ValueError("Pinned Python runtime version changed")
    for name, version in lock["packages"].items():
        if metadata.version(name) != version:
            raise ValueError(f"Pinned package version changed: {name}")
    package = Path(metadata.distribution("ultralytics").locate_file("ultralytics"))
    for name, expected in protocol["environment"]["installed_implementation_sha256"].items():
        if digest(package / Path(name)) != expected:
            raise ValueError(f"Pinned Ultralytics implementation changed: {name}")


def _check_lock(protocol_path: Path, config_path: Path) -> None:
    path = project_path(LOCK)
    if not path.is_file():
        raise ValueError("YOLO26 v2 executable controls are not sealed")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("protocol_path") != PROTOCOL:
        raise ValueError("Executable lock belongs to another protocol")
    entries = lock["artifacts"]
    for entry in entries:
        if digest(project_path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Sealed execution artifact changed: {entry['path']}")
    bound = {entry["path"] for entry in entries}
    for required in (PROTOCOL, *CONFIGS.values(), *EXECUTION_SOURCES):
        if required not in bound:
            raise ValueError(f"Execution lock does not bind {required}")


def load_controls(arm: str = "E0", *, require_lock: bool = True) -> tuple[dict, dict, Path]:
    """Return an identity-checked (config, protocol, config_path); no real-image access.

    ``require_lock=False`` is restricted to preparation/preflight. Production
    runner entry points must retain the default and perform full snapshot checks
    separately before and after dataset operations.
    """
    if arm not in CONFIGS:
        raise ValueError("Only E0/E1 are executable here; E2/E3 await verified negative bundles")
    protocol_path = project_path(PROTOCOL)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    config_path = project_path(CONFIGS[arm])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["protocol"] != PROTOCOL or config["protocol_sha256"] != digest(protocol_path):
        raise ValueError("Configuration does not bind the prospective v2 protocol")
    if config["architecture"] != arm or config["experiment_id"] != protocol["experiments"][arm]["experiment_id"]:
        raise ValueError("Experiment architecture/id changed")
    for key in ("dataset", "initialization", "seeds", "default_seed", "train", "evaluation", "checkpoint"):
        if config[key] != protocol[key]:
            raise ValueError(f"Common controlled parameter changed: {key}")
    if config["model"] != protocol["models"][arm]:
        raise ValueError("Architecture does not match its pinned topology")
    if config["output_root"] != f"04_experiments/controlled_yolo26_v2/{config['experiment_id']}":
        raise ValueError("Controlled output path changed")
    if config["negative_mode"] != "none" or config["negative_count"] != 0:
        raise ValueError("E0/E1 cannot include additional negatives")
    if config["seeds"] != [17, 42, 123] or config["train"]["optimizer"] != "MuSGD":
        raise ValueError("Fixed seed/explicit optimizer plan changed")
    if config["train"]["nms"] is not True or config["evaluation"]["nms"] is not True:
        raise ValueError("The standard-NMS inference path must be explicit")
    if config["checkpoint"]["filename"] != "last.pt" or config["checkpoint"]["epoch"] != 100:
        raise ValueError("The paper checkpoint is fixed prospectively at epoch100 last.pt")
    for item in (protocol["analysis_plan"], protocol["models"]["E0"], protocol["models"]["E1"],
                 protocol["initialization"]):
        if digest(project_path(item["path"])) != item["sha256"]:
            raise ValueError(f"Pinned input changed: {item['path']}")
    release = project_path(protocol["dataset"]["release_record"])
    if digest(release) != protocol["dataset"]["release_record_sha256"]:
        raise ValueError("Frozen release record changed")
    dataset = project_path(config["dataset"]["path"])
    pin_release(dataset, config["dataset"], full=False)
    outside_dataset(project_path(config["output_root"]), dataset)
    _check_environment(protocol)
    bootstrap()
    from ultralytics.cfg import get_cfg
    # Validate that every supplied option is recognized by the pinned library.
    get_cfg(overrides=dict(config["train"], seed=config["default_seed"]))
    if require_lock:
        _check_lock(protocol_path, config_path)
    return deepcopy(config), deepcopy(protocol), config_path


def create_lock(extra_paths: tuple[str, ...] | list[str] = ()) -> dict:
    """Seal a complete checked implementation once; refuse overwriting any old lock.

    The caller includes runner/launcher/test files only after implementation and
    required checks have passed. Creating this lock does not authorize all arms
    or start training; execution remains an explicit caller action.
    """
    lock_path = project_path(LOCK)
    if lock_path.exists():
        raise FileExistsError("Never overwrite an existing executable freeze; create a new version")
    for arm in CONFIGS:
        load_controls(arm, require_lock=False)
    paths = [PROTOCOL, *CONFIGS.values(), *EXECUTION_SOURCES, *extra_paths]
    records = []
    for relative in dict.fromkeys(paths):
        path = project_path(relative)
        records.append({"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256": digest(path)})
    record = {
        "schema_version": 1, "protocol_path": PROTOCOL,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Prospective controlled YOLO26 E0/E1 execution identities; no result or external-benchmark freeze",
        "artifacts": records,
        "test_locked": True,
        "change_policy": "Preserve this version and record a prospective new protocol/lock for any change; never silently reseal after outcomes."
    }
    with lock_path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    return record
