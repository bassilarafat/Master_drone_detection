"""Experiment inputs, separate from model topology and generated run outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import PROJECT_ROOT, project_path


def snapshot_root_for_yaml(data_yaml: Path) -> Path | None:
    """Recognize a frozen dataset, including a removed freeze marker.

    The published YAML must be used directly: an alternate YAML could point
    train/val at different files while retaining the same dataset root.
    """
    import yaml

    data_yaml = Path(data_yaml).resolve()
    candidates = [data_yaml.parent]
    if data_yaml.is_file():
        values = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        if isinstance(values, dict) and isinstance(values.get("path"), str):
            root = Path(values["path"]).expanduser()
            candidates.append((root if root.is_absolute() else data_yaml.parent / root).resolve())
    for root in candidates:
        if (root / "FREEZE.json").is_file() or "_Clean_v" in root.name:
            if data_yaml != root / "data.yaml":
                raise ValueError("Use the frozen dataset's own data.yaml; alternate YAML files are not allowed.")
            if not (root / "FREEZE.json").is_file():
                raise ValueError(f"Frozen dataset is missing its FREEZE.json marker: {root}")
            return root
    return None


def _enforce_dataset_training_policy(dataset_root: Path) -> None:
    """Honor external project policy without changing a frozen release.

    A matching directory name outside this project's dataset directory does
    not inherit the policy. Policy applies only to the exact project dataset
    root, so temporary fixtures and independent checkouts remain isolated.
    """
    import json

    dataset_root = Path(dataset_root).resolve()
    project_root = Path(PROJECT_ROOT).resolve()
    if dataset_root.parent != (project_root / "02_datasets").resolve():
        return
    policy_path = project_root / "configs" / "datasets" / f"{dataset_root.name}.training_policy.json"
    if not policy_path.is_file():
        return
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read dataset training policy: {policy_path}: {exc}") from exc
    if (not isinstance(policy, dict) or policy.get("schema_version") != 1
            or policy.get("dataset_name") != dataset_root.name
            or type(policy.get("training_allowed")) is not bool):
        raise ValueError(f"Invalid dataset training policy; training is blocked: {policy_path}")
    relative = policy.get("dataset_relative_path")
    if relative is not None and (project_root / relative).resolve() != dataset_root:
        raise ValueError(f"Dataset training policy points at a different dataset: {policy_path}")
    if policy["training_allowed"] is False:
        reason = policy.get("reason", "Dataset training has been explicitly prohibited.")
        documentation = policy.get("documentation", "docs/scene_sequence_audit.md")
        raise ValueError(
            f"Training prohibited for {dataset_root.name} by {policy_path}: {reason} "
            f"See {documentation}."
        )


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: Path
    data: Path
    project: Path
    pretrained: Path | None = None
    train: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or self.name in {".", ".."} or any(
            char in self.name for char in '/\\:*?"<>|'
        ):
            raise ValueError("Experiment name must be a single directory name.")
        reserved = {"model", "data", "project", "name", "pretrained", "exist_ok", "resume"}
        collisions = reserved.intersection(self.train)
        if collisions:
            raise ValueError(f"Reserved training options: {', '.join(sorted(collisions))}")

    def training_kwargs(self) -> dict[str, Any]:
        kwargs = {
            **self.train,
            "data": str(self.data),
            "project": str(self.project),
            "name": self.name,
            "pretrained": str(self.pretrained) if self.pretrained else True,
            "exist_ok": False,
        }
        if snapshot_root_for_yaml(self.data) is not None:
            # Even cache=False leaves Ultralytics label-cache writes enabled;
            # training.py separately suppresses those writes for this snapshot.
            kwargs["cache"] = False
        return kwargs

    def validate_files(self) -> dict[str, Any] | None:
        """Fail before model initialization or a possible automatic download."""
        for label, path in (("model", self.model), ("dataset YAML", self.data),
                            ("pretrained weights", self.pretrained)):
            if path is not None and not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")
        root = snapshot_root_for_yaml(self.data)
        if root is not None:
            _enforce_dataset_training_policy(root)
            from .dataset_snapshot import verify_snapshot

            if self.train.get("cache") not in (None, False):
                raise ValueError("Image caching must be disabled for a frozen dataset (cache: false).")
            device = self.train.get("device")
            if (isinstance(device, (list, tuple)) and len(device) > 1) or (
                isinstance(device, str) and "," in device
            ):
                raise ValueError("Frozen dataset training requires a single device so cache writes remain guarded.")
            if self.project.resolve().is_relative_to(root):
                raise ValueError("Experiment outputs must be outside the frozen dataset.")
            if self.train.get("save_dir") is not None:
                save_dir = Path(self.train["save_dir"]).resolve()
                if save_dir.is_relative_to(root):
                    raise ValueError("Experiment outputs must be outside the frozen dataset.")
            return verify_snapshot(root)
        return None

    def as_dict(self) -> dict[str, Any]:
        return {"model": str(self.model), "train": self.training_kwargs()}


def load_experiment(path: str | Path, *, root: Path = PROJECT_ROOT) -> ExperimentConfig:
    import yaml

    config_path = project_path(path, root=root)
    with config_path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"Expected an experiment mapping in {config_path}")
    required = {"name", "model", "data"}
    missing = required - values.keys()
    unknown = values.keys() - (required | {"project", "pretrained", "train"})
    if missing or unknown:
        raise ValueError(f"Invalid experiment keys: missing={sorted(missing)}, unknown={sorted(unknown)}")
    for key in required | ({"project", "pretrained"} & values.keys()):
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f"{key} must be a non-empty string in {config_path}")
    train = values.get("train", {})
    if not isinstance(train, dict) or any(not isinstance(key, str) for key in train):
        raise ValueError("train must be a mapping of option names to values")
    return ExperimentConfig(
        name=values["name"],
        model=project_path(values["model"], root=root),
        data=project_path(values["data"], root=root),
        project=project_path(values.get("project", "04_experiments"), root=root),
        pretrained=project_path(values["pretrained"], root=root) if "pretrained" in values else None,
        train=dict(train),
    )
