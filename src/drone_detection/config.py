"""Experiment inputs, separate from model topology and generated run outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import PROJECT_ROOT, project_path


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
        return {
            **self.train,
            "data": str(self.data),
            "project": str(self.project),
            "name": self.name,
            "pretrained": str(self.pretrained) if self.pretrained else True,
            "exist_ok": False,
        }

    def validate_files(self) -> None:
        """Fail before model initialization or a possible automatic download."""
        for label, path in (("model", self.model), ("dataset YAML", self.data),
                            ("pretrained weights", self.pretrained)):
            if path is not None and not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")

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
