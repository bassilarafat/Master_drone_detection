"""Load local models without modifying the installed Ultralytics package."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from ..paths import project_path

if TYPE_CHECKING:
    from ultralytics import YOLO


def register_custom_modules() -> None:
    """Register GAM for YAML parsing and historical checkpoint deserialization.

    Call this before constructing or loading models. The registration is local
    to the current Python process and can be repeated. New checkpoints retain
    the stable ``drone_detection.models.attention.GAM`` import path; existing
    checkpoints can still resolve ``ultralytics.nn.modules.gam.GAM``.

    GAM keeps its input channel count, and Ultralytics passes its YAML arguments
    unchanged. The supplied GAM YAML therefore uses an explicit 512 channels
    for scale s; changing scale also requires reviewing that argument.
    """
    from . import attention

    # Install the alias first, including for environments whose old package
    # imports GAM during initialization. No site-packages files are changed.
    sys.modules["ultralytics.nn.modules.gam"] = attention

    from ultralytics.nn import modules, tasks

    modules.gam = attention
    modules.GAM = attention.GAM
    tasks.GAM = attention.GAM


def _local_file(path: str | Path, label: str) -> Path:
    resolved = project_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def load_model(
    model_path: str | Path, *, pretrained: str | Path | None = None
) -> YOLO:
    """Build or load a local YOLO model, optionally transferring local weights.

    Missing files fail before invoking Ultralytics so a typo cannot trigger a
    model download. Transferring weights is distinct from resuming a run, and
    layers with changed indices or shapes may not transfer.
    """
    model_file = _local_file(model_path, "Model")
    weights_file = _local_file(pretrained, "Pretrained weights") if pretrained is not None else None
    register_custom_modules()

    from ultralytics import YOLO

    model = YOLO(str(model_file))
    if weights_file is not None:
        model.load(str(weights_file))
    return model


def transfer_backbone_weights(target, source, *, last_module_exclusive: int) -> list[str]:
    """Transfer matching backbone tensors through a declared module boundary."""
    target_state = target.state_dict()
    source_state = source.state_dict()
    keys = [
        key
        for key in target_state
        if key.startswith("model.")
        and int(key.split(".", 2)[1]) < last_module_exclusive
    ]
    if any(key not in source_state or source_state[key].shape != target_state[key].shape for key in keys):
        raise ValueError("Declared shared backbone tensors differ in name or shape")
    target.load_state_dict({key: source_state[key] for key in keys}, strict=False)
    if not all(target.state_dict()[key].equal(source_state[key]) for key in keys):
        raise ValueError("Backbone transfer verification failed")
    return keys
