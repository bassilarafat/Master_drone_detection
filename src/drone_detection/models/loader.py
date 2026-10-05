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
