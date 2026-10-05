"""Resolve repository paths independently of a notebook's working directory."""

from __future__ import annotations

import os
from pathlib import Path

# Editable installation is the supported workflow. An explicit root also allows
# an installed wheel to operate on a separate research checkout.
PROJECT_ROOT = Path(
    os.environ.get("DRONE_DETECTION_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()


def project_path(value: str | Path, *, root: Path = PROJECT_ROOT) -> Path:
    """Interpret relative paths against the project root, never the current cwd."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()
