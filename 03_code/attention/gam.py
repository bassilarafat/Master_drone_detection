"""Compatibility import for older notebooks; implementation lives in src/."""

import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from drone_detection.models.attention import GAM

__all__ = ["GAM"]
