"""Compatibility import for shared bounding-box metrics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drone_detection.metrics import calculate_iou

__all__ = ["calculate_iou"]
