"""Compatibility entry point for the EXP002 evaluation command."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drone_detection.evaluation import (
    DATASET_DIR,
    DEFAULT_MODEL,
    PROJECT_ROOT,
    RESULTS_DIR,
    TEST_IMAGES,
    TEST_LABELS,
    calculate_iou,
    evaluate,
    load_ground_truth,
    main,
    parse_args,
)


if __name__ == "__main__":
    main()
