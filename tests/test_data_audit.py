from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from drone_detection.data_audit import normalize_dataset_labels


class DatasetNormalizationTests(unittest.TestCase):
    def test_clips_pixel_corners_and_preserves_yolo_xywh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "images" / "train"
            label_dir = root / "labels" / "train"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            Image.new("RGB", (100, 50)).save(image_dir / "sample.jpg")
            (label_dir / "sample.txt").write_text("0 0.95 0.5 0.2 0.4\n", encoding="utf-8")

            output = root / "report"
            report = normalize_dataset_labels(root, output, write=True)

            self.assertEqual(report["changed_boxes"], 1)
            self.assertEqual(
                (label_dir / "sample.txt").read_text(encoding="utf-8"),
                "0 0.925000000000 0.500000000000 0.150000000000 0.400000000000\n",
            )
            self.assertTrue((output / "backups" / "train" / "sample.txt").is_file())
            self.assertEqual(json.loads((output / "report.json").read_text())["errors"], 0)


if __name__ == "__main__":
    unittest.main()
