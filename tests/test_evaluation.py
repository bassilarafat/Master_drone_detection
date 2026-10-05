"""Exercise prediction exports using tiny local fixtures and a mocked model."""

from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from drone_detection import evaluation
from drone_detection.paths import PROJECT_ROOT


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        from PIL import Image

        temporary = tempfile.TemporaryDirectory(prefix="drone-evaluation-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model_path = self.root / "best.pt"
        self.model_path.touch()
        self.dataset_dir = self.root / "dataset"
        self.output_dir = self.root / "output"
        images = self.dataset_dir / "images" / "val"
        labels = self.dataset_dir / "labels" / "val"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        Image.new("RGB", (100, 80)).save(images / "sample.jpg")
        (labels / "sample.txt").write_text("0 0.5 0.5 0.4 0.5\n")
        self.model = Mock()
        loader = patch("drone_detection.models.load_model", return_value=self.model)
        self.load_model = loader.start()
        self.addCleanup(loader.stop)

    def evaluate(self, *, split: str = "val") -> Path:
        with contextlib.redirect_stdout(io.StringIO()):
            return evaluation.evaluate(
                self.model_path,
                960,
                "cpu",
                0.25,
                dataset_dir=self.dataset_dir,
                output_dir=self.output_dir,
                split=split,
            )

    def test_multiple_predictions_use_one_ground_truth_read(self) -> None:
        coordinates = Mock()
        coordinates.cpu.return_value.numpy.return_value = [30, 20, 70, 60]
        prediction = SimpleNamespace(xyxy=[coordinates], conf=[0.9], cls=[0])
        self.model.predict.return_value = [SimpleNamespace(boxes=[prediction, prediction])]

        with patch.object(
            evaluation, "load_ground_truth", wraps=evaluation.load_ground_truth,
        ) as read_ground_truth:
            output_path = self.evaluate()

        read_ground_truth.assert_called_once_with("sample", self.dataset_dir, "val")
        self.assertEqual(output_path.name, "exp002_val_predictions.csv")
        with output_path.open(newline="") as output_file:
            reader = csv.DictReader(output_file)
            rows = list(reader)
            self.assertEqual(reader.fieldnames, evaluation.PREDICTION_COLUMNS)
        self.assertEqual([int(row["pred_id"]) for row in rows], [0, 1])
        self.assertEqual([float(row["best_iou"]) for row in rows], [1.0, 1.0])
        self.assertEqual(
            [float(rows[0][axis]) for axis in ("x1", "y1", "x2", "y2")],
            [30, 20, 70, 60],
        )
        options = self.model.predict.call_args.kwargs
        self.assertEqual(options["imgsz"], 960)
        self.assertEqual(options["conf"], 0.25)
        self.assertEqual(options["iou"], 0.7)
        self.assertEqual(options["workers"], 0)

    def test_empty_predictions_keep_csv_columns(self) -> None:
        self.model.predict.return_value = [SimpleNamespace(boxes=[])]

        output_path = self.evaluate()

        with output_path.open(newline="") as output_file:
            reader = csv.DictReader(output_file)
            self.assertEqual(reader.fieldnames, evaluation.PREDICTION_COLUMNS)
            self.assertEqual(list(reader), [])

    def test_split_without_jpg_images_fails_before_model_load(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "No JPG images found"):
            self.evaluate(split="test")

        self.load_model.assert_not_called()
        self.assertFalse(self.output_dir.exists())

    def test_cli_relative_paths_use_project_root(self) -> None:
        with contextlib.chdir(self.root):
            args = evaluation.parse_args([
                "--model", "weights/best.pt",
                "--dataset-dir", "datasets/sample",
                "--output-dir", "results/sample",
            ])

        self.assertEqual(args.model, PROJECT_ROOT / "weights" / "best.pt")
        self.assertEqual(args.dataset_dir, PROJECT_ROOT / "datasets" / "sample")
        self.assertEqual(args.output_dir, PROJECT_ROOT / "results" / "sample")

    def test_original_cli_defaults_are_preserved(self) -> None:
        args = evaluation.parse_args([])

        self.assertEqual(args.model, evaluation.DEFAULT_MODEL)
        self.assertEqual(args.dataset_dir, evaluation.DATASET_DIR)
        self.assertEqual(args.output_dir, evaluation.RESULTS_DIR)
        self.assertEqual((args.imgsz, args.device, args.conf, args.split), (960, "0", 0.25, "test"))


if __name__ == "__main__":
    unittest.main()
