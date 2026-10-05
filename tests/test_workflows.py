"""Regression checks for experiment isolation and lightweight entry points."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from drone_detection.config import ExperimentConfig, load_experiment
from drone_detection.paths import PROJECT_ROOT
from drone_detection.training import run_experiment


class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="drone-workflows-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_paths_follow_project_root_when_cwd_changes(self) -> None:
        (self.root / "experiment.yaml").write_text(
            "name: trial\nmodel: weights/model.pt\ndata: dataset/data.yaml\n"
            "train:\n  epochs: 5\n", encoding="utf-8",
        )
        with contextlib.chdir(PROJECT_ROOT / "notebooks"):
            config = load_experiment("experiment.yaml", root=self.root)
        self.assertEqual(config.model, self.root / "weights" / "model.pt")
        self.assertEqual(config.data, self.root / "dataset" / "data.yaml")
        self.assertEqual(config.project, self.root / "04_experiments")

    def test_nested_options_cannot_override_output_safety(self) -> None:
        for option in ("name", "project", "exist_ok", "resume", "model", "data", "pretrained"):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "Reserved"):
                ExperimentConfig("trial", self.root, self.root, self.root, train={option: True})

    def test_missing_inputs_fail_before_model_loading(self) -> None:
        config = ExperimentConfig("trial", self.root / "missing.pt", self.root, self.root)
        with patch("drone_detection.models.load_model") as loader:
            with self.assertRaisesRegex(FileNotFoundError, "Missing model"):
                run_experiment(config)
        loader.assert_not_called()

    def test_training_records_actual_incremented_run_directory(self) -> None:
        model_path = self.root / "model.pt"
        dataset_path = self.root / "data.yaml"
        model_path.touch()
        dataset_path.touch()
        actual_output = self.root / "trial2"
        actual_output.mkdir()
        config = ExperimentConfig("trial", model_path, dataset_path, self.root, train={"epochs": 3})
        model = Mock(trainer=SimpleNamespace(save_dir=actual_output))
        with patch("drone_detection.models.load_model", return_value=model), contextlib.redirect_stdout(io.StringIO()):
            run_experiment(config)
        self.assertFalse(model.train.call_args.kwargs["exist_ok"])
        self.assertEqual(model.train.call_args.kwargs["epochs"], 3)
        manifest = json.loads((actual_output / "experiment.json").read_text())
        self.assertEqual(manifest["model"], str(model_path))
        self.assertFalse((self.root / "trial").exists())

    def test_entry_points_do_not_import_torch(self) -> None:
        code = (
            "import sys; import drone_detection.cli; import drone_detection.evaluation; "
            "import drone_detection.models; "
            "assert 'torch' not in sys.modules; assert 'ultralytics' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_model_loader_resolves_relative_paths_from_root(self) -> None:
        from drone_detection.models.loader import _local_file

        with contextlib.chdir(self.root):
            self.assertEqual(
                _local_file("configs/models/yolo26s_p2.yaml", "Model"),
                PROJECT_ROOT / "configs" / "models" / "yolo26s_p2.yaml",
            )


if __name__ == "__main__":
    unittest.main()
