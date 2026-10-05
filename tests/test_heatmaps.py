"""Check heatmap alignment and classifier capture without trained checkpoints."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from drone_detection.heatmaps import DetectorHeatmaps, collect_images, read_ground_truth, restore_heatmap


class HeatmapGeometryTests(unittest.TestCase):
    def test_odd_letterbox_padding_is_removed_on_either_axis(self) -> None:
        # A 6x10 image becomes 19x32: six pixels before, seven after.
        # Bright padding must never leak into the restored content.
        for shape, content in (
            ((6, 10), (slice(6, 25), slice(None))),
            ((10, 6), (slice(None), slice(6, 25))),
        ):
            with self.subTest(shape=shape):
                grid = np.ones((32, 32), dtype=np.float32)
                grid[content] = 0.17
                restored = restore_heatmap(grid, shape, (32, 32))
                self.assertEqual(restored.shape, shape)
                np.testing.assert_allclose(restored, 0.17, atol=1e-6)

    def test_low_scores_keep_their_absolute_scale(self) -> None:
        restored = restore_heatmap(np.full((4, 4), 0.007), (13, 5), (32, 32))
        np.testing.assert_allclose(restored, 0.007, atol=1e-7)

    def test_nonpositive_dimensions_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            restore_heatmap(np.zeros((2, 2)), (0, 10), (32, 32))


class HeatmapInputTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="drone-heatmap-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_source_order_deduplication_and_limit(self) -> None:
        for name in ("c.jpg", "b.PNG", "a.jpg", "notes.txt"):
            (self.root / name).touch()
        (self.root / "directory.jpg").mkdir()
        # Explicit image order wins; directory contents are then sorted.
        result = collect_images([self.root / "c.jpg", self.root, self.root / "a.jpg"], 3)
        self.assertEqual(result, [self.root / name for name in ("c.jpg", "a.jpg", "b.PNG")])
        self.assertEqual(collect_images([self.root], 2), [self.root / "a.jpg", self.root / "b.PNG"])

    def test_missing_sources_are_not_silently_ignored_after_limit(self) -> None:
        present = self.root / "a.jpg"
        present.touch()
        with self.assertRaises(FileNotFoundError):
            collect_images([present, self.root / "missing.jpg"], 1)
        with self.assertRaisesRegex(ValueError, "No supported images"):
            collect_images([], 4)

    def test_ground_truth_is_optional(self) -> None:
        for path in (self.root / "photo.jpg", self.root / "images/val/photo.jpg"):
            with self.subTest(path=path):
                self.assertEqual(read_ground_truth(path, (80, 100), class_id=0), [])

    def test_ground_truth_inference_class_filter_and_coordinates(self) -> None:
        labels = self.root / "labels/val"
        labels.mkdir(parents=True)
        (labels / "photo.txt").write_text(
            "0 0.5 0.5 0.4 0.5\n\n1 0.5 0.5 0.8 0.8\n0 0.5 0.5 0 0.3\n",
            encoding="utf-8",
        )
        boxes = read_ground_truth(self.root / "images/val/photo.jpg", (80, 100), class_id=0)
        np.testing.assert_allclose(boxes, [[30, 20, 70, 60]])

    def test_explicit_labels_directory_supports_external_images(self) -> None:
        labels = self.root / "custom-labels"
        labels.mkdir()
        (labels / "photo.txt").write_text("0 0.5 0.5 1.2 1.2\n", encoding="utf-8")
        boxes = read_ground_truth(self.root / "photo.jpg", (80, 100), class_id=0, labels_dir=labels)
        self.assertEqual(boxes, [[0, 0, 100, 80]])


class ClassifierCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls) -> None:
        import torch

        torch.set_num_threads(cls.previous_threads)

    def make_detector(self, *, end2end: bool = False):
        import torch
        from ultralytics.nn.modules.head import Detect

        head = Detect(nc=2, reg_max=1, end2end=True, ch=(8, 16)).eval()
        head.stride = torch.tensor([4.0, 8.0])
        head.f = [1, 2]
        head.end2end = end2end
        model = Mock(
            task="detect", names={0: "other", 1: "UAV"},
            model=SimpleNamespace(model=[head]), callbacks={"on_predict_start": []},
        )
        model.add_callback.side_effect = lambda event, callback: model.callbacks[event].append(callback)
        with patch("drone_detection.models.load_model", return_value=model):
            detector = DetectorHeatmaps(self.fake_checkpoint(), class_id=1)
        self.addCleanup(detector.close)
        predictor = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(model=[head])))
        return detector, head, predictor

    @staticmethod
    def fake_checkpoint() -> Path:
        # Loading is mocked; tests never open or download checkpoint files.
        return Path("synthetic-heatmap-test.pt")

    def test_maps_match_selected_class_logits_of_both_inference_branches(self) -> None:
        import torch

        for end2end in (False, True):
            with self.subTest(end2end=end2end):
                detector, head, predictor = self.make_detector(end2end=end2end)
                detector._attach(predictor)
                features = [torch.randn(1, 8, 8, 8), torch.randn(1, 16, 4, 4)]
                with torch.inference_mode():
                    _, raw = head(features)
                branch = "one2one" if end2end else "one2many"
                selected = raw[branch]["scores"][0, 1].sigmoid().numpy()
                self.assertEqual(detector.branch, branch)
                np.testing.assert_allclose(detector.raw_maps["P2/4"], selected[:64].reshape(8, 8))
                np.testing.assert_allclose(detector.raw_maps["P3/8"], selected[64:].reshape(4, 4))
                detector._remove_hooks()
                self.assertTrue(all(not layer._forward_hooks for layer in head.cv3))
                self.assertTrue(all(not layer._forward_hooks for layer in head.one2one_cv3))

    def test_failed_inference_removes_hooks_and_reattaching_does_not_duplicate_them(self) -> None:
        detector, head, predictor = self.make_detector()

        def fail_prediction(**kwargs):
            detector._attach(predictor)
            detector._attach(predictor)
            self.assertEqual([len(layer._forward_hooks) for layer in head.cv3], [1, 1])
            raise RuntimeError("synthetic inference failure")

        detector.model.predict.side_effect = fail_prediction
        with self.assertRaisesRegex(RuntimeError, "synthetic inference failure"):
            detector.extract(np.zeros((32, 32, 3), dtype=np.uint8), image_size=32, device="cpu", confidence=0.25)
        self.assertEqual(detector.handles, [])
        self.assertTrue(all(not layer._forward_hooks for layer in head.cv3))


if __name__ == "__main__":
    unittest.main()
