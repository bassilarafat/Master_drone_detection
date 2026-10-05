"""CPU model integration checks, with synthetic tensors and no downloads."""

from __future__ import annotations

import os
import unittest

from drone_detection.paths import PROJECT_ROOT


class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        settings = PROJECT_ROOT / ".cache" / "ultralytics"
        settings.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(settings))
        import torch

        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls) -> None:
        import torch

        torch.set_num_threads(cls.previous_threads)

    def test_p2_has_four_expected_detection_strides(self) -> None:
        from drone_detection.models import load_model

        model = load_model(PROJECT_ROOT / "configs/models/yolo26s_p2.yaml")
        self.assertEqual(model.model.stride.tolist(), [4, 8, 16, 32])

    def test_gam_receives_detection_gradient_only_in_corrected_graph(self) -> None:
        import torch

        from drone_detection.models import load_model
        from drone_detection.models.attention import GAM

        for relative_path, connected in (("yolo26s_gam.yaml", True), ("legacy/yolo26s_gam.yaml", False)):
            with self.subTest(model=relative_path):
                network = load_model(PROJECT_ROOT / "configs/models" / relative_path).model
                attention = next(layer for layer in network.modules() if isinstance(layer, GAM))
                network.train()
                output = network(torch.randn(2, 3, 64, 64))
                branch = output["one2many"]
                (branch["scores"].sum() + branch["boxes"].sum()).backward()
                gradients = [parameter.grad for parameter in attention.parameters()]
                if connected:
                    self.assertTrue(all(gradient is not None for gradient in gradients))
                    self.assertTrue(any(gradient.abs().sum().item() > 0 for gradient in gradients))
                else:
                    self.assertTrue(all(gradient is None for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
