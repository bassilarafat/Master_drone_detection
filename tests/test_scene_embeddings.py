import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from drone_detection.scene_embeddings import fit_layout, preprocess, valid_patch_mask


class FullImageEmbeddingInputTests(unittest.TestCase):
    def test_letterbox_preserves_whole_wide_scene(self):
        self.assertEqual(fit_layout(1280, 720), (0, 113, 518, 291))
        mask = valid_patch_mask(1280, 720).reshape(37, 37)
        self.assertFalse(mask[:8].any())
        self.assertTrue(mask[8:29].all())
        self.assertFalse(mask[29:].any())

    def test_content_edges_survive_without_center_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (100, 50), "red")
            for y in range(50):
                image.putpixel((0, y), (0, 255, 0))
                image.putpixel((99, y), (0, 0, 255))
            path = root / "scene.png"
            image.save(path)
            row = dict(relative_path="scene.png", width=100, height=50,
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            array = preprocess(row, root, 140)
            self.assertEqual(array.shape, (3, 140, 140))
            self.assertGreater(array[1, 70, 0], array[0, 70, 0])
            self.assertGreater(array[2, 70, -1], array[0, 70, -1])

    def test_changed_image_bytes_are_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scene.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Frozen image changed"):
                preprocess(dict(relative_path="scene.png", width=100, height=50, sha256="old"), root)

    def test_invalid_grid_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_layout(100, 50, 512)
        self.assertAlmostEqual(float(np.linalg.norm(valid_patch_mask(50, 50))), 37)


if __name__ == "__main__":
    unittest.main()
