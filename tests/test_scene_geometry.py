import csv
import hashlib
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from drone_detection.scene_geometry import (
    extract_features, verify_features, build_feature_cache, verify_candidate_pairs, verify_candidate_graph,
)


def encoded(image):
    return cv2.imencode(".png", image)[1].tobytes()


def textured(seed=2):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (480, 640), np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), .6)
    for index in range(24):
        location = tuple(map(int, rng.integers([10, 20], [590, 440])))
        cv2.putText(image, f"building {index}", location, cv2.FONT_HERSHEY_SIMPLEX, .4, int(index * 10), 1)
    return image


class SceneGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)

    def test_projective_view_of_textured_scene_has_distributed_support(self):
        image = textured()
        transform = np.array([[.95, .02, 12], [-.01, .94, 14], [.00002, -.00003, 1]], np.float64)
        other = cv2.warpPerspective(image, transform, (640, 480))
        result = verify_features(extract_features(encoded(image)), extract_features(encoded(other)))
        self.assertEqual(result["geometry_decision"], "CONFIRMED_GEOMETRY")
        self.assertGreater(result["ransac_inliers"], 30)
        self.assertTrue(result["homography_plausible"])

    def test_independent_random_scenes_do_not_create_confirmed_edge(self):
        result = verify_features(extract_features(encoded(textured(2))), extract_features(encoded(textured(99))))
        self.assertNotEqual(result["geometry_decision"], "CONFIRMED_GEOMETRY")

    def test_featureless_sky_requires_review_even_with_identical_phash(self):
        first = np.full((480, 640), 140, np.uint8)
        second = np.full((480, 640), 145, np.uint8)
        result = verify_features(extract_features(encoded(first)), extract_features(encoded(second)))
        self.assertEqual(result["geometry_decision"], "REVIEW")
        self.assertEqual(result["geometry_reason"], "insufficient_local_texture")
        self.assertEqual(result["phash_distance"], 0)

    def test_identical_foreground_object_cannot_confirm_background_scene(self):
        foreground = textured()[60:420, 40:600]
        first, second = np.full((480, 640), 130, np.uint8), np.full((480, 640), 170, np.uint8)
        first[60:420, 40:600] = foreground
        second[60:420, 40:600] = foreground
        unmasked = verify_features(extract_features(encoded(first)), extract_features(encoded(second)))
        boxes = [(.5, .5, 560/640, 360/480)]
        background = verify_features(extract_features(encoded(first), boxes=boxes),
                                     extract_features(encoded(second), boxes=boxes))
        self.assertEqual(unmasked["geometry_decision"], "CONFIRMED_GEOMETRY")
        self.assertNotEqual(background["geometry_decision"], "CONFIRMED_GEOMETRY")
        self.assertEqual(background["geometry_reason"], "insufficient_local_texture")

    def test_features_remain_bound_to_image_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            (dataset / "images/train").mkdir(parents=True)
            (dataset / "labels/train").mkdir(parents=True)
            path = dataset / "images/train/00001.png"
            (dataset / "labels/train/00001.txt").write_text("")
            raw = encoded(textured())
            path.write_bytes(raw)
            index = root / "index.csv"
            with index.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_id", "relative_path", "sha256"])
                writer.writeheader()
                writer.writerow(dict(image_id="train/00001.png", relative_path="images/train/00001.png",
                                     sha256=hashlib.sha256(raw).hexdigest()))
            cache = build_feature_cache(dataset, index, root / "cache", workers=1)
            self.assertTrue(cache.is_file())
            self.assertEqual(path.read_bytes(), raw)
            path.write_bytes(encoded(textured(33)))
            # A fresh cache cannot use content different from its pinned SHA.
            with self.assertRaisesRegex(ValueError, "changed before feature extraction"):
                build_feature_cache(dataset, index, root / "fresh_cache", workers=1)

    def test_pair_resume_is_bound_to_original_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            (dataset / "images/train").mkdir(parents=True)
            (dataset / "labels/train").mkdir(parents=True)
            rows = []
            for index in range(2):
                raw = encoded(textured(index + 2))
                name = f"0000{index}.png"
                (dataset / "images/train" / name).write_bytes(raw)
                (dataset / "labels/train" / Path(name).with_suffix(".txt")).write_text("")
                rows.append(dict(image_id="train/" + name, relative_path="images/train/" + name,
                                 sha256=hashlib.sha256(raw).hexdigest()))
            index_file = root / "index.csv"
            with index_file.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            feature_index = build_feature_cache(dataset, index_file, root / "features", workers=1)
            candidates = root / "candidates.csv"
            candidates.write_text("image_a,image_b,cosine_similarity\ntrain/00000.png,train/00001.png,.91\n")
            output = root / "pairs.csv"
            first = verify_candidate_pairs(candidates, feature_index, output, workers=1)
            second = verify_candidate_pairs(candidates, feature_index, output, workers=1)
            self.assertEqual(first["new_pairs"], 1)
            self.assertEqual(second["new_pairs"], 0)
            candidates.write_text(candidates.read_text().replace(".91", ".95"))
            with self.assertRaisesRegex(ValueError, "inputs/policy changed"):
                verify_candidate_pairs(candidates, feature_index, output, workers=1)

    def test_adaptive_graph_skips_only_confirmed_paths_and_resumes_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            (dataset / "images/train").mkdir(parents=True)
            (dataset / "labels/train").mkdir(parents=True)
            image = textured()
            rows = []
            for index in range(3):
                transform = np.array([[1, 0, index * 3], [0, 1, index * 2], [0, 0, 1]], np.float64)
                raw = encoded(cv2.warpPerspective(image, transform, (640, 480)))
                name = f"{index}.png"
                (dataset / "images/train" / name).write_bytes(raw)
                (dataset / "labels/train" / f"{index}.txt").write_text("")
                rows.append(dict(image_id="train/" + name, relative_path="images/train/" + name,
                                 sha256=hashlib.sha256(raw).hexdigest()))
            index_file = root / "index.csv"
            with index_file.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            with contextlib.redirect_stdout(io.StringIO()):
                feature_index = build_feature_cache(dataset, index_file, root / "features", workers=1)
            candidates = root / "candidates.csv"
            candidates.write_text("image_a,image_b,cosine_similarity\ntrain/0.png,train/1.png,.99\n"
                                  "train/1.png,train/2.png,.98\ntrain/0.png,train/2.png,.97\n")
            output = root / "graph.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                first = verify_candidate_graph(candidates, feature_index, output, workers=1, batch_size=1)
                second = verify_candidate_graph(candidates, feature_index, output, workers=1, batch_size=1)
            self.assertEqual(first["new_direct_geometry_evaluations"], 2)
            self.assertEqual(first["new_decision_counts"]["SAME_SCENE_TRANSITIVE"], 1)
            self.assertEqual(first["geometry_components"], 1)
            self.assertEqual(second["new_pairs"], 0)
            with Path(first["confirmed_edges_file"]).open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)


if __name__ == "__main__":
    unittest.main()
