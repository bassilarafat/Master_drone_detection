import unittest

import numpy as np

from drone_detection.scene_graph import Components, exact_neighbors


class SceneGraphTests(unittest.TestCase):
    def test_connected_paths_join_frames_without_direct_endpoint_edge(self):
        graph = Components(["a", "b", "c", "d"])
        graph.union("a", "b")
        graph.union("b", "c")
        self.assertEqual(graph.groups(), [["a", "b", "c"], ["d"]])

    def test_unknown_image_cannot_silently_enter_graph(self):
        graph = Components(["a"])
        with self.assertRaises(ValueError):
            graph.union("a", "outside")

    def test_global_neighbors_include_cross_split_and_exclude_self(self):
        matrix = np.array([[1, 0], [.999, .01], [0, 1]], dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        neighbors, _ = exact_neighbors(matrix, 1, device="cpu")
        self.assertEqual(neighbors[0, 0], 1)
        self.assertEqual(neighbors[1, 0], 0)
        self.assertTrue(all(i not in row for i, row in enumerate(neighbors)))


if __name__ == "__main__":
    unittest.main()
