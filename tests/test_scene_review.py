import unittest

from drone_detection.scene_review import risk_reason


class SceneReviewRiskTests(unittest.TestCase):
    def test_partial_geometry_is_an_uncertainty_constraint_not_scene_confirmation(self):
        row = dict(cosine_similarity=.8, ransac_inliers=12, inlier_ratio=.8, homography_plausible=True,
                   hull_coverage_a=.05, hull_coverage_b=.06, grid_cells_a=4, grid_cells_b=5)
        self.assertEqual(risk_reason(row), "partial_distributed_background_geometry")

    def test_blank_sky_remains_an_uncertainty_constraint(self):
        row = dict(cosine_similarity=.92, keypoints_a=0, keypoints_b=2, phash_distance=4, background_ssim128=.9)
        self.assertEqual(risk_reason(row), "strong_global_retrieval_identity_unresolved")

    def test_high_single_cosine_without_corroboration_requires_colocation(self):
        self.assertEqual(risk_reason(dict(cosine_similarity=.95, phash_distance=40, background_ssim128=.1,
                                         keypoints_a=100, keypoints_b=100)), "strong_global_retrieval_identity_unresolved")

    def test_localized_overlay_is_uncertain_even_with_low_cosine_and_one_grid_cell(self):
        self.assertEqual(risk_reason(dict(cosine_similarity=.51, ransac_inliers=23,
            inlier_ratio=1, median_reprojection_px=.04, p95_reprojection_px=.22,
            grid_cells_a=1, grid_cells_b=1)), "coherent_local_context_or_overlay_requires_colocation")

    def test_changed_viewpoint_with_moderate_match_ratio_remains_uncertain(self):
        self.assertEqual(risk_reason(dict(cosine_similarity=.459, ransac_inliers=48,
            inlier_ratio=.696, median_reprojection_px=.2, p95_reprojection_px=1.2)),
            "coherent_local_context_or_overlay_requires_colocation")


if __name__ == "__main__":
    unittest.main()
