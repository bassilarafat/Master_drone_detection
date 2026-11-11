import unittest

import cv2
import numpy as np

from drone_detection.scene_gpu_geometry import batch_mutual_matches


class ExactDescriptorTests(unittest.TestCase):
    def test_nearly_identical_descriptors_and_ties_keep_direct_ratio_result(self):
        rng=np.random.default_rng(7)
        a=np.abs(rng.normal(size=(12,128))).astype(np.float32)
        a/=np.linalg.norm(a,axis=1,keepdims=True)
        b=np.repeat(a,10,axis=0)
        b+=rng.normal(0,1e-6,b.shape).astype(np.float32)
        matcher=cv2.BFMatcher(cv2.NORM_L2)
        forward={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(a,b,k=2) if m.distance<.75*n.distance}
        backward={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(b,a,k=2) if m.distance<.75*n.distance}
        expected={(i,j) for i,j in forward.items() if backward.get(j)==i}
        self.assertEqual(set(batch_mutual_matches([(a,b)],device='cpu')[0]),expected)

    def test_batched_direct_euclidean_matches_opencv_mutual_ratio(self):
        rng=np.random.default_rng(42)
        a=rng.random((50,128),dtype=np.float32)
        b=np.concatenate([a[:30]+rng.normal(0,.001,(30,128)).astype(np.float32),rng.random((20,128),dtype=np.float32)])
        matcher=cv2.BFMatcher(cv2.NORM_L2)
        forward={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(a,b,k=2) if m.distance<.75*n.distance}
        backward={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(b,a,k=2) if m.distance<.75*n.distance}
        expected={(i,j) for i,j in forward.items() if backward.get(j)==i}
        actual=batch_mutual_matches([(a,b),(np.empty((0,128),np.float32),b)],device='cpu')
        self.assertEqual(set(actual[0]),expected)
        self.assertEqual(actual[1],[])


if __name__=='__main__':unittest.main()
