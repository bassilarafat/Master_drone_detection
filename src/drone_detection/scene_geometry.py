"""Cached local features and conservative geometric scene-edge verification.

Geometry can confirm a shared visible scene; it never supplies a native video
ID or source frame number. Missing geometric support is not proof of different
scenes. Sky/low-texture images require review or conservative allocation by the
calling scene-audit workflow.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .data_audit import digest

FEATURE_VERSION = 2
GEOMETRY_VERSION = 2
GRAPH_VERSION = 1


@dataclass(frozen=True)
class GeometryPolicy:
    ratio: float = 0.75
    ransac_px: float = 3.0
    min_inliers: int = 24
    min_inlier_ratio: float = 0.55
    min_hull_coverage: float = 0.10
    min_grid_cells: int = 6
    max_median_error: float = 1.5
    max_p95_error: float = 3.5
    min_projected_area_ratio: float = 0.08
    max_projected_area_ratio: float = 12.0
    min_projected_overlap: float = 0.15


POLICY = GeometryPolicy()
METRIC_FIELDS = [
    "phash_distance", "ssim128", "background_ssim128", "background_overlap_fraction",
    "keypoints_a", "keypoints_b", "mutual_ratio_matches",
    "ransac_inliers", "inlier_ratio", "hull_coverage_a", "hull_coverage_b",
    "grid_cells_a", "grid_cells_b", "median_reprojection_px", "p95_reprojection_px",
    "homography_plausible", "projected_area_ratio", "projected_overlap",
    "geometry_decision", "geometry_reason",
]
_local = threading.local()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _read_index(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["image_id"] = row.get("image_id") or row.get("image")
        if not row["image_id"] or "sha256" not in row:
            raise ValueError("Image index requires image_id (old_split/name) and sha256")
        row.setdefault("relative_path", "images/" + row["image_id"])
        if Path(row["relative_path"]).is_absolute() or ".." in Path(row["relative_path"]).parts:
            raise ValueError("Unsafe image path in index")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Image index contains duplicate image IDs")
    return rows


def _sift(nfeatures):
    key = f"sift_{nfeatures}"
    if not hasattr(_local, key):
        setattr(_local, key, cv2.SIFT_create(nfeatures=nfeatures))
    return getattr(_local, key)


def extract_features(image_bytes: bytes, *, max_side=640, nfeatures=1024, boxes=(), margin_px=8) -> dict:
    """Read bytes once; local geometry uses aspect-preserving resized images."""
    full = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if full is None:
        raise ValueError("Image could not be decoded")
    height, width = full.shape
    scale = min(1.0, max_side / max(height, width))
    gray = cv2.resize(full, (max(1, round(width * scale)), max(1, round(height * scale))),
                      interpolation=cv2.INTER_AREA) if scale < 1 else full
    mask = np.full(gray.shape, 255, np.uint8)
    # Boxes are normalized YOLO xywh. An eight-pixel margin at feature scale
    # is at least eight original pixels because we never upscale images.
    for x, y, box_width, box_height in boxes:
        x1 = max(0, int(np.floor((x-box_width/2) * gray.shape[1])) - margin_px)
        y1 = max(0, int(np.floor((y-box_height/2) * gray.shape[0])) - margin_px)
        x2 = min(gray.shape[1], int(np.ceil((x+box_width/2) * gray.shape[1])) + margin_px)
        y2 = min(gray.shape[0], int(np.ceil((y+box_height/2) * gray.shape[0])) + margin_px)
        mask[y1:y2, x1:x2] = 0
    keypoints, descriptors = _sift(nfeatures).detectAndCompute(gray, mask)
    if descriptors is None:
        points = np.empty((0, 2), np.float32)
        descriptors = np.empty((0, 128), np.float32)
    else:
        order = sorted(range(len(keypoints)), key=lambda i: -keypoints[i].response)[:nfeatures]
        points = np.array([keypoints[i].pt for i in order], np.float32)
        descriptors = descriptors[order].astype(np.float32)
        descriptors /= np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-12)
        np.sqrt(descriptors, out=descriptors)  # RootSIFT; squared L2 norm is one.
    tiny = cv2.resize(full, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(tiny)[:8, :8].flatten()[1:]
    phash = sum(int(bit) << i for i, bit in enumerate(low > np.median(low)))
    return dict(points=points, descriptors=descriptors.astype(np.float16),
                shape=np.array(gray.shape, np.int32),
                gray128=cv2.resize(full, (128, 128), interpolation=cv2.INTER_AREA),
                background_mask128=cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST),
                phash=np.array(phash, np.uint64), source_shape=np.array(full.shape, np.int32))


def build_feature_cache(dataset, image_index, cache_dir, *, max_side=640, nfeatures=1024, workers=6):
    """Build/resume source-SHA-bound NPZ features outside the frozen dataset."""
    dataset, image_index, cache_dir = map(lambda p: Path(p).resolve(), (dataset, image_index, cache_dir))
    if cache_dir == dataset or dataset in cache_dir.parents:
        raise ValueError("Feature caches must be outside the dataset")
    rows = _read_index(image_index)
    payload = {}
    manifest_path = dataset / "MANIFEST.csv"
    if manifest_path.is_file():
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            payload = {row["path"]: row["sha256"] for row in csv.DictReader(handle)}
    settings = dict(feature_version=FEATURE_VERSION, detector="RootSIFT", max_side=max_side,
                    nfeatures=nfeatures, image_index_sha256=digest(image_index),
                    opencv_version=cv2.__version__, images=len(rows),
                    descriptor_scope="Background only: GT UAV boxes excluded with 8px feature-scale margin",
                    label_manifest_sha256=digest(manifest_path) if manifest_path.is_file() else None,
                    phash_protocol="OpenCV grayscale resize32x32 INTER_AREA, DCT8x8 excluding DC, median63")
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "feature_cache.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != settings:
        raise ValueError("Feature cache settings/index changed; choose a new cache directory")
    if not meta_path.exists():
        _json(meta_path, settings)
    cv2.setNumThreads(1)

    def one(row):
        name = row["image_id"].replace("/", "__").replace("\\", "__") + ".npz"
        target = cache_dir / name
        relative = Path(row["relative_path"])
        label_relative = Path("labels") / Path(*relative.parts[1:]).with_suffix(".txt")
        label_path = dataset / label_relative
        if not label_path.is_file():
            raise ValueError(f"Missing GT label; cannot safely select background: {label_path}")
        label_raw = label_path.read_bytes()
        label_hash = hashlib.sha256(label_raw).hexdigest()
        if payload and payload.get(label_relative.as_posix()) != label_hash:
            raise ValueError(f"Label differs from frozen manifest: {label_path}")
        if target.exists():
            with np.load(target, allow_pickle=False) as saved:
                if str(saved["source_sha256"]) != row["sha256"]:
                    raise ValueError(f"Cached source SHA differs: {row['image_id']}")
                if str(saved["label_sha256"]) != label_hash:
                    raise ValueError(f"Cached GT SHA differs: {row['image_id']}")
                count, phash = len(saved["points"]), int(saved["phash"])
                texture = float(saved["gray128"].std())
                background_fraction = float(np.mean(saved["background_mask128"] > 0))
        else:
            image = dataset / row["relative_path"]
            raw = image.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError(f"Image changed before feature extraction: {image}")
            boxes = []
            for line in label_raw.decode("utf-8-sig").splitlines():
                if not line.strip():
                    continue
                values = list(map(float, line.split()))
                if len(values) != 5 or values[0] != 0 or not np.isfinite(values).all() or min(values[3:]) <= 0:
                    raise ValueError(f"Invalid GT label for background masking: {label_path}")
                boxes.append(values[1:])
            features = extract_features(raw, max_side=max_side, nfeatures=nfeatures, boxes=boxes)
            features["source_sha256"] = np.array(row["sha256"])
            features["label_sha256"] = np.array(label_hash)
            temporary = target.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez(handle, **features)
            temporary.replace(target)
            count, phash = len(features["points"]), int(features["phash"])
            texture = float(features["gray128"].std())
            background_fraction = float(np.mean(features["background_mask128"] > 0))
        return dict(image_id=row["image_id"], sha256=row["sha256"], feature_file=target.name,
                    label_sha256=label_hash, keypoints=count, gray_std=texture,
                    background_fraction=background_fraction, phash=str(phash), phash_hex=f"{phash:016x}")

    output = cache_dir / "feature_index.csv"
    fields = ["image_id", "sha256", "feature_file", "label_sha256", "keypoints", "gray_std",
              "background_fraction", "phash", "phash_hex"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Dataset is small enough for these extraction futures; pair
            # verification below deliberately bounds its much larger queue.
            for index, result in enumerate(executor.map(one, rows), 1):
                writer.writerow(result)
                if index % 500 == 0 or index == len(rows):
                    handle.flush()
                    print(f"local features {index}/{len(rows)}", flush=True)
    return output


@lru_cache(maxsize=1024)
def _load_feature(path: str) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        return dict(points=saved["points"], descriptors=saved["descriptors"].astype(np.float32),
                    shape=saved["shape"], gray128=saved["gray128"],
                    background_mask128=saved["background_mask128"], phash=int(saved["phash"]))


def _coverage(points, shape):
    height, width = map(int, shape)
    if len(points) < 3:
        return 0.0, 0
    area = abs(cv2.contourArea(cv2.convexHull(points.astype(np.float32)))) / (width * height)
    bins = np.minimum(3, np.maximum(0, (points / np.array([width, height]) * 4).astype(int)))
    occupied = len({tuple(point) for point in bins})
    return float(area), occupied


def _plausibility(homography, shape_a, shape_b, policy):
    height, width = map(int, shape_a)
    target_h, target_w = map(int, shape_b)
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    if not np.isfinite(homography).all() or abs(homography[2, 2]) < 1e-12:
        return False, 0.0, 0.0
    homography = homography / homography[2, 2]
    denominators = np.c_[corners, np.ones(4)] @ homography[2]
    if not (np.all(denominators > 1e-5) or np.all(denominators < -1e-5)):
        return False, 0.0, 0.0
    projected = cv2.perspectiveTransform(corners[None], homography)[0]
    if not np.isfinite(projected).all() or not cv2.isContourConvex(projected):
        return False, 0.0, 0.0
    signed_area = cv2.contourArea(projected, oriented=True)
    if signed_area <= 0:  # Reject reflected or folded transformations.
        return False, 0.0, 0.0
    area_ratio = signed_area / (target_h * target_w)
    destination = np.array([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]], np.float32)
    try:
        intersection, _ = cv2.intersectConvexConvex(projected, destination)
    except cv2.error:
        return False, float(area_ratio), 0.0
    overlap = max(0.0, float(intersection)) / min(signed_area, target_h * target_w)
    edges = np.linalg.norm(np.roll(projected, -1, axis=0) - projected, axis=1)
    plausible = (policy.min_projected_area_ratio <= area_ratio <= policy.max_projected_area_ratio
                 and overlap >= policy.min_projected_overlap
                 and edges.min() > 1 and edges.max() / edges.min() < 12)
    return bool(plausible), float(area_ratio), float(overlap)


def verify_features(a: dict, b: dict, *, policy=POLICY, compute_ssim=True, matched_indices=None) -> dict:
    """Verify geometric support; no-support is not a different-scene verdict."""
    result = dict(phash_distance=(int(a["phash"]) ^ int(b["phash"])).bit_count(), ssim128="",
                  background_ssim128="", background_overlap_fraction=0.0,
                  keypoints_a=len(a["points"]), keypoints_b=len(b["points"]), mutual_ratio_matches=0,
                  ransac_inliers=0, inlier_ratio=0.0, hull_coverage_a=0.0, hull_coverage_b=0.0,
                  grid_cells_a=0, grid_cells_b=0, median_reprojection_px="", p95_reprojection_px="",
                  homography_plausible=False, projected_area_ratio=0.0, projected_overlap=0.0,
                  geometry_decision="NO_GEOMETRIC_SUPPORT", geometry_reason="too_few_mutual_matches")
    if compute_ssim:
        from skimage.metrics import structural_similarity

        score, score_map = structural_similarity(a["gray128"], b["gray128"], data_range=255, full=True)
        valid = ((a["background_mask128"] > 0) & (b["background_mask128"] > 0)).astype(np.uint8)
        valid = cv2.erode(valid, np.ones((7, 7), np.uint8)).astype(bool)
        result["ssim128"] = float(score)
        result["background_overlap_fraction"] = float(valid.mean())
        if valid.any():
            result["background_ssim128"] = float(score_map[valid].mean())
    if min(len(a["points"]), len(b["points"])) < 16:
        result.update(geometry_decision="REVIEW", geometry_reason="insufficient_local_texture")
        return result
    if matched_indices is None:
        desc_a, desc_b = a["descriptors"].astype(np.float32), b["descriptors"].astype(np.float32)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        forward = matcher.knnMatch(desc_a, desc_b, k=2)
        backward = matcher.knnMatch(desc_b, desc_a, k=2)
        passed_a = {m.queryIdx: m.trainIdx for neighbors in forward if len(neighbors) == 2
                    for m, n in [neighbors] if m.distance < policy.ratio * n.distance}
        passed_b = {m.queryIdx: m.trainIdx for neighbors in backward if len(neighbors) == 2
                    for m, n in [neighbors] if m.distance < policy.ratio * n.distance}
        matches = [(i, j) for i, j in passed_a.items() if passed_b.get(j) == i]
    else:
        matches = matched_indices
    result["mutual_ratio_matches"] = len(matches)
    if len(matches) < 8:
        return result
    points_a = np.array([a["points"][i] for i, _ in matches], np.float32)
    points_b = np.array([b["points"][j] for _, j in matches], np.float32)
    homography, mask = cv2.findHomography(points_a, points_b, cv2.USAC_MAGSAC,
                                          policy.ransac_px, maxIters=2000, confidence=0.999)
    if homography is None or mask is None:
        result.update(geometry_decision="REVIEW", geometry_reason="homography_not_established")
        return result
    keep = mask.ravel().astype(bool)
    inlier_a, inlier_b = points_a[keep], points_b[keep]
    count, ratio = int(keep.sum()), float(keep.mean())
    result.update(ransac_inliers=count, inlier_ratio=ratio)
    if count < 4:
        result.update(geometry_decision="REVIEW", geometry_reason="insufficient_ransac_support")
        return result
    coverage_a, grid_a = _coverage(inlier_a, a["shape"])
    coverage_b, grid_b = _coverage(inlier_b, b["shape"])
    predicted = cv2.perspectiveTransform(inlier_a[None], homography)[0]
    errors = np.linalg.norm(predicted - inlier_b, axis=1)
    median, p95 = float(np.median(errors)), float(np.quantile(errors, .95))
    plausible, area_ratio, overlap = _plausibility(homography, a["shape"], b["shape"], policy)
    result.update(hull_coverage_a=coverage_a, hull_coverage_b=coverage_b,
                  grid_cells_a=grid_a, grid_cells_b=grid_b,
                  median_reprojection_px=median, p95_reprojection_px=p95,
                  homography_plausible=plausible, projected_area_ratio=area_ratio,
                  projected_overlap=overlap)
    strong = (count >= policy.min_inliers and ratio >= policy.min_inlier_ratio
              and min(coverage_a, coverage_b) >= policy.min_hull_coverage
              and min(grid_a, grid_b) >= policy.min_grid_cells
              and median <= policy.max_median_error and p95 <= policy.max_p95_error and plausible)
    result.update(geometry_decision="CONFIRMED_GEOMETRY" if strong else "REVIEW",
                  geometry_reason="distributed_consistent_projective_support" if strong else "borderline_geometric_support")
    return result


def verify_candidate_pairs(candidates_csv, feature_index, output_csv, *, workers=6, resume=True,
                           compute_ssim=True, policy=POLICY):
    """Verify bounded batches; append resumably and preserve retrieval signals."""
    candidates_csv, feature_index, output_csv = map(lambda p: Path(p).resolve(),
                                                    (candidates_csv, feature_index, output_csv))
    with feature_index.open(newline="", encoding="utf-8") as handle:
        features = {row["image_id"]: row for row in csv.DictReader(handle)}
    settings = dict(geometry_version=GEOMETRY_VERSION, policy=asdict(policy),
                    candidates_sha256=digest(candidates_csv), feature_index_sha256=digest(feature_index),
                    compute_ssim=compute_ssim, opencv_version=cv2.__version__)
    metadata_path = output_csv.with_suffix(output_csv.suffix + ".metadata.json")
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if any(previous.get(key) != value for key, value in settings.items()):
            raise ValueError("Geometry inputs/policy changed; choose a new output CSV")
    elif output_csv.exists():
        raise ValueError("Existing geometry CSV has no pinned input metadata; choose a new output")
    seen = set()
    if output_csv.exists():
        if not resume:
            raise FileExistsError(output_csv)
        with output_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                seen.add(tuple(sorted((row["image_a"], row["image_b"]))))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not metadata_path.exists():
        _json(metadata_path, settings)
    cv2.setNumThreads(1)
    counts = {}

    def one(row):
        a, b = row["image_a"], row["image_b"]
        if a not in features or b not in features:
            raise ValueError(f"Candidate references an image outside feature cache: {a}, {b}")
        result = verify_features(_load_feature(str(feature_index.parent / features[a]["feature_file"])),
                                 _load_feature(str(feature_index.parent / features[b]["feature_file"])),
                                 policy=policy, compute_ssim=compute_ssim)
        return {**row, **result}

    with candidates_csv.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"image_a", "image_b"} <= set(reader.fieldnames):
            raise ValueError("Candidate CSV requires image_a and image_b")
        fields = list(reader.fieldnames) + [f for f in METRIC_FIELDS if f not in reader.fieldnames]
        existing = output_csv.exists()
        if existing:
            with output_csv.open(newline="", encoding="utf-8") as prior:
                if next(csv.reader(prior)) != fields:
                    raise ValueError("Existing geometry CSV header differs from the requested schema")
        with output_csv.open("a" if existing else "w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            if not existing:
                writer.writeheader()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                batch, total = [], 0
                for row in reader:
                    key = tuple(sorted((row["image_a"], row["image_b"])))
                    if key in seen or key[0] == key[1]:
                        continue
                    seen.add(key)
                    batch.append(row)
                    if len(batch) < 256:
                        continue
                    for result in executor.map(one, batch):
                        writer.writerow(result)
                        counts[result["geometry_decision"]] = counts.get(result["geometry_decision"], 0) + 1
                    total += len(batch)
                    destination.flush()
                    print(f"geometry pairs newly processed: {total}; {counts}", flush=True)
                    batch = []
                for result in executor.map(one, batch):
                    writer.writerow(result)
                    counts[result["geometry_decision"]] = counts.get(result["geometry_decision"], 0) + 1
                total += len(batch)
    _json(metadata_path, dict(
        **settings, new_pairs=total, new_decision_counts=counts, completed_pairs=len(seen),
        no_support_semantics="No geometric support is not proof of different scenes; review retrieval-risk candidates.",
        frozen_dataset_modified=False,
    ))
    return dict(new_pairs=total, counts=counts, output=str(output_csv))


class _Components:
    def __init__(self, images):
        self.parent = {image: image for image in images}

    def find(self, image):
        if image not in self.parent:
            raise ValueError(f"Image is not in the feature cache: {image}")
        current = image
        while self.parent[current] != current:
            current = self.parent[current]
        while self.parent[image] != image:
            previous = self.parent[image]
            self.parent[image] = current
            image = previous
        return current

    def union(self, first, second):
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def verify_candidate_graph(candidates_csv, feature_index, output_csv, *, workers=6, resume=True,
                           compute_ssim=True, policy=POLICY, batch_size=64, seed_edges=None):
    """Adaptively verify bridges and skip only confirmed-path relationships.

    SAME_SCENE_TRANSITIVE rows are *not* geometric edges. Components are rebuilt
    on resume from explicit confirmed seed/direct edges, never skipped rows.
    Input order therefore matters and is pinned by the candidate-file hash.
    """
    if batch_size < 1 or batch_size > 1024:
        raise ValueError("Graph batch_size must be between 1 and 1024")
    candidates_csv, feature_index, output_csv = map(lambda p: Path(p).resolve(),
                                                    (candidates_csv, feature_index, output_csv))
    with feature_index.open(newline="", encoding="utf-8") as handle:
        features = {row["image_id"]: row for row in csv.DictReader(handle)}
    components = _Components(features)
    seeds = []
    if seed_edges is not None:
        with Path(seed_edges).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                decision = row.get("decision") or row.get("geometry_decision") or row.get("review_decision")
                if decision not in {"CONFIRMED_GEOMETRY", "CONFIRMED_VISUAL", "CONFIRMED_SAME_SCENE"}:
                    raise ValueError("Every seed edge must explicitly confirm the same scene")
                components.union(row["image_a"], row["image_b"])
                seeds.append(dict(image_a=row["image_a"], image_b=row["image_b"],
                                  edge_provenance="confirmed_seed", decision=decision))
    settings = dict(graph_version=GRAPH_VERSION, geometry_version=GEOMETRY_VERSION,
                    policy=asdict(policy), candidates_sha256=digest(candidates_csv),
                    feature_index_sha256=digest(feature_index),
                    seed_edges_sha256=digest(seed_edges) if seed_edges is not None else None,
                    compute_ssim=compute_ssim, batch_size=batch_size, opencv_version=cv2.__version__)
    meta_path = output_csv.with_suffix(output_csv.suffix + ".metadata.json")
    seen = set()
    existing = output_csv.exists()
    if existing:
        if not resume:
            raise FileExistsError(output_csv)
        if not meta_path.is_file():
            raise ValueError("Existing graph CSV has no pinned input metadata")
        previous = json.loads(meta_path.read_text())
        if any(previous.get(key) != value for key, value in settings.items()):
            raise ValueError("Graph inputs/policy changed; choose a new output CSV")
        with output_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                seen.add(tuple(sorted((row["image_a"], row["image_b"]))))
                if row["geometry_decision"] == "CONFIRMED_GEOMETRY":
                    components.union(row["image_a"], row["image_b"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not meta_path.exists():
        _json(meta_path, settings)
    cv2.setNumThreads(1)
    counts = Counter()
    new_total = evaluated = 0

    def one(row):
        a, b = row["image_a"], row["image_b"]
        result = verify_features(_load_feature(str(feature_index.parent / features[a]["feature_file"])),
                                 _load_feature(str(feature_index.parent / features[b]["feature_file"])),
                                 policy=policy, compute_ssim=compute_ssim)
        return {**row, **result, "direct_geometry_evaluated": True}

    def transitive(row):
        a, b = features[row["image_a"]], features[row["image_b"]]
        result = {key: "" for key in METRIC_FIELDS}
        result.update(phash_distance=(int(a["phash"]) ^ int(b["phash"])).bit_count(),
                      keypoints_a=a["keypoints"], keypoints_b=b["keypoints"],
                      geometry_decision="SAME_SCENE_TRANSITIVE",
                      geometry_reason="connected_by_confirmed_seed_or_direct_edges_only",
                      direct_geometry_evaluated=False)
        return {**row, **result}

    with candidates_csv.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"image_a", "image_b"} <= set(reader.fieldnames):
            raise ValueError("Candidate CSV requires image_a and image_b")
        fields = list(reader.fieldnames) + [f for f in METRIC_FIELDS + ["direct_geometry_evaluated"]
                                          if f not in reader.fieldnames]
        if existing:
            with output_csv.open(newline="", encoding="utf-8") as old:
                if next(csv.reader(old)) != fields:
                    raise ValueError("Existing graph CSV has a different header")
        with output_csv.open("a" if existing else "w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            if not existing:
                writer.writeheader()

            def save(result):
                nonlocal new_total
                writer.writerow(result)
                new_total += 1
                counts[result["geometry_decision"]] += 1

            with ThreadPoolExecutor(max_workers=workers) as executor:
                batch = []
                for row in reader:
                    a, b = row["image_a"], row["image_b"]
                    key = tuple(sorted((a, b)))
                    if key in seen or a == b:
                        continue
                    seen.add(key)
                    if components.find(a) == components.find(b):
                        save(transitive(row))
                    else:
                        batch.append(row)
                    if len(batch) < batch_size:
                        if new_total and new_total % 5000 == 0:
                            destination.flush()
                            print(f"graph candidates {new_total}; actual matches {evaluated}; {dict(counts)}", flush=True)
                        continue
                    for result in executor.map(one, batch):
                        evaluated += 1
                        if result["geometry_decision"] == "CONFIRMED_GEOMETRY":
                            components.union(result["image_a"], result["image_b"])
                        save(result)
                    batch = []
                    destination.flush()
                    if evaluated % 512 == 0:
                        print(f"graph candidates {new_total}; actual matches {evaluated}; {dict(counts)}", flush=True)
                for result in executor.map(one, batch):
                    evaluated += 1
                    if result["geometry_decision"] == "CONFIRMED_GEOMETRY":
                        components.union(result["image_a"], result["image_b"])
                    save(result)
    edge_path = output_csv.with_suffix(output_csv.suffix + ".confirmed_edges.csv")
    with edge_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_a", "image_b", "edge_provenance", "decision"])
        writer.writeheader()
        writer.writerows(seeds)
        with output_csv.open(newline="", encoding="utf-8") as pairs:
            for row in csv.DictReader(pairs):
                if row["geometry_decision"] == "CONFIRMED_GEOMETRY":
                    writer.writerow(dict(image_a=row["image_a"], image_b=row["image_b"],
                                         edge_provenance="direct_local_geometry", decision="CONFIRMED_GEOMETRY"))
    group_path = output_csv.with_suffix(output_csv.suffix + ".components.csv")
    roots = sorted({components.find(image) for image in features})
    names = {root: f"GEOM_{index:05d}" for index, root in enumerate(roots, 1)}
    with group_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "geometry_component_id"])
        writer.writeheader()
        for image in sorted(features):
            writer.writerow(dict(image_id=image, geometry_component_id=names[components.find(image)]))
    summary = dict(**settings, new_pairs=new_total, new_direct_geometry_evaluations=evaluated,
                   new_decision_counts=dict(counts), completed_candidate_pairs=len(seen),
                   geometry_components=len(roots), confirmed_edges_file=str(edge_path),
                   components_file=str(group_path),
                   transitive_semantics="Skipped relationships are not direct geometric edges; confirmed paths only.")
    _json(meta_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    features = sub.add_parser("features")
    features.add_argument("--dataset", required=True)
    features.add_argument("--image-index", required=True)
    features.add_argument("--cache-dir", required=True)
    features.add_argument("--workers", type=int, default=6)
    features.add_argument("--max-side", type=int, default=640)
    features.add_argument("--nfeatures", type=int, default=1024)
    for name in ("pairs", "graph-pairs"):
        pairs = sub.add_parser(name)
        pairs.add_argument("--candidates", required=True)
        pairs.add_argument("--feature-index", required=True)
        pairs.add_argument("--output", required=True)
        pairs.add_argument("--workers", type=int, default=6)
        pairs.add_argument("--no-ssim", action="store_true")
        if name == "graph-pairs":
            pairs.add_argument("--batch-size", type=int, default=64)
            pairs.add_argument("--seed-edges")
    args = parser.parse_args(argv)
    if args.command == "features":
        print(build_feature_cache(args.dataset, args.image_index, args.cache_dir, workers=args.workers,
                                  max_side=args.max_side, nfeatures=args.nfeatures))
    elif args.command == "pairs":
        print(json.dumps(verify_candidate_pairs(args.candidates, args.feature_index, args.output,
                                                workers=args.workers, compute_ssim=not args.no_ssim), indent=2))
    else:
        verify_candidate_graph(args.candidates, args.feature_index, args.output, workers=args.workers,
                               compute_ssim=not args.no_ssim, batch_size=args.batch_size,
                               seed_edges=args.seed_edges)


if __name__ == "__main__":
    main()
