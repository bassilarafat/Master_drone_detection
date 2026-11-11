"""Global visual retrieval and confirmed scene components for all dataset images."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .data_audit import digest, dump_csv


class Components:
    def __init__(self, images):
        self.parent = {image: image for image in images}
        self.size = {image: 1 for image in images}

    def find(self, image):
        if image not in self.parent:
            raise ValueError(f"Unknown graph image: {image}")
        root = image
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[image] != image:
            previous = self.parent[image]
            self.parent[image] = root
            image = previous
        return root

    def union(self, first, second):
        a, b = self.find(first), self.find(second)
        if a == b:
            return False
        if self.size[a] < self.size[b] or (self.size[a] == self.size[b] and a > b):
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        return True

    def groups(self):
        result = defaultdict(list)
        for image in sorted(self.parent):
            result[self.find(image)].append(image)
        return sorted(result.values(), key=lambda group: group[0])


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def exact_neighbors(values, k, *, device="cuda:0", batch_size=512):
    """Exact global cosine retrieval, excluding each image from its neighbors."""
    import torch

    values = np.asarray(values, dtype=np.float32)
    if len(values) < 2 or k < 1:
        raise ValueError("Retrieval needs two images and positive k")
    if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=3e-4):
        raise ValueError("Cosine retrieval requires normalized embeddings")
    k = min(k, len(values)-1)
    matrix = torch.from_numpy(values.copy()).to(device)
    neighbors, scores = [], []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            similarity = matrix[start:stop] @ matrix.T
            similarity[torch.arange(stop-start, device=device), torch.arange(start, stop, device=device)] = -2
            best, indices = torch.topk(similarity, k, dim=1, sorted=True)
            neighbors.append(indices.cpu().numpy())
            scores.append(best.cpu().numpy())
    return np.concatenate(neighbors), np.concatenate(scores)


def retrieve_candidates(index_csv, embeddings_dir, feature_index, output, *, k=60, phash_k=20,
                        phash_max=8, device="cuda:0", known_links=None):
    index_csv, embeddings_dir, feature_index, output = map(Path, (index_csv, embeddings_dir, feature_index, output))
    if output.exists():
        raise FileExistsError(output)
    rows = read_csv(index_csv)
    ids = [r["image_id"] for r in rows]
    positions = {image: i for i, image in enumerate(ids)}
    features = {r["image_id"]: r for r in read_csv(feature_index)}
    if set(ids) != set(features) or len(positions) != len(ids):
        raise ValueError("Retrieval/feature indices must cover each image exactly once")
    extraction_index = read_csv(embeddings_dir / "image_index.csv")
    if [r["image_id"] for r in extraction_index] != ids:
        raise ValueError("Embedding rows differ from the declared image index")
    progress = json.loads((embeddings_dir / "progress.json").read_text(encoding="utf-8"))
    extraction = json.loads((embeddings_dir / "extraction.json").read_text(encoding="utf-8"))
    if progress.get("complete") is not True or progress.get("processed") != len(ids):
        raise ValueError("Only complete embedding extractions can build a scene graph")
    if digest(embeddings_dir / "embeddings.npy") != extraction["descriptors_sha256"] or digest(embeddings_dir / "patch_mean_embeddings.npy") != extraction["patch_descriptors_sha256"]:
        raise ValueError("Embedding arrays changed after extraction")
    for row, extracted in zip(rows, extraction_index):
        if row["sha256"] != extracted["sha256"]:
            raise ValueError("Embedding input hashes differ from the image index")
        if row["sha256"] != features[row["image_id"]]["sha256"]:
            raise ValueError("Feature hashes differ from the image index")
    cls = np.load(embeddings_dir / "embeddings.npy", mmap_mode="r")
    patch = np.load(embeddings_dir / "patch_mean_embeddings.npy", mmap_mode="r")
    if len(cls) != len(ids) or len(patch) != len(ids):
        raise ValueError("Incomplete embeddings")
    output.mkdir(parents=True)
    pairs = {}

    def add(a, b, source, rank=9999):
        if a == b:
            return
        key = (min(a, b), max(a, b))
        if key not in pairs:
            pairs[key] = [set(), rank, False]
        pairs[key][0].add(source)
        pairs[key][1] = min(rank, pairs[key][1])

    for name, values in (("dino_cls", cls), ("dino_patchmean", patch)):
        neighbors, scores = exact_neighbors(values, k, device=device)
        np.savez(output / f"{name}_neighbors.npz", indices=neighbors, cosine=scores)
        for a, selected in enumerate(neighbors):
            for rank, b in enumerate(selected, 1):
                add(a, int(b), name, rank)
        print(f"global retrieval {name}: {len(pairs)} unique pairs", flush=True)
    phashes = np.array([int(features[image]["phash"]) for image in ids], dtype=np.uint64)
    lookup = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    phash_k = min(phash_k, len(ids)-1)
    if phash_k > 0:
        for start in range(0, len(ids), 128):
            xor = np.bitwise_xor(phashes[start:start+128, None], phashes[None, :])
            distances = lookup[xor.view(np.uint8).reshape(len(xor), len(ids), 8)].sum(axis=2)
            distances[np.arange(len(xor)), np.arange(start, start+len(xor))] = 999
            chosen = np.argpartition(distances, phash_k-1, axis=1)[:, :phash_k]
            for offset, selected in enumerate(chosen):
                for b in selected:
                    if distances[offset, b] <= phash_max:
                        add(start+offset, int(b), "phash_additional")
    if known_links:
        for link in json.loads(Path(known_links).read_text(encoding="utf-8")):
            if link.get("same_scene_confirmed") is True:
                a = positions[link["split_a"] + "/" + link["image_a"]]
                b = positions[link["split_b"] + "/" + link["image_b"]]
                add(a, b, "prior_visual_review", 0)
                pairs[(min(a,b), max(a,b))][2] = True
    keys = sorted(pairs)
    cos_cls, cos_patch = np.empty(len(keys), np.float32), np.empty(len(keys), np.float32)
    for start in range(0, len(keys), 4096):
        selected = np.array(keys[start:start+4096])
        cos_cls[start:start+len(selected)] = np.einsum("ij,ij->i", cls[selected[:,0]], cls[selected[:,1]])
        cos_patch[start:start+len(selected)] = np.einsum("ij,ij->i", patch[selected[:,0]], patch[selected[:,1]])
    candidates = []
    for number, (a, b) in enumerate(keys):
        sources, rank, known = pairs[(a,b)]
        fa, fb = features[ids[a]], features[ids[b]]
        candidates.append(dict(
            image_a=ids[a], image_b=ids[b], original_split_a=rows[a]["old_split"], original_split_b=rows[b]["old_split"],
            cosine_similarity=float(max(cos_cls[number], cos_patch[number])), cosine_cls=float(cos_cls[number]),
            cosine_patch=float(cos_patch[number]), phash_distance=int(phashes[a] ^ phashes[b]).bit_count(),
            embedding_rank_min=rank, retrieval_sources=";".join(sorted(sources)),
            known_visual_same_scene=known, keypoints_a=int(fa["keypoints"]), keypoints_b=int(fb["keypoints"]),
            ssim="", review_decision="PENDING",
        ))
    candidates.sort(key=lambda r: (not r["known_visual_same_scene"],
                                  min(r["keypoints_a"], r["keypoints_b"]) < 16,
                                  -r["cosine_similarity"], r["embedding_rank_min"], r["image_a"], r["image_b"]))
    target = output / "scene_pair_candidates.csv"
    dump_csv(target, candidates, list(candidates[0]))
    report = dict(images=len(ids), candidates=len(candidates), k_per_descriptor=k, phash_k=phash_k,
                  phash_max=phash_max, retrieval="Exact global cosine kNN; union of CLS, patchmean, pHash and documented prior review",
                  image_index_sha256=digest(index_csv), feature_index_sha256=digest(feature_index),
                  cls_sha256=digest(embeddings_dir / "embeddings.npy"), patch_sha256=digest(embeddings_dir / "patch_mean_embeddings.npy"),
                  candidate_sha256=digest(target), no_single_similarity_threshold_confirms_a_scene=True)
    (output / "retrieval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return target


def make_groups(index_csv, geometry_csv, output, *, review_csv=None, constraints_csv=None):
    """Confirmed edges define scenes; uncertain constraints only define blocks."""
    rows = read_csv(index_csv)
    ids = [row["image_id"] for row in rows]
    scene = Components(ids)
    reviews = []
    if review_csv:
        with Path(review_csv).open(newline="", encoding="utf-8-sig") as handle:
            reviews = [row for row in csv.DictReader(handle)
                       if row["review_decision"] in {"DIFFERENT_SCENE", "UNCERTAIN_COLOCATE", "CONFIRMED_SAME_SCENE"}]
    veto = {tuple(sorted((row["image_a"], row["image_b"]))) for row in reviews
            if row["review_decision"] in {"DIFFERENT_SCENE", "UNCERTAIN_COLOCATE"}}
    with Path(geometry_csv).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["geometry_decision"] == "CONFIRMED_GEOMETRY" and tuple(sorted((row["image_a"], row["image_b"]))) not in veto:
                scene.union(row["image_a"], row["image_b"])
    for row in reviews:
        if row["review_decision"] == "CONFIRMED_SAME_SCENE":
            scene.union(row["image_a"], row["image_b"])
    blocks = Components(ids)
    for group in scene.groups():
        for image in group[1:]:
            blocks.union(group[0], image)
    for row in reviews:
        if row["review_decision"] == "UNCERTAIN_COLOCATE":
            blocks.union(row["image_a"], row["image_b"])
    constraints = read_csv(constraints_csv) if constraints_csv else []
    for row in constraints:
        if row.get("constraint") != "COLOCATE_UNRESOLVED_IDENTITY":
            raise ValueError("Allocation constraints cannot assert a confirmed scene")
        blocks.union(row["image_a"], row["image_b"])
    scene_ids = {image: f"SCENE_{i:05d}" for i, group in enumerate(scene.groups(), 1) for image in group}
    block_ids = {image: f"BLOCK_{i:05d}" for i, group in enumerate(blocks.groups(), 1) for image in group}
    result = [dict(image=row["image_id"], old_split=row["old_split"], image_sha256=row["sha256"],
                   scene_group_id=scene_ids[row["image_id"]], allocation_block_id=block_ids[row["image_id"]]) for row in rows]
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    dump_csv(output / "scene_groups.csv", result, list(result[0]))
    sizes = sorted([len(group) for group in scene.groups()], reverse=True)
    report = dict(images=len(ids), scene_groups=len(sizes), singleton_groups=sum(size == 1 for size in sizes),
                  allocation_blocks=len(blocks.groups()), largest_scene_groups=sizes[:30],
                  largest_allocation_blocks=sorted([len(group) for group in blocks.groups()], reverse=True)[:30],
                  geometry_sha256=digest(geometry_csv), review_sha256=digest(review_csv) if review_csv else None)
    report["allocation_constraints_sha256"] = digest(constraints_csv) if constraints_csv else None
    (output / "groups_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return result, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    retrieval = sub.add_parser("retrieve")
    for name in ("index", "embeddings", "features", "output"):
        retrieval.add_argument("--" + name, required=True)
    retrieval.add_argument("--k", type=int, default=60)
    retrieval.add_argument("--phash-k", type=int, default=20)
    retrieval.add_argument("--known-links")
    retrieval.add_argument("--device", default="cuda:0")
    groups = sub.add_parser("groups")
    for name in ("index", "geometry", "output"):
        groups.add_argument("--" + name, required=True)
    groups.add_argument("--review")
    groups.add_argument("--constraints")
    args = parser.parse_args(argv)
    if args.command == "retrieve":
        retrieve_candidates(args.index, args.embeddings, args.features, args.output, k=args.k,
                            phash_k=args.phash_k, device=args.device, known_links=args.known_links)
    else:
        make_groups(args.index, args.geometry, args.output, review_csv=args.review, constraints_csv=args.constraints)


if __name__ == "__main__":
    main()
