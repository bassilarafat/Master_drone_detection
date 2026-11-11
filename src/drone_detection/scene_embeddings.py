"""DINOv2 inference on a frozen image index, with no network or training.

The official repository and checkpoint must already exist locally. Images are
letterboxed without cropping. CLS and valid-region mean patch descriptors are
saved separately for complementary scene retrieval.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_NAME = "dinov2_vitb14"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def fit_layout(width: int, height: int, side: int = 518) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0 or side < 14 or side % 14:
        raise ValueError("Positive image dimensions and a side divisible by14 are required")
    factor = min(side / width, side / height)
    resized_width = max(1, min(side, round(width * factor)))
    resized_height = max(1, min(side, round(height * factor)))
    return ((side - resized_width) // 2, (side - resized_height) // 2,
            resized_width, resized_height)


def preprocess(row: dict, dataset: Path, side: int = 518) -> np.ndarray:
    """Hash and decode the same byte buffer; never repair or write an image."""
    path = dataset / row["relative_path"]
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != row["sha256"]:
        raise ValueError(f"Frozen image changed: {row['relative_path']}")
    with Image.open(io.BytesIO(content)) as opened:
        image = opened.convert("RGB")
        if image.size != (int(row["width"]), int(row["height"])):
            raise ValueError(f"Index/image dimensions differ: {row['relative_path']}")
        x, y, width, height = fit_layout(*image.size, side)
        image = image.resize((width, height), Image.Resampling.BICUBIC)
        neutral = tuple(round(value * 255) for value in IMAGENET_MEAN)
        canvas = Image.new("RGB", (side, side), neutral)
        canvas.paste(image, (x, y))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = (array - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def valid_patch_mask(width: int, height: int, side: int = 518) -> np.ndarray:
    """Use patches whose centers lie in original image content, not padding."""
    x, y, resized_width, resized_height = fit_layout(width, height, side)
    centers = np.arange(side // 14, dtype=np.float32) * 14 + 7
    mask = ((centers[None, :] >= x) & (centers[None, :] < x + resized_width)
            & (centers[:, None] >= y) & (centers[:, None] < y + resized_height))
    if not mask.any():
        raise ValueError("No valid content patch remains after letterboxing")
    return mask.reshape(-1).astype(np.float32)


def load_index(dataset: Path, index_path: Path, *, limit: int | None = None) -> tuple[list[dict], dict]:
    """Bind every indexed image to the frozen manifest without repeating all file reads.

    The caller performs full snapshot verification before/after the workflow;
    extraction additionally checks each image's bytes as that image is read.
    """
    freeze = dataset / "FREEZE.json"
    manifest = dataset / "MANIFEST.csv"
    frozen = json.loads(freeze.read_text(encoding="utf-8"))
    if sha256(manifest) != frozen["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    with manifest.open(newline="", encoding="utf-8") as handle:
        images = {row["path"]: row["sha256"] for row in csv.DictReader(handle)
                  if row["path"].startswith("images/")}
    with index_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    paths = [row["relative_path"] for row in rows]
    if len(paths) != len(set(paths)) or set(paths) != set(images):
        raise ValueError("Index must cover each frozen image exactly once")
    for row in rows:
        if images[row["relative_path"]] != row["sha256"]:
            raise ValueError(f"Index SHA differs from manifest: {row['relative_path']}")
    identity = dict(dataset_name=frozen["dataset_name"], manifest_sha256=sha256(manifest),
                    freeze_sha256=sha256(freeze), input_index_sha256=sha256(index_path),
                    indexed_images=len(rows))
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    return rows, identity


def repo_fingerprint(repo: Path) -> str:
    files = [repo / "hubconf.py"] + sorted((repo / "dinov2").rglob("*.py"))
    if not files[0].is_file() or len(files) < 2:
        raise ValueError(f"Not a local official DINOv2 repository: {repo}")
    result = hashlib.sha256()
    for path in files:
        result.update(path.relative_to(repo).as_posix().encode())
        result.update(bytes.fromhex(sha256(path)))
    return result.hexdigest()


def extract(dataset, index_path, repo, checkpoint, output, *, side=518, batch_size=4,
            device="cuda:0", workers=4, resume=False, limit=None):
    import torch
    import torch.nn.functional as functional

    dataset, index_path, repo, checkpoint, output = [
        Path(value).resolve() for value in (dataset, index_path, repo, checkpoint, output)
    ]
    if output == dataset or dataset in output.parents:
        raise ValueError("Embedding outputs must be outside the frozen dataset")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if batch_size <= 0 or workers <= 0:
        raise ValueError("Positive batch_size and workers are required")
    fit_layout(1, 1, side)
    rows, source = load_index(dataset, index_path, limit=limit)
    identity = dict(source, model=MODEL_NAME, checkpoint_sha256=sha256(checkpoint),
                    model_repository_url="https://github.com/facebookresearch/dinov2",
                    checkpoint_source_url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
                    extractor_code_sha256=sha256(Path(__file__)),
                    repository_code_sha256=repo_fingerprint(repo), side=side,
                    descriptor="L2-normalized DINOv2 CLS768 and valid-region mean-patch768, separately",
                    preprocessing="Full RGB image; aspect-preserving bicubic resize; ImageNet-mean letterbox; ImageNet normalization",
                    patch_mean_mask="Patch center lies inside resized image content",
                    images=len(rows), subset=limit is not None)
    started = time.monotonic()
    progress_path = output / "progress.json"
    done = 0
    if output.exists():
        if not resume:
            raise FileExistsError(f"Use a new output directory or explicit --resume: {output}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["identity"] != identity:
            raise ValueError("Resume provenance differs from this model/source/preprocessing")
        if progress.get("complete"):
            raise FileExistsError("Complete embeddings are preserved; select a new output version")
        done = int(progress["processed"])
        if not 0 <= done <= len(rows):
            raise ValueError("Resume progress is outside the image index")
        cls_array = np.lib.format.open_memmap(output / "embeddings.npy", mode="r+")
        patch_array = np.lib.format.open_memmap(output / "patch_mean_embeddings.npy", mode="r+")
        if cls_array.shape != (len(rows), 768) or patch_array.shape != (len(rows), 768):
            raise ValueError("Resume embedding dimensions are invalid")
    else:
        output.mkdir(parents=True, exist_ok=False)
        cls_array = np.lib.format.open_memmap(output / "embeddings.npy", mode="w+", dtype="float32", shape=(len(rows), 768))
        patch_array = np.lib.format.open_memmap(output / "patch_mean_embeddings.npy", mode="w+", dtype="float32", shape=(len(rows), 768))
        fields = list(rows[0]) + ["feature_row", "content_x", "content_y", "content_width", "content_height"]
        with (output / "image_index.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for number, row in enumerate(rows):
                x, y, width, height = fit_layout(int(row["width"]), int(row["height"]), side)
                writer.writerow(dict(row, feature_row=number, content_x=x, content_y=y,
                                     content_width=width, content_height=height))
        _write_json(progress_path, dict(identity=identity, processed=0, complete=False))
    print(f"Loading local {MODEL_NAME}; images={len(rows)}; resume={done}", flush=True)
    model = torch.hub.load(str(repo), MODEL_NAME, source="local", pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False).to(device)
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        with torch.inference_mode():
            for start in range(done, len(rows), batch_size):
                batch_rows = rows[start:start + batch_size]
                tensors = list(pool.map(lambda row: preprocess(row, dataset, side), batch_rows))
                inputs = torch.from_numpy(np.stack(tensors)).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cuda):
                    features = model.forward_features(inputs)
                    cls = functional.normalize(features["x_norm_clstoken"].float(), dim=1)
                    patches = features["x_norm_patchtokens"].float()
                    masks = np.stack([valid_patch_mask(int(row["width"]), int(row["height"]), side)
                                      for row in batch_rows])
                    mask = torch.from_numpy(masks).to(device).unsqueeze(-1)
                    mean_patch = functional.normalize((patches * mask).sum(1) / mask.sum(1), dim=1)
                if not torch.isfinite(cls).all() or not torch.isfinite(mean_patch).all():
                    raise ValueError("Nonfinite embedding")
                end = start + len(batch_rows)
                cls_array[start:end] = cls.cpu().numpy()
                patch_array[start:end] = mean_patch.cpu().numpy()
                done = end
                if done % 500 < batch_size or done == len(rows):
                    cls_array.flush()
                    patch_array.flush()
                    elapsed = time.monotonic() - started
                    _write_json(progress_path, dict(identity=identity, processed=done, complete=False,
                                                   updated_at_utc=datetime.now(timezone.utc).isoformat()))
                    print(f"DINOv2 embeddings: {done}/{len(rows)}; elapsed={elapsed:.1f}s", flush=True)
    finally:
        pool.shutdown(wait=True)
        cls_array.flush()
        patch_array.flush()
    if any(not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=2e-4)
           for values in (cls_array, patch_array)):
        raise ValueError("Final descriptors are not finite unit vectors")
    # Protect against a concurrent metadata or model-source replacement.
    if sha256(dataset / "FREEZE.json") != source["freeze_sha256"] or sha256(dataset / "MANIFEST.csv") != source["manifest_sha256"]:
        raise ValueError("Frozen release metadata changed during extraction")
    if sha256(index_path) != source["input_index_sha256"]:
        raise ValueError("Input image index changed during extraction")
    if sha256(checkpoint) != identity["checkpoint_sha256"] or repo_fingerprint(repo) != identity["repository_code_sha256"]:
        raise ValueError("Model checkpoint or code changed during extraction")
    if sha256(Path(__file__)) != identity["extractor_code_sha256"]:
        raise ValueError("Embedding extractor code changed during extraction")
    result = dict(identity, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.monotonic() - started, inference_only=True,
                  training_performed=False, torch_version=torch.__version__, numpy_version=np.__version__,
                  device=str(device), gpu=torch.cuda.get_device_name(device) if cuda else None,
                  batch_size=batch_size, precision="float16 forward, float32 normalized output" if cuda else "float32",
                  descriptors_sha256=sha256(output / "embeddings.npy"),
                  patch_descriptors_sha256=sha256(output / "patch_mean_embeddings.npy"),
                  feature_index_sha256=sha256(output / "image_index.csv"))
    _write_json(output / "extraction.json", result)
    _write_json(progress_path, dict(identity=identity, processed=done, complete=True))
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "index", "repo", "checkpoint", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--side", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    extract(args.dataset, args.index, args.repo, args.checkpoint, args.output,
            side=args.side, batch_size=args.batch_size, workers=args.workers,
            device=args.device, resume=args.resume, limit=args.limit)


if __name__ == "__main__":
    main()
