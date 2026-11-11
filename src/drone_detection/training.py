"""Training orchestration; model definitions and CLI parsing live elsewhere."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, snapshot_root_for_yaml


@contextmanager
def _without_snapshot_cache_writes(dataset_root: Path):
    """Suppress Ultralytics label-cache files without editing the installation.

    cache=False only disables image caching. Label scanning otherwise writes
    labels/train.cache and labels/val.cache into the dataset automatically.
    Non-snapshot cache writes retain their original behavior.
    """
    from unittest.mock import patch

    from ultralytics.data import dataset, utils

    original_save = utils.save_dataset_cache_file
    root = dataset_root.resolve()

    def save_cache(prefix, path, values, version):
        if Path(path).resolve().is_relative_to(root):
            values["version"] = version
            return None
        return original_save(prefix, path, values, version)

    with patch.object(dataset, "save_dataset_cache_file", save_cache), patch.object(
        utils, "save_dataset_cache_file", save_cache
    ):
        yield


def run_experiment(config: ExperimentConfig) -> Any:
    import json

    snapshot = config.validate_files()
    snapshot_identity = (
        {key: snapshot[key] for key in ("manifest_sha256", "freeze_sha256")}
        if snapshot is not None else None
    )
    snapshot_root = snapshot_root_for_yaml(config.data)
    training_options = config.training_kwargs()
    initial_record = {"model": str(config.model), "train": training_options}
    from .models import load_model

    model = load_model(config.model, pretrained=config.pretrained)
    if snapshot is None:
        results = model.train(**training_options)
    else:
        from .dataset_snapshot import verify_snapshot

        training_error = None
        verification_error = None
        try:
            with _without_snapshot_cache_writes(snapshot_root):
                results = model.train(**training_options)
        except BaseException as exc:
            training_error = exc
            raise
        finally:
            try:
                after = verify_snapshot(snapshot_root)
                for field, expected in snapshot_identity.items():
                    if after.get(field) != expected:
                        raise ValueError(f"Snapshot identity changed: {field}")
            except (OSError, ValueError) as exc:
                verification_error = exc
            trainer = getattr(model, "trainer", None)
            save_dir = getattr(trainer, "save_dir", None)
            if save_dir is not None and Path(save_dir).is_dir():
                record = dict(initial_record)
                record["dataset_snapshot"] = snapshot
                record["dataset_snapshot_verified_after_training"] = verification_error is None
                record["training_status"] = "failed" if training_error or verification_error else "completed"
                if verification_error:
                    record["dataset_snapshot_error"] = str(verification_error)
                (Path(save_dir) / "experiment.json").write_text(
                    json.dumps(record, indent=2) + "\n", encoding="utf-8"
                )
            if verification_error:
                raise ValueError(f"Frozen dataset changed during the experiment: {verification_error}") from verification_error
    # Ultralytics can increment an existing name; use its actual output directory.
    output_dir = Path(model.trainer.save_dir)
    if snapshot is None:
        (output_dir / "experiment.json").write_text(
            json.dumps(config.as_dict(), indent=2) + "\n", encoding="utf-8"
        )
    print(f"Training complete. Results: {output_dir}")
    return results
