"""Training orchestration; model definitions and CLI parsing live elsewhere."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ExperimentConfig


def run_experiment(config: ExperimentConfig) -> Any:
    import json

    config.validate_files()
    from .models import load_model

    model = load_model(config.model, pretrained=config.pretrained)
    results = model.train(**config.training_kwargs())
    # Ultralytics can increment an existing name; use its actual output directory.
    output_dir = Path(model.trainer.save_dir)
    (output_dir / "experiment.json").write_text(
        json.dumps(config.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Training complete. Results: {output_dir}")
    return results
