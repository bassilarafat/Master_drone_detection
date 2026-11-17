"""Resume a local checkpoint with tuned batch/workers and frozen-data guards."""
import argparse
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / '.cache/ultralytics-local')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    from drone_detection.config import load_experiment
    from drone_detection.models import load_model
    from drone_detection.training import _without_snapshot_cache_writes
    config = load_experiment('configs/experiments/yolo26s_clean_v2_local.yaml')
    identity = config.validate_files()
    checkpoint = (ROOT / args.checkpoint).resolve()
    if not checkpoint.is_relative_to(ROOT / '04_experiments/local_clean_v2'):
        raise ValueError('Checkpoint must belong to this local training workflow.')
    backup = checkpoint.with_name('before_performance_tuning.pt')
    if not backup.exists():
        shutil.copy2(checkpoint, backup)
    model = load_model(checkpoint)
    if Path(model.ckpt['train_args']['data']).resolve() != config.data.resolve():
        raise ValueError('Checkpoint dataset does not match Clean_v2.')
    record = dict(checkpoint=str(checkpoint), batch=config.train['batch'],
                  workers=config.train['workers'], resumed_at_utc=datetime.now(timezone.utc).isoformat(),
                  dataset_snapshot=identity, status='running',
                  note='Batch/workers changed after epoch 3 for local throughput tuning.')
    record_path = checkpoint.parent.parent / 'performance_resume.json'
    record_path.write_text(json.dumps(record, indent=2))
    try:
        with _without_snapshot_cache_writes(config.data.parent):
            model.train(resume=True, data=str(config.data), batch=config.train['batch'],
                        workers=config.train['workers'], cache=False, device='0')
        record['status'] = 'completed'
    except BaseException:
        record['status'] = 'interrupted_or_failed'
        raise
    finally:
        after = config.validate_files()
        if after != identity:
            raise ValueError('Dataset identity changed during training.')
        record['dataset_verified_after_training'] = True
        record_path.write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
