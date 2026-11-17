"""Measure actual augmented training throughput without saving candidate weights."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / '.cache/ultralytics-local')
OUT = ROOT / '04_experiments/local_clean_v2/benchmark'


class BenchmarkComplete(Exception):
    pass


def candidate(batch, workers):
    import torch
    from drone_detection.config import load_experiment
    from drone_detection.models import load_model
    from drone_detection.training import _without_snapshot_cache_writes

    config = load_experiment('configs/experiments/yolo26s_clean_v2_local.yaml')
    model = load_model(config.model)
    count = 0
    start = None
    result = {'batch': batch, 'workers': workers, 'status': 'failed'}

    def measure(trainer):
        nonlocal count, start
        count += 1
        if count == 20:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
        if count == 100:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            result.update(status='ok', images_per_second=80 * trainer.batch_size / elapsed,
                          seconds=elapsed,
                          peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                          actual_batch=trainer.batch_size)
            raise BenchmarkComplete()

    model.add_callback('on_train_batch_end', measure)
    options = config.training_kwargs()
    options.update(batch=batch, workers=workers, project=str(OUT),
                   name=f'b{batch}_w{workers}', save=False, plots=False, val=False,
                   epochs=100)
    try:
        with _without_snapshot_cache_writes(config.data.parent):
            model.train(**options)
    except BenchmarkComplete:
        pass
    except Exception as exc:
        result['error'] = str(exc)
    finally:
        loader = getattr(getattr(model, 'trainer', None), 'train_loader', None)
        if loader is not None:
            loader.close()
        validator = getattr(getattr(model, 'trainer', None), 'validator', None)
        loader = getattr(validator, 'dataloader', None)
        if loader is not None:
            loader.close()
        (OUT / f'b{batch}_w{workers}.json').write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int)
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.batch:
        candidate(args.batch, args.workers)
        return
    from drone_detection.config import load_experiment
    config = load_experiment('configs/experiments/yolo26s_clean_v2_local.yaml')
    config.validate_files()
    results = []
    for batch, workers in [(4, 2), (8, 2), (8, 4), (12, 4)]:
        print(f'Benchmark batch={batch}, workers={workers}', flush=True)
        with (OUT / f'b{batch}_w{workers}.log').open('w') as log:
            process = subprocess.run([sys.executable, '-u', __file__, '--batch', str(batch),
                                      '--workers', str(workers)], cwd=ROOT,
                                     stdout=log, stderr=subprocess.STDOUT)
        path = OUT / f'b{batch}_w{workers}.json'
        result = json.loads(path.read_text()) if path.exists() else {
            'batch': batch, 'workers': workers, 'status': 'failed', 'exit_code': process.returncode}
        results.append(result)
        print(json.dumps(result), flush=True)
        (OUT / 'results.json').write_text(json.dumps(results, indent=2))
    config.validate_files()
    print('Dataset integrity verified after benchmarks.', flush=True)


if __name__ == '__main__':
    main()
