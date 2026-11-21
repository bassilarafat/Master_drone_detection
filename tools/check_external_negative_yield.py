"""Run the frozen exploratory yield check on an allocation-screened source.

This produces candidate detections and reviewable crops, without assuming that
the source existence labels independently verify UAV-free images.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from drone_detection.data_audit import digest
from drone_detection.paper_common import read_csv, write_csv
from drone_detection.paper_runner import configure_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_path, output = args.plan.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT / "06_results"):
        raise ValueError("Write outside all datasets")
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(plan_path.read_text())
    lock = json.loads(plan_path.with_suffix(".lock.json").read_text())
    assert digest(plan_path) == lock["plan_sha256"]
    audit_path = ROOT / plan["required_source_audit"]
    audit = json.loads(audit_path.read_text())
    assert audit["status"] == plan["required_source_status"]
    assert audit["allocation_separation_under_declared_screen"]
    assert audit["low_texture_external_frames"] == 0
    exact = json.loads((audit_path.parent / "exact_overlap_check.json").read_text())
    assert exact["status"] == "NO_EXACT_OVERLAP" and exact["overlap_count"] == 0
    index_path = ROOT / plan["source_index"]
    index = read_csv(index_path)
    assert len(index) == audit["source_frames"]
    assert all(row["source_split"] == "train" and int(row["annotation_exist"]) == 0 for row in index)
    checkpoint = ROOT / plan["model"]["checkpoint"]
    assert digest(checkpoint) == plan["model"]["sha256"]
    for row in index:
        frame = (ROOT / row["relative_path"]).resolve()
        assert frame.is_relative_to(index_path.parent / "frames")
        assert digest(frame) == row["sha256"]
    configure_runtime()
    from ultralytics import YOLO
    model = YOLO(str(checkpoint))
    output.mkdir(parents=True)
    (output / "candidates").mkdir()
    state = dict(status="PREDICTING", evidence_level="EXPLORATORY", frames_expected=len(index),
                 plan_sha256=digest(plan_path), source_audit_sha256=digest(audit_path),
                 checkpoint_sha256=digest(checkpoint), started_at_utc=datetime.now(timezone.utc).isoformat())

    def progress(**values):
        state.update(values)
        temporary = output / "progress.json.tmp"
        temporary.write_text(json.dumps(state, indent=2)+"\n", encoding="utf-8")
        temporary.replace(output / "progress.json")

    proposed, total_detections = [], 0
    try:
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for start in range(0, len(index), plan["inference"]["batch"]):
                rows = index[start:start+plan["inference"]["batch"]]
                paths = [str(ROOT / row["relative_path"]) for row in rows]
                settings = plan["inference"]
                results = model.predict(paths, imgsz=settings["imgsz"], conf=settings["confidence"],
                                        iou=settings["nms_iou"], max_det=settings["max_det"],
                                        rect=settings["rect"], augment=settings["augment"],
                                        nms=settings["nms"], device=settings["device"],
                                        save=False, save_txt=False, save_crop=False, verbose=False)
                assert len(results) == len(rows)
                assert not model.predictor.model.end2end
                for row, result in zip(rows, results):
                    assert Path(result.path).resolve() == (ROOT / row["relative_path"]).resolve()
                    predictions = [dict(box=list(map(float, box)), confidence=float(score), category=int(category))
                                   for box, score, category in zip(result.boxes.xyxy.cpu().tolist(),
                                                                  result.boxes.conf.cpu().tolist(),
                                                                  result.boxes.cls.cpu().tolist())]
                    total_detections += len(predictions)
                    record = dict(sequence=row["sequence"], frame_index=int(row["frame_index"]),
                                  source_split="train", frame_file=row["relative_path"],
                                  frame_sha256=row["sha256"], annotation_exist=0,
                                  negative_verified=False, predictions=predictions)
                    handle.write(json.dumps(record, allow_nan=False)+"\n")
                    for pred_id, prediction in enumerate(predictions):
                        proposed.append(dict(sequence=row["sequence"], frame_index=int(row["frame_index"]),
                                             frame_file=row["relative_path"], frame_sha256=row["sha256"],
                                             width=int(row["width"]), height=int(row["height"]),
                                             pred_id=pred_id, **prediction))
                handle.flush()
                if start % 100 == 0:
                    progress(frames_processed=start+len(rows), detections_at_threshold=total_detections)
                    print(f"Yield check {start+len(rows)}/{len(index)} frames, detections={total_detections}", flush=True)
        review, seen = [], set()
        for prediction in sorted(proposed, key=lambda r: (-r["confidence"], r["sequence"], r["frame_index"], r["pred_id"])):
            key = (prediction["sequence"], prediction["frame_index"])
            if key in seen:
                continue
            seen.add(key)
            width, height = prediction["width"], prediction["height"]
            if min(width, height) < 256:
                continue
            x1, y1, x2, y2 = prediction["box"]
            x, y = max(0, min(width-256, round((x1+x2)/2)-128)), max(0, min(height-256, round((y1+y2)/2)-128))
            bounds = [x, y, x+256, y+256]
            name = f"{prediction['sequence']}_{prediction['frame_index']:06d}_{prediction['pred_id']}"
            target = output / "candidates" / (name+".png")
            with Image.open(ROOT / prediction["frame_file"]) as image:
                image.convert("RGB").crop(bounds).save(target)
            review.append(dict(sample_id=name, method="hard_candidate", source_split="external_train",
                               sequence=prediction["sequence"], frame_index=prediction["frame_index"],
                               full_context=prediction["frame_file"], source_frame_sha256=prediction["frame_sha256"],
                               crop_path=target.relative_to(ROOT).as_posix(), crop_sha256=digest(target),
                               crop_xyxy=json.dumps(bounds), confidence=prediction["confidence"],
                               mining_checkpoint_sha256=state["checkpoint_sha256"],
                               verification="PENDING", reviewer="", reviewed_at_utc="", notes=""))
            if len(review) == plan["crop"]["max_proposals"]:
                break
        fields = ["sample_id", "method", "source_split", "sequence", "frame_index", "full_context",
                  "source_frame_sha256", "crop_path", "crop_sha256", "crop_xyxy", "confidence",
                  "mining_checkpoint_sha256", "verification", "reviewer", "reviewed_at_utc", "notes"]
        write_csv(output / "review.csv", review, fields)
        write_csv(output / "candidate_catalog.csv", review, fields)
        assert digest(checkpoint) == state["checkpoint_sha256"]
        assert digest(plan_path) == state["plan_sha256"]
        for row in index:
            assert digest(ROOT / row["relative_path"]) == row["sha256"]
        progress(status="COMPLETED", frames_processed=len(index), detections_at_threshold=total_detections,
                 proposed_crops=len(review), enough_proposals_for_512=len(review) >= 512,
                 independently_verified_negatives=None, training_performed=False,
                 ended_at_utc=datetime.now(timezone.utc).isoformat(),
                 interpretation="Yield of unverified candidates at the predeclared threshold; no operational false-positive rate claimed")
    except BaseException as exc:
        progress(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
