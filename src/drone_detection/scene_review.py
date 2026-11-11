"""Separate confirmed scene edges from conservative unresolved allocation links."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .data_audit import digest
from .scene_graph import Components, read_csv

RISK_POLICY = dict(
    version=5,
    strong_retrieval="Either DINO descriptor cosine>=.85: broad conservative co-allocation only; semantically similar distinct scenes may share an allocation block.",
    coherent_local_patch="inliers>=6, ratio>=.50, median error<=1.5, p95<=3.5: weak context or shared overlay; conservative co-allocation only, never automatic scene confirmation.",
    localized_context="inliers>=24, ratio>=.75, min hull>=.01, min grid>=2, median error<=1.5 and p95<=3.5; no embedding/whole-frame homography prerequisite for review",
    unidentifiable_backgrounds="All images with fewer than16 background keypoints share one conservative allocation family; this is not a same-scene assertion.",
    partial_geometry="inliers>=8, ratio>=.30, plausible homography, min hull>=.015, min grid>=3, cosine>=.65",
    corroborated_retrieval="cosine>=.94 and (pHash<=16 or background SSIM>=.60)",
    dual_descriptor="CLS cosine>=.97 and patch-mean cosine>=.94",
    low_texture="min keypoints<16, cosine>=.90 and (pHash<=12 or background SSIM>=.70)",
    action="Keep unresolved high-risk pairs in one allocation block without asserting the same scene.",
    non_link_semantics="No confirmed link is not proof that two scenes are different.",
)


def _number(row, name, default=0.0):
    value = row.get(name)
    return float(value) if value not in (None, "") else default


def risk_reason(row):
    cosine = _number(row, "cosine_similarity")
    distance = _number(row, "phash_distance", 64)
    ssim = _number(row, "background_ssim128", _number(row, "ssim128", -1))
    if cosine >= .85:
        return "strong_global_retrieval_identity_unresolved"
    if (_number(row, "ransac_inliers") >= 6 and _number(row, "inlier_ratio") >= .50
        and _number(row, "median_reprojection_px", 999) <= 1.5
        and _number(row, "p95_reprojection_px", 999) <= 3.5):
        return "coherent_local_context_or_overlay_requires_colocation"
    if (_number(row, "ransac_inliers") >= 24 and _number(row, "inlier_ratio") >= .75
        and min(_number(row, "hull_coverage_a"), _number(row, "hull_coverage_b")) >= .01
        and min(_number(row, "grid_cells_a"), _number(row, "grid_cells_b")) >= 2
        and _number(row, "median_reprojection_px", 999) <= 1.5
        and _number(row, "p95_reprojection_px", 999) <= 3.5):
        return "localized_consistent_background_requires_review"
    partial = (_number(row, "ransac_inliers") >= 8 and _number(row, "inlier_ratio") >= .30
               and str(row.get("homography_plausible")).lower() == "true"
               and min(_number(row, "hull_coverage_a"), _number(row, "hull_coverage_b")) >= .015
               and min(_number(row, "grid_cells_a"), _number(row, "grid_cells_b")) >= 3 and cosine >= .65)
    if partial:
        return "partial_distributed_background_geometry"
    if cosine >= .94 and (distance <= 16 or ssim >= .60):
        return "corroborated_high_retrieval_similarity"
    if _number(row, "cosine_cls") >= .97 and _number(row, "cosine_patch") >= .94:
        return "both_dino_descriptors_very_close"
    if min(_number(row, "keypoints_a"), _number(row, "keypoints_b")) < 16 and cosine >= .90 and (distance <= 12 or ssim >= .70):
        return "low_texture_identity_unresolved"
    return ""


def prepare_review(index_csv, geometry_csv, output, *, manual_json=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    images = {row["image_id"]: row for row in read_csv(index_csv)}
    graph = Components(images)
    manual = {}
    if manual_json:
        for row in json.loads(Path(manual_json).read_text(encoding="utf-8")):
            key = tuple(sorted((row["image_a"], row["image_b"])))
            if key in manual or not set(key) <= images.keys():
                raise ValueError("Manual scene review has duplicate or unknown image IDs")
            if row["decision"] not in {"CONFIRMED_SAME_SCENE", "DIFFERENT_SCENE", "UNCERTAIN_COLOCATE"}:
                raise ValueError("Manual scene review must use an explicit decision")
            manual[key] = row
    with Path(geometry_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = tuple(sorted((row["image_a"], row["image_b"])))
            override = manual.get(key, {}).get("decision")
            if row["geometry_decision"] == "CONFIRMED_GEOMETRY" and override not in {"DIFFERENT_SCENE", "UNCERTAIN_COLOCATE"}:
                graph.union(*key)
    for key, row in manual.items():
        if row["decision"] == "CONFIRMED_SAME_SCENE":
            graph.union(*key)
    output.mkdir(parents=True)
    fields = ["image_a", "image_b", "review_decision", "reviewer", "reason", "risk_reason", "geometry_decision",
              "cosine_similarity", "cosine_cls", "cosine_patch", "phash_distance", "ssim", "ransac_inliers",
              "inlier_ratio", "hull_coverage_a", "hull_coverage_b", "group_a", "group_b"]
    counts, risk_counts = Counter(), Counter()
    representatives = {}
    with (output / "scene_pair_review.csv").open("w", newline="", encoding="utf-8") as dest:
        writer = csv.DictWriter(dest, fieldnames=fields)
        writer.writeheader()
        with Path(geometry_csv).open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                a, b = row["image_a"], row["image_b"]
                key = tuple(sorted((a, b)))
                reason = risk_reason(row)
                same = graph.find(a) == graph.find(b)
                if key in manual:
                    decision = manual[key]["decision"]
                    reviewer = "Codex visual inspection"
                    rationale = manual[key].get("reason", "Explicit visual decision")
                elif row["geometry_decision"] == "CONFIRMED_GEOMETRY":
                    decision, reviewer, rationale = "CONFIRMED_SAME_SCENE", "background_geometry_policy", row["geometry_reason"]
                elif same:
                    decision, reviewer, rationale = "CONFIRMED_PATH", "confirmed_graph", "Connected through independently confirmed background edges"
                elif reason:
                    decision, reviewer, rationale = "UNCERTAIN_COLOCATE", "conservative_allocation_policy", "Identity unresolved; keep together without claiming a confirmed scene"
                    risk_counts[reason] += 1
                    group_pair = tuple(sorted((graph.find(a), graph.find(b))))
                    if group_pair not in representatives or _number(row, "cosine_similarity") > _number(representatives[group_pair], "cosine_similarity"):
                        representatives[group_pair] = dict(row, risk_reason=reason, group_a=group_pair[0], group_b=group_pair[1])
                else:
                    decision, reviewer, rationale = "NO_CONFIRMED_LINK", "not_visually_adjudicated", "No accepted scene edge; this is not a different-scene ground-truth label"
                counts[decision] += 1
                writer.writerow(dict(image_a=a, image_b=b, review_decision=decision, reviewer=reviewer,
                                     reason=rationale, risk_reason=reason, geometry_decision=row["geometry_decision"],
                                     cosine_similarity=row.get("cosine_similarity", ""), cosine_cls=row.get("cosine_cls", ""),
                                     cosine_patch=row.get("cosine_patch", ""), phash_distance=row.get("phash_distance", ""),
                                     ssim=row.get("ssim128", ""), ransac_inliers=row.get("ransac_inliers", ""),
                                     inlier_ratio=row.get("inlier_ratio", ""), hull_coverage_a=row.get("hull_coverage_a", ""),
                                     hull_coverage_b=row.get("hull_coverage_b", ""), group_a=graph.find(a), group_b=graph.find(b)))
    selected = sorted(representatives.values(), key=lambda row: (-_number(row, "cosine_similarity"), -_number(row, "ransac_inliers")))
    (output / "borderline_representatives.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    report = dict(decisions=dict(counts), uncertain_risk_reasons=dict(risk_counts),
                  uncertain_component_pairs=len(selected), policy=RISK_POLICY,
                  geometry_sha256=digest(geometry_csv), manual_sha256=digest(manual_json) if manual_json else None)
    (output / "review_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "geometry", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--manual")
    args = parser.parse_args(argv)
    prepare_review(args.index, args.geometry, args.output, manual_json=args.manual)


if __name__ == "__main__":
    main()
