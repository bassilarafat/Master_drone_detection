"""Publish the completed design/source diagnostics into the editable dashboard."""
from collections import Counter
import csv
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DESIGN = Path("06_results/negative_mining_design_20261120")
SOURCE = Path("06_results/external_negative_source_audit_20261120")
POOL = Path("06_results/external_discovery_pool_audit_20261121")
YIELD = Path("06_results/external_negative_yield_check_20261121")
REVIEW = "docs/NEGATIVE_MINING_DESIGN_REVIEW.md"
DECISION = "configs/paper/negative_mining_design_decision_20261121.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source_link(path):
    value = str(path).replace("\\", "/")
    target = value[5:] if value.startswith("docs/") else "../"+value
    return f"[{value}]({target})"


def card(row):
    sources = [row["source_artifact"], *row["supporting_artifacts"].split(" | ")]
    return (f"#### {row['id']} · {row['title']}\n\n"
            f"**{row['status']}** · {row['evidence_level']} · RQ3\n\n"
            f"- **Population** — {row['dataset']} · {row['split']}\n"
            f"- **Finding** — {row['finding']}\n"
            f"- **Limit** — {row['limitations']}\n"
            f"- **Next dependency** — {row['next_dependency']}\n"
            f"- **Evidence** — {' · '.join(source_link(p) for p in sources if p)}\n\n")


def main():
    existing_ids = {row["id"] for row in read_csv("docs/experiment_registry.csv")}
    published_ids = {"D-DESIGN-STRICT", "D-EXT-SOURCE", "D-EXT-BOUNDARY", "D-EXT-YIELD"}
    if existing_ids & published_ids:
        raise ValueError("This preparation snapshot was already published; preserve it and use a new version")
    pool = read_json(POOL / "pool_audit.json")
    exact = read_json(POOL / "exact_overlap_check.json")
    outcome = read_json(YIELD / "progress.json")
    assert pool["status"] == "NO_DECLARED_PROTOCOL_CROSSING_FOUND"
    assert exact["overlap_count"] == 0
    assert outcome["status"] == "COMPLETED" and outcome["frames_processed"] == 2398
    count, detections = outcome["proposed_crops"], outcome["detections_at_threshold"]
    source = read_json(SOURCE / "source_audit.json")
    document = Path(REVIEW).read_text(encoding="utf-8")
    document = document.replace("The full-pool audit is processing all 2,398 predeclared candidate frames.",
                                "The full-pool audit completed all 2,398 predeclared candidate frames.")
    boundary = f"""## Completed full-pool findings

The joint index contains 11,849 images. Retrieval produced **{pool['candidate_pairs']:,} cross-boundary candidate pairs**, all evaluated by the existing geometric pipeline. It produced **{pool['risk_or_confirmed_links']} confirmed or policy-risk links** and **{pool['low_texture_external_frames']} external low-texture frames**. The candidate family therefore occupies a separate component under the declared allocation screen. This is a protocol-qualified claim over the indexed pool.

Freshly decoded external pixels had **zero exact overlaps** with the released reference images, using decoded-pixel hashes reused by matching original image-byte identities. A CPU/GPU parity sample checked 128 pair decisions with zero differences.

The geometric pipeline found 9,623 pairs with too few mutual matches and 808 pairs with insufficient reference texture. No pair supplied enough support for a RANSAC fit. These outcomes supply bounded screening evidence. They do not prove that all possible relationships were recovered or establish native recording identities.

[Full audit](../{POOL.as_posix()}/pool_audit.json) · [Exact-overlap check](../{POOL.as_posix()}/exact_overlap_check.json) · [Frame provenance](../{POOL.as_posix()}/frame_provenance_check.json) · [CPU/GPU parity](../{POOL.as_posix()}/cpu_gpu_parity.json)

## Completed exploratory yield check

At the frozen confidence threshold 0.50, E1 seed 17 returned **{detections} detection(s)** over the 2,398 predeclared external Train frames. The one-crop-per-source rule produced **{count} candidate crop(s)** for review. A matched study requires 512 accepted hard crops per arm, so the current source is **insufficient at the declared settings**.

Candidate crops have UAV-free verification pending. The detection count is a proposal-yield diagnostic. Ground-truth negative status and an operational false-positive rate were not established by this check. The source is one nighttime filename family, with limited diversity relative to the Clean_v2 building-background errors.

The result is preserved under its predeclared threshold. Further source or threshold changes require a new prospective version with this exposure history. A different allocation-separated discovery source with enough reviewable hard examples is required before preparing the 512/512 matched bundle.

[Frozen rules](../configs/paper/external_negative_yield_check_v1.json) · [Yield outcome](../{YIELD.as_posix()}/progress.json) · [Complete raw predictions](../{YIELD.as_posix()}/predictions.jsonl) · [Candidate review](../{YIELD.as_posix()}/review.csv)

"""
    document = document.replace("## Dependencies before E2/E3", boundary+"## Dependencies before E2/E3")
    document = document.replace("1. Confirm full-pool allocation separation and inspect any flagged source links.",
                                "1. Identify another allocation-separated discovery source with sufficient candidate yield. The audited nighttime source passed the finite boundary screen but fell short of 512 reviewable proposals.")
    Path(REVIEW).write_text(document, encoding="utf-8")
    registry_path = Path("docs/experiment_registry.csv")
    rows = read_csv(registry_path)
    fields = list(rows[0])
    old_count = len(rows)
    previous_counts = Counter(row["evidence_level"] for row in rows)
    old_oof = next(row for row in rows if row["id"] == "D-OOF")
    old_oof.update(status="Rejected under full-block requirement",
                   finding="Full allocation-block separation was selected by the researcher. Internal Train OOF leaves zero eligible discovery-training images.",
                   limitations="Clean_v2 Train is one allocation block. Direct buffering is insufficient for the chosen requirement.",
                   next_dependency="Find an independently allocated discovery source with sufficient hard-candidate yield.",
                   source_artifact=DECISION, supporting_artifacts=" | ".join([REVIEW, str(DESIGN / "design_audit.json")]))

    def append(identifier, title, level, artifact, finding, limitation, supporting, **extra):
        if any(row["id"] == identifier for row in rows):
            raise ValueError(f"Record already exists: {identifier}")
        row = dict.fromkeys(fields, "")
        row.update(id=identifier, branch="D", research_question="RQ3", title=title,
                   kind="Source/design diagnostic", evidence_level=level, status="Complete",
                   source_artifact=str(artifact).replace("\\", "/"), finding=finding, limitations=limitation,
                   next_dependency="A sufficient allocation-separated candidate source, independent review and a prospective E2/E3 source amendment.",
                   supporting_artifacts=" | ".join(str(path).replace("\\", "/") for path in supporting), **extra)
        rows.append(row)
        return row

    new = [append("D-DESIGN-STRICT", "Full-block mining-design feasibility", "DIAGNOSTIC", DESIGN / "design_audit.json",
                  "6,299 Train images occupy one block. Strict internal mining leaves zero discovery-training images. Buffered designs were rejected by the researcher.",
                  "Buffered role manifests are diagnostic proposals with residual same-block dependence.", [DECISION, REVIEW], dataset="Clean_v2", split="Train"),
           append("D-EXT-SOURCE", "RGPT Train source inventory", "DIAGNOSTIC", SOURCE / "source_audit.json",
                  "160 Train clips contain 149,528 annotation entries. Conservative filename-family reservation leaves 8 clips and 2,398 absence-flag candidates.",
                  "One nighttime filename family. Archive identity, native recording separation and UAV-free verification remain unverified.", [REVIEW, SOURCE / "upstream_provenance_review.json"], dataset="Local antiuav_rgpt", split="Train metadata"),
           append("D-EXT-BOUNDARY", "Full external candidate-pool boundary audit", "DIAGNOSTIC", POOL / "pool_audit.json",
                  f"2,398 external Train frames and 9,451 reference images. {pool['candidate_pairs']:,} evaluated crossing pairs, zero declared-policy links and zero exact overlaps.",
                  "Finite retrieval and texture limits. Native recording independence remains unproven.", [REVIEW, POOL / "exact_overlap_check.json", POOL / "cpu_gpu_parity.json"], dataset="External candidate pool / Clean_v2 reference", split="External Train source only"),
           append("D-EXT-YIELD", "Fixed-threshold independent-source candidate yield", "EXPLORATORY", YIELD / "progress.json",
                  f"At confidence .50, {detections} detection(s) over 2,398 source frames produced {count} unverified crop(s), below 512 required hard negatives.",
                  "A proposal-yield diagnostic. UAV-free status and population-level false-alarm performance are unverified.", [REVIEW, YIELD / "review.csv", "configs/paper/external_negative_yield_check_v1.json"],
                  dataset="External Train absence-flag frames", split="External discovery", model="YOLO26s-P2", seed="17", epochs="100 (existing detector)",
                  checkpoint="04_experiments/controlled_yolo26_v3/E1_Y26_P2/seed_17/run/weights/last.pt",
                  checkpoint_sha256=outcome["checkpoint_sha256"], protocol="configs/paper/external_negative_yield_check_v1.json",
                  evaluator="Frozen exploratory candidate yield, confidence .50")]
    rows = sorted(rows, key=lambda row: (row["branch"], row["id"]))
    with registry_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    dashboard_path = Path("docs/RESEARCH_PROGRESS_DASHBOARD.md")
    dashboard = dashboard_path.read_text(encoding="utf-8")
    dashboard = dashboard.replace("As of 20 November 2026", "As of 21 November 2026")
    dashboard = re.sub(r"#### D-OOF · .*?(?=\n#### |\n</details>)", card(old_oof), dashboard, flags=re.S)
    match = re.search(r"(<summary><b>D · Background learning</b>.*?</summary>)(.*?)(</details>)", dashboard, re.S)
    if match is None:
        raise ValueError("Missing branch-D card group")
    header = re.sub(r"\d+ records", f"{sum(row['branch']=='D' for row in rows)} records", match[1])
    dashboard = dashboard[:match.start()]+header+match[2]+"\n"+"".join(card(row) for row in new)+match[3]+dashboard[match.end():]
    counts = Counter(row["evidence_level"] for row in rows)
    for level, number in previous_counts.items():
        dashboard = dashboard.replace(f"| {level} | {number} |", f"| {level} | {counts[level]} |")
    dashboard = dashboard.replace(f"**{old_count} records**", f"**{len(rows)} records**")
    dashboard = dashboard.replace("**92** with execution or audit evidence", "**96** with execution or audit evidence")
    dashboard += f"\n## Negative discovery update · 21 November 2026\n\nThe researcher requires full allocation-block separation. Internal OOF was retired. The external source passed its declared overlap screen, but yielded only {count} unverified candidate crop(s) at confidence .50, below the required 512. E2/E3 remain blocked by candidate-source sufficiency, verification and source/protocol integration. {source_link(REVIEW)}\n"
    dashboard_path.write_text(dashboard, encoding="utf-8")
    for name in ("docs/research_questions_and_evidence.md", "docs/RESEARCH_PROGRESS_FLOWCHART.md"):
        path = Path(name)
        text = path.read_text(encoding="utf-8")
        text = text.replace('T["Clean_v2 Train scene groups only"]', 'T["Retired internal OOF · researcher requires full allocation-block separation"]')
        if name.endswith("evidence.md"):
            text += f"\n## Current negative-discovery decision\n\nFull allocation-block separation is required. Internal buffered OOF was rejected. The complete external source screen found zero declared allocation crossings, but the fixed-threshold yield was only {count} unverified crop(s). The 512/512 matched bundle remains unavailable. {source_link(REVIEW)}\n"
        path.write_text(text, encoding="utf-8")
    claims = Path("docs/SCIENTIFIC_CLAIMS_STATUS.md")
    claim = f"| A sufficient independent negative bundle is ready | DIAGNOSTIC boundary screen passed, EXPLORATORY yield {count} crop(s) at confidence .50 | Source feasibility only | The source passed the finite allocation screen but supplies insufficient unverified proposals. | 512 verified hard negatives or operational false-alarm benefit established. | Another sufficient source, independent review and matched random controls. | {source_link(REVIEW)} |\n"
    claims.write_text(claims.read_text(encoding="utf-8").rstrip()+"\n"+claim, encoding="utf-8")
    html_path = Path("docs/research_progress_dashboard.html")
    html = html_path.read_text(encoding="utf-8")
    embedded = json.dumps(read_csv(registry_path), ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    html = re.sub(r'(<script id="snapshot" type="application/json">).*?(</script>)',
                  lambda match: match[1]+embedded+match[2], html, flags=re.S)
    html = html.replace("Repository snapshot · 20 November 2026", "Repository snapshot · 21 November 2026")
    panel = f'<section><h2>Negative-data preparation</h2><div class="panel"><p>Full allocation-block separation is required. The 2,398-frame external source passed the declared overlap screen, but supplied only <strong>{count} unverified candidate crop(s)</strong> at confidence 0.50. The required 512 hard negatives remain unavailable.</p><p><a href="NEGATIVE_MINING_DESIGN_REVIEW.md">Source audits and current dependencies</a></p></div></section>'
    html = html.replace('<section id="register">', panel+'<section id="register">')
    html_path.write_text(html, encoding="utf-8")
    print(f"Updated registry and dashboard with {len(rows)} records, preserving six official controlled runs")


if __name__ == "__main__":
    import os
    os.chdir(ROOT)
    main()
