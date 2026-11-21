"""Publish the second exploratory yield snapshot without replacing version 1."""
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
YIELD = "06_results/external_negative_yield_check_v2_20261121"
PLAN = "configs/paper/external_negative_yield_check_v2.json"
IDENTIFIER = "D-EXT-YIELD-V2"


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    import os
    os.chdir(ROOT)
    registry = read_csv("docs/experiment_registry.csv")
    if any(row["id"] == IDENTIFIER for row in registry):
        raise ValueError("Version 2 already published; preserve the published snapshot")
    outcome = json.loads(Path(YIELD, "progress.json").read_text(encoding="utf-8"))
    assert outcome["status"] == "COMPLETED"
    assert (outcome["frames_processed"], outcome["detections_at_threshold"], outcome["proposed_crops"]) == (2398, 4, 4)
    assert outcome["training_performed"] is False
    review = read_csv(Path(YIELD, "review.csv"))
    notes = {
        660: "Blue luminous patch at left frame edge; identity indeterminate. Adjacent to frame 659.",
        659: "Blue luminous patch at left frame edge; identity indeterminate. Adjacent to frame 660; not an independent event.",
        404: "Dim red point near camera overlay; identity indeterminate.",
        780: "Bright green/yellow luminous streak; identity indeterminate.",
    }
    timestamp = datetime.now(timezone.utc).isoformat()
    for row in review:
        assert row["verification"] == "PENDING"
        row.update(verification="UNCERTAIN", reviewer="Codex assistant visual inspection",
                   reviewed_at_utc=timestamp,
                   notes=notes[int(row["frame_index"])]+" Crop and full context inspected; independent UAV-free review pending. Not accepted as a verified negative.")
    write_csv(Path(YIELD, "review.csv"), review)
    base = next(row for row in registry if row["id"] == "D-EXT-YIELD")
    row = dict(base)
    row.update(id=IDENTIFIER, title="Operating-threshold external-source candidate yield v2",
               evaluator="Frozen exploratory candidate yield v2, confidence .25",
               protocol=PLAN, source_artifact=YIELD+"/progress.json",
               finding="At confidence .25, 4 detections over 2,398 source frames produced 4 unverified crops, including one adjacent-frame pair; below 512 required hard negatives.",
               limitations="Exploratory amendment after v1 exposure. Four uncertain candidates, not four independent verified negatives. No operational false-alarm rate established.",
               supporting_artifacts=" | ".join(["docs/NEGATIVE_MINING_DESIGN_REVIEW.md", YIELD+"/review.csv", PLAN, YIELD+"/context_review_sheet.png"]))
    registry.append(row)
    registry.sort(key=lambda item: (item["branch"], item["id"]))
    write_csv("docs/experiment_registry.csv", registry)

    path = Path("docs/NEGATIVE_MINING_DESIGN_REVIEW.md")
    document = path.read_text(encoding="utf-8")
    document = document.replace("[Full audit](../06_results/external_discovery_pool_audit_20261121/pool_audit.json) ?", "[Full audit](../06_results/external_discovery_pool_audit_20261121/pool_audit.json) ·")
    document = document.replace("A different allocation-separated discovery source with enough reviewable hard examples is required before preparing the 512/512 matched bundle.", "The tested cohort is insufficient under version 1. Additional source development must retain this result and declare subsequent changes prospectively.")
    addition = f"""## Operating-threshold yield check, version 2

Version 2 was frozen after observing version 1 and before generating version-2 predictions. It changes confidence from 0.50 to the existing official operating threshold 0.25. The model, 2,398-frame source cohort and crop rules stay fixed. This is exploratory source development, with prior exposure recorded in the plan.

| Version | Confidence | Source frames | Detections | Proposed crops | Accepted independently verified negatives |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.50 | 2,398 | 1 | 1 | Not established |
| 2 | 0.25 | 2,398 | 4 | 4 | Not established |

All four version-2 crops remain **UNCERTAIN** after assistant inspection of crops and full contexts. Two are neighboring frames 659/660 showing the same blue edge feature; the others show a dim red point and a luminous streak. These are four frame proposals, with correlated observations, and cannot be called four independent negative events. Independent review is pending. Both checks fall below the required 512 hard crops even before verification.

The filename-family reservation is an engineering choice that protects potential external evaluation families. The inventory contains 27 Train filename families; another 4,938 absence-flag entries were excluded by that reservation. Their allocation separation and candidate yield have not been established. The present result applies to the tested cohort at the two declared thresholds. It does not establish that all local or external sources are unusable. A broader source policy would need a new prospective version, exposure accounting and explicit effects on the reserved external evaluation population.

[Version-2 frozen rules](../{PLAN}) · [Outcome](../{YIELD}/progress.json) · [Raw predictions](../{YIELD}/predictions.jsonl) · [Candidate review](../{YIELD}/review.csv) · [Context sheet](../{YIELD}/context_review_sheet.png) · [All Train source families](../06_results/external_negative_source_audit_20261120/source_family_options.csv)

"""
    document = document.replace("## Dependencies before E2/E3", addition+"## Dependencies before E2/E3")
    document = document.replace("1. Identify another allocation-separated discovery source with sufficient candidate yield. The audited nighttime source passed the finite boundary screen but fell short of 512 reviewable proposals.", "1. Establish a sufficient allocation-separated discovery cohort through a prospectively declared source policy. The tested nighttime cohort passed the finite boundary screen but supplied only one/four proposals at confidence 0.50/0.25.")
    document = document.replace("The frozen external_negative_yield_check_v1.json plan drives check_external_negative_yield.py.", "The separately frozen version-1 and version-2 plans drive check_external_negative_yield.py.")
    path.write_text(document, encoding="utf-8")

    path = Path("docs/RESEARCH_PROGRESS_DASHBOARD.md")
    document = path.read_text(encoding="utf-8")
    document = document.replace("**138 records**", "**139 records**").replace("**96** with execution or audit evidence", "**97** with execution or audit evidence").replace("| EXPLORATORY | 11 |", "| EXPLORATORY | 12 |")
    document = document.replace("**Next dependency: prepare verified E2/E3 data and resolve the mining design.**", "**Next dependency: a sufficient allocation-separated source and verified E2/E3 data.**")
    match = re.search(r"(<summary><b>D · Background learning</b>.*?</summary>)(.*?)(</details>)", document, re.S)
    assert match
    card = f"\n#### {IDENTIFIER} · {row['title']}\n\n**Complete** · EXPLORATORY · RQ3\n\n- **Finding** — {row['finding']}\n- **Limit** — {row['limitations']}\n- **Next dependency** — {row['next_dependency']}\n- **Evidence** — [Version-2 outcome](../{YIELD}/progress.json) · [Frozen plan](../{PLAN}) · [Candidate review](../{YIELD}/review.csv)\n\n"
    document = document[:match.start()]+match[1].replace("12 records", "13 records")+match[2]+card+match[3]+document[match.end():]
    document += "\nVersion 2 used the official operating confidence .25 and produced four uncertain candidate crops from the same 2,398 frames, including an adjacent-frame pair. Version 1 (.50, one crop) is preserved. Neither yield supplies the required 512 verified hard negatives. The source reservation heuristic and possible future policy changes are explicit in the design review.\n"
    path.write_text(document, encoding="utf-8")
    path = Path("docs/research_progress_dashboard.html")
    document = path.read_text(encoding="utf-8")
    snapshot = json.dumps(registry, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    document = re.sub(r'(<script id="snapshot" type="application/json">).*?(</script>)', lambda m: m[1]+snapshot+m[2], document, flags=re.S)
    document = document.replace("supplied only <strong>1 unverified candidate crop(s)</strong> at confidence 0.50.", "supplied <strong>one unverified crop at confidence 0.50 and four at 0.25</strong> in separately frozen exploratory versions. All four remain uncertain; two are adjacent frames.")
    path.write_text(document, encoding="utf-8")
    path = Path("docs/RESULTS_MASTER_TABLE.md")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"| {IDENTIFIER} | EXPLORATORY | Complete | {row['finding']} | [Source](../{YIELD}/progress.json) |\n")
    path = Path("docs/SCIENTIFIC_CLAIMS_STATUS.md")
    document = path.read_text(encoding="utf-8").replace("EXPLORATORY yield 1 crop(s) at confidence .50", "EXPLORATORY yield: 1 crop at .50; 4 at .25, all uncertain")
    path.write_text(document, encoding="utf-8")
    path = Path("docs/research_questions_and_evidence.md")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\nThe prospectively amended version-2 source diagnostic at confidence .25 yielded four uncertain crop proposals, including an adjacent-frame pair; the .50 version yielded one. Neither meets the 512-hard-crop requirement. Source-family reservation is a declared engineering heuristic; other families remain unaudited under a revised policy. [Current design review](NEGATIVE_MINING_DESIGN_REVIEW.md)\n")
    path = Path("docs/RESEARCH_PROGRESS_FLOWCHART.md")
    document = path.read_text(encoding="utf-8").replace('Frozen confidence 0.50 yield check · one candidate crop', 'Frozen yield checks · 0.50: one crop / 0.25: four crops').replace('Crop identity uncertain · independent review pending', 'Four crop identities uncertain · adjacent-frame pair · independent review pending').replace('Another allocation-separated discovery source', 'Prospective source development with full allocation-block separation')
    path.write_text(document, encoding="utf-8")
    print(json.dumps({"records": len(registry), "evidence_levels": Counter(r["evidence_level"] for r in registry)}, indent=2))


if __name__ == "__main__":
    main()
