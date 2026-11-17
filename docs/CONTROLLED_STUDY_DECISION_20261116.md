# Approved controlled-study decisions — 16 November 2026

The researcher approved these decision and analysis rules. This supplement fixes the study direction without adding results or starting training. The paper skeleton and its verified historical/descriptive findings remain unchanged.

The machine-readable analysis plan is [analysis_plan_yolo26_v1.json](../configs/paper/analysis_plan_yolo26_v1.json). Its companion SHA-256 record binds this document, the plan, and the reviewed protocol/configuration/runner inputs. It is an **analysis-plan freeze**, not a claim that the complete executable training protocol is ready. The older `protocol_yolo26.json` remains in its existing preflight status; the historical YOLOv8 control lock is not repurposed.

## Fixed experiment matrix and order

| Arm | Configuration | Seeds |
| --- | --- | --- |
| E0 | YOLO26s baseline | 17, 42, 123 |
| E1 | YOLO26s-P2 | 17, 42, 123 |
| E2 | P2 plus 512 source-matched random verified background crops | 17, 42, 123 |
| E3 | P2 plus 512 source-matched model-guided verified hard background crops | 17, 42, 123 |

Finish executable readiness first, then E0 Seed 17 and E1 Seed 17. Review architecture, shared public initialization, dataloader, logs, evaluator and no-Test compliance. This is a technical-compliance gate. Complete E0/E1 Seeds 42 and 123 regardless of the direction or magnitude of Seed 17's valid result. Then prepare and independently verify the matched negative bundles and complete all three E2 and E3 seeds. Valid negative findings do not justify dropping seeds or replacing them. Technical failures require preserved logs and a documented repair, not silently omitted results.

Fusion is outside this study's E2/E3 definitions. The completed local AdamW run remains a **completed descriptive local baseline**, not Controlled E0.

## Comparisons and reporting

The architecture comparison is **E1 minus E0**, paired by training seed, for the whole declared P2 variant. The primary negative-selection comparison is **E3 minus E2**. Positive images/labels, negative counts, crop dimensions, source-image frequencies, augmentation, initialization, schedule and actual optimizer-update budget must match between E2 and E3. Same-source matching controls source composition; the contrast estimates the declared selection procedure, not an unrestricted claim about every possible hard-negative method.

**E2 minus E1** is the secondary overall random-background augmentation comparison. **E3 minus E1** is the secondary overall hard-negative augmentation comparison. Both can include additional-data/update effects relative to E1; neither isolates hardness. E3 minus E1 alone does not establish that background negatives in general are useful.

Report every seed, arithmetic mean and **sample SD** with denominator `n-1` (`n=3`, `ddof=1`). For each contrast report the three same-seed signed differences and their mean/sample SD. Label percentage-point differences separately from relative percentages and count differences. Report unavailable or failed runs explicitly. Seed SD is training variability; it is not a confidence interval over videos and does not establish statistical significance.

Report AP50, AP50:95, precision, recall, Tiny/Small recall and TP/FP/FN using one pinned evaluator and resolved operating settings. Latency and background-only performance require their own declared measurement population, denominator and protocol; do not fill missing measurements from training throughput or positive-image false positives.

## Development and final confirmation

All model, architecture and operating-point decisions use **Validation only**. Clean_v2 Test has already been evaluated in the descriptive matrix. It must not drive new decisions or be relabeled as untouched. No additional Test inference is part of this supplement.

The [external protocol template](../configs/paper/external_validation_protocol.template.json) is deliberately **incomplete and not frozen**. Before final inference, specify dataset/version/source, verified RGB/visible channel, exact video/sample manifest, inclusion/exclusion and sampling rules, exposure/overlap audit, model-selection rule, checkpoint hashes, confidence/head/NMS settings, evaluator and metrics, per-video and aggregate definitions, and the confidence-interval method and statistical unit. Preserve recording dependence; individual correlated frames are not automatically independent sampling units.

An already inspected sampled RGPT evaluation stays preliminary. Do not silently substitute an assumed RGBT benchmark identity, and do not relabel an exposed sample as independent untouched confirmation. Once the final external protocol is fully specified, bind its inputs and hash it before inference. Changes after outcome exposure require a new version and explicit exposure history, not retrospective preregistration.

## Executable-readiness findings

The read-only check found these concrete gaps; no code or saved results were changed to conceal them:

- The current YOLO26 runner supports E0/Seed 17 only. E1 exists as a YAML but lacks an equivalent run path; the remaining seeds and E2/E3 are not wired in.
- The loader accepts only `preflight_only`, while the runner also mentions an authorized-training status. The status contract must be made consistent.
- Learning-rate schedule, warmup, augmentation, stopping, precision and accumulation are not all bound in a complete resolved, hash-checked run configuration. The E0/E1 YAMLs need parity checking against that configuration.
- The current evaluator uses `last.pt`. The exact prospective checkpoint/model-selection rule must be explicit before runs; “Validation only” does not silently authorize switching to `best.pt`.
- The actual inference head/NMS path is not bound by consuming the declared `nms_free` setting. Pin and record effective settings, including operating confidence versus the AP prediction floor.
- This particular YOLO26 runner lacks the before/after frozen snapshot and cache-write guards present in other project workflows. Its `modified:false` output is not proof of verification. These guards must be integrated before relying on it for frozen-release runs.
- Matched negative bundles, crop/source manifests and measured optimizer-update parity still need YOLO26 integration and verification.
- Bind the environment and implementation identities and audit the existing E0 output directory before reuse. Do not overwrite an earlier attempt or assume completion from a directory name.

The synthetic preflight pass remains useful evidence of construction and one update, not proof of full training/evaluation compliance. A later executable freeze must resolve these gaps and preserve existing protocol versions. This supplement creates no E0–E3 scores, external benchmark results, training authorization state changes, or queued training jobs.
