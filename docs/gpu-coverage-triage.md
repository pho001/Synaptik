# GPU Coverage Triage

Phase 14 turns benchmark coverage reports into an ordered backlog for v1.3 GPU region expansion. The gate is coverage and materialization behavior, not raw timing.

## What The Triage Ranks

`GpuCoverageGapTriage` reads benchmark reports through `GpuCoverageSummary` and emits ranked `GpuCoverageGap` records. The categories keep CPU exits visible: rejected GPU candidates, CPU materialization, tensor-array fallback, CPU fallback, device handoff, short selected regions, low GPU coverage, and non-device-owned storage residency.

The ranking is deterministic. It uses raw counts and fixed severity constants so later phases close measured exits before adding speculative operation coverage.

## Hot Path Targets

`GpuHotPathCoverageTargets` defines the checked workload set used by Phase 14 and downstream v1.3 work:

| Workload | Purpose |
| --- | --- |
| `transformer_block_hot_path` | transformer region length, normalization, epilogues, and device handoffs |
| `mlp_classifier_small` | linear/bias/activation and storage-sensitive epilogue coverage |
| `conv2d_resnet_3x3` | conv-style lowering and longer GPU region coverage |
| `layer_norm_small` | normalization, reduction-style, and storage residency coverage |

`GpuCoverageTriageReport` combines these targets with top ranked gaps, category counts, requirement-family counts, and downstream phase targets.

## Requirement Families

| Family | Downstream Scope |
| --- | --- |
| `GPUDAG` | GPU region internal lowered DAG contract |
| `GPUSTORAGE` | dtype, memory binding, and storage residency blockers |
| `GPUNORM` | normalization, reduction, softmax-ish, conv, and loss-adjacent lowering |
| `GPUFUSEX` | fused elementwise chains and epilogue subregions |
| `GPUMULTI` | multi-op GPU region execution without intermediate CPU materialization |
| `GPUHARDEN` | coverage regression hardening and stayed-on-GPU gates |

## Evidence Commands

```bash
./gradlew classes
./gradlew test --tests GpuCoverageGapTriageTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageTriageReportTest
./gradlew test --tests BenchmarkSuiteSessionTest --tests GpuCoverageSummaryTest
```

## Artifact Hygiene

`profiles/platform/.../tuning/abc/* remained unstaged`.

Local tuning profile files are not canonical Phase 14 evidence. Canonical evidence is the checked source, tests, docs, and `.planning/phases/14-coverage-gap-triage-and-hot-path-targets/14-HOT-PATH-TARGETS.md`.
