# GPU Coverage Triage

Phase 14 turns benchmark coverage reports into an ordered backlog for v1.3 GPU partition expansion. The gate is coverage and materialization behavior, not raw timing.

## What The Triage Ranks

`GpuCoverageGapTriage` reads benchmark reports through `GpuCoverageSummary` and emits ranked `GpuCoverageGap` records. The categories keep CPU exits visible: rejected GPU candidates, CPU materialization, tensor-array fallback, CPU fallback, device handoff, short selected partitions, low GPU coverage, and non-device-owned storage residency.

The ranking is deterministic. It uses raw counts and fixed severity constants so later phases close measured exits before adding speculative operation coverage.

## Cross-Backend Router Evidence

`CrossBackendRouterEvidence` is the Phase 46 router audit layer. It does not change execution. It derives evidence from `ExecutionTrace` attributes and combines the route/path facts that previously had to be read from separate accelerator and coverage sections:

- common transport path counts: `BUFFER_BINDING`, `TENSOR_ARRAY`, `CPU_FALLBACK`, and unavailable paths,
- backend route counts: Metal `MPS_GRAPH` / `CUSTOM_KERNEL` and CUDA `cudaExecutionPath`,
- rejected route and capability reason counts, including CUDA `CAPABILITY_MISSING` rows,
- native copy strategies and output-buffer write statuses,
- selected partition length, lowered primitive count, fused subpattern count, layout materialization, dtype residency, CPU materialization, and device handoff counts.

`CrossBackendRouterRegressionGate` evaluates that evidence for representative workloads. The gate fails hidden tensor-array replay, unexpected CPU fallback, internal CPU materialization, lost native buffer binding, lost lowered-partition evidence, missing required visible rejection reasons, unexpected copy/write statuses, and unsupported route overclaims. A CUDA capability skip is valid fallback evidence only when it remains explicit; it is not support. MPSGraph remains `MPSGRAPH_RESULT_COPY` / `COPY_REQUIRED`; only the scoped custom Metal route may report `TRUE_OUTPUT_BUFFER_WRITE` / `PROVEN_TRUE_WRITE`.

## Hot Path Targets

`GpuHotPathCoverageTargets` defines the checked workload set used by Phase 14 and downstream v1.3 work:

| Workload | Purpose |
| --- | --- |
| `transformer_block_hot_path` | transformer partition length, normalization, epilogues, and device handoffs |
| `mlp_classifier_small` | linear/bias/activation and storage-sensitive epilogue coverage |
| `conv2d_resnet_3x3` | conv-style lowering and longer GPU partition coverage |
| `layer_norm_small` | normalization, reduction-style, and storage residency coverage |

`GpuCoverageTriageReport` combines these targets with top ranked gaps, category counts, requirement-family counts, and downstream phase targets.

## Requirement Families

| Family | Downstream Scope |
| --- | --- |
| `GPUDAG` | GPU partition internal lowered DAG contract |
| `GPUSTORAGE` | dtype, memory binding, and storage residency blockers |
| `GPUNORM` | normalization, reduction, softmax-ish, conv, and loss-adjacent lowering |
| `GPUFUSEX` | fused elementwise chains and epilogue subpartitions |
| `GPUMULTI` | multi-op GPU partition execution without intermediate CPU materialization |
| `GPUHARDEN` | coverage regression hardening and stayed-on-GPU gates |

## Evidence Commands

```bash
./gradlew classes
./gradlew test --tests GpuCoverageGapTriageTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageTriageReportTest
./gradlew test --tests BenchmarkSuiteSessionTest --tests GpuCoverageSummaryTest
./gradlew test --tests CrossBackendRouterEvidenceTest --tests BenchmarkSessionTest
```

## Artifact Hygiene

`profiles/platform/.../tuning/abc/* remained unstaged`.

Local tuning profile files are not canonical Phase 14 evidence. Canonical evidence is the checked source, tests, docs, and `.planning/phases/14-coverage-gap-triage-and-hot-path-targets/14-HOT-PATH-TARGETS.md`.
