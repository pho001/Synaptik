# Phase 14 Hot Path Targets

This file is the source-of-truth target list for Phases 15-20 in v1.3 Coverage-Driven GPU Region Expansion.

## Target Workloads

| Workload | Kind | Requirement Families | Primary Owner |
| --- | --- | --- | --- |
| `transformer_block_hot_path` | transformer | `GPUDAG`, `GPUNORM`, `GPUFUSEX`, `GPUMULTI`, `GPUHARDEN` | Phase 19 |
| `mlp_classifier_small` | mlp | `GPUSTORAGE`, `GPUFUSEX`, `GPUMULTI`, `GPUHARDEN` | Phase 18 |
| `conv2d_resnet_3x3` | conv | `GPUNORM`, `GPUMULTI`, `GPUHARDEN` | Phase 17 |
| `layer_norm_small` | normalization | `GPUSTORAGE`, `GPUNORM`, `GPUMULTI`, `GPUHARDEN` | Phase 17 |

## Top Coverage Gap Categories

| Category | Triage Meaning |
| --- | --- |
| `CPU_MATERIALIZATION` | Device-owned values had to become CPU-readable. |
| `TENSOR_ARRAY_FALLBACK` | Accelerator execution used tensor-array bridge instead of native buffer coverage. |
| `CPU_FALLBACK` | Accelerator-selected flow executed via explicit CPU fallback. |
| `DEVICE_HANDOFF` | Execution crossed CPU/GPU or GPU/CPU boundaries. |
| `REJECTED_CANDIDATE` | Planner saw an accelerator-compatible candidate but rejected it with a reason. |
| `LOW_REGION_LENGTH` | Selected GPU region was too short to represent the desired hot path. |
| `LOW_GPU_COVERAGE` | Run trace stayed below the Phase 14 coverage threshold. |
| `STORAGE_RESIDENCY` | Storage residency was not `DEVICE_OWNED`. |

## Downstream Phase Targets

| Phase | Family | Scope |
| --- | --- | --- |
| Phase 15 | `GPUDAG` | GPU region internal lowered DAG contract and debug metadata. |
| Phase 16 | `GPUSTORAGE` | dtype, memory binding, and storage residency blockers. |
| Phase 17 | `GPUNORM` | normalization, reduction, softmax-ish, conv, and loss-adjacent lowering. |
| Phase 18 | `GPUFUSEX` | fused elementwise chains and epilogue subregions. |
| Phase 19 | `GPUMULTI` | multi-op GPU region execution without intermediate CPU materialization. |
| Phase 20 | `GPUHARDEN` | coverage regression hardening and hot-path stayed-on-GPU gates. |

## Evidence Commands

```bash
./gradlew test --tests GpuCoverageTriageReportTest --tests GpuCoverageGapTriageTest --tests GpuHotPathCoverageTargetsTest
```

## Artifact Hygiene

`profiles/platform/.../tuning/abc/* remained unstaged`.
