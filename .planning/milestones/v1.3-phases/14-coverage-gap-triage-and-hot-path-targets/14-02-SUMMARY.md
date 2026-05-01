---
phase: 14-coverage-gap-triage-and-hot-path-targets
plan: "02"
status: complete
---

# 14-02 Summary - Hot Path Coverage Targets

## Hot path target registry

Added `GpuHotPathCoverageTarget` and `GpuHotPathCoverageTargets` as the checked Java registry for v1.3 hot-path coverage work.

| Workload | Kind | Owner Phase |
| --- | --- | --- |
| `transformer_block_hot_path` | transformer | 19 |
| `mlp_classifier_small` | mlp | 18 |
| `conv2d_resnet_3x3` | conv | 17 |
| `layer_norm_small` | normalization | 17 |

## Verification

| Command | Status |
| --- | --- |
| `./gradlew test --tests GpuHotPathCoverageTargetsTest --tests BenchmarkSuiteSessionTest` | passed |

## Artifact Hygiene

`profiles/platform/.../tuning/abc/* remained unstaged`.
