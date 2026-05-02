# Phase 28 Verification: Coverage Regression Closure

## Verdict

**PASS.**

Phase 28 completes `GPUCLOSE-01`, `GPUCLOSE-02`, and `GPUCLOSE-03` by hardening target policies against actual v1.4 execution truth, extending regression gates for supported families, and rendering final report evidence without relying on raw timing or local benchmark artifacts.

## Requirement Evidence

| Requirement | Evidence | Status |
|---|---|---|
| `GPUCLOSE-01` | `GpuHotPathCoverageTargets` now hard-gates native-executable targets and keeps visible-blocker expectations for unsupported/capability-gated targets. Suite reports render region length, lowered primitive count, backend path counters, CPU exits, and handoffs. | Passed |
| `GPUCLOSE-02` | `GpuCoverageRegressionGateTest` fails supported targets on hidden tensor-array replay, CPU fallback, lost native buffer binding, CPU materialization, shortened region evidence, and lost lowered primitive evidence. Missing target summaries and missing visible reasons name workload/backend. | Passed |
| `GPUCLOSE-03` | `GpuCoverageBaseline.v14Closure(...)`, `coverageDeltaVsBaseline`, text/JSON renderer tests, docs, and profile artifact checks provide deterministic before/after coverage evidence while keeping local profile outputs unstaged. | Passed |

## Source And Test Evidence

| Area | Evidence |
|---|---|
| Target policy truth | `src/main/java/tuning/benchmark/report/GpuHotPathCoverageTargets.java`, `src/test/java/GpuHotPathCoverageTargetsTest.java` |
| Regression gate behavior | `src/main/java/tuning/benchmark/report/GpuCoverageRegressionGate.java`, `src/test/java/GpuCoverageRegressionGateTest.java` |
| Coverage delta baseline | `src/main/java/tuning/benchmark/report/GpuCoverageBaseline.java`, `src/main/java/tuning/benchmark/report/GpuCoverageComparison.java`, `src/test/java/BenchmarkSessionTest.java` |
| Text/JSON report evidence | `src/main/java/tuning/benchmark/report/TextBenchmarkSuiteReportRenderer.java`, `src/main/java/tuning/benchmark/report/JsonBenchmarkSuiteReportRenderer.java`, `src/test/java/BenchmarkSuiteSessionTest.java` |
| Report/docs contract | `docs/gpu-lowering-coverage.md`, `docs/compute-flow.md`, `docs/testing.md`, `docs/calibration-autotune.md` |

## Commands

| Command | Result |
|---|---|
| `./gradlew test --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageRegressionGateTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests StandardWorkloadsTest` | Passed |
| `./gradlew metalTest` | Passed |
| `git diff --check` | Passed |
| `command -v nvcc` | Not available locally; CUDA native gate not run |
| `git status --short profiles/platform` | Local profile artifacts remain dirty but are not staged Phase 28 evidence |

## Supported Vs Visible-Blocker Targets

| Target | Metal | CUDA |
|---|---|---|
| `reduction_chain_small` | Hard native/buffer gate | Hard native/buffer gate |
| `layer_norm_small` | Hard native/buffer gate | Hard native/buffer gate |
| `rms_norm_small` | Hard native/buffer gate | Hard native/buffer gate |
| `transformer_block_hot_path` | Hard native/buffer gate for verified forward SDPA path | Visible `CAPABILITY_MISSING` blocker |
| `mlp_classifier_small` | Hard fused/lowered hot-path gate | Hard fused/lowered hot-path gate |
| `conv2d_resnet_3x3` | Visible conv capability/layout blocker | Visible conv capability/layout blocker |
| `max_pool2d_small` | Visible pool capability blocker | Visible pool capability blocker |
| `cross_entropy_small` | Visible loss/index blocker | Visible loss/index blocker |
| `bool_compare_where_small` | Visible BOOL-producing compute blocker | Visible BOOL-producing compute blocker |

## Residual Risks

- Native conv/pool execution remains unsupported and explicitly capability-gated.
- Native BOOL-producing compare/logical/reduction output execution remains unsupported; external BOOL predicate residency for `WHERE` is separate.
- CUDA native execution remains locally toolchain-gated because `nvcc` is unavailable.
- Local profile artifacts remain dirty in the working tree and are intentionally not Phase 28 proof.
