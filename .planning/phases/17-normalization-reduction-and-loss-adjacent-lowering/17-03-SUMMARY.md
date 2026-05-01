---
phase: 17-normalization-reduction-and-loss-adjacent-lowering
plan: "03"
status: complete
requirements-completed: [GPUNORM-02, GPUNORM-03]
completed: 2026-05-01
---

# Phase 17 Plan 03: Trace And Parity Summary

Softmax-ish support and loss-adjacent fallback are now covered by parity checks, prepare trace rendering, coverage summaries, and benchmark report assertions.

## Softmax and loss-adjacent evidence

- Added CPU parity tests for `LOG_SOFTMAX`, `CROSS_ENTROPY_LOSS_INDICES`, and `LAYER_NORM` accelerator-configured flows.
- Verified selected `LOG_SOFTMAX` GPU manifests still render original `LOG_SOFTMAX` and lowered `SOFTMAX` plus `LOG` primitives.
- Verified index-target cross entropy exposes `UNSUPPORTED_DTYPE`, `family=LOSS_ADJACENT`, and `target=transformer_block_hot_path`.
- Verified layer norm fallback exposes `REDUCTION_ADJACENT`, `family=NORMALIZATION`, and `target=layer_norm_small`.
- Added coverage and benchmark report fixtures that keep Phase 17 selected and rejected evidence visible in text and JSON reports.

CPU parity remained the correctness oracle.

loss-adjacent fallback remained visible.

## Verification

| Command | Result |
|---------|--------|
| `./gradlew test --tests PreparedExecutionBuildTest` | Passed |
| `./gradlew test --tests CompiledGraphTraceTest` | Passed |
| `./gradlew test --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest` | Passed |
| `./gradlew test --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest` | Passed |
| `rg -n "phaseSeventeenLogSoftmaxResidualFlowMatchesCpuAndStaysSupported|phaseSeventeenCrossEntropyIndexFallbackMatchesCpuAndReportsUnsupportedDType|phaseSeventeenLayerNormFallbackMatchesCpuAndReportsReductionAdjacent|target=transformer_block_hot_path|target=layer_norm_small" src/test/java/PreparedExecutionBuildTest.java` | Passed |
| `rg -n "prepareTraceRendersPhaseSeventeenNormAndLossEvidence|family=LOSS_ADJACENT|family=NORMALIZATION|target=layer_norm_small|target=transformer_block_hot_path" src/test/java/CompiledGraphTraceTest.java` | Passed |
| `rg -n "coverageSummaryCountsPhaseSeventeenNormAndLossReasons|benchmarkReportsRenderPhaseSeventeenNormAndLossEvidence|family=NORMALIZATION|family=LOSS_ADJACENT|target=layer_norm_small|target=transformer_block_hot_path" src/test/java/GpuCoverageSummaryTest.java src/test/java/BenchmarkSessionTest.java` | Passed |

## Requirement Coverage

- `GPUNORM-02`: Trace, coverage, and benchmark report assertions now tie softmax support and norm/loss rejection evidence to Phase 17 hot-path blockers.
- `GPUNORM-03`: CPU parity tests prove numerically sensitive softmax, loss, and normalization flows remain correct while unsupported accelerator coverage stays explicit.

## Commits

| Commit | Description |
|--------|-------------|
| `cc64608` | Added softmax, loss, and layer norm parity evidence tests. |
| `52a784b` | Added Phase 17 prepare trace evidence rendering. |
| `beaa901` | Added coverage and benchmark report Phase 17 evidence. |

## Deviations from Plan

None - plan executed exactly as written.

## Hygiene

profiles/platform/.../tuning/abc/* remained unstaged

## Self-Check: PASSED
