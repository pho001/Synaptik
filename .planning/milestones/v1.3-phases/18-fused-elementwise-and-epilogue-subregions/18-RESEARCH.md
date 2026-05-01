# Phase 18 Research: Fused Elementwise And Epilogue Subregions

**Phase:** 18 - Fused Elementwise And Epilogue Subregions
**Date:** 2026-05-01
**Status:** Complete

## RESEARCH COMPLETE

## Goal

Plan Phase 18 so GPU fusion becomes a region-internal lowering and execution concept for supported elementwise chains and matmul/linear epilogues. The work must deepen the Phase 12 compound-region implementation without reusing CPU `Operation.OpType.FUSED`, generated ASM, or CPU vector dispatch internals.

## Source Of Truth

- `.planning/ROADMAP.md` Phase 18 requires `GPUFUSEX-01`, `GPUFUSEX-02`, and `GPUFUSEX-03`.
- `.planning/phases/14-coverage-gap-triage-and-hot-path-targets/14-HOT-PATH-TARGETS.md` names `mlp_classifier_small` as Phase 18's primary target and includes `transformer_block_hot_path` as a secondary fusion consumer.
- `.planning/phases/15-gpu-region-internal-lowered-dag-contract/15-VERIFICATION.md` verifies `GpuLoweredRegionManifest`, original-op metadata, lowered primitives, fused summary metadata, dtype/layout assumptions, and trace/report rendering.
- `.planning/phases/17-normalization-reduction-and-loss-adjacent-lowering/17-VERIFICATION.md` verifies the shared Metal/CUDA coverage matrix, backend legality detail, softmax-ish support, visible fallback, and CPU parity rules.
- `.planning/milestones/v1.2-phases/12-fused-gpu-region-execution/12-VERIFICATION.md` verifies the first compound summaries: `LINEAR_BIAS_ACTIVATION`, `ELEMENTWISE_CHAIN`, `REDUCTION_ADJACENT`, and explicit CPU `FUSED` rejection.

## Existing Implementation

The current code already has the right base abstractions:

- `src/main/java/backend/accelerator/lowering/GpuCompoundPatternDetector.java` classifies DAGs as `LINEAR_BIAS_ACTIVATION`, `ELEMENTWISE_CHAIN`, `REDUCTION_ADJACENT`, or `CPU_FUSED_UNSUPPORTED`.
- `src/main/java/backend/accelerator/lowering/GpuCompoundRegionSummary.java` carries a single compound summary with backend, pattern type, support status, ordered node ids, DAG node types, post ops, reason, and detail.
- `src/main/java/backend/accelerator/lowering/GpuLoweredRegionManifest.java` includes a `fusedSummary` field plus original ops and lowered primitive manifests.
- `src/main/java/graph/optimizer/region/GenericGpuRegionOptimizationPolicy.java` fuses a whole accelerator partition only when every node is fusable; otherwise it emits single-op units.
- `src/main/java/backend/metal/lowering/MetalRegionLowerer.java` and `src/main/java/backend/cuda/lowering/CudaRegionLowerer.java` choose fused elementwise lowering families only when a selected region has one `FUSED_ELEMENTWISE` execution unit.
- `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` and `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java` expose `compoundSummary()`, buffer decisions, and fallback behavior through the shared executable contract.
- `src/main/java/graph/execution/PreparedExecution.java` already renders `gpuCompound*` attributes when an accelerator executable reports a compound summary.

## Planning Implications

Phase 18 should not create a public GPU tensor API and should not create a GPU equivalent of `Operation.OpType.FUSED`. The correct shape is:

1. Keep graph operations as normal semantic operations.
2. Let partitioning select a device-owned region.
3. Let region optimization identify safe fused subpatterns inside that region.
4. Let accelerator lowering record subpattern metadata and reject unsupported subpatterns with stable reason codes.
5. Let Metal/CUDA prepared executables execute supported subpatterns through backend-owned DAG/buffer paths or expose visible fallback in REQUIRED/AUTO mode.

## Recommended Plan Shape

Four waves are enough and align with existing phase patterns:

1. **Shared fusion subpattern metadata contract**: introduce list-capable fusion subpattern metadata beside the legacy single `GpuCompoundRegionSummary`, stable reason codes, tests, and docs.
2. **Elementwise chain subregions**: teach GPU region optimization/lowering to identify supported elementwise subchains inside a larger selected region without shortening the selected region or materializing interiors.
3. **Matmul/linear epilogue subregions**: detect supported `MATMUL` or `LINEAR` plus bias plus activation as an epilogue subpattern when dtype/layout/backend gates allow it.
4. **Trace/report/docs and CPU isolation closure**: render fused subpattern count/span/primitive count/rejection evidence and prove CPU fused execution remains independent.

## Validation Architecture

Use existing Gradle/JUnit infrastructure with focused tests after each wave:

- Model and detector tests: `backend.accelerator.lowering.GpuCompoundPatternDetectorTest`, `GpuLoweredRegionManifestTest`, and `AcceleratorSubgraphLowererTest`.
- Region optimizer tests: `graph.optimizer.region.DefaultRegionOptimizerTest`.
- Backend lowerer tests: `backend.metal.lowering.MetalRegionLowererTest` and `backend.cuda.lowering.CudaRegionLowererTest`.
- Prepared execution and trace tests: `PreparedExecutionBuildTest`, `CompiledGraphTraceTest`, `GpuCoverageSummaryTest`, and `BenchmarkSessionTest`.
- CPU guardrails: `OptimizerFuseTest`, `backend.cpu.fused.plan.LoweredFusedOperationBuilderTest`, and `SourceTreeHygieneTest`.

Full focused command:

```bash
./gradlew classes
./gradlew test --tests backend.accelerator.lowering.GpuCompoundPatternDetectorTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest --tests graph.optimizer.region.DefaultRegionOptimizerTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests OptimizerFuseTest --tests backend.cpu.fused.plan.LoweredFusedOperationBuilderTest --tests SourceTreeHygieneTest
```

Native Metal/CUDA runs remain capability-gated. Portable Java gates must still prove metadata, fallback visibility, CPU parity, and no hidden CPU materialization.

## Landmines

- Do not import `backend.cpu.fused.*` into `backend.accelerator`, `backend.metal`, or `backend.cuda`.
- Do not make `Operation.OpType.FUSED` a valid GPU lowering input.
- Do not mark an epilogue or elementwise chain supported unless dtype, layout, backend, and buffer-binding legality are checked.
- Do not let a fused subpattern shorten the selected region without recording rejection or candidate-span evidence.
- Do not count tensor-array bridge fallback as native buffer GPU coverage.
- Do not stage local `profiles/platform/.../tuning/abc/*` artifacts.

