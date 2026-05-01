# Phase 19: Multi-Op GPU Region Execution - Research

**Date:** 2026-05-01
**Status:** Complete

## Research Question

What needs to be known to plan Phase 19 so Metal and CUDA can execute longer GPU regions with multiple lowered operations without hidden CPU materialization?

## Findings

### Current Architecture

- `AcceleratorSubgraphLowerer` already lowers selected accelerator candidates into an `AcceleratorDagSpec` and a `GpuLoweredRegionManifest`.
- `GpuLoweredRegionManifest` already records original operation ids, lowered primitive ids/types, dtype/layout assumptions, selected region length, fused subpatterns, rejections, and backend extensions.
- `MetalRegionLowerer` and `CudaRegionLowerer` currently emit one `LoweredExecutionUnit` for a selected backend region, with special fused families only when the region is a single whole fused elementwise unit.
- `MetalNodePreparer` and `CudaGpuNodePreparer` prepare one accelerator executable at the partition anchor and mark interior nodes as `PartitionExecutionRole.INTERIOR`.
- `PreparedMetalExecutable` and `PreparedCudaExecutable` already expose `gpuLoweredRegionManifest()`, evaluate native buffer binding, fall back explicitly, and mark output device buffer bindings as `DEVICE_OWNED`.
- `AcceleratorPreparedInputResolver` can call `context.requireCpuReadable(..., ACCELERATOR_PREPARED_INPUT)` when CPU fallback layout-prepared inputs are applied. This is a key risk for hidden internal CPU materialization.
- Tensor-array bridge paths call `requireCpuReadable(..., CPU_CONSUMER)` for external inputs and should remain visible fallback/non-native coverage, not native buffer coverage.

### Planning Implications

- Phase 19 should not invent a public GPU tensor API. The required residency boundary already exists in `ExecutionState`, `ExecutionContext`, and device buffer binding records.
- The first implementation target is not broad new primitive math. It is proving that already-supported primitives can coexist inside one selected region and run without CPU exits between internal steps.
- Unsupported normalization, reduction, conv, and loss-adjacent pieces should preserve Phase 17 rejection detail unless a narrow supported implementation is added with parity tests.
- The highest-value tests should combine prepare/build assertions with runtime traces: selected region length, lowered primitive count, fused subpattern count, no internal `ACCELERATOR_PREPARED_INPUT`, `BUFFER_BINDING` path evidence, and explicit fallback when unsupported.
- Phase 20 will harden gates. Phase 19 should add stable fields and focused tests, but avoid turning this phase into the full regression gate milestone.

## Candidate Implementation Tracks

1. **Multi-op region contract and legality tests**
   - Extend lowerer and region-lowerer tests to prove a selected partition remains one backend-owned region with multiple lowered primitives.
   - Assert unsupported internal primitives split/reject before execution with stable manifest reasons.

2. **Internal device handoff and materialization guard**
   - Add trace/test assertions that supported region interiors do not emit `ACCELERATOR_PREPARED_INPUT` or `CPU_CONSUMER` materialization.
   - If needed, adjust accelerator input preparation so CPU fallback prepared-input logic is not applied to native-buffer-supported internals.

3. **Metal/CUDA prepared executable parity**
   - Keep shared planning/lowering contract backend-neutral.
   - Let Metal/CUDA execute through backend-specific prepared executables and native bridge paths.
   - Preserve required-mode failure when native buffer execution is unavailable.

4. **Coverage evidence**
   - Extend trace/report summaries so Phase 19 can prove multi-op region execution with region length, lowered primitive count, fused subpattern count, materialization counts/reasons, device handoff counts, and runtime path.
   - Use `transformer_block_hot_path` and `mlp_classifier_small` first; report visible blockers for `conv2d_resnet_3x3` and `layer_norm_small` if not implemented.

## Validation Architecture

### Automated Tests

- `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest`
- `./gradlew test --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest`
- `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests graph.execution.DeviceLayoutViewPropagationTest`
- `./gradlew test --tests SourceTreeHygieneTest`
- `./gradlew classes`

### Evidence Requirements

- At least one test must assert `selectedRegionLength` greater than one for a supported GPU region.
- At least one test must assert lowered primitive count and fused subpattern count for a multi-op region.
- At least one runtime trace test must assert no internal `ACCELERATOR_PREPARED_INPUT` materialization for a supported native buffer path.
- Coverage/report tests must distinguish `BUFFER_BINDING`, `TENSOR_ARRAY`, and `CPU_FALLBACK`.
- Docs and summaries must repeat that tensor-array bridge execution does not count as native buffer GPU coverage.

## Risks

- Native bridge limitations may prevent real execution for some multi-op shapes. Mitigation: keep native checks capability-gated and use portable fake/stub bridge tests for Java-side contracts.
- CPU fallback prepared-input logic may be necessary for fallback parity. Mitigation: preserve fallback behavior but keep it explicit and prevent it from being counted as supported internal GPU handoff.
- Over-expanding operation coverage could destabilize Phase 19. Mitigation: consume Phase 17 rejections and defer broad vendor-library routing to `GPULIB-*`.

## Research Complete

Phase 19 should be planned as an integration phase: prove longer backend-owned regions over existing supported primitives, protect `ExecutionState` residency, expose coverage evidence, and keep unsupported internals visibly rejected.
