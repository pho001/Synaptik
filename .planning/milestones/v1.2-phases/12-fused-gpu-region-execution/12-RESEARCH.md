# Phase 12 Research: Fused GPU Region Execution

**Phase:** 12 - Fused GPU Region Execution
**Date:** 2026-04-30
**Status:** Complete

## Research Complete

Phase 12 should extend the existing accelerator lowering path. The repo already has the important pieces:

- `LoweringPipeline` invokes backend `RegionLowerer` implementations after partition, fusion, and memory planning.
- `MetalRegionLowerer` and `CudaRegionLowerer` classify selected accelerator regions as graph or fused-elementwise graph families.
- `AcceleratorSubgraphLowerer` already produces backend-neutral `AcceleratorDagSpec` values and has legacy `AcceleratorMatMulSpec` recognition for matmul/linear, bias, and post-ops.
- `MetalPartitionPlan` and `CudaGpuPartitionPlan` already carry the selected accelerator DAG into backend preparers.
- `PreparedMetalExecutable` and `PreparedCudaExecutable` already expose runtime buffer decisions and execution stats in run traces.
- `ExecutionState` already records device residency, device buffer bindings, and CPU materialization traces.
- `GpuLoweringCoverageMatrix` already classifies fused compound gaps as `DEFERRED_FUSED_REGION`.

The gap is not "no lowering". The gap is that compound GPU intent is not explicit enough. Today a longer region may lower to a DAG and execute as one backend graph, but tests and traces cannot reliably say "this selected region is the supported `LINEAR_BIAS_ACTIVATION` pattern" or fail when a supported compound pattern is shortened, materialized through Java arrays, or routed through CPU `Operation.OpType.FUSED` internals.

## Existing Pipeline Shape

Current flow:

1. Tensor graph compile builds `CompiledNode` snapshots.
2. Partition planning selects backend regions.
3. Region optimization may produce `ExecutionUnitKind.FUSED_ELEMENTWISE` for elementwise units.
4. `AcceleratorSubgraphLowerer` tries to lower the partition candidate into `AcceleratorDagSpec`.
5. Backend-specific partition plans are attached to selected partitions.
6. `LoweringPipeline` emits `LoweredRegion` and `LoweredExecutionUnit` descriptors.
7. Metal/CUDA preparers build a prepared accelerator executable at the anchor node.
8. Runtime execution uses native buffer binding, tensor-array bridge, or CPU fallback according to backend capability and config.

Phase 12 should add an explicit compound summary alongside the existing DAG:

`selected GPU region -> compound pattern summary -> accelerator DAG -> backend primitive/graph execution -> trace/residency evidence`

## Recommended Architecture

Add backend-neutral compound metadata under `backend.accelerator.lowering`:

- `GpuCompoundPatternType`
  - `NONE`
  - `LINEAR_BIAS_ACTIVATION`
  - `ELEMENTWISE_CHAIN`
  - `REDUCTION_ADJACENT`
  - `CPU_FUSED_UNSUPPORTED`
- `GpuCompoundRegionSummary`
  - backend
  - pattern type
  - supported flag
  - stable reason code
  - ordered node ids
  - external input ids
  - output node ids
  - DAG node type names
  - optional post-op names
  - detail string
- `GpuCompoundPatternDetector`
  - classifies a lowered `AcceleratorSubgraphSpec` plus `AcceleratorDagSpec` and optional `AcceleratorMatMulSpec`
  - rejects `Operation.OpType.FUSED` explicitly
  - produces stable reasons for unsupported compound candidates

Wire the summary into existing artifacts instead of adding a separate GPU fusion pipeline:

- extend `AcceleratorSubgraphLoweringResult` with a `GpuCompoundRegionSummary summary`
- keep compatibility constructors so existing tests can create lowering results without a summary
- carry summary through `MetalPartitionPlan` and `CudaGpuPartitionPlan`
- attach summary or a `GpuCompoundLoweringArtifact` to `LoweredExecutionUnit` when useful for plan tests
- expose summary from `PreparedAcceleratorExecutable` with a default method, implemented by Metal/CUDA executables
- add trace attributes in `PreparedExecution.buildStepMetadata`:
  - `gpuCompoundPattern`
  - `gpuCompoundSupported`
  - `gpuCompoundReason`
  - `gpuCompoundNodeCount`
  - `gpuCompoundDagNodeTypes`

## Target Pattern Details

### Linear + Bias + Activation

Supported minimal target:

- `LINEAR(input, weight, bias) -> RELU`
- or `MATMUL(input, weight) -> ADD(bias) -> RELU`

Required checks:

- backend is Metal or CUDA
- dtype is the existing accelerator-supported FLOAT32 path unless current code clearly supports more
- rank and shape constraints stay consistent with `AcceleratorMatMulSpec`
- bias is contiguous, no storage offset, and either vector or full output shape per existing matmul post-op rules
- activation is one of the already supported post-op types, with `RELU` as the required acceptance target
- the full selected region includes the compute, bias, and activation nodes
- no Java array round trip occurs between the fused operations for native buffer paths

### Elementwise Chain

Supported minimal target:

- `ADD -> RELU -> EXP`
- single published output
- all nodes have existing accelerator DAG node types
- all intermediate nodes are partition interiors or backend-owned, not independent CPU steps

Required checks:

- one accelerator prepared step covers the chain
- lowering family is `METAL_FUSED_ELEMENTWISE_GRAPH` or `CUDA_FUSED_ELEMENTWISE_GRAPH`
- trace names the pattern as `ELEMENTWISE_CHAIN`
- runtime CPU materialization trace has no `CPU_CONSUMER` entry between chain nodes for native buffer paths
- parity test compares output to CPU baseline

### Reduction-Adjacent

Reduction-adjacent candidates are third priority. For Phase 12, it is acceptable to leave `LAYER_NORM`, `RMS_NORM`, `SUM`, `MEAN`, `REDUCE_MIN`, and `REDUCE_MAX` unsupported if the coverage matrix and planner diagnostics say so with stable reasons. A narrow supported subset may be implemented only if parity and backend capability tests are added in the same plan.

## Threats And Risks

- **Hidden fallback:** A supported compound pattern can silently execute CPU fallback and still produce correct output. Mitigation: required-mode tests must fail before fallback, and AUTO-mode traces must expose fallback reason.
- **Region shortening:** Backend selection may select only a subpart of a supported pattern. Mitigation: tests must assert selected decisions contain all node ids for supported target graphs.
- **CPU FUSED coupling:** GPU path might consume `Operation.OpType.FUSED` or CPU ASM internals. Mitigation: GPU legality and lowering must reject `FUSED` with a stable GPU compound reason.
- **Backend parity overclaiming:** Metal MPSGraph may support more than CUDA. Mitigation: summary and matrix must be backend-specific and may show narrower CUDA coverage.
- **Trace drift:** Summary may exist in lowering but not show up in traces. Mitigation: run trace tests assert exact trace attribute keys.
- **CPU hot-path regression:** Changes to shared fusion concepts might break CPU fused ASM/vector execution. Mitigation: run existing CPU fused and prepared execution tests.

## Validation Architecture

Use existing Gradle/JUnit infrastructure.

Primary commands:

- `./gradlew classes`
- `./gradlew test --tests backend.accelerator.lowering.*`
- `./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest`
- `./gradlew test --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest`

Optional native commands:

- `./gradlew metalTest`
- `./gradlew buildCudaGraphShim cudaTest`

Validation must prove:

- `GPUFUSE-01`: Metal and CUDA select and lower a full linear+bias+activation region.
- `GPUFUSE-02`: Metal and CUDA execute a representative elementwise chain as one compound GPU region with device-owned intermediates where native buffer capability allows.
- `GPUFUSE-03`: GPU compound summaries and execution paths do not depend on `backend.cpu.fused` internals and CPU fused tests still pass.
- `GPUFUSE-04`: reduction-adjacent candidates are either implemented with parity tests or rejected with stable reason codes and matrix/docs rows.

