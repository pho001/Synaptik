# Phase 24 Context: Normalization GPU Lowering

**Gathered:** 2026-05-01
**Status:** Ready for planning
**Source:** Auto context from Phase 24 roadmap, Phase 22 semantics, and Phase 23 reduction execution.

<domain>
## Phase Boundary

Phase 24 closes the `LAYER_NORM` and `RMS_NORM` fallback family for legal dense `FLOAT32` Metal/CUDA regions. The implementation should lower normalization to backend-owned GPU sub-DAGs made from already-supported reductions and elementwise primitives, adding only the missing primitive support needed for epsilon and CUDA execution.

This phase does not add public GPU tensors, does not widen native compute beyond legal `FLOAT32`, and does not hide unsupported dtype/layout/rank cases behind CPU replay.
</domain>

<decisions>
## Implementation Decisions

### D-01 Normalization Lowering Shape
- Lower `LAYER_NORM` and `RMS_NORM` as region-internal GPU DAG expansion, not as CPU `Operation.OpType.FUSED` and not as a single opaque native-only op.
- `LAYER_NORM` formula: `mean = mean(input, trailing normalized axes, keepDims=true)`, `centered = input - mean`, `variance = mean(centered * centered, trailing normalized axes, keepDims=true)`, `invStd = inv(sqrt(variance + epsilon))`, `output = centered * invStd * gamma + beta`.
- `RMS_NORM` formula: `meanSquares = mean(input * input, trailing normalized axes, keepDims=true)`, `invRms = inv(sqrt(meanSquares + epsilon))`, `output = input * invRms * gamma`.
- Repeated single-axis `MEAN` nodes are acceptable for multi-axis trailing normalized ranks; reduce from the last axis down to preserve shape-contract clarity with keep-dims outputs.

### D-02 ABI Delta
- Add only minimal shared DAG ABI required for normalization, expected to include scalar epsilon addition such as `ADD_SCALAR`.
- Do not add a public `Tensor` scalar-add API just to satisfy native lowering; the scalar primitive can be internal to accelerator DAG lowering.
- Reuse existing binary `MUL`, `SUB`, `ADD`, `SQRT`, `INV`, and Phase 23 `MEAN` primitives where possible.

### D-03 Backend Execution Scope
- Metal may use MPSGraph broadcasting and scalar constants behind the shared DAG primitive contract.
- CUDA must execute the lowered DAG natively for supported dense `FLOAT32` rank 1-4 shapes; this likely requires dense/broadcast binary kernels for normalization inputs and parameter tensors.
- Supported parameter broadcasting is suffix/tail broadcasting where `gamma` and `beta` shapes exactly match the normalized trailing dimensions.

### D-04 Rejection Scope
- Keep unsupported dtype, rank, non-dense direct compute layout, invalid normalizedRank, and unsupported broadcast cases as explicit stable rejections.
- BFLOAT16 continuation optimizations in CPU kernels are not part of native GPU compute in Phase 24.

### D-05 Coverage Gate
- `layer_norm_small` and `rms_norm_small` should move from normalization blockers to native/lowered GPU coverage for legal `FLOAT32` cases.
- Reports must still expose CPU materialization only at true graph output, CPU consumer, or gradient publication boundaries.
</decisions>

<canonical_refs>
## Canonical References

Downstream agents MUST read these before planning or implementing.

### Planning And Requirements
- `.planning/ROADMAP.md` - Phase 24 goal, dependencies, and success criteria.
- `.planning/REQUIREMENTS.md` - `GPUNORMX-01`, `GPUNORMX-02`, and `GPUNORMX-03`.
- `.planning/phases/22-coverage-truth-and-semantics-lock/22-02-SUMMARY.md` - normalization semantics contract summary.
- `.planning/phases/23-forward-reductions-native-execution/23-02-SUMMARY.md` - reduction execution support Phase 24 depends on.

### Operation Semantics
- `src/main/java/operations/normalization/layerNorm.java` - normalized rank and epsilon descriptor.
- `src/main/java/operations/normalization/rmsNorm.java` - normalized rank and epsilon descriptor.
- `src/main/java/tensor/ops/normalization/TensorNormalizationOps.java` - public forward/backward formulas and parameter shape validation.
- `src/main/java/backend/cpu/kernels/nn/CpuLayerNormKernel.java` - CPU forward parity oracle.
- `src/main/java/backend/cpu/kernels/nn/CpuRmsNormKernel.java` - CPU forward parity oracle.

### GPU Lowering And Execution
- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java` - shared DAG lowering entry point.
- `src/main/java/backend/accelerator/dag/AcceleratorDagNodeType.java` - native ABI op codes.
- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java` - planner coverage source of truth.
- `src/main/java/backend/metal/lowering/MetalPartitionSupport.java` - Metal planner rejection reasons.
- `src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java` - CUDA planner rejection reasons.
- `src/main/native/apple/synaptik_apple_mps_stub.m` - Metal native graph execution.
- `src/main/native/cuda/synaptik_cuda_graph_stub.cu` - CUDA native graph execution.
</canonical_refs>

<specifics>
## Specific Ideas

- Prefer a specialized normalization DAG builder inside `AcceleratorSubgraphLowerer` over graph-rewriting the public tensor graph.
- Test lowering with input shape `[2, 3]`, gamma/beta shape `[3]`, normalizedRank `1`, epsilon `1e-5`.
- Test multi-axis contract with input shape `[2, 4, 8, 1]`, gamma/beta shape `[8, 1]`, normalizedRank `2`.
- For CUDA binary broadcasting, start with suffix broadcast rules only: input rank may be lower than output rank, and each aligned dimension must be `1` or equal to the output dimension.
</specifics>

<deferred>
## Deferred Ideas

- Native BFLOAT16 normalization compute.
- Public device tensor API.
- Vendor library routing for normalization primitives.
- High-rank normalization beyond native rank 1-4.
- Backward normalization native GPU execution; Phase 24 focuses on forward coverage.
</deferred>

---

*Phase: 24-normalization-gpu-lowering*
*Context gathered: 2026-05-01 via auto planning context*
