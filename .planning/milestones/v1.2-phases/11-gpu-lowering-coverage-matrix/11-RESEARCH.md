---
phase: 11-gpu-lowering-coverage-matrix
status: complete
researched: 2026-04-30
requirements: [GPULOWER-01, GPULOWER-02, GPULOWER-03]
---

# Phase 11 Research: GPU Lowering Coverage Matrix

## Research Goal

Answer what needs to be known to plan Phase 11 well: how Synaptik can broaden Metal and CUDA operation lowering for common NN/tensor patterns while keeping unsupported operations, dtypes, layouts, and native capability gaps explicit.

## Phase Scope

Phase 11 owns the checked-in Metal/CUDA lowering coverage contract and the next set of planner/lowering support improvements. It should make supported regions longer, especially around `matmul/linear -> layout/view -> elementwise/softmax-ish`, while preserving CPU fallback visibility.

Out of scope:

- Fused GPU compound execution internals such as `linear + bias + activation` as one backend-native fused kernel. Phase 12 owns that.
- Coverage ratio benchmark gates and representative workload regression thresholds. Phase 13 owns that.
- Public device tensor APIs. Public `Tensor` remains logical.
- Universal support for every operation, dtype, rank, or layout.

## Current Code Findings

### Existing accelerator DAG coverage

- `backend.accelerator.lowering.AcceleratorSubgraphLowerer` is the shared source that turns selected Metal/CUDA partition candidates into `AcceleratorDagSpec`.
- `AcceleratorDagNodeType` currently represents:
  - matmul/linear: `MATMUL`, `LINEAR`
  - elementwise: `ADD`, `SUB`, `MUL`, `DIV`, `RELU`, `TANH`, `SIGMOID`, `ABS`, `EXP`, `LOG`, `NEG`, `SQRT`, `INV`, `CLAMP_MIN`, `CLAMP_MAX`, `MUL_SCALAR`, `WHERE`
  - layout: `RESHAPE`, `CONTIGUOUS`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`
  - softmax and backward-adjacent ops: `SOFTMAX`, `SOFTMAX_GRAD`, `LOG_SOFTMAX_GRAD`, `REDUCE_MIN_GRAD`, `REDUCE_MAX_GRAD`, `MIN_GRAD`, `MAX_GRAD`
  - attention specializations: `SDPA`, `SDPA_BACKWARD_QUERY`, `SDPA_BACKWARD_KEY`, `SDPA_BACKWARD_VALUE`
- `LOG_SOFTMAX`, `SUM`, `MEAN`, `REDUCE_MIN`, `REDUCE_MAX`, `LAYER_NORM`, `RMS_NORM`, and loss ops are CPU-supported but not directly represented as forward accelerator DAG nodes today.
- `AcceleratorPostOpType` already captures many unary/binary post-ops after matmul-like compute. That is useful for Phase 12 but should not be treated as a full fused GPU execution contract in Phase 11.

### Existing backend allowlists

- `MetalPartitionSupport` has a source-level allowlist and stable human-readable unsupported reasons.
- `CudaGpuRegionLegalityAdapter` has an inline allowlist with less diagnostic detail than Metal.
- Metal and CUDA forward allowlists currently overlap for matmul/linear, common elementwise ops, `WHERE`, `SOFTMAX`, and layout-adjacent ops.
- Metal has additional explicit rejection reasons for direct forward SDPA because native MPSGraph scale/mask semantics are not verified against CPU semantics.
- CUDA direct non-dense compute remains conservative after Phase 10. CUDA layout/view residency is safe only through metadata-only view propagation or explicit dense materialization.

### Existing tests

- `MetalRegionLowererTest` already covers Metal graph region lowering, pure elementwise fused-region family labeling, dtype rejection, `WHERE` predicate rules, and direct SDPA rejection diagnostics.
- `CudaRegionLowererTest` currently covers CUDA fused elementwise lowering family but does not have the same legality/rejection matrix as Metal.
- `PreparedExecutionBuildTest` has many prepared-execution assertions for selected Metal and CUDA candidates, backend selection decisions, cost rejection, and prior Phase 10 layout-adjacent flows.
- `CompiledGraphTraceTest`, `MetalLayoutAwareDeviceFlowTest`, and `CudaLayoutTransformDeviceFlowTest` provide trace/materialization patterns that Phase 11 can reuse.

### Documentation state

- `docs/metal-backend.md` documents current Metal allowlist and direct SDPA caveats.
- `docs/native-bridges-and-blas.md` documents native ABI and layout-transform contracts.
- There is no dedicated checked-in Metal/CUDA lowering coverage matrix that maps operation families to `supported`, `fallback`, or `unsupported` with stable reason codes.

## Recommended Architecture

### 1. Shared coverage model as executable contract

Add a backend-neutral coverage model under `backend.accelerator.lowering` rather than duplicating source-of-truth lists in Metal and CUDA:

- `GpuLoweringCoverageStatus`: `SUPPORTED`, `FALLBACK`, `UNSUPPORTED`
- `GpuLoweringUnsupportedReason`: stable reason codes such as `SUPPORTED`, `UNSUPPORTED_OPERATION`, `UNSUPPORTED_DTYPE`, `UNSUPPORTED_LAYOUT`, `UNSUPPORTED_RANK_OR_SHAPE`, `CAPABILITY_MISSING`, `NATIVE_ABI_MISMATCH`, and `DEFERRED_FUSED_REGION`
- `GpuLoweringOperationFamily`: rows for matmul/linear, elementwise chains, layout/view-adjacent nodes, softmax/log-softmax-ish flows, reductions, normalization, loss-adjacent ops, attention/SDPA, convolution/pooling, index/gather/scatter, compare/bool, and backward-adjacent gradients
- `GpuLoweringCoverageEntry`: backend, operation type, family, status, reason, and notes
- `GpuLoweringCoverageMatrix`: deterministic entries for Metal and CUDA

The matrix should be small and testable. It should classify individual `Operation.OpType` values where possible and group unsupported families with explicit reason codes where direct lowering is not safe yet.

### 2. Backend adapters consume the shared model but keep final capability ownership

Use the common model to avoid hand-growing divergent allowlists:

- `MetalPartitionSupport.plannerUnsupportedReason(...)` should consult shared coverage entries, then apply Metal-specific dtype, external input role, and SDPA semantic checks.
- `CudaGpuRegionLegalityAdapter` should gain a stable `plannerUnsupportedReason(...)` helper and use the same shared coverage entries before CUDA-specific backend/layout capability checks.
- Backend-specific native capability stays backend-owned. The shared matrix says what semantic lowering is claimed; Metal/CUDA still reject at runtime or preparation for missing native symbols, layout ABI mismatch, dtype gaps, or disabled accelerator config.

### 3. Narrow lowering expansion for softmax-ish flows

The highest-value low-risk expansion is softmax-ish coverage:

- `SOFTMAX` is already in the DAG and allowlists.
- `LOG_SOFTMAX` can be represented as an accelerator DAG decomposition into `SOFTMAX` followed by `LOG`, both of which are already DAG-level primitives.
- Tests can prove the lowered DAG uses two native-supported nodes without adding a new native ABI op code.

Reduction, normalization, and loss-adjacent families should be matrix-first unless the executor finds a narrow, testable native representation. They must not be hidden CPU fallbacks. Entries should explain why `SUM`, `MEAN`, `REDUCE_*`, `LAYER_NORM`, `RMS_NORM`, `NLL_LOSS`, and `CROSS_ENTROPY_*` remain fallback/unsupported in this phase if not implemented.

### 4. Selected and rejected candidate evidence

Phase 11 should not only test source-level allowlists. It should prove:

- selected high-value GPU candidates stay in Metal/CUDA regions when dtype/layout/capability/cost rules allow,
- rejected candidates name unsupported operation/dtype/layout/capability reasons,
- unsupported reduction/normalization/loss-adjacent families fall back visibly and do not corrupt CPU/Metal/CUDA safeguards,
- docs and tests prevent drift between the matrix and actual planner/lowerer behavior.

## Validation Architecture

Use focused JUnit/Gradle gates:

- Shared coverage matrix:
  - `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest`
- Metal/CUDA legality alignment:
  - `./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest`
- Lowerer expansion and selected/rejected candidates:
  - `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests PreparedExecutionBuildTest`
- Trace/docs closure:
  - `./gradlew test --tests CompiledGraphTraceTest`
  - `rg -n "GPU Lowering Coverage Matrix|GPULOWER-01|GPULOWER-02|GPULOWER-03|UNSUPPORTED_OPERATION|LOG_SOFTMAX" docs .planning src/test/java`

Optional native checks remain capability-gated:

- `./gradlew classes`
- `./gradlew metalTest`
- `./gradlew buildCudaGraphShim cudaTest`

## Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Matrix drifts from source allowlists | Add a test that checks documented statuses against `GpuLoweringCoverageMatrix` and adapter support methods. |
| CUDA accepts non-dense compute silently | Keep CUDA layout rejection conservative and include an explicit matrix/test entry for direct non-dense CUDA compute. |
| Phase 11 grows into Phase 12 fusion | Use `DEFERRED_FUSED_REGION` entries for fused compound patterns and leave compound execution to Phase 12. |
| Phase overclaims reductions/norm/loss | Mark unsupported/fallback rows with `UNSUPPORTED_OPERATION` or `DEFERRED_FUSED_REGION`, then prove rejected candidates fall back visibly. |
| Native ABI changes break dense execution | Prefer DAG decompositions using existing op codes before adding native ABI codes; run `./gradlew classes` and targeted backend tests. |
| Local benchmark/profile artifacts get committed | Keep tests focused and do not stage `profiles/platform/.../tuning/abc/*`. |

## Planning Implications

Recommended waves:

1. Shared coverage matrix, docs contract, and drift tests.
2. Metal/CUDA legality adapters consume common coverage decisions and expose stable rejection reasons.
3. Narrow softmax-ish lowering expansion plus selected/rejected candidate tests.
4. Trace/docs closure and final phase verification evidence.

## Research Complete

Phase 11 should be planned as a four-wave implementation. The central design is an executable coverage matrix plus shared legality helpers, with cautious softmax-ish expansion and explicit fallback for unsupported reductions, normalization, and loss-adjacent families.

