# GPU Lowering Coverage Matrix

This document is the checked-in coverage contract for Phase 11 GPU lowering. It covers `GPULOWER-01`, informs `GPULOWER-02`, and defines the stable reason-code vocabulary required by `GPULOWER-03`.

The source of truth lives in `backend.accelerator.lowering.GpuLoweringCoverageMatrix`. The tables below are intentionally conservative: a `supported` row means the operation family is represented in the shared accelerator DAG and admitted by the current planner contract when dtype, layout, runtime enablement, cost, and backend capability checks also pass. A `fallback` or `unsupported` row must remain visible in traces and reports when backend selection rejects or materializes a boundary.

## Status Legend

| Status | Meaning |
|---|---|
| `supported` | The operation has a checked-in GPU lowering path for the listed backend subject to backend-owned dtype, layout, and capability checks. |
| `fallback` | The operation is recognized by the matrix but currently exits to CPU or another legal path with a stable reason. |
| `unsupported` | The operation is intentionally outside current GPU lowering coverage and must reject explicitly. |

## Reason-Code Legend

| Reason code | Meaning |
|---|---|
| `SUPPORTED` | The row is supported. Non-supported rows must not use this reason. |
| `UNSUPPORTED_OPERATION` | The operation is not in the tested GPU planner or native DAG coverage. |
| `UNSUPPORTED_DTYPE` | The operation depends on a dtype outside current GPU output/input contracts, such as index targets or BOOL outputs. |
| `UNSUPPORTED_LAYOUT` | The operation needs a layout that is not legal for the selected backend path. |
| `UNSUPPORTED_RANK_OR_SHAPE` | The operation rank or shape is outside the native DAG or backend bridge contract. |
| `CAPABILITY_MISSING` | The semantic operation is known, but the backend native capability is not currently enabled or verified. |
| `NATIVE_ABI_MISMATCH` | The Java/native ABI version or symbol contract does not match the required lowering path. |
| `DEFERRED_FUSED_REGION` | Compound fused GPU region execution is deliberately deferred to Phase 12. |

## Metal

| Family | Representative operations | Status | Reason |
|---|---|---|---|
| matmul/linear | `MATMUL`, `LINEAR` | supported | `SUPPORTED` |
| elementwise chains | `ADD`, `SUB`, `MUL`, `DIV`, `RELU`, `TANH`, `SIGMOID`, `ABS`, `EXP`, `LOG`, `NEG`, `SQRT`, `INV`, `MUL_SCALAR`, `WHERE`, `CLAMP_MIN`, `CLAMP_MAX` | supported | `SUPPORTED` |
| layout/view-adjacent nodes | `RESHAPE`, `CONTIGUOUS`, `NOOP`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE` | supported | `SUPPORTED` |
| softmax/log-softmax-ish flows | `SOFTMAX` | supported | `SUPPORTED` |
| softmax/log-softmax-ish flows | `LOG_SOFTMAX` | supported | `SUPPORTED`; lowered as SOFTMAX followed by LOG |
| reductions | `SUM`, `MEAN`, `REDUCE_MIN`, `REDUCE_MAX` | fallback | `UNSUPPORTED_OPERATION` |
| normalization pieces | `LAYER_NORM`, `RMS_NORM` | fallback | `DEFERRED_FUSED_REGION` |
| loss-adjacent ops | `NLL_LOSS`, `CROSS_ENTROPY_LOSS` | unsupported | `UNSUPPORTED_OPERATION` |
| loss-adjacent ops | `CROSS_ENTROPY_LOSS_INDICES`, `CROSS_ENTROPY_LOSS_INDICES_GRAD` | unsupported | `UNSUPPORTED_DTYPE` |
| attention/SDPA | `SCALED_DOT_PRODUCT_ATTENTION` | fallback | `CAPABILITY_MISSING` |
| attention/SDPA | `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD` | supported | `SUPPORTED` |
| conv/pool | `CONV2D`, `MAX_POOL2D` | unsupported | `UNSUPPORTED_OPERATION` |
| index/scatter/gather | `GATHER`, `SCATTER_ADD` | unsupported | `UNSUPPORTED_OPERATION` |
| compare/bool | `GT`, `EQ` | unsupported | `UNSUPPORTED_DTYPE` |
| backward-adjacent | `SOFTMAX_GRAD`, `LOG_SOFTMAX_GRAD`, `REDUCE_MIN_GRAD`, `REDUCE_MAX_GRAD`, `MIN_GRAD`, `MAX_GRAD` | supported | `SUPPORTED` |
| fused compound patterns | `FUSED` | fallback | `DEFERRED_FUSED_REGION` |

## CUDA

| Family | Representative operations | Status | Reason |
|---|---|---|---|
| matmul/linear | `MATMUL`, `LINEAR` | supported | `SUPPORTED` |
| elementwise chains | `ADD`, `SUB`, `MUL`, `DIV`, `RELU`, `TANH`, `SIGMOID`, `ABS`, `EXP`, `LOG`, `NEG`, `SQRT`, `INV`, `MUL_SCALAR`, `WHERE`, `CLAMP_MIN`, `CLAMP_MAX` | supported | `SUPPORTED` |
| layout/view-adjacent nodes | `RESHAPE`, `CONTIGUOUS`, `NOOP`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE` | supported | `SUPPORTED` |
| softmax/log-softmax-ish flows | `SOFTMAX` | supported | `SUPPORTED` |
| softmax/log-softmax-ish flows | `LOG_SOFTMAX` | supported | `SUPPORTED`; lowered as SOFTMAX followed by LOG |
| reductions | `SUM`, `MEAN`, `REDUCE_MIN`, `REDUCE_MAX` | fallback | `UNSUPPORTED_OPERATION` |
| normalization pieces | `LAYER_NORM`, `RMS_NORM` | fallback | `DEFERRED_FUSED_REGION` |
| loss-adjacent ops | `NLL_LOSS`, `CROSS_ENTROPY_LOSS` | unsupported | `UNSUPPORTED_OPERATION` |
| loss-adjacent ops | `CROSS_ENTROPY_LOSS_INDICES`, `CROSS_ENTROPY_LOSS_INDICES_GRAD` | unsupported | `UNSUPPORTED_DTYPE` |
| attention/SDPA | `SCALED_DOT_PRODUCT_ATTENTION` | fallback | `UNSUPPORTED_OPERATION` |
| attention/SDPA | `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD` | unsupported | `CAPABILITY_MISSING` |
| conv/pool | `CONV2D`, `MAX_POOL2D` | unsupported | `UNSUPPORTED_OPERATION` |
| index/scatter/gather | `GATHER`, `SCATTER_ADD` | unsupported | `UNSUPPORTED_OPERATION` |
| compare/bool | `GT`, `EQ` | unsupported | `UNSUPPORTED_DTYPE` |
| backward-adjacent | `SOFTMAX_GRAD`, `LOG_SOFTMAX_GRAD`, `REDUCE_MIN_GRAD`, `REDUCE_MAX_GRAD`, `MIN_GRAD`, `MAX_GRAD` | supported | `SUPPORTED` |
| fused compound patterns | `FUSED` | fallback | `DEFERRED_FUSED_REGION` |

## Runtime Boundary

The public `Tensor` remains logical and device residency stays in `ExecutionState` and `DeviceBufferBinding`. The matrix does not grant public device tensors, and it does not bypass backend-owned dtype, layout, cost, capability, or ABI checks.

## Planner Rejection Sources

Shared matrix entries classify semantic support for operation families and provide the stable fallback or unsupported reason code used by Metal and CUDA planner diagnostics.

Metal adds backend-owned dtype, external-input role, and SDPA semantic gates before it accepts a matrix-supported operation. CUDA keeps direct non-dense CUDA compute remains conservative unless Phase 10 metadata-only view propagation or dense materialization makes the layout legal for the consumer. Backend runtime capability and native ABI checks remain backend-owned and can still reject a matrix-supported row at prepare or execution time.

`LOG_SOFTMAX` support does not add a native ABI op code. It is lowered as SOFTMAX followed by LOG using existing accelerator DAG primitives.

## Adding A GPU-Lowerable Operation Family

Use the regular tensor operation checklist first, then add GPU coverage deliberately:

1. Add or update the semantic `Operation.OpType`.
2. Add a `GpuLoweringCoverageMatrix` entry for Metal and CUDA.
3. Add or update `AcceleratorDagNodeType` only when a native ABI op is truly required.
4. prefer decomposition through existing DAG primitives where safe.
5. Update `AcceleratorSubgraphLowerer`.
6. Update Metal/CUDA planner tests.
7. Keep backend-specific native capability checks backend-owned.
8. Add selected and rejected candidate tests.

Phase 12 owns fused GPU compound execution for patterns such as linear plus bias plus activation and longer elementwise chains. Phase 13 owns coverage benchmark gates and report thresholds for GPU region length, fallback counts, CPU materialization counts, and device handoffs.
