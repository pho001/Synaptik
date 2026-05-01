# GPU Lowering Coverage Matrix

This document is the checked-in coverage contract for Phase 11 GPU lowering and Phase 12 GPU compound region lowering. It covers `GPULOWER-01`, informs `GPULOWER-02`, defines the stable reason-code vocabulary required by `GPULOWER-03`, and records Phase 12 `LINEAR_BIAS_ACTIVATION`, `ELEMENTWISE_CHAIN`, and `REDUCTION_ADJACENT` compound behavior.

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
| `DEFERRED_FUSED_REGION` | Compound fused GPU region execution is recognized but deliberately deferred outside the current supported subset. |
| `CPU_FUSED_OPERATION_UNSUPPORTED` | CPU `Operation.OpType.FUSED` remains CPU-only and is not consumed by GPU compound lowering. |
| `COMPOUND_PATTERN_UNSUPPORTED` | A candidate compound region is recognized but not in the current supported GPU compound subset. |
| `COMPOUND_REGION_SHORTENED` | A supported compound target pattern was shortened before GPU lowering and must fail required-mode checks. |

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
| fused compound patterns | `FUSED` | unsupported | `CPU_FUSED_OPERATION_UNSUPPORTED`; CPU `Operation.OpType.FUSED` remains CPU-only for Phase 12 |

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
| fused compound patterns | `FUSED` | unsupported | `CPU_FUSED_OPERATION_UNSUPPORTED`; CPU `Operation.OpType.FUSED` remains CPU-only for Phase 12 |

## Runtime Boundary

The public `Tensor` remains logical and device residency stays in `ExecutionState` and `DeviceBufferBinding`. The matrix does not grant public device tensors, and it does not bypass backend-owned dtype, layout, cost, capability, or ABI checks.

Metal and CUDA coverage is backend-specific. Shared rows describe the common semantic contract, but each backend can still reject a row through its own native capability, dtype, layout, ABI, or buffer-binding gates.

## GPU Compound Region Lowering

GPU compound region lowering is the Phase 12 path that lets Metal and CUDA execute selected multi-node regions without importing CPU fused ASM/vector internals. Supported compound summaries currently include `LINEAR_BIAS_ACTIVATION` and representative `ELEMENTWISE_CHAIN` regions. `REDUCTION_ADJACENT` candidates such as `SUM`, `MEAN`, `REDUCE_MIN`, `REDUCE_MAX`, `LAYER_NORM`, and `RMS_NORM` are recognized but reject with stable reason codes until a narrow GPU implementation with parity tests exists.

`Operation.OpType.FUSED remains CPU-only`. GPU compound regions lower from normal graph operations through `AcceleratorSubgraphLowerer`, backend-specific Metal/CUDA legality, and backend-specific prepared executables. CPU fused operations remain in the CPU planning and execution path.

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

Phase 12 owns GPU compound region lowering for patterns such as `LINEAR_BIAS_ACTIVATION`, `ELEMENTWISE_CHAIN`, and `REDUCTION_ADJACENT`. CPU `Operation.OpType.FUSED` remains CPU-only; GPU compound regions arise from normal graph operations lowered through accelerator DAG primitives. Phase 13 owns coverage benchmark gates and report thresholds for GPU region length, fallback counts, CPU materialization counts, and device handoffs.

## Coverage-Driven Expansion

Phase 14 adds coverage triage on top of the Phase 13 report contract. `GpuCoverageGapTriage`,
`GpuCoverageTriageReport`, and `GpuHotPathCoverageTargets` rank measured GPU exits before later phases add broader
lowering, storage, fusion, or multi-op execution support.

The source-of-truth target list is
`.planning/phases/14-coverage-gap-triage-and-hot-path-targets/14-HOT-PATH-TARGETS.md`. Later lowering and fusion work
should close ranked gaps for `transformer_block_hot_path`, `mlp_classifier_small`, `conv2d_resnet_3x3`, and
`layer_norm_small` before adding speculative operation coverage.

## Phase 15 GPU Lowered Region Manifest

Phase 15 adds the [GPU Lowered Region Manifest](gpu-lowered-region-manifest.md) as the trace/report contract for selected GPU regions. The manifest describes selected regions as internal lowered DAGs with original op mapping, lowered primitive mapping, dtype/layout/storage assumptions, fused subpattern placeholders, rejection evidence, and candidate-shortening evidence.

The Phase 15 DAG-level reason constants are:

- `DAG_PRIMITIVE_UNSUPPORTED`
- `DAG_REGION_BOUNDARY_MATERIALIZATION`
- `DAG_CANDIDATE_SHORTENED`
- `DAG_FUSED_SUBPATTERN_REJECTED`

## Phase 13 Coverage Gates

Phase 11 defines which operation families Metal and CUDA may lower, and Phase 12 defines which multi-node compound
regions may become GPU-owned regions. Phase 13 turns that support into an auditable GPU coverage summary so benchmark
and regression reports can prove that selected regions actually stayed on the accelerator path.

The Phase 13 report fields include `gpuCoverageRatio`, `selectedRegionCount`, `maxSelectedRegionLength`,
`rejectedCandidateReasonCounts`, `cpuMaterializationReasonCounts`, and `deviceHandoffCount`. These fields connect the
lowering matrix and compound-region decisions to runtime evidence: selected GPU regions, rejected candidates, CPU
materialization boundaries, tensor-array bridge usage, CPU fallback, and device handoffs.

The portable coverage gate checks coverage/materialization behavior, not raw timing. It is allowed to fail fast on a
hidden tensor-array fallback even if a benchmark is fast, because tensor-array execution is a CPU-visible bridge path
rather than native buffer ownership. Native Metal and CUDA checks are native capability-gated evidence; when local CUDA
is capability-skipped, portable Java tests remain the required proof for schema, fallback, and report contracts.
