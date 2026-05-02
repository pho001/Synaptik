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
| `UNSUPPORTED_INDEX_SEMANTICS` | The operation depends on index-target semantics such as INT32 targets, bounds checks, ignore-index, or per-class scatter behavior that are not yet native GPU primitives. |
| `UNSUPPORTED_IGNORE_INDEX` | The operation depends on ignore-index semantics that have not been proven for native GPU execution. |
| `UNSUPPORTED_DUPLICATE_INDEX` | The operation depends on duplicate-index accumulation semantics that have not been proven for native GPU execution. |
| `UNSUPPORTED_BOUNDS_CHECK` | The operation depends on index bounds behavior that has not been proven for native GPU execution. |
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
| layout/view-adjacent nodes | `RESHAPE`, `CONTIGUOUS`, `NOOP`, `PERMUTE`, `EXPAND`, `EXPAND_DIMS`, `SQUEEZE` | supported | `SUPPORTED`; router distinguishes metadata-only views, dense materialization, broadcast materialization, and unsupported strided compute |
| softmax/log-softmax-ish flows | `SOFTMAX` | supported | `SUPPORTED` |
| softmax/log-softmax-ish flows | `LOG_SOFTMAX` | supported | `SUPPORTED`; lowered as SOFTMAX followed by LOG |
| reductions | `SUM`, `MEAN`, `REDUCE_MIN`, `REDUCE_MAX` | supported | `SUPPORTED`; axis and keep-dims metadata lower through the shared DAG ABI |
| normalization pieces | `LAYER_NORM`, `RMS_NORM` | supported | `SUPPORTED`; lowered as repeated keep-dims `MEAN` plus elementwise normalization DAG with epsilon scalar |
| loss-adjacent ops | `NLL_LOSS`, `CROSS_ENTROPY_LOSS` | unsupported | `DAG_PRIMITIVE_UNSUPPORTED` |
| loss-adjacent ops | `CROSS_ENTROPY_LOSS_INDICES`, `CROSS_ENTROPY_LOSS_INDICES_GRAD` | unsupported | `UNSUPPORTED_INDEX_SEMANTICS` |
| attention/SDPA | `SCALED_DOT_PRODUCT_ATTENTION` | supported | `SUPPORTED`; direct unmasked FLOAT32 rank-3/4 native MPSGraph primitive SDPA DAG; masked direct SDPA remains `UNSUPPORTED_MASK_SEMANTICS` until BOOL mask semantics match backend mask behavior |
| attention/SDPA | `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD` | supported | `SUPPORTED` |
| conv/pool | `CONV2D`, `CONV2D_GEMM`, `CONV2D_BACKWARD_INPUT`, `CONV2D_BACKWARD_WEIGHT`, `CONV2D_BACKWARD_INPUT_GEMM`, `CONV2D_BACKWARD_WEIGHT_GEMM`, `MAX_POOL2D`, `MAX_POOL2D_BACKWARD_INPUT`, `AVG_POOL2D`, `AVG_POOL2D_BACKWARD_INPUT` | unsupported | `CAPABILITY_MISSING` |
| index/scatter/gather | `GATHER`, `TAKE_ALONG_AXIS` | supported | `SUPPORTED`; dense `FLOAT32` value/output with `INT32` indices lowers to MPSGraph `gatherAlongAxis` |
| index/scatter/gather | `GATHER_GRAD`, `TAKE_ALONG_AXIS_GRAD`, `SCATTER_ADD` | unsupported | `UNSUPPORTED_DUPLICATE_INDEX` |
| compare/bool | `GT`, `GE`, `LT`, `LE`, `EQ`, `NE`, `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT`, `REDUCE_ALL`, `REDUCE_ANY` | supported | `SUPPORTED`; dense scoped BOOL outputs execute through dtype ABI v3 and can feed legal `WHERE` mask-chain consumers without CPU materialization |
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
| reductions | `SUM`, `MEAN`, `REDUCE_MIN`, `REDUCE_MAX` | supported | `SUPPORTED`; axis and keep-dims metadata lower through the shared DAG ABI |
| normalization pieces | `LAYER_NORM`, `RMS_NORM` | supported | `SUPPORTED`; lowered as repeated keep-dims `MEAN` plus elementwise normalization DAG with epsilon scalar |
| loss-adjacent ops | `NLL_LOSS`, `CROSS_ENTROPY_LOSS` | unsupported | `DAG_PRIMITIVE_UNSUPPORTED` |
| loss-adjacent ops | `CROSS_ENTROPY_LOSS_INDICES`, `CROSS_ENTROPY_LOSS_INDICES_GRAD` | unsupported | `UNSUPPORTED_INDEX_SEMANTICS` |
| attention/SDPA | `SCALED_DOT_PRODUCT_ATTENTION` | fallback | `CAPABILITY_MISSING`; CUDA direct forward SDPA native/lowered path is not implemented and must provide backend evidence before this can become supported |
| attention/SDPA | `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD` | unsupported | `CAPABILITY_MISSING` |
| conv/pool | `CONV2D`, `CONV2D_GEMM`, `CONV2D_BACKWARD_INPUT`, `CONV2D_BACKWARD_WEIGHT`, `CONV2D_BACKWARD_INPUT_GEMM`, `CONV2D_BACKWARD_WEIGHT_GEMM`, `MAX_POOL2D`, `MAX_POOL2D_BACKWARD_INPUT`, `AVG_POOL2D`, `AVG_POOL2D_BACKWARD_INPUT` | unsupported | `CAPABILITY_MISSING` |
| index/scatter/gather | `GATHER`, `TAKE_ALONG_AXIS` | unsupported | `CAPABILITY_MISSING` |
| index/scatter/gather | `GATHER_GRAD`, `TAKE_ALONG_AXIS_GRAD`, `SCATTER_ADD` | unsupported | `UNSUPPORTED_DUPLICATE_INDEX` |
| compare/bool | `GT`, `GE`, `LT`, `LE`, `EQ`, `NE`, `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT`, `REDUCE_ALL`, `REDUCE_ANY` | unsupported | `UNSUPPORTED_DTYPE` |
| backward-adjacent | `SOFTMAX_GRAD`, `LOG_SOFTMAX_GRAD`, `REDUCE_MIN_GRAD`, `REDUCE_MAX_GRAD`, `MIN_GRAD`, `MAX_GRAD` | supported | `SUPPORTED` |
| fused compound patterns | `FUSED` | unsupported | `CPU_FUSED_OPERATION_UNSUPPORTED`; CPU `Operation.OpType.FUSED` remains CPU-only for Phase 12 |

## Runtime Boundary

The public `Tensor` remains logical and device residency stays in `ExecutionState` and `DeviceBufferBinding`. The matrix does not grant public device tensors, and it does not bypass backend-owned dtype, layout, cost, capability, or ABI checks.

Metal and CUDA coverage is backend-specific. Shared rows describe the common semantic contract, but each backend can still reject a row through its own native capability, dtype, layout, ABI, or buffer-binding gates.

Phase 33 adds coverage evidence for Metal layout repair. `layout_broadcast_repair_small` requires native buffer execution,
`BROADCAST_GPU_MATERIALIZATION` evidence, and no `CPU_CONSUMER` materialization on the supported hot path. This is not a
claim of universal strided-native compute; direct `STRIDED_NATIVE_COMPUTE` remains a named unsupported route until a
consumer primitive proves that contract.

## DType residency is not native dtype compute

`dtype residency is not native dtype compute`. Phase 16 dtype residency records whether values such as `BFLOAT16`, `INT32`, and `BOOL` can stay represented in runtime storage and trace/report metadata. It does not widen Metal or CUDA native arithmetic beyond the backend role gates.

Metal currently accepts `FLOAT32` compute/output broadly, `BFLOAT16` compute/output only for scoped operation families (`MATMUL`, `LINEAR`, arithmetic elementwise and activation ops, scalar multiply/clamp, `SOFTMAX`, `LOG_SOFTMAX`, `SUM`, `MEAN`, `REDUCE_MIN`, `REDUCE_MAX`, `LAYER_NORM`, and `RMS_NORM`), and `BOOL` compute/output only for scoped compare/logical/reduction mask families. `BOOL` can also be a predicate-style external input for `WHERE`, but reports distinguish that from native BOOL-producing compute through role-specific `dtypeResidency` evidence. CUDA native dense buffer execution remains `FLOAT32` only. Other dtype-role combinations reject with `UNSUPPORTED_DTYPE` and stable details such as `backend=GPU_METAL role=operation dtype=BFLOAT16 code=UNSUPPORTED_OPERATION_DTYPE` or `backend=GPU_CUDA role=output dtype=INT32`.

Reports use `dtypeResidency` evidence to explain why a region stayed resident, shortened, or exited. A dtype-resident internal value can still be materialized later for a real CPU consumer, graph output, or gradient publication, and that CPU boundary remains reportable.

Metal dtype capability truth is role-specific. `MetalMpsCapabilities` distinguishes storage representability, external
input legality, predicate input legality, native compute legality, native output legality, and operation-specific dtype
support. The optional Metal dtype ABI v3 probes prove that a native shim can describe widened dtype roles; BF16 support
is still operation-scoped and capability-gated, BOOL output is limited to scoped compare/logical/reduction families,
and INT32 plus FLOAT64 remain non-native compute/output.

## Phase 30 BF16 Metal contract

Phase 30 adds native Metal BF16 compute/output for the scoped families above. The contract is intentionally narrower
than "Metal supports BF16 everywhere":

- BF16 storage, external inputs, compute nodes, and outputs use dtype ABI v3 metadata.
- BF16 upload/readback roundtrip is exact raw BF16 storage equality.
- BF16 MATMUL and reduction parity use explicit BF16 numeric tolerance.
- BF16 LayerNorm/RMSNorm and softmax/log-softmax parity use a separate normalization/softmax tolerance.
- BF16 hot-path targets `mlp_classifier_small_bf16`, `layer_norm_small_bf16`, `rms_norm_small_bf16`, and
  `reduction_chain_small_bf16` require Metal native buffer evidence, dtype residency evidence, zero CPU
  materializations, zero CPU fallback, and zero tensor-array fallback.

Unsupported BF16 and BOOL families must remain visible. Conv/pool, loss-adjacent ops, gather/take gradients, scatter,
masked SDPA, arbitrary BOOL consumers, and generic INT32 compute/output remain rejected or fallback rows until their own
semantic, native execution, parity, trace, and regression-gate evidence exists.

## Phase 17 normalization, reduction, and loss-adjacent contract

Phase 17 covers `GPUNORM-01` and `GPUNORM-02` by making normalization, reduction, softmax-ish, conv, and loss-adjacent gaps explicit in the shared Metal/CUDA matrix. The source-of-truth targets are `target=layer_norm_small`, `target=conv2d_resnet_3x3`, and `target=transformer_block_hot_path`.

Phase 23 closes the forward reduction subset. `SUM`, `MEAN`, `REDUCE_MIN`, and `REDUCE_MAX` are supported rows for legal dense `FLOAT32` Metal/CUDA regions. Phase 24 extends that support to legal dense `FLOAT32` `LAYER_NORM` and `RMS_NORM` by lowering them into repeated keep-dims `MEAN`, epsilon scalar add, and elementwise DAG primitives. Phase 31 closes the Metal BOOL compare/logical/reduction subset. Loss, conv/pool, indexing, and CUDA BOOL outputs remain separate closure targets.

`LOG_SOFTMAX remains lowered as SOFTMAX followed by LOG`. That support is separate from loss-adjacent operations such as `NLL_LOSS`, `CROSS_ENTROPY_LOSS`, and `CROSS_ENTROPY_LOSS_INDICES`, where unsupported loss-adjacent rows must remain visible fallback, not silent CPU replay.

Forward reductions (`SUM`, `MEAN`, `REDUCE_MIN`, `REDUCE_MAX`) and legal dense `FLOAT32` or scoped `BFLOAT16` normalization (`LAYER_NORM`, `RMS_NORM`) are now supported for the `target=layer_norm_small` and BF16 coverage paths. Unsupported normalization variants still reject explicitly: unsupported dtype or BF16 operation family uses `UNSUPPORTED_DTYPE`, non-dense direct inputs use `UNSUPPORTED_LAYOUT`, and invalid rank or tail parameter shape uses `UNSUPPORTED_RANK_OR_SHAPE`. `CONV2D` is a matrix-visible blocker for `target=conv2d_resnet_3x3` and remains unsupported until conv lowering is explicitly added. Index-target loss rows use `UNSUPPORTED_INDEX_SEMANTICS` because `INT32` targets, bounds checks, ignore-index handling, and reduction denominator semantics are outside the current native accelerator DAG compute contract.

## Phase 26 loss and indexing contract

Phase 26 makes loss-adjacent and indexing gaps explicit without claiming native GPU execution. `NLL_LOSS` and dense `CROSS_ENTROPY_LOSS` remain `DAG_PRIMITIVE_UNSUPPORTED` until a backend-owned loss primitive or lowered loss sub-DAG exists. `CROSS_ENTROPY_LOSS_INDICES` and `CROSS_ENTROPY_LOSS_INDICES_GRAD` remain `UNSUPPORTED_INDEX_SEMANTICS` because legal support must preserve INT32 targets, bounds behavior, ignore-index masking, reduction denominators, and per-class gradient scatter behavior.

Phase 32 promotes scoped Metal forward `GATHER` and `TAKE_ALONG_AXIS` from residency-only evidence to native index compute for dense `FLOAT32` value/output tensors with `INT32` index inputs whose bounds can be proven from a static leaf index tensor. `TAKE_ALONG_AXIS` maps directly to MPSGraph `gatherAlongAxis`; `GATHER` expands the reduced index tensor on the gathered axis, runs `gatherAlongAxis`, then squeezes the axis back to the public shape. Unproven or out-of-range bounds reject with `UNSUPPORTED_BOUNDS_CHECK` because MPSGraph out-of-bounds behavior does not match CPU exception semantics. CUDA forward gather/take remains `CAPABILITY_MISSING`. `GATHER_GRAD`, `TAKE_ALONG_AXIS_GRAD`, and `SCATTER_ADD` remain `UNSUPPORTED_DUPLICATE_INDEX` because duplicate-index accumulation must match CPU semantics before GPU support can be claimed.

`native support is not implied by matrix text alone`. A supported row means lowering, legality, native execution, trace/report, and parity evidence exist for the legal scoped case. A fallback or unsupported row means the blocker is recognized, diagnosed, and kept visible for planning and reports.

## Phase 27 conv/pool and BOOL output contract

Phase 27 expands the matrix surface for conv/pool and BOOL-producing operations without claiming native GPU execution prematurely. Metal and CUDA now list every targeted conv/pool op explicitly: `CONV2D`, `CONV2D_GEMM`, `CONV2D_BACKWARD_INPUT`, `CONV2D_BACKWARD_WEIGHT`, `CONV2D_BACKWARD_INPUT_GEMM`, `CONV2D_BACKWARD_WEIGHT_GEMM`, `MAX_POOL2D`, `MAX_POOL2D_BACKWARD_INPUT`, `AVG_POOL2D`, and `AVG_POOL2D_BACKWARD_INPUT`. These rows are `CAPABILITY_MISSING` until backend-owned primitives or verified library routing preserve NCHW rank-4 shape, stride, padding, dilation, groups, pooling tie behavior, and average-pool divisor semantics.

BOOL-producing operations are also explicit: `GT`, `GE`, `LT`, `LE`, `EQ`, `NE`, `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT`, `REDUCE_ALL`, and `REDUCE_ANY`. Phase 31 promotes the Metal rows to supported for dense scoped BOOL outputs through MPSGraph compare/logical/reduction primitives. Existing BOOL residency evidence for external `WHERE` predicate inputs remains storage/role support only; native BOOL-producing GPU compute is proved separately by operation dtype legality, native buffer execution, exact BOOL byte parity, and `dtypeResidency` evidence for `role=compute` or `role=internalValue`.

CUDA BOOL-producing rows remain `UNSUPPORTED_DTYPE` until a CUDA-native implementation provides the same DAG/native contract: ABI node type, lowerer mapping, legality gates, native execution or vendor-library routing, CPU parity tests, and report evidence for selected region length, lowered primitive count, backend path, and CPU exits.

## GPU Compound Region Lowering

GPU compound region lowering is the Phase 12 path that lets Metal and CUDA execute selected multi-node regions without importing CPU fused ASM/vector internals. Supported compound summaries currently include `LINEAR_BIAS_ACTIVATION`, representative `ELEMENTWISE_CHAIN`, and legal forward `NORMALIZATION` regions. Reduction-adjacent operations that have been implemented (`SUM`, `MEAN`, `REDUCE_MIN`, `REDUCE_MAX`, `LAYER_NORM`, `RMS_NORM`) are no longer rejected solely because they are reduction-adjacent; unsupported variants still reject with stable dtype, layout, rank, or operation reason codes.

`Operation.OpType.FUSED remains CPU-only`. GPU compound regions lower from normal graph operations through `AcceleratorSubgraphLowerer`, backend-specific Metal/CUDA legality, and backend-specific prepared executables. CPU fused operations remain in the CPU planning and execution path.

## Phase 18 fused elementwise and epilogue subregions

Phase 18 covers `GPUFUSEX-01`, `GPUFUSEX-02`, and `GPUFUSEX-03` by making GPU fusion a list-capable region-internal metadata contract. `GpuFusionSubpatternSummary` records the fused subpattern type, support state, original operation span, lowered primitive ids, lowered primitive count, rejection reason, and detail.

`GPU fusion is region-internal lowering/fusion, not CPU fused ASM reuse`. Trace and benchmark report evidence includes `gpuFusedSubpatternCount`, `gpuFusedSubpatternTypes`, `gpuFusedSubpatternOriginalNodeIds`, `gpuFusedSubpatternLoweredPrimitiveCount`, and `gpuFusedSubpatternReasons` beside existing fallback, materialization, and buffer-decision fields.

`GPU fusion is region-internal lowering/fusion`. Partitioning still selects a device-owned region, and lowering records supported subpatterns inside that region. Elementwise chains satisfy `GPUFUSEX-01`; matmul or linear epilogues such as bias plus activation satisfy `GPUFUSEX-02` only when dtype, layout, backend, and capability gates allow it.

`CPU Operation.OpType.FUSED remains CPU-only`. `GPUFUSEX-03` forbids importing CPU fused ASM/vector internals into accelerator, Metal, or CUDA packages. GPU fusion can share semantic operation knowledge with CPU fusion, but not CPU implementation internals or public `Operation.OpType.FUSED` nodes.

Phase 18 verification treated local tuning outputs as non-canonical evidence; `profiles/platform/.../tuning/abc/* remained unstaged`.

## Phase 19 multi-op GPU region execution

Phase 19 extends the checked-in contract from named compound summaries to selected multi-op Metal/CUDA regions. A
`selected GPU partition can execute as one backend-owned lowered region` when the shared lowering matrix, backend
legality, dtype/layout gates, runtime capability checks, and native-buffer binding policy all accept the candidate.
`ExecutionState and device buffer bindings carry supported internal values` between supported internal steps; public
`Tensor` objects still remain logical and only become CPU-readable at real graph output, CPU consumer, or gradient
publication boundaries.

`tensor-array bridge execution is not native buffer GPU coverage`. Coverage reports keep `nativeBufferStepCount` tied to
`BUFFER_BINDING` and keep `tensorArrayStepCount`, CPU fallback, device handoffs, and CPU materializations separately
visible. The evidence for hot-path residency is trace/report based rather than timing-only.

`GPU fusion remains region-internal lowering/fusion, not CPU fused ASM reuse`. Multi-op GPU regions may contain lowered
layout/view, matmul/linear, elementwise, softmax/log-softmax-ish, normalization, and supported fused subpattern metadata,
but they do not consume CPU `Operation.OpType.FUSED` nodes or import CPU fused ASM/vector internals. Unsupported
normalization variants, conv, and loss-adjacent blockers remain visible support/rejection outcomes until a
backend-specific implementation with parity evidence exists. `vendor library routing is deferred to GPULIB-*`; current
GPU lowering still routes through the shared accelerator DAG contract before backend-specific execution.

Phase 19 closure treated local tuning outputs as non-canonical evidence; `profiles/platform/.../tuning/abc/* remained unstaged`.

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

## Phase 20 coverage regression hardening

Phase 20 coverage regression hardening closes the v1.3 coverage loop with hard gates over the report/trace fields.
`hot path stayed on GPU is trace/report evidence, not timing-only`: timing may explain a benchmark result, but gate
pass/fail comes from selected regions, lowered primitives, fused subpatterns, CPU exits, native buffer binding, and
device handoff evidence.

Benchmark reports render `targetCoverageGates`, `nativeEvidence`, and `capabilitySkipped` so portable Java proof stays
separate from capability-gated native Metal/CUDA evidence. `tensor-array bridge execution is not native buffer GPU coverage`, and local tuning outputs are not closure proof; Phase 20 hygiene recorded that
`profiles/platform/.../tuning/abc/* remained unstaged`.

## Phase 28 coverage regression closure

Phase 28 tightens the v1.4 closure gate around target truth rather than timing. A target whose operation family is
`NATIVE_EXECUTABLE` in `GpuTargetCoverageTruth` must use a hard native/buffer policy: native buffer evidence is required,
tensor-array bridge execution and CPU fallback must be zero, unexpected CPU materialization must be zero, and selected
region plus lowered primitive counts must stay above the target threshold. This currently covers reduction targets,
legal normalization targets, Metal forward SDPA, and the fused/MLP-style hot path.

Targets that remain unsupported or capability-gated are still audit targets, but they pass only when the report exposes
stable visible blockers. Conv/pool, CUDA BOOL-producing outputs, loss/index blockers, and CUDA forward SDPA
must surface reason evidence such as `CAPABILITY_MISSING`, `UNSUPPORTED_DTYPE`, `UNSUPPORTED_INDEX_SEMANTICS`, operation
family names, or target names. They do not count as native coverage closure until backend-owned execution and parity
evidence exist.

Suite reports include `coverageDeltaVsBaseline` so reviewers can compare trace-derived coverage counters against the
v1.4 pre-closure baseline without using raw latency medians as proof. The relevant evidence fields are
`maxSelectedRegionLength`, `loweredPrimitiveCount`, `nativeBufferStepCount`, `tensorArrayStepCount`,
`cpuFallbackStepCount`, `cpuMaterializationCount`, `fallbackCount`, `deviceHandoffCount`, `targetCoverageGates`,
`nativeEvidence`, `capabilitySkipped`, and visible reason-code lists.
