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

Phase 44 adds a real scoped custom Metal kernel route on top of the MPSGraph coverage matrix. Supported Metal rows still report `metalExecutionRoute=MPS_GRAPH` by default, but the dense buffer-bound `FLOAT32` single-node `RELU` candidate can report `metalExecutionRoute=CUSTOM_KERNEL` and `metalNativeCopyStrategy=TRUE_OUTPUT_BUFFER_WRITE`. This is not a universal replacement for MPSGraph: unsupported custom candidates, unsupported dtypes, unavailable custom symbols, non-buffer transport, and non-dense runtime bindings remain on MPSGraph, tensor-array fallback, or visible CPU fallback with rejected route evidence.

| Family | Representative operations | Status | Reason |
|---|---|---|---|
| matmul/linear | `MATMUL`, `LINEAR` | supported | `SUPPORTED` |
| elementwise chains | `ADD`, `SUB`, `MUL`, `DIV`, `RELU`, `TANH`, `SIGMOID`, `ABS`, `EXP`, `LOG`, `NEG`, `SQRT`, `INV`, `ERF`, `FLOOR`, `CEIL`, `SIGN`, `MUL_SCALAR`, `WHERE`, `CLAMP_MIN`, `CLAMP_MAX` | supported | `SUPPORTED`; `FLOOR`/`CEIL` produce floating outputs, not integer casts. |
| layout/view-adjacent nodes | `RESHAPE`, `CONTIGUOUS`, `NOOP`, `PERMUTE`, `EXPAND`, `EXPAND_DIMS`, `SQUEEZE`, `SLICE`, `CONCAT`, `PAD`, `TILE` | supported | `SUPPORTED`; router distinguishes metadata-only views, dense materialization, broadcast materialization, and unsupported strided compute. Static Metal layout descriptors support dense `FLOAT32`/`BFLOAT16` slice with `step=1`, concat, constant pad, and tile. |
| backward layout writes | `SLICE_GRAD` | supported | `SUPPORTED`; scoped static dense `FLOAT32`/`BFLOAT16` `step=1` slice gradients lower to zero-fill pad with explicit before/after attributes. Strided slice gradients and unsupported dtypes remain visible rejections. |
| dtype conversion | `CAST` | supported | `SUPPORTED`; scoped explicit cast-pair policy supports identity casts plus `FLOAT32 <-> BFLOAT16` conversion. `FLOAT64`, runtime `INT64`, and general `BOOL`/`INT32` numeric casts remain unsupported. |
| softmax/log-softmax-ish flows | `SOFTMAX` | supported | `SUPPORTED` |
| softmax/log-softmax-ish flows | `LOG_SOFTMAX` | supported | `SUPPORTED`; lowered as SOFTMAX followed by LOG |
| reductions | `SUM`, `MEAN`, `REDUCE_MIN`, `REDUCE_MAX`, `REDUCE_PROD`, `ARGMAX`, `CUMSUM` | supported | `SUPPORTED`; axis and keep-dims metadata lower through the shared DAG ABI. Metal `ARGMAX` produces public `INT64` indices through the native buffer path; CUDA `ARGMAX` remains unsupported. `CUMSUM` is a shape-preserving scan with static axis/exclusive/reverse metadata. |
| normalization pieces | `LAYER_NORM`, `RMS_NORM` | supported | `SUPPORTED`; lowered as repeated keep-dims `MEAN` plus elementwise normalization DAG with epsilon scalar |
| loss-adjacent ops | `NLL_LOSS`, `CROSS_ENTROPY_LOSS` | supported | `SUPPORTED`; dense `FLOAT32`, dense-layout, rank 1..4, mean-reduced scalar loss lowers to existing DAG primitives |
| loss-adjacent ops | `CROSS_ENTROPY_LOSS_INDICES`, `CROSS_ENTROPY_LOSS_INDICES_GRAD` | supported | `SUPPORTED`; dense `FLOAT32` logits/sampleScale, dense static in-bounds `INT32` targets, ignore-index masking, `NONE`/`SUM`/`MEAN` reductions, and scatter-style gradient subtraction lower to MPSGraph |
| attention/SDPA | `SCALED_DOT_PRODUCT_ATTENTION` | supported | `SUPPORTED`; direct FLOAT32/BFLOAT16 rank-3/4 native MPSGraph primitive SDPA DAG supports unmasked, dense external BOOL masked, causal, and external+causal effective mask modes |
| attention/SDPA | `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD` | supported | `SUPPORTED`; unmasked, dense external BOOL masked, causal, and external+causal SDPA producers lower to native Metal backward DAG nodes |
| attention/SDPA | `SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS` | supported | `SUPPORTED`; lowers from the producer SDPA descriptor to a native Metal `softmax(QK^T * scale + mask)` DAG without CPU runtime-cache materialization |
| conv/pool | `CONV2D`, `CONV2D_GEMM` | supported | `SUPPORTED`; direct dense `FLOAT32` NCHW/OIHW forward Conv2D lowers to MPSGraph `convolution2D` for groups=1, dilation=1, stride/padding, and optional bias; `CONV2D_GEMM` preserves the original conv descriptor and routes to the same native primitive |
| conv/pool | `MAX_POOL2D`, `AVG_POOL2D` | supported | `SUPPORTED`; direct dense `FLOAT32` NCHW forward pooling lowers to MPSGraph pooling for compact kernel/stride/padding metadata, with `AVG_POOL2D` scoped to `countIncludePad=false` |
| conv/pool | `CONV2D_BACKWARD_INPUT`, `CONV2D_BACKWARD_WEIGHT`, `CONV2D_BACKWARD_INPUT_GEMM`, `CONV2D_BACKWARD_WEIGHT_GEMM` | supported | `SUPPORTED`; dense `FLOAT32` rank-4 NCHW/OIHW conv backward lowers to MPSGraph `convolution2DDataGradient` / `convolution2DWeightsGradient`, scoped to groups=1 and dilation=1 |
| conv/pool | `AVG_POOL2D_BACKWARD_INPUT` | supported | `SUPPORTED`; dense `FLOAT32` rank-4 NCHW average-pool backward lowers to MPSGraph `avgPooling2DGradient`, scoped to `countIncludePad=false` |
| conv/pool | `MAX_POOL2D_BACKWARD_INPUT` | supported | `SUPPORTED`; dense `FLOAT32` rank-4 NCHW max-pool backward now carries the original source tensor and lowers to MPSGraph `maxPooling2DGradient` with CPU first-max tie parity tests |
| index/scatter/gather | `GATHER`, `GATHER_AXIS`, `GATHER_ND`, `TAKE_ALONG_AXIS` | supported | `SUPPORTED`; dense `FLOAT32`/`BFLOAT16` value/output with static in-bounds `INT32` indices lowers to MPSGraph. `GATHER`/`GATHER_AXIS`/`TAKE_ALONG_AXIS` use `gatherAlongAxis`; `GATHER_ND` uses `gatherNDWithUpdatesTensor` for static non-negative tuple indices, slice suffix outputs, and validated `batch_dims`. |
| index/scatter/gather | `GATHER_GRAD`, `GATHER_AXIS_GRAD`, `TAKE_ALONG_AXIS_GRAD`, `SCATTER_ADD` | supported | `SUPPORTED`; dense `FLOAT32`/`BFLOAT16` gradients/values with static in-bounds `INT32` indices lower to MPSGraph `scatterAlongAxis` add |
| index/scatter/gather | `SCATTER_ELEMENTS`, `SCATTER_ND` | supported | `SUPPORTED`; dense `FLOAT32`/`BFLOAT16` data/updates with static non-negative in-bounds `INT32` indices lower to MPSGraph scatter writes for `NONE`/`ADD`/`MUL`/`MAX`/`MIN`. `SCATTER_ELEMENTS` uses `scatterAlongAxis`; `SCATTER_ND` uses `scatterNDWithDataTensor` with validated `batch_dims` and slice suffix updates. |
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
| loss-adjacent ops | `NLL_LOSS`, `CROSS_ENTROPY_LOSS` | unsupported | `DAG_PRIMITIVE_UNSUPPORTED`; Phase 42 validates dense `FLOAT32` rank 1..4, dense target shape, class axis, and scalar mean-output contract before rejecting |
| loss-adjacent ops | `CROSS_ENTROPY_LOSS_INDICES`, `CROSS_ENTROPY_LOSS_INDICES_GRAD` | unsupported | `UNSUPPORTED_INDEX_SEMANTICS` |
| attention/SDPA | `SCALED_DOT_PRODUCT_ATTENTION` | fallback | `CAPABILITY_MISSING`; Phase 42 validates unmasked, dense external BOOL masked, causal, and external+causal mask modes before rejecting because CUDA direct forward SDPA native/lowered path is not implemented |
| attention/SDPA | `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD` | unsupported | `CAPABILITY_MISSING` |
| conv/pool | `CONV2D`, `CONV2D_GEMM`, `MAX_POOL2D`, `AVG_POOL2D` | unsupported | `CAPABILITY_MISSING`; Phase 42 validates dense `FLOAT32` NCHW/OIHW forward conv/pool contracts, groups, dilation, shape/rank/layout, and average-pool divisor blockers before rejecting |
| conv/pool | `CONV2D_BACKWARD_INPUT`, `CONV2D_BACKWARD_WEIGHT`, `CONV2D_BACKWARD_INPUT_GEMM`, `CONV2D_BACKWARD_WEIGHT_GEMM`, `MAX_POOL2D_BACKWARD_INPUT`, `AVG_POOL2D_BACKWARD_INPUT` | unsupported | `CAPABILITY_MISSING` |
| index/scatter/gather | `GATHER`, `TAKE_ALONG_AXIS` | unsupported | `CAPABILITY_MISSING` |
| index/scatter/gather | `GATHER_GRAD`, `TAKE_ALONG_AXIS_GRAD`, `SCATTER_ADD` | unsupported | `UNSUPPORTED_DUPLICATE_INDEX` |
| compare/bool | `GT`, `GE`, `LT`, `LE`, `EQ`, `NE`, `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT`, `REDUCE_ALL`, `REDUCE_ANY` | unsupported | `UNSUPPORTED_DTYPE` |
| backward-adjacent | `SOFTMAX_GRAD`, `LOG_SOFTMAX_GRAD`, `REDUCE_MIN_GRAD`, `REDUCE_MAX_GRAD`, `MIN_GRAD`, `MAX_GRAD` | supported | `SUPPORTED` |
| fused compound patterns | `FUSED` | unsupported | `CPU_FUSED_OPERATION_UNSUPPORTED`; CPU `Operation.OpType.FUSED` remains CPU-only for Phase 12 |

## Runtime Boundary

The public `Tensor` remains logical and device residency stays in `ExecutionState` and `DeviceBufferBinding`. The matrix does not grant public device tensors, and it does not bypass backend-owned dtype, layout, cost, capability, or ABI checks.

Metal and CUDA coverage is backend-specific. Shared rows describe the common semantic contract, but each backend can still reject a row through its own native capability, dtype, layout, ABI, or buffer-binding gates.

## Phase 40 CUDA parity baseline

Phase 40 adds code-level CUDA-vs-Metal parity reporting on top of this matrix. `GpuBackendParityReporter.cudaAgainstMetal()` is the source for CUDA-vs-Metal parity rows. It compares every Metal row against the corresponding CUDA row and groups gaps by the evidence required before CUDA support can be claimed.

CUDA supported rows still require backend-owned dtype, layout, capability, native execution, CPU parity, trace/report evidence, and gates. A `SUPPORTED` matrix row is not enough on its own when native execution, capability probes, or report evidence are missing.

CUDA unsupported rows for dense loss, SDPA, conv/pool, gather/take, BOOL-producing compute, and selected training/index-gradient families are v1.6 targets or explicit blockers. Capability skip and optional native test skip must not be counted as support.

Phase 41 closes the CUDA dtype/layout/index residency contract without overclaiming compute support. `CUDADTYPE-01` is covered by `CudaDTypeRolePolicy`: CUDA `FLOAT32` remains native compute/output, `INT32` is index-input/residency only, `BOOL` is predicate-input/residency only, `BFLOAT16` is residency-only, and `FLOAT64` remains unsupported. `CUDADTYPE-02` is covered by CUDA layout diagnostics: metadata-only views remain distinct from dense `FLOAT32` GPU materialization, broadcast repair rejects with `CUDA_LAYOUT_BROADCAST_UNSUPPORTED`, and strided native compute rejects with `CUDA_STRIDED_COMPUTE_UNSUPPORTED`. `CUDAINDEX-01` is covered by `CudaPartitionSupport`: CUDA forward `GATHER` / `TAKE_ALONG_AXIS` validate dtype, layout, rank/axis/shape, and static bounds before legal cases end in `CAPABILITY_MISSING` until native CUDA execution exists.

Phase 42 adds CUDA NN operation parity diagnostics without claiming new native execution. `CUDANN-01` is covered by `CudaNnSemantics`: forward SDPA validates dense `FLOAT32` rank-3/4 tensors and classifies unmasked, external BOOL masked, causal, and external+causal mask modes before final `CAPABILITY_MISSING`. `CUDANN-02` is covered by forward conv/pool semantic checks for dense `FLOAT32` NCHW/OIHW contracts, groups, dilation, shape/layout, and average-pool divisor blockers before final `CAPABILITY_MISSING`. `CUDANN-03` is covered by dense-loss checks for dense `FLOAT32` `NLL_LOSS` and `CROSS_ENTROPY_LOSS`; legal dense cases end in `DAG_PRIMITIVE_UNSUPPORTED`, while index-target losses remain `UNSUPPORTED_INDEX_SEMANTICS`.

Phase 43 adds CUDA training/index semantic closure without overclaiming native training support. `CUDATRAIN-01` is covered by target truth and hot-path tests that keep CUDA backward rows independent from forward support and Metal native rows. `CUDATRAIN-02` is covered by `CudaIndexWriteSemantics`: CUDA `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD` validate dense `FLOAT32` values/output, dense static `INT32` indices, rank/axis/shape, and static bounds before final `UNSUPPORTED_DUPLICATE_INDEX`. `CUDATRAIN-03` is covered by `CUDATRAIN` hot-path metadata and training target policies that require visible blockers for unsupported CUDA training rows and keep gradient publication separate from hidden internal CPU materialization.

Phase 33 adds coverage evidence for Metal layout repair. `layout_broadcast_repair_small` requires native buffer execution,
`BROADCAST_GPU_MATERIALIZATION` evidence, and no `CPU_CONSUMER` materialization on the supported hot path. This is not a
claim of universal strided-native compute; direct `STRIDED_NATIVE_COMPUTE` remains a named unsupported route until a
consumer primitive proves that contract.

## DType residency is not native dtype compute

`dtype residency is not native dtype compute`. Phase 16 dtype residency records whether values such as `BFLOAT16`, `INT32`, and `BOOL` can stay represented in runtime storage and trace/report metadata. It does not widen Metal or CUDA native arithmetic beyond the backend role gates.

Metal currently accepts `FLOAT32` and `BFLOAT16` compute/output for the Metal-supported floating operation families, `BOOL` compute/output only for scoped compare/logical/reduction mask families, scoped `INT64` output for `ARGMAX`, and explicit `CAST` only for identity and `FLOAT32 <-> BFLOAT16` cast pairs. Public `ARGMAX` now produces `INT64` indices on Metal instead of relying on the older scoped `INT32` output path. `BOOL` can also be a predicate-style external input for `WHERE`, but reports distinguish that from native BOOL-producing compute through role-specific `dtypeResidency` evidence. CUDA native dense buffer execution remains `FLOAT32` only. Other dtype-role combinations reject with `UNSUPPORTED_DTYPE`, `UNSUPPORTED_CAST_PAIR`, or `RESIDENCY_ONLY_NOT_COMPUTE` and stable details such as `backend=GPU_CUDA role=COMPUTE_OUTPUT dtype=INT32 code=RESIDENCY_ONLY_NOT_COMPUTE`.

Reports use `dtypeResidency` evidence to explain why a region stayed resident, shortened, or exited. A dtype-resident internal value can still be materialized later for a real CPU consumer, graph output, or gradient publication, and that CPU boundary remains reportable.

Metal dtype capability truth is role-specific. `MetalMpsCapabilities` distinguishes storage representability, external
input legality, predicate input legality, native compute legality, native output legality, and operation-specific dtype
support. The optional Metal dtype ABI v3 probes prove that a native shim can describe widened dtype roles; BF16 support
tracks the Metal floating operation coverage, BOOL output is limited to scoped compare/logical/reduction families,
`INT32` is limited to index roles and scoped index outputs, `CAST` is limited by `MetalCastPolicy`, and `FLOAT64`
remains non-native compute/output.

## Phase 30 BF16 Metal contract

Phase 30 introduced native Metal BF16 compute/output, and the current contract extends it to dtype parity with Metal-supported `FLOAT32` floating operation families. The contract is still narrower than "Metal supports BF16 everywhere":

- BF16 storage, external inputs, compute nodes, and outputs use dtype ABI v3 metadata.
- BF16 upload/readback roundtrip is exact raw BF16 storage equality.
- BF16 MATMUL, reduction, conv/pool, loss, index/scatter, normalization, SDPA, and elementwise parity use explicit BF16 numeric tolerance.
- BF16 hot-path targets `mlp_classifier_small_bf16`, `layer_norm_small_bf16`, `rms_norm_small_bf16`, and
  `reduction_chain_small_bf16` require Metal native buffer evidence, dtype residency evidence, zero CPU
  materializations, zero CPU fallback, and zero tensor-array fallback.

Unsupported BF16 and BOOL families must remain visible. BF16 now follows the existing Metal `FLOAT32` operation-family coverage and keeps the same shape/layout/semantic limits; arbitrary BOOL consumers, non-dense/unsupported SDPA mask layouts, generic INT32 compute/output, grouped/dilated conv, and unsupported pooling variants remain rejected or fallback rows until their own semantic, native execution, parity, trace, and regression-gate evidence exists.

## Phase 17 normalization, reduction, and loss-adjacent contract

Phase 17 covers `GPUNORM-01` and `GPUNORM-02` by making normalization, reduction, softmax-ish, conv, and loss-adjacent gaps explicit in the shared Metal/CUDA matrix. The source-of-truth targets are `target=layer_norm_small`, `target=conv2d_resnet_3x3`, and `target=transformer_block_hot_path`.

Phase 23 closes the initial forward reduction subset. `SUM`, `MEAN`, `REDUCE_MIN`, and `REDUCE_MAX` are supported rows for legal dense `FLOAT32` Metal/CUDA regions. Todo 69 extends the Metal-only reduction/scan subset with dense `FLOAT32/BFLOAT16` `REDUCE_PROD`, dense `FLOAT32/BFLOAT16` `ARGMAX` with public `INT64` indices, and dense `FLOAT32/BFLOAT16` `CUMSUM` with static axis/exclusive/reverse metadata. The old scoped `ARGMAX` `INT32` output path is no longer a public graph support claim because `ARGMAX` now returns `INT64`. Phase 24 extends reduction support to legal dense `FLOAT32` `LAYER_NORM` and `RMS_NORM` by lowering them into repeated keep-dims `MEAN`, epsilon scalar add, and elementwise DAG primitives. Phase 31 closes the Metal BOOL compare/logical/reduction subset. Phase 35 promotes scoped Metal `CONV2D`, `CONV2D_GEMM`, `MAX_POOL2D`, and `AVG_POOL2D` forward execution. Phase 58 promotes scoped Metal conv backward, pool backward, and index-target cross entropy forward/backward. Grouped/dilated conv variants and CUDA BOOL outputs remain separate closure targets.

`LOG_SOFTMAX remains lowered as SOFTMAX followed by LOG`. Loss-adjacent support is similarly scoped by row and backend:
Metal dense and index-target loss rows are supported only for their locked contracts, while CUDA loss rows must remain
visible fallback/rejection instead of silent CPU replay.

Forward reductions (`SUM`, `MEAN`, `REDUCE_MIN`, `REDUCE_MAX`, Metal `REDUCE_PROD`, and Metal `ARGMAX`) plus Metal `CUMSUM` scan and legal dense `FLOAT32/BFLOAT16` normalization (`LAYER_NORM`, `RMS_NORM`) are now supported for the `target=layer_norm_small` and BF16 coverage paths. Metal `ARGMAX` is scoped to dense floating inputs and public `INT64` index outputs. Unsupported normalization variants still reject explicitly: unsupported dtype uses `UNSUPPORTED_DTYPE`, non-dense direct inputs use `UNSUPPORTED_LAYOUT`, and invalid rank or tail parameter shape uses `UNSUPPORTED_RANK_OR_SHAPE`. Scoped Metal `CONV2D`, `CONV2D_GEMM`, `CONV2D_BACKWARD_INPUT`, `CONV2D_BACKWARD_WEIGHT`, `CONV2D_BACKWARD_INPUT_GEMM`, and `CONV2D_BACKWARD_WEIGHT_GEMM` are now supported for dense `FLOAT32/BFLOAT16` rank-4 NCHW/OIHW tensors; unsupported Conv2D variants still reject for dtype mismatch, layout, rank/shape, groups, dilation, or metadata encoding limits. Scoped Metal direct `MAX_POOL2D`, `MAX_POOL2D_BACKWARD_INPUT`, `AVG_POOL2D`, and `AVG_POOL2D_BACKWARD_INPUT` are supported for dense `FLOAT32/BFLOAT16` NCHW pooling with `AVG_POOL2D`/backward scoped to `countIncludePad=false`. Dense Metal loss rows are supported for `NLL_LOSS` and dense `CROSS_ENTROPY_LOSS`; index-target `CROSS_ENTROPY_LOSS_INDICES` and `CROSS_ENTROPY_LOSS_INDICES_GRAD` are supported for dense `FLOAT32/BFLOAT16` logits/sampleScale, dense static in-bounds `INT32` targets, public ignore-index masking, and `NONE`/`SUM`/`MEAN` reductions.

## Phase 26 loss and indexing contract

Phase 26 made loss-adjacent and indexing gaps explicit. Phase 37 promoted dense Metal `NLL_LOSS` and dense `CROSS_ENTROPY_LOSS`. Phase 58 promotes index-target Metal cross entropy: `CROSS_ENTROPY_LOSS_INDICES` lowers to MPSGraph softmax/log/gather/reduction, and `CROSS_ENTROPY_LOSS_INDICES_GRAD` lowers to MPSGraph softmax plus `scatterAlongAxis` sample-scale subtraction. `cross_entropy_small` is now a positive Metal native-buffer gate; CUDA keeps the visible `UNSUPPORTED_INDEX_SEMANTICS` blocker.

Phase 32 promotes scoped Metal forward `GATHER` and `TAKE_ALONG_AXIS` from residency-only evidence to native index compute for dense `FLOAT32/BFLOAT16` value/output tensors with `INT32` index inputs whose bounds can be proven from a static leaf index tensor. `TAKE_ALONG_AXIS` maps directly to MPSGraph `gatherAlongAxis`; `GATHER` expands the reduced index tensor on the gathered axis, runs `gatherAlongAxis`, then squeezes the axis back to the public shape. Todo 68 extends this truth to ONNX-style `GATHER_AXIS` for the scoped 1-D-index case by broadcasting the index tensor to the output shape before `gatherAlongAxis`. Todo 73 extends the same read-only index contract to `GATHER_ND` through MPSGraph `gatherNDWithUpdatesTensor`, including slice suffix outputs and validated `batch_dims`; negative ONNX-style tuple indices stay CPU-owned because the Metal contract currently requires static non-negative in-bounds tuples. Unproven or out-of-range bounds reject with `UNSUPPORTED_BOUNDS_CHECK` because MPSGraph out-of-bounds behavior does not match CPU exception semantics. CUDA forward gather/take/gatherND remains `CAPABILITY_MISSING` or `UNSUPPORTED_INDEX_SEMANTICS`, but Phase 41 now validates scoped contracts first so dtype, layout, rank, and bounds mistakes do not collapse into a generic capability gap.

Phase 58 promotes the Metal side of the index-write and index-gradient blocker. `GATHER_GRAD`, `TAKE_ALONG_AXIS_GRAD`, and `SCATTER_ADD` now lower to MPSGraph `scatterAlongAxis` with `MPSGraphScatterModeAdd` for the narrow supported contract: dense `FLOAT32/BFLOAT16` values/output, dense static `INT32` indices, legal rank/axis/shape, and proven in-bounds indices. Todo 68 adds the matching scoped `GATHER_AXIS_GRAD` path for 1-D ONNX gather indices. The ONNX write primitives `SCATTER_ELEMENTS` and `SCATTER_ND` are also Metal-supported for dense `FLOAT32/BFLOAT16` data and updates, static non-negative in-bounds `INT32` indices, `NONE`/`ADD`/`MUL`/`MAX`/`MIN` reductions, and validated duplicate handling. `NONE` rejects duplicate target indices before native execution to match CPU semantics. CUDA keeps the visible `UNSUPPORTED_DUPLICATE_INDEX` / `UNSUPPORTED_INDEX_SEMANTICS` blockers until it has equivalent native duplicate-index and tuple-write evidence.

Todo 68 also adds attributed Metal layout DAG nodes for `SLICE`, `CONCAT`, `PAD`, and `TILE`. `SLICE` is intentionally scoped to static positive unit steps (`step=1`); strided slice compute remains a future contract. `CONCAT` is bounded by the current five-input DAG ABI. `PAD` supports constant scalar pads with non-negative static before/after vectors, and `TILE` supports positive static repeats. These rows are layout ownership improvements, not a claim that arbitrary strided runtime compute is native.

`native support is not implied by matrix text alone`. A supported row means lowering, legality, native execution, trace/report, and parity evidence exist for the legal scoped case. A fallback or unsupported row means the blocker is recognized, diagnosed, and kept visible for planning and reports.

## Phase 27 conv/pool and BOOL output contract

Phase 27 expands the matrix surface for conv/pool and BOOL-producing operations without claiming native GPU execution prematurely. Metal and CUDA now list every targeted conv/pool op explicitly: `CONV2D`, `CONV2D_GEMM`, `CONV2D_BACKWARD_INPUT`, `CONV2D_BACKWARD_WEIGHT`, `CONV2D_BACKWARD_INPUT_GEMM`, `CONV2D_BACKWARD_WEIGHT_GEMM`, `MAX_POOL2D`, `MAX_POOL2D_BACKWARD_INPUT`, `AVG_POOL2D`, and `AVG_POOL2D_BACKWARD_INPUT`. Phase 35 promotes scoped Metal `CONV2D`, `CONV2D_GEMM`, `MAX_POOL2D`, and `AVG_POOL2D` rows to supported after DAG lowering, MPSGraph buffer execution, and prepared execution evidence. Phase 58 promotes scoped Metal conv backward plus max/average-pool backward rows after native MPSGraph buffer execution and CPU parity tests.

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
stable visible blockers. CUDA conv/pool, CUDA BOOL-producing outputs, CUDA loss/index blockers, and CUDA forward SDPA
must surface reason evidence such as `CAPABILITY_MISSING`, `UNSUPPORTED_DTYPE`, `UNSUPPORTED_INDEX_SEMANTICS`, operation
family names, or target names. They do not count as native coverage closure until backend-owned execution and parity
evidence exist.

Phase 36 originally added `scatter_index_gradient_small` to the visible-blocker set. Phase 58 promotes the Metal side:
Metal reports must now provide hard native evidence for `SCATTER_ADD`, `SCATTER_ELEMENTS`, `SCATTER_ND`,
`GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD`, while CUDA still must surface `UNSUPPORTED_DUPLICATE_INDEX`,
`UNSUPPORTED_INDEX_SEMANTICS`, or one of the named index-write/gradient operations rather than passing because
forward `GATHER` / `TAKE_ALONG_AXIS` coverage exists.
The native forward target remains `gather_take_small`, which separately requires `INT32` dtype residency and native
buffer evidence.

Suite reports include `coverageDeltaVsBaseline` so reviewers can compare trace-derived coverage counters against the
v1.4 pre-closure baseline without using raw latency medians as proof. The relevant evidence fields are
`maxSelectedRegionLength`, `loweredPrimitiveCount`, `nativeBufferStepCount`, `tensorArrayStepCount`,
`cpuFallbackStepCount`, `cpuMaterializationCount`, `internalCpuMaterializationCount`,
`gradientPublicationMaterializationCount`, `fallbackCount`, `deviceHandoffCount`, `targetCoverageGates`,
`nativeEvidence`, `capabilitySkipped`, and visible reason-code lists.

## Phase 38 training/backward contract

Training support is per backward operation, not inherited from forward support. `GpuTargetCoverageTruth` now carries
explicit backward rows for `SOFTMAX_GRAD`, `LOG_SOFTMAX_GRAD`, `REDUCE_MIN_GRAD`, `REDUCE_MAX_GRAD`, `MIN_GRAD`,
`MAX_GRAD`, `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD`, conv/pool backward variants, index gradients, and index-target
loss gradients. Metal marks only the backward rows with existing prepared-execution, native buffer trace, and parity
evidence as `NATIVE_EXECUTABLE`. `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD` is scoped to FLOAT32 rank-3/4 SDPA
producers and supports unmasked, dense external BOOL masked, causal, and external+causal mask modes. CUDA rows remain
matrix-supported-only or explicit rejection until CUDA-native evidence exists.

Forward conv/pool, gather/take, dense loss, BOOL compare, and SDPA support does not imply backward support. Metal
currently marks scoped conv/pool backward, index gradients/scatter, and index-target loss gradients as native
executable only where native buffer execution and parity evidence exist. CUDA still reports conv/pool backward as
`CAPABILITY_MISSING`, index gradients and scatter as `UNSUPPORTED_DUPLICATE_INDEX`, and index-target loss gradients
as `UNSUPPORTED_INDEX_SEMANTICS`.

Phase 38 training gates add explicit `training_*` hot-path targets. Supported targets allow bounded
`GRADIENT_PUBLICATION` materialization, require zero `internalCpuMaterializationCount`, and still fail tensor-array
bridge replay, CPU fallback, or shorter selected regions. `training_cross_entropy_small` is a hard native Metal
target after Phase 58 and remains a visible CUDA blocker. Unsupported training targets such as
`training_transformer_block_hot_path` pass only when the report exposes stable blocker evidence for SDPA backward.

Phase 43 applies the same training target discipline to CUDA. CUDA `training_reduction_chain_small` and
`training_layer_norm_small` require native buffer evidence and zero hidden internal CPU materialization, while CUDA
`training_transformer_block_hot_path`, `training_dense_loss_small`, `training_cross_entropy_small`, and
`scatter_index_gradient_small` remain visible blocker targets until native execution and parity evidence exist.
