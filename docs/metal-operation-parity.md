# Metal Operation Parity Matrix

Generated from `MetalOperationParityMatrix`; do not hand-edit status rows.

| Operation | CPU kernel | Metal coverage | Planner supported | DAG lowerable | Native MPSGraph mapped | Buffer executable | Custom route eligible | CPU fallback only | Reason | Note |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| ADD | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| SUB | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| MUL | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| DIV | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| MIN | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | MPSGraph-first elementwise parity gap closed through native accelerator DAG mapping |
| MAX | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | MPSGraph-first elementwise parity gap closed through native accelerator DAG mapping |
| GT | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | greater-than compare has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| GE | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | greater-or-equal compare has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| LT | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | less-than compare has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| LE | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | less-or-equal compare has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| EQ | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | equal compare has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| NE | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | not-equal compare has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| LOGICAL_AND | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | logical AND has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| LOGICAL_OR | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | logical OR has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| LOGICAL_NOT | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | logical NOT has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| MIN_GRAD | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG backward-adjacent path |
| MAX_GRAD | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG backward-adjacent path |
| REDUCE_MIN | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG forward reduce-min path; target=reduction_chain_small |
| REDUCE_MAX | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG forward reduce-max path; target=reduction_chain_small |
| REDUCE_PROD | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native Metal MPSGraph product reduction path for dense FLOAT32/BFLOAT16 inputs; target=reduction_chain_small |
| CUMSUM | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native Metal MPSGraph cumulative sum path for dense FLOAT32/BFLOAT16 inputs with static axis/exclusive/reverse metadata |
| ARGMAX | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native Metal MPSGraph argmax path for dense FLOAT32/BFLOAT16 inputs with public INT64 index outputs |
| REDUCE_ALL | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | BOOL all reduction has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| REDUCE_ANY | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | BOOL any reduction has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| SOFTMAX | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG softmax path; target=transformer_block_hot_path |
| SOFTMAX_GRAD | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG backward-adjacent path |
| LOG_SOFTMAX | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | lowered as SOFTMAX followed by LOG using existing accelerator DAG primitives; target=transformer_block_hot_path |
| LOG_SOFTMAX_GRAD | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG backward-adjacent path |
| NLL_LOSS | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | dense FLOAT32/BFLOAT16 rank 1..4 NLL loss lowers to target multiply, all-axis SUM, and scalar mean scaling; target=loss_dense_small target=transformer_block_hot_path |
| CROSS_ENTROPY_LOSS | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | dense FLOAT32/BFLOAT16 rank 1..4 cross entropy lowers to SOFTMAX, LOG, target multiply, all-axis SUM, and scalar mean scaling; target=loss_dense_small target=transformer_block_hot_path |
| CROSS_ENTROPY_LOSS_INDICES | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal index-target cross entropy lowers to MPSGraph softmax/log/gather/reduction with dense FLOAT32/BFLOAT16 logits, dense INT32 in-bounds targets, ignore-index masking, and NONE/SUM/MEAN reductions; target=transformer_block_hot_path |
| CROSS_ENTROPY_LOSS_INDICES_GRAD | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal index-target cross entropy gradient lowers to MPSGraph softmax plus scatterAlongAxis sample-scale subtraction with dense FLOAT32/BFLOAT16 logits/sampleScale and dense INT32 in-bounds targets; target=transformer_block_hot_path |
| REDUCE_MIN_GRAD | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG backward-adjacent path |
| REDUCE_MAX_GRAD | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG backward-adjacent path |
| GATHER | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | forward gather lowers to Metal gatherAlongAxis with expanded INT32 indices; scoped to dense FLOAT32/BFLOAT16 value input and static in-bounds INT32 index input; target=gather_take_small |
| GATHER_GRAD | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal gather gradient lowers to MPSGraph scatterAlongAxis add with dense FLOAT32/BFLOAT16 gradients, static in-bounds INT32 indices, and duplicate accumulation on device; target=scatter_index_gradient_small |
| GATHER_AXIS | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | ONNX-style gatherAxis lowers to Metal gatherAlongAxis with broadcast INT32 indices for dense FLOAT32/BFLOAT16 value input and static in-bounds 1-D index input; target=gather_take_small |
| GATHER_AXIS_GRAD | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal gatherAxis gradient lowers to MPSGraph scatterAlongAxis add with broadcast INT32 indices, dense FLOAT32/BFLOAT16 gradients, and duplicate accumulation on device; target=scatter_index_gradient_small |
| GATHER_ND | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal gather-nd lowers to MPSGraph gatherNDWithUpdatesTensor for dense FLOAT32/BFLOAT16 values, dense static non-negative in-bounds INT32 tuple indices, slice suffix outputs, and validated batch_dims |
| GATHER_ND_GRAD | no | unsupported | no | no | no | no | no | yes | UNSUPPORTED_INDEX_SEMANTICS | gather-nd gradient remains CPU-owned until tuple-index duplicate accumulation and batch_dims semantics are proven for Metal |
| TAKE_ALONG_AXIS | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | take-along-axis lowers to Metal gatherAlongAxis with INT32 indices; scoped to dense FLOAT32/BFLOAT16 value input and static in-bounds INT32 index input; target=gather_take_small |
| TAKE_ALONG_AXIS_GRAD | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal take-along-axis gradient lowers to MPSGraph scatterAlongAxis add with dense FLOAT32/BFLOAT16 gradients, static in-bounds INT32 indices, and duplicate accumulation on device; target=scatter_index_gradient_small |
| SCATTER_ADD | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal scatter-add lowers to MPSGraph scatterAlongAxis add onto the base tensor with dense FLOAT32/BFLOAT16 values, static in-bounds INT32 indices, and duplicate accumulation on device; target=scatter_index_gradient_small |
| SCATTER_AXIS_ADD | yes | unsupported | no | no | no | no | no | yes | UNSUPPORTED_INDEX_SEMANTICS | scatter-axis-add remains CPU-owned until rank-changing gather inverse writes and duplicate-index accumulation are proven for Metal |
| SCATTER_ELEMENTS | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal scatter-elements lowers to MPSGraph scatterAlongAxis with dense FLOAT32/BFLOAT16 data/updates, static non-negative in-bounds INT32 indices, and NONE/ADD/MUL/MAX/MIN reduction parity |
| SCATTER_ND | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal scatter-nd lowers to MPSGraph scatterNDWithDataTensor with dense FLOAT32/BFLOAT16 data/updates, static non-negative in-bounds INT32 tuple indices, slice suffix updates, validated batch_dims, and NONE/ADD/MUL/MAX/MIN reduction parity |
| SCALED_DOT_PRODUCT_ATTENTION | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | direct FLOAT32/BFLOAT16 rank-3/4 native MPSGraph primitive SDPA DAG supports unmasked, dense external BOOL masked, causal, and external+causal effective mask modes; target=transformer_block_hot_path target=masked_sdpa_small |
| SCALED_DOT_PRODUCT_ATTENTION_BACKWARD | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal backward SDPA is represented by the shared accelerator DAG for unmasked, dense external BOOL masked, causal, and external+causal 3/4-input SDPA producers |
| SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | attention weights publication lowers from the producer SDPA descriptor to a native Metal softmax(QK^T * scale + mask) DAG without CPU cache materialization; target=masked_sdpa_small |
| LINEAR | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG matmul/linear path |
| CONV2D | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal direct FLOAT32/BFLOAT16 dense NCHW/OIHW Conv2D forward lowers to MPSGraph convolution2D; scoped to groups=1, dilation=1, stride/padding, and optional bias; target=conv2d_resnet_3x3 |
| MAX_POOL2D | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal direct FLOAT32/BFLOAT16 dense NCHW MAX_POOL2D forward lowers to MPSGraph maxPooling2D; scoped to compact kernel/stride/padding metadata and CPU tie behavior parity; target=max_pool2d_small |
| AVG_POOL2D | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal direct FLOAT32/BFLOAT16 dense NCHW AVG_POOL2D forward lowers to MPSGraph avgPooling2D; scoped to countIncludePad=false and compact kernel/stride/padding metadata; target=avg_pool2d_small |
| LAYER_NORM | yes | supported | yes | yes | no | yes | no | no | SUPPORTED | lowered as repeated MEAN plus elementwise normalization DAG with epsilon scalar; target=layer_norm_small |
| RMS_NORM | yes | supported | yes | yes | no | yes | no | no | SUPPORTED | lowered as repeated MEAN plus elementwise RMS normalization DAG with epsilon scalar; target=rms_norm_small |
| WHERE | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| MATMUL | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG matmul/linear path |
| NEG | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| INV | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| LOG | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| EXP | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| FAST_EXP | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| ERF | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | MPSGraph-first elementwise parity gap closed through native accelerator DAG mapping |
| TANH | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| FAST_TANH | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| POW | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | MPSGraph-first elementwise parity gap closed through native accelerator DAG mapping |
| POW_TENSOR | yes | unsupported | no | no | no | no | no | yes | CAPABILITY_MISSING | Metal tensor-exponent POW_TENSOR remains unsupported until native accelerator DAG execution maps binary power semantics |
| SQRT | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| ABS | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| FLOOR | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | MPSGraph-first elementwise parity gap closed through native accelerator DAG mapping |
| CEIL | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | MPSGraph-first elementwise parity gap closed through native accelerator DAG mapping |
| SIGN | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | MPSGraph-first elementwise parity gap closed through native accelerator DAG mapping |
| MUL_SCALAR | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| SUM | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG forward sum reduction path; target=reduction_chain_small |
| MEAN | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG forward mean reduction path; target=reduction_chain_small |
| RELU | yes | supported | yes | yes | yes | yes | yes | no | SUPPORTED | native accelerator DAG elementwise path |
| CLAMP_MIN | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| CLAMP_MAX | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| SIGMOID | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| CONTIGUOUS | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | layout/view-adjacent accelerator DAG metadata or materialization path |
| RESHAPE | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | layout/view-adjacent accelerator DAG metadata or materialization path |
| EXPAND | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal MPSGraph layout path maps broadcast EXPAND and single-index SELECT into native accelerator DAG shape ops |
| SELECT | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal MPSGraph layout path maps broadcast EXPAND and single-index SELECT into native accelerator DAG shape ops |
| SLICE | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal MPSGraph layout path supports dense FLOAT32/BFLOAT16 static slice, concat, constant pad, and tile descriptors |
| SLICE_BACKWARD | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal SLICE_BACKWARD supports static dense FLOAT32/BFLOAT16 step=1 backward layout writes by lowering to zero-fill pad with explicit before/after attributes |
| CONCAT | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal MPSGraph layout path supports dense FLOAT32/BFLOAT16 static slice, concat, constant pad, and tile descriptors |
| PAD | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal MPSGraph layout path supports dense FLOAT32/BFLOAT16 static slice, concat, constant pad, and tile descriptors |
| TILE | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal MPSGraph layout path supports dense FLOAT32/BFLOAT16 static slice, concat, constant pad, and tile descriptors |
| UNFOLD_AXIS | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | GPU_METAL UNFOLD_AXIS supports scoped dense FLOAT32/BFLOAT16 native axis sliding-window materialization with step=1; wider dtype/rank coverage is rejected by planner semantics |
| UNFOLD2D | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | GPU_METAL UNFOLD2D supports scoped dense FLOAT32/BFLOAT16 native im2col lowering with stride=1 and dilation=1; wider geometry is rejected by planner semantics |
| FOLD2D | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | GPU_METAL FOLD2D supports scoped dense FLOAT32/BFLOAT16 native col2im accumulation lowering with stride=1 and dilation=1; wider geometry is rejected by planner semantics |
| PERMUTE | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | layout/view-adjacent accelerator DAG metadata or materialization path |
| EXPAND_DIMS | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | layout/view-adjacent accelerator DAG metadata or materialization path |
| SQUEEZE | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | layout/view-adjacent accelerator DAG metadata or materialization path |
| CAST | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal CAST supports scoped identity casts plus FLOAT32 <-> BFLOAT16 conversion through explicit cast-pair policy; FLOAT64, runtime INT64, and general BOOL/INT32 numeric casts remain unsupported |
| CONST_SCALAR | no | unsupported | no | no | no | no | no | yes | UNSUPPORTED_OPERATION | CONST_SCALAR is an internal CPU fused-plan scalar node, not a public standalone GPU compute op; GPU DAG lowering carries scalar values as primitive metadata |
| NOOP | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | layout/view-adjacent accelerator DAG metadata or materialization path |
| FUSED | yes | unsupported | no | no | no | no | no | yes | CPU_FUSED_OPERATION_UNSUPPORTED | CPU Operation.OpType.FUSED remains CPU-only for Phase 12; GPU compound partitions lower from normal graph operations |
