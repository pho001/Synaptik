# Metal Operation Parity Matrix

Generated from `MetalOperationParityMatrix`; do not hand-edit status rows.

| Operation | CPU kernel | CPU fusable | Metal coverage | Planner supported | DAG lowerable | Native MPSGraph mapped | Buffer executable | Custom route eligible | CPU fallback only | Reason | Note |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| ADD | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| SUB | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| MUL | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| DIV | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| MIN | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | MPSGraph-first elementwise parity gap closed through native accelerator DAG mapping |
| MAX | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | MPSGraph-first elementwise parity gap closed through native accelerator DAG mapping |
| GT | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | greater-than compare has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| GE | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | greater-or-equal compare has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| LT | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | less-than compare has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| LE | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | less-or-equal compare has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| EQ | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | equal compare has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| NE | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | not-equal compare has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| LOGICAL_AND | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | logical AND has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| LOGICAL_OR | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | logical OR has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| LOGICAL_NOT | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | logical NOT has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| MIN_GRAD | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG backward-adjacent path |
| MAX_GRAD | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG backward-adjacent path |
| REDUCE_MIN | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG forward reduce-min path; target=reduction_chain_small |
| REDUCE_MAX | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG forward reduce-max path; target=reduction_chain_small |
| REDUCE_PROD | yes | no | unsupported | no | no | no | no | no | yes | UNSUPPORTED_OPERATION | operation is not in the checked-in GPU lowering coverage matrix |
| CUMSUM | yes | no | unsupported | no | no | no | no | no | yes | UNSUPPORTED_OPERATION | operation is not in the checked-in GPU lowering coverage matrix |
| ARGMAX | yes | no | unsupported | no | no | no | no | no | yes | UNSUPPORTED_OPERATION | operation is not in the checked-in GPU lowering coverage matrix |
| REDUCE_ALL | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | BOOL all reduction has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| REDUCE_ANY | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | BOOL any reduction has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate |
| SOFTMAX | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG softmax path; target=transformer_block_hot_path |
| SOFTMAX_GRAD | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG backward-adjacent path |
| LOG_SOFTMAX | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | lowered as SOFTMAX followed by LOG using existing accelerator DAG primitives; target=transformer_block_hot_path |
| LOG_SOFTMAX_GRAD | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG backward-adjacent path |
| NLL_LOSS | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | dense FLOAT32/BFLOAT16 rank 1..4 NLL loss lowers to target multiply, all-axis SUM, and scalar mean scaling; target=loss_dense_small target=transformer_block_hot_path |
| CROSS_ENTROPY_LOSS | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | dense FLOAT32/BFLOAT16 rank 1..4 cross entropy lowers to SOFTMAX, LOG, target multiply, all-axis SUM, and scalar mean scaling; target=loss_dense_small target=transformer_block_hot_path |
| CROSS_ENTROPY_LOSS_INDICES | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal index-target cross entropy lowers to MPSGraph softmax/log/gather/reduction with dense FLOAT32/BFLOAT16 logits, dense INT32 in-bounds targets, ignore-index masking, and NONE/SUM/MEAN reductions; target=transformer_block_hot_path |
| CROSS_ENTROPY_LOSS_INDICES_GRAD | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal index-target cross entropy gradient lowers to MPSGraph softmax plus scatterAlongAxis sample-scale subtraction with dense FLOAT32/BFLOAT16 logits/sampleScale and dense INT32 in-bounds targets; target=transformer_block_hot_path |
| REDUCE_MIN_GRAD | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG backward-adjacent path |
| REDUCE_MAX_GRAD | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG backward-adjacent path |
| GATHER | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | forward gather lowers to Metal gatherAlongAxis with expanded INT32 indices; scoped to dense FLOAT32/BFLOAT16 value input and static in-bounds INT32 index input; target=gather_take_small |
| GATHER_GRAD | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal gather gradient lowers to MPSGraph scatterAlongAxis add with dense FLOAT32/BFLOAT16 gradients, static in-bounds INT32 indices, and duplicate accumulation on device; target=scatter_index_gradient_small |
| GATHER_AXIS | yes | no | unsupported | no | no | no | no | no | yes | UNSUPPORTED_OPERATION | operation is not in the checked-in GPU lowering coverage matrix |
| GATHER_AXIS_GRAD | yes | no | unsupported | no | no | no | no | no | yes | UNSUPPORTED_OPERATION | operation is not in the checked-in GPU lowering coverage matrix |
| GATHER_ND | yes | no | unsupported | no | no | no | no | no | yes | UNSUPPORTED_INDEX_SEMANTICS | gather-nd remains CPU-owned until tuple-index read, slice suffix addressing, and static bounds checks are proven for Metal |
| GATHER_ND_GRAD | yes | no | unsupported | no | no | no | no | no | yes | UNSUPPORTED_INDEX_SEMANTICS | gather-nd gradient remains CPU-owned until tuple-index duplicate accumulation and batch_dims semantics are proven for Metal |
| TAKE_ALONG_AXIS | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | take-along-axis lowers to Metal gatherAlongAxis with INT32 indices; scoped to dense FLOAT32/BFLOAT16 value input and static in-bounds INT32 index input; target=gather_take_small |
| TAKE_ALONG_AXIS_GRAD | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal take-along-axis gradient lowers to MPSGraph scatterAlongAxis add with dense FLOAT32/BFLOAT16 gradients, static in-bounds INT32 indices, and duplicate accumulation on device; target=scatter_index_gradient_small |
| SCATTER_ADD | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal scatter-add lowers to MPSGraph scatterAlongAxis add onto the base tensor with dense FLOAT32/BFLOAT16 values, static in-bounds INT32 indices, and duplicate accumulation on device; target=scatter_index_gradient_small |
| SCATTER_ELEMENTS | yes | no | unsupported | no | no | no | no | no | yes | UNSUPPORTED_INDEX_SEMANTICS | scatter-elements remains CPU-owned until rank-preserving write, reduction, and duplicate-index policies are proven for Metal |
| SCATTER_ND | yes | no | unsupported | no | no | no | no | no | yes | UNSUPPORTED_INDEX_SEMANTICS | scatter-nd remains CPU-owned until tuple-index write, slice update, reduction, and duplicate-index policies are proven for Metal |
| SCALED_DOT_PRODUCT_ATTENTION | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | direct FLOAT32/BFLOAT16 rank-3/4 native MPSGraph primitive SDPA DAG supports unmasked, dense external BOOL masked, causal, and external+causal effective mask modes; target=transformer_block_hot_path target=masked_sdpa_small |
| SCALED_DOT_PRODUCT_ATTENTION_BACKWARD | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal backward SDPA is represented by the shared accelerator DAG for unmasked, dense external BOOL masked, causal, and external+causal 3/4-input SDPA producers |
| SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | attention weights publication lowers from the producer SDPA descriptor to a native Metal softmax(QK^T * scale + mask) DAG without CPU cache materialization; target=masked_sdpa_small |
| LINEAR | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG matmul/linear path |
| CONV2D | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal direct FLOAT32/BFLOAT16 dense NCHW/OIHW Conv2D forward lowers to MPSGraph convolution2D; scoped to groups=1, dilation=1, stride/padding, and optional bias; target=conv2d_resnet_3x3 |
| CONV2D_GEMM | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal CONV2D_GEMM descriptor preserves original NCHW/OIHW tensors and routes to the same MPSGraph convolution2D primitive as direct CONV2D; scoped to groups=1, dilation=1, stride/padding, and optional bias; target=conv2d_resnet_3x3 |
| CONV2D_BACKWARD_INPUT | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal CONV2D_BACKWARD_INPUT lowers to MPSGraph convolution2DDataGradient for dense FLOAT32/BFLOAT16 rank-4 NCHW/OIHW tensors; scoped to groups=1 and dilation=1 |
| CONV2D_BACKWARD_WEIGHT | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal CONV2D_BACKWARD_WEIGHT lowers to MPSGraph convolution2DWeightsGradient for dense FLOAT32/BFLOAT16 rank-4 NCHW/OIHW tensors; scoped to groups=1 and dilation=1 |
| CONV2D_BACKWARD_INPUT_GEMM | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal CONV2D_BACKWARD_INPUT_GEMM preserves original conv descriptor and routes to the same MPSGraph convolution2DDataGradient primitive |
| CONV2D_BACKWARD_WEIGHT_GEMM | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal CONV2D_BACKWARD_WEIGHT_GEMM preserves original conv descriptor and routes to the same MPSGraph convolution2DWeightsGradient primitive |
| MAX_POOL2D | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal direct FLOAT32/BFLOAT16 dense NCHW MAX_POOL2D forward lowers to MPSGraph maxPooling2D; scoped to compact kernel/stride/padding metadata and CPU tie behavior parity; target=max_pool2d_small |
| MAX_POOL2D_BACKWARD_INPUT | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal MAX_POOL2D_BACKWARD_INPUT lowers to MPSGraph maxPooling2DGradient with the original source tensor for dense FLOAT32/BFLOAT16 rank-4 NCHW tensors; scoped to first-max tie parity |
| AVG_POOL2D | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal direct FLOAT32/BFLOAT16 dense NCHW AVG_POOL2D forward lowers to MPSGraph avgPooling2D; scoped to countIncludePad=false and compact kernel/stride/padding metadata; target=avg_pool2d_small |
| AVG_POOL2D_BACKWARD_INPUT | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal AVG_POOL2D_BACKWARD_INPUT lowers to MPSGraph avgPooling2DGradient for dense FLOAT32/BFLOAT16 rank-4 NCHW tensors; scoped to countIncludePad=false |
| LAYER_NORM | yes | no | supported | yes | yes | no | yes | no | no | SUPPORTED | lowered as repeated MEAN plus elementwise normalization DAG with epsilon scalar; target=layer_norm_small |
| RMS_NORM | yes | no | supported | yes | yes | no | yes | no | no | SUPPORTED | lowered as repeated MEAN plus elementwise RMS normalization DAG with epsilon scalar; target=rms_norm_small |
| WHERE | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| MATMUL | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG matmul/linear path |
| NEG | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| INV | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| LOG | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| EXP | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| FAST_EXP | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| TANH | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| FAST_TANH | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| POW | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | MPSGraph-first elementwise parity gap closed through native accelerator DAG mapping |
| SQRT | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| ABS | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| MUL_SCALAR | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| SUM | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG forward sum reduction path; target=reduction_chain_small |
| MEAN | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG forward mean reduction path; target=reduction_chain_small |
| RELU | yes | yes | supported | yes | yes | yes | yes | yes | no | SUPPORTED | native accelerator DAG elementwise path |
| CLAMP_MIN | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| CLAMP_MAX | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| SIGMOID | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED | native accelerator DAG elementwise path |
| CONTIGUOUS | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | layout/view-adjacent accelerator DAG metadata or materialization path |
| RESHAPE | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | layout/view-adjacent accelerator DAG metadata or materialization path |
| EXPAND | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal MPSGraph layout path maps broadcast EXPAND and single-index SELECT into native accelerator DAG shape ops |
| SELECT | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | Metal MPSGraph layout path maps broadcast EXPAND and single-index SELECT into native accelerator DAG shape ops |
| SLICE | yes | no | unsupported | no | no | no | no | no | yes | UNSUPPORTED_OPERATION | operation is not in the checked-in GPU lowering coverage matrix |
| SLICE_GRAD | yes | no | unsupported | no | no | no | no | no | yes | UNSUPPORTED_OPERATION | operation is not in the checked-in GPU lowering coverage matrix |
| CONCAT | yes | no | unsupported | no | no | no | no | no | yes | UNSUPPORTED_OPERATION | operation is not in the checked-in GPU lowering coverage matrix |
| PAD | yes | no | unsupported | no | no | no | no | no | yes | UNSUPPORTED_OPERATION | operation is not in the checked-in GPU lowering coverage matrix |
| TILE | yes | no | unsupported | no | no | no | no | no | yes | UNSUPPORTED_OPERATION | operation is not in the checked-in GPU lowering coverage matrix |
| PERMUTE | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | layout/view-adjacent accelerator DAG metadata or materialization path |
| EXPAND_DIMS | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | layout/view-adjacent accelerator DAG metadata or materialization path |
| SQUEEZE | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | layout/view-adjacent accelerator DAG metadata or materialization path |
| CAST | yes | no | unsupported | no | no | no | no | no | yes | UNSUPPORTED_OPERATION | operation is not in the checked-in GPU lowering coverage matrix |
| CONST_SCALAR | no | no | unsupported | no | no | no | no | no | yes | UNSUPPORTED_OPERATION | CONST_SCALAR is an internal CPU fused-plan scalar node, not a public standalone GPU compute op; GPU DAG lowering carries scalar values as primitive metadata |
| NOOP | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED | layout/view-adjacent accelerator DAG metadata or materialization path |
| FUSED | yes | no | unsupported | no | no | no | no | no | yes | CPU_FUSED_OPERATION_UNSUPPORTED | CPU Operation.OpType.FUSED remains CPU-only for Phase 12; GPU compound regions lower from normal graph operations |
