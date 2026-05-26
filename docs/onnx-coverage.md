# ONNX Coverage Report

Generated from `OnnxCoverageMatrix`; do not hand-edit status rows.

## Summary

- Import: supported=79, partial=0, unsupported=1
- Export: supported=74, partial=0, unsupported=6
- CPU: supported=79, partial=0, unsupported=1
- Metal: supported=68, partial=9, unsupported=3
- CUDA: supported=35, partial=9, unsupported=36
- Round-trip evidence: round_trip_tested=73, explicitly_classified=1, import_only_tested=5, rejection_tested=1, not_applicable=0
- Limitation categories: none=30, static_semantic_limit=22, static_attribute_limit=21, multi_output_limit=2, runtime_shape_limit=4, data_dependent_shape_limit=1

## Matrix

| ONNX op | Synaptik mapping | Import | Export | CPU | Metal | CUDA | Round-trip evidence | Mapped op types | Limitation category | Limitations |
|---|---|---|---|---|---|---|---|---|---|---|
| Add | add | supported | supported | supported | supported | supported | round_trip_tested | ADD | none |  |
| Sub | sub | supported | supported | supported | supported | supported | round_trip_tested | SUB | none |  |
| Mul | mul | supported | supported | supported | supported | supported | round_trip_tested | MUL | none |  |
| Div | div | supported | supported | supported | supported | supported | round_trip_tested | DIV | none |  |
| Min | min | supported | supported | supported | supported | unsupported | round_trip_tested | MIN | static_semantic_limit | variadic import lowers to a left-associated binary chain |
| Max | max | supported | supported | supported | supported | unsupported | round_trip_tested | MAX | static_semantic_limit | variadic import lowers to a left-associated binary chain |
| Pow | scalar or tensor-exponent pow | supported | supported | supported | supported | unsupported | round_trip_tested | POW, POW_TENSOR | static_semantic_limit | tensor exponent maps to CPU POW_TENSOR; accelerator native coverage remains scalar-pow scoped |
| Neg | neg | supported | supported | supported | supported | supported | round_trip_tested | NEG | none |  |
| Abs | abs | supported | supported | supported | supported | supported | round_trip_tested | ABS | none |  |
| Relu | relu | supported | supported | supported | supported | supported | round_trip_tested | RELU | none |  |
| LeakyRelu | where(x >= 0, x, alpha * x) | supported | supported | supported | partial | partial | round_trip_tested | GE, WHERE, MUL, MUL_SCALAR | static_semantic_limit | canonical export recognizes the conservative where/ge/scale composition |
| Elu | where(x >= 0, x, alpha * (exp(x) - 1)) | supported | supported | supported | partial | partial | round_trip_tested | GE, WHERE, EXP, SUB, MUL, MUL_SCALAR | static_semantic_limit | canonical export recognizes the conservative where/ge/exp/sub/scale composition |
| HardSigmoid | clip(alpha * x + beta, 0, 1) | supported | supported | supported | partial | partial | round_trip_tested | MUL, MUL_SCALAR, ADD, CLAMP_MIN, CLAMP_MAX | static_semantic_limit | canonical export recognizes clampMin(clampMax(alpha*x + beta, 1), 0) |
| Softplus | log(exp(x) + 1) | supported | supported | supported | partial | partial | round_trip_tested | EXP, ADD, LOG | static_semantic_limit | canonical export recognizes log(exp(x) + 1); not numerically stabilized with thresholding yet |
| Tanh | tanh | supported | supported | supported | supported | supported | round_trip_tested | TANH | none |  |
| Sigmoid | sigmoid | supported | supported | supported | supported | supported | round_trip_tested | SIGMOID | none |  |
| Exp | exp | supported | supported | supported | supported | supported | round_trip_tested | EXP | none |  |
| Log | log | supported | supported | supported | supported | supported | round_trip_tested | LOG | none |  |
| Sqrt | sqrt | supported | supported | supported | supported | supported | round_trip_tested | SQRT | none |  |
| Reciprocal | inv | supported | supported | supported | supported | supported | round_trip_tested | INV | none |  |
| Erf | erf | supported | supported | supported | supported | unsupported | round_trip_tested | ERF | none |  |
| Floor | floor | supported | supported | supported | supported | unsupported | round_trip_tested | FLOOR | none |  |
| Ceil | ceil | supported | supported | supported | supported | unsupported | round_trip_tested | CEIL | none |  |
| Sign | sign | supported | supported | supported | supported | unsupported | round_trip_tested | SIGN | none |  |
| Equal | eq | supported | supported | supported | supported | unsupported | round_trip_tested | EQ | none |  |
| Greater | gt | supported | supported | supported | supported | unsupported | round_trip_tested | GT | none |  |
| GreaterOrEqual | ge | supported | supported | supported | supported | unsupported | round_trip_tested | GE | none |  |
| Less | lt | supported | supported | supported | supported | unsupported | round_trip_tested | LT | none |  |
| LessOrEqual | le | supported | supported | supported | supported | unsupported | round_trip_tested | LE | none |  |
| Not | logical_not | supported | supported | supported | supported | unsupported | round_trip_tested | LOGICAL_NOT | none |  |
| And | logical_and | supported | supported | supported | supported | unsupported | round_trip_tested | LOGICAL_AND | none |  |
| Or | logical_or | supported | supported | supported | supported | unsupported | round_trip_tested | LOGICAL_OR | none |  |
| Where | where | supported | supported | supported | supported | supported | round_trip_tested | WHERE | none |  |
| Identity | pass-through | supported | unsupported | supported | partial | partial | import_only_tested | NOOP | static_semantic_limit | import-only pass-through; export preserves the producer op instead |
| Clip | clampMin/clampMax | supported | supported | supported | supported | supported | round_trip_tested | CLAMP_MIN, CLAMP_MAX | static_semantic_limit | scalar bounds only |
| Cast | cast | supported | supported | supported | supported | unsupported | round_trip_tested | CAST | static_semantic_limit | CPU import/export supports runtime INT64 tensors; Metal runtime cast is scoped to identity and FLOAT32 <-> BFLOAT16 pairs |
| MatMul | matmul | supported | supported | supported | supported | supported | round_trip_tested | MATMUL | none |  |
| Gemm | matmul plus optional bias/scale | supported | supported | supported | supported | supported | round_trip_tested | MATMUL, ADD | static_attribute_limit | rank-2 transposition flags and scalar alpha/beta only |
| Conv | conv2d | supported | supported | supported | supported | unsupported | round_trip_tested | PAD, CONV2D | static_attribute_limit | rank-4 NCHW/OIHW, static attributes; asymmetric pads import as explicit Pad + Conv |
| MaxPool | maxPool2d | supported | supported | supported | supported | unsupported | round_trip_tested | MAX_POOL2D | static_attribute_limit | rank-4 NCHW, static attributes; ceil_mode=1 is CPU/import supported and accelerator-native unsupported |
| AveragePool | avgPool2d | supported | supported | supported | supported | unsupported | round_trip_tested | AVG_POOL2D | static_attribute_limit | rank-4 NCHW, static attributes; ceil_mode=1 is CPU/import supported; Metal native row is scoped to count_include_pad=false and ceil_mode=false |
| Col2Im | fold2d | supported | supported | supported | supported | supported | round_trip_tested | FOLD2D | static_attribute_limit | 2-D static image_shape/block_shape only; symmetric spatial pads map to Window2dOptions; Synaptik UNFOLD2D/UNFOLD_AXIS have no standard ONNX op |
| LayerNormalization | layerNorm | supported | supported | supported | supported | supported | round_trip_tested | LAYER_NORM | static_attribute_limit | single output; axis must select trailing normalized dimensions |
| BatchNormalization | batchNorm with external statistics | supported | supported | supported | partial | partial | round_trip_tested | SUB, DIV, MUL, ADD | multi_output_limit | single-output inference form only; export recognizes canonical external-statistics batchNorm graphs |
| Transpose | permute | supported | supported | supported | supported | supported | round_trip_tested | PERMUTE | none |  |
| Reshape | reshape | supported | supported | supported | supported | supported | round_trip_tested | RESHAPE | static_attribute_limit | constant target shape |
| Flatten | reshape | supported | supported | supported | supported | supported | round_trip_tested | RESHAPE | static_attribute_limit | static axis reshape |
| Expand | expand | supported | supported | supported | supported | unsupported | round_trip_tested | EXPAND | static_attribute_limit | constant target shape |
| Pad | pad | supported | supported | supported | supported | unsupported | round_trip_tested | PAD | static_attribute_limit | constant mode, static non-negative pads, scalar constant value |
| Tile | tile | supported | supported | supported | supported | unsupported | round_trip_tested | TILE | static_attribute_limit | constant positive repeats |
| ConstantOfShape | constant leaf materialization | supported | unsupported | supported | partial | partial | import_only_tested |  | runtime_shape_limit | import-time static shape input only; no first-class runtime operation |
| Range | shape constant or constant tensor leaf | supported | unsupported | supported | partial | partial | import_only_tested |  | runtime_shape_limit | import-time static scalar inputs only; runtime data-dependent length is unsupported |
| Squeeze | squeeze | supported | supported | supported | supported | supported | round_trip_tested | SQUEEZE | static_attribute_limit | constant axes |
| Unsqueeze | expand_dims | supported | supported | supported | supported | supported | round_trip_tested | EXPAND_DIMS | static_attribute_limit | constant axes |
| Slice | slice | supported | supported | supported | supported | unsupported | round_trip_tested | SLICE | static_attribute_limit | constant positive-step slice parameters |
| Concat | concat | supported | supported | supported | supported | unsupported | round_trip_tested | CONCAT | static_attribute_limit | runtime tensors or shape-only axis-0 constants |
| Split | slice per output | supported | supported | supported | supported | unsupported | round_trip_tested | SLICE | multi_output_limit | multi-output export is supported for graph-output slice siblings that exactly cover one axis |
| Shape | shape constant | supported | unsupported | supported | unsupported | unsupported | import_only_tested |  | runtime_shape_limit | import-time static shape plumbing only |
| Size | size constant | supported | unsupported | supported | unsupported | unsupported | import_only_tested |  | runtime_shape_limit | import-time static shape plumbing only |
| Gather | gatherAxis or shape gather | supported | supported | supported | supported | unsupported | round_trip_tested | GATHER_AXIS | static_attribute_limit | runtime mapping uses GATHER_AXIS; shape-only mapping is axis-0 only |
| GatherElements | takeAlongAxis | supported | supported | supported | supported | unsupported | round_trip_tested | TAKE_ALONG_AXIS | static_attribute_limit | runtime indices may be INT32 or INT64 on CPU/ONNX; accelerator native rows remain backend-scoped |
| GatherND | gatherNd | supported | supported | supported | supported | unsupported | round_trip_tested | GATHER_ND | static_attribute_limit | runtime indices may be INT32 or INT64 on CPU/ONNX; batch_dims supported; Metal native row is scoped to static non-negative in-bounds INT32 tuple indices |
| ScatterElements | scatterElements | supported | supported | supported | supported | unsupported | round_trip_tested | SCATTER_ELEMENTS | static_attribute_limit | runtime indices may be INT32 or INT64 on CPU/ONNX; forward reductions none/add/mul/max/min; backward only none/add; accelerator native rows remain backend-scoped |
| ScatterND | scatterNd | supported | supported | supported | supported | unsupported | round_trip_tested | SCATTER_ND | static_attribute_limit | runtime indices may be INT32 or INT64 on CPU/ONNX; forward reductions none/add/mul/max/min; backward only none/add; accelerator native rows remain backend-scoped |
| ReduceSum | sum | supported | supported | supported | supported | supported | round_trip_tested | SUM | static_semantic_limit | multi-axis reductions are imported as repeated single-axis reductions |
| ReduceMean | mean | supported | supported | supported | supported | supported | round_trip_tested | MEAN | static_semantic_limit | multi-axis reductions are imported as repeated single-axis reductions |
| ReduceMax | reduce_max | supported | supported | supported | supported | supported | round_trip_tested | REDUCE_MAX | static_semantic_limit | multi-axis reductions are imported as repeated single-axis reductions |
| ReduceMin | reduce_min | supported | supported | supported | supported | supported | round_trip_tested | REDUCE_MIN | static_semantic_limit | multi-axis reductions are imported as repeated single-axis reductions |
| ReduceProd | reduce_prod | supported | supported | supported | supported | unsupported | round_trip_tested | REDUCE_PROD | static_semantic_limit | multi-axis reductions are imported as repeated single-axis reductions |
| ReduceL1 | abs then ReduceSum | supported | supported | supported | supported | supported | round_trip_tested | ABS, SUM | static_semantic_limit | canonical export recognizes abs(x) followed by single-axis sum |
| ReduceL2 | mul then ReduceSum then sqrt | supported | supported | supported | supported | supported | round_trip_tested | MUL, SUM, SQRT | static_semantic_limit | canonical export recognizes sqrt(sum(x*x)) single-axis pattern |
| ReduceLogSum | ReduceSum then log | supported | supported | supported | supported | supported | round_trip_tested | SUM, LOG | static_semantic_limit | canonical export recognizes log(sum(x)) single-axis pattern |
| ReduceLogSumExp | exp then ReduceSum then log | supported | supported | supported | supported | supported | round_trip_tested | EXP, SUM, LOG | static_semantic_limit | canonical export recognizes log(sum(exp(x))) single-axis pattern; not numerically stabilized with max-shift yet |
| ArgMax | argMax | supported | supported | supported | supported | unsupported | round_trip_tested | ARGMAX | static_semantic_limit | output is INT64; select_last_index=0/1 supported on CPU/import/export; accelerator native rows remain first-index scoped unless backend tie policy is proven |
| CumSum | cumSum | supported | supported | supported | supported | unsupported | round_trip_tested | CUMSUM | static_attribute_limit | axis input must be a static INT64/INT32 scalar constant; BOOL input is unsupported; Metal supports FLOAT32/BFLOAT16 inputs |
| GlobalAveragePool | repeated mean over spatial axes | supported | supported | supported | supported | supported | round_trip_tested | MEAN | static_semantic_limit | canonical export recognizes rank-4 keepdims spatial mean chain |
| Softmax | softmax | supported | supported | supported | supported | supported | round_trip_tested | SOFTMAX | none |  |
| LogSoftmax | logSoftmax | supported | supported | supported | supported | supported | round_trip_tested | LOG_SOFTMAX | none |  |
| Constant | initializer tensor or shape constant | supported | supported | supported | partial | partial | explicitly_classified |  | static_semantic_limit | leaf tensors can export as graph inputs, initializers, trainable inputs, or Constant nodes according to OnnxLeafTensorPolicy |
| NonZero | unsupported dynamic-shape op | unsupported | unsupported | unsupported | unsupported | unsupported | rejection_tested |  | data_dependent_shape_limit | runtime output shape depends on input values; requires a dynamic-shape execution model |
