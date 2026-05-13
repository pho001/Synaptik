# ONNX Coverage Report

Generated from `OnnxCoverageMatrix`; do not hand-edit status rows.

## Summary

- Import: supported=78, partial=0, unsupported=1
- Export: supported=70, partial=1, unsupported=8
- CPU: supported=78, partial=0, unsupported=1
- Metal: supported=50, partial=9, unsupported=20
- CUDA: supported=34, partial=9, unsupported=36
- Round-trip evidence: round_trip_tested=70, explicitly_classified=1, import_only_tested=7, rejection_tested=1, not_applicable=0

## Matrix

| ONNX op | Synaptik mapping | Import | Export | CPU | Metal | CUDA | Round-trip evidence | Mapped op types | Limitations |
|---|---|---|---|---|---|---|---|---|---|
| Add | add | supported | supported | supported | supported | supported | round_trip_tested | ADD |  |
| Sub | sub | supported | supported | supported | supported | supported | round_trip_tested | SUB |  |
| Mul | mul | supported | supported | supported | supported | supported | round_trip_tested | MUL |  |
| Div | div | supported | supported | supported | supported | supported | round_trip_tested | DIV |  |
| Min | min | supported | supported | supported | supported | unsupported | round_trip_tested | MIN | binary form only |
| Max | max | supported | supported | supported | supported | unsupported | round_trip_tested | MAX | binary form only |
| Pow | scalar-exponent pow | supported | supported | supported | supported | unsupported | round_trip_tested | POW | tensor exponent is unsupported |
| Neg | neg | supported | supported | supported | supported | supported | round_trip_tested | NEG |  |
| Abs | abs | supported | supported | supported | supported | supported | round_trip_tested | ABS |  |
| Relu | relu | supported | supported | supported | supported | supported | round_trip_tested | RELU |  |
| LeakyRelu | where(x >= 0, x, alpha * x) | supported | supported | supported | partial | partial | round_trip_tested | GE, WHERE, MUL, MUL_SCALAR | canonical export recognizes the conservative where/ge/scale composition |
| Elu | where(x >= 0, x, alpha * (exp(x) - 1)) | supported | supported | supported | partial | partial | round_trip_tested | GE, WHERE, EXP, SUB, MUL, MUL_SCALAR | canonical export recognizes the conservative where/ge/exp/sub/scale composition |
| HardSigmoid | clip(alpha * x + beta, 0, 1) | supported | supported | supported | partial | partial | round_trip_tested | MUL, MUL_SCALAR, ADD, CLAMP_MIN, CLAMP_MAX | canonical export recognizes clampMin(clampMax(alpha*x + beta, 1), 0) |
| Softplus | log(exp(x) + 1) | supported | supported | supported | partial | partial | round_trip_tested | EXP, ADD, LOG | canonical export recognizes log(exp(x) + 1); not numerically stabilized with thresholding yet |
| Tanh | tanh | supported | supported | supported | supported | supported | round_trip_tested | TANH |  |
| Sigmoid | sigmoid | supported | supported | supported | supported | supported | round_trip_tested | SIGMOID |  |
| Exp | exp | supported | supported | supported | supported | supported | round_trip_tested | EXP |  |
| Log | log | supported | supported | supported | supported | supported | round_trip_tested | LOG |  |
| Sqrt | sqrt | supported | supported | supported | supported | supported | round_trip_tested | SQRT |  |
| Reciprocal | inv | supported | supported | supported | supported | supported | round_trip_tested | INV |  |
| Erf | erf | supported | supported | supported | unsupported | unsupported | round_trip_tested | ERF |  |
| Floor | floor | supported | supported | supported | unsupported | unsupported | round_trip_tested | FLOOR |  |
| Ceil | ceil | supported | supported | supported | unsupported | unsupported | round_trip_tested | CEIL |  |
| Sign | sign | supported | supported | supported | unsupported | unsupported | round_trip_tested | SIGN |  |
| Equal | eq | supported | supported | supported | supported | unsupported | round_trip_tested | EQ |  |
| Greater | gt | supported | supported | supported | supported | unsupported | round_trip_tested | GT |  |
| GreaterOrEqual | ge | supported | supported | supported | supported | unsupported | round_trip_tested | GE |  |
| Less | lt | supported | supported | supported | supported | unsupported | round_trip_tested | LT |  |
| LessOrEqual | le | supported | supported | supported | supported | unsupported | round_trip_tested | LE |  |
| Not | logical_not | supported | supported | supported | supported | unsupported | round_trip_tested | LOGICAL_NOT |  |
| And | logical_and | supported | supported | supported | supported | unsupported | round_trip_tested | LOGICAL_AND |  |
| Or | logical_or | supported | supported | supported | supported | unsupported | round_trip_tested | LOGICAL_OR |  |
| Where | where | supported | supported | supported | supported | supported | round_trip_tested | WHERE |  |
| Identity | pass-through | supported | unsupported | supported | partial | partial | import_only_tested | NOOP | import-only pass-through; export preserves the producer op instead |
| Clip | clampMin/clampMax | supported | supported | supported | supported | supported | round_trip_tested | CLAMP_MIN, CLAMP_MAX | scalar bounds only |
| Cast | cast | supported | supported | supported | unsupported | unsupported | round_trip_tested | CAST | runtime INT64 is unsupported |
| MatMul | matmul | supported | supported | supported | supported | supported | round_trip_tested | MATMUL |  |
| Gemm | matmul plus optional bias/scale | supported | supported | supported | supported | supported | round_trip_tested | MATMUL, ADD | rank-2 transposition flags and scalar alpha/beta only |
| Conv | conv2d | supported | supported | supported | supported | unsupported | round_trip_tested | CONV2D | rank-4 NCHW/OIHW, symmetric spatial pads, static attributes |
| MaxPool | maxPool2d | supported | supported | supported | supported | unsupported | round_trip_tested | MAX_POOL2D | rank-4 NCHW, static attributes, ceil_mode=0 |
| AveragePool | avgPool2d | supported | supported | supported | supported | unsupported | round_trip_tested | AVG_POOL2D | rank-4 NCHW, static attributes, ceil_mode=0; Metal native row is scoped to count_include_pad=false |
| LayerNormalization | layerNorm | supported | supported | supported | supported | supported | round_trip_tested | LAYER_NORM | single output; axis must select trailing normalized dimensions |
| BatchNormalization | batchNorm with external statistics | supported | unsupported | supported | partial | partial | import_only_tested | SUB, DIV, MUL, ADD | single-output inference form only; export has no first-class batchNorm descriptor |
| Transpose | permute | supported | supported | supported | supported | supported | round_trip_tested | PERMUTE |  |
| Reshape | reshape | supported | supported | supported | supported | supported | round_trip_tested | RESHAPE | constant target shape |
| Flatten | reshape | supported | supported | supported | supported | supported | round_trip_tested | RESHAPE | static axis reshape |
| Expand | expand | supported | supported | supported | supported | unsupported | round_trip_tested | EXPAND | constant target shape |
| Pad | pad | supported | supported | supported | unsupported | unsupported | round_trip_tested | PAD | constant mode, static non-negative pads, scalar constant value |
| Tile | tile | supported | supported | supported | unsupported | unsupported | round_trip_tested | TILE | constant positive repeats |
| ConstantOfShape | constant leaf materialization | supported | unsupported | supported | partial | partial | import_only_tested |  | import-time static shape input only; no first-class runtime operation |
| Range | shape constant or constant tensor leaf | supported | unsupported | supported | partial | partial | import_only_tested |  | import-time static scalar inputs only; runtime data-dependent length is unsupported |
| Squeeze | squeeze | supported | supported | supported | supported | supported | round_trip_tested | SQUEEZE | constant axes |
| Unsqueeze | expand_dims | supported | supported | supported | supported | supported | round_trip_tested | EXPAND_DIMS | constant axes |
| Slice | slice | supported | supported | supported | unsupported | unsupported | round_trip_tested | SLICE | constant positive-step slice parameters |
| Concat | concat | supported | supported | supported | unsupported | unsupported | round_trip_tested | CONCAT | runtime tensors or shape-only axis-0 constants |
| Split | slice per output | supported | unsupported | supported | unsupported | unsupported | import_only_tested | SLICE | import-only multi-output lowering with static split sizes |
| Shape | shape constant | supported | unsupported | supported | unsupported | unsupported | import_only_tested |  | import-time static shape plumbing only |
| Size | size constant | supported | unsupported | supported | unsupported | unsupported | import_only_tested |  | import-time static shape plumbing only |
| Gather | gatherAxis or shape gather | supported | supported | supported | unsupported | unsupported | round_trip_tested | GATHER_AXIS | runtime mapping uses GATHER_AXIS; shape-only mapping is axis-0 only |
| GatherElements | takeAlongAxis | supported | supported | supported | supported | unsupported | round_trip_tested | TAKE_ALONG_AXIS | runtime indices are INT32 |
| GatherND | gatherNd | supported | supported | supported | unsupported | unsupported | round_trip_tested | GATHER_ND | runtime indices are INT32; batch_dims supported |
| ScatterElements | scatterElements | supported | supported | supported | unsupported | unsupported | round_trip_tested | SCATTER_ELEMENTS | forward reductions none/add/mul/max/min; backward only none/add |
| ScatterND | scatterNd | supported | supported | supported | unsupported | unsupported | round_trip_tested | SCATTER_ND | forward reductions none/add/mul/max/min; backward only none/add |
| ReduceSum | sum | supported | supported | supported | supported | supported | round_trip_tested | SUM | multi-axis reductions are imported as repeated single-axis reductions |
| ReduceMean | mean | supported | supported | supported | supported | supported | round_trip_tested | MEAN | multi-axis reductions are imported as repeated single-axis reductions |
| ReduceMax | reduce_max | supported | supported | supported | supported | supported | round_trip_tested | REDUCE_MAX | multi-axis reductions are imported as repeated single-axis reductions |
| ReduceMin | reduce_min | supported | supported | supported | supported | supported | round_trip_tested | REDUCE_MIN | multi-axis reductions are imported as repeated single-axis reductions |
| ReduceProd | reduce_prod | supported | supported | supported | unsupported | unsupported | round_trip_tested | REDUCE_PROD | multi-axis reductions are imported as repeated single-axis reductions |
| ReduceL1 | abs then ReduceSum | supported | supported | supported | supported | supported | round_trip_tested | ABS, SUM | canonical export recognizes abs(x) followed by single-axis sum |
| ReduceL2 | mul then ReduceSum then sqrt | supported | supported | supported | supported | supported | round_trip_tested | MUL, SUM, SQRT | canonical export recognizes sqrt(sum(x*x)) single-axis pattern |
| ReduceLogSum | ReduceSum then log | supported | supported | supported | supported | supported | round_trip_tested | SUM, LOG | canonical export recognizes log(sum(x)) single-axis pattern |
| ReduceLogSumExp | exp then ReduceSum then log | supported | supported | supported | supported | supported | round_trip_tested | EXP, SUM, LOG | canonical export recognizes log(sum(exp(x))) single-axis pattern; not numerically stabilized with max-shift yet |
| ArgMax | argMax | supported | supported | supported | unsupported | unsupported | round_trip_tested | ARGMAX | output is INT32 because runtime INT64 tensors are unsupported; select_last_index=0 only |
| CumSum | cumSum | supported | supported | supported | unsupported | unsupported | round_trip_tested | CUMSUM | axis input must be a static INT64/INT32 scalar constant; BOOL input is unsupported |
| GlobalAveragePool | repeated mean over spatial axes | supported | supported | supported | supported | supported | round_trip_tested | MEAN | canonical export recognizes rank-4 keepdims spatial mean chain |
| Softmax | softmax | supported | supported | supported | supported | supported | round_trip_tested | SOFTMAX |  |
| LogSoftmax | logSoftmax | supported | supported | supported | supported | supported | round_trip_tested | LOG_SOFTMAX |  |
| Constant | initializer tensor or shape constant | supported | partial | supported | partial | partial | explicitly_classified |  | export usually serializes leaves as graph inputs or initializers rather than Constant nodes |
| NonZero | unsupported dynamic-shape op | unsupported | unsupported | unsupported | unsupported | unsupported | rejection_tested |  | runtime output shape depends on input values; requires a dynamic-shape execution model |
