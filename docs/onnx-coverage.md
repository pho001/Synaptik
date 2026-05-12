# ONNX Coverage Report

Generated from `OnnxCoverageMatrix`; do not hand-edit status rows.

## Summary

- Import: supported=78, partial=0, unsupported=1
- Export: supported=65, partial=1, unsupported=13
- CPU: supported=78, partial=0, unsupported=1
- Metal: supported=50, partial=9, unsupported=20
- CUDA: supported=34, partial=9, unsupported=36
- Round-trip evidence: round_trip_tested=17, explicitly_classified=49, import_only_tested=12, rejection_tested=1, not_applicable=0

## Matrix

| ONNX op | Synaptik mapping | Import | Export | CPU | Metal | CUDA | Round-trip evidence | Mapped op types | Limitations |
|---|---|---|---|---|---|---|---|---|---|
| Add | add | supported | supported | supported | supported | supported | round_trip_tested | ADD |  |
| Sub | sub | supported | supported | supported | supported | supported | explicitly_classified | SUB |  |
| Mul | mul | supported | supported | supported | supported | supported | explicitly_classified | MUL |  |
| Div | div | supported | supported | supported | supported | supported | explicitly_classified | DIV |  |
| Min | min | supported | supported | supported | supported | unsupported | explicitly_classified | MIN | binary form only |
| Max | max | supported | supported | supported | supported | unsupported | explicitly_classified | MAX | binary form only |
| Pow | scalar-exponent pow | supported | supported | supported | supported | unsupported | explicitly_classified | POW | tensor exponent is unsupported |
| Neg | neg | supported | supported | supported | supported | supported | explicitly_classified | NEG |  |
| Abs | abs | supported | supported | supported | supported | supported | explicitly_classified | ABS |  |
| Relu | relu | supported | supported | supported | supported | supported | explicitly_classified | RELU |  |
| LeakyRelu | where(x >= 0, x, alpha * x) | supported | supported | supported | partial | partial | round_trip_tested | GE, WHERE, MUL, MUL_SCALAR | canonical export recognizes the conservative where/ge/scale composition |
| Elu | where(x >= 0, x, alpha * (exp(x) - 1)) | supported | supported | supported | partial | partial | round_trip_tested | GE, WHERE, EXP, SUB, MUL, MUL_SCALAR | canonical export recognizes the conservative where/ge/exp/sub/scale composition |
| HardSigmoid | clip(alpha * x + beta, 0, 1) | supported | supported | supported | partial | partial | round_trip_tested | MUL, MUL_SCALAR, ADD, CLAMP_MIN, CLAMP_MAX | canonical export recognizes clampMin(clampMax(alpha*x + beta, 1), 0) |
| Softplus | log(exp(x) + 1) | supported | supported | supported | partial | partial | round_trip_tested | EXP, ADD, LOG | canonical export recognizes log(exp(x) + 1); not numerically stabilized with thresholding yet |
| Tanh | tanh | supported | supported | supported | supported | supported | explicitly_classified | TANH |  |
| Sigmoid | sigmoid | supported | supported | supported | supported | supported | explicitly_classified | SIGMOID |  |
| Exp | exp | supported | supported | supported | supported | supported | explicitly_classified | EXP |  |
| Log | log | supported | supported | supported | supported | supported | explicitly_classified | LOG |  |
| Sqrt | sqrt | supported | supported | supported | supported | supported | explicitly_classified | SQRT |  |
| Reciprocal | inv | supported | supported | supported | supported | supported | round_trip_tested | INV |  |
| Erf | erf | supported | supported | supported | unsupported | unsupported | round_trip_tested | ERF |  |
| Floor | floor | supported | supported | supported | unsupported | unsupported | round_trip_tested | FLOOR |  |
| Ceil | ceil | supported | supported | supported | unsupported | unsupported | round_trip_tested | CEIL |  |
| Sign | sign | supported | supported | supported | unsupported | unsupported | round_trip_tested | SIGN |  |
| Equal | eq | supported | supported | supported | supported | unsupported | explicitly_classified | EQ |  |
| Greater | gt | supported | supported | supported | supported | unsupported | explicitly_classified | GT |  |
| GreaterOrEqual | ge | supported | supported | supported | supported | unsupported | explicitly_classified | GE |  |
| Less | lt | supported | supported | supported | supported | unsupported | explicitly_classified | LT |  |
| LessOrEqual | le | supported | supported | supported | supported | unsupported | explicitly_classified | LE |  |
| Not | logical_not | supported | supported | supported | supported | unsupported | explicitly_classified | LOGICAL_NOT |  |
| And | logical_and | supported | supported | supported | supported | unsupported | explicitly_classified | LOGICAL_AND |  |
| Or | logical_or | supported | supported | supported | supported | unsupported | explicitly_classified | LOGICAL_OR |  |
| Where | where | supported | supported | supported | supported | supported | explicitly_classified | WHERE |  |
| Identity | pass-through | supported | unsupported | supported | partial | partial | import_only_tested | NOOP | import-only pass-through; export preserves the producer op instead |
| Clip | clampMin/clampMax | supported | supported | supported | supported | supported | explicitly_classified | CLAMP_MIN, CLAMP_MAX | scalar bounds only |
| Cast | cast | supported | supported | supported | unsupported | unsupported | explicitly_classified | CAST | runtime INT64 is unsupported |
| MatMul | matmul | supported | supported | supported | supported | supported | round_trip_tested | MATMUL |  |
| Gemm | matmul plus optional bias/scale | supported | supported | supported | supported | supported | explicitly_classified | MATMUL, ADD | rank-2 transposition flags and scalar alpha/beta only |
| Conv | conv2d | supported | supported | supported | supported | unsupported | round_trip_tested | CONV2D | rank-4 NCHW/OIHW, symmetric spatial pads, static attributes |
| MaxPool | maxPool2d | supported | supported | supported | supported | unsupported | explicitly_classified | MAX_POOL2D | rank-4 NCHW, static attributes, ceil_mode=0 |
| AveragePool | avgPool2d | supported | supported | supported | supported | unsupported | explicitly_classified | AVG_POOL2D | rank-4 NCHW, static attributes, ceil_mode=0; Metal native row is scoped to count_include_pad=false |
| LayerNormalization | layerNorm | supported | supported | supported | supported | supported | round_trip_tested | LAYER_NORM | single output; axis must select trailing normalized dimensions |
| BatchNormalization | batchNorm with external statistics | supported | unsupported | supported | partial | partial | import_only_tested | SUB, DIV, MUL, ADD | single-output inference form only; export has no first-class batchNorm descriptor |
| Transpose | permute | supported | supported | supported | supported | supported | explicitly_classified | PERMUTE |  |
| Reshape | reshape | supported | supported | supported | supported | supported | explicitly_classified | RESHAPE | constant target shape |
| Flatten | reshape | supported | supported | supported | supported | supported | explicitly_classified | RESHAPE | static axis reshape |
| Expand | expand | supported | supported | supported | supported | unsupported | explicitly_classified | EXPAND | constant target shape |
| Pad | pad | supported | supported | supported | unsupported | unsupported | explicitly_classified | PAD | constant mode, static non-negative pads, scalar constant value |
| Tile | tile | supported | supported | supported | unsupported | unsupported | explicitly_classified | TILE | constant positive repeats |
| ConstantOfShape | constant leaf materialization | supported | unsupported | supported | partial | partial | import_only_tested |  | import-time static shape input only; no first-class runtime operation |
| Range | shape constant or constant tensor leaf | supported | unsupported | supported | partial | partial | import_only_tested |  | import-time static scalar inputs only; runtime data-dependent length is unsupported |
| Squeeze | squeeze | supported | supported | supported | supported | supported | explicitly_classified | SQUEEZE | constant axes |
| Unsqueeze | expand_dims | supported | supported | supported | supported | supported | explicitly_classified | EXPAND_DIMS | constant axes |
| Slice | slice | supported | supported | supported | unsupported | unsupported | explicitly_classified | SLICE | constant positive-step slice parameters |
| Concat | concat | supported | supported | supported | unsupported | unsupported | explicitly_classified | CONCAT | runtime tensors or shape-only axis-0 constants |
| Split | slice per output | supported | unsupported | supported | unsupported | unsupported | import_only_tested | SLICE | import-only multi-output lowering with static split sizes |
| Shape | shape constant | supported | unsupported | supported | unsupported | unsupported | import_only_tested |  | import-time static shape plumbing only |
| Size | size constant | supported | unsupported | supported | unsupported | unsupported | import_only_tested |  | import-time static shape plumbing only |
| Gather | gatherAxis or shape gather | supported | supported | supported | unsupported | unsupported | explicitly_classified | GATHER_AXIS | runtime mapping uses GATHER_AXIS; shape-only mapping is axis-0 only |
| GatherElements | takeAlongAxis | supported | supported | supported | supported | unsupported | round_trip_tested | TAKE_ALONG_AXIS | runtime indices are INT32 |
| GatherND | gatherNd | supported | supported | supported | unsupported | unsupported | round_trip_tested | GATHER_ND | runtime indices are INT32; batch_dims supported |
| ScatterElements | scatterElements | supported | supported | supported | unsupported | unsupported | round_trip_tested | SCATTER_ELEMENTS | forward reductions none/add/mul/max/min; backward only none/add |
| ScatterND | scatterNd | supported | supported | supported | unsupported | unsupported | round_trip_tested | SCATTER_ND | forward reductions none/add/mul/max/min; backward only none/add |
| ReduceSum | sum | supported | supported | supported | supported | supported | explicitly_classified | SUM | multi-axis reductions are imported as repeated single-axis reductions |
| ReduceMean | mean | supported | supported | supported | supported | supported | explicitly_classified | MEAN | multi-axis reductions are imported as repeated single-axis reductions |
| ReduceMax | reduce_max | supported | supported | supported | supported | supported | explicitly_classified | REDUCE_MAX | multi-axis reductions are imported as repeated single-axis reductions |
| ReduceMin | reduce_min | supported | supported | supported | supported | supported | explicitly_classified | REDUCE_MIN | multi-axis reductions are imported as repeated single-axis reductions |
| ReduceProd | reduce_prod | supported | supported | supported | unsupported | unsupported | explicitly_classified | REDUCE_PROD | multi-axis reductions are imported as repeated single-axis reductions |
| ReduceL1 | abs then ReduceSum | supported | unsupported | supported | supported | supported | import_only_tested | ABS, SUM | import-only composed lowering; export recognition is not implemented |
| ReduceL2 | mul then ReduceSum then sqrt | supported | unsupported | supported | supported | supported | import_only_tested | MUL, SUM, SQRT | import-only composed lowering; final sqrt is applied after all requested axes |
| ReduceLogSum | ReduceSum then log | supported | unsupported | supported | supported | supported | import_only_tested | SUM, LOG | import-only composed lowering; final log is applied after all requested axes |
| ReduceLogSumExp | exp then ReduceSum then log | supported | unsupported | supported | supported | supported | import_only_tested | EXP, SUM, LOG | import-only direct lowering; not numerically stabilized with max-shift yet |
| ArgMax | argMax | supported | supported | supported | unsupported | unsupported | explicitly_classified | ARGMAX | output is INT32 because runtime INT64 tensors are unsupported; select_last_index=0 only |
| CumSum | cumSum | supported | supported | supported | unsupported | unsupported | explicitly_classified | CUMSUM | axis input must be a static INT64/INT32 scalar constant; BOOL input is unsupported |
| GlobalAveragePool | repeated mean over spatial axes | supported | unsupported | supported | supported | supported | import_only_tested | MEAN | import-only static rank >= 3 lowering to MEAN |
| Softmax | softmax | supported | supported | supported | supported | supported | explicitly_classified | SOFTMAX |  |
| LogSoftmax | logSoftmax | supported | supported | supported | supported | supported | explicitly_classified | LOG_SOFTMAX |  |
| Constant | initializer tensor or shape constant | supported | partial | supported | partial | partial | explicitly_classified |  | export usually serializes leaves as graph inputs or initializers rather than Constant nodes |
| NonZero | unsupported dynamic-shape op | unsupported | unsupported | unsupported | unsupported | unsupported | rejection_tested |  | runtime output shape depends on input values; requires a dynamic-shape execution model |
