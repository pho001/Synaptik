# Task 0056: Deterministic Dual-Route Catalog and Fixed Route Identity

## Status

Ready

Authorized from clean production/documentation base
`5b80d37c585f7e8579856678b84cf4d52cbb2397`. Task 0055 is Complete at `ddeff1b2`;
the signed-32 native program-image remediation is Complete at `ce569b65`, and current ABI/schema
documentation remediations are Complete at `486ff493` plus `5b80d37c`.

The normative [per-wire route-evidence audit](0056-route-evidence-audit.md) is part of this task.
It resolves every row to an exact installed-SDK selector or finite selector composition and to the
current Model enum source plus bounded signature; broad reason codes are summaries, not evidence
substitutes.

This task is the sole Metal implementation frontier. It is an internal architecture/catalog task,
not an operation-capability task and not a performance experiment.

## Change class

Class C. The implementation changes backend-private route metadata, preparation identity, candidate
plumbing, route-forcing test seams, and tests. It may refactor current Metal Java/native internals
only as required to make the two route families consume one validated schema-13 plan. It changes no
public API, Model semantics, capability row, native export name, program wire, data-type wire, or
attribute wire.

## Goal

Install the smallest deterministic internal architecture in which every prepared Metal partition:

1. is lowered once to the existing bounded schema-13 program and validated through the existing
   capability, topology, descriptor, attribute, lifecycle, buffer, and publication gates;
2. embeds exactly one immutable backend-private route identity before resource declaration and
   finalization;
3. reaches either the MPSGraph family or the custom-kernel family only after shared validation;
4. cannot switch, retry, fall back, benchmark, or autotune at Runtime; and
5. can be forced by package-private tests so each implemented route is checked independently against
   the Model-owned result set.

The task also lands one exhaustive, executable catalog for all 115 current schema-13 operation
wires. The catalog records structural route potential, not capability or correctness approval.
Existing production routes and the exact `34 capability-true + 81 capability-false = 115` count stay
fixed.

## Non-goals

- no new capability-true operation, data type, shape, attribute form, gradient, or profile domain;
- no new MPSGraph selector, composition, custom kernel, oracle, device run, or numerical proof;
- no ABI-6, schema-14, export, public configuration, environment variable, system property, public
  route enum, or public route-forcing API;
- no hot route lookup, route switch, retry, fallback, benchmark, telemetry-driven selection,
  persistent cache, threshold, winner, or autotuner;
- no use of local timing as qualification, route choice, cache identity, or tuning authority;
- no reinterpretation of the blocked Task 0026/0027/0030-0037/0051/0053 records;
- no claim that an SDK selector is Model-correct merely because its shape/cardinality can be made to
  match; and
- no bitwise equality requirement between two legal implementations when the selected numerical
  profile permits multiple observable results.

## Current source and SDK ground truth

The implementation must re-audit these exact current sources before editing and fail planning drift
rather than silently changing this ledger:

- `MetalMpsGraphProgram.NodeKind` is schema 13 with stable operation wires `1..115`, attribute wires
  `0..41`, type wires `1..6`, rank `0..16`, and variable `0..N` input / `1..N` output cardinality.
- `MetalNegPartitionPreparer` currently fixes `CUSTOM_TASK0052` when any wire `20..34` occurs,
  otherwise fixes eligible singleton `NEG` to `CUSTOM_SINGLE_NEG`, otherwise uses `MPSGRAPH`.
- `MetalNegPreparationPlan` already retains one closed route value. It is the starting point, not a
  reason to add a second route-selection object.
- `MetalNegTuningBatch.Candidate` uses stable route wires 1 `CUSTOM_SINGLE_NEG`, 2 `MPSGRAPH`, and 3
  `CUSTOM_TASK0052`; all current workload/exact-policy/candidate/compatibility/route/codec versions
  are 14.
- `synaptik_metal_mpsgraph_executable_create` already decodes and validates the schema-13 image. A
  Task-0052-containing program becomes a whole-program custom execution recipe: wires `20..34` use
  reviewed custom kernels while other currently executable nodes may remain MPSGraph steps. The
  legacy singleton-NEG pipeline remains a separate exact route identity.
- Java and native currently admit exactly operation wires `1..34` in bounded domains; wires
  `35..115` are registered but fail closed as unsupported.

The installed SDK audit source is
`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/Headers/`.
The relevant current headers are `MPSGraphArithmeticOps.h`, `MPSGraphActivationOps.h`,
`MPSGraphReductionOps.h`, `MPSGraphCumulativeOps.h`, `MPSGraphTensorShapeOps.h`,
`MPSGraphGatherOps.h`, `MPSGraphScatterNDOps.h`, `MPSGraphOneHotOps.h`,
`MPSGraphImToColOps.h`, `MPSGraphMatrixMultiplicationOps.h`, `MPSGraphConvolutionOps.h`,
`MPSGraphPoolingOps.h`, `MPSGraphNormalizationOps.h`, `MPSGraphLossOps.h`,
`MPSGraphSortOps.h`, `MPSGraphTopKOps.h`, `MPSGraphRandomOps.h`, and `MPSGraphRNNOps.h`.
Header presence is structural evidence only. Availability annotations, accepted data types, exact
shape restrictions, output types, special values, ordering, reductions, and opaque algorithms still
require route-specific proof before capability.

## Closed catalog meanings

### MPSGraph state

Each kind has exactly one state:

- `DIRECT`: the current SDK has one named MPSGraph operation that can produce at least one
  structurally exact, bounded occurrence of this Model kind without another semantic operation.
  Descriptor construction and typed constants do not by themselves make a route composed.
- `COMPOSED`: no single selector produces such an occurrence, but an explicit finite composition of
  current MPSGraph primitives can structurally produce at least one. The exact composition is named
  below.
- `UNAVAILABLE`: current SDK structure cannot expose at least one mandatory observable contract for
  any exact occurrence. A hidden value, opaque state, wrong mandatory output type/cardinality, or
  missing atomic validation/skip behavior is a structural blocker.

This is deliberately an occurrence-existence catalog, not a whole-kind/domain claim. A `DIRECT`
entry may name excluded signatures or attributes. A later capability task must narrow and prove its
complete proposed domain; it cannot cite `DIRECT` or `COMPOSED` alone.

### Custom-kernel state

Each kind has exactly one state:

- `AVAILABLE`: current reviewed production source contains a custom implementation for at least one
  exact bounded occurrence of this kind.
- `PENDING`: schema 13 can represent the kind, but no current independently approved custom
  implementation owns an exact occurrence.
- `UNAVAILABLE-WITH-PROOF`: a cited architecture/representation proof establishes that no custom
  kernel route can represent any exact occurrence.

The current count for `UNAVAILABLE-WITH-PROOF` is zero. Schema 13 and the explicit custom program
lifecycle impose no known structural impossibility across the current 115 kinds. Missing source,
proof, numerical qualification, or engineering work is `PENDING`, never a fabricated impossibility.
The current custom `AVAILABLE` count is 16 kinds: singleton `NEG` plus the fifteen Task-0052 kinds.
A mixed Task-0052 whole-program route does not make its MPSGraph steps custom implementations.

## Reason-code ledger

The catalog implementation uses closed enum reason identities, not runtime prose or free-form
strings. The task document owns their exact interpretation.

### MPSGraph reasons

| Code | Exact structural reason |
|---|---|
| `MD-ARITH` | One named arithmetic selector matches an exact bounded occurrence. |
| `MD-PRED` | One classification, comparison, logical, or conditional selector matches. |
| `MD-SHAPE` | One reshape/broadcast/transpose/expand/squeeze/pad/slice/update/concat/tile selector matches; current `CONTIGUOUS` uses same-shape `reshapeTensor` materialization. |
| `MD-REDUCE` | One reduction/mean/variance/cumulative selector matches; arg selectors require the cast composition code instead. |
| `MD-INDEX` | One gather/scatter/one-hot selector matches the required indexed operation. |
| `MD-MATMUL` | One matrix-multiplication selector matches a bounded occurrence. |
| `MD-ATTENTION-BASE` | Direct unmasked, noncausal, output-only SDPA exists; BOOL mask, causal and weights-output forms are excluded pending an explicit composition and proof. |
| `MD-CONV-BASE` | Direct unbiased convolution exists; the optional bias form requires a following typed broadcast add and remains unapproved. |
| `MD-DENSE-CE` | Direct dense-label softmax-cross-entropy structurally matches a bounded reduction form. |
| `MD-BN-INFER` | Direct normalization can consume input/scale/bias/running mean/running variance for bounded inference. |
| `MD-SOFTMAX` | Direct axis softmax structurally matches. |
| `MD-SORT` | Direct one-output sort structurally matches; exact stability/NaN/zero semantics remain unproved. |
| `MD-POOL2D` | Named 2D pool selector maps a bounded 2D geometry; winner/divisor semantics remain unproved. |
| `MD-IM2COL` | Named `imToCol`/`colToIm` selector structurally maps a bounded 2D window occurrence. |
| `MC-SCALAR` | Create an exact typed scalar tensor, then use the corresponding binary selector. |
| `MC-UNARY` | `LOG1P=log(1+x)`, `EXPM1=exp(x)-1`, exact GELU/GELU-tanh formulas, or `SILU=x*sigmoid(x)`. |
| `MC-SELECT` | Unit-length slice on the selected axis, then squeeze that axis. |
| `MC-STACK` | Expand each input at the normalized axis, then concatenate. |
| `MC-UNFOLD-AXIS` | Emit each exact axis slice/window, reshape as required, then concatenate in canonical order. |
| `MC-FOLD-AXIS` | Generate exact destinations and scatter-add every window contributor. |
| `MC-WINDOW3D` | Materialize canonical NCDHW windows with slices/reshape/concat; fold scatter-adds them; pool reduces them with exact max/average rules still needing proof. |
| `MC-MSE` | Subtract, square by multiplication, then apply `NONE`, `SUM`, or exact-divisor `MEAN`. |
| `MC-INDEX-CE` | Build index labels with one-hot/gather plus log-softmax and requested reduction/ignore masking. |
| `MC-BN-TRAIN` | Explicit mean/population variance, saved inverse standard deviation, affine output, unbiased running variance, and two running-statistic updates produce five outputs. |
| `MC-LAYER-NORM` | Explicit trailing-slice mean, population variance, epsilon, reciprocal square root, and optional affine transform. |
| `MC-RMS-NORM` | Explicit mean square, epsilon, reciprocal square root, and optional scale. |
| `MC-LOG-SOFTMAX` | Stable axis max/subtract/exp/sum/log/subtract composition. |
| `MC-INDEX64` | Direct arg-sort/top-k/arg-extrema result is INT32; cast the mandatory index output to Model INT64. |
| `MC-LOGSUMEXP` | Stable max/subtract/exp/sum/log/add reduction. |
| `MC-STDDEV` | Variance followed by square root. |
| `MC-NORM` | L1 is `sum(abs(x))`; L2 is `sqrt(sum(x*x))` over exact axes. |
| `MU-RNG` | Direct dropout hides randomness and mask and returns one output; MPSGraph Philox state is opaque and cannot expose Model's exact raw INT64 `[key,counter]`, exact `+N mod 2^64`, BOOL mask, and three-output transition. |
| `MU-RECURRENT` | Current RNN/GRU/LSTM selectors do not expose the required runtime INT64 valid lengths, atomic validation, skipped padded work, exact positive-zero padding, and required final state outputs. |

### Custom reasons

| Code | Exact current state |
|---|---|
| `CA-NEG` | Reviewed singleton FLOAT32 custom NEG implementation exists in production. |
| `CA-0052` | Reviewed Task-0052 custom kernel exists for the named FLOAT32 comparison/extrema/scan kind. |
| `CP-POINT` | Pointwise custom source plus complete-domain result-set proof and route tests are absent. |
| `CP-MOVE` | Index/layout/window custom source plus exact mapping, validation, alias, and publication proof are absent. |
| `CP-CONTRACT` | Contraction/convolution/attention custom source plus shape-wide numerical proof are absent. |
| `CP-AGGREGATE` | Reduction/order/loss/normalization/pooling custom source plus contributor/order/special-value proof are absent. |
| `CP-STATE` | Explicit RNG/recurrent custom source plus exact state-transition/validation/publication proof are absent. |

## Exhaustive schema-13 route catalog

Every wire `1..115` occurs exactly once. `M` is the MPSGraph state/reason; `C` is the custom state/reason.

| Wire | Exact Model kind | M | C |
|---:|---|---|---|
| 1 | `UnaryElementwiseKind.NEG` | `DIRECT / MD-ARITH` | `AVAILABLE / CA-NEG` |
| 2 | `BinaryArithmeticKind.ADD` | `DIRECT / MD-ARITH` | `PENDING / CP-POINT` |
| 3 | `BinaryArithmeticKind.SUB` | `DIRECT / MD-ARITH` | `PENDING / CP-POINT` |
| 4 | `BinaryArithmeticKind.MUL` | `DIRECT / MD-ARITH` | `PENDING / CP-POINT` |
| 5 | `BinaryArithmeticKind.DIV` | `DIRECT / MD-ARITH` | `PENDING / CP-POINT` |
| 6 | `ShapeTransformKind.RESHAPE` | `DIRECT / MD-SHAPE` | `PENDING / CP-MOVE` |
| 7 | `ShapeTransformKind.EXPAND` | `DIRECT / MD-SHAPE` | `PENDING / CP-MOVE` |
| 8 | `AxisTransformKind.PERMUTE` | `DIRECT / MD-SHAPE` | `PENDING / CP-MOVE` |
| 9 | `AxisTransformKind.EXPAND_DIMS` | `DIRECT / MD-SHAPE` | `PENDING / CP-MOVE` |
| 10 | `AxisTransformKind.SQUEEZE` | `DIRECT / MD-SHAPE` | `PENDING / CP-MOVE` |
| 11 | `ContiguousKind.CONTIGUOUS` | `DIRECT / MD-SHAPE` | `PENDING / CP-MOVE` |
| 12 | `UnaryElementwiseKind.ABS` | `DIRECT / MD-ARITH` | `PENDING / CP-POINT` |
| 13 | `AggregateReductionKind.SUM` | `DIRECT / MD-REDUCE` | `PENDING / CP-AGGREGATE` |
| 14 | `AggregateReductionKind.MEAN` | `DIRECT / MD-REDUCE` | `PENDING / CP-AGGREGATE` |
| 15 | `MatmulKind.MATMUL` | `DIRECT / MD-MATMUL` | `PENDING / CP-CONTRACT` |
| 16 | `AxisGatherKind.GATHER` | `DIRECT / MD-INDEX` | `PENDING / CP-MOVE` |
| 17 | `OneHotKind.ONE_HOT` | `DIRECT / MD-INDEX` | `PENDING / CP-MOVE` |
| 18 | `AxisScatterKind.SCATTER_ELEMENTS` | `DIRECT / MD-INDEX` | `PENDING / CP-MOVE` |
| 19 | `WindowTransformKind.UNFOLD_AXIS` | `COMPOSED / MC-UNFOLD-AXIS` | `PENDING / CP-MOVE` |
| 20 | `BinaryComparisonKind.GREATER_THAN` | `DIRECT / MD-PRED` | `AVAILABLE / CA-0052` |
| 21 | `BinaryComparisonKind.GREATER_OR_EQUAL` | `DIRECT / MD-PRED` | `AVAILABLE / CA-0052` |
| 22 | `BinaryComparisonKind.LESS_THAN` | `DIRECT / MD-PRED` | `AVAILABLE / CA-0052` |
| 23 | `BinaryComparisonKind.LESS_OR_EQUAL` | `DIRECT / MD-PRED` | `AVAILABLE / CA-0052` |
| 24 | `BinaryComparisonKind.EQUAL` | `DIRECT / MD-PRED` | `AVAILABLE / CA-0052` |
| 25 | `BinaryComparisonKind.NOT_EQUAL` | `DIRECT / MD-PRED` | `AVAILABLE / CA-0052` |
| 26 | `BinaryArithmeticKind.MIN` | `DIRECT / MD-ARITH` | `AVAILABLE / CA-0052` |
| 27 | `BinaryArithmeticKind.MAX` | `DIRECT / MD-ARITH` | `AVAILABLE / CA-0052` |
| 28 | `ScalarElementwiseKind.MIN` | `COMPOSED / MC-SCALAR` | `AVAILABLE / CA-0052` |
| 29 | `ScalarElementwiseKind.MAX` | `COMPOSED / MC-SCALAR` | `AVAILABLE / CA-0052` |
| 30 | `ScalarElementwiseKind.CLAMP` | `DIRECT / MD-ARITH` | `AVAILABLE / CA-0052` |
| 31 | `AggregateReductionKind.MIN` | `DIRECT / MD-REDUCE` | `AVAILABLE / CA-0052` |
| 32 | `AggregateReductionKind.MAX` | `DIRECT / MD-REDUCE` | `AVAILABLE / CA-0052` |
| 33 | `CumulativeScanKind.CUM_SUM` | `DIRECT / MD-REDUCE` | `AVAILABLE / CA-0052` |
| 34 | `CumulativeScanKind.CUM_PROD` | `DIRECT / MD-REDUCE` | `AVAILABLE / CA-0052` |
| 35 | `ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION` | `DIRECT / MD-ATTENTION-BASE` | `PENDING / CP-CONTRACT` |
| 36 | `Conv2dKind.CONV2D` | `DIRECT / MD-CONV-BASE` | `PENDING / CP-CONTRACT` |
| 37 | `Conv3dKind.CONV3D` | `DIRECT / MD-CONV-BASE` | `PENDING / CP-CONTRACT` |
| 38 | `BinaryArithmeticKind.POW` | `DIRECT / MD-ARITH` | `PENDING / CP-POINT` |
| 39 | `CastKind.CAST` | `DIRECT / MD-ARITH` | `PENDING / CP-POINT` |
| 40 | `FloatingClassificationKind.IS_FINITE` | `DIRECT / MD-PRED` | `PENDING / CP-POINT` |
| 41 | `FloatingClassificationKind.IS_NAN` | `DIRECT / MD-PRED` | `PENDING / CP-POINT` |
| 42 | `FloatingClassificationKind.IS_INF` | `DIRECT / MD-PRED` | `PENDING / CP-POINT` |
| 43 | `BooleanLogicalKind.AND` | `DIRECT / MD-PRED` | `PENDING / CP-POINT` |
| 44 | `BooleanLogicalKind.OR` | `DIRECT / MD-PRED` | `PENDING / CP-POINT` |
| 45 | `BooleanLogicalKind.NOT` | `DIRECT / MD-PRED` | `PENDING / CP-POINT` |
| 46 | `ScalarElementwiseKind.ADD` | `COMPOSED / MC-SCALAR` | `PENDING / CP-POINT` |
| 47 | `ScalarElementwiseKind.SUB` | `COMPOSED / MC-SCALAR` | `PENDING / CP-POINT` |
| 48 | `ScalarElementwiseKind.MUL` | `COMPOSED / MC-SCALAR` | `PENDING / CP-POINT` |
| 49 | `ScalarElementwiseKind.DIV` | `COMPOSED / MC-SCALAR` | `PENDING / CP-POINT` |
| 50 | `ScalarElementwiseKind.POW` | `COMPOSED / MC-SCALAR` | `PENDING / CP-POINT` |
| 51 | `WhereSelectionKind.WHERE` | `DIRECT / MD-PRED` | `PENDING / CP-POINT` |
| 52 | `UnaryElementwiseKind.RECIPROCAL` | `DIRECT / MD-ARITH` | `PENDING / CP-POINT` |
| 53 | `UnaryElementwiseKind.LOG` | `DIRECT / MD-ARITH` | `PENDING / CP-POINT` |
| 54 | `UnaryElementwiseKind.LOG1P` | `COMPOSED / MC-UNARY` | `PENDING / CP-POINT` |
| 55 | `UnaryElementwiseKind.EXP` | `DIRECT / MD-ARITH` | `PENDING / CP-POINT` |
| 56 | `UnaryElementwiseKind.EXPM1` | `COMPOSED / MC-UNARY` | `PENDING / CP-POINT` |
| 57 | `UnaryElementwiseKind.ERF` | `DIRECT / MD-ARITH` | `PENDING / CP-POINT` |
| 58 | `UnaryElementwiseKind.SQRT` | `DIRECT / MD-ARITH` | `PENDING / CP-POINT` |
| 59 | `UnaryElementwiseKind.RSQRT` | `DIRECT / MD-ARITH` | `PENDING / CP-POINT` |
| 60 | `UnaryElementwiseKind.FLOOR` | `DIRECT / MD-ARITH` | `PENDING / CP-POINT` |
| 61 | `UnaryElementwiseKind.CEIL` | `DIRECT / MD-ARITH` | `PENDING / CP-POINT` |
| 62 | `UnaryElementwiseKind.SIGN` | `DIRECT / MD-ARITH` | `PENDING / CP-POINT` |
| 63 | `UnaryElementwiseKind.RELU` | `DIRECT / MD-ARITH` | `PENDING / CP-POINT` |
| 64 | `UnaryElementwiseKind.SIGMOID` | `DIRECT / MD-ARITH` | `PENDING / CP-POINT` |
| 65 | `UnaryElementwiseKind.TANH` | `DIRECT / MD-ARITH` | `PENDING / CP-POINT` |
| 66 | `UnaryElementwiseKind.GELU` | `COMPOSED / MC-UNARY` | `PENDING / CP-POINT` |
| 67 | `UnaryElementwiseKind.GELU_TANH_APPROXIMATION` | `COMPOSED / MC-UNARY` | `PENDING / CP-POINT` |
| 68 | `UnaryElementwiseKind.SILU` | `COMPOSED / MC-UNARY` | `PENDING / CP-POINT` |
| 69 | `AxisGatherKind.GATHER_ELEMENTS` | `DIRECT / MD-INDEX` | `PENDING / CP-MOVE` |
| 70 | `AxisScatterKind.SCATTER_ADD` | `DIRECT / MD-INDEX` | `PENDING / CP-MOVE` |
| 71 | `GatherNdKind.GATHER_ND` | `DIRECT / MD-INDEX` | `PENDING / CP-MOVE` |
| 72 | `ScatterNdKind.SCATTER_ND` | `DIRECT / MD-INDEX` | `PENDING / CP-MOVE` |
| 73 | `SelectKind.SELECT` | `COMPOSED / MC-SELECT` | `PENDING / CP-MOVE` |
| 74 | `PadKind.PAD` | `DIRECT / MD-SHAPE` | `PENDING / CP-MOVE` |
| 75 | `SliceKind.SLICE` | `DIRECT / MD-SHAPE` | `PENDING / CP-MOVE` |
| 76 | `SliceKind.SLICE_UPDATE` | `DIRECT / MD-SHAPE` | `PENDING / CP-MOVE` |
| 77 | `TensorCompositionKind.CONCAT` | `DIRECT / MD-SHAPE` | `PENDING / CP-MOVE` |
| 78 | `TensorCompositionKind.STACK` | `COMPOSED / MC-STACK` | `PENDING / CP-MOVE` |
| 79 | `TileKind.TILE` | `DIRECT / MD-SHAPE` | `PENDING / CP-MOVE` |
| 80 | `WindowTransformKind.FOLD_AXIS` | `COMPOSED / MC-FOLD-AXIS` | `PENDING / CP-MOVE` |
| 81 | `WindowTransformKind.UNFOLD2D` | `DIRECT / MD-IM2COL` | `PENDING / CP-MOVE` |
| 82 | `WindowTransformKind.FOLD2D` | `DIRECT / MD-IM2COL` | `PENDING / CP-MOVE` |
| 83 | `WindowTransformKind.UNFOLD3D` | `COMPOSED / MC-WINDOW3D` | `PENDING / CP-MOVE` |
| 84 | `WindowTransformKind.FOLD3D` | `COMPOSED / MC-WINDOW3D` | `PENDING / CP-MOVE` |
| 85 | `LossKind.MEAN_SQUARED_ERROR` | `COMPOSED / MC-MSE` | `PENDING / CP-AGGREGATE` |
| 86 | `LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` | `DIRECT / MD-DENSE-CE` | `PENDING / CP-AGGREGATE` |
| 87 | `LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` | `COMPOSED / MC-INDEX-CE` | `PENDING / CP-AGGREGATE` |
| 88 | `BatchNormKind.BATCH_NORM_INFERENCE` | `DIRECT / MD-BN-INFER` | `PENDING / CP-AGGREGATE` |
| 89 | `BatchNormKind.BATCH_NORM_TRAINING` | `COMPOSED / MC-BN-TRAIN` | `PENDING / CP-AGGREGATE` |
| 90 | `LayerNormKind.LAYER_NORM` | `COMPOSED / MC-LAYER-NORM` | `PENDING / CP-AGGREGATE` |
| 91 | `RmsNormKind.RMS_NORM` | `COMPOSED / MC-RMS-NORM` | `PENDING / CP-AGGREGATE` |
| 92 | `SoftmaxKind.SOFTMAX` | `DIRECT / MD-SOFTMAX` | `PENDING / CP-AGGREGATE` |
| 93 | `SoftmaxKind.LOG_SOFTMAX` | `COMPOSED / MC-LOG-SOFTMAX` | `PENDING / CP-AGGREGATE` |
| 94 | `OrderingKind.SORT` | `DIRECT / MD-SORT` | `PENDING / CP-AGGREGATE` |
| 95 | `OrderingKind.ARGSORT` | `COMPOSED / MC-INDEX64` | `PENDING / CP-AGGREGATE` |
| 96 | `TopKKind.TOP_K` | `COMPOSED / MC-INDEX64` | `PENDING / CP-AGGREGATE` |
| 97 | `Pool2dKind.MAX_POOL2D` | `DIRECT / MD-POOL2D` | `PENDING / CP-AGGREGATE` |
| 98 | `Pool2dKind.AVERAGE_POOL2D` | `DIRECT / MD-POOL2D` | `PENDING / CP-AGGREGATE` |
| 99 | `Pool3dKind.MAX_POOL3D` | `COMPOSED / MC-WINDOW3D` | `PENDING / CP-AGGREGATE` |
| 100 | `Pool3dKind.AVERAGE_POOL3D` | `COMPOSED / MC-WINDOW3D` | `PENDING / CP-AGGREGATE` |
| 101 | `DropoutKind.DROPOUT` | `UNAVAILABLE / MU-RNG` | `PENDING / CP-STATE` |
| 102 | `GraphRngKind.INITIAL_STATE` | `UNAVAILABLE / MU-RNG` | `PENDING / CP-STATE` |
| 103 | `RecurrentScanKind.RNN_TANH` | `UNAVAILABLE / MU-RECURRENT` | `PENDING / CP-STATE` |
| 104 | `RecurrentScanKind.GRU_RESET_AFTER` | `UNAVAILABLE / MU-RECURRENT` | `PENDING / CP-STATE` |
| 105 | `RecurrentScanKind.LSTM` | `UNAVAILABLE / MU-RECURRENT` | `PENDING / CP-STATE` |
| 106 | `AggregateReductionKind.PROD` | `DIRECT / MD-REDUCE` | `PENDING / CP-AGGREGATE` |
| 107 | `AggregateReductionKind.ALL` | `DIRECT / MD-REDUCE` | `PENDING / CP-AGGREGATE` |
| 108 | `AggregateReductionKind.ANY` | `DIRECT / MD-REDUCE` | `PENDING / CP-AGGREGATE` |
| 109 | `AggregateReductionKind.ARG_MAX` | `COMPOSED / MC-INDEX64` | `PENDING / CP-AGGREGATE` |
| 110 | `AggregateReductionKind.ARG_MIN` | `COMPOSED / MC-INDEX64` | `PENDING / CP-AGGREGATE` |
| 111 | `AggregateReductionKind.LOG_SUM_EXP` | `COMPOSED / MC-LOGSUMEXP` | `PENDING / CP-AGGREGATE` |
| 112 | `AggregateReductionKind.VARIANCE` | `DIRECT / MD-REDUCE` | `PENDING / CP-AGGREGATE` |
| 113 | `AggregateReductionKind.STANDARD_DEVIATION` | `COMPOSED / MC-STDDEV` | `PENDING / CP-AGGREGATE` |
| 114 | `AggregateReductionKind.L1_NORM` | `COMPOSED / MC-NORM` | `PENDING / CP-AGGREGATE` |
| 115 | `AggregateReductionKind.L2_NORM` | `COMPOSED / MC-NORM` | `PENDING / CP-AGGREGATE` |

The fixed totals are:

- MPSGraph: `76 DIRECT + 34 COMPOSED + 5 UNAVAILABLE = 115`;
- custom kernel: `16 AVAILABLE + 99 PENDING + 0 UNAVAILABLE-WITH-PROOF = 115`; and
- production capability: unchanged `34 true + 81 false = 115`.

## Minimal implementation design

### One catalog, no second registry

Add one package-private `MetalOperationRouteCatalog` beside `MetalMpsGraphProgram`. It is keyed only by
`MetalMpsGraphProgram.NodeKind`; it must not reflect over Model enums, duplicate wire numbers, or
parse class names. Use exhaustive Java switches so a future `NodeKind` addition fails compilation or
the exact-coverage test until it receives both states and both reasons.

Use closed package-private enums for the two state vocabularies and the reason identities. Return
immutable shared entries; do not allocate per lookup, build a map on the hot path, or retain prose.
The catalog must be queryable only during cold analysis/tests. It must not be consulted by Runtime.

The catalog is descriptive. Capability remains the sole admission authority. Route-specific
approval remains the sole permission to select an implementation. Tests must prove that changing a
catalog state alone changes neither capability nor the current heuristic route.

### One fixed prepared route identity

Keep the existing three exact implementation identities because they denote different prepared
resources and existing candidate wires:

- `CUSTOM_SINGLE_NEG` — custom-kernel family, candidate wire 1;
- `MPSGRAPH` — MPSGraph family, candidate wire 2; and
- `CUSTOM_TASK0052` — custom-kernel family, candidate wire 3.

Do not add a generic route string, integer scattered through callers, or a fourth identity. Move the
stable wire and the two-value family (`MPSGRAPH`, `CUSTOM_KERNEL`) onto the one closed route enum (or
an equally small package-private value if a mechanical extraction materially reduces duplication).
`MetalNegTuningBatch.Candidate` must delegate to that identity rather than own a second route-wire
truth. Preserve wire values 1/2/3.

`MetalNegPreparationPlan` must continue to hold exactly one non-null final route identity. The route
is selected once during analysis after schema/capability/validation and before route-dependent
internal buffers/workspace are declared. Finalization and execution switch only on that retained
identity. Runtime receives an already prepared executable and cannot reconsider it.

### Shared spine and route-specific boundary

Both route families use the same facts, in this order:

1. current capability and profile admission;
2. topology/value/descriptor/attribute lowering to one canonical schema-13 program;
3. Java program-image bounds and semantic preflight;
4. exact feed, target, internal-value, splat, and required-byte facts;
5. one selected route identity;
6. route-dependent resource declarations and cold lowering;
7. the existing prepared-resource ownership and deferred release rules;
8. the existing run-owned input/output/internal bindings, synchronous completion barrier, canary and
   failure behavior; and
9. exact public result publication only after success.

Route-specific code begins only at executable construction: MPSGraph graph/executable versus custom
pipeline/program steps. It may request different cold resources or workspace, but it may not bypass
schema validation, capability, index validation, buffer-size checks, lifecycle ownership, completion,
or publication. A failed create or run returns that failure; it never tries the other family.

Do not widen ABI 5 or schema 13 merely to label the route. The route is Java preparation/candidate
identity, while the schema image remains canonical route-neutral workload input. Native construction
continues through existing exports. If implementation discovers that shared validation cannot be
preserved without an ABI/schema change, stop this task as Blocked and write a successor plan; do not
smuggle a route bit into reserved schema words.

### Existing route freeze

The following decisions remain byte-for-byte in policy meaning:

- an eligible singleton FLOAT32 `NEG` with no selected handoff defaults to `CUSTOM_SINGLE_NEG`;
- a freshly authenticated current candidate decision may select `MPSGRAPH` for that exact existing
  singleton-NEG domain;
- any occurrence of a wire `20..34` fixes the whole partition to `CUSTOM_TASK0052`;
- every other currently supported partition uses `MPSGRAPH`; and
- every wire `35..115` remains capability-false and cannot reach either executor.

A `DIRECT`, `COMPOSED`, or `AVAILABLE` catalog entry cannot alter those rules. Each future change
needs a separate task with an exact bounded domain, complete Model-result-set proof, route-specific
approval, capability change if applicable, and independent review.

## Internal route forcing and parity protocol

Add no public knob. The production `prepare` entry calls one private core with no override. A
package-private test-only method in the Metal package may call that core with one exact route
identity. It must be unreachable from public integration/configuration and must:

- run all normal schema/capability/validation and declaration logic;
- reject an identity not approved for the exact current occurrence/domain;
- reject a stale/foreign candidate, unsupported kind, or incompatible route rather than substitute a
  route;
- embed the forced identity in the returned prepared plan; and
- preserve the same finalization, buffers, lifecycle, completion, failure, and publication path as
  production.

For every currently implemented alternative, test each route separately against the Model oracle:

1. prepare and run the custom route; assert the observable result belongs to the Model-owned result
   set for the selected profile;
2. prepare and run the MPSGraph route independently; assert that result belongs to the same
   Model-owned result set;
3. assert trace/prepared metadata names the forced route and that the plan never changes it; and
4. exercise a route-specific create/run failure and prove there is no second create/run attempt.

The existing singleton-NEG dual-route domain is the mandatory parity case. Task-0052 operations have
only the custom route approved: their current tests remain custom-vs-Model checks, and forcing
MPSGraph must fail before native construction. Do not compare route outputs bitwise unless the Model
result set for that exact case is singleton/exact. Under `ACCELERATOR`, two different permitted
FLOAT32 results are parity success. Compare structure, exact outputs such as BOOL/INT/index/state,
and floating membership according to the operation's Model contract.

No new real-device test or timing run belongs to this task. Use existing deterministic mocks and
existing packaged/native test facilities only for the current implemented routes.

## Identity and cache implications

- Schema-13 program bytes remain route-neutral and unchanged.
- Workload compatibility remains route-neutral so it can describe the same validated workload for
  multiple candidates.
- Candidate/decision bytes retain route wire 1/2/3 and session-scoped target compatibility.
- A prepared plan and native resource are route-specific. Any future executable/cache key must be
  the tuple `(workload compatibility, route wire identity, route-policy version, target session)`;
  workload digest alone is insufficient.
- The repository has no persistent Metal executable cache; this task adds none.
- Catalog metadata alone does not change wire meaning and therefore does not bump ABI 5, schema 13,
  or identity version 14. If implementation changes a candidate wire, accepted candidate set,
  route-policy meaning, or codec bytes, it must bump every affected candidate/compatibility/route/
  codec version together and update all decoders/tests with no dual decoder. A refactor that merely
  centralizes the existing 1/2/3 truth must prove byte-for-byte codec stability instead.

## Future controlled-autotune seam

The only preserved seam is the existing cold authenticated tuning handoff. A future separately
approved controlled environment may evaluate multiple **already correctness-approved** complete
route identities for one exact compatibility key and return one authenticated decision before
resource declaration. That future work must preserve the safe heuristic when no decision exists,
session compatibility, strict structural dominance rules, and the prohibition on local timing as
authority.

This task adds no measurement producer, timer, benchmark, matrix, cache, threshold, exploration,
background service, dynamic candidate generation, or Runtime decision. Unsupported/unapproved
routes never become candidates merely because the catalog says structurally possible.

## Implementation order

1. Recompute the 115-kind source inventory and current SDK selector names. If any count/state/reason
   differs, update this task first in a planning-only review; do not silently implement a different
   catalog.
2. Add closed catalog state/reason enums and the exhaustive `NodeKind` catalog with zero behavior
   change.
3. Centralize existing route family/wire identity on the current closed route identity; remove the
   duplicate candidate wire truth while preserving encoded bytes and identities.
4. Make the shared schema/capability/validation/value/buffer facts explicit before the
   route-specific construction switch. Delete any duplicated validation exposed by the refactor;
   retain route-required declarations.
5. Add the package-private test forcing seam and fail-closed approval gate. Do not route public or
   production callers through an override-bearing API.
6. Add catalog exhaustiveness/count tests, fixed-route tests, independent Model-result-set route
   tests, no-fallback failure tests, and codec stability tests.
7. Update current Metal package/backend docs and planning status. Do not edit historical task or
   evidence claims except to add an explicit later-supersession sentence when genuinely necessary.
8. Run focused verification once, then the one repository validation required by the integration
   contract. Remove throwaway artifacts and commit one atomic implementation.

## Required tests

Permanent tests earn their place only for these observable/invariant risks:

- all `NodeKind.values()` resolve exactly once and totals are MPSGraph `76/34/5`, custom
  `16/99/0`;
- representative entries from every reason class and all five MPSGraph-unavailable kinds have the
  exact closed states/reasons;
- catalog state changes cannot widen capability or route eligibility;
- current heuristic mapping remains exact for singleton NEG, ordinary wires `1..19`, mixed
  Task-0052 programs, and unsupported wires `35..115`;
- every plan has exactly one route identity before declaration/finalization and route-dependent
  workspace/internal-buffer invariants remain fail closed;
- candidate wires and encoded compatibility/candidate/decision bytes remain stable unless an
  explicit coordinated identity bump is required;
- forced singleton NEG custom and MPSGraph routes each satisfy the Model result set independently;
- Task-0052 MPSGraph forcing and every unavailable/unapproved force fail before native create;
- create/run failure performs no alternate-route native call and publishes no result;
- concurrent invocations never mutate route identity or share run-owned writable state; and
- no public Config/Engine/Metal API exposes route forcing or catalog state.

## Validation

Implementation validation, not this planning commit, must include:

1. focused catalog/route/candidate/codec/preparation/finalization/execution tests;
2. focused Metal capability and conformance tests proving the `34/81` matrix is unchanged;
3. native build plus exact export audit and schema-13 malformed/unsupported tests;
4. packaged signed-library focused route tests when native code or package inputs changed;
5. public Engine singleton-NEG tuned/untuned coverage if candidate plumbing changed;
6. architecture checks, documentation links/fences/final-newline checks, and `git diff --check`;
7. an audit alignment check proving wires `1..115`, exact kind/state equality, both fixed totals,
   every installed header selector citation, and every Model source link;
8. the repository build exactly once after focused tests; and
9. independent Class C review before `Complete`.

This planning-only commit runs documentation validation only. It runs no production build, device
probe, oracle, benchmark, performance test, or native package build.

## Completion criteria

Task 0056 may become `Complete` only when:

- one executable exhaustive catalog accounts for all 115 kinds with the fixed state totals;
- the normative per-wire evidence audit remains exactly aligned with the catalog and every cited
  selector, installed header, Model kind, and bounded signature is still current;
- one immutable route identity is embedded in every plan and used without hot reconsideration;
- both route families share the schema-13/capability/validation/lifecycle/buffer/publication spine;
- all current production route decisions, capability rows, candidate bytes, and public APIs are
  unchanged unless an explicitly coordinated identity bump was required;
- internal forcing proves each approved route independently against Model result sets and proves no
  fallback/retry;
- local timing is absent from qualification and selection;
- focused and repository validation pass; and
- independent review approves the atomic implementation.
