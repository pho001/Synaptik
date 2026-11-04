# Task 0056: Deterministic Dual-Route Catalog and Fixed Route [Identity](../../../../glossary.md#term-identity-meanings)

## Reader context

Planning chooses Metal as backend owner; Metal preparation chooses the implementation route. A catalog cell says an SDK selector or custom implementation is structurally available for some occurrence, not that it is admitted, numerically approved or selected. `MetalPreparedRoute` carries one selected recipe into finalization; Runtime does not try the catalog after failure.

Package-private forcing was a test seam, not an application option. Route names, candidate counts and versions below belong to this checkpoint. Read current routing separately rather than inferring support from `DIRECT`, `COMPOSED` or `AVAILABLE`.

- [Current preparation-fixed routes](../../../../backend-guide/metal-backend.md#physical-authentication-and-fixed-routes) — Explains backend ownership versus implementation choice.
- [Current occurrence and candidate boundary](../../../../backend-guide/metal-backend.md#capability-whole-partition-analysis-and-route-selection) — Distinguishes structural catalog cells from executable permission.

## Status

Complete

Implemented at `4f35576c` with the focused [conformance](../../../../glossary.md#term-conformance)-[matrix](../../../../glossary.md#term-matrix-meanings) correction `4a5cbdef` and
independent-review remediation `1718b28a`. The Class C reviewer found and approved remediation of
one P2 decision-empty [handoff](../../../../glossary.md#term-handoff)-[authentication](../../../../glossary.md#term-authentication) gap. A follow-up external final review then found a
separate P2 catalog-description mismatch: wire 39 used the arithmetic reason despite the [normative](../../../../glossary.md#term-normative)
`M-CAST` evidence and installed `SHAPE::castTensor:toType:name:` selector. The catalog, exhaustive
test, and documentation now use the dedicated closed `MD_CAST` reason with no behavior or
capability change. Task 0055 remains Complete at `ddeff1b2`; the signed-32 native program-image
remediation remains Complete at `ce569b65`, and the [ABI](../../../../glossary.md#term-application-binary-interface-abi)/[schema](../../../../glossary.md#term-schema) documentation remediations remain
Complete at `486ff493` plus `5b80d37c`.

The normative [per-wire route-evidence audit](0056-route-evidence-audit.md) remains part of this
task. It resolves every row to an exact installed-[SDK](../../../../glossary.md#term-sdk) selector or [finite](../../../../glossary.md#term-finite) selector composition and
to the current [Model](../../../../glossary.md#term-model) enum source plus bounded [signature](../../../../glossary.md#term-signature); broad reason codes are summaries, not
evidence substitutes.

This completed internal architecture/catalog task changed no [operation](../../../../glossary.md#operation) capability, performance
authority, [device](../../../../glossary.md#term-device) contract, [oracle](../../../../glossary.md#term-test-oracle), timing policy, benchmark, or [autotuning](../../../../glossary.md#autotuning--model-autotuning) behavior.

## Change class

Class C. The implementation changes [backend](../../../../glossary.md#backend)-private route [metadata](../../../../glossary.md#term-metadata), preparation [identity](../../../../glossary.md#term-identity-meanings), candidate
plumbing, route-forcing test seams, and tests. It may refactor current [Metal](../../../../glossary.md#term-metal) Java/native internals
only as required to make the two route families consume one validated [schema](../../../../glossary.md#term-schema)-13 plan. It changes no
public [API](../../../../glossary.md#term-application-programming-interface-api), [Model](../../../../glossary.md#term-model) semantics, capability row, native export name, program wire, data-type wire, or
attribute wire.

## Goal

Install the smallest deterministic internal architecture in which every prepared [Metal](../../../../glossary.md#term-metal) [partition](../../../../glossary.md#partition):

1. is lowered once to the existing bounded schema-13 program and validated through the existing
   capability, topology, descriptor, attribute, lifecycle, buffer, and publication gates;<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [descriptor glossary](../../../../glossary.md#term-descriptor); [lifecycle glossary](../../../../glossary.md#lifecycle); [buffer glossary](../../../../glossary.md#term-buffer); [publication glossary](../../../../glossary.md#term-publication))</span>
2. embeds exactly one immutable backend-private route identity before resource declaration and
   finalization;<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [backend glossary](../../../../glossary.md#backend); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
3. reaches either the [MPSGraph](../../../../glossary.md#term-mpsgraph) family or the custom-[kernel](../../../../glossary.md#kernel) family only after shared [validation](../../../../glossary.md#term-validation);
4. cannot switch, retry, fall back, benchmark, or autotune at [Runtime](../../../../glossary.md#term-runtime); and
5. can be forced by package-private tests so each implemented route is checked independently against
   the Model-owned result set.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>

The task also lands one exhaustive, executable catalog for all 115 current [schema](../../../../glossary.md#term-schema)-13 [operation](../../../../glossary.md#operation)
wires. The catalog records structural route potential, not capability or correctness approval.
Existing production routes and the exact `34 capability-true + 81 capability-false = 115` count stay
fixed.

## Non-goals

- no new capability-true [operation](../../../../glossary.md#operation), [data type](../../../../glossary.md#data-type--datatype), [shape](../../../../glossary.md#shape), attribute form, [gradient](../../../../glossary.md#term-gradient), or profile domain;
- no new [MPSGraph](../../../../glossary.md#term-mpsgraph) selector, composition, custom [kernel](../../../../glossary.md#kernel), [oracle](../../../../glossary.md#term-test-oracle), [device](../../../../glossary.md#term-device) [run](../../../../glossary.md#term-run), or numerical proof;
- no ABI-6, schema-14, export, public configuration, environment variable, system property, public
  route enum, or public route-forcing API;<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- no hot route lookup, route switch, retry, fallback, benchmark, telemetry-driven selection,
  persistent cache, threshold, winner, or autotuner;<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>
- no use of local timing as [qualification](../../../../glossary.md#term-qualification), route choice, [cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings), or tuning authority;
- no reinterpretation of the blocked Task 0026/0027/0030-0037/0051/0053 records;
- no claim that an SDK selector is Model-correct merely because its shape/cardinality can be made to
  match; and<span data-glossary-navigation="true"> ([SDK glossary](../../../../glossary.md#term-sdk); [Model glossary](../../../../glossary.md#term-model); [cardinality glossary](../../../../glossary.md#term-cardinality))</span>
- no bitwise equality requirement between two legal implementations when the selected numerical
  profile permits multiple observable results.

## Current source and [SDK](../../../../glossary.md#term-sdk) ground truth

The implementation must re-audit these exact current sources before editing and fail [planning](../../../../glossary.md#term-planning) drift
rather than silently changing this ledger:

- `MetalMpsGraphProgram.NodeKind` is [schema](../../../../glossary.md#term-schema) 13 with stable [operation](../../../../glossary.md#operation) wires `1..115`, attribute wires
  `0..41`, type wires `1..6`, [rank](../../../../glossary.md#term-rank) `0..16`, and variable `0..N` input / `1..N` output [cardinality](../../../../glossary.md#term-cardinality).
- `MetalNegPartitionPreparer` currently fixes `CUSTOM_TASK0052` when any wire `20..34` occurs,
  otherwise fixes eligible singleton <span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton))</span>`NEG` to `CUSTOM_SINGLE_NEG`, otherwise uses [`MPSGRAPH`](../../../../glossary.md#term-mpsgraph).
- `MetalNegPreparationPlan` already retains one closed route value. It is the starting point, not a
  reason to add a second route-selection object.
- `MetalNegTuningBatch.Candidate` uses stable route wires 1 `CUSTOM_SINGLE_NEG`, 2 `MPSGRAPH`, and 3
  `CUSTOM_TASK0052`; all current workload/exact-policy/candidate/compatibility/route/codec versions
  are 14.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>
- `synaptik_metal_mpsgraph_executable_create` already decodes and validates the schema-13 image. A
  Task-0052-containing program becomes a whole-program custom execution recipe: wires `20..34` use
  reviewed custom kernels while other currently executable nodes may remain MPSGraph steps. The
  legacy singleton-NEG pipeline remains a separate exact route identity.<span data-glossary-navigation="true"> ([kernels glossary](../../../../glossary.md#kernel); [nodes glossary](../../../../glossary.md#node); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Java and native currently admit exactly operation wires `1..34` in bounded domains; wires
  `35..115` are registered but [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) as unsupported.

The installed SDK audit source is
`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/Headers/`.
The relevant current headers are `MPSGraphArithmeticOps.h`, `MPSGraphActivationOps.h`,
`MPSGraphReductionOps.h`, `MPSGraphCumulativeOps.h`, `MPSGraphTensorShapeOps.h`,
`MPSGraphGatherOps.h`, `MPSGraphScatterNDOps.h`, `MPSGraphOneHotOps.h`,
`MPSGraphImToColOps.h`, `MPSGraphMatrixMultiplicationOps.h`, `MPSGraphConvolutionOps.h`,
`MPSGraphPoolingOps.h`, `MPSGraphNormalizationOps.h`, `MPSGraphLossOps.h`,
`MPSGraphSortOps.h`, `MPSGraphTopKOps.h`, `MPSGraphRandomOps.h`, and `MPSGraphRNNOps.h`.
Header presence is structural evidence only. Availability annotations, accepted data types, exact
[shape](../../../../glossary.md#shape) restrictions, output types, special values, ordering, reductions, and [opaque](../../../../glossary.md#term-opaque) algorithms still
require route-specific proof before capability.

## Closed catalog meanings

### [MPSGraph](../../../../glossary.md#term-mpsgraph) state

Each kind has exactly one state:

- `DIRECT`: the current SDK has one named MPSGraph operation that can produce at least one
  structurally exact, bounded occurrence of this Model kind without another semantic operation.
  Descriptor construction and typed constants do not by themselves make a route composed.<span data-glossary-navigation="true"> ([SDK glossary](../../../../glossary.md#term-sdk); [operation glossary](../../../../glossary.md#operation); [Model glossary](../../../../glossary.md#term-model); [Descriptor glossary](../../../../glossary.md#term-descriptor); [constants glossary](../../../../glossary.md#term-constant))</span>
- `COMPOSED`: no single selector produces such an occurrence, but an explicit finite composition of
  current MPSGraph primitives can structurally produce at least one. The exact composition is named
  below.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite))</span>
- `UNAVAILABLE`: current SDK structure cannot expose at least one mandatory observable contract for
  any exact occurrence. A hidden value, opaque state, wrong mandatory output type/cardinality, or
  missing atomic validation/skip behavior is a structural blocker.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [cardinality glossary](../../../../glossary.md#term-cardinality); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [validation glossary](../../../../glossary.md#term-validation))</span>

This is deliberately an [occurrence](../../../../glossary.md#term-occurrence)-existence catalog, not a whole-kind/domain claim. A `DIRECT`
entry may name excluded [signatures](../../../../glossary.md#term-signature) or attributes. A later capability task must narrow and prove its
complete proposed domain; it cannot cite `DIRECT` or `COMPOSED` alone.

### Custom-[kernel](../../../../glossary.md#kernel) state

Each kind has exactly one state:

- `AVAILABLE`: current reviewed production source contains a custom implementation for at least one
  exact bounded occurrence of this kind.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- `PENDING`: schema 13 can represent the kind, but no current independently approved custom
  implementation owns an exact occurrence.
- `UNAVAILABLE-WITH-PROOF`: a cited architecture/representation proof establishes that no custom
  kernel route can represent any exact occurrence.

The current count for `UNAVAILABLE-WITH-PROOF` is zero. [Schema](../../../../glossary.md#term-schema) 13 and the explicit custom program
[lifecycle](../../../../glossary.md#lifecycle) impose no known structural impossibility across the current 115 kinds. Missing source,
proof, numerical [qualification](../../../../glossary.md#term-qualification), or engineering work is `PENDING`, never a fabricated impossibility.
The current custom `AVAILABLE` count is 16 kinds: [singleton](../../../../glossary.md#term-singleton) `NEG` plus the fifteen Task-0052 kinds.
A mixed Task-0052 whole-program route does not make its [MPSGraph](../../../../glossary.md#term-mpsgraph) steps custom implementations.

## Reason-code ledger

The catalog implementation uses closed enum reason [identities](../../../../glossary.md#term-identity-meanings), not [runtime](../../../../glossary.md#term-runtime) prose or free-form
strings. The task document owns their exact interpretation.

### [MPSGraph](../../../../glossary.md#term-mpsgraph) reasons

| Code | Exact structural reason |
|---|---|
| `MD-ARITH` | One named arithmetic selector matches an exact bounded [occurrence](../../../../glossary.md#term-occurrence). |
| `MD-CAST` | The named cast selector maps the exact bounded [dtype](../../../../glossary.md#data-type--datatype) conversion. |
| `MD-PRED` | One classification, comparison, logical, or conditional selector matches. |
| `MD-SHAPE` | One reshape/broadcast/[transpose](../../../../glossary.md#term-transpose)/expand/squeeze/pad/[slice](../../../../glossary.md#slice)/update/concat/tile selector matches; current [`CONTIGUOUS`](../../../../glossary.md#term-contiguous) uses same-[shape](../../../../glossary.md#shape) `reshapeTensor` [materialization](../../../../glossary.md#materialization). |
| `MD-REDUCE` | One reduction/[mean](../../../../glossary.md#term-mean)/[variance](../../../../glossary.md#term-variance)/cumulative selector matches; arg selectors require the cast composition code instead. |
| `MD-INDEX` | One [gather](../../../../glossary.md#gather)/scatter/one-hot selector matches the required indexed [operation](../../../../glossary.md#operation). |
| `MD-MATMUL` | One [matrix](../../../../glossary.md#term-matrix-meanings)-multiplication selector matches a bounded occurrence. |
| `MD-ATTENTION-BASE` | Direct unmasked, noncausal, output-only SDPA exists; [BOOL](../../../../glossary.md#data-type--datatype) [mask](../../../../glossary.md#term-boolean-mask), causal and [weights](../../../../glossary.md#term-weight)-output forms are excluded pending an explicit composition and proof. |
| `MD-CONV-BASE` | Direct unbiased [convolution](../../../../glossary.md#term-channels-first-convolution-layers) exists; the optional [bias](../../../../glossary.md#term-bias) form requires a following typed broadcast add and remains unapproved. |
| `MD-DENSE-CE` | Direct dense-label [softmax](../../../../glossary.md#softmax--log-softmax)-cross-entropy structurally matches a bounded reduction form. |
| `MD-BN-INFER` | Direct [normalization](../../../../glossary.md#term-normalization) can consume input/scale/bias/running mean/running variance for bounded [inference](../../../../glossary.md#term-inference). |
| `MD-SOFTMAX` | Direct [axis](../../../../glossary.md#term-axis) softmax structurally matches. |
| `MD-SORT` | Direct one-output sort structurally matches; exact stability/[NaN](../../../../glossary.md#term-nan)/zero semantics remain unproved. |
| `MD-POOL2D` | Named 2D pool selector maps a bounded 2D geometry; winner/divisor semantics remain unproved. |
| `MD-IM2COL` | Named `imToCol`/`colToIm` selector structurally maps a bounded 2D window occurrence. |
| `MC-SCALAR` | Create an exact typed [scalar](../../../../glossary.md#term-scalar) [tensor](../../../../glossary.md#tensor), then use the corresponding [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) selector. |
| `MC-UNARY` | `LOG1P=log(1+x)`, `EXPM1=exp(x)-1`, exact [GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu)/GELU-[tanh](../../../../glossary.md#term-tanh) formulas, or `SILU=x*sigmoid(x)`. |
| `MC-SELECT` | Unit-length slice on the selected axis, then squeeze that axis. |
| `MC-STACK` | Expand each input at the [normalized axis](../../../../glossary.md#normalized-axis), then concatenate. |
| `MC-UNFOLD-AXIS` | Emit each exact axis slice/window, reshape as required, then concatenate in [canonical](../../../../glossary.md#term-canonical-form) order. |
| `MC-FOLD-AXIS` | Generate exact destinations and scatter-add every window contributor. |
| `MC-WINDOW3D` | Materialize canonical [NCDHW](../../../../glossary.md#conv3d--ncdhw) windows with slices/reshape/concat; fold scatter-adds them; pool reduces them with exact max/average rules still needing proof. |
| `MC-MSE` | Subtract, square by multiplication, then apply `NONE`, `SUM`, or exact-divisor `MEAN`. |
| `MC-INDEX-CE` | Build index labels with one-hot/gather plus [log-softmax](../../../../glossary.md#softmax--log-softmax) and requested reduction/ignore masking. |
| `MC-BN-TRAIN` | Explicit mean/[population variance](../../../../glossary.md#population-variance), saved inverse [standard deviation](../../../../glossary.md#term-standard-deviation), affine output, unbiased running variance, and two running-[statistic](../../../../glossary.md#term-statistic) updates produce five outputs. |
| `MC-LAYER-NORM` | Explicit trailing-slice mean, population variance, [epsilon](../../../../glossary.md#epsilon), reciprocal [square root](../../../../glossary.md#norms-elementary-functions-and-probability-distributions), and optional [affine transform](../../../../glossary.md#affine-transform). |
| `MC-RMS-NORM` | Explicit mean square, epsilon, reciprocal square root, and optional scale. |
| `MC-LOG-SOFTMAX` | Stable axis max/subtract/[exp](../../../../glossary.md#term-expx)/sum/[log](../../../../glossary.md#term-log)/subtract composition. |
| `MC-INDEX64` | Direct arg-sort/top-k/arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum) result is [INT32](../../../../glossary.md#data-type--datatype); cast the mandatory index output to [Model](../../../../glossary.md#term-model) [INT64](../../../../glossary.md#data-type--datatype). |
| `MC-LOGSUMEXP` | Stable max/subtract/exp/sum/log/add reduction. |
| `MC-STDDEV` | Variance followed by square root. |
| `MC-NORM` | [L1](../../../../glossary.md#term-l1) is `sum(abs(x))`; [L2](../../../../glossary.md#term-l2) is `sqrt(sum(x*x))` over exact axes. |
| `MU-RNG` | Direct [dropout](../../../../glossary.md#dropout--inverted-dropout) hides randomness and mask and returns one output; MPSGraph [Philox](../../../../glossary.md#graph-rng-state--graphrngstate) state is [opaque](../../../../glossary.md#term-opaque) and cannot expose Model's exact raw INT64 `[key,counter]`, exact `+N mod 2^64`, BOOL mask, and three-output transition. |
| `MU-RECURRENT` | Current [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) selectors do not expose the required [runtime](../../../../glossary.md#term-runtime) INT64 valid lengths, [atomic](../../../../glossary.md#resource-transaction-and-rollback) [validation](../../../../glossary.md#term-validation), skipped padded work, exact positive-zero [padding](../../../../glossary.md#padding), and required final state outputs. |

### Custom reasons

| Code | Exact current state |
|---|---|
| `CA-NEG` | Reviewed [singleton](../../../../glossary.md#term-singleton) [FLOAT32](../../../../glossary.md#term-float32) custom NEG implementation exists in production. |
| `CA-0052` | Reviewed Task-0052 custom [kernel](../../../../glossary.md#kernel) exists for the named FLOAT32 comparison/[extrema](../../../../glossary.md#saturation-pole-and-extremum)/[scan](../../../../glossary.md#term-cumulative-scan) kind. |
| `CP-POINT` | [Pointwise](../../../../glossary.md#term-pointwise) custom source plus complete-domain result-set proof and route tests are absent. |
| `CP-MOVE` | Index/[layout](../../../../glossary.md#layout)/window custom source plus exact mapping, [validation](../../../../glossary.md#term-validation), [alias](../../../../glossary.md#aliasing-and-arena), and [publication](../../../../glossary.md#term-publication) proof are absent. |
| `CP-CONTRACT` | [Contraction](../../../../glossary.md#contraction-dimension)/[convolution](../../../../glossary.md#term-channels-first-convolution-layers)/attention custom source plus [shape](../../../../glossary.md#shape)-wide numerical proof are absent. |
| `CP-AGGREGATE` | Reduction/order/[loss](../../../../glossary.md#loss)/[normalization](../../../../glossary.md#term-normalization)/[pooling](../../../../glossary.md#term-pooling) custom source plus contributor/order/special-value proof are absent. |
| `CP-STATE` | Explicit [RNG](../../../../glossary.md#graph-rng-state--graphrngstate)/recurrent custom source plus exact state-transition/validation/publication proof are absent. |

## Exhaustive [schema](../../../../glossary.md#term-schema)-13 route catalog

Every wire `1..115` occurs exactly once. `M` is the [MPSGraph](../../../../glossary.md#term-mpsgraph) state/reason; `C` is the custom state/reason.

| Wire | Exact [Model](../../../../glossary.md#term-model) kind | M | C |
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
| 39 | `CastKind.CAST` | `DIRECT / MD-CAST` | `PENDING / CP-POINT` |
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
- custom [kernel](../../../../glossary.md#kernel): `16 AVAILABLE + 99 PENDING + 0 UNAVAILABLE-WITH-PROOF = 115`; and
- production capability: unchanged `34 true + 81 false = 115`.

## Minimal implementation design

### One catalog, no second [registry](../../../../glossary.md#term-registry)

Add one package-private `MetalOperationRouteCatalog` beside `MetalMpsGraphProgram`. It is keyed only by
`MetalMpsGraphProgram.NodeKind`; it must not reflect over [Model](../../../../glossary.md#term-model) enums, duplicate wire numbers, or
parse class names. Use exhaustive Java switches so a future `NodeKind` addition fails [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) or
the exact-coverage test until it receives both states and both reasons.

Use closed package-private enums for the two state vocabularies and the reason [identities](../../../../glossary.md#term-identity-meanings). Return
[immutable](../../../../glossary.md#term-immutable) shared entries; do not allocate per [lookup](../../../../glossary.md#term-lookup), build a map on the [hot path](../../../../glossary.md#term-hot-path), or retain prose.
The catalog must be queryable only during cold analysis/tests. It must not be consulted by [Runtime](../../../../glossary.md#term-runtime).

The catalog is descriptive. Capability remains the sole [admission](../../../../glossary.md#term-admission) authority. Route-specific
approval remains the sole permission to select an implementation. Tests must prove that changing a
catalog state alone changes neither capability nor the current [heuristic](../../../../glossary.md#term-heuristic) route.

### One fixed prepared route [identity](../../../../glossary.md#term-identity-meanings)

Keep the existing three exact implementation identities because they denote different prepared
resources and existing candidate wires:

- `CUSTOM_SINGLE_NEG` — custom-[kernel](../../../../glossary.md#kernel) family, candidate wire 1;
- [`MPSGRAPH`](../../../../glossary.md#term-mpsgraph) — MPSGraph family, candidate wire 2; and
- `CUSTOM_TASK0052` — custom-kernel family, candidate wire 3.

Do not add a generic route string, integer scattered through callers, or a fourth identity. Move the
stable wire and the two-value family (`MPSGRAPH`, `CUSTOM_KERNEL`) onto the one closed route enum (or
an equally small package-private value if a mechanical extraction materially reduces duplication).
`MetalNegTuningBatch.Candidate` must delegate to that identity rather than own a second route-wire
truth. Preserve wire values 1/2/3.

`MetalNegPreparationPlan` must continue to hold exactly one non-null final route identity. The route
is selected once during analysis after [schema](../../../../glossary.md#term-schema)/capability/[validation](../../../../glossary.md#term-validation) and before route-dependent
internal [buffers](../../../../glossary.md#term-buffer)/workspace are declared. Finalization and execution switch only on that retained
identity. [Runtime](../../../../glossary.md#term-runtime) receives an already [prepared executable](../../../../glossary.md#prepared-executable--preparedexecutable) and cannot reconsider it.

### Shared spine and route-specific boundary

Both route families use the same facts, in this order:

1. current capability and profile [admission](../../../../glossary.md#term-admission);
2. [topology](../../../../glossary.md#term-topology)/value/[descriptor](../../../../glossary.md#term-descriptor)/attribute [lowering](../../../../glossary.md#lowering) to one [canonical](../../../../glossary.md#term-canonical-form) [schema](../../../../glossary.md#term-schema)-13 program;
3. Java program-image bounds and semantic [preflight](../../../../glossary.md#term-preflight);
4. exact feed, [target](../../../../glossary.md#term-target), internal-value, splat, and required-byte facts;
5. one selected route [identity](../../../../glossary.md#term-identity-meanings);
6. route-dependent resource declarations and cold lowering;
7. the existing prepared-resource [ownership](../../../../glossary.md#term-ownership) and deferred [release](../../../../glossary.md#term-release) rules;
8. the existing run-owned input/output/internal bindings, synchronous completion barrier, canary and
   failure behavior; and<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding); [synchronous glossary](../../../../glossary.md#term-synchronous); [canary glossary](../../../../glossary.md#term-canary))</span>
9. exact public result [publication](../../../../glossary.md#term-publication) only after success.

Route-specific code begins only at executable construction: [MPSGraph](../../../../glossary.md#term-mpsgraph) [graph](../../../../glossary.md#graph)/executable versus custom
pipeline/program steps. It may request different cold resources or workspace, but it may not bypass
schema [validation](../../../../glossary.md#term-validation), capability, index validation, [buffer](../../../../glossary.md#term-buffer)-size checks, [lifecycle](../../../../glossary.md#lifecycle) [ownership](../../../../glossary.md#term-ownership-meanings), completion,
or publication. A failed create or [run](../../../../glossary.md#term-run) returns that failure; it never tries the other family.

Do not widen [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5 or schema 13 merely to label the route. The route is Java preparation/candidate
identity, while the schema image remains canonical route-neutral workload input. Native construction
continues through existing exports. If implementation discovers that shared validation cannot be
preserved without an ABI/schema change, stop this task as Blocked and write a successor plan; do not
smuggle a route bit into reserved schema words.

### Existing route freeze

The following decisions remain byte-for-byte in policy meaning:

- an eligible [singleton](../../../../glossary.md#term-singleton) [FLOAT32](../../../../glossary.md#term-float32) `NEG` with no selected [handoff](../../../../glossary.md#term-handoff) defaults to `CUSTOM_SINGLE_NEG`;
- a freshly authenticated current candidate decision may select [`MPSGRAPH`](../../../../glossary.md#term-mpsgraph) for that exact existing
  singleton-NEG domain;
- any [occurrence](../../../../glossary.md#term-occurrence) of a wire `20..34` fixes the whole [partition](../../../../glossary.md#partition) to `CUSTOM_TASK0052`;
- every other currently supported partition uses `MPSGRAPH`; and
- every wire `35..115` remains capability-false and cannot reach either executor.

A `DIRECT`, `COMPOSED`, or `AVAILABLE` catalog entry cannot alter those rules. Each future change
needs a separate task with an exact bounded domain, complete [Model](../../../../glossary.md#term-model)-result-set proof, route-specific
approval, capability change if applicable, and independent review.

## Internal route forcing and parity protocol

Add no public knob. The production [`prepare`](../../../../glossary.md#prepare) entry calls one private core with no override. A
package-private test-only method in the [Metal](../../../../glossary.md#term-metal) package may call that core with one exact route
[identity](../../../../glossary.md#term-identity-meanings). It must be unreachable from public integration/configuration and must:

- [run](../../../../glossary.md#term-run) all normal [schema](../../../../glossary.md#term-schema)/capability/[validation](../../../../glossary.md#term-validation) and declaration logic;
- reject an identity not approved for the exact current [occurrence](../../../../glossary.md#term-occurrence)/domain;
- reject a stale/foreign candidate, unsupported kind, or incompatible route rather than substitute a
  route;
- embed the forced identity in the returned prepared plan; and
- preserve the same finalization, buffers, lifecycle, completion, failure, and publication path as
  production.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [lifecycle glossary](../../../../glossary.md#lifecycle); [publication glossary](../../../../glossary.md#term-publication))</span>

For every currently implemented alternative, test each route separately against the [Model](../../../../glossary.md#term-model) [oracle](../../../../glossary.md#term-test-oracle):

1. prepare and run the custom route; assert the observable result belongs to the Model-owned result
   set for the selected profile;
2. prepare and run the MPSGraph route independently; assert that result belongs to the same
   Model-owned result set;
3. assert [trace](../../../../glossary.md#trace)/prepared [metadata](../../../../glossary.md#term-metadata) names the forced route and that the plan never changes it; and
4. exercise a route-specific create/run failure and prove there is no second create/run attempt.

The existing [singleton](../../../../glossary.md#term-singleton)-NEG dual-route domain is the mandatory parity case. Task-0052 operations have
only the custom route approved: their current tests remain custom-vs-Model checks, and forcing
[MPSGraph](../../../../glossary.md#term-mpsgraph) must fail before native construction. Do not compare route outputs bitwise unless the Model
result set for that exact case is singleton/exact. Under [`ACCELERATOR`](../../../../glossary.md#term-accelerator), two different permitted
[FLOAT32](../../../../glossary.md#term-float32) results are parity success. Compare structure, exact outputs such as [BOOL](../../../../glossary.md#data-type--datatype)/INT/index/state,
and floating membership according to the [operation](../../../../glossary.md#operation)'s Model contract.

No new real-[device](../../../../glossary.md#term-device) test or timing run belongs to this task. Use existing deterministic mocks and
existing packaged/native test facilities only for the current implemented routes.

## [Identity](../../../../glossary.md#term-identity-meanings) and [cache](../../../../glossary.md#term-cache) implications

- [Schema](../../../../glossary.md#term-schema)-13 program bytes remain route-neutral and unchanged.
- Workload compatibility remains route-neutral so it can describe the same validated workload for
  multiple candidates.
- Candidate/decision bytes retain route wire 1/2/3 and session-scoped [target](../../../../glossary.md#term-target) compatibility.
- A prepared plan and native resource are route-specific. Any future executable/cache key must be
  the tuple <span data-glossary-navigation="true"> ([cache key glossary](../../../../glossary.md#term-cache-key))</span>`(workload compatibility, route wire identity, route-policy version, target session)`;
  workload digest alone is insufficient.<span data-glossary-navigation="true"> ([digest glossary](../../../../glossary.md#term-digest))</span>
- The repository has no persistent [Metal](../../../../glossary.md#term-metal) executable cache; this task adds none.
- Catalog metadata alone does not change wire meaning and therefore does not bump ABI 5, schema 13,
  or identity version 14. If implementation changes a candidate wire, accepted candidate set,
  route-policy meaning, or codec bytes, it must bump every affected candidate/compatibility/route/
  codec version together and update all decoders/tests with no dual decoder. A refactor that merely
  centralizes the existing 1/2/3 truth must prove byte-for-byte codec stability instead.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [codec glossary](../../../../glossary.md#term-codec))</span>

## Future controlled-autotune seam

The only preserved seam is the existing cold authenticated tuning [handoff](../../../../glossary.md#term-handoff). A future separately
approved controlled environment may evaluate multiple **already correctness-approved** complete
route [identities](../../../../glossary.md#term-identity-meanings) for one exact [compatibility key](../../../../glossary.md#cache-and-memoization) and return one authenticated decision before
resource declaration. That future work must preserve the safe [heuristic](../../../../glossary.md#term-heuristic) when no decision exists,
session compatibility, strict [structural dominance](../../../../glossary.md#term-structural-dominance-index-mappings-and-cost-dominance) rules, and the prohibition on local timing as
authority.

This task adds no measurement [producer](../../../../glossary.md#term-producer), timer, benchmark, [matrix](../../../../glossary.md#term-matrix-meanings), [cache](../../../../glossary.md#term-cache), threshold, exploration,
background service, dynamic candidate generation, or [Runtime](../../../../glossary.md#term-runtime) decision. Unsupported/unapproved
routes never become candidates merely because the catalog says structurally possible.

## Implementation order

1. Recompute the 115-kind source inventory and current SDK selector names. If any count/state/reason
   differs, update this task first in a planning-only review; do not silently implement a different
   catalog.<span data-glossary-navigation="true"> ([SDK glossary](../../../../glossary.md#term-sdk); [planning glossary](../../../../glossary.md#term-planning))</span>
2. Add closed catalog state/reason enums and the exhaustive `NodeKind` catalog with zero behavior
   change.
3. Centralize existing route family/wire identity on the current closed route identity; remove the
   duplicate candidate wire truth while preserving encoded bytes and identities.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
4. Make the shared schema/capability/validation/value/buffer facts explicit before the
   route-specific construction switch. Delete any duplicated validation exposed by the refactor;
   retain route-required declarations.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [validation glossary](../../../../glossary.md#term-validation); [buffer glossary](../../../../glossary.md#term-buffer))</span>
5. Add the package-private test forcing seam and fail-closed approval gate. Do not route public or
   production callers through an override-bearing API.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
6. Add catalog exhaustiveness/count tests, fixed-route tests, independent Model-result-set route
   tests, no-fallback failure tests, and codec stability tests.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [codec glossary](../../../../glossary.md#term-codec))</span>
7. Update current Metal package/backend docs and planning status. Do not edit historical task or
   evidence claims except to add an explicit later-supersession sentence when genuinely necessary.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [backend glossary](../../../../glossary.md#backend))</span>
8. Run focused verification once. The explicit implementation direction supersedes the planned
   final full repository build and atomic-commit wording: run no final full build, remove throwaway
   artifacts, and retain logical implementation, remediation, and documentation commits.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [commit glossary](../../../../glossary.md#term-commit); [artifacts glossary](../../../../glossary.md#term-artifact); [commits glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>

## Required tests

Permanent tests earn their place only for these observable/[invariant](../../../../glossary.md#term-invariant) risks:

- all `NodeKind.values()` resolve exactly once and totals are [MPSGraph](../../../../glossary.md#term-mpsgraph) `76/34/5`, custom
  `16/99/0`;
- representative entries from every reason class and all five MPSGraph-unavailable kinds have the
  exact closed states/reasons;
- catalog state changes cannot widen capability or route eligibility;
- current [heuristic](../../../../glossary.md#term-heuristic) mapping remains exact for [singleton](../../../../glossary.md#term-singleton) NEG, ordinary wires `1..19`, mixed
  Task-0052 programs, and unsupported wires `35..115`;
- every plan has exactly one route identity before declaration/finalization and route-dependent
  workspace/internal-buffer invariants remain fail closed;<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- candidate wires and encoded compatibility/candidate/decision bytes remain stable unless an
  explicit coordinated identity bump is required;
- forced singleton NEG custom and MPSGraph routes each satisfy the [Model](../../../../glossary.md#term-model) result set independently;
- Task-0052 MPSGraph forcing and every unavailable/unapproved force fail before native create;
- create/[run](../../../../glossary.md#term-run) failure performs no alternate-route native call and publishes no result;
- concurrent invocations never mutate route [identity](../../../../glossary.md#term-identity-meanings) or share run-owned writable state; and
- no public Config/[Engine](../../../../glossary.md#term-engine)/[Metal](../../../../glossary.md#term-metal) [API](../../../../glossary.md#term-application-programming-interface-api) exposes route forcing or catalog state.

## [Validation](../../../../glossary.md#term-validation)

Implementation validation, not this [planning](../../../../glossary.md#term-planning) [commit](../../../../glossary.md#term-commit), must include:

1. focused catalog/route/candidate/[codec](../../../../glossary.md#term-codec)/preparation/finalization/execution tests;
2. focused [Metal](../../../../glossary.md#term-metal) capability and [conformance](../../../../glossary.md#term-conformance) tests proving the `34/81` [matrix](../../../../glossary.md#term-matrix-meanings) is unchanged;
3. native build plus exact export audit and [schema](../../../../glossary.md#term-schema)-13 malformed/unsupported tests;
4. packaged signed-library focused route tests when [native code](../../../../glossary.md#native-numerical-library-and-instruction-architecture) or package inputs changed;
5. public [Engine](../../../../glossary.md#term-engine) [singleton](../../../../glossary.md#term-singleton)-NEG tuned/untuned coverage if candidate plumbing changed;
6. architecture checks, documentation links/fences/final-newline checks, and `git diff --check`;
7. an audit [alignment](../../../../glossary.md#term-alignment) check proving wires `1..115`, exact kind/state equality, both fixed totals,
   every installed header selector citation, and every Model source link;<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
8. no final full repository build; the explicit implementation direction requires focused
   verification only; and
9. independent Class C review before `Complete`.

This planning-only commit runs documentation validation only. It runs no production build, [device](../../../../glossary.md#term-device)
probe, [oracle](../../../../glossary.md#term-test-oracle), benchmark, performance test, or native package build.

## Completion criteria

Task 0056 may become `Complete` only when:

- one executable exhaustive catalog accounts for all 115 kinds with the fixed state totals;
- the normative per-wire evidence audit remains exactly aligned with the catalog and every cited
  selector, installed header, Model kind, and bounded signature is still current;<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative); [Model glossary](../../../../glossary.md#term-model); [signature glossary](../../../../glossary.md#term-signature))</span>
- one [immutable](../../../../glossary.md#term-immutable) route [identity](../../../../glossary.md#term-identity-meanings) is embedded in every plan and used without hot reconsideration;
- both route families share the [schema](../../../../glossary.md#term-schema)-13/capability/[validation](../../../../glossary.md#term-validation)/[lifecycle](../../../../glossary.md#lifecycle)/[buffer](../../../../glossary.md#term-buffer)/[publication](../../../../glossary.md#term-publication) spine;
- all current production route decisions, capability rows, candidate bytes, and public APIs are
  unchanged unless an explicitly coordinated identity bump was required;
- internal forcing proves each approved route independently against Model result sets and proves no
  fallback/retry;
- local timing is absent from [qualification](../../../../glossary.md#term-qualification) and selection;
- focused validation passes and the explicitly excluded final full build is not [run](../../../../glossary.md#term-run); and
- independent review approves the implemented and remediated logical [commits](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings).

## Completion record

- `MetalOperationRouteCatalog` exhaustively describes all 115 `NodeKind` values with fixed totals
  MPSGraph <span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>`76/34/5` and custom `16/99/0`. Capability remains `34 true / 81 false`; catalog state
  neither admits an occurrence nor authorizes a route.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- `MetalPreparedRoute` is the one package-private [canonical](../../../../glossary.md#term-canonical-form) [identity](../../../../glossary.md#term-identity-meanings) for
  `CUSTOM_SINGLE_NEG=1`, `MPSGRAPH=2`, and `CUSTOM_TASK0052=3`; it owns the two-value family and
  candidate encoding delegates to it. Every prepared plan retains one final identity.
- The package-private forcing seam runs ordinary capability/semantic validation, authenticates
  every supplied handoff, and accepts only an exact freshly generated candidate. Independent
  custom/MPSGraph singleton-NEG result-set tests, incompatible-force tests, fixed-identity tests,
  and create/run no-fallback tests pass.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [run glossary](../../../../glossary.md#term-run))</span>
- Focused Java catalog/route/candidate/codec/preparation/finalization/execution/trace/tuning tests
  passed. Focused capability, backend-conformance, Metal/Engine public-shape, Metal Javadoc, and
  architecture checks passed.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [trace glossary](../../../../glossary.md#trace); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Metal glossary](../../../../glossary.md#term-metal); [Engine glossary](../../../../glossary.md#term-engine); [shape glossary](../../../../glossary.md#shape); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- A fresh native build retained exactly thirteen defined exports. Two raw ABI/schema-13 malformed
  and unsupported-operation tests plus one prepared native route test passed with zero skips. Two
  public Engine singleton-NEG tuned/untuned integration tests passed with zero skips.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema); [operation glossary](../../../../glossary.md#operation))</span>
- Documentation link/anchor/fence/newline/whitespace [validation](../../../../glossary.md#term-validation) and `git diff --check` passed.
  No performance run, benchmark, numerical oracle, package/archive action, or final full
  repository build was run.<span data-glossary-navigation="true"> ([numerical oracle glossary](../../../../glossary.md#adr-ci-and-qualification))</span>
- Independent Class C review found one P2 empty-[handoff](../../../../glossary.md#term-handoff) [authentication](../../../../glossary.md#term-authentication) defect. Remediation
  `1718b28a` authenticates exact partition and compatibility before optional-decision inspection
  or route forcing; final re-review returned <span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition))</span>`APPROVE` with zero remaining findings.
- Follow-up external final review found the separate wire-39 cast-reason mismatch. The dedicated
  closed `MD_CAST` correction aligns the catalog with `M-CAST` and
  `SHAPE::castTensor:toType:name:`; exhaustive catalog coverage still passes and capability remains
  `34 true / 81 false`.
