# Compile API

## Purpose and implementation status

This reference separates the compile-time model values implemented today from the compiler and
engine APIs that remain planned. The repository does not yet provide a runnable graph compiler.

Compilation will answer two questions: what the computation means, and which backend identity owns
each planned region. It will not create physical buffers, choose concrete kernels, or construct
prepared executables.

## Current model contracts

The `io.github.pho001.synaptik.model.operation`, `io.github.pho001.synaptik.model.tensor`, and
`io.github.pho001.synaptik.model.graph` packages now provide compiler-neutral data that later
compiler work can produce and consume:

| Current contract | Meaning | Deliberate boundary |
|---|---|---|
| `OperationSignature` | Exact attributes variant and inclusive local input/output-count bounds | Structural occurrence contract, not operand inference or executable support |
| `TensorProducer` | Identity of one pre-capture expression occurrence with exact operation, ordered input Tensors, and ordered output descriptors | Public-expression origin, not `CompiledNode`, graph membership, or compiler state |
| `TensorProvenance` | Exact producer plus one zero-based output index, with derived operation, inputs, and descriptor | Indexed result association, not `NodeId`, `ValueId`, or graph output binding |
| `GraphRngState` | Opaque public wrapper around one explicit key/counter state-expression occurrence | Expression identity and threading boundary, not a generator, exposed numerical Tensor, runtime state, or bitstream |
| `CompiledNode` | One operation occurrence with ordered input/output `ValueId` snapshots | Local list and signature-cardinality validation, not graph-wide closure or descriptor compatibility |
| `CompiledGraphModel` | Immutable ordered graph values, topological nodes, declared input/output boundaries, and exact node phases | Structural graph state, not compiler passes, partitions, storage, or execution |
| `GraphPhase` | Exactly `FORWARD` or `BACKWARD` compile-time node classification | Not a compile mode, optimizer phase, or runtime schedule |
| `PublicationBinding` | Standalone `TensorId`-to-`ValueId` association | Not an owning publication plan and not a `CompiledGraphModel` component |

`CompiledGraphModel` validates structural closure when constructed. It snapshots its lists and
phase map, requires resolvable references and topological node order, enforces producer and phase
coverage rules, and stores no derived indexes. This validation does not capture an expression,
infer descriptors, transform a graph, perform autograd, plan backend ownership, or make the model
executable.

Before a node enters that container, `Operation` validates its exact kind/attributes pairing and
derives a family-owned `OperationSignature`. `CompiledNode` preserves its existing local list
rules and then checks that the final ordered input and output counts lie within that signature's
inclusive bounds. This catches a unary operation connected to two inputs, for example, without
claiming that operand Shapes or data types are compatible. Zero-input, bounded, variadic-input,
and multi-output occurrences remain representable when their kind explicitly declares them.

A `PublicationBinding` carries only two identities. A later compiler-owned `PublicationPlan` will
group bindings with their owning compilation context and publication policy. The binding itself
does not retain a public `Tensor`, gradient role, runtime target, storage, backend, or execution
state.

`GraphRngState.initial(key, counter)` is also current model construction. It creates a fresh
zero-input, one-output `GraphRngKind.INITIAL_STATE` occurrence whose exact
`GraphRngStateAttrs(long key, long counter)` words are retained in provenance. The private state
Tensor is fixed `INT64 Shape[2]`, unresolved-layout, non-gradient, unlabeled, and storage-free,
with output index zero. Both `long` values are unsigned 64-bit bit patterns: key identifies a
caller-selected stream/domain, and counter identifies the next abstract logical sample position.
Equal words request replay-equivalent positions but do not merge expression identities.

This is compiler-visible semantic input, not implemented compiler support. A future capture pass
may preserve the state producer and ordered state edges, and a future serializer must preserve the
raw words losslessly. This task defines neither capture nor serialization, common-subexpression
policy, gradient rules, a byte encoding, a stable enum token, a random algorithm, prepared state,
or execution. Without a selected portable algorithm, no cross-backend or cross-version bitstream
follows from equal state.

The public `Tensor` model is also current. Its seven binary arithmetic methods, six binary
comparison methods, three boolean logical methods, nineteen unary elementwise methods, three
floating-classification methods, seven exact-typed plus seven exact-FLOAT64 scalar arithmetic
methods, and six range/one-bound clamp
methods, plus one static conditional-selection method and one explicit cast method, fifteen
full/axis numeric aggregate methods, one binding-aware target-Shape SUM method, six full/axis
boolean aggregate methods, fourteen ordinary
multi-axis methods, twelve floating advanced/statistical reduction methods, two axis-removing
masked aggregate methods, six axis-only `argMin`/`argMax` methods, two one-axis `cumSum` methods,
and two one-axis `cumProd` methods, plus one-axis `softmax`, `logSoftmax`, two trailing-Shape
`layerNorm`
methods, and two trailing-Shape `rmsNorm`
methods, plus explicit-axis five-input `batchNormInference` and `batchNormTraining` methods,
scalar `select`, two
tensor-index axis-gather methods, the `embedding` convenience over axis-zero Gather, one
trailing-axis `oneHot` index-encoding method, and two Gather-ND methods, plus two functional
Scatter Elements methods, one Gather-compatible functional Scatter Add method, and three
functional Scatter-ND methods, plus one matrix-multiplication method, construct
storage-free expressions with immutable producer-and-output-index provenance. Four full-ordering
methods add ascending or explicitly directed `sort` and `argsort` requests. Every current
single-output expression creates one identity-distinct producer whose ordered descriptor list has
one entry and whose provenance index is zero. The producer snapshots the exact operation and input
Tensor references and retains no output Tensor objects. The
parameterless `contiguous` method
adds the same expression provenance for a canonical-layout request, and the two `reshape`
overloads add normalized target-shape expressions. Two `expand` overloads add directional
right-aligned target-shape expressions.
Binary ADD, SUB, MUL, MIN, and MAX now accept same-category floating or signed-integral operands;
DIV and POW remain floating-only. All six comparisons accept the same floating or integral
pairing rules and produce BOOL results. Integral promotion uses `INT32 < INT64`, with conceptual
sign extension into the INT64 domain, while mixed numeric categories require an explicit cast and
BOOL is excluded. Numeric unary and conditional-selection results remain floating. Unary
`rsqrt`, `log1p`, `expm1`, exact `gelu`, fixed `geluTanhApproximation`, and canonical `silu` are
first-class transforms rather than stored decompositions. Exact GELU selects `x * Phi(x)`; the
tanh spelling selects only its fixed conventional `0.044715` approximation; and SiLU selects
`x * sigmoid(x)`. There is no `swish` alias or configurable approximation. Together with `exp`
and `tanh`, they record portable mathematical requests without selecting an algorithm, bitwise
result, fixed accuracy bound, gradient rule, or backend route. Floating classifications,
comparisons, and logical results are unresolved-layout `BOOL` descriptors with false gradient
eligibility. Logical AND and OR require exact BOOL inputs and derive a local broadcast shape;
logical NOT requires exact BOOL and retains the exact input shape. Scalar parameters remain exact
typed `ScalarValue` operation attributes rather than Tensor inputs. Scalar ADD, SUB, MUL, MIN, and
MAX plus one-bound clamp conveniences accept exact matching floating or integral values; scalar
DIV, POW, and first-class range CLAMP remain floating-only. Constant padding uses exact equality
for all six current data types. Tensor-to-Tensor and scalar arithmetic share the seven semantic
kinds, while the accepted integral subset is the five operations above; public pairwise extrema
are named `minimum`/`maximum`, while aggregate reductions remain `min`/`max`. Floating extrema
propagate NaN, order infinities normally, and select negative zero for minimum or positive zero
for maximum. Range
`CLAMP` remains first-class; `clampMin` creates one scalar `MAX` producer and `clampMax` one scalar
`MIN` producer. Integral ADD, SUB, and MUL have fixed-width two's-complement modular meaning, and
integral MIN, MAX, and comparisons use signed order. These are current model semantics and
metadata-construction facts, not compiler
capture, validation, gradient, lowering, backend, or execution claims. An `OperationSignature`
still validates only the exact attribute class and occurrence cardinality because an `Operation`
has no operand
descriptor. Future compiler graph validation is expected to revalidate operand domains and exact
scalar/input data-type equality for captured or otherwise constructed occurrences; no compiler
capture or revalidation is implemented today. `Tensor.where` requires an exact BOOL
condition, promotes two floating branches, composes branch-first and condition-second local
broadcasts, propagates gradient eligibility from the branches only, and records exact ordered
condition/true-branch/false-branch provenance. It constructs no selected values or gradient rule.
`Tensor.cast` accepts every current source/target data-type pair, retains the exact input shape,
leaves layout unresolved, and retains a true gradient request only for floating-to-floating casts.
Every call remains a fresh explicit expression, including a same-type request, with typed target
attributes and exact one-input provenance.
`Tensor.sumToShape(Shape)` currently constructs one fresh numeric SUM expression whose descriptor
and `SumToShapeAttrs` retain the exact target Shape. The target is right-aligned with the input:
leading axes reduce, aligned target-one axes reduce and remain, and equal aligned axes preserve.
Provable static incompatibility fails locally; any pair involving an unresolved Dimension remains
an obligation for later binding validation. The result preserves exact input type and gradient
eligibility, has unresolved layout, and records exact one-input/output-index-zero provenance.
This is model metadata that a future compiler may capture; current compiler capture, constraint
representation or proof, adjoint construction, canonicalization, lowering, backend selection, and
execution remain unimplemented.
`Tensor.matmul` currently constructs one fresh two-input MATMUL expression with a locally derived
vector, matrix, or broadcast-batch Shape and same-category promoted numeric type. Unequal static
contraction dimensions fail locally; unresolved contraction equality and the accepted
unresolved-versus-static batch singleton-or-equal cases remain obligations for later compiler
validation or concrete binding. This is current model-expression metadata only: expression
capture, graph-wide validation, constraint proof, gradients, lowering, backend support, and
execution remain planned.
`Tensor.linear(weight)` and `Tensor.linear(weight, bias)` are also current model construction, but
they add no LINEAR operation. Conventional `[outFeatures, inFeatures]` weight is explicitly
transposed through PERMUTE `[1, 0]`, followed by MATMUL and optional exact rank-one
`[outFeatures]` ADD bias. Complete caller-controlled validation precedes the first intermediate.
No-bias returns the MATMUL product after two wrapper/ID allocations; biased construction returns
ADD after three. The biased final Shape is structurally equal to the product Shape and reuses its
ordered Dimension references, although the outer Shape object may differ. These visible primitive
producer chains are inputs a future compiler may capture and canonicalize; this page does not
claim capture, linear-pattern recognition, fusion, gradient construction, lowering, backend
support, or execution.
`Tensor.scaledDotProductAttention` is current first-class model construction with three ordered
query/key/value inputs and an optional fourth BOOL mask input. It locally derives the promoted
type and exact broadcast-batch/output Shape, retains optional exact scale and causal eligibility,
and records one `SCALED_DOT_PRODUCT_ATTENTION` output. Unresolved embedding positivity/equality,
key/value sequence equality, batch singleton-or-equal, and mask singleton-or-equal facts remain
obligations for later compiler validation or concrete binding. This is compiler-visible metadata,
not current compiler support: no capture, revalidation, constraint representation, legal
decomposition, attention gradient, saved value, backend lowering, or execution is implemented.
`Tensor.conv2d(weight, attrs)` and `Tensor.conv2d(weight, bias, attrs)` are current first-class
grouped NCHW cross-correlation model construction. Each call records one `CONV2D` occurrence with
exact ordered inputs, intrinsic stride/padding/dilation/group attributes, promoted floating
descriptor, and static or canonical-symbolic output Shape. Input and weight have Shapes
`[N, C_in, H, W]` and `[C_out, C_in/groups, K_h, K_w]`; optional bias has `[C_out]`.
Unresolved channel divisibility, grouped weight/input equality, bias/output equality, and dynamic
spatial non-negativity remain obligations for future compiler validation or concrete binding.
This metadata does not mean the current repository compiles convolution. Capture, constraint
representation and proof, legal decomposition, convolution gradients/adjoints and saved values,
backend lowering, algorithm selection, and execution remain planned in their owning layers.
`Tensor.maxPool2d(attrs)` is current first-class NCHW maximum-pooling model construction. One
`MAX_POOL2D` occurrence records exact ordered input `[input]`, `MaxPool2dAttrs`, one output at
index zero, the unchanged floating type and gradient request, exact batch/channel Dimensions, and
static or canonical-symbolic floor/ceil spatial extents. Dynamic spatial non-negativity remains a
future compiler-validation or concrete-binding obligation. Literal ceil mode retains every
ceiling-grid window, even when the terminal window is all-padding; padding exclusion, negative-
infinity empty windows, NaN propagation, signed-zero ordering, and first-logical-sample ties are
semantic metadata. This does not mean the current repository compiles pooling. A future compiler
owns capture, operand revalidation and binding proof, any legal decomposition, gradients and
adjoints, and any saved-index decision. Backend lowering, algorithms, kernels, and execution
remain planned in their owning layers.
`Tensor.averagePool2d(attrs)` is current first-class NCHW average-pooling model construction. One
`AVERAGE_POOL2D` occurrence records exact ordered input `[input]`, `AveragePool2dAttrs`, one output
at index zero, the unchanged floating type and gradient request, exact batch/channel Dimensions,
and static or canonical-symbolic floor/ceil spatial extents. Literal ceil mode retains terminal
all-padding windows. Its semantic metadata fixes a `kernelHeight * kernelWidth` divisor,
conceptual positive-zero padding that counts in that divisor, FLOAT32 accumulation/division for
BFLOAT16 and FLOAT32, FLOAT64 accumulation/division for FLOAT64, one final division, and the
documented NaN/infinity/signed-zero/all-padding policies. Dynamic spatial non-negativity remains a
future compiler-validation or concrete-binding obligation. This does not mean the repository
currently compiles average pooling. A future compiler owns capture, operand revalidation and
binding proof, any legal decomposition that preserves the fixed divisor and special-value
meaning, and gradient or adjoint construction. Backend lowering, algorithms, tolerances, kernels,
and execution remain planned in their owning layers.
`Tensor.sort(axis[, descending])` and `Tensor.argsort(axis[, descending])` currently construct
distinct stable, one-input, one-output ordering expressions. Both normalize the axis, preserve the
exact input Shape reference, leave layout unresolved, and use fixed NaN-last ordering in both
directions with stable logical-index ties. Sort preserves input type and gradient eligibility;
argsort uses non-differentiable INT64. These are current model-expression and provenance facts,
not implemented compiler capture, operand revalidation, gradient construction, algorithm
selection, lowering, backend support, runtime behavior, or execution.
`Tensor.topK(k, axis[, largest, sorted])` currently constructs one compiler-neutral
`TopKKind.TOP_K` model occurrence with ordered outputs `[values, indices]`. The values output
preserves input type and gradient eligibility; indices are non-differentiable INT64. Both share
one fresh Shape whose selected dimension is static `k`, one exact producer, and provenance slots
zero and one. A known static selected extent is checked during Tensor construction. When that
extent is dynamic or expression-based, the obligation `bound extent >= k` is deliberately
deferred: future compiler or binding validation must reject an insufficient bound rather than
clamp, pad, wrap, or reduce the output count. These are current model-expression facts, not a
claim that a compiler captures TOP_K, infers or revalidates its descriptors, constructs gradients,
selects an algorithm, lowers it, reports backend support, or executes it.
`Tensor.embedding(indices)` currently validates a rank-two floating weight receiver and exact
INT32/INT64 indices, then constructs the existing ordinary axis-zero GATHER occurrence directly.
The result Shape is the complete indices Shape plus the exact weight axis-one Dimension; result
type and gradient eligibility come only from weights. This is public model-expression construction,
not compiler support: capture, constant-index analysis, repeated-index autograd construction,
dynamic binding, safe bounds enforcement, lowering, backend support, and execution remain owned by
later lifecycle layers. No `EMBEDDING` kind or padding/sparse/max-norm/frequency option exists.
`Tensor.oneHot(depth)` currently validates exact INT32/INT64 receiver metadata before positive
static depth, preserves every input Dimension reference, and appends one fresh
`StaticDimension(depth)`. It constructs one storage-free, non-differentiable BOOL result with one
`ONE_HOT` producer and exact sole-input provenance. This is a current model-expression inventory
entry, not compiler support. Construction reads no index values; valid eventual execution
requires `0 <= i < depth`, while capture, constant analysis, dynamic bounds enforcement,
gradients, lowering, backend support, and execution remain planned in their owning layers.
`Tensor.rsqrt`, `log1p`, `expm1`, `gelu`, `geluTanhApproximation`, and `silu` accept floating input,
retain its exact type, Shape, and gradient eligibility, leave layout unresolved, and record
one-input parameterless provenance.
Their selected special-value semantics distinguish signed zero, infinities, and NaN, but current
construction neither reads values nor promises correct rounding or a fixed numerical tolerance.
`Tensor.isFinite`, `isNaN`, and `isInf` accept floating input and construct fixed BOOL results with
the exact input Shape, unresolved layout, false gradient eligibility, and one-input parameterless
provenance. They record graph-visible value classifications; they do not eagerly classify host
storage or define compiler validation, gradients, lowering, backend support, or execution.
`Tensor.sum`, `prod`, reduction `min`, and reduction `max` accept floating or signed-integral
inputs; `mean` remains floating-only. They construct full, axis-removing, or retained-axis
expressions. Full forms have canonical rank-zero Shape and use the canonical no-attributes
singleton; axis forms normalize the caller axis and store `AxisReductionAttrs`. Every result
preserves exact input type and gradient eligibility, leaves layout unresolved, and records
one-input provenance. Integral sum/product use exact-type modular arithmetic modulo `2^32` or
`2^64` with reassociation permitted; integral min/max use signed order. Their empty identities are
zero, one, the input type's maximum, and the input type's minimum, respectively, for full domains
and empty selected-axis slices. Construction records these semantics without aggregating or
comparing values, implementing a gradient or numerical algorithm, or providing compiler,
lowering, backend, or executable behavior. Aggregate `MIN`/`MAX` remain typed separately from the
equally named two-input binary elementwise kinds.
The masked `Tensor.sum(axis, mask)` and `Tensor.mean(axis, mask)` forms require floating input and
an exact BOOL mask. They require ordinary right-aligned broadcasting of the mask to produce
exactly the input Shape; callers make other axis intent visible with an explicit reshape,
dimension insertion, or expansion. Each form removes the normalized axis and records one
first-class two-input `SUM` or `MEAN` occurrence with `MaskedReductionAttrs(axis)` and ordered
`[input, mask]` provenance. False positions exclude their inputs, including NaN and infinity;
an all-false sum is zero and an all-false mean is NaN without a payload guarantee. Construction
does not align storage, inspect values, select elements, count true positions, compute a result,
capture or decompose the occurrence, define gradients, lower it, or execute work.
`Tensor.all` and `Tensor.any` require exact BOOL input and construct full, axis-removing, or retained-axis
expressions with exact BOOL result type, false gradient eligibility, unresolved layout, and
one-input provenance. Aggregate `ALL`/`ANY` remain typed separately from elementwise `AND`/`OR`.
Construction does not inspect truth values or define empty-domain identities, compiler behavior,
backend support, or execution.
`Tensor.argMin` and `Tensor.argMax` accept floating or integral input and one positive or negative
axis. Their convenience forms explicitly use `FIRST_INDEX`, while complete forms retain an
explicit first- or last-index policy in shared `ArgExtremaAttrs`. Axis removal or retention follows
the ordinary structural Shape rules, but every result is fixed unresolved-layout INT64 with false
gradient eligibility, one-input provenance, and output index zero. Integral candidates use signed
order. Floating candidates prefer NaN for both extrema directions, treat multiple NaNs as ties,
order negative zero below positive zero, and order infinities normally. Construction rejects a
statically empty selected axis, accepts an unselected zero axis and unbound selected extent, and
does not compare values, select an index, define gradients, capture or validate a compiled graph,
lower, report backend support, or execute. A future compiler must prove or validate that a dynamic
selected extent is positive before index selection, but its callable API and failure type remain
unspecified.
The 26 multi-axis/statistical methods accept ordered distinct positive or negative axes. Caller
order is retained in immutable `MultiAxisReductionAttrs` or `StatisticalReductionAttrs`; Shape
derivation uses membership to remove selected axes or retain them with extent one. An empty axis
list selects a point domain and creates a fresh occurrence rather than requesting full reduction.
Ordinary type domains remain unchanged. Floating-only `LOG_SUM_EXP`, `VARIANCE`,
`STANDARD_DEVIATION`, `L1_NORM`, and `L2_NORM` preserve exact input type/eligibility and remain
first-class operations rather than stored decompositions. Variance and standard deviation default
to correction zero or retain explicit non-negative correction, rejecting a statically known
domain count at most correction while deferring dynamic proof.

The model now fixes floating and BOOL empty/special-value reduction semantics, including NaN,
infinity, signed zero, positive-zero/one or infinite identities, ALL/ANY identities, stable
log-sum-exp targets, corrected statistical formulas, and non-negative norm targets. These are
compiler-visible requested meanings, not implemented capture, operand revalidation, numerical
algorithms, gradients, lowering, backend support, or execution. A future compiler must revalidate
dynamic corrected-domain counts before execution without inventing a callable API here.
`Tensor.cumSum` and `Tensor.cumProd` are current model expressibility for shape-preserving
cumulative addition and multiplication. Each family accepts floating or integral input and one
positive or negative axis. Each short form explicitly selects inclusive forward traversal; each
complete form retains exact exclusive and reverse flags. Every result retains the exact input
Shape, data type, and gradient eligibility, leaves layout unresolved, and records the matching
`CUM_SUM` or `CUM_PROD` with exact one-input provenance. Integral cumulative product has exact-
width modular meaning; floating product has the selected NaN, zero-times-infinity, sign-parity,
and positive-one identity semantics documented by the Tensor API. Construction does not read or
accumulate values.

Those model expressions are compiler-visible inputs, not implemented compiler adoption. Capture,
dynamic validation, canonicalization, product-adjoint construction from prefix and suffix scans,
gradient boundary policy, saved-value lifetime, numerical lowering, backend support, and execution
remain planned. In particular, the current compiler API exposes no callable compilation path that
turns either scan kind into executable work.
`Tensor.softmax` and `Tensor.logSoftmax` accept floating input and one positive or negative axis.
Every result retains the exact input Shape, data type, and gradient eligibility, leaves layout
unresolved, and records the requested first-class SOFTMAX or LOG_SOFTMAX kind with exact one-input
provenance. Construction does not read values, calculate probabilities or logarithms, select a
numerical algorithm, define a gradient rule, capture or decompose a graph operation, lower a
backend operation, or execute work.
`Tensor.layerNorm` is current model metadata construction for exact trailing-Shape population
normalization. The no-affine form records `LayerNormAttrs` and ordered input `[input]`; the affine
form records `AffineLayerNormAttrs` and ordered inputs `[input, scale, bias]`. Both retain exact
normalized Shape and typed epsilon parameters, exact input result Shape, one output at index zero,
and no saved-statistic output. Local construction rejects known static mismatches and defers an
unresolved trailing-dimension equality when output Shape is still exact. This is compiler-visible
metadata, not compiler support: capture, operand revalidation, representing and proving deferred
constraints, saved-statistic lifetime, gradients or adjoints, legal decomposition, lowering, and
execution remain planned in their owning layers.
`Tensor.rmsNorm` is current model metadata construction for exact trailing-Shape uncentered
root-mean-square normalization. One `RmsNormAttrs(normalizedShape, epsilon)` value supports the
safe ordered inputs `[input]` and `[input, scale]`, each with exactly one output at index zero. The
result retains the exact input Shape; the scaled form requires scale Shape exactly equal to the
normalized Shape. Local construction rejects known static trailing mismatches and defers
unresolved equality. This metadata does not imply compiler support: capture and operand
revalidation, representation and proof of deferred constraints, compiler-generated saved values,
gradients or adjoints, legal decomposition, lowering, backend preparation, tolerance enforcement,
and runtime execution remain planned in their owning layers.
`Tensor.batchNormInference` is current stateless model metadata construction with exact ordered
inputs `[input, scale, bias, runningMean, runningVariance]` and exactly one output at index zero.
It retains `BatchNormInferenceAttrs(normalizedChannelAxis, epsilon)`, preserves the exact input
Shape, requires rank-one `[C]` affine/statistic vectors, defers unresolved channel-extent equality,
and uses ordered floating promotion with exact-result-typed positive epsilon. The channel axis is
layout-neutral, and running variance is interpreted directly by the formula with epsilon inside
the denominator square root. This remains compiler-visible metadata, not implemented compiler
support: capture, operand revalidation, deferred-constraint representation and proof, saved-value
and gradient construction, legal decomposition, lowering, backend preparation, tolerance
enforcement, runtime execution, and numerical evaluation remain planned in their owning layers.
`Tensor.batchNormTraining` is current five-output model metadata with exact ordered inputs
`[input, scale, bias, runningMean, runningVariance]`. Its producer describes normalized output,
next running mean, next running variance, saved batch mean, and saved inverse standard deviation
in that order. `BatchNormTrainingResult` exposes positions zero through two; positions three and
four remain real producer descriptors for future compiler capture. The model records a normalized
channel axis, exact typed new-batch-weight momentum and epsilon, the all-non-channel reduction
domain, biased forward variance, correction-one running variance, output Shapes, types, gradient
eligibility, and indexed provenance. This does not implement capture of any position, deferred
`C == 0 || N >= 2` proof, saved-value materialization or lifetime, autograd/backward construction,
publication, liveness, graph optimization, lowering, preparation, execution, or cross-step
statistic ownership. Those remain planned in the compiler, training extension, runtime, and
backend layers that own them.
`Tensor.meanSquaredError(target, reduction)` is current one-output model metadata with exact
ordered inputs `[prediction, target]`. It records `LossKind.MEAN_SQUARED_ERROR` and one
`MeanSquaredErrorAttrs` carrying explicit `NONE`, `SUM`, or `MEAN` reduction. Local construction
accepts only floating inputs, promotes them in input order, rejects unequal known static Shapes
without broadcasting, and defers unequal pairs involving an unresolved Dimension. `NONE` retains
the exact prediction Shape; `SUM` and `MEAN` use the scalar Shape. Mean divides by the complete
logical element count, so scalar count is one, empty sum is positive zero, and empty mean is NaN.
This is compiler-visible requested meaning only. Capture, revalidation, representation and proof
of deferred equality, gradient or adjoint construction, legal decomposition, optimization,
lowering, backend support, runtime execution, and training coordination remain planned in their
owning layers.
`Tensor.categoricalCrossEntropyWithLogits(target, classAxis, reduction)` is current one-output
model metadata with ordered inputs `[logits, target]`. Exact floating target type dispatches to the
unchanged dense target-weighted stable-log-softmax meaning, including floating promotion,
combined gradient eligibility, exact logits/target Shape compatibility, and sample-count mean.
Exact INT32 or INT64 target type dispatches to selected-class negative log-softmax: the target
Shape is constrained to logits Shape without the normalized class axis, result type and gradient
eligibility come only from logits, `NONE` retains exact target Shape, and `MEAN` divides by target
count. The four-argument overload adds one exact target-typed INT32/INT64 `ScalarValue` ignore
index. Matching ignore precedes bounds and logits evaluation, contributes positive zero, and is
excluded from the mean denominator; it is attributes metadata, so provenance remains exactly
`[logits, target]`.

Construction reads no values. Dense target normalization, unresolved mapped Shape equality,
non-ignored index bounds, and class-extent alternatives therefore remain obligations. Empty or
all-ignored index means are NaN and sums are positive zero; with ignore, a zero class extent may
remain valid when every target is ignored. This is compiler-visible requested meaning only. It
does not implement capture, revalidation, constant analysis, proof, bounds checks, gradients or
adjoints, decomposition, optimization, lowering, preparation, execution, publication, or training
coordination; those remain planned in their owning lifecycle layers.
`Tensor.contiguous()` accepts every current data type and preserves the exact Shape, data type, and
gradient eligibility. It creates new canonical dense row-major, zero-offset layout geometry for a
fully static Shape and leaves a dynamic Shape unresolved. Every call is fresh, unlabeled, and
storage-free, records `CONTIGUOUS` with the canonical no-attributes singleton and exact one-input
provenance, and does not inspect input layout, storage, or values. Resolved result geometry does
not allocate or copy storage. Compiler capture, redundant-request canonicalization,
materialization policy, lowering, and execution remain planned.
`Tensor.reshape(long...)` accepts all current data types, normalizes an empty request or one
inferable `-1`, and rejects locally provable invalid counts. `Tensor.reshape(Shape)` retains an
exact normalized target and defers count equality when either Shape is dynamic. Both overloads
retain type and gradient eligibility, record exact RESHAPE/target-shape semantics with one-input
provenance, and stay unlabeled and storage-free. Only resolved contiguous input plus a static
target produces same-offset canonical view metadata; all other result layout remains unresolved.
This is current model expression construction, not compiler capture, graph-wide dynamic constraint
solving, reshape-chain canonicalization, materialization planning, backend alias/copy lowering, or
execution.
`Tensor.expand(long...)` treats every requested extent as a literal non-negative dimension, while
`Tensor.expand(Shape)` retains the exact target reference. Both overloads require target rank at
least input rank and accept aligned dimensions only when they are structurally equal or the input
is a static singleton; new leading target axes are valid. A fully static target and any resolved
input layout produce new same-offset view geometry with preserved aligned strides and zero strides
for leading or expanded-singleton axes. Dynamic target or unresolved input geometry stays
unresolved. Every result retains exact type and gradient eligibility, records exact
EXPAND/target-shape semantics with one-input provenance, and remains fresh, unlabeled, and
storage-free. This is current model construction, not value repetition, storage aliasing, dynamic
constraint solving, gradient behavior, compiler capture or canonicalization, materialization,
lowering, or execution.
`Tensor.permute(int...)` accepts every current data type, requires a complete output-to-input axis
mapping, normalizes each negative axis once, and reorders exact Dimension references. Any resolved
input layout produces a new same-offset view descriptor with exact reordered strides; unresolved
input layout remains unresolved. `Tensor.transpose()` requires rank two and uses the same PERMUTE
construction with normalized axes `[1, 0]`. Every result preserves type and gradient eligibility,
records exact normalized attributes and one-input provenance, and remains fresh, unlabeled, and
storage-free. This is current model expression construction, not graph capture, permutation
canonicalization, physical aliasing, materialization, lowering, or execution.
`Tensor.expandDims(int)` inserts one static singleton at a caller position normalized against the
result rank. `Tensor.squeeze(int)` instead normalizes an existing input axis and removes it only
when its dimension is statically known as one. Both preserve exact type and gradient eligibility,
retain exact unaffected Dimension references, record matching `AxisTransformAttrs` and one-input
provenance, and leave unresolved input geometry unresolved. Resolved input geometry produces one
new same-offset logical view descriptor with a deterministic checked stride inserted or the
selected stride removed. This is current model expression construction, not graph capture,
dynamic singleton constraint solving, inverse-pair canonicalization, physical aliasing,
materialization, gradient behavior, lowering, or execution.
`Tensor.slice(long[], long[], int[], long[])` clones four parallel request arrays, normalizes raw
axes and bounds against selected static dimensions, clamps bounds by step direction, derives a
same-rank Shape, and records normalized start/length/axis/signed-step sequences in `SliceAttrs`.
Empty arrays and zero-extent results are valid. Resolved non-empty all-positive geometry produces
checked start-adjusted, step-multiplied logical view metadata; unresolved input, empty results, and
any negative step remain layout-unresolved. `Tensor.sliceAxis(int, long, long)` supplies one
step-one entry, its four-argument overload supplies an explicit signed step, and `Tensor.flip`
creates one negative-step SLICE occurrence for explicit axes. Empty flip axes mean identity.
Every success preserves exact type and gradient eligibility, records one identity-distinct
producer with exact `[input]`, one output descriptor, and provenance index zero, and remains fresh,
unlabeled, and storage-free. This is current model expression construction, not graph capture,
slice-chain or flip canonicalization, physical aliasing or copying, gradient-scatter construction,
materialization, backend/ONNX lowering, or execution.
`Tensor.sliceUpdate(update, starts, axes, steps)` is now current model construction for functional
signed multi-axis replacement. It derives normalized finite `SliceAttrs` lengths from selected
static update Dimensions, requires exact base/update type and same-rank Shape compatibility,
retains the exact base Shape, combines eligibility, leaves layout unresolved, and records one
`SLICE_UPDATE` occurrence with ordered `[base, update]` provenance. Static first/final coordinate
bounds fail locally. A non-negative start on an unresolved base axis retains its upper-bound
obligation for later binding or execution; no start is clamped or shifted to fit.
`Tensor.cropToShape(targetShape, prefixShape)` is also current model construction. It records one
`SLICE` occurrence with exact `CropToShapeAttrs`, retains the supplied target Shape as the result,
and interprets the prefix Shape as per-axis logical extents preceding the region. Fully static
`prefix + target <= input` bounds are checked locally; inequalities involving any unresolved
Dimension remain deferred. Neither primitive captures a graph, represents a constraint, constructs
an adjoint, mutates values, chooses materialization, lowers, or executes work. A later compiler may
compose these public primitives for Slice, Select, Pad, or Concat adjoints, but no such traversal,
binding proof, saved-value policy, accumulation, or canonicalization is implemented today.
`Tensor.select(int, long)` normalizes one source axis and one scalar coordinate, removes the
selected Dimension, preserves every unaffected exact Dimension reference, and records normalized
`SelectAttrs` with exact one-input provenance. A static selected extent supplies immediate
negative-index normalization and bounds checks; a non-negative coordinate on a dynamic selected
extent remains representable with its upper bound deferred, while a negative coordinate is
rejected. Resolved input geometry with a non-empty result produces checked selected-stride removal
and offset advancement in one new logical view descriptor; unresolved input and empty results stay
unresolved. The fresh result preserves exact type and eligibility and has no label or storage.
This is current model expression construction, not value selection, physical aliasing, gradient
construction, graph capture or canonicalization, materialization, backend lowering, or execution.
`Tensor.gather` and `gatherElements` consume exact ordered `[data, indices]` inputs with `INT32` or
`INT64` indices. They normalize one data axis and apply canonical axis-replacement or same-rank
aligned-Shape rules. Every fresh result retains data type and gradient eligibility, leaves layout
unresolved, and records exact two-input provenance. Construction interprets no index
values, checks no index-value bounds, and adds no gradient rule, graph capture, canonicalization,
materialization, lowering, or execution behavior.
`Tensor.gatherNd` consumes exact ordered `[data, indices]` inputs with `INT32` or `INT64` indices.
Its short form uses zero shared batch Dimensions; its complete form retains one normalized
non-negative batch count. Construction validates both ranks, structurally equal leading batch
Dimensions, and a statically known positive tuple depth from the final indices Dimension. It
derives the result as the indices prefix without tuple depth followed by the untouched data
suffix, including canonical scalar and exact retained Dimension references. Every result is fresh,
unlabeled, storage-free, and unresolved-layout, preserves data type and gradient eligibility, and
records `GATHER_ND` with exact `[data, indices]` provenance. It reads no index value, checks no
index-value bound, and adds no gradient, graph capture, compiler transformation, materialization,
lowering, or execution behavior.
Both `Tensor.scatterElements` overloads consume exact ordered `[data, indices, updates]` inputs.
They require `INT32` or `INT64` indices and exact matching data/update types. `NONE` is permitted
for every current type and arithmetic reductions for floating or integral values. Each method
normalizes one raw data axis, validates the same-rank Shape relationship, and returns a fresh
result with the exact data Shape/type, data/update gradient-
eligibility OR, unresolved layout, and exact three-input provenance. Construction reads no values,
checks no index bound or duplicate target, mutates no input, and performs no write or reduction.
It adds no gradient rule, graph capture, compiler transformation, materialization, lowering,
backend behavior, or execution behavior.
`Tensor.scatterAdd(indices, updates, axis)` is now the current Gather-compatible fixed-add model
primitive. It records `SCATTER_ADD` with normalized `IndexAxisAttrs` and ordered
`[data, indices, updates]` inputs. Updates must have the exact Shape that `data.gather(indices,
axis)` would produce, while the functional result retains the exact data Shape and accumulates
duplicate targets. Construction validates metadata only and leaves layout unresolved. It does not
capture a graph, construct a Gather/embedding adjoint, inspect or bounds-check indices, lower the
operation, or execute addition.

A later compiler/autograd owner may use this general public primitive when constructing Gather or
embedding data adjoints. That owner must preserve the same occurrence, Shape, numeric, duplicate-
accumulation, and eventual index-bounds obligations; this documentation does not claim that such
adjoint construction or compiler support exists today.
The three `Tensor.scatterNd` overloads consume exact ordered `[data, indices, updates]` inputs with
`INT32` or `INT64` indices and exact matching data/update types. Their defaults select
`ScatterReduction.NONE` and zero shared batch Dimensions; complete construction retains the exact
supplied reduction and normalized non-negative batch count. Construction validates reduction
eligibility, both ranks, structurally equal leading batch Dimensions, static positive tuple depth
from the final indices Dimension, and the exact indices-prefix-plus-data-suffix updates Shape.
Every result is fresh, unlabeled, storage-free, and unresolved-layout, retains the exact data
Shape/type, combines data/update gradient eligibility, and records `SCATTER_ND` with exact
three-input provenance. It reads no index or update value, checks no index bound or duplicate
target, mutates no input, performs no write or reduction, and adds no gradient, graph capture,
compiler transformation, materialization, lowering, backend behavior, or execution behavior.
Static `Tensor.concat(int, Tensor...)` and `Tensor.stack(int, Tensor...)` snapshot ordered non-empty
inputs, normalize an existing or inserted axis, enforce exact type and operation-specific Shape
rules, and create fresh unresolved-layout results with eligibility OR and exact ordered provenance.
Instance `Tensor.unstack(int)` requires a static `int`-sized selected extent and returns an
immutable ordered List of scalar `SELECT` expressions after upfront count validation. Each fresh
result removes that axis, has an independent one-output producer, and uses provenance index zero
over the same input; its `SelectAttrs` stores the coordinate. A zero extent returns no result or
ID. This is
current model expression construction, not compiler capture, decomposition, grouping, backward
construction, materialization, backend lowering, ONNX mapping, or execution.
`Tensor.unfold`, `Tensor.foldAxis`, both `Tensor.unfold2d` forms, and `Tensor.fold2d` construct the
current public storage-free window-transform expressions. General-axis fold restores an explicit
target extent under overlap summation. The 2D forms preserve canonical rank-three im2col/col2im,
retain exact static or symbolic channel/spatial formulas, and distinguish direct conceptual-zero
padding from an exact typed padding scalar. `fold2d` accepts only complete structural formula
matches rather than recording equality between unrelated unresolved symbols. Every result
preserves input data type and gradient eligibility, leaves layout unresolved, and records exact
one-input provenance. None of these model contracts reads values or provides compiler capture,
canonicalization, gradient generation, lowering, or execution.
That origin metadata gives a future compiler an expression to traverse, but no current API
captures it into `CompiledGraphModel`, performs inference or optimization, or produces compile
artifacts.

A future capture pass can observe that two result Tensors belong to the same expression occurrence
only when their provenance carries the same exact `TensorProducer` reference with different output
indices. It can read the producer's exact operation, immutable ordered inputs, immutable ordered
output descriptors, and signature-validated count. Capture may then assign graph-local `NodeId`
and `ValueId` values for that compilation. The producer does not prescribe those identities, and
the same public expression may be captured into separate graphs with different graph-local IDs.

Dropout is the first current production occurrence that uses this shared-output seam. Its public
output selects producer slot zero and its opaque next-state wrapper retains slot two. The producer
also describes slot one as a same-Shape BOOL auxiliary keep mask even though `DropoutResult` exposes
no mask Tensor. A later capture pass can therefore create a graph value for slot one by reading the
reachable producer's ordered descriptors; it does not need a public sibling-output lookup. A later
autograd pass may retain that captured forward mask for backward construction. Neither capture,
auxiliary-value lifetime policy, nor a dropout gradient rule is implemented today.

## Current expression input and planned compiler output

Conceptually, compilation will receive a requested tensor output and declarative `CompileConfig`:

```java
// Conceptual API; not currently runnable.
CompiledGraph graph = CompiledGraph.compile(output, CompileConfig.auto());
```

- `output` will identify a current public `Tensor` expression for the future compiler to capture.
  Public Tensor state plus binary arithmetic, binary comparison, boolean logical, conditional
  selection, cast, unary numeric including exact GELU, fixed tanh-approximation GELU, and SiLU,
  floating-classification, scalar, numeric aggregate expression
  construction for sum, mean,
  product, minimum, and maximum, binding-aware exact-target-Shape sum, masked sum and mean
  construction, boolean aggregate expression
  construction for all and any, ordered multi-axis ordinary/log-sum-exp/statistical/norm
  construction, axis-only arg-min/arg-max construction, and shape-preserving cumulative sum/product
  scan, softmax/log-softmax, trailing-Shape layer- and RMS-normalization construction, explicit
  five-input batch-normalization inference and five-output training/statistic-transition
  construction, exact-shape mean-squared-error construction with explicit loss reduction, and
  dense- or index-target categorical-cross-entropy-with-logits construction with exact target-type
  dispatch and one exact typed ignore overload,
  plus
  first-class
  scaled-dot-product-attention
  construction with optional BOOL mask and scale/causal attributes, plus grouped NCHW Conv2d
  construction with optional bias and exact geometry/group attributes, plus NCHW maximum- and
  average-pooling construction with operation-specific attributes and exact floor/ceil spatial
  expressions, plus static-resolved or
  dynamic-unresolved contiguous
  request construction plus conditional-view reshape, expand, permutation, and rank-two transpose
  construction, conditional-view expand-dimensions/squeeze construction, and general/single-axis
  signed-step slice plus one-occurrence flip construction, functional signed slice update, exact
  target/prefix-Shape crop construction, plus conditional-view scalar-select construction and
  unresolved-layout Gather/Gather Elements and trailing-axis one-hot construction, plus
  unresolved-layout Gather-ND,
  functional Scatter Elements, Gather-compatible Scatter Add, and functional Scatter-ND
  construction, plus
  unresolved constant-pad and complete-pattern tile
  construction, plus ordered concat/stack and immutable repeated-SELECT unstack construction,
  plus public general-axis unfold/fold and direct- or typed-padding NCHW unfold/fold window-
  transform construction, plus
  explicit-state training-dropout construction with public output and next-state results and one
  non-public producer mask slot,
  are implemented;
  the compiler entry point, traversal, capture,
  scan/reduction/normalization/loss/attention inference and canonicalization, layer-, RMS-, batch-
  normalization, and loss deferred-constraint proof, saved-statistic and gradient construction,
  optional
  softmax, layer-normalization, RMS-normalization, attention, or activation decomposition,
  activation/attention-gradient construction,
  redundant-cast, redundant-contiguous, reshape/expand/permutation/rank-edit, slice chains, slice
  updates, crops, selects,
  axis-gather/one-hot/Gather-ND/axis-scatter/Scatter-ND/pad/tile/composition/window-transform
  canonicalization or decomposition,
  shared-producer traversal and output-slot capture,
  compiler-generated use of public `FOLD_AXIS`, deferred slice-update upper-bound and crop-bound
  proof, their possible use in adjoint construction, deferred
  dynamic reshape count validation, expand compatibility constraints, dynamic select upper-bound
  validation placement, attention, convolution, and pooling deferred-constraint proof, legal
  convolution/pooling decomposition and gradient construction, maximum-pooling saved-index
  policy, average-pooling fixed-divisor preservation, layout materialization
  planning, and conversion into graph values and nodes remain planned.
- `GraphRngState` is an implemented opaque model expression value rather than a public numerical
  `Tensor` output. Current dropout places its private state Tensor at producer input one and wraps
  producer output two as the next state. Capturing, serializing, preserving or deliberately
  transforming that state edge remains compiler work; current compilation exposes no such API.
- `CompileConfig` will describe compile mode, backend intent, optimization, scoring, and
  publication policy as data. It will not contain live backend services.
- `PublicationPlan` will be compiler-owned context around publication bindings. It is planned and
  is separate from the current model graph.
- `CompileArtifacts` will combine a `CompiledGraphModel`, planned partitions, a logical memory
  plan, a `PublicationPlan`, and diagnostics. It is planned and will remain non-executable.
- `CompiledGraph` will be an engine facade over immutable `CompileArtifacts`, not the same object
  as the current `CompiledGraphModel`.

The planned artifacts deliberately contain no device buffers, backend executable objects, runtime
residency, prepared schedules, or mutable run state.

## Planned lifecycle and failures

```text
expression -> capture -> inference and validation -> optimization
           -> optional autograd -> backend ownership -> logical plans
           -> CompileArtifacts
```

Compilation is expected to reject invalid shapes, data types, operations, graph structure, or
unsatisfied backend capabilities before preparation. Exact exception types and callable signatures
remain to be specified by compiler and engine tasks; callers must not code against invented
exceptions from this conceptual page.

## Example interpretation

If a future graph contains a matrix multiplication followed by a small elementwise operation,
capability analysis may find both CPU and Metal valid. Backend-neutral scoring may assign both
nodes to Metal to avoid a transfer boundary. The artifact records only `owner = Metal`; it does
not record MPSGraph or a custom Metal kernel. Metal prepare makes that later choice.

This scenario is conceptual. The current graph DTOs can represent and structurally validate node
relationships, but they cannot run this compilation or select ownership.

## Related contracts

- [Current Tensor and graph-model API](tensor-api.md)
- [Lifecycle](../architecture/lifecycle.md)
- [Partition scoring](../architecture/partition-scoring.md)
- [Compiling graphs user guide](../user-guide/compiling-graphs.md)
- [Roadmap](../planning/roadmap.md)
