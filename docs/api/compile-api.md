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
full/axis numeric aggregate methods, six full/axis boolean aggregate methods, fourteen ordinary
multi-axis methods, twelve floating advanced/statistical reduction methods, two axis-removing
masked aggregate methods, six axis-only `argMin`/`argMax` methods, and two
one-axis `cumSum` methods, plus one-axis `softmax`, `logSoftmax`, scalar `select`, two
tensor-index axis-gather methods, the `embedding` convenience over axis-zero Gather, one
trailing-axis `oneHot` index-encoding method, and two Gather-ND methods, plus two functional
Scatter Elements
methods and three functional Scatter-ND methods, plus one matrix-multiplication method, construct
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
`Tensor.matmul` currently constructs one fresh two-input MATMUL expression with a locally derived
vector, matrix, or broadcast-batch Shape and same-category promoted numeric type. Unequal static
contraction dimensions fail locally; unresolved contraction equality and the accepted
unresolved-versus-static batch singleton-or-equal cases remain obligations for later compiler
validation or concrete binding. This is current model-expression metadata only: expression
capture, graph-wide validation, constraint proof, gradients, lowering, backend support, and
execution remain planned.
`Tensor.sort(axis[, descending])` and `Tensor.argsort(axis[, descending])` currently construct
distinct stable, one-input, one-output ordering expressions. Both normalize the axis, preserve the
exact input Shape reference, leave layout unresolved, and use fixed NaN-last ordering in both
directions with stable logical-index ties. Sort preserves input type and gradient eligibility;
argsort uses non-differentiable INT64. These are current model-expression and provenance facts,
not implemented compiler capture, operand revalidation, gradient construction, algorithm
selection, lowering, backend support, runtime behavior, or execution.
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
`Tensor.cumSum` accepts floating or integral input and one positive or negative axis. Its short
form explicitly selects inclusive forward traversal; its complete form retains exact exclusive
and reverse flags. Every result retains the exact input Shape, data type, and gradient eligibility,
leaves layout unresolved, and records `CUM_SUM` with exact one-input provenance. Construction does
not read or accumulate values, create a gradient rule, capture a graph, lower a backend operation,
or execute work.
`Tensor.softmax` and `Tensor.logSoftmax` accept floating input and one positive or negative axis.
Every result retains the exact input Shape, data type, and gradient eligibility, leaves layout
unresolved, and records the requested first-class SOFTMAX or LOG_SOFTMAX kind with exact one-input
provenance. Construction does not read values, calculate probabilities or logarithms, select a
numerical algorithm, define a gradient rule, capture or decompose a graph operation, lower a
backend operation, or execute work.
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
Specialized fixed-add gather adjoints are not current public model kinds or Tensor methods. A
future compiler task may define selected backend-independent generated semantics when ordinary
composition is insufficient; this page does not predefine those operations.
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
`Tensor.unfold`, `Tensor.unfold2d`, and `Tensor.fold2d` construct the current public storage-free
window-transform expressions. Their signatures and behavior are unchanged: they use checked
static Shape arithmetic, preserve input data type and gradient eligibility, leave layout
unresolved, and record exact one-input provenance. No public `Tensor.foldAxis` or helper path
constructs the retained `FOLD_AXIS`/`FoldAxisAttrs` semantic pair. Task 0023 owns its first
compiler-generated construction and operand compatibility. None of these model contracts reads
values or provides compiler capture, canonicalization, gradient generation, lowering, or
execution.
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
  product, minimum, and maximum, masked sum and mean construction, boolean aggregate expression
  construction for all and any, ordered multi-axis ordinary/log-sum-exp/statistical/norm
  construction, axis-only arg-min/arg-max construction, and shape-preserving cumulative-
  sum and softmax/log-softmax construction, plus static-resolved or dynamic-unresolved contiguous
  request construction plus conditional-view reshape, expand, permutation, and rank-two transpose
  construction, conditional-view expand-dimensions/squeeze construction, and general/single-axis
  signed-step slice plus one-occurrence flip construction, plus conditional-view scalar-select construction and
  unresolved-layout Gather/Gather Elements and trailing-axis one-hot construction, plus
  unresolved-layout Gather-ND,
  functional Scatter Elements, and functional Scatter-ND
  construction, plus
  unresolved constant-pad and complete-pattern tile
  construction, plus ordered concat/stack and immutable repeated-SELECT unstack construction,
  plus public general-axis unfold and NCHW unfold/fold window-transform construction, plus
  explicit-state training-dropout construction with public output and next-state results and one
  non-public producer mask slot,
  are implemented;
  the compiler entry point, traversal, capture,
  scan/reduction/normalization inference and canonicalization, optional softmax or activation
  decomposition, activation-gradient construction,
  redundant-cast, redundant-contiguous, reshape/expand/permutation/rank-edit/slice-chain/select/
  axis-gather/one-hot/Gather-ND/axis-scatter/Scatter-ND/pad/tile/composition/window-transform
  canonicalization or decomposition,
  shared-producer traversal and output-slot capture,
  compiler-generated `FOLD_AXIS` construction, deferred
  dynamic reshape count validation, expand compatibility constraints, and dynamic select upper-
  bound validation placement, layout materialization
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
