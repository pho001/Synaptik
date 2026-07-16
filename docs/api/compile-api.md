# Compile API

## Purpose and implementation status

This reference separates the compile-time model values implemented today from the compiler and
engine APIs that remain planned. The compiler module now contains package-private structural
forward capture, binding-free captured-graph verification inference, mandatory dense
canonicalization, an immutable logical-splat sidecar for explicitly supplied fixed sources, and
one config-controlled forward optimization pipeline with guarded exact arithmetic rewriting,
bounded constant folding, and constant-aware cleanup. The repository still provides no public or
runnable graph compiler.
The current config module provides four immutable standalone input values that a later compile
configuration aggregate can contain: `BackendIntent`, `CompileMode`, and
`GraphOptimizationConfig`, plus `PartitionScoringConfig`. The current planning module provides the
immutable `OperationCapabilityQuery` and the explicitly supplied `BackendCapabilityProvider`
collaboration. It also contains one package-private per-query hard-eligibility step, but it does
not expose that step publicly. A second package-private step now selects one `BackendId` owner by
optional preferred-class match and provider order. The module still provides no reusable/public
capability matrix, public planner orchestration or owner selector, general cost scoring,
owner-map assembly, or compiler integration. It now provides the public
immutable `PlannedPartition(owner, nodeIds)` recipe and a package-private generator that groups a
complete node-to-owner assignment into maximal consecutive same-owner runs. A following
package-private step derives public immutable `LogicalMemoryRequirement` and `LogicalMemoryPlan`
recipes from a closed graph and ordered complete partitions.

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

### Current package-private structural capture

The compiler module now contains package-private `GraphCapture`. Its package-private static entry
point accepts one non-null, non-empty ordered `List<Tensor>` and returns a
`CompiledGraphModel`. This is an internal compiler implementation boundary, not a public compile
API or a callable lifecycle facade.

Capture traverses exact Tensor and `TensorProducer` references in requested-result order and
producer-input order. A first-seen provenance-free Tensor becomes one graph input. A first-seen
producer is emitted once after its reachable input producers, and every ordered producer output
descriptor becomes a graph value even when no public Tensor wrapper exposes that position.
Produced Tensors resolve through `TensorProvenance.outputIndex()`. Requested graph outputs retain
caller order, while every captured node is classified `GraphPhase.FORWARD`.

`ValueId` and `NodeId` allocation restarts at zero for each call and follows deterministic
depth-first postorder encounter. Capture preserves repeated operand positions, distinct
identity-separated producer occurrences, exact operation and descriptor references, zero-input
semantic source nodes, and reachable opaque RNG-state edges. The returned graph retains none of
the traversed Tensor, producer, provenance, or `GraphRngState` references.

The request rejects a null list, an empty list, a null element, a repeated exact Tensor reference,
or distinct wrappers that resolve to one graph output value. No caller collection is mutated.
The returned graph owns immutable collection snapshots through `CompiledGraphModel`.

An overload accepts a package-private immutable constant-ingress request. Each binding pairs one
exact provenance-free, non-gradient Tensor leaf identity with one exact typed logical splat. A
logical splat means that the same `ScalarValue` applies at every logical coordinate; it contains no
Shape, dense payload, storage, Tensor, buffer, or backend value. Capture maps a binding only at the
existing Tensor-identity-to-`ValueId` boundary and rejects a binding that is not encountered as a
reachable leaf. It does not read `Tensor.hostStorage()` or infer a constant from a factory method,
descriptor, layout, label, storage contents, or the mere absence of provenance.

The result of that overload is an internal immutable sidecar around the unchanged
`CompiledGraphModel`. Every fixed source remains structurally a graph input, while the sidecar's
derived bindable-input list contains exactly the graph inputs without facts, in graph-input order.
Consequently an explicitly fixed source cannot also be supplied later as a caller input. An
otherwise identical leaf absent from ingress remains bindable, including a leaf created by a
scalar, zero, one, full, import, allocation, or random factory path. There is no public constant-
binding or compile entry point.

Capture itself performs structural forward capture only. The following package-private
verification-inference and transformation boundaries revalidate, canonicalize, and optionally
optimize the captured graph. Capture does not perform those later steps. None of these internal
steps constructs gradients or backward work, derives publication bindings, invokes planning,
produces diagnostics or `CompileArtifacts`, prepares backends, allocates physical memory, or
executes values. No public `GraphCompiler`, `CompiledGraph`, compile entry point, configuration
aggregate, capture method, validation method, or optimizer method exists.

### Current package-private verification inference

Package-private `CapturedGraphInference.inferAndValidate(CompiledGraphModel)` is the current
semantic boundary after structural capture. It visits stored nodes in deterministic topological
order, resolves ordered input and output descriptors, selects a typed operation-family rule, and
independently derives the complete expected `TensorDescriptor` list. It checks data-type domains
and promotion, operand roles, ranks, axes, Shape and attribute relationships, output roles,
resolved-or-unresolved layout, and gradient-eligibility metadata. The first unequal expected and
stored descriptor fails with node, operation-kind, output-position, and value context.

The pass covers every current production operation-kind enum and accepted attributes variant in
five cohesive families: elementwise, reduction/normalization/ordering, indexing,
layout/composition/window, and structured numeric/loss/RNG-state operations. Dispatch is closed
and typed. An unrecognized custom `OperationKind` fails rather than bypassing validation.

Some model operations determine an exact output descriptor while leaving a Shape relationship
undecidable. The compiler represents such a relationship as a package-private
`DeferredGraphConstraint`: one owning `NodeId`, one semantic subject, and one typed immutable
predicate. A conservative three-valued proof returns proven, disproven, or deferred from static,
structural, and available bounded dimension facts. Proven predicates disappear; disproven
predicates fail in deterministic rule order; deferred predicates remain ordered in the successful
result. The evaluator never binds or unifies symbols, and checked-arithmetic overflow makes a
proof unavailable rather than false.

On success, package-private `ValidatedGraph` retains the exact accepted compiler sidecar and
snapshots only the unresolved constraints. Its graph-only compatibility path wraps the accepted
`CompiledGraphModel` with no constant facts. The semantic inference rules continue to inspect the
graph alone; sidecar construction independently checks that every fact names a non-gradient graph
input of the same data type. For example, a reshape from Shape
`[N]` to Shape `[M]`, where `N` and `M` are distinct unresolved dimensions, retains an
element-count-equality constraint because the output descriptor is already exact but equality is
not locally decidable. This proves neither a concrete binding nor executability. Value-dependent
index bounds, duplicate scatter targets, target contents, numerical results, and storage validity
remain outside descriptor-only verification.

This pass is reusable compiler-internal state, not a public artifact. It performs no graph
transformation, concrete dimension binding, autograd, publication or planning orchestration,
diagnostic aggregation, preparation, backend lowering, runtime work, or execution. The current
canonicalization and optional optimization pipeline reuses it after mandatory canonicalization
and after every changed graph candidate.

### Current package-private canonicalization and forward optimization

Package-private `ForwardGraphOptimization.optimize(ValidatedGraph, GraphOptimizationConfig)`
consumes only a successful verification result and the standalone optimization permission. It
always rebuilds the graph through `GraphCanonicalization` before validating the rebuilt candidate.
Graph inputs receive dense `ValueId` values first in boundary order. Node outputs then receive
dense values in stored topological and output-slot order, while `NodeId` values follow stored node
order. Operations, descriptors, phases, repeated input positions, every output slot, and ordered
graph boundaries are preserved.

When optional optimization is disabled, that canonicalization and validation still occur. When it
is enabled, the internal pipeline runs exactly once in this order:

```text
guarded exact arithmetic rewriting -> constant folding
                                   -> sidecar-aware forward dead-code elimination
                                   -> exact forward common-subexpression elimination
                                   -> sidecar-aware forward dead-code elimination cleanup
```

The first scan recognizes exactly seven internal, one-output `FORWARD` identities: duplicate-input
binary `MIN` and `MAX`; scalar `MUL` by exact typed positive one for `BFLOAT16`, `FLOAT32`,
`FLOAT64`, `INT32`, and `INT64`; scalar `DIV` and `POW` by exact typed positive one for the three
floating types; and scalar `ADD` and `SUB` by exact typed zero for `INT32` and `INT64`. Every bypass
requires complete equality between the input and output `TensorDescriptor`, including Shape,
layout, data type, and `requiresGrad`; the equal descriptor must have `requiresGrad == false`.
The result must not be a graph output. Duplicate extrema compare already-remapped input IDs, while
scalar rules require the exact `ScalarValueAttrs` carrier, matching scalar/input/output type, and
matching typed value. A prior bypass can therefore expose a later eligible occurrence during the
same scan without iteration.

`ScalarValueAttrs` is immutable operation metadata already stored in the graph. Reading its exact
typed `ScalarValue` for an identity proof is not Tensor constant discovery or host-storage
inspection. The scan deliberately does not add floating ADD/SUB-zero, multiplication by
zero, cancellation, other powers, bounds/clamp identities, broader algebra, or relaxed numerical
rules. It retains graph-output, gradient-eligible, non-forward, and multi-output occurrences.

The following constant-folding scan reads only logical splat facts already present in the
compiler sidecar. It folds exactly these current rows:

| Family | Current selected kinds and domains |
|---|---|
| Boolean logical | `NOT`, `AND`, and `OR` over exact `BOOL` splats |
| Binary arithmetic | `ADD`, `SUB`, `MUL`, `MIN`, and `MAX` over `INT32`/`INT64` splats, including mixed-width signed promotion |
| Binary comparison | All six comparisons over `INT32`/`INT64` splats, producing exact `BOOL` splats |
| Scalar elementwise | `ADD`, `SUB`, `MUL`, `MIN`, and `MAX` over one integral splat and an exact same-typed integral `ScalarValueAttrs` value |

Integral addition, subtraction, and multiplication use the current fixed-width modular semantics;
extrema and comparisons use signed order. If either binary operand is `INT64`, an `INT32` operand
is sign-extended and the result domain is `INT64`. The scalar input remains the left operand, so
subtraction preserves operand order. A folded splat can feed a later selected occurrence during
the same topological scan, but the compiler does not seek a fixed point or run a second folding
scan.

Each selected fold replaces one internal, one-output, non-gradient `FORWARD` occurrence with one
synthetic constant graph input. Original inputs remain first; synthetic sources follow in original
fold-occurrence order; retained outputs and nodes are then rebuilt densely and deterministically.
Graph-output producers, all multi-output producers, `BACKWARD` nodes, and gradient-eligible values
remain occurrences. Equal splats are not interned.

Floating and BFLOAT16 splats retain exact type and bits in the sidecar, including signed zeros and
NaN payloads, but no floating or BFLOAT16 operation is evaluated. Casts, unselected scalar kinds,
unary numerical functions, reductions, scans, normalization, loss, linear algebra, indexing,
layout/view operations, random/state operations, and generic partial evaluation are also outside
the current fold matrix. The compiler reads no host storage, creates no dense constant payload,
and allocates no physical value.

The existing graph-only dead-code elimination (DCE) entry retains every graph input and graph
output, treats every non-forward node and its dependency closure as live roots, and retains all
output slots of every live node. Its sidecar-aware overload first applies those same liveness
rules, then removes only fixed constant inputs that are neither graph outputs nor inputs to a
retained node. It always preserves bindable inputs, even when unused. This prevents obsolete
ingress or synthetic constants from becoming downstream materialization obligations.
Exact common-subexpression elimination (CSE) compares the forward phase, complete immutable
operation value, ordered remapped inputs, and every ordered output descriptor. It merges a whole
multi-output occurrence slot by slot. A producer containing a graph output may neither merge nor
serve as a representative, so distinct requested boundaries remain distinct. Each changed
candidate is immediately revalidated through the verification pass; an unchanged helper result is
not redundantly revalidated.

These are internal graph transformations, not a public compiler surface. They do not rewrite
casts, broader arithmetic, algebra, or views; construct autograd; derive publication, planning, or
diagnostics; create `CompileArtifacts`; or affect trace, preparation, runtime, backend, or engine
behavior. Compiler task 0005 must transport the exact immutable facts, or a named component with
identical semantics, into future compile artifacts and must expose only the derived bindable
inputs. Planning may then see the unchanged logical graph. Future prepare/backend work owns
physical splat materialization, storage allocation, lowering, and execution.

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
Equal words request replay-equivalent positions but do not merge expression identities in the
public expression model.

Current package-private structural capture preserves this state producer and its ordered state
edges when they are reachable from a requested Tensor result. A future serializer must preserve
the raw words losslessly. Current exact internal CSE may merge equal initial-state and dropout
graph occurrences only when their complete operation, ordered inputs, phase, and all output
descriptors match; this does not merge their pre-capture expression identities or select a random
algorithm. No current code defines serialization, gradient rules, a byte encoding, a stable enum
token, prepared state, or execution. Without a selected portable algorithm, no cross-backend or
cross-version bitstream follows from equal state.

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
metadata-construction facts, not compiler validation, gradient, lowering, backend, or execution
claims. Package-private structural capture preserves them, and the following package-private
verification pass independently revalidates them. An `OperationSignature`
still validates only the exact attribute class and occurrence cardinality because an `Operation`
has no operand
descriptor. Current package-private compiler verification revalidates operand domains and exact
scalar/input data-type equality for captured or otherwise constructed occurrences. `Tensor.where` requires an exact BOOL
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
Current package-private capture preserves this model metadata structurally, and current
package-private verification represents and proves or retains the Shape obligation. Adjoint
construction, canonicalization, lowering, backend selection, and execution remain unimplemented.
`Tensor.matmul` currently constructs one fresh two-input MATMUL expression with a locally derived
vector, matrix, or broadcast-batch Shape and same-category promoted numeric type. Unequal static
contraction dimensions fail locally; unresolved contraction equality and the accepted
unresolved-versus-static batch singleton-or-equal cases remain obligations for later compiler
validation or concrete binding. Package-private structural capture plus graph-wide verification
and conservative constraint proof are current; gradients, lowering, backend support, concrete
binding, and execution remain planned.
`Tensor.linear(weight)` and `Tensor.linear(weight, bias)` are also current model construction, but
they add no LINEAR operation. Conventional `[outFeatures, inFeatures]` weight is explicitly
transposed through PERMUTE `[1, 0]`, followed by MATMUL and optional exact rank-one
`[outFeatures]` ADD bias. Complete caller-controlled validation precedes the first intermediate.
No-bias returns the MATMUL product after two wrapper/ID allocations; biased construction returns
ADD after three. The biased final Shape is structurally equal to the product Shape and reuses its
ordered Dimension references, although the outer Shape object may differ. These visible primitive
producer chains are current inputs to package-private structural capture. This page does not claim
linear-pattern recognition, canonicalization, fusion, gradient construction, lowering, backend
support, or execution.
`Tensor.scaledDotProductAttention` and `Tensor.scaledDotProductAttentionWithWeights` are current
first-class model construction with three ordered query/key/value inputs and an optional fourth
BOOL mask input. Both locally derive the promoted type and exact broadcast-batch, weights, and
output Shapes and retain optional exact scale and causal eligibility. The original four methods
record an exact one-output `SCALED_DOT_PRODUCT_ATTENTION` occurrence with output slot zero and no
hidden weights descriptor or Tensor. The four explicit methods return
`ScaledDotProductAttentionResult(output, weights)` whose slots zero and one retain one exact
shared producer, operation, attributes reference, ordered inputs, and output descriptors. The
kind's sole signature accepts three to four inputs and one to two outputs; producer descriptor
count identifies the requested occurrence form without adding another kind. Unresolved embedding
positivity/equality, key/value sequence equality, batch singleton-or-equal, and mask
singleton-or-equal facts remain obligations for later compiler validation or concrete binding.
Package-private structural capture preserves this compiler-visible metadata, and package-private
verification revalidates it and proves or retains its typed Shape constraints. Legal
decomposition, attention gradient, saved-value lifetime, backend lowering, and execution remain
planned.
`Tensor.conv2d(weight, attrs)` and `Tensor.conv2d(weight, bias, attrs)` are current first-class
grouped NCHW cross-correlation model construction. Each call records one `CONV2D` occurrence with
exact ordered inputs, intrinsic stride/padding/dilation/group attributes, promoted floating
descriptor, and static or canonical-symbolic output Shape. Input and weight have Shapes
`[N, C_in, H, W]` and `[C_out, C_in/groups, K_h, K_w]`; optional bias has `[C_out]`.
Unresolved channel divisibility, grouped weight/input equality, bias/output equality, and dynamic
spatial non-negativity remain obligations for future compiler validation or concrete binding.
This metadata can be structurally captured and package-private verification now represents and
proves or retains its descriptor-only constraints, but the repository still cannot compile or run
convolution. Legal decomposition, convolution gradients/adjoints and saved values, concrete
binding, backend lowering, algorithm selection, and execution remain planned in their owning
layers.
`Tensor.maxPool2d(attrs)` is current first-class NCHW maximum-pooling model construction. One
`MAX_POOL2D` occurrence records exact ordered input `[input]`, `MaxPool2dAttrs`, one output at
index zero, the unchanged floating type and gradient request, exact batch/channel Dimensions, and
static or canonical-symbolic floor/ceil spatial extents. Dynamic spatial non-negativity remains a
future compiler-validation or concrete-binding obligation. Literal ceil mode retains every
ceiling-grid window, even when the terminal window is all-padding; padding exclusion, negative-
infinity empty windows, NaN propagation, signed-zero ordering, and first-logical-sample ties are
semantic metadata. Package-private structural capture and descriptor verification are current,
but the repository does not compile pooling. Concrete binding, legal decomposition, gradients and
adjoints, and saved-index decisions remain planned, as do backend lowering, algorithms, kernels,
and execution.
`Tensor.averagePool2d(attrs)` is current first-class NCHW average-pooling model construction. One
`AVERAGE_POOL2D` occurrence records exact ordered input `[input]`, `AveragePool2dAttrs`, one output
at index zero, the unchanged floating type and gradient request, exact batch/channel Dimensions,
and static or canonical-symbolic floor/ceil spatial extents. Literal ceil mode retains terminal
all-padding windows. Its semantic metadata fixes a `kernelHeight * kernelWidth` divisor,
conceptual positive-zero padding that counts in that divisor, FLOAT32 accumulation/division for
BFLOAT16 and FLOAT32, FLOAT64 accumulation/division for FLOAT64, one final division, and the
documented NaN/infinity/signed-zero/all-padding policies. Dynamic spatial non-negativity remains a
future compiler-validation or concrete-binding obligation. This does not mean the repository
currently compiles average pooling. Package-private structural capture and descriptor verification
are current. Concrete binding, any legal decomposition that preserves the fixed divisor and
special-value meaning, gradient or adjoint construction, backend lowering, algorithms,
tolerances, kernels, and execution remain planned in their owning layers.
`Tensor.sort(axis[, descending])` and `Tensor.argsort(axis[, descending])` currently construct
distinct stable, one-input, one-output ordering expressions. Both normalize the axis, preserve the
exact input Shape reference, leave layout unresolved, and use fixed NaN-last ordering in both
directions with stable logical-index ties. Sort preserves input type and gradient eligibility;
argsort uses non-differentiable INT64. These model-expression and provenance facts are structurally
capturable and package-private operand/descriptor revalidation is current; gradient construction,
algorithm selection, lowering, backend support, runtime behavior, and execution remain planned.
`Tensor.topK(k, axis[, largest, sorted])` currently constructs one compiler-neutral
`TopKKind.TOP_K` model occurrence with ordered outputs `[values, indices]`. The values output
preserves input type and gradient eligibility; indices are non-differentiable INT64. Both share
one fresh Shape whose selected dimension is static `k`, one exact producer, and provenance slots
zero and one. A known static selected extent is checked during Tensor construction. When that
extent is dynamic or expression-based, the obligation `bound extent >= k` is deliberately
deferred: future compiler or binding validation must reject an insufficient bound rather than
clamp, pad, wrap, or reduce the output count. Package-private capture preserves the shared TOP_K
producer and both output positions. The following package-private pass revalidates both descriptors
and proves or retains the selected-extent obligation. It does not construct gradients, select an
algorithm, lower the operation, report backend support, bind an extent, or execute it.
`Tensor.embedding(indices)` currently validates a rank-two floating weight receiver and exact
INT32/INT64 indices, then constructs the existing ordinary axis-zero GATHER occurrence directly.
The result Shape is the complete indices Shape plus the exact weight axis-one Dimension; result
type and gradient eligibility come only from weights. This is public model-expression construction,
and is structurally capturable. Constant-index analysis, repeated-index autograd construction,
dynamic binding, safe bounds enforcement, lowering, backend support, and execution remain owned by
later lifecycle layers. No `EMBEDDING` kind or padding/sparse/max-norm/frequency option exists.
`Tensor.oneHot(depth)` currently validates exact INT32/INT64 receiver metadata before positive
static depth, preserves every input Dimension reference, and appends one fresh
`StaticDimension(depth)`. It constructs one storage-free, non-differentiable BOOL result with one
`ONE_HOT` producer and exact sole-input provenance. This is a current model-expression inventory
entry and is structurally capturable. Construction reads no index values; valid eventual execution
requires `0 <= i < depth`, while constant analysis, dynamic bounds enforcement,
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
compiler-visible requested meanings that current package-private capture preserves structurally.
Current package-private verification revalidates operands and proves, rejects, or retains dynamic
corrected-domain constraints. Numerical algorithms, gradients, lowering, backend support,
concrete binding, and execution remain planned.
`Tensor.cumSum` and `Tensor.cumProd` are current model expressibility for shape-preserving
cumulative addition and multiplication. Each family accepts floating or integral input and one
positive or negative axis. Each short form explicitly selects inclusive forward traversal; each
complete form retains exact exclusive and reverse flags. Every result retains the exact input
Shape, data type, and gradient eligibility, leaves layout unresolved, and records the matching
`CUM_SUM` or `CUM_PROD` with exact one-input provenance. Integral cumulative product has exact-
width modular meaning; floating product has the selected NaN, zero-times-infinity, sign-parity,
and positive-one identity semantics documented by the Tensor API. Construction does not read or
accumulate values.

Those model expressions are compiler-visible inputs to current package-private structural capture.
Dynamic validation, canonicalization, product-adjoint construction from prefix and suffix scans,
gradient boundary policy, saved-value lifetime, numerical lowering, backend support, and execution
remain planned. In particular, the current compiler API exposes no callable compilation path that
turns either scan kind into executable work.
`Tensor.softmax` and `Tensor.logSoftmax` accept floating input and one positive or negative axis.
Every result retains the exact input Shape, data type, and gradient eligibility, leaves layout
unresolved, and records the requested first-class SOFTMAX or LOG_SOFTMAX kind with exact one-input
provenance. Construction does not read values, calculate probabilities or logarithms, select a
numerical algorithm, define a gradient rule, decompose a graph operation, lower a backend
operation, or execute work; package-private structural capture can preserve the occurrence.
`Tensor.layerNorm` is current model metadata construction for exact trailing-Shape population
normalization. The no-affine form records `LayerNormAttrs` and ordered input `[input]`; the affine
form records `AffineLayerNormAttrs` and ordered inputs `[input, scale, bias]`. Both retain exact
normalized Shape and typed epsilon parameters, exact input result Shape, one output at index zero,
and no saved-statistic output. Local construction rejects known static mismatches and defers an
unresolved trailing-dimension equality when output Shape is still exact. Package-private capture
preserves this metadata structurally, and package-private compiler verification revalidates the
operands and proves or retains deferred constraints. Saved-statistic lifetime, gradients or
adjoints, legal decomposition, lowering, concrete binding, and execution remain planned in their
owning layers.
`Tensor.rmsNorm` is current model metadata construction for exact trailing-Shape uncentered
root-mean-square normalization. One `RmsNormAttrs(normalizedShape, epsilon)` value supports the
safe ordered inputs `[input]` and `[input, scale]`, each with exactly one output at index zero. The
result retains the exact input Shape; the scaled form requires scale Shape exactly equal to the
normalized Shape. Local construction rejects known static trailing mismatches and defers
unresolved equality. Package-private capture preserves this metadata structurally, and current
package-private verification revalidates operands and proves or retains deferred constraints.
Compiler-generated saved values, gradients or adjoints, legal decomposition, lowering, backend
preparation, tolerance enforcement, concrete binding, and runtime execution remain planned in
their owning layers.
`Tensor.batchNormInference` is current stateless model metadata construction with exact ordered
inputs `[input, scale, bias, runningMean, runningVariance]` and exactly one output at index zero.
It retains `BatchNormInferenceAttrs(normalizedChannelAxis, epsilon)`, preserves the exact input
Shape, requires rank-one `[C]` affine/statistic vectors, defers unresolved channel-extent equality,
and uses ordered floating promotion with exact-result-typed positive epsilon. The channel axis is
layout-neutral, and running variance is interpreted directly by the formula with epsilon inside
the denominator square root. Package-private capture preserves this compiler-visible metadata;
current verification revalidates the operands and proves or retains channel constraints.
Saved-value and gradient construction, legal decomposition, lowering, backend preparation,
tolerance enforcement, concrete binding, runtime execution, and numerical evaluation remain
planned in their owning layers.
`Tensor.batchNormTraining` is current five-output model metadata with exact ordered inputs
`[input, scale, bias, runningMean, runningVariance]`. Its producer describes normalized output,
next running mean, next running variance, saved batch mean, and saved inverse standard deviation
in that order. `BatchNormTrainingResult` exposes positions zero through two; positions three and
four remain real producer descriptors that package-private capture preserves. The model records a normalized
channel axis, exact typed new-batch-weight momentum and epsilon, the all-non-channel reduction
domain, biased forward variance, correction-one running variance, output Shapes, types, gradient
eligibility, and indexed provenance. Structural capture of every position and package-private
proof or retention of the `C == 0 || N >= 2` constraint are current. Saved-value materialization
or lifetime, autograd/backward construction, publication, liveness, graph optimization, concrete
binding, lowering, preparation, execution, and cross-step statistic ownership remain planned in
the compiler, training extension, runtime, and backend layers that own them.
`Tensor.meanSquaredError(target, reduction)` is current one-output model metadata with exact
ordered inputs `[prediction, target]`. It records `LossKind.MEAN_SQUARED_ERROR` and one
`MeanSquaredErrorAttrs` carrying explicit `NONE`, `SUM`, or `MEAN` reduction. Local construction
accepts only floating inputs, promotes them in input order, rejects unequal known static Shapes
without broadcasting, and defers unequal pairs involving an unresolved Dimension. `NONE` retains
the exact prediction Shape; `SUM` and `MEAN` use the scalar Shape. Mean divides by the complete
logical element count, so scalar count is one, empty sum is positive zero, and empty mean is NaN.
This compiler-visible requested meaning is structurally capturable, and package-private
verification now revalidates it and proves or retains deferred equality. Gradient or adjoint
construction, legal decomposition, optimization, concrete binding, lowering, backend support,
runtime execution, and training coordination remain planned in their owning layers.
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
remain valid when every target is ignored. This compiler-visible requested meaning is structurally
capturable. It does not implement revalidation, constant analysis, proof, bounds checks, gradients or
adjoints, decomposition, optimization, lowering, preparation, execution, publication, or training
coordination; those remain planned in their owning lifecycle layers.
`Tensor.contiguous()` accepts every current data type and preserves the exact Shape, data type, and
gradient eligibility. It creates new canonical dense row-major, zero-offset layout geometry for a
fully static Shape and leaves a dynamic Shape unresolved. Every call is fresh, unlabeled, and
storage-free, records `CONTIGUOUS` with the canonical no-attributes singleton and exact one-input
provenance, and does not inspect input layout, storage, or values. Resolved result geometry does
not allocate or copy storage. Package-private structural capture is current; redundant-request
canonicalization, materialization policy, lowering, and execution remain planned.
`Tensor.reshape(long...)` accepts all current data types, normalizes an empty request or one
inferable `-1`, and rejects locally provable invalid counts. `Tensor.reshape(Shape)` retains an
exact normalized target and defers count equality when either Shape is dynamic. Both overloads
retain type and gradient eligibility, record exact RESHAPE/target-shape semantics with one-input
provenance, and stay unlabeled and storage-free. Only resolved contiguous input plus a static
target produces same-offset canonical view metadata; all other result layout remains unresolved.
Package-private structural capture can preserve this current model expression. Graph-wide dynamic
constraint solving, reshape-chain canonicalization, materialization planning, backend alias/copy
lowering, and execution remain planned.
`Tensor.expand(long...)` treats every requested extent as a literal non-negative dimension, while
`Tensor.expand(Shape)` retains the exact target reference. Both overloads require target rank at
least input rank and accept aligned dimensions only when they are structurally equal or the input
is a static singleton; new leading target axes are valid. A fully static target and any resolved
input layout produce new same-offset view geometry with preserved aligned strides and zero strides
for leading or expanded-singleton axes. Dynamic target or unresolved input geometry stays
unresolved. Every result retains exact type and gradient eligibility, records exact
EXPAND/target-shape semantics with one-input provenance, and remains fresh, unlabeled, and
storage-free. Package-private structural capture can preserve this model construction, but it does
not repeat values or establish storage aliasing, dynamic constraint solving, gradient behavior,
canonicalization, materialization, lowering, or execution.
`Tensor.permute(int...)` accepts every current data type, requires a complete output-to-input axis
mapping, normalizes each negative axis once, and reorders exact Dimension references. Any resolved
input layout produces a new same-offset view descriptor with exact reordered strides; unresolved
input layout remains unresolved. `Tensor.transpose()` requires rank two and uses the same PERMUTE
construction with normalized axes `[1, 0]`. Every result preserves type and gradient eligibility,
records exact normalized attributes and one-input provenance, and remains fresh, unlabeled, and
storage-free. Package-private structural capture can preserve the occurrence; permutation
canonicalization, physical aliasing, materialization, lowering, and execution remain planned.
`Tensor.expandDims(int)` inserts one static singleton at a caller position normalized against the
result rank. `Tensor.squeeze(int)` instead normalizes an existing input axis and removes it only
when its dimension is statically known as one. Both preserve exact type and gradient eligibility,
retain exact unaffected Dimension references, record matching `AxisTransformAttrs` and one-input
provenance, and leave unresolved input geometry unresolved. Resolved input geometry produces one
new same-offset logical view descriptor with a deterministic checked stride inserted or the
selected stride removed. Package-private structural capture can preserve the occurrence; dynamic
singleton constraint solving, inverse-pair canonicalization, physical aliasing, materialization,
gradient behavior, lowering, and execution remain planned.
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
unlabeled, and storage-free. Package-private structural capture can preserve the occurrence;
slice-chain or flip canonicalization, physical aliasing or copying, gradient-scatter construction,
materialization, backend/ONNX lowering, and execution remain planned.
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
Dimension remain deferred. Both primitives are structurally capturable, but neither represents a
constraint, constructs an adjoint, mutates values, chooses materialization, lowers, or executes
work. A later compiler may compose them for Slice, Select, Pad, or Concat adjoints, but no such
adjoint construction, binding proof, saved-value policy, accumulation, or canonicalization is
implemented today.
`Tensor.select(int, long)` normalizes one source axis and one scalar coordinate, removes the
selected Dimension, preserves every unaffected exact Dimension reference, and records normalized
`SelectAttrs` with exact one-input provenance. A static selected extent supplies immediate
negative-index normalization and bounds checks; a non-negative coordinate on a dynamic selected
extent remains representable with its upper bound deferred, while a negative coordinate is
rejected. Resolved input geometry with a non-empty result produces checked selected-stride removal
and offset advancement in one new logical view descriptor; unresolved input and empty results stay
unresolved. The fresh result preserves exact type and eligibility and has no label or storage.
Package-private structural capture can preserve this expression. Value selection, physical
aliasing, gradient construction, canonicalization, materialization, backend lowering, and
execution remain planned.
`Tensor.gather` and `gatherElements` consume exact ordered `[data, indices]` inputs with `INT32` or
`INT64` indices. They normalize one data axis and apply canonical axis-replacement or same-rank
aligned-Shape rules. Every fresh result retains data type and gradient eligibility, leaves layout
unresolved, and records exact two-input provenance. Package-private structural capture can
preserve the occurrence. Construction interprets no index values, checks no index-value bounds,
and adds no gradient rule, canonicalization, materialization, lowering, or execution behavior.
`Tensor.gatherNd` consumes exact ordered `[data, indices]` inputs with `INT32` or `INT64` indices.
Its short form uses zero shared batch Dimensions; its complete form retains one normalized
non-negative batch count. Construction validates both ranks, structurally equal leading batch
Dimensions, and a statically known positive tuple depth from the final indices Dimension. It
derives the result as the indices prefix without tuple depth followed by the untouched data
suffix, including canonical scalar and exact retained Dimension references. Every result is fresh,
unlabeled, storage-free, and unresolved-layout, preserves data type and gradient eligibility, and
records `GATHER_ND` with exact `[data, indices]` provenance. Package-private structural capture can
preserve the occurrence. Construction reads no index value, checks no index-value bound, and adds
no gradient, compiler transformation, materialization, lowering, or execution behavior.
Both `Tensor.scatterElements` overloads consume exact ordered `[data, indices, updates]` inputs.
They require `INT32` or `INT64` indices and exact matching data/update types. `NONE` is permitted
for every current type and arithmetic reductions for floating or integral values. Each method
normalizes one raw data axis, validates the same-rank Shape relationship, and returns a fresh
result with the exact data Shape/type, data/update gradient-
eligibility OR, unresolved layout, and exact three-input provenance. Construction reads no values,
checks no index bound or duplicate target, mutates no input, and performs no write or reduction.
Package-private structural capture can preserve the occurrence; gradient rules, compiler
transformations, materialization, lowering, backend behavior, and execution remain planned.
`Tensor.scatterAdd(indices, updates, axis)` is now the current Gather-compatible fixed-add model
primitive. It records `SCATTER_ADD` with normalized `IndexAxisAttrs` and ordered
`[data, indices, updates]` inputs. Updates must have the exact Shape that `data.gather(indices,
axis)` would produce, while the functional result retains the exact data Shape and accumulates
duplicate targets. Construction validates metadata only and leaves layout unresolved.
Package-private structural capture can preserve the occurrence, but no code constructs a
Gather/embedding adjoint, inspects or bounds-checks indices, lowers the operation, or executes
addition.

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
three-input provenance. Package-private structural capture can preserve the occurrence. Model
construction reads no index or update value, checks no index bound or duplicate target, mutates no
input, performs no write or reduction, and adds no gradient, compiler transformation,
materialization, lowering, backend behavior, or execution behavior.
Static `Tensor.concat(int, Tensor...)` and `Tensor.stack(int, Tensor...)` snapshot ordered non-empty
inputs, normalize an existing or inserted axis, enforce exact type and operation-specific Shape
rules, and create fresh unresolved-layout results with eligibility OR and exact ordered provenance.
Instance `Tensor.unstack(int)` requires a static `int`-sized selected extent and returns an
immutable ordered List of scalar `SELECT` expressions after upfront count validation. Each fresh
result removes that axis, has an independent one-output producer, and uses provenance index zero
over the same input; its `SelectAttrs` stores the coordinate. A zero extent returns no result or
ID. Package-private structural capture preserves each independent occurrence; decomposition,
grouping, backward construction, materialization, backend lowering, ONNX mapping, and execution
remain planned.
`Tensor.unfold`, `Tensor.foldAxis`, both `Tensor.unfold2d` forms, and `Tensor.fold2d` construct the
current public storage-free window-transform expressions. General-axis fold restores an explicit
target extent under overlap summation. The 2D forms preserve canonical rank-three im2col/col2im,
retain exact static or symbolic channel/spatial formulas, and distinguish direct conceptual-zero
padding from an exact typed padding scalar. `fold2d` accepts only complete structural formula
matches rather than recording equality between unrelated unresolved symbols. Every result
preserves input data type and gradient eligibility, leaves layout unresolved, and records exact
one-input provenance. Package-private structural capture can preserve these occurrences, while the
model contracts provide no canonicalization, gradient generation, lowering, or execution.
That origin metadata is now traversed by the package-private structural capture described above.
The internal step observes that two result Tensors belong to the same expression occurrence only
when their provenance carries the same exact `TensorProducer` reference. Different output indices
then select different output positions of one captured node. The producer does not prescribe
graph-local identity: each capture assigns fresh `NodeId` and `ValueId` sequences from its own
deterministic traversal.

Dropout is the first current production occurrence that uses this shared-output seam. Its public
output selects producer slot zero and its opaque next-state wrapper retains slot two. The producer
also describes slot one as a same-Shape BOOL auxiliary keep mask even though `DropoutResult` exposes
no mask Tensor. Current internal capture creates graph values for all three slots by reading the
reachable producer's ordered descriptors and also preserves the zero-input initial-state producer
and its edge into dropout. It needs no public sibling-output lookup. Auxiliary-value lifetime
policy, dropout gradient construction, and backward saved-value selection remain planned.

Explicit attention output-and-weights construction uses the same foundation without an auxiliary
or hidden output. `ScaledDotProductAttentionResult.output()` and `weights()` expose the exact
wrappers for producer slots zero and one. If either wrapper is requested, current internal capture
emits their shared producer once and creates both ordered graph values. Attention operand
revalidation, backward preservation policy, adjoints, and lifetime policy remain planned.

## Current compile-configuration inputs

`io.github.pho001.synaptik.config.compile.BackendIntent` is current declarative compile
configuration. It holds exactly one `Optional<BackendRequirement>` named `hardRequirement`:

```java
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.BackendIdRequirement;
import io.github.pho001.synaptik.config.compile.BackendIntent;

BackendIntent unconstrained = BackendIntent.unconstrained();
BackendIntent requireCuda =
        BackendIntent.requiring(
                new BackendIdRequirement(new BackendId("cuda")));
```

The first value has no hard eligibility target. The second retains the exact supplied requirement
reference for later planning. Neither construction evaluates availability or capability, ranks
candidates, chooses ownership, discovers a service, or invokes a compiler. In particular,
`unconstrained()` does not mean “automatic backend selection succeeded”; it records only the
absence of a hard target.

The canonical constructor rejects a null optional with `NullPointerException("hardRequirement")`.
The `requiring` factory rejects a null requirement with `NullPointerException("requirement")`.
Current internal planning evaluates the hard target before its cost-free baseline owner selection.
General score calculation, planning cost profiles, `CompileConfig`, compiler consumption, and a
public no-match failure remain planned. The separate current `PartitionScoringConfig` value
described below supplies only soft preference input; it does not change these hard-intent
semantics.

`io.github.pho001.synaptik.config.compile.CompileMode` is also current. It contains exactly three
requested graph scopes in declaration order:

```java
import io.github.pho001.synaptik.config.compile.CompileMode;

CompileMode inferenceScope = CompileMode.FORWARD_ONLY;
CompileMode gradientScope = CompileMode.FORWARD_AND_BACKWARD;
CompileMode trainingStepDirection = CompileMode.TRAINING_STEP;
```

`FORWARD_ONLY` requests forward graph construction and requested forward publications.
`FORWARD_AND_BACKWARD` requests later compiler autograd expansion plus combined forward/backward
compile-time graph work. `TRAINING_STEP` records the architecture's future training-step direction;
it does not itself introduce an optimizer, optimizer-update graph, session, schedule, or execution
behavior. No current compiler interprets any of the three values.

`io.github.pho001.synaptik.config.compile.GraphOptimizationConfig` is another current value:

```java
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;

GraphOptimizationConfig noOptionalOptimization =
        GraphOptimizationConfig.disabled();
GraphOptimizationConfig standardOptimization =
        GraphOptimizationConfig.standard();
```

The first value requests that the compiler skip optional semantics-preserving optimization.
It cannot disable graph capture, ordering, inference, validation, mandatory canonical
representation, mode-required autograd, publication binding, planning, preparation, or execution.
The second permits the compiler's standard optional semantics-preserving pipeline without freezing
a pass list, pass order, internal graph shape, or implementation strategy. It permits no
approximate mathematics, changed numerical semantics, backend-specific fusion, preparation, or
execution behavior. Direct construction retains either primitive boolean value, and both factories
return fresh values. The current package-private compiler transformation boundary consumes this
permission; no public compiler entry point or `CompileConfig` aggregate does.

`io.github.pho001.synaptik.config.compile.PartitionScoringConfig` is the fourth current value. It
holds exactly one `Optional<DeviceClass>` named `preferredDeviceClass`:

```java
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;

PartitionScoringConfig neutralRanking = PartitionScoringConfig.neutral();
PartitionScoringConfig preferAccelerator =
        PartitionScoringConfig.preferring(DeviceClass.ACCELERATOR);
```

The first value contains an empty optional and means only that no explicit coarse device-class
preference was supplied. It selects no aggregate default or fallback and promises neither equal
candidate scores nor successful ownership selection. The second contains the exact supplied enum
reference as a soft preference that planning may apply only after hard eligibility. It does
not remove another eligible candidate, weaken a hard `BackendRequirement`, or guarantee that an
accelerator candidate wins.

Direct construction retains the exact non-null `Optional<DeviceClass>` reference and rejects null
with `NullPointerException("preferredDeviceClass")`. `preferring(null)` rejects null with
`NullPointerException("deviceClass")`; both factories return fresh records. The value performs no
candidate enumeration or evaluation, score calculation or comparison, profile lookup, ownership
or device selection, route or kernel selection, compilation, preparation, runtime work, or
execution. Current package-private planning consumes only its optional class preference: the first
eligible class match wins, or the first eligible backend wins when no preference or match exists.
The backend-neutral cost classification, planning cost inputs, public scoring evaluation,
compiler consumer, and aggregate remain planned.

## Current operation-capability contracts

`io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery` describes one immutable
operation occurrence for a compile-time capability question. Its exact ordered components are an
`Operation`, a `List<TensorDescriptor>` of inputs, and a `List<TensorDescriptor>` of outputs. The
canonical constructor checks the three top-level references, scans inputs in encounter order,
snapshots input membership, scans outputs in encounter order, snapshots output membership, and
then validates only the two counts against `operation.signature()`.

The snapshots retain the exact descriptor element references and are immutable. Empty or repeated
inputs are valid when the signature permits them, and repeated output descriptor references and
valid multi-output occurrences are representable. Null failures identify the top-level component
or first encountered element, such as `inputs[1]`; signature count failures retain the existing
`OperationSignature` behavior. The query performs no operand compatibility, graph, availability,
requirement, scoring, device, route, prepare, runtime, or execution validation.

`BackendCapabilityProvider` names one stable non-null `BackendId` through `backendId()` and answers
`supports(query)` for that same backend. For an immutable query and unchanged immutable provider
configuration, the boolean answer is deterministic. Implementations reject a null query with
`NullPointerException("query")`. `true` means only that the backend can semantically own the
described occurrence; `false` carries no reason.

The provider is supplied explicitly to compile-time planning. The public contract performs
no registry, discovery, classpath scan, `ServiceLoader` lookup, availability or hard-requirement
evaluation, scoring, route or kernel selection, preparation, or execution. The repository supplies
no production provider implementation and no compiler consumer. Compile-time plans will retain a
`BackendId`, not a provider object.

Current package-private planning implementation validates a complete association between ordered
providers and caller-supplied availability snapshots by equal `BackendId`, then applies non-empty
availability and the optional exact hard target before invoking each still-possible provider once.
It returns only an immutable provider-ordered list of eligible `BackendId` references; an empty
list is a valid internal no-match result. Exact-device and device-class requirements consult the
matching snapshot only for availability. They do not make the boolean answer device-level, select
a device, or retain one.

Current package-private baseline selection treats that list as the complete candidate set. It
validates the full snapshot input for nulls and duplicate equal identities, requires one equal-ID
snapshot for each eligible backend, and permits extra unique snapshots. Empty eligibility fails
before snapshot-element reads. A neutral configuration returns the first eligible identity; a
present preference returns the first provider-order class match or falls back to the first
eligible identity. Empty snapshots are preference nonmatches, and the exact eligibility identity
reference is returned. No provider is called again and no device, route, or kernel is selected.

Reusable or public capability matrices, a public eligibility result/evaluator or owner selector,
public planning orchestration, numeric or cost scoring and profiles, owner-map assembly, compiler
integration, device-level capability or selection, preparation, runtime, execution, and typed
rejection diagnostics remain planned.

## Current planned-partition recipe

`io.github.pho001.synaptik.planning.partition.PlannedPartition` is current public compile-time
data with exactly two ordered components:

```java
BackendId owner
List<NodeId> nodeIds
```

The owner is one non-null backend ownership identity. The node list is non-null, non-empty,
contains no null or duplicate identity by equality, and is copied into an immutable ordered
snapshot. The record retains the exact owner and element references, uses ordinary record
equality and hashing, and carries no graph values, partition ID, boundary edges, transfer or
memory facts, selected device, route, kernel, executable, or runtime state.

The package-private generator accepts one immutable `CompiledGraphModel` and one complete
`Map<NodeId, BackendId>`. Equal map keys associate with graph nodes, but output membership always
uses the exact `NodeId` references from `graph.nodes()`. It validates all keys, graph coverage, and
owners before constructing a result. For each maximal run it retains the exact owner reference
mapped to the first node and compares later owners with `BackendId.equals`.

For graph order `[n0, n1, n2, n3, n4]` and owner values
`[cpu, cpu, metal, metal, cpu]`, the current result meaning is:

```text
PlannedPartition(cpu,   [n0, n1])
PlannedPartition(metal, [n2, n3])
PlannedPartition(cpu,   [n4])
```

The final CPU node stays separate because only consecutive positions in the stored validated
topological order are adjacent. Conversely, consecutive independent nodes with equal owners join.
Graph edges, phase changes, fan-out, merges, repeated input positions, graph input/output values,
and multiple output values from one producer do not independently create boundaries. A valid
zero-node pass-through graph produces an immutable empty list.

The generator itself is intentionally not public. No current API assembles the complete owner map
from capability and selection results or invokes partitioning for callers. The following internal
logical-memory step is current, but compiler orchestration and `CompileArtifacts` remain planned.
A partition is an immutable ownership recipe, not a promise of one fused kernel, one executable,
or any prepare/runtime behavior.

## Current logical-memory recipes

`io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement` is current public
compile-time data with exactly these ordered components:

```java
ValueId valueId
TensorDescriptor descriptor
Optional<PlannedPartition> producerPartition
List<PlannedPartition> consumerPartitions
boolean graphOutput
```

`producerPartition` is empty for a graph input in a generated plan. Each generated consumer list
contains a supplied partition at most once and follows supplied partition order, regardless of
repeated input positions or multiple consuming nodes inside one partition. `graphOutput` is true
exactly when an equal value identity occurs in `CompiledGraphModel.outputs()`. The generated
requirement retains the exact graph value identity and descriptor references plus exact supplied
partition-element references.

`LogicalMemoryPlan` contains one immutable ordered `List<LogicalMemoryRequirement>`. Its public
constructor permits an empty standalone plan, rejects null elements and duplicate equal
`ValueId` values, snapshots list membership, and retains exact requirement references. The public
requirement constructor likewise snapshots consumer membership and rejects null or equal
duplicate partitions. Directly constructed DTOs have no owning graph and therefore do not prove
that caller-supplied producer, consumer, or output facts are graph-valid.

The package-private derivation accepts one `CompiledGraphModel` and one ordered
`List<PlannedPartition>`. It completes all partition null, graph-membership, coverage, graph-order,
and adjacent-owner maximality checks before constructing a requirement. It then emits exactly one
requirement for every `graph.values()` entry in that encounter order. Producer facts come from
node outputs; consumer facts come from node inputs. Repeated inputs, fan-out, merges, unused graph
inputs, unused produced values, multi-output nodes, forward/backward uses, and zero-node
pass-through graphs require no synthetic node or role.

The primitive facts describe overlapping logical roles: graph input, partition input, partition
output, same-owner or cross-owner boundary, graph-output preservation, and partition-internal
value. The records do not store another role enum. They retain `TensorDescriptor` so static,
dynamic, and expression Shapes remain representable without calculating an element or byte
count.

This is logical planning only. The derivation accepts no `PublicationBinding` or `TensorId` and
creates no `PublicationPlan`. It selects no alias, copy, transfer, lifetime, physical slot,
allocation, device, route, kernel, schedule, executable, or runtime residency. The derivation is
not public, and no current compiler invokes it end to end.

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
  scaled-dot-product-attention construction with optional BOOL mask and scale/causal attributes,
  preserving the original one-output family and adding an explicit same-producer output/weights
  family, plus grouped NCHW Conv2d
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
  the public compiler entry point and every lifecycle orchestration surface,
  saved-statistic and gradient construction,
  optional
  softmax, layer-normalization, RMS-normalization, attention, or activation decomposition,
  activation/attention-gradient construction,
  redundant-cast, redundant-contiguous, reshape/expand/permutation/rank-edit, slice chains, slice
  updates, crops, selects,
  axis-gather/one-hot/Gather-ND/axis-scatter/Scatter-ND/pad/tile/composition/window-transform
  canonicalization or decomposition,
  compiler-generated use of public `FOLD_AXIS`, possible use of slice-update and crop metadata in
  adjoint construction, concrete dynamic binding and value-dependent select/index validation,
  legal
  convolution/pooling decomposition and gradient construction, maximum-pooling saved-index
  policy, average-pooling fixed-divisor preservation, layout materialization
  planning remain planned. Package-private identity-based structural forward capture into graph
  values and nodes, binding-free verification inference, mandatory dense graph-local
  canonicalization, the guarded seven-rule exact arithmetic scan, the bounded logical-splat fold
  matrix, and the exact one-shot sidecar-aware DCE/CSE/DCE cleanup are current internal behavior.
  All other operation-specific rewrites listed above remain planned.
- `GraphRngState` is an implemented opaque model expression value rather than a public numerical
  `Tensor` output. Current dropout places its private state Tensor at producer input one and wraps
  producer output two as the next state. Package-private structural capture preserves the
  reachable initial-state producer, state edge, and every dropout output slot. Direct state
  boundary selection, serialization, value rewriting, lifetime policy, and public compilation
  remain planned. Current internal canonicalization preserves every state slot, exact CSE merges
  only complete equal occurrences, and DCE retains all slots of a live dropout node.
- `CompileConfig` will aggregate the current compile mode, `BackendIntent`, graph-optimization,
  and `PartitionScoringConfig` values with any later justified planning-cost inputs as data. It will not
  contain live backend services. Its exact surface and defaults remain planned.
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

This scenario is conceptual. Current internal capture can build the structural forward graph and
map explicit logical-splat ingress, package-private verification can validate its operation
descriptors and retain unresolved constraints, and the current internal transformation boundary
can canonicalize and apply its guarded seven-rule scan, bounded exact constant-folding matrix, and
sidecar-aware forward DCE/CSE/DCE sequence. None can run the illustrated lifecycle, bind concrete
dimensions, select ownership, or expose a public compilation result.

## Related contracts

- [Current Tensor and graph-model API](tensor-api.md)
- [Lifecycle](../architecture/lifecycle.md)
- [Partition scoring](../architecture/partition-scoring.md)
- [Compiling graphs user guide](../user-guide/compiling-graphs.md)
- [Roadmap](../planning/roadmap.md)
