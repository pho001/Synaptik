# Compile API

## Purpose and implementation status

This reference separates the compile-time contracts implemented today from the public engine
lifecycle that remains planned. The compiler module contains package-private `GraphCompiler`.
Its five-argument entry compiles a forward-only or combined one/two-stage functional derivative
Tensor expression into package-private immutable `GraphCompilation`. A second package-private
nine-argument entry completes publication and backend-neutral planning and returns public
immutable `CompileArtifacts`.

The current stages are fail-closed autograd preflight, formula construction through public Tensor
operations, one phase-aware capture, binding-free captured-graph verification with retained
occurrence-local Shape predicates, mandatory dense
canonicalization, explicit logical-splat facts, one bounded exact whole-graph optimization
pipeline, publication-role validation, one owner selection per final graph node, maximal
same-owner partitioning, logical-memory derivation, and immutable artifact assembly. The
repository still provides no public compiler entry point, `CompileConfig` aggregate, engine
`CompiledGraph`, prepared execution, or runnable public graph compiler. It does provide the public
immutable `FunctionalGradientRequest` input value used by package-private compiler integration.
The current config module provides four immutable standalone input values that a later compile
configuration aggregate can contain: `BackendIntent`, `CompileMode`, and
`GraphOptimizationConfig`, plus `PartitionScoringConfig`. The current planning module provides the
immutable `OperationCapabilityQuery` and the explicitly supplied `BackendCapabilityProvider`
collaboration. It keeps per-query hard eligibility and baseline comparison package-private and
exposes one public `BackendOwnerPlanning.selectOwner(...)` collaboration that composes them
without exposing the intermediate. It also exposes the existing stateless
`MaximalSameOwnerPartitioning.partition(...)` and `LogicalMemoryPlanning.plan(...)` operations in
their owning packages. The module still provides no reusable/public capability matrix, public
graph-wide planner workflow, general cost scoring, or owner-map assembly. Compiler owns the
graph-wide loop and invokes those three Planning operations directly.

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
| `ForwardPublicationBinding` | Standalone `TensorId`-to-`ValueId` association used for requested forward outputs inside current compiler publication plans | Not an owning publication plan, a gradient binding, or a `CompiledGraphModel` component |

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

The original capture entries remain forward-only. One package-private combined entry receives
forward outputs, ordered target/gradient roles, the exact identity set of original forward
producers, and merged explicit splat ingress. It traverses forward and gradient roots together,
assigns graph-local IDs once, and classifies original producer occurrences as `FORWARD` and
generated occurrences as `BACKWARD`. The graph-output boundary contains the distinct forward
values first, then each gradient value absent from that prefix at its first role occurrence.
Target roles retain boundary ordinals, so two targets can share one gradient `ValueId` without an
identity node.

Capture itself performs no inference, transformation, or derivative-rule selection. The
following package-private boundaries revalidate, canonicalize, and optionally optimize the
captured graph. The original graph-stage entry stops there; the complete entry derives
publication, invokes Planning, and constructs `CompileArtifacts` only after final graph
validation. Neither entry prepares backends, allocates physical memory, or executes values. No public
`GraphCompiler`, `CompiledGraph`, compile entry point, configuration aggregate, capture method,
validation method, or optimizer method exists.

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

### Current package-private canonicalization and whole-graph optimization

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
                                   -> sidecar-aware whole-graph dead-code elimination
                                   -> exact phase-local common-subexpression elimination
                                   -> sidecar-aware whole-graph dead-code elimination cleanup
```

The first scan recognizes exactly seven internal one-output identities in either graph phase:
duplicate-input
binary `MIN` and `MAX`; scalar `MUL` by exact typed positive one for `BFLOAT16`, `FLOAT32`,
`FLOAT64`, `INT32`, and `INT64`; scalar `DIV` and `POW` by exact typed positive one for the three
floating types; and scalar `ADD` and `SUB` by exact typed zero for `INT32` and `INT64`. Every bypass
requires complete equality between the input and output `TensorDescriptor`, including Shape,
layout, data type, and `requiresGrad`; the equal descriptor must have `requiresGrad == false`.
The result must not be a graph output. The occurrence phase is preserved. Duplicate extrema
compare already-remapped input IDs, while
scalar rules require the exact `ScalarValueAttrs` carrier, matching scalar/input/output type, and
matching typed value. A prior bypass can therefore expose a later eligible occurrence during the
same scan without iteration.

`ScalarValueAttrs` is immutable operation metadata already stored in the graph. Reading its exact
typed `ScalarValue` for an identity proof is not Tensor constant discovery or host-storage
inspection. The scan deliberately does not add floating ADD/SUB-zero, multiplication by
zero, cancellation, other powers, bounds/clamp identities, broader algebra, or relaxed numerical
rules. It retains graph-output, gradient-eligible, and multi-output occurrences.

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

Each selected fold replaces one internal, one-output, non-gradient occurrence in either phase with
one
synthetic constant graph input. Original inputs remain first; synthetic sources follow in original
fold-occurrence order; retained outputs and nodes are then rebuilt densely and deterministically.
Graph-output producers, all multi-output producers, and gradient-eligible values remain
occurrences. Equal splats are not interned.

Floating and BFLOAT16 splats retain exact type and bits in the sidecar, including signed zeros and
NaN payloads, but no floating or BFLOAT16 operation is evaluated. Casts, unselected scalar kinds,
unary numerical functions, reductions, scans, normalization, loss, linear algebra, indexing,
layout/view operations, random/state operations, and generic partial evaluation are also outside
the current fold matrix. The compiler reads no host storage, creates no dense constant payload,
and allocates no physical value.

Dead-code elimination (DCE) derives liveness only from the complete ordered graph-output boundary,
so an unused `BACKWARD` node is removable just like unused `FORWARD` work. Every live occurrence
retains all output slots. The sidecar-aware overload then removes only fixed constant inputs that
are neither graph outputs nor inputs to a retained node; it always preserves bindable inputs,
even when unused.

Exact common-subexpression elimination (CSE) compares graph phase, complete immutable operation
value, ordered remapped inputs, and every ordered output descriptor. It merges a whole
multi-output occurrence slot by slot within `FORWARD` or within `BACKWARD`, never across phases.
A producer containing a graph output may neither merge nor serve as a representative, so
requested boundaries remain distinct. Each changed candidate is immediately revalidated through
the verification pass; an unchanged helper result is not redundantly revalidated.

These are internal graph transformations, not a public compiler surface. They do not rewrite
casts, broader arithmetic, algebra, or views; construct autograd; derive publication, planning, or
diagnostics; or affect trace, preparation, runtime, backend, or engine behavior. The current
complete compiler entry transports the exact immutable facts into `CompileConstantPlan`, exposing
only derived bindable inputs and exact logical-splat sources. Planning sees the unchanged logical
graph. Future prepare/backend work owns physical splat materialization, storage allocation,
lowering, and execution.

### Current package-private pre-capture autograd

Model task 0025 is Complete, so a producer now returns the canonical exact Tensor wrapper for
every output slot, including hidden dropout masks and batch-normalization saved statistics,
without changing public Tensor methods or ergonomic result carriers. Compiler task 0004 is
Complete and supplies the first internal autograd consumer of that identity contract. Compiler
tasks 0004A and 0004B are also Complete with the exact-composition and shared-algebra local-rule
extensions described below. Compiler task 0005A is Complete and closes the exact 48-kind current
elementwise/activation inventory with explicit derivative and non-differentiable-role policies.
Compiler task 0005B is Complete and adds binding-aware expansion plus the current reduction,
scan, softmax, statistics, norm, and Layer/RMS/batch-normalization first-order matrix.
Compiler task 0005C is Complete and adds dynamic slice/window constraints plus the current
layout, Gather/scatter, ordering/top-K, and explicit-state dropout first-order matrix.
Compiler task 0005D is Complete and adds the current two-output attention, grouped convolution,
pooling, and loss first-order matrix without adding a public gradient API or backward-only
operation vocabulary.
Compiler task 0005E closed that matrix against the production Model inventory at its checkpoint:
37 operation-kind enum families, 107 constants, and 128 complete
kind/attributes/input-range/output-range fingerprints. The checkpoint includes both
`SLICE_UPDATE` attributes variants. Model 0025E subsequently added one recurrent family, three
constants, and three exact signatures without Compiler adoption. The current complete Model
inventory is therefore 38 families, 110 constants, and 131 signatures, while the supported
Compiler inventory remains the exact original 128 signatures.

Package-private `GraphCompiler.compile` takes `CompileMode`, ordered forward outputs, an optional
public `FunctionalGradientRequest`, explicit forward constant ingress, and
`GraphOptimizationConfig` directly; there is no request aggregate. It returns package-private
mode-neutral `GraphCompilation`. `FORWARD_ONLY` requires no functional request, produces no
`BACKWARD` nodes, and has empty gradient results. `FORWARD_AND_BACKWARD` and the current internal
`TRAINING_STEP` path require the same request and may carry one combined graph.
`GraphCompilation` retains the final `ValidatedGraph`, ordered forward output values, and ordered
public `GradientPublicationBinding` values. Its package-private component names remain
`forwardOutputs` and `gradientResults`; the public publication plan instead exposes
`forwardBindings()` and `gradientBindings()`. It is graph-stage state, not the current public
cross-package `CompileArtifacts`.

The current functional request contains exactly one or two reverse-mode stages. Stage-one outputs
identify exact requested forward Tensors; stage-two outputs identify generated first-stage
gradients by target-list index. A one-stage request has `createGraph=false`; a two-stage request
has `true` for stage one and `false` for stage two. Each stage has a non-empty,
exact-identity-unique target list drawn from the complete original forward inventory and an
output-aligned list of optional cotangent seeds. An absent seed is valid only for a scalar,
floating, gradient-eligible output and creates an exact typed positive one. A present seed must
have the output's exact Shape and floating data type and must not request gradients. Multiple
selected outputs contribute a deterministic sum of vector-Jacobian products.

`DisconnectedPolicy.ERROR` rejects a target without a differentiable route.
`DisconnectedPolicy.ZERO` returns an ordinary exact typed zero expression; several result roles
may share that value. Results are ordered by derivative order and then stage-local target index.
`DerivativeGraphMetadata` classifies final nodes as derivative order zero, one, or two while
unchanged Model `GraphPhase` continues to distinguish only forward and backward nodes.

### Functional-gradient examples

These are compiler-package integration sketches because `GraphCompiler` remains package-private;
they describe current semantics without claiming a public compile facade.

For scalar `loss = x * x`, request stage one from `loss` to `x` with its default seed and
`createGraph=true`, then request stage two from
`FirstStageGradientReference(0)` to `x` with its default seed and `createGraph=false`. One combined
compile returns an order-one role for `2 * x` and an order-two role for `2`; both are ordinary
values in the same graph, and no Tensor gradient field is written.

For a vector output, supply an exact-Shape explicit seed to select its vector-Jacobian product.
A multi-output stage supplies one aligned seed per output and sums those products in output order.
The second stage may then seed a selected first-stage gradient with an exact-Shape vector to form
a Hessian-vector product. Omitting a seed for any non-scalar output is rejected before derivative
formulas are allocated.

Before constructing the seed or any formula Tensor, `AutogradPreflight` iteratively inventories
the complete original forward request. The package-private first-order coverage checker assigns
each exact selected output/input role one current disposition: conditionally differentiable
(`D`), intentionally non-differentiable (`ND`), or fail-closed (`FC`). Every `ND` or `FC`
decision has a deterministic reason. Every `D` decision names exactly one existing formula-family
owner, but remains conditional until preflight validates the occurrence's exact attributes,
cardinality, data-type and Shape relationships, canonical auxiliaries, normalization path, and
derivative policy. Unknown kinds, unclassified kind/attributes pairings, malformed
cardinalities, illegal slots, and unsupported prerequisites fail closed before derivative Tensor
identity allocation.

The same family-owner decision is retained in the successful preflight plan and drives
`FirstOrderAutograd` dispatch, so role selection and formula routing do not maintain independent
kind branches. Named `ElementwiseGradientRules`, `ReductionGradientRules`,
`NormalizationGradientRules`, `LinearAlgebraGradientRules`, `LayoutGradientRules`,
`IndexingGradientRules`, `OrderingGradientRules`, and `StochasticGradientRules` then build
formulas only with existing public Tensor operations. `AttentionGradientRules`,
`ConvolutionGradientRules`, `PoolingGradientRules`, and `LossGradientRules` own the structured
Compiler 0005D formulas. Exact Tensor identity keys ordered contributions during one compile
request. Reverse accumulation visits producer postorder in reverse, selected output slots in
ascending order for that producer, and input positions in ascending order; ordinary
left-associated `Tensor.add` accumulates contributions. This ephemeral state is neither Tensor
state nor graph intermediate representation (IR).

The current matrix is the union of the Compiler 0004–0004B rows and the Compiler 0005A–0005D
family completions:

| Family | Current exact supported variants |
|---|---|
| Elementwise and activation | All seven promoted floating binary arithmetic kinds: `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, and `POW`; all eight exact-type floating scalar kinds, including first-class `CLAMP`; promoted floating branch-only `WHERE`; floating-to-floating `CAST`; and all nineteen floating unary kinds from `ABS` through `SILU` |
| Reductions, scans, and softmax | Ordinary full, single-axis, and ordered multi-axis floating `SUM`, `MEAN`, `PROD`, `MIN`, and `MAX`; masked floating `SUM` and `MEAN`; binding-aware floating `SUM_TO_SHAPE`; floating `LOG_SUM_EXP`, `VARIANCE`, `STANDARD_DEVIATION`, `L1_NORM`, and `L2_NORM`; floating `CUM_SUM` and `CUM_PROD` for every exclusive/reverse combination; floating `SOFTMAX` and `LOG_SOFTMAX` |
| Normalization | No-affine and affine floating `LAYER_NORM`; no-scale and scaled floating `RMS_NORM`; all five floating inputs of batch-normalization inference; and the exact output-slot-specific batch-normalization-training roles for public slots zero through two |
| Linear algebra | Every current floating `MATMUL` vector/matrix rank pairing, with role-aware mixed-floating normalization and batch unbroadcasting |
| Logical layout and selection | Floating `CONTIGUOUS`, `RESHAPE`, binding-aware `EXPAND`, `EXPAND_DIMS`, `SQUEEZE`, `PERMUTE`, both `SLICE`/`SLICE_UPDATE` attribute forms, `SELECT`, `PAD`, `TILE`, `CONCAT`, and `STACK` |
| Window transforms | Floating `UNFOLD_AXIS`, `FOLD_AXIS`, both `UNFOLD2D` attribute forms, and `FOLD2D` through their exact public inverse or overlap-add operation |
| Indexing and scatter | Floating data for `GATHER`, `GATHER_ELEMENTS`, and `GATHER_ND`; floating base and update roles for `SCATTER_ADD`, `SCATTER_ELEMENTS`, and `SCATTER_ND` with `NONE`, `ADD`, `MUL`, `MIN`, or `MAX` as applicable |
| Ordering | One-output floating `SORT` through one exact matching stable `ARGSORT`; floating `TOP_K` values slot zero through the canonical indices at slot one |
| Stochastic | Floating `DROPOUT` values slot zero through the exact same-occurrence BOOL keep mask; mask and graph-state roles remain non-differentiable |
| Attention | Both public outputs of exact two-output floating `SCALED_DOT_PRODUCT_ATTENTION`: query, key, and value from values slot zero; query and key from canonical weights slot one; an optional BOOL mask remains non-differentiable |
| Convolution and pooling | Every floating input, weight, and optional bias role of grouped NCHW `CONV2D`; the sole floating input of fixed-count `AVERAGE_POOL2D` and exact first-logical-winner `MAX_POOL2D` |
| Losses | Prediction and target of floating `MEAN_SQUARED_ERROR`; logits and dense floating targets of dense categorical cross-entropy with logits; logits only for index-target categorical cross-entropy with a positive static class depth |

Forward and generated expressions use one model-owned Tensor algebra. For a selected mixed-
floating input, the compiler first reverses ordinary broadcasting with `sumToShape` when needed
and then uses ordinary `cast` exactly when the contribution type differs from the input type.
The resulting contribution has that input's exact Shape and floating data type before
deterministic accumulation. `WHERE` applies this rule independently to its two branch roles, and
MATMUL applies it after its rank-specific formula and batch-unbroadcast step. The BOOL condition,
mask, and other non-floating control roles receive no cotangent.

Binary DIV constructs the left contribution as `g / right` and the right contribution in the
exact order `-(g * left) / (right * right)`. Scalar DIV constructs `g / scalar`. These are
ordinary Tensor expressions: they inherit the same division, multiplication, negation, NaN,
infinity, signed-zero, overflow, underflow, rounding, validation, and exact-optimization
contracts as forward expressions. No gradient-specific singularity or exceptional-value rule is
added.

Binary `MIN` and `MAX` route `g` to the strictly selected operand and split an exact numeric tie
equally. Each selected contribution then uses ordinary `sumToShape`-then-`cast` normalization.
Binary `POW(left, right)` constructs `g * right * pow(left, right - 1)` for `left` and
`g * output * log(left)` for `right`; it deliberately does not replace the first formula with
`output / left`.

Scalar `MIN` and `MAX` give the Tensor receiver `g` on its strict side, `g * 0.5` at an exact
numeric tie, and exact positive zero otherwise. Scalar `POW` constructs
`g * exponent * pow(input, exponentMinusOne)`. The derived scalar uses exactly one subtraction in
the represented type: BFLOAT16 expands to binary32, subtracts binary32 one, and rounds back;
FLOAT32 and FLOAT64 subtract one in their own formats. First-class `CLAMP` differentiates the
exact ordered forward composition `MIN(MAX(input, minValue), maxValue)`. With distinct finite
bounds the contribution is zero outside, `g` strictly inside, and `g / 2` at either endpoint;
when both extrema stages tie it is `g / 4`.

The added unary formulas are:

| Kind | Input contribution |
|---|---|
| `ABS` | `where(x > 0, g, where(x < 0, -g, 0))` |
| `RECIPROCAL` | `-g / (x * x)` |
| `LOG` | `g / x` |
| `LOG1P` | `g / (x + 1)` |
| `SQRT` | `g / (2 * output)` |
| `RSQRT` | `-0.5 * g * output * output * output` |
| `RELU` | `where(x > 0, g, 0)` |

Exact GELU uses the derivative of `x * Phi(x)`. Fixed tanh-approximation GELU differentiates its
specified `0.044715` cubic approximation, and SiLU differentiates `x * sigmoid(x)`. Each of these
three activations selects `g` at positive infinity and exact positive zero at negative infinity
to avoid a spurious infinity-times-zero NaN; finite and NaN inputs use the ordinary analytic
formula. Every other raw formula retains ordinary shared Tensor exceptional behavior without a
compiler-inserted finite, domain, singularity, or continuous-extension mask.

Piecewise extrema, `CLAMP`, `ABS`, and `RELU` use represented-value comparisons. Equal finite
values, equal same-sign infinities, and opposite signed zeros are ties. An unordered NaN makes
the ordered and equality tests false, so these piecewise rules return exact positive zero.
Comparisons, Boolean logic, floating classifications, the `WHERE` condition, scalar attributes
and clamp bounds, and non-floating cast roles remain non-differentiable.

`FLOOR`, `CEIL`, and `SIGN` use a direct exact positive-zero cotangent as their selected
first-order local convention. They do not construct `g * 0` or a floating comparison. This
convention does not change the forward operations' model semantics.

Ordinary MEAN restores and expands `g`, creates an input-shaped exact logical-one expression,
reduces those ones across the same full, single-axis, or ordered multi-axis domain, expands the
count, and divides. The same expression works for static, dynamic, expression, zero-sized, and
empty-axis-list domains without a host-computed count or type-specific threshold. Masked MEAN
constructs its per-slice true count with ordinary `WHERE` and `SUM`, divides the restored
cotangent by that count, then applies a final ordinary `WHERE`. The selected local convention
returns exact zero at every input coordinate of an all-false slice; it does not prescribe branch
evaluation or suppress the intermediate quotient.

For input `x` and output cotangent `g`, ERF constructs
`g * exp(-(x * x)) * (2 / sqrt(pi))`. The coefficient is exact scalar-operation metadata with
fixed BFLOAT16 bits `0x3F90`, FLOAT32 bits `0x3F906EBB`, or FLOAT64 bits
`0x3FF20DD750429B6D`; compilation does not evaluate a host transcendental function or add a
coefficient Tensor leaf.

Masked SUM restores the removed axis, expands `g` to the data Shape, and selects that value where
the exact BOOL mask is true or an explicit typed zero where it is false. The mask receives no
cotangent. A forward `sumToShape` is inverted only when every right-aligned target Dimension is
the exact input Dimension, is static one, or retains the exact target-one-or-target-equal-source
predicate needed by the inverse `expand`. Leading input axes are permitted. Conversely, a
binding-dependent forward `EXPAND` emits, in aligned-axis order,
`source == 1 || source == target`; its inverse `SUM_TO_SHAPE` is admitted only through that same
predicate. Deferred predicates remain attached to the occurrence for later binding validation.
The compiler does not bind a Dimension, choose a stride, or treat a deferred result as execution
permission.

`PROD` uses sequential keep-dimension product stages and reverses each stage with exclusive
prefix and suffix `CUM_PROD`; it never divides by the selected input. `CUM_PROD` replaces
represented zeros with one for safe products, counts zeros with `CUM_SUM`, and accumulates in the
opposite scan direction. These formulas handle zero-free, one-zero, multiple-zero, exclusive,
reverse, and zero-length structure without a gradient-specific infinity or NaN repair.

Reduction `MIN` and `MAX` compare each input with the exact restored forward output and divide the
restored cotangent equally among all matching coordinates. Opposite signed zeros are ties. A NaN
forward output matches no coordinate, so the selected contribution is exact positive zero.
`LOG_SUM_EXP` uses `restored(g) * exp(input - restored(output))`.

`VARIANCE` and `STANDARD_DEVIATION` derive selected-domain count by reducing exact logical ones
and construct correction as a typed logical-one reduction rather than a host floating
conversion. Standard deviation returns exact zero when the saved result is not strictly
positive. `L1_NORM` selects sign with exact zero at signed zero or NaN; `L2_NORM` returns exact
zero when its saved norm is not strictly positive. Other exceptional values follow the ordinary
Tensor formulas in their recorded order.

`SOFTMAX` consumes exact forward output `y` and constructs
`y * (g - sum(g * y, axis, true))`. `LOG_SOFTMAX` constructs
`g - exp(y) * sum(g, axis, true)`. Neither rule recomputes a softmax occurrence or selects a
forward numerical algorithm.

Layer and root-mean-square (RMS) normalization rebuild their population statistics with ordinary
Tensor operations over the exact trailing normalized axes. Mixed-floating occurrences compute in
the selected forward-output type, align affine operands by visible reshape/expand expressions,
and cast each completed Shape-correct contribution back to its selected input type. Layer
normalization supports input, scale, and bias roles; RMS normalization supports input and
optional scale.

Batch-normalization inference supports input, scale, bias, running mean, and running variance.
Batch-normalization training is output-slot-aware: normalized output slot zero contributes to
input/scale/bias, next-running-mean slot one contributes to input/running mean, and
next-running-variance slot two contributes to input/running variance. Its formulas retrieve the
exact same-occurrence saved mean and inverse-standard-deviation wrappers from slots three and
four. Those saved slots are formula inputs, not independent cotangent roots, publications,
physical buffers, or a runtime tape.

MATMUL handles vector/vector, vector/matrix, matrix/vector, and matrix/matrix formulas with public
rank edits, last-two-axis permutation, multiplication or MATMUL, followed by `sumToShape` for
each selected operand where batch broadcasting may have occurred and then one ordinary cast when
the promoted contribution type differs from that operand. Integral MATMUL remains
non-differentiable.

SLICE writes `g` into an input-shaped typed zero. `SliceAttrs` placement uses the exact stored
starts, axes, and steps; `CropToShapeAttrs` placement uses the exact prefix Shape. For
SLICE_UPDATE, the base role replaces the selected region of `g` with an update-shaped zero. The
update role uses `sliceByLength` with the exact recorded finite lengths for `SliceAttrs`, including
empty, signed-step, and unresolved selected-base regions, or `cropToShape` with the exact target
and prefix Shapes for `CropToShapeAttrs`. No raw end is reconstructed. SELECT writes an
axis-restored `g` at the selected coordinate. PAD crops away its before-width prefix. TILE
reshapes to interleaved repeat/input axes, sums the repeat axes, then restores the input Shape.
CONCAT crops `g` by each ordered symbolic input prefix, and STACK selects the corresponding
inserted-axis coordinate. Repeated input positions remain repeated contributions.

UNFOLD_AXIS uses `foldAxis` with the original selected extent and step; FOLD_AXIS uses `unfold`
with the original window extent and step. Both UNFOLD2D forms use `fold2d` with the exact original
input Shape and window, while FOLD2D uses the conceptual-positive-zero `unfold2d` form. Typed
unfold padding is scalar metadata and receives no cotangent. Dynamic two-dimensional spatial
domains retain separate height and width constraints requiring the padded input or target extent
to contain the effective dilated kernel. An undecidable relation remains an occurrence-owned
deferred graph constraint; a contradicted relation fails inference. This compiler stage neither
binds Dimensions nor chooses materialization.

GATHER, GATHER_ELEMENTS, and GATHER_ND route floating data cotangents through their matching
additive scatter, so repeated indices accumulate. SCATTER_ADD and configurable scatter ADD
preserve the base cotangent and Gather the update cotangent. Scatter NONE replacement zeros the
addressed base positions and Gathers the update cotangent; its forward uniqueness rule remains a
value-dependent execution obligation.

Scatter MUL counts represented zeros and constructs safe per-target products without dividing by
zero. A zero-free update receives the base times every other update; a sole zero receives the base
times all non-zero updates; every update in a multiple-zero group receives exact positive zero.
The base contribution multiplies `g` by the canonical Model product of all updates. NaN, infinity,
signed-zero, overflow, underflow, and rounding therefore follow the ordinary recorded Tensor
formula and Model forward product, except for the explicit positive-zero multiple-zero branch.
Scatter MIN/MAX compare the base and every duplicate update with the canonical forward result and
split `g` equally among numeric-equality winners. Opposite signed zeros tie. A canonical NaN
result matches no candidate, so every candidate receives exact positive zero.

SORT constructs exactly one stable ARGSORT occurrence with the exact original input, normalized
axis, and direction, then replacement-scatters `g` through that permutation. This is the sole
matching recomputation in the current matrix. TOP_K instead uses the exact canonical slot-one
indices from its original producer; `k == 0` routes an input-shaped zero. Equal keys, NaN-last
membership, signed-zero order, and equal-cutoff membership freeze the forward route without
averaging. DROPOUT uses the exact canonical same-occurrence BOOL mask and
`where(mask, g / (1 - probability), zero)`. It neither infers nor resamples a mask and does not
advance or differentiate graph RNG state. At either signed probability zero, the denominator is
positive one and the all-kept mask routes `g`.

The original rules remain unchanged: broadcasted elementwise contributions use `sumToShape`;
ordinary SUM restores removed axes before expansion; CUM_SUM reverses scan direction while
retaining exclusivity; and PERMUTE uses the inverse axis order. WHERE routes no cotangent through
its BOOL condition.

Generated BFLOAT16, FLOAT32, and FLOAT64 exact scalar bases are
provenance-free, storage-free, non-gradient scalar leaves. BFLOAT16 uses exact bits `0x0000` and
`0x3F80`; FLOAT32 and FLOAT64 use exact positive `0.0` and `1.0`. Each base is registered
explicitly as one logical splat, and Shape-specific values are ordinary `expand` expressions.
One request-local cache keys zero, one, extrema/clamp bounds, and other Tensor comparison
operands by exact data type and represented bits, preserves deterministic first-use order, and
creates at most one base for each exact value.
No storage, factory history, label, descriptor, layout, Shape, or provenance absence implies a
constant. Only bases reachable from returned gradient expressions remain in combined-capture
ingress; direct-zero local formulas therefore do not retain an unreachable unit seed.

Arithmetic-only coefficients remain scalar operation metadata. Compiler 0005A and the additive
Compiler 0005D loss rule fix these exact BFLOAT16/FLOAT32/FLOAT64 bit triples:

| Coefficient | BFLOAT16 | FLOAT32 | FLOAT64 |
|---|---:|---:|---:|
| `0.5` | `0x3F00` | `0x3F000000` | `0x3FE0000000000000` |
| `-0.5` | `0xBF00` | `0xBF000000` | `0xBFE0000000000000` |
| `2` | `0x4000` | `0x40000000` | `0x4000000000000000` |
| `-2` | `0xC000` | `0xC0000000` | `0xC000000000000000` |
| `invSqrt2` | `0x3F35` | `0x3F3504F3` | `0x3FE6A09E667F3BCD` |
| `invSqrt2Pi` | `0x3ECC` | `0x3ECC422A` | `0x3FD9884533D43651` |
| `sqrt2OverPi` | `0x3F4C` | `0x3F4C422A` | `0x3FE9884533D43651` |
| `0.044715` | `0x3D37` | `0x3D372713` | `0x3FA6E4E26D4801F7` |
| `0.134145` | `0x3E09` | `0x3E095D4F` | `0x3FC12BA9D1F60179` |

The existing ERF coefficient remains `0x3F90`, `0x3F906EBB`, and
`0x3FF20DD750429B6D`. Compilation derives none of these coefficients with a host transcendental
function.

The compiler captures forward outputs and generated gradient roots together once. The request
includes target-specific roles, the original forward-producer identity set, and merged explicit
splats. Original producer occurrences become `FORWARD`; generated producers become `BACKWARD`.
The graph boundary keeps requested forward values first, then stable first-occurrence-distinct
gradient values. Every target role remains ordered even when multiple targets share one final
gradient `ValueId`.

Authoritative inference and validation follow the combined capture. Canonicalization, the exact
task-0003A rewrites, task-0003B folding, dead-code elimination, and initially phase-local
common-subexpression elimination run once over the complete immutable selected graph under their
existing guards. Every changed candidate is revalidated through the current verification pass.
A later construction, capture, validation, or optimization failure may consume temporary Tensor
IDs; IDs remain opaque and are not rolled back.

Everything outside the table fails closed when it lies on a selected route. The remaining matrix
is intentionally explicit:

| Classification | Current deferred or rejected families |
|---|---|
| Structured fail-closed boundaries | One-output attention lacks canonical same-occurrence weights and is rejected; index-target categorical cross-entropy rejects a dynamic or zero class depth |
| Current source-backed closure | The supported closed inventory contains exactly 128 signature fingerprints; the boundary test adds the three explicitly deferred RNN, GRU, and LSTM signatures and proves that the disjoint 131-signature union equals the complete 38-family, 110-constant Model inventory. Every supported legal output/input role has a `D`, `ND`, or `FC` disposition, and generated formula edges remain inside the same classified Tensor algebra. Every deferred recurrent boundary role fails closed with the unknown/unclassified reason and allocates no Tensor ID. |
| Non-differentiable roles and outputs | Comparisons, BOOL logic/classification, `ALL`, `ANY`, arg-extrema results, batch-training saved auxiliary roots, one-hot and other index roles, ordering indices, dropout masks, padding constants, select coordinates, and graph RNG state |

Unknown/custom kinds, wrong attribute classes or cardinalities, missing canonical outputs, and
descriptor contradictions also fail deterministically. Later formulas must continue using the
shared Tensor algebra; this table does not reserve a gradient-specific numerical or optimization
policy for them.

This strategy adds no `Tensor.gradient`, `Tensor.backward`, mutable gradient field, ThreadLocal
scope, model derivative rule, placeholder/`ValueId` conversion map, direct graph-node formula
language, public gradient registry, runtime tape, physical saved buffer, backend autograd, or
public compile entry point. It adds no publication, optimizer update, training session, planning,
backend lowering, preparation, execution, or runtime behavior. The functional request supports
exactly one optional second reverse-mode stage; further orders remain future work.

### Current package-private artifact compilation

The complete `GraphCompiler.compile(...)` overload has these nine direct inputs:

```text
CompileMode
ordered forward Tensors
optional public FunctionalGradientRequest
explicit logical-splat ingress
GraphOptimizationConfig
BackendIntent
PartitionScoringConfig
ordered BackendCapabilityProvider values
BackendAvailabilitySnapshot values
```

It validates the nine top-level references in declaration order before graph construction, then
invokes the unchanged five-argument graph-stage entry exactly once. Provider and snapshot list
elements are inspected only when a final graph node is planned. A valid zero-node pass-through
graph therefore asks no capability question, accepts unused list elements without inspecting
them, produces no partitions, and still receives one logical-memory requirement per graph value.
Caller collections are neither retained nor mutated.

After graph-stage success, Compiler performs this exact sequence:

```text
final GraphCompilation
  -> PublicationPlan
  -> CompileConstantPlan + CompileDiagnostics
  -> one OperationCapabilityQuery and owner selection per final node
  -> complete Map<NodeId, BackendId>
  -> MaximalSameOwnerPartitioning.partition(...)
  -> LogicalMemoryPlanning.plan(...)
  -> CompileArtifacts
```

`PublicationPlan` is a public final output-only type with package-private construction. It retains
the exact final `CompiledGraphModel` reference and immutable membership snapshots exposed by
`forwardBindings()` and `gradientBindings()`. Forward `ForwardPublicationBinding` values pair
requested Tensor IDs with the final forward graph-output prefix in request order. Gradient
`GradientPublicationBinding` values retain derivative order, stage-local target index, requested
target Tensor ID, and final gradient `valueId()` in target order. Target IDs are unique within
each derivative order. Gradient values may repeat or equal a forward value; the graph boundary
contains the forward prefix followed by each previously unseen gradient value in binding order.

`CompileConstantPlan` is a public final output-only source-role classification with
package-private construction. Its immutable `bindableInputs()` list and immutable
`constantSources()` list classify every final graph input exactly once in graph-input order. A
`ConstantSource(ValueId, ScalarValue)` retains the exact input identity and exact typed scalar
whose bits repeat at every logical coordinate. It is not a dense payload, Tensor, storage object,
materialization instruction, or physical allocation.

`CompileDiagnostics` is a public final output-only successful-compile diagnostic bundle with
package-private construction. It exposes ordered immutable
`DeferredConstraintDiagnostic(NodeId, subject, predicate)` projections. Subject and predicate
text are nonblank and deterministic for diagnostics, but the predicate text is not a public
binding language, trace schema, or serialization format. The exact immutable internal constraints
remain privately retained for later compiler-owned binding validation. A rejected compile returns
no partial artifact and is not converted into a successful diagnostic.

`CompileArtifacts` is the exact eight-component public immutable record:

```java
public record CompileArtifacts(
        CompileMode mode,
        CompiledGraphModel graph,
        List<PlannedPartition> partitions,
        LogicalMemoryPlan memory,
        PublicationPlan publication,
        CompileConstantPlan constants,
        CompileDiagnostics diagnostics,
        DerivativeGraphMetadata derivatives) {}
```

Its canonical constructor validates components in declaration order, snapshots partition-list
membership, retains exact immutable element and plan references, and cross-checks graph identity,
mode/phase/result roles, maximal graph-order partitions, derived logical memory, complete
graph-input source classification, constant type and gradient eligibility, and diagnostic node
membership. The aggregate contains no provider, availability snapshot, selected device, route,
kernel, physical buffer, transfer, schedule, executable, runtime residency, or mutable run state.
It is an immutable recipe for later prepare work, not `GraphCompilation`, a public compile facade,
engine `CompiledGraph`, or `PreparedExecution`.

For each final `CompiledNode`, Compiler resolves ordered input and output descriptors, constructs
one `OperationCapabilityQuery` retaining the exact operation and descriptor references, and calls
`BackendOwnerPlanning.selectOwner(...)` once. Planning validates and evaluates hard eligibility
once, then applies its preferred-class/provider-order baseline once. Compiler retains only the
selected exact `BackendId` in its construction-local owner map; providers and snapshots do not
enter the artifacts. The first no-hard-eligible node fails immediately with graph occurrence
context and retains Planning's terminal failure as its cause. Provider-thrown runtime exceptions
and other Planning composition failures propagate unchanged.

This current boundary adds no public lifecycle call, cost-bearing scoring,
trace event, concrete-dimension binding, prepare/runtime/backend/engine behavior, physical memory,
or more than two reverse-mode stages.

Before a node enters that container, `Operation` validates its exact kind/attributes pairing and
derives a family-owned `OperationSignature`. `CompiledNode` preserves its existing local list
rules and then checks that the final ordered input and output counts lie within that signature's
inclusive bounds. This catches a unary operation connected to two inputs, for example, without
claiming that operand Shapes or data types are compatible. Zero-input, bounded, variadic-input,
and multi-output occurrences remain representable when their kind explicitly declares them.

A `ForwardPublicationBinding` carries only the public Tensor identity and graph-local value
identity for one requested forward result. A `GradientPublicationBinding` instead carries
derivative order, stage-local target index, target Tensor identity, and final gradient
`valueId()`. Current compiler-owned `PublicationPlan` groups both ordered binding lists with the
exact final graph. Neither binding retains a public `Tensor`, publication policy, runtime target,
storage, backend, or execution state.

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
Tensor references and retains the exact canonical Tensor wrapper for that output. Its public
indexed lookup returns that same wrapper without reconstruction. The parameterless `contiguous`
method
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
for maximum, independent of operand order. Equal nonzero candidates produce their numeric value
without a selected-operand promise. Floating ordered comparisons are false if either operand is
NaN and treat negative and positive zero as equal. Floating `EQUAL` is exact represented numeric
equality rather than bit or tolerance equality, and `NOT_EQUAL` is its logical complement.
FLOAT64, FLOAT32, and BFLOAT16 use their represented values under existing promotion. Range
`CLAMP` remains first-class with exactly ordered `MIN(MAX(input, minValue), maxValue)` meaning;
`clampMin` creates one scalar `MAX` producer and `clampMax` one scalar `MIN` producer. Integral
ADD, SUB, and MUL have fixed-width two's-complement modular meaning, and integral MIN, MAX, and
comparisons use signed order. These are current model semantics and metadata-construction facts,
not compiler validation, derivative-policy, lowering, backend, or execution claims. Package-
private structural capture preserves them, and the following package-private verification pass
independently revalidates them. An `OperationSignature`
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
construction is current only for the locally provable floating subset described in the autograd
matrix above; binding-dependent inversion, lowering, backend selection, and execution remain
unimplemented.
`Tensor.matmul` currently constructs one fresh two-input MATMUL expression with a locally derived
vector, matrix, or broadcast-batch Shape and same-category promoted numeric type. Unequal static
contraction dimensions fail locally; unresolved contraction equality and the accepted
unresolved-versus-static batch singleton-or-equal cases remain obligations for later compiler
validation or concrete binding. Package-private structural capture plus graph-wide verification
and conservative constraint proof are current. Current package-private autograd constructs
role-aware floating MATMUL cotangents for all four vector/matrix rank pairings when the selected
operand has the output type; integral and cross-floating selected roles remain unsupported.
Lowering, backend support, concrete binding, and execution remain planned.
`Tensor.linear(weight)` and `Tensor.linear(weight, bias)` are also current model construction, but
they add no LINEAR operation. Conventional `[outFeatures, inFeatures]` weight is explicitly
transposed through PERMUTE `[1, 0]`, followed by MATMUL and optional exact rank-one
`[outFeatures]` ADD bias. Complete caller-controlled validation precedes the first intermediate.
No-bias returns the MATMUL product after two wrapper/ID allocations; biased construction returns
ADD after three. The biased final Shape is structurally equal to the product Shape and reuses its
ordered Dimension references, although the outer Shape object may differ. These visible primitive
producer chains are current inputs to package-private structural capture. This page does not claim
linear-pattern recognition, canonicalization, fusion, lowering, backend support, or execution.
The visible floating PERMUTE/MATMUL/ADD chain is differentiable when every selected primitive row
meets the current closed autograd guards; there is no separate LINEAR rule.
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
decomposition, saved-value lifetime, backend lowering, and execution remain planned. Current
package-private first-order autograd supports both public outputs only for the explicit
two-output occurrence: values slot zero may select query, key, and value, while canonical weights
slot one may select query and key. It reuses the exact same-occurrence weights wrapper, constructs
selection-safe products through `where`, and applies the configured or logical symbolic default
scale. The one-output occurrence fails closed because it has no canonical weights output.
`Tensor.conv2d(weight, attrs)` and `Tensor.conv2d(weight, bias, attrs)` are current first-class
grouped NCHW cross-correlation model construction. Each call records one `CONV2D` occurrence with
exact ordered inputs, intrinsic stride/padding/dilation/group attributes, promoted floating
descriptor, and static or canonical-symbolic output Shape. Input and weight have Shapes
`[N, C_in, H, W]` and `[C_out, C_in/groups, K_h, K_w]`; optional bias has `[C_out]`.
Unresolved channel divisibility, grouped weight/input equality, bias/output equality, and dynamic
spatial non-negativity remain obligations for future compiler validation or concrete binding.
This metadata can be structurally captured and package-private verification now represents and
proves or retains its descriptor-only constraints, but the repository still cannot compile or run
convolution. Current package-private first-order autograd constructs every selected input, weight,
and optional bias cotangent for grouped convolution through exact-group `unfold2d`, matrix
contraction, reduction, and overlap-accumulating `fold2d`. Legal decomposition, saved values,
concrete binding, backend lowering, algorithm selection, and execution remain planned in their
owning layers.
`Tensor.maxPool2d(attrs)` is current first-class NCHW maximum-pooling model construction. One
`MAX_POOL2D` occurrence records exact ordered input `[input]`, `MaxPool2dAttrs`, one output at
index zero, the unchanged floating type and gradient request, exact batch/channel Dimensions, and
static or canonical-symbolic floor/ceil spatial extents. Dynamic spatial non-negativity remains a
future compiler-validation or concrete-binding obligation. Literal ceil mode retains every
ceiling-grid window, even when the terminal window is all-padding; padding exclusion, negative-
infinity empty windows, NaN propagation, signed-zero ordering, and first-logical-sample ties are
semantic metadata. Package-private structural capture and descriptor verification are current,
but the repository does not compile pooling. Current package-private first-order autograd
reconstructs the exact first eligible logical winner from the original input and the
same-occurrence public output, including padding exclusion and the specified NaN and signed-zero
equality, then routes through public one-hot and overlap-accumulating fold expressions. It adds no
saved-index output. Concrete binding, legal decomposition, backend lowering, algorithms, kernels,
and execution remain planned.
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
are current. Current package-private first-order autograd divides by a logical typed
`kernelHeight * kernelWidth` count and routes every window position through public expansion and
overlap-accumulating fold expressions. Concrete binding, any legal decomposition that preserves
the fixed divisor and special-value meaning, backend lowering, algorithms, tolerances, kernels,
and execution remain planned in their owning layers.
`Tensor.sort(axis[, descending])` and `Tensor.argsort(axis[, descending])` currently construct
distinct stable, one-input, one-output ordering expressions. Both normalize the axis, preserve the
exact input Shape reference, leave layout unresolved, and use fixed NaN-last ordering in both
directions with stable logical-index ties. Sort preserves input type and gradient eligibility;
argsort uses non-differentiable INT64. These model-expression and provenance facts are structurally
capturable and package-private operand/descriptor revalidation is current. Floating SORT gradient
construction is also current: the compiler constructs one separate stable `ARGSORT` occurrence
with the exact original input, normalized axis, and direction, then routes the cotangent through
that permutation. Algorithm selection, lowering, backend support, runtime behavior, and execution
remain planned.

Preflight requires the exact SORT input/output descriptors, `SortAttrs`, normalized axis,
direction, and matching one-input/one-output ARGSORT constructibility before any derivative Tensor
is allocated. SORT still has one output: this is not a hidden index output, public sort result,
producer reconstruction, or Model API change. The generated ARGSORT is a BACKWARD occurrence in
the one combined capture.
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
algorithm, lower the operation, report backend support, bind an extent, or execute it. Current
floating TOP_K values-slot autograd uses the exact canonical indices wrapper at producer slot one
and never recomputes selection. Indices remain non-differentiable; stable cutoff membership,
NaN membership, direction, and sorted-output order are routed without selected-set averaging.
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
The current Model also exposes fixed `RNN_TANH`, `GRU_RESET_AFTER`, and `LSTM` recurrent-scan
expressions through the advanced low-level static `RecurrentScan.rnn`, `gru`, and `lstm`
namespace. Each occurrence has one `FORWARD` or `REVERSE` attribute, fully static time-major
descriptors, one ordinary non-gradient `INT64[batch]` valid-length input, and two or three outputs
from one exact flat producer. Generic capture preserves that producer as one `CompiledNode`, with
every declared output position and ordered input edge. This structural fact and the public Model
construction namespace are not Compiler adoption: current `CapturedGraphInference` rejects
`RecurrentScanKind` as an unsupported operation kind before planning or capability selection.
Current autograd preflight likewise rejects each of the exact three deferred recurrent signatures
with the unknown/unclassified reason before constructing any derivative Tensor or allocating an
ID. The inventory boundary names those three signatures explicitly rather than broadly filtering
the family, and proves their disjoint union with the 128 supported signatures is the complete
131-signature Model inventory. Compiler task 0006A owns forward inference, final validation, and
inventory adoption; later explicit work owns backpropagation through time. No current capability
provider, public compiler API, executable, or backend route supports the family.
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
verification now revalidates it and proves or retains deferred equality. Current package-private
first-order autograd supports both prediction and target roles, restores `NONE`, `SUM`, and
`MEAN` cotangents through logical Tensor counts, and uses exact typed scalar-operation
coefficients `2` and `-2`. Legal decomposition, optimization, concrete binding, lowering,
backend support, runtime execution, and training coordination remain planned in their owning
layers.
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
capturable. Current package-private first-order autograd supports both dense floating logits and
target roles. It supports only the logits role for index targets, requires a positive static class
depth, clamps ignored targets before one-hot construction, and excludes ignored rows through a
final `where`; dynamic or zero class depth fails closed. Revalidation, constant analysis, proof,
bounds checks, decomposition, optimization, lowering, preparation, execution, publication, and
training coordination remain planned in their owning lifecycle layers.
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
least input rank and allow new leading target axes. Structural equality and a static source
singleton are immediately compatible. A fully static unequal aligned pair is rejected when the
source is not one; zero is an ordinary static extent. If either aligned dimension is unresolved,
Model construction now retains the exact target and the later condition
`source == 1 || source == target`, leaves layout unresolved, and creates no
`DeferredGraphConstraint`. A fully static target and resolved input layout otherwise produce new
same-offset view geometry with preserved aligned strides and zero strides for leading or
expanded-singleton axes. Every result retains exact type and gradient eligibility, records exact
EXPAND/target-shape semantics with one-input provenance, and remains fresh, unlabeled, and
storage-free.

Package-private compiler inference now adopts that broadened occurrence. For each non-equal,
non-static-source-singleton unresolved aligned pair, it emits the occurrence-owned predicate
`AnyOf(DimensionEqual(source, 1), DimensionEqual(source, target))` in aligned-axis order and
leaves layout unresolved. Floating EXPAND autograd constructs
`gradient.sumToShape(input.descriptor().shape())`; preflight admits the route only when that
inverse uses the same forward predicate relationship. Binding-dependent `SUM_TO_SHAPE` similarly
inverts through `gradient.expand(input.descriptor().shape())` under the matching
target-one-or-target-equal-source obligation. Existing Model reduction semantics are unchanged.
This compiler contract binds no dimension and claims no value repetition, storage aliasing,
materialization, backend lowering, or execution.
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
slice-chain or flip canonicalization, physical aliasing or copying, materialization,
backend/ONNX lowering, and execution remain planned. Closed first-order autograd currently
constructs the normalized floating SLICE cotangent described above; unsupported types and
non-normalized metadata fail during preflight.
`Tensor.sliceByLength(starts, lengths, axes, steps)` is current Model expression construction. It
records the same exact `SLICE`/`SliceAttrs` occurrence after proving non-negative first/final
coordinates and every statically decidable selected-extent upper bound. Selected result
Dimensions are exact static lengths; unaffected Dimension references remain exact. An unresolved
selected extent defers only its upper-bound comparison, and Model stores no constraint object.
Empty or negative-step results remain layout-unresolved; resolved input geometry produces a view
only for a non-empty all-positive request. Current compiler inference derives the exact selected
lengths and proves or retains each non-empty signed region's upper-bound obligation. Fail-closed
preflight validates the exact occurrence before current floating autograd uses that same
length-defined form to invert SLICE_UPDATE without requiring a static selected base extent.
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
Dimension remain deferred. Both primitives are structurally capturable, but neither mutates
values, chooses materialization, lowers, or executes work. Closed first-order autograd currently
constructs guarded floating cotangents for normalized SLICE and for both normalized SLICE_UPDATE
data roles. The update role uses exact recorded lengths, so unresolved selected base extents are
supported when their bounds can be proved or retained. Compiler inference also distinguishes
target-relative extraction from placement, preserves exact Shapes, and retains unresolved bounds.
Concrete binding, saved-value policy, and canonicalization remain planned.
`Tensor.sliceUpdate(update, prefixShape)` is also current Model expression construction. It
records exact `SLICE_UPDATE`/`CropToShapeAttrs` with ordered inputs `[base, update]`, retains the
exact update Shape as the target region and the exact caller prefix Shape, and returns the exact
base Shape/type with eligibility OR and unresolved layout. A fully static axis proves checked
`prefix + update <= base`; if any base, prefix, or update extent is unresolved, Model defers that
whole axis fit without partial arithmetic or a constraint object. Structural capture can preserve
the occurrence. Current compiler inference distinguishes both slice kinds and both exact
attributes variants, retains `prefix + target <= base` obligations, and validates update Shape
against the exact target. Floating autograd uses target-relative placement for the source/base
cotangent and exact target-relative crop for the update cotangent. This adds no execution,
binding, or materialization behavior.
`Tensor.select(int, long)` normalizes one source axis and one scalar coordinate, removes the
selected Dimension, preserves every unaffected exact Dimension reference, and records normalized
`SelectAttrs` with exact one-input provenance. A static selected extent supplies immediate
negative-index normalization and bounds checks; a non-negative coordinate on a dynamic selected
extent remains representable with its upper bound deferred, while a negative coordinate is
rejected. Resolved input geometry with a non-empty result produces checked selected-stride removal
and offset advancement in one new logical view descriptor; unresolved input and empty results stay
unresolved. The fresh result preserves exact type and eligibility and has no label or storage.
Package-private structural capture can preserve this expression. Value selection, physical
aliasing, canonicalization, materialization, backend lowering, and execution remain planned.
Closed first-order autograd currently constructs the guarded floating SELECT cotangent described
above; dynamic-coordinate binding remains outside this binding-free phase.
`Tensor.gather` and `gatherElements` consume exact ordered `[data, indices]` inputs with `INT32` or
`INT64` indices. They normalize one data axis and apply canonical axis-replacement or same-rank
aligned-Shape rules. Every fresh result retains data type and gradient eligibility, leaves layout
unresolved, and records exact two-input provenance. Package-private structural capture can
preserve the occurrence. Construction interprets no index values, checks no index-value bounds,
and adds no Model-owned gradient rule, canonicalization, materialization, lowering, or execution
behavior. Current compiler autograd routes a floating data cotangent through matching
`scatterAdd` or additive `scatterElements`; indices remain non-differentiable.
`Tensor.gatherNd` consumes exact ordered `[data, indices]` inputs with `INT32` or `INT64` indices.
Its short form uses zero shared batch Dimensions; its complete form retains one normalized
non-negative batch count. Construction validates both ranks, structurally equal leading batch
Dimensions, and a statically known positive tuple depth from the final indices Dimension. It
derives the result as the indices prefix without tuple depth followed by the untouched data
suffix, including canonical scalar and exact retained Dimension references. Every result is fresh,
unlabeled, storage-free, and unresolved-layout, preserves data type and gradient eligibility, and
records `GATHER_ND` with exact `[data, indices]` provenance. Package-private structural capture can
preserve the occurrence. Construction reads no index value, checks no index-value bound, and adds
no Model-owned gradient, materialization, lowering, or execution behavior. Current compiler
autograd routes a floating data cotangent through matching additive Scatter-ND; coordinate tuples
remain non-differentiable.
Both `Tensor.scatterElements` overloads consume exact ordered `[data, indices, updates]` inputs.
They require `INT32` or `INT64` indices and exact matching data/update types. `NONE` is permitted
for every current type and arithmetic reductions for floating or integral values. Each method
normalizes one raw data axis, validates the same-rank Shape relationship, and returns a fresh
result with the exact data Shape/type, data/update gradient-
eligibility OR, unresolved layout, and exact three-input provenance. Construction reads no values,
checks no index bound or duplicate target, mutates no input, and performs no write or reduction.
Package-private structural capture can preserve the occurrence. Current compiler autograd
supports both floating data roles for every current reduction: NONE masks addressed base
positions, ADD preserves the base, MUL uses the zero-count/safe-product policy, and MIN/MAX share
among exact numeric-equality winners. Canonicalization, materialization, lowering, backend
behavior, and execution remain planned.
For `MUL`, `MIN`, and `MAX`, the current Model contract combines the base exactly once with every
addressed update exactly once, counts duplicate targets as distinct contributions, and preserves
the exact representation of an unaddressed base coordinate. Its floating and integral result is
independent of encounter, layout, stride, atomic, tree, and backend order. Compiler work must
preserve that represented-value contract; it may not infer a fixed accumulation sequence or add a
derivative or subgradient policy to the Model.
`Tensor.scatterAdd(indices, updates, axis)` is now the current Gather-compatible fixed-add model
primitive. It records `SCATTER_ADD` with normalized `IndexAxisAttrs` and ordered
`[data, indices, updates]` inputs. Updates must have the exact Shape that `data.gather(indices,
axis)` would produce, while the functional result retains the exact data Shape and accumulates
duplicate targets. Construction validates metadata only and leaves layout unresolved.
Package-private structural capture can preserve the occurrence. Current compiler autograd uses
this primitive for floating GATHER data cotangents and preserves its Shape, duplicate
accumulation, and eventual index-bounds obligations. It does not inspect or bounds-check indices,
lower the operation, or execute addition.
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
input, performs no write or reduction, and adds no Model-owned gradient, materialization,
lowering, backend behavior, or execution behavior. Current compiler autograd supports both
floating data roles for NONE, ADD, MUL, MIN, and MAX through exact matching Gather-ND/Scatter-ND
geometry; tuple indices remain non-differentiable.
For arithmetic reduction, every tuple contributes its complete update suffix slice scalar by
scalar. Current Model `MUL`, `MIN`, and `MAX` semantics include each target's base and every
addressed scalar exactly once, count duplicate tuples as distinct contributions, preserve the
exact representation of an unaddressed coordinate, and are independent of encounter, layout,
stride, atomic, tree, and backend order. `NONE`, configurable `ADD`, and fixed `SCATTER_ADD`
remain unchanged. Compiler work must preserve this forward contract without assigning a Model
derivative or subgradient policy.
Static `Tensor.concat(int, Tensor...)` and `Tensor.stack(int, Tensor...)` snapshot ordered non-empty
inputs, normalize an existing or inserted axis, enforce exact type and operation-specific Shape
rules, and create fresh unresolved-layout results with eligibility OR and exact ordered provenance.
Instance `Tensor.unstack(int)` requires a static `int`-sized selected extent and returns an
immutable ordered List of scalar `SELECT` expressions after upfront count validation. Each fresh
result removes that axis, has an independent one-output producer, and uses provenance index zero
over the same input; its `SelectAttrs` stores the coordinate. A zero extent returns no result or
ID. Package-private structural capture preserves each independent occurrence; decomposition,
grouping, materialization, backend lowering, ONNX mapping, and execution remain planned. Closed
first-order autograd currently constructs guarded floating CONCAT and STACK cotangents, including
STACK occurrences produced by repeated SELECT operations during `unstack`.
`Tensor.unfold`, `Tensor.foldAxis`, both `Tensor.unfold2d` forms, and `Tensor.fold2d` construct the
current public storage-free window-transform expressions. General-axis fold restores an explicit
target extent under overlap summation. The 2D forms preserve canonical rank-three im2col/col2im,
retain exact static or symbolic channel/spatial formulas, and distinguish direct conceptual-zero
padding from an exact typed padding scalar. `fold2d` accepts only complete structural formula
matches rather than recording equality between unrelated unresolved symbols. Every result
preserves input data type and gradient eligibility, leaves layout unresolved, and records exact
one-input provenance. Package-private structural capture can preserve these occurrences. Current
compiler inference retains unresolved two-dimensional height/width domain constraints, and
current floating autograd uses each exact public inverse or overlap-add transformation. The Model
still owns no gradient rule, canonicalization, lowering, or execution.
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
policy and physical saved-value lifetime remain planned. Current floating values-slot autograd
retrieves the exact canonical mask wrapper at producer slot one and constructs
`where(mask, g / (1 - probability), zero)`. It never resamples, infers a mask from values, or
advances/differentiates graph RNG state.

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
reference for Planning. Neither construction evaluates availability or capability, ranks
candidates, chooses ownership, discovers a service, or invokes a compiler. In particular,
`unconstrained()` does not mean “automatic backend selection succeeded”; it records only the
absence of a hard target.

The canonical constructor rejects a null optional with `NullPointerException("hardRequirement")`.
The `requiring` factory rejects a null requirement with `NullPointerException("requirement")`.
Current internal planning evaluates the hard target before its cost-free baseline owner selection,
and the package-private complete compiler entry supplies this intent once per final graph node.
General score calculation, planning cost profiles, `CompileConfig`, and a public compile failure
remain planned. The separate current `PartitionScoringConfig` value
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
Package-private `GraphCompiler` currently interprets it as a forward-only graph-stage or complete
artifact request and requires the functional request to be absent. `FORWARD_AND_BACKWARD`
requests current package-private one/two-stage pre-capture autograd plus combined
forward/backward graph-stage work.
`TRAINING_STEP` currently performs the same internal graph and artifact construction, but the enum value
does not itself introduce an optimizer, optimizer-update graph, session, publication delivery,
schedule, or execution behavior. Public compiler consumption of all three values remains planned.

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
The package-private complete compiler entry supplies this value to the current per-occurrence
Planning collaboration. Backend-neutral cost classification, planning cost inputs, public numeric
scoring evaluation, and the aggregate remain planned.

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

The provider is supplied explicitly to compile-time planning. The public contract performs no
registry, discovery, classpath scan, `ServiceLoader` lookup, availability or hard-requirement
evaluation, scoring, route or kernel selection, preparation, or execution. The repository
supplies no production provider implementation. The current package-private compiler artifact
entry is the concrete consumer: it supplies providers once per final graph node and retains only a
selected `BackendId`, not a provider object.

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
Public `BackendOwnerPlanning.selectOwner(...)` validates its five top-level references in
declaration order and composes these two internal steps exactly once for one occurrence. The
eligibility result remains package-private and does not escape.

Reusable or public capability matrices, a public eligibility result/evaluator, public graph-wide
planning orchestration, numeric or cost scoring and profiles, device-level capability or
selection, preparation, runtime, execution, and typed trace/rejection schemas remain planned.
Compiler currently owns owner-map assembly and contextualizes only the terminal no-hard-eligible
failure.

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

Public stateless `MaximalSameOwnerPartitioning.partition(...)` accepts one immutable
`CompiledGraphModel` and one complete `Map<NodeId, BackendId>`. Equal map keys associate with graph
nodes, but output membership always uses the exact `NodeId` references from `graph.nodes()`. It
validates all keys, graph coverage, and owners before constructing a result. For each maximal run
it retains the exact owner reference mapped to the first node and compares later owners with
`BackendId.equals`.

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

Compiler currently assembles the complete owner map from per-occurrence results and invokes this
operation directly. Public visibility serves that concrete cross-package consumer; it does not
create a graph-wide Planning facade or public compile workflow. A partition is an immutable
ownership recipe, not a promise of one fused kernel, one executable, or any prepare/runtime
behavior.

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

Public stateless `LogicalMemoryPlanning.plan(...)` accepts one `CompiledGraphModel` and one
ordered `List<PlannedPartition>`. It completes all partition null, graph-membership, coverage,
graph-order, and adjacent-owner maximality checks before constructing a requirement. It then emits
exactly one requirement for every `graph.values()` entry in that encounter order. Producer facts
come from node outputs; consumer facts come from node inputs. Repeated inputs, fan-out, merges,
unused graph inputs, unused produced values, multi-output nodes, forward/backward uses, and
zero-node pass-through graphs require no synthetic node or role.

The primitive facts describe overlapping logical roles: graph input, partition input, partition
output, same-owner or cross-owner boundary, graph-output preservation, and partition-internal
value. The records do not store another role enum. They retain `TensorDescriptor` so static,
dynamic, and expression Shapes remain representable without calculating an element or byte
count.

This is logical planning only. The derivation accepts no `ForwardPublicationBinding`,
`GradientPublicationBinding`, or `TensorId` and creates no `PublicationPlan`. It selects no alias,
copy, transfer, lifetime, physical slot,
allocation, device, route, kernel, schedule, executable, or runtime residency. The current
package-private compiler invokes it after partition generation and retains the exact plan in
`CompileArtifacts`.

## Current expression input and compile output

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
  signed-step slice plus one-occurrence flip construction, finite length-defined extraction across
  unresolved selected extents, functional signed slice update, exact target-relative symbolic
  placement, and exact target/prefix-Shape crop construction, plus conditional-view scalar-select
  construction and
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
  saved-statistic construction and gradient construction outside the closed support table above,
  optional
  softmax, layer-normalization, RMS-normalization, attention, or activation decomposition,
  redundant-cast, redundant-contiguous, reshape/expand/permutation/rank-edit, slice chains, slice
  updates, crops, and selects beyond the current exact inference/preflight/gradient matrix,
  axis-gather/one-hot/Gather-ND/axis-scatter/Scatter-ND/pad/tile/composition/window-transform
  canonicalization or decomposition,
  concrete dynamic binding and value-dependent select/index validation,
  legal
  convolution/pooling decomposition, maximum-pooling saved-index
  policy, average-pooling fixed-divisor preservation, layout materialization
  planning remain planned. Package-private identity-based forward-only and phase-aware combined
  capture, closed first-order autograd, binding-free verification inference, mandatory dense
  graph-local canonicalization, the guarded seven-rule exact arithmetic scan, the bounded
  logical-splat fold matrix, and exact one-shot whole-graph DCE/phase-local-CSE/DCE cleanup are
  current internal behavior.
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
- `PublicationPlan` is current compiler-owned output-only context around ordered forward and
  gradient publication bindings and the exact final graph. It is separate from the model graph
  and adds no publication delivery policy or runtime behavior.
- `CompileConstantPlan` is current compiler-owned output-only classification of final graph inputs
  as caller-bindable or exact logical splats.
- `CompileDiagnostics` is current compiler-owned output-only projection of successful deferred
  graph constraints without exposing the internal predicate vocabulary.
- `CompileArtifacts` currently combines mode, the exact final `CompiledGraphModel`, maximal
  backend-owned partitions, the derived logical memory plan, publication roles, constant/input
  roles, diagnostics, and derivative-order metadata. Its public record and nested output data are
  implemented, while the only complete construction entry remains package-private.
- `CompiledGraph` will be an engine facade over immutable `CompileArtifacts`, not the same object
  as the current `CompiledGraphModel`.

The planned artifacts deliberately contain no device buffers, backend executable objects, runtime
residency, prepared schedules, or mutable run state.

## Current internal lifecycle and planned public failures

```text
forward Tensor outputs
  -> if the internal mode requests first-order gradients:
     preflight -> Tensor formulas -> combined expression
  -> one phase-aware capture
  -> inference and validation
  -> canonicalization and one-shot exact whole-graph optimization
  -> publication, backend ownership, partitions, and logical memory
  -> CompileArtifacts
```

The current internal entry rejects invalid shapes, data types, operations, graph structure,
publication roles, planning composition, unsatisfied hard capability, partitions, memory, source
roles, or diagnostics before returning an artifact. A no-hard-eligible occurrence becomes a
compiler-owned `IllegalStateException` with node index, ID, and operation-kind context, while the
Planning failure remains its cause. Provider runtime failures otherwise propagate unchanged.
There is still no public callable compiler failure contract; exact public request signatures and
exception taxonomy remain for compiler and engine tasks.

## Example interpretation

If a future graph contains a matrix multiplication followed by a small elementwise operation,
capability analysis may find both CPU and Metal valid. Backend-neutral scoring may assign both
nodes to Metal to avoid a transfer boundary. The artifact records only `owner = Metal`; it does
not record MPSGraph or a custom Metal kernel. Metal prepare makes that later choice.

This scenario remains conceptual because no production provider or public compile entry exists.
Current package-private `GraphCompiler` can construct and validate its bounded forward-only or
combined first-order graph, query explicitly supplied test or future backend providers, select
backend ownership, derive partitions and logical memory, and return `CompileArtifacts`. It cannot
deliver publications, bind concrete dimensions, prepare work, execute a backend, or expose a
public lifecycle result.

## Related contracts

- [Current Tensor and graph-model API](tensor-api.md)
- [Lifecycle](../architecture/lifecycle.md)
- [Partition scoring](../architecture/partition-scoring.md)
- [Compiling graphs user guide](../user-guide/compiling-graphs.md)
- [Roadmap](../planning/roadmap.md)
