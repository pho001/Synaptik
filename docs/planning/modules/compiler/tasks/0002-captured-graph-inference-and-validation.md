# Task 0002: Captured-[Graph](../../../../glossary.md#graph) [Inference](../../../../glossary.md#term-inference) and [Validation](../../../../glossary.md#term-validation)

## Status

Complete

## Goal

Add the [compiler](../../../../glossary.md#term-compiler)'s first semantic [graph](../../../../glossary.md#graph) pass after package-private [structural capture](../../../../glossary.md#term-structural-capture). The pass
must independently derive the [descriptor](../../../../glossary.md#term-descriptor) facts implied by every current production [operation](../../../../glossary.md#operation)
[occurrence](../../../../glossary.md#term-occurrence), reject incompatible operands or stored output descriptors, and turn each current
[shape](../../../../glossary.md#shape) obligation that cannot be decided locally into a typed package-private [deferred graph
constraint](../../../../glossary.md#deferred-graph-constraint).

The pass is verification [inference](../../../../glossary.md#term-inference), not graph rewriting:

```text
captured CompiledGraphModel
  -> resolve ordered input/output descriptors
  -> infer the descriptor each operation occurrence requires
  -> compare inferred and stored descriptors
  -> prove, reject, or retain each symbolic obligation
  -> ValidatedGraph(exact same graph, unresolved constraints)
```

Task 0001 already guarantees [immutable](../../../../glossary.md#term-immutable) structural closure and [topological order](../../../../glossary.md#term-topological-order). This task must
not duplicate [capture](../../../../glossary.md#term-capture-meanings) or silently trust [model](../../../../glossary.md#term-model)-time [Tensor](../../../../glossary.md#tensor) construction. It validates graph model
data through `Operation`, attributes, ordered input descriptors, and ordered output descriptors
only; it must not retain or reconstruct `Tensor`, [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), or [provenance](../../../../glossary.md#provenance) state.

## [Scope](../../../../glossary.md#term-scope)

- Add one package-private, stateless [compiler](../../../../glossary.md#term-compiler) entry point with this exact [shape](../../../../glossary.md#shape):

  ```java
  static ValidatedGraph inferAndValidate(CompiledGraphModel graph)
  ```

- Return the exact input graph reference on success together with an immutable deterministic list
  of only the constraints that remain unresolved after compile-time proof.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [immutable glossary](../../../../glossary.md#term-immutable); [compile glossary](../../../../glossary.md#compile))</span>
- Revalidate every current production operation kind and accepted attributes variant. No current
  family may bypass inference because its Tensor construction already checked the request.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Independently derive and compare complete output `TensorDescriptor` values, including data
  type, Shape, resolved-or-unresolved layout, and gradient-eligibility metadata.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Validate ordered operand roles, data-type domains and promotion, ranks, axes, Shape
  relationships, attribute-to-descriptor relationships, output count/roles, and multi-output
  descriptor relationships beyond the occurrence-count checks already owned by<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
  `OperationSignature` and `CompiledNode`.
- Represent the current [dimension](../../../../glossary.md#dimension)/Shape obligations listed in
  [Deferred constraint coverage](#deferred-constraint-coverage) through a small typed internal
  predicate model. Prove or disprove them only from current immutable model facts.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Use static extents, structural equality of named and canonical expression dimensions, and
  conservative bounds already carried by constrained unknown expressions. Do not bind a symbol,
  guess a value, or introduce an algebra system.<span data-glossary-navigation="true"> ([structural equality glossary](../../../../glossary.md#term-structural-equality); [canonical glossary](../../../../glossary.md#term-canonical-form); [symbol glossary](../../../../glossary.md#term-symbol))</span>
- Reject a statically disproven obligation. Omit a proven obligation from the result. Retain an
  undecidable obligation once, in deterministic node/rule/axis order, with its owning [`NodeId`](../../../../glossary.md#nodeid)
  and a concise semantic subject.
- [Fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) for an unrecognized [`OperationKind`](../../../../glossary.md#operationkind) implementation or a current production
  kind/attributes variant that has no compiler inference rule.
- Keep [validation](../../../../glossary.md#term-validation) phase-neutral: the [descriptor](../../../../glossary.md#term-descriptor) contract of an [operation](../../../../glossary.md#operation) is the same for a
  `FORWARD` or future `BACKWARD` [node](../../../../glossary.md#node). This task does not create or reinterpret backward work.
- Add focused compiler tests covering every current operation family and attributes variant,
  descriptor mismatch category, constraint proof outcome, failure-order contract, immutability,
  deterministic ordering, unsupported kinds, and valid captured multi-output/state graphs.
- Update the Compile API and targeted Tensor API status statements after implementation. Add the
  reusable deferred-graph-constraint distinction to the glossary.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Complete the required separate documentation-focused review in the same overall change.

### [Validation](../../../../glossary.md#term-validation) boundary after [capture](../../../../glossary.md#term-capture-meanings)

[`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) construction and `GraphCapture` already prove structure: value and [node](../../../../glossary.md#node) ID
uniqueness, resolvable references, [producer](../../../../glossary.md#term-producer) closure, [topological](../../../../glossary.md#term-topological-order) availability, [graph](../../../../glossary.md#graph) boundaries,
phase coverage, deterministic capture order, and local [operation](../../../../glossary.md#operation) [occurrence](../../../../glossary.md#term-occurrence) [cardinality](../../../../glossary.md#term-cardinality). This
task consumes those invariants and must not add a second structural graph validator.

The new pass proves semantic agreement:

- each ordered input [descriptor](../../../../glossary.md#term-descriptor) is legal for its operation role;
- the attributes are consistent with the input ranks and [Shapes](../../../../glossary.md#shape);
- every stored output descriptor equals the independently inferred descriptor for that output
  position; and
- every [compile](../../../../glossary.md#compile)-visible symbolic obligation is proven, rejected, or retained explicitly.

The pass builds only construction-local [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) [lookup](../../../../glossary.md#term-lookup) data. It stores no producer/use index,
mutates no graph collection, allocates no new graph ID, and changes no descriptor reference.

### Required [operation](../../../../glossary.md#operation)-family coverage

The implementation may group closely related rules into cohesive package-private classes, but it
must cover the complete production inventory below. Family [dispatch](../../../../glossary.md#term-dispatch) uses typed `instanceof` or
enum switches; do not use class names, reflection, annotations, a service loader, or a [mutable](../../../../glossary.md#term-mutable)
[registry](../../../../glossary.md#term-registry).

| Group | Required [inference](../../../../glossary.md#term-inference) and revalidation |
|---|---|
| [Binary](../../../../glossary.md#term-binary-descriptor-and-metadata), [scalar](../../../../glossary.md#term-scalar), [unary](../../../../glossary.md#term-unary), comparison, logical, classification, selection, and cast | Exact operand domains; same-category promotion; integral-operation subset; scalar attribute/data-type equality; ordered [broadcasting](../../../../glossary.md#broadcasting); fixed [BOOL](../../../../glossary.md#data-type--datatype) results; [Shape](../../../../glossary.md#shape) retention; cast [target](../../../../glossary.md#term-target) and [gradient](../../../../glossary.md#term-gradient) policy; unresolved [layout](../../../../glossary.md#layout) and gradient propagation. |
| Ordinary, multi-[axis](../../../../glossary.md#term-axis), masked, statistical, arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum), and [sum-to-Shape](../../../../glossary.md#sum-to-shape) reductions | Exact kind/attributes variants; numeric/BOOL domains; normalized distinct axes; [mask](../../../../glossary.md#term-boolean-mask) type and broadcast-to-input rule; reduced Shape; retained type or fixed [INT64](../../../../glossary.md#data-type--datatype); correction/domain constraint; non-empty selected arg-extrema [extent](../../../../glossary.md#term-extent); target-one-or-input-equal constraints; unresolved layout and gradient policy. |
| [Cumulative scan](../../../../glossary.md#cumulative-scan), [softmax](../../../../glossary.md#softmax--log-softmax), [layer normalization](../../../../glossary.md#layer-normalization), [RMS normalization](../../../../glossary.md#rms-normalization), and [batch normalization](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training) | Floating/integral family domains as applicable; normalized axes and trailing Shapes; exact [epsilon](../../../../glossary.md#epsilon)/scalar typing; affine/[statistic](../../../../glossary.md#term-statistic) [vector](../../../../glossary.md#term-vector) roles; all one- and five-output [descriptor](../../../../glossary.md#term-descriptor) relationships; [channel](../../../../glossary.md#term-channel)/trailing equality and [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-statistic constraints; unresolved layout and exact gradient propagation. |
| Sort, argsort, and top-K | Accepted input domains; [normalized axis](../../../../glossary.md#normalized-axis); Shape retention or selected-static-`k` replacement; values/indices output order and types; selected-extent-at-least-`k` constraint; layout and gradient policy. |
| [Scalar select](../../../../glossary.md#scalar-select), [Gather](../../../../glossary.md#gather), [Gather Elements](../../../../glossary.md#gather-elements), one-hot, [Gather-ND](../../../../glossary.md#gather-nd), [Scatter Elements](../../../../glossary.md#scatter-elements), Gather-compatible [Scatter Add](../../../../glossary.md#scatter-add), and [Scatter-ND](../../../../glossary.md#scatter-nd) | Index [data type](../../../../glossary.md#data-type--datatype); normalized axes; [tuple depth](../../../../glossary.md#term-tuple-depth) and batch prefix; exact data/update types; family-specific Shape formulas; fixed BOOL one-hot result; functional data-shaped scatter results; layout and gradient policy. |
| [Contiguous](../../../../glossary.md#term-contiguous), reshape, expand, permute, expand-[dimensions](../../../../glossary.md#dimension), squeeze, [slice](../../../../glossary.md#slice), [slice update](../../../../glossary.md#slice-update), [target-relative crop](../../../../glossary.md#target-relative-crop), pad, tile, concat, stack, unfold/fold axis, and unfold/fold 2D | [Rank](../../../../glossary.md#term-rank)/axis/attribute relationships; exact Shape arithmetic; [element](../../../../glossary.md#term-element)-count, [singleton](../../../../glossary.md#term-singleton)/expansion, slice-region, crop-region, and window constraints; exact resolved-layout derivation where the current [model](../../../../glossary.md#term-model) contract produces geometry; otherwise unresolved layout; type and gradient retention. |
| [Matrix multiplication](../../../../glossary.md#matrix-multiplication--matmul) and [scaled dot-product attention](../../../../glossary.md#scaled-dot-product-attention) | Numeric/floating promotion; ranks and ordered roles; [contraction](../../../../glossary.md#contraction-dimension)/[embedding](../../../../glossary.md#embedding)/sequence equality; singleton-or-equal batch and mask broadcast obligations; optional mask and scale rules; exact output/[weights](../../../../glossary.md#term-weight) descriptor order and gradient policy. |
| Grouped [NCHW](../../../../glossary.md#term-nchw) [convolution](../../../../glossary.md#term-channels-first-convolution-layers) and NCHW maximum/average [pooling](../../../../glossary.md#term-pooling) | Floating promotion/domain; rank and optional [bias](../../../../glossary.md#term-bias) roles; grouped channel divisibility/equality; exact [spatial](../../../../glossary.md#term-spatial-axes) symbolic formulas; non-negative numerator obligations; exact result Shape/type/layout/gradient [metadata](../../../../glossary.md#term-metadata). |
| [Mean-squared error](../../../../glossary.md#mean-squared-error--mse) and dense/index [categorical cross entropy](../../../../glossary.md#categorical-cross-entropy) with logits | Floating promotion; index versus [dense target](../../../../glossary.md#dense-target) domains; class-axis and reduction Shape rules; mapped descriptor equalities and [compile](../../../../glossary.md#compile)-visible extent constraints; exact result type/layout/gradient policy. |
| Initial [graph RNG state](../../../../glossary.md#graph-rng-state--graphrngstate) and [dropout](../../../../glossary.md#dropout--inverted-dropout) | Exact zero-input state descriptor; exact input/state roles; probability/scalar typing; ordered public output, BOOL mask, and next-state descriptors; unresolved layouts and gradient policy. |

The implementation must derive these facts directly from descriptors, typed attributes, and
public model Shape/layout operations. It must not create temporary [Tensor](../../../../glossary.md#tensor) expressions as an
inference technique because that would consume public Tensor [identities](../../../../glossary.md#term-identity-meanings), create new [provenance](../../../../glossary.md#provenance),
and make [compiler](../../../../glossary.md#term-compiler) [validation](../../../../glossary.md#term-validation) depend on mutable [API](../../../../glossary.md#term-application-programming-interface-api) state.

Deferral must not broaden a current model contract. A relation becomes a deferred constraint only
where the current Tensor construction contract deliberately permits that unresolved relation
while still selecting one exact output descriptor. Current contracts that reject an unprovable
relation because no exact output Dimension can be selected remain rejecting compiler rules. This
includes general `ShapeBroadcast` pairs, directional expand pairs, composition equalities, and
unequal unresolved [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) or attention batch candidates for which the model cannot choose one
result Dimension. By contrast, the explicitly retained unresolved/static MATMUL and attention
broadcast cases, output-descriptor-independent mask alternatives, and the other obligations below
are deferred because their exact result descriptor is already determined.

### Deferred constraint coverage

Create one small package-private typed predicate vocabulary sufficient for current [compile](../../../../glossary.md#compile)-visible
[Shape](../../../../glossary.md#shape) obligations. It must support exactly these [atomic](../../../../glossary.md#resource-transaction-and-rollback) relations plus typed conjunction and
disjunction:

- two [Dimensions](../../../../glossary.md#dimension) are equal;
- one Dimension is at least a non-negative [constant](../../../../glossary.md#term-constant);
- one Dimension is divisible by a positive constant;
- two Shapes have equal logical [element](../../../../glossary.md#term-element) counts;
- one Shape has a logical element count equal to, or at least, a non-negative constant; and
- a non-negative start plus a Shape/Dimension [extent](../../../../glossary.md#term-extent) fits within a containing Dimension.

This vocabulary must express the current obligations, including:

- reshape element-count equality;
- [sum-to-Shape](../../../../glossary.md#sum-to-shape) [target](../../../../glossary.md#term-target)-one-or-input-equal alternatives;
- non-empty arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum) and top-K selected extents;
- statistical reduction domain count greater than correction;
- [scalar](../../../../glossary.md#term-scalar)-select and [slice](../../../../glossary.md#slice)/crop/update upper bounds when they depend on an unresolved extent;
- [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) [contraction](../../../../glossary.md#contraction-dimension) equality and unresolved-versus-static [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [singleton](../../../../glossary.md#term-singleton)-or-equal rules;
- attention embedding/sequence equality, batch singleton-or-equal, mask singleton-or-equal, and
  positive embedding requirements;<span data-glossary-navigation="true"> ([embedding glossary](../../../../glossary.md#embedding); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- layer/RMS/batch-normalization extent equalities and the documented empty-channel or batch-count
  alternative for batch-normalization training;
- [convolution](../../../../glossary.md#term-channels-first-convolution-layers) group divisibility, [channel](../../../../glossary.md#term-channel)/[bias](../../../../glossary.md#term-bias) equalities, and [spatial](../../../../glossary.md#term-spatial-axes) numerator non-negativity;
- [pooling](../../../../glossary.md#term-pooling) and window-transform numerator non-negativity and column/target compatibility; and
- topologically encountered loss Shape/extent relations that depend only on descriptors and
  typed attributes.<span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss))</span>

The proof evaluator is deliberately conservative:

- equal static values and structurally equal Dimensions/Shapes prove equality;
- unequal static values disprove equality;
- fully static element counts decide count predicates with exact non-negative integer
  multiplication and never wrap, including when the mathematical product exceeds `long`;
- recursive interval inspection may use only static values, the non-negative baseline of a named
  dynamic Dimension, constrained-unknown minimum and optional maximum, and the current positive-
  coefficient linear, product, floor-division, and ceiling-division expression forms; linear
  signed offsets are preserved and derived bounds must not be clamped to zero, while product or
  division bounds are unavailable unless their operand bounds already prove the required
  non-negative domain;<span data-glossary-navigation="true"> ([linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- interval bounds use checked `long` arithmetic. Overflow, an absent upper bound, or a bound that
  depends on an unavailable opposing value makes that proof unavailable rather than false;<span data-glossary-navigation="true"> ([Overflow glossary](../../../../glossary.md#term-overflow))</span>
- a Dimension equality may additionally be disproven by non-overlapping known intervals, but
  different symbols or structurally different expressions with overlapping intervals remain
  deferred;
- Dimension divisibility is decided only for a static value or divisor one; this task does not
  derive modular facts from symbolic coefficients or products;
- Shape element-count equality is proven by structural Shape equality or decided when both counts
  are fully static; otherwise conservative count intervals may only disprove equality when they
  do not overlap. Count-equal-to-constant and count-at-least predicates may use the same checked
  interval product, with an unavailable or overflowing non-static interval left deferred;
- fit-within is decided from static values or the conservative interval facts above and otherwise
  remains deferred;
- `allOf` and `anyOf` use ordinary three-valued `PROVEN`, `DISPROVEN`, `DEFERRED` composition; and
- no cancellation, substitution, factorization, affine comparison, symbol unification, or
  equivalence beyond existing model-value equality is permitted.<span data-glossary-navigation="true"> ([cancellation glossary](../../../../glossary.md#term-cancellation); [model glossary](../../../../glossary.md#term-model))</span>

Different unresolved symbols are not equal merely because one [operation](../../../../glossary.md#operation) requires them to be.
The requirement itself is retained as a constraint. No [binding](../../../../glossary.md#term-binding) map, [symbol](../../../../glossary.md#term-symbol) assignment, union-find
solver, substitution, [serialization](../../../../glossary.md#term-serialization), [cache](../../../../glossary.md#term-cache), or public [evaluation](../../../../glossary.md#term-evaluation) [API](../../../../glossary.md#term-application-programming-interface-api) is added.

Value-dependent obligations are not [graph](../../../../glossary.md#graph) [descriptor](../../../../glossary.md#term-descriptor) [inference](../../../../glossary.md#term-inference). Index values and bounds,
duplicate scatter targets, dense categorical target [normalization](../../../../glossary.md#term-normalization), ignore-index contents, [host
storage](../../../../glossary.md#host-storage) contents, [NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity) behavior, and numerical result [validation](../../../../glossary.md#term-validation) cannot be proved from
[`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) because graph values retain descriptors rather than element data. This task
must not pretend to represent or discharge them. They remain explicit future constant-analysis,
binding, preparation, or execution validation work in the [lifecycle](../../../../glossary.md#lifecycle) [owner](../../../../glossary.md#term-owner) that has the required
values.

### [Validation](../../../../glossary.md#term-validation) and failure order

The entry point validates in this deterministic order:

1. reject a null [graph](../../../../glossary.md#graph) with `NullPointerException("graph")`;
2. construct a temporary value [lookup](../../../../glossary.md#term-lookup) from the already validated `graph.values()` order;
3. visit [nodes](../../../../glossary.md#node) in stored [topological order](../../../../glossary.md#term-topological-order);
4. resolve that node's input and output [descriptors](../../../../glossary.md#term-descriptor) in position order;
5. select the typed family rule and validate operands/attributes in documented role order;
6. derive the complete expected output descriptor list;
7. compare expected and stored descriptors in output-position order; and
8. evaluate generated predicates in rule then axis order, rejecting the first disproven predicate
   and appending each deferred predicate.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>

An unsupported kind failure must identify `nodes[index]`, its [`NodeId`](../../../../glossary.md#nodeid), the concrete kind type,
and `kind.name()`. Every semantic failure must be prefixed with the zero-based node position,
`NodeId`, and typed [operation](../../../../glossary.md#operation)-kind diagnostic. A descriptor mismatch must additionally identify
the zero-based output position, [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), expected descriptor, and stored descriptor. A
disproven constraint must identify its semantic subject and typed predicate. Exact family-detail
wording is not a [serialization](../../../../glossary.md#term-serialization) or public diagnostic [schema](../../../../glossary.md#term-schema), but the common context is exact:

```text
nodes[<index>] NodeId[value=<id>] <kind-class-name>.<kind.name()>: <detail>
```

Unsupported [dispatch](../../../../glossary.md#term-dispatch) uses detail `unsupported operation kind`. Descriptor mismatch detail begins
`output[<position>] ValueId[value=<id>] expected=<descriptor>, stored=<descriptor>`. A disproven
predicate begins `constraint <subject> failed: <predicate>`. Checked descriptor-derivation
arithmetic uses detail `descriptor derivation failed: <cause-message>` and retains the arithmetic
exception as its cause. Family-specific detail after the common prefix may remain concise and is
not a serialization or public diagnostics contract. Tests must lock this prefix, the specialized
detail above, and the ordering needed to distinguish the first failure.

All unsupported-kind, operand, attribute-to-descriptor, output-descriptor, and disproven-
constraint failures throw `IllegalArgumentException`. Checked arithmetic failure while deriving
an expected descriptor is wrapped as an `IllegalArgumentException` with the same node context and
the arithmetic failure as its cause. This internal exception behavior is not a new public [compile](../../../../glossary.md#compile)
exception taxonomy.

The result preserves one deferred constraint per owning [occurrence](../../../../glossary.md#term-occurrence) and semantic role. Do not
deduplicate equal-looking constraints from distinct nodes, because occurrence context is needed
for later diagnostics and graph transformation.

### Relationship to the architecture [compile](../../../../glossary.md#compile) order

This task supplies a reusable correctness pass and applies it to raw captured [graphs](../../../../glossary.md#graph) as a safe
[compiler](../../../../glossary.md#term-compiler) ingress boundary. It does not claim that raw-[capture](../../../../glossary.md#term-capture-meanings) [validation](../../../../glossary.md#term-validation) replaces the
architecture's [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form)-then-[inference](../../../../glossary.md#term-inference)-and-validation sequence.

Task 0003 must consume only a successful `ValidatedGraph`, perform its separately specified
canonicalization/forward transformations, and invoke this same inference/validation [operation](../../../../glossary.md#operation) on
every resulting graph candidate before that candidate can advance. Thus the eventual transformed
pipeline still performs canonicalization before the authoritative inference/validation of the
transformed graph, while malformed captured [metadata](../../../../glossary.md#term-metadata) cannot enter transformation code unchecked.

## Out of [scope](../../../../glossary.md#term-scope)

- [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction) changes, public [capture](../../../../glossary.md#term-capture-meanings) access, a public [compiler](../../../../glossary.md#term-compiler) [facade](../../../../glossary.md#term-facade), `GraphCompiler`,
  `CompiledGraph`, [engine](../../../../glossary.md#term-engine) orchestration, or a [compile](../../../../glossary.md#compile) configuration aggregate
- changing, replacing, normalizing, canonicalizing, or copying the accepted graph, nodes, values,
  IDs, Operations, attributes, descriptors, layouts, or phases<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node); [descriptors glossary](../../../../glossary.md#term-descriptor); [layouts glossary](../../../../glossary.md#layout))</span>
- persistent [producer](../../../../glossary.md#term-producer)/use indexes or a new [graph](../../../../glossary.md#graph) container component
- dead-code elimination, common-subexpression elimination, constant folding, algebraic
  simplification, same-type cast removal, view folding, layout materialization, graph candidate
  generation, optional optimization, or any task-0003 transformation<span data-glossary-navigation="true"> ([dead-code elimination glossary](../../../../glossary.md#term-dead-code-elimination-dce); [common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [constant folding glossary](../../../../glossary.md#term-constant-folding); [view glossary](../../../../glossary.md#term-view-meanings); [materialization glossary](../../../../glossary.md#materialization))</span>
- autograd, adjoint rules, saved-value selection, gradient accumulation, backward graph
  construction, phase assignment, compile-mode interpretation, or task 0004<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [adjoint glossary](../../../../glossary.md#term-adjoint); [gradient accumulation glossary](../../../../glossary.md#term-gradient-accumulation); [backward graph glossary](../../../../glossary.md#backward-graph))</span>
- publication bindings/plans, backend intent, capability queries, owner selection, partitions,
  logical memory orchestration, diagnostics aggregation, trace emission/payloads, or<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [bindings glossary](../../../../glossary.md#term-binding); [backend intent glossary](../../../../glossary.md#backend-intent--backendintent); [owner glossary](../../../../glossary.md#term-owner); [trace glossary](../../../../glossary.md#trace))</span>
  `CompileArtifacts`
- public or runtime dimension binding, concrete input-Shape binding, symbol substitution,
  constraint serialization, a general constraint solver, or embedding unresolved constraints in
  a public artifact before its consumer is designed<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [dimension glossary](../../../../glossary.md#dimension); [Shape glossary](../../../../glossary.md#shape); [symbol glossary](../../../../glossary.md#term-symbol); [serialization glossary](../../../../glossary.md#term-serialization); [embedding glossary](../../../../glossary.md#embedding); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- inspection of Tensor labels, host storage, element values, constant propagation, index values,
  duplicate targets, numerical results, or runtime failure policy<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [host storage glossary](../../../../glossary.md#host-storage); [element glossary](../../../../glossary.md#term-element); [constant glossary](../../../../glossary.md#term-constant); [targets glossary](../../../../glossary.md#term-target))</span>
- physical buffers, materialization decisions, preparation, backend lowering, kernel/route
  selection, schedules, executables, runtime residency, publication delivery, or execution<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [residency glossary](../../../../glossary.md#residency))</span>
- changes to model/config/planning/trace/runtime/prepare/engine/backend source or tests, module
  dependencies, compiler/root/shared Gradle configuration, architecture rules/docs/tests,
  backend conformance tests, or integration tests<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- a detailed task 0003, 0004, or 0005 specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially core invariants, compiler
  responsibilities, dependency rules, and compile lifecycle<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [dependency glossary](../../../../glossary.md#term-dependency); [compile glossary](../../../../glossary.md#compile); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Compiler task 0001](0001-tensor-expression-graph-capture.md)
- [Model master plan](../../model/master-plan.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model capability contract closure audit](../../model/model-capability-contract-closure-audit.md)
- [Model task 0009: Compiled graph model](../../model/tasks/0009-compiled-graph-model.md)<span data-glossary-navigation="true"> ([Compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel))</span>
- [Model task 0018K: Operation signature and construction hardening](../../model/tasks/0018k-operation-signature-and-construction-hardening.md)<span data-glossary-navigation="true"> ([Operation signature glossary](../../../../glossary.md#operation-signature))</span>
- [Model task 0018M: Symbolic extent expressions](../../model/tasks/0018m-symbolic-extent-expressions.md)<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>
- [Model task 0018M1: Dynamic extent adoption](../../model/tasks/0018m1-dynamic-extent-adoption.md)
- [Model task 0023A: Binding-aware sum-to-Shape](../../model/tasks/0023a-binding-aware-sum-to-shape.md)<span data-glossary-navigation="true"> ([Binding glossary](../../../../glossary.md#term-binding); [sum-to-Shape glossary](../../../../glossary.md#sum-to-shape))</span>
- [Planning contract closure audit](../../planning/planning-contract-closure-audit.md)
- [Config master plan](../../config/master-plan.md)
- [Trace master plan](../../trace/master-plan.md)<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace))</span>
- [Runtime master plan](../../runtime/master-plan.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Prepare master plan](../../prepare/master-plan.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
- [Training master plan](../../../extensions/training/master-plan.md)
- [Compile API](../../../../api/compile-api.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Glossary](../../../../glossary.md)

## Architecture constraints

- [`Tensor`](../../../../glossary.md#tensor) remains public mutable API state and is not graph IR. Inference consumes only immutable
  graph, operation, descriptor, Shape, Dimension, layout, and attributes values.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [Inference glossary](../../../../glossary.md#term-inference); [immutable glossary](../../../../glossary.md#term-immutable); [descriptor glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [layout glossary](../../../../glossary.md#layout))</span>
- [`Operation`](../../../../glossary.md#operation) and its typed attributes remain model-owned semantic descriptions. Compiler applies
  those semantics to graph occurrences; it does not move semantic kinds or attributes into the
  compiler and does not add backend support to <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>`Operation`.
- [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) remains model-owned immutable graph state. Validation does not add a field
  or mutate it. <span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation))</span>`ValidatedGraph` is package-private compiler pass state, not a second public graph
  model or compile artifact.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Dynamic and expression Dimensions remain model values without runtime bindings. The compiler
  may represent and conservatively reason about obligations; it must not assign concrete sizes.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [bindings glossary](../../../../glossary.md#term-binding))</span>
- Compiler output remains compile-time state only. No physical, prepared, executable, backend,
  or runtime object may appear in inference or constraint types.
- No dependency or module-boundary change is authorized. Compiler continues to avoid runtime,
  prepare, engine, and concrete-backend dependencies.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine))</span>
- No trace payload or public diagnostics taxonomy is selected. Exceptions from this internal pass
  are focused validation failures until task 0005 owns compilation diagnostics.<span data-glossary-navigation="true"> ([trace payload glossary](../../../../glossary.md#trace-payload--tracepayload); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- `ARCHITECTURE.md` remains unchanged. Stop if implementation needs a public binding contract,
  another module's behavior, an artifact component, a dependency change, or a different lifecycle
  owner.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [owner glossary](../../../../glossary.md#term-owner))</span>

## Package impact

Existing package used:

- `io.github.pho001.synaptik.compiler` — remains the cohesive package-private front-end boundary
  for structural capture and captured-graph semantic inference/validation. No public compiler
  declaration is added.<span data-glossary-navigation="true"> ([structural capture glossary](../../../../glossary.md#term-structural-capture))</span><span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [compiler glossary](../../../../glossary.md#term-compiler))</span>

Packages added or changed:

- No Java package is added. The compiler root gains package-private pass/result/constraint types
  and cohesive family inference helpers. A subpackage would require widening Java visibility
  before a cross-package consumer exists; task 0005 may refine the public/internal package map
  when concrete orchestration requires it.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening))</span>

Type placement:

- `io.github.pho001.synaptik.compiler.CapturedGraphInference` — package-private stateless entry
  point, graph/value lookup, typed family dispatch, deterministic node failure context, descriptor
  comparison, and constraint collection.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [dispatch glossary](../../../../glossary.md#term-dispatch); [node glossary](../../../../glossary.md#node); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- `io.github.pho001.synaptik.compiler.ValidatedGraph` — package-private immutable pair of the exact
  accepted <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>[`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) reference and ordered unresolved constraints.
- `io.github.pho001.synaptik.compiler.DeferredGraphConstraint` — package-private immutable
  occurrence context plus the closed internal predicate vocabulary required above; companion
  package-private proof status/evaluator implementation may live in the same source file.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- `io.github.pho001.synaptik.compiler.ElementwiseInference` — package-private cohesive rules for
  elementwise, scalar, comparison, logical, selection, classification, and cast families.<span data-glossary-navigation="true"> ([elementwise glossary](../../../../glossary.md#term-elementwise); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- `io.github.pho001.synaptik.compiler.ReductionNormalizationInference` — package-private cohesive
  rules for reductions, scans, normalization, and ordering.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>
- `io.github.pho001.synaptik.compiler.IndexingInference` — package-private cohesive rules for
  scalar/tensor indexing, gathers, scatters, and one-hot.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor))</span>
- `io.github.pho001.synaptik.compiler.LayoutInference` — package-private cohesive rules for
  layout/Shape transformations, slices, composition, and windows.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [Shape glossary](../../../../glossary.md#shape))</span>
- `io.github.pho001.synaptik.compiler.StructuredOperationInference` — package-private cohesive
  rules for matrix multiplication, attention, convolution, pooling, losses, and explicit graph
  RNG/dropout.<span data-glossary-navigation="true"> ([matrix multiplication glossary](../../../../glossary.md#matrix-multiplication--matmul); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [losses glossary](../../../../glossary.md#loss); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>

Tests mirror the production package. `CapturedGraphInferenceTest` owns entry/result/failure/proof
contracts and inventory coverage; the five family-focused test files mirror the [inference](../../../../glossary.md#term-inference) groups
and use both valid task-0001 [capture](../../../../glossary.md#term-capture-meanings) and directly constructed invalid [graph](../../../../glossary.md#graph) occurrences.

## Affected files

Expected production paths:

- add
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CapturedGraphInference.java`
- add `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ValidatedGraph.java`
- add
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/DeferredGraphConstraint.java`
- add
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseInference.java`
- add
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionNormalizationInference.java`
- add
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/IndexingInference.java`
- add `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LayoutInference.java`
- add
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/StructuredOperationInference.java`

Expected test paths:

- add
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/CapturedGraphInferenceTest.java`
- add
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ElementwiseInferenceTest.java`
- add
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ReductionNormalizationInferenceTest.java`
- add
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/IndexingInferenceTest.java`
- add `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/LayoutInferenceTest.java`
- add
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/StructuredOperationInferenceTest.java`

Expected documentation/[planning](../../../../glossary.md#term-planning) paths:

- update [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- update targeted current/planned boundaries in [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- update [Glossary](../../../../glossary.md)
- create and finalize this task specification
- update [Compiler master plan](../master-plan.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- update [Roadmap](../../../roadmap.md)

Review without modification unless a documented conflict requires stopping: `GraphCapture`, its
tests, all [model](../../../../glossary.md#term-model) source/tests, config, planning, [trace](../../../../glossary.md#trace), [runtime](../../../../glossary.md#term-runtime), [prepare](../../../../glossary.md#prepare), [engine](../../../../glossary.md#term-engine), training,
[backend](../../../../glossary.md#backend), build/[Gradle](../../../../glossary.md#term-gradle), architecture, [conformance](../../../../glossary.md#term-conformance), and integration paths.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the exact twenty paths listed under
[Affected files](#affected-files): eight production files, six test files, and six documentation
or [planning](../../../../glossary.md#term-planning) files.

The larger-than-normal file count is justified by one [atomic](../../../../glossary.md#resource-transaction-and-rollback) correctness boundary: a [graph](../../../../glossary.md#graph) cannot
be called semantically validated while silently trusting some current [operation](../../../../glossary.md#operation) families. The
five family helpers and mirrored tests keep that complete inventory out of one god class. Do not
split semantic families into later partially validated [compiler](../../../../glossary.md#term-compiler) states.

No existing production or test source, [model](../../../../glossary.md#term-model) contract, build file, or architecture document is
authorized. If complete [validation](../../../../glossary.md#term-validation) needs a twenty-first path or another [module](../../../../glossary.md#term-module) change, stop and
report the concrete missing contract instead of expanding scope.

## Acceptance criteria

- The [compiler](../../../../glossary.md#term-compiler) exposes exactly one new package-private static
  `CapturedGraphInference.inferAndValidate(CompiledGraphModel)` entry point and no public compiler
  API.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- A null [graph](../../../../glossary.md#graph) fails first with `NullPointerException("graph")`.
- A successful result retains the exact graph reference, owns an immutable ordered constraint
  snapshot, and contains no Tensor, producer, provenance, backend, prepared, runtime, or physical
  state.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [snapshot glossary](../../../../glossary.md#term-snapshot); [Tensor glossary](../../../../glossary.md#tensor); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- The input graph, nodes, values, IDs, operations, attributes, descriptors, layouts, phases, and
  collections are never mutated, replaced, normalized, or reallocated as graph state.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [layouts glossary](../../../../glossary.md#layout))</span>
- Every current production operation kind and accepted attributes variant is handled by typed
  family inference. Unsupported kind implementations fail closed with the required node context.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [inference glossary](../../../../glossary.md#term-inference); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- Each operation revalidates ordered operand roles/domains and independently derives the complete
  output descriptor list. The first mismatching output reports node position/ID, kind, output
  position/ID, expected descriptor, and stored descriptor.
- Data-type promotion, Shape derivation, layout derivation, gradient eligibility, multi-output
  order, state edges, and hidden outputs match current model contracts for every family.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [model glossary](../../../../glossary.md#term-model))</span>
- The typed internal constraint vocabulary contains only the required atomic and boolean forms,
  uses immutable model values, and exposes no public binding/evaluation/serialization surface.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [boolean glossary](../../../../glossary.md#data-type--datatype); [binding glossary](../../../../glossary.md#term-binding); [evaluation glossary](../../../../glossary.md#term-evaluation); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- Static, structural, and bounded proofs follow the conservative rules above. Proven predicates
  disappear, disproven predicates fail in deterministic order, and undecidable predicates remain
  ordered in the result.
- Equal-looking constraints from distinct [node](../../../../glossary.md#node) occurrences remain distinct and retain their own
  [`NodeId`](../../../../glossary.md#nodeid) and subject.
- Different unresolved symbols are never unified or assigned values. Arithmetic proof attempts
  never wrap.
- Value-dependent index, duplicate-target, target-content, numerical-result, and storage
  obligations are neither claimed as proven nor disguised as descriptor constraints.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- The pass accepts valid task-0001 captures including pass-through graphs, multi-output top-K and
  attention, five-output batch normalization training, and hidden dropout mask/state slots.<span data-glossary-navigation="true"> ([captures glossary](../../../../glossary.md#term-capture-meanings))</span><span data-glossary-navigation="true"> ([batch normalization glossary](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- Focused tests include at least one valid occurrence for every current enum constant and every
  accepted attributes variant, plus manually constructed invalid graphs for each rule category.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [constant glossary](../../../../glossary.md#term-constant))</span>
- Tests lock deterministic validation order, descriptor mismatch context, constraint order,
  three proof outcomes, constraint immutability, fail-closed dispatch, and no public declarations.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- No test uses reflection to corrupt model objects; invalid semantic graphs are constructed only
  through current public immutable model constructors.
- Compile API marks package-private captured-graph verification inference and typed deferred
  constraints current without claiming a runnable compiler, transformation, binding, autograd,
  orchestration, diagnostics, planning, prepare, backend, runtime, or execution.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [autograd glossary](../../../../glossary.md#autograd); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare))</span>
- Tensor API current/planned statements no longer say that all compiler operand revalidation and
  constraint representation are unimplemented. They continue to distinguish unresolved
  compile-time constraints from concrete binding and value-dependent execution validation.
- The glossary defines the reusable deferred graph constraint distinction and keeps current
  internal status separate from a public binding contract.<span data-glossary-navigation="true"> ([deferred graph constraint glossary](../../../../glossary.md#deferred-graph-constraint))</span>
- `GraphCapture`, Public API, architecture docs/tests, other modules, Gradle, backend conformance,
  and integration tests remain unchanged for the recorded reasons.<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Exactly the twenty authorized paths change. Task 0002, its master-plan row, and the roadmap are
  [`Ready`](../../../../glossary.md#term-ready) before implementation and become `Complete` only after all implementation,
  documentation, and validation evidence is final. Tasks 0003–0005 remain Draft rows without
  detailed specifications.
- A separate documentation-focused agent pass has finalized affected Javadocs, API references,
  glossary impact, terminology, links, status, and evidence in this same overall change.

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) family-focused tests as needed, for example:

```bash
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.CapturedGraphInferenceTest
./gradlew :modules:compiler:test --tests 'io.github.pho001.synaptik.compiler.*InferenceTest'
```

After executable code stabilizes, run one final [compiler](../../../../glossary.md#term-compiler) [module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :modules:compiler:test
```

This task closes the compiler [capture](../../../../glossary.md#term-capture-meanings)-and-validation milestone. The implementation context must
therefore run one capability-[checkpoint](../../../../glossary.md#term-checkpoint) validation after the module suite:

```bash
./gradlew test :testing:architecture-tests:test
```

No architecture-test source change is expected because dependencies and architecture rules do
not change. The checkpoint verifies that the complete new compiler inventory remains inside the
existing boundaries. Record XML test counts and [Gradle](../../../../glossary.md#term-gradle) outcomes; do not infer counts from console
task summaries.

The implementation context hands the exact test/checkpoint evidence to the documentation-focused
agent. That pass must not repeat successful Java tests unless it changes executable Java behavior
or records a concrete stale-evidence risk.

Documentation pass:

```bash
./gradlew :modules:compiler:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

The documentation pass must also verify package-private visibility, complete `@param`, `@return`,
and expected `@throws` coverage for every new contract-relevant declaration, generated [Javadoc](../../../../glossary.md#term-javadoc),
local Markdown links/anchors, balanced fences, final newlines, glossary terminology, exact
twenty-path [scope](../../../../glossary.md#term-scope), [operation](../../../../glossary.md#operation)-family inventory coverage, no [model](../../../../glossary.md#term-model)/build/[dependency](../../../../glossary.md#term-dependency)/architecture
change, synchronized task/master/roadmap status, and absence of a task 0003-or-later detailed
specification.

Repository-wide validation after this capability checkpoint is deferred to [continuous
integration](../../../../glossary.md#term-continuous-integration-ci) unless executable code changes after the recorded checkpoint or a concrete
repository-wide risk is found.

## Dependencies

- [Compiler](../../../../glossary.md#term-compiler) task 0001, deterministic package-private structural [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction) — Complete.
- [Model](../../../../glossary.md#term-model) task 0009, [immutable](../../../../glossary.md#term-immutable) structurally closed [compiled graph](../../../../glossary.md#compiled-graph--compiledgraphmodel) — Complete.
- Model task 0018K, family-owned [operation](../../../../glossary.md#operation) [signatures](../../../../glossary.md#term-signature) and [occurrence](../../../../glossary.md#term-occurrence) [cardinality](../../../../glossary.md#term-cardinality) — Complete.
- Model tasks 0018M/0018M1, canonical symbolic extent values and current operation adoption —
  Complete.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [extent glossary](../../../../glossary.md#term-extent))</span>
- Model task 0023A and subsequent model capability closure, current deferred-shape obligations
  and operation inventory — Complete.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- [Planning](../../../../glossary.md#term-planning) task 0006 and its `CLOSED` contract audit — Complete, but not consumed by this task.
- Current compiler [dependency](../../../../glossary.md#term-dependency) on model — already present and sufficient.

Config 0004+, [Trace](../../../../glossary.md#trace) 0003+, [Runtime](../../../../glossary.md#term-runtime), [Prepare](../../../../glossary.md#prepare), Training, and concrete [backends](../../../../glossary.md#backend) are not dependencies.
This task consumes no config, planning evaluator, trace [DTO](../../../../glossary.md#data-transfer-object--dto), prepared/runtime contract, [backend
capability](../../../../glossary.md#backend-capability), or execution state despite those existing compiler [Gradle](../../../../glossary.md#term-gradle) dependencies.

## Follow-up tasks

Future Draft [compiler](../../../../glossary.md#term-compiler) rows, in order:

- 0003 — consume a successful `ValidatedGraph`, add bounded canonicalization and forward
  optimization, and re-run this pass on every transformed candidate before it advances.<span data-glossary-navigation="true"> ([canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [run glossary](../../../../glossary.md#term-run))</span>
- 0004 — construct backward graph state and reuse inference/validation for generated operation
  occurrences; gradient rules and saved values remain wholly deferred to that task.<span data-glossary-navigation="true"> ([backward graph glossary](../../../../glossary.md#backward-graph); [inference glossary](../../../../glossary.md#term-inference); [operation glossary](../../../../glossary.md#operation); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- 0005 — define publication/planning/diagnostic orchestration and immutable compile artifacts,
  including the eventual owning boundary for unresolved constraints only when its concrete
  downstream consumer is known.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [planning glossary](../../../../glossary.md#term-planning); [immutable glossary](../../../../glossary.md#term-immutable); [compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>

Concrete dynamic [binding](../../../../glossary.md#term-binding), value-dependent [validation](../../../../glossary.md#term-validation), [prepare](../../../../glossary.md#prepare)/[runtime](../../../../glossary.md#term-runtime) input [Shape](../../../../glossary.md#shape) contracts, and
[backend](../../../../glossary.md#backend) execution checks remain later [lifecycle](../../../../glossary.md#lifecycle) work. Do not create speculative detailed tasks or
public [artifacts](../../../../glossary.md#term-artifact) for them here.

## Architecture impact

Expected impact: None.

This task implements the existing [compiler](../../../../glossary.md#term-compiler)-owned [inference](../../../../glossary.md#term-inference) and [validation](../../../../glossary.md#term-validation) responsibilities with
package-private [compile](../../../../glossary.md#compile)-time state. It changes no [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings), [dependency direction](../../../../glossary.md#java-platform-and-build-tools), public
[lifecycle](../../../../glossary.md#lifecycle), or [artifact](../../../../glossary.md#term-artifact) [shape](../../../../glossary.md#shape). If correct implementation requires public [dimension](../../../../glossary.md#dimension) [binding](../../../../glossary.md#term-binding),
another module's source, a new [dependency](../../../../glossary.md#term-dependency), a `CompileArtifacts` component, or a different compile
order, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the compiler master plan, and
docs/planning/modules/compiler/tasks/0002-captured-graph-inference-and-validation.md. Read the
directly referenced compiler/model source, tests, API documentation, and completed model
contracts needed to verify the complete current operation inventory.

Implement Compiler 0002 exactly within its twenty authorized paths. Do not implement task-0003
transformation, autograd/backward work, public binding or compiler APIs, publication/planning/
diagnostic orchestration, CompileArtifacts, trace emission, prepare/runtime/backend/engine work,
value-dependent execution validation, or dependency/build/architecture changes. Stop on any
scope, inventory, or architecture conflict rather than trusting an operation family silently.

After executable implementation, the final compiler suite, and the capture-and-validation
capability checkpoint, hand the actual diff and exact evidence to a separate documentation-
focused agent or thread with clean context. That targeted pass must follow
docs/developer-guide/documentation-rules.md, finalize package-private Javadocs, Compile/Tensor API,
glossary, planning status/evidence, terminology, and links in the same overall change, and must
not repeat successful Java tests unless executable behavior changes or a concrete stale-evidence
risk is recorded.

Update this task with local decisions, exact validation evidence, implementation notes,
completion summary, and final status. Do not mark it Complete before the documentation pass and
every acceptance criterion finish.
```

## Local decisions

- Family [dispatch](../../../../glossary.md#term-dispatch) remains one closed typed `instanceof` chain in `CapturedGraphInference`; each
  cohesive helper uses enum switches or typed attributes and no registry, reflection, class-name
  dispatch, or temporary Tensor construction.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [Tensor glossary](../../../../glossary.md#tensor))</span>
- `ValidatedGraph` snapshots constraints but retains the exact accepted [graph](../../../../glossary.md#graph) reference.
- Deferred predicates and their three-valued evaluator remain package-private in the constraint
  source. Exact static element-count proof uses <span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>`BigInteger`; non-static interval arithmetic uses
  checked <span data-glossary-navigation="true"> ([interval arithmetic glossary](../../../../glossary.md#interval-arithmetic-and-directed-rounding))</span>`long` operations and returns deferred when a bound is unavailable or overflows.
- Attention inference receives the structurally validated stored output count only to select its
  accepted one- or two-output signature; it derives both descriptors independently and does not
  trust stored descriptor contents.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [signature glossary](../../../../glossary.md#term-signature); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>

## Known limitations

- No concrete dynamic [binding](../../../../glossary.md#term-binding) is supplied. An accepted `ValidatedGraph` may therefore retain
  unresolved constraints and is not by itself executable.
- Value-dependent validity cannot be decided from descriptor-only graph values and remains later
  lifecycle work.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [graph glossary](../../../../glossary.md#graph); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- The pass validates current production operation families only. An unrecognized custom kind
  fails closed until a separately specified compiler extension boundary exists.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Raw captured graphs receive ingress validation; task 0003 still owns canonicalization and the
  authoritative post-transformation inference/validation invocation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [inference glossary](../../../../glossary.md#term-inference))</span>
- No public compile exception taxonomy or trace payload exists; task 0005 owns that later
  aggregation boundary.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [trace payload glossary](../../../../glossary.md#trace-payload--tracepayload))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Final implementation-focused [run](../../../../glossary.md#term-run): `./gradlew :modules:compiler:test --tests
  'io.github.pho001.synaptik.compiler.*InferenceTest'` — `BUILD SUCCESSFUL`; JUnit XML records 30
  tests with zero skipped, failures, or errors across the six new inference test classes.<span data-glossary-navigation="true"> ([JUnit glossary](../../../../glossary.md#term-junit); [inference glossary](../../../../glossary.md#term-inference))</span>
- Final implementation [module](../../../../glossary.md#term-module) run: `./gradlew :modules:compiler:test` — `BUILD SUCCESSFUL`.
  Compiler JUnit XML records 42 tests with zero skipped, failures, or errors: 12 existing graph-
  capture tests and 30 inference/validation tests.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- Final [capture](../../../../glossary.md#term-capture-meanings)-and-validation capability [checkpoint](../../../../glossary.md#term-checkpoint): `./gradlew test
  :testing:architecture-tests:test` — `BUILD SUCCESSFUL`. Architecture-test XML records 3 tests
  and the repository XML aggregate records 1,179 tests, all with zero skipped, failures, or errors.
- No Java source changed after the recorded final [compiler](../../../../glossary.md#term-compiler) suite and checkpoint.
- Documentation-focused context `/root/docs_compiler_0002` independently reviewed the final
  implementation, all six focused test files, generated compiler Javadoc, Compile API, targeted
  Tensor API boundaries, glossary terminology, planning state, and the directly relevant model
  contracts. It applied the General, API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor); [planning glossary](../../../../glossary.md#term-planning); [model glossary](../../../../glossary.md#term-model))</span>
- The documentation context changed Javadoc only in the eight already authorized production
  paths; no executable Java changed, so it reused the successful implementation test and
  capability-checkpoint evidence instead of repeating Java tests.
- Final resumed documentation [Javadoc](../../../../glossary.md#term-javadoc) run: `./gradlew :modules:compiler:javadoc` —
  `BUILD SUCCESSFUL`; seven tasks up to date with no warnings. Generated output
  includes every package-private entry/result, family helper, predicate, proof status/evaluator,
  parameter, result, and documented expected failure.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>
- Markdown validation: `python3 /tmp/validate_synaptik_markdown.py` — passed for 233 Markdown
  files, 4,199 local links, 255 local anchors, and 2,946 fence markers; final-newline and trailing-
  whitespace checks passed. The command was run once before final planning-status edits and again
  on the final combined documentation state, with the final counts recorded below.
- Final documentation/[scope](../../../../glossary.md#term-scope) checks: `git diff --check`, exact changed-path inventory,
  package-private declaration scan, operation-kind dispatch inventory scan, changed-area scan, and
  task-spec inventory scan passed. Exactly the twenty authorized paths changed; tasks 0003–0005
  remain Draft master-plan rows and no task 0003-or-later detailed specification exists.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [operation glossary](../../../../glossary.md#operation); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Package visibility cross-check: `javap -classpath modules/compiler/build/classes/java/main -p`
  over all eight new production types showed package-private enclosing declarations and the sole
  package-private `inferAndValidate` entry point.
- The implementation follow-up closed every issue from the independent acceptance audit within the
  existing six authorized test paths. Coverage now includes hand-built invalid graphs across every
  inference rule category; distinct equal-looking constraints retained per [`NodeId`](../../../../glossary.md#nodeid); deterministic
  constraint, node, first-disproof, and descriptor-before-constraint order; descriptor mismatches
  for data type, Shape, layout, gradient eligibility, and output position one; zero-node
  pass-through capture; and all five batch-normalization-training producer outputs.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [descriptor glossary](../../../../glossary.md#term-descriptor); [data type glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [producer glossary](../../../../glossary.md#term-producer))</span>
- No-change conclusions: `ARCHITECTURE.md` and architecture source/tests remain unchanged because
  ownership, dependency direction, and lifecycle order did not change; Gradle/build and dependency
  files remain unchanged because the existing compiler-to-model boundary is sufficient; Public
  API remains unchanged because every new declaration is enclosed by package-private types;
  model/config/planning/trace/runtime/prepare/engine/training/backend source and tests remain
  unchanged because this pass consumes immutable model data only; backend conformance and
  integration tests remain unchanged because no backend or end-to-end executable behavior exists;
  task 0003+ specifications remain absent because transformation, autograd, publication, and
  artifact design are still Draft future work.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [lifecycle glossary](../../../../glossary.md#lifecycle); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [trace glossary](../../../../glossary.md#trace); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [immutable glossary](../../../../glossary.md#term-immutable); [conformance glossary](../../../../glossary.md#term-conformance); [autograd glossary](../../../../glossary.md#autograd); [publication glossary](../../../../glossary.md#term-publication); [artifact glossary](../../../../glossary.md#term-artifact))</span>

## Implementation notes

- Added the package-private captured-graph entry/result boundary, closed typed deferred-predicate
  vocabulary, conservative proof evaluator, and five cohesive family inference helpers.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [inference glossary](../../../../glossary.md#term-inference))</span>
- Typed coverage includes every production operation-kind enum and every accepted attributes
  variant across elementwise, reduction/normalization/ordering, indexing, layout/window,
  structured numeric/loss, RNG, and dropout families. Tests exercise enum inventories and
  attributes variants directly and validate representative public captured graphs, multi-output
  attention, top-K, and hidden dropout state.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [elementwise glossary](../../../../glossary.md#term-elementwise); [normalization glossary](../../../../glossary.md#term-normalization); [layout glossary](../../../../glossary.md#layout); [loss glossary](../../../../glossary.md#loss); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- The implementation derives complete descriptor lists, validates ordered roles and attributes,
  evaluates constraints after descriptor comparison, retains unresolved occurrence context, and
  fails closed for a custom <span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>[`OperationKind`](../../../../glossary.md#operationkind).
- Graph capture, model/config/planning/trace/runtime/prepare/engine/training/backend source and
  tests, Gradle/build configuration, architecture documents/tests, conformance tests, and
  integration tests were reviewed as out of implementation scope and were not modified.<span data-glossary-navigation="true"> ([Graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [model glossary](../../../../glossary.md#term-model); [planning glossary](../../../../glossary.md#term-planning); [trace glossary](../../../../glossary.md#trace); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [scope glossary](../../../../glossary.md#term-scope))</span>
- The separate documentation-focused pass expanded and finalized package-private Javadocs,
  documented current binding-free verification in the Compile API, corrected targeted Tensor API
  current/planned boundaries, and defined the reusable deferred-graph-constraint distinction in
  the glossary.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor))</span>
- The final shared change contains exactly twenty paths: all 14 authorized production/test paths
  and all six authorized documentation/planning paths.

## Completion summary

- Completed changes: executable captured-graph semantic verification, typed deferred constraints,
  conservative proof evaluation, complete typed family dispatch, and focused inventory tests.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [evaluation glossary](../../../../glossary.md#term-evaluation); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Files changed or created: all eight authorized compiler production paths and all six authorized
  compiler test paths, plus the six authorized documentation/planning paths.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: the final focused inference suite, compiler module suite, and capture-and-
  validation capability checkpoint passed with the XML evidence above; final compiler Javadoc,
  Markdown, link/anchor, whitespace, exact-scope, status, inventory, and diff checks passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [inference glossary](../../../../glossary.md#term-inference); [module glossary](../../../../glossary.md#term-module); [capture glossary](../../../../glossary.md#term-capture-meanings); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean-context `/root/docs_compiler_0002` completed the independent
  targeted review and finalization.
- Documentation impact: Compile API, targeted Tensor API boundaries, glossary, package-private
  Javadocs, compiler master plan, roadmap, and this task now describe the current internal pass
  without claiming concrete binding, transformation, public compilation, or execution.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor); [binding glossary](../../../../glossary.md#term-binding); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- Javadoc review: every new package-private contract-relevant declaration has meaningful generated
  documentation with complete parameter/result and expected-failure coverage.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>
- Glossary impact: added the deferred graph constraint definition and distinguished internal
  three-valued proof from public/concrete binding and value-dependent validation.<span data-glossary-navigation="true"> ([deferred graph constraint glossary](../../../../glossary.md#deferred-graph-constraint))</span>
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
