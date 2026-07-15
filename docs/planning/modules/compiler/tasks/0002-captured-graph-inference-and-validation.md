# Task 0002: Captured-Graph Inference and Validation

## Status

Complete

## Goal

Add the compiler's first semantic graph pass after package-private structural capture. The pass
must independently derive the descriptor facts implied by every current production operation
occurrence, reject incompatible operands or stored output descriptors, and turn each current
shape obligation that cannot be decided locally into a typed package-private deferred graph
constraint.

The pass is verification inference, not graph rewriting:

```text
captured CompiledGraphModel
  -> resolve ordered input/output descriptors
  -> infer the descriptor each operation occurrence requires
  -> compare inferred and stored descriptors
  -> prove, reject, or retain each symbolic obligation
  -> ValidatedGraph(exact same graph, unresolved constraints)
```

Task 0001 already guarantees immutable structural closure and topological order. This task must
not duplicate capture or silently trust model-time Tensor construction. It validates graph model
data through `Operation`, attributes, ordered input descriptors, and ordered output descriptors
only; it must not retain or reconstruct `Tensor`, `TensorProducer`, or provenance state.

## Scope

- Add one package-private, stateless compiler entry point with this exact shape:

  ```java
  static ValidatedGraph inferAndValidate(CompiledGraphModel graph)
  ```

- Return the exact input graph reference on success together with an immutable deterministic list
  of only the constraints that remain unresolved after compile-time proof.
- Revalidate every current production operation kind and accepted attributes variant. No current
  family may bypass inference because its Tensor construction already checked the request.
- Independently derive and compare complete output `TensorDescriptor` values, including data
  type, Shape, resolved-or-unresolved layout, and gradient-eligibility metadata.
- Validate ordered operand roles, data-type domains and promotion, ranks, axes, Shape
  relationships, attribute-to-descriptor relationships, output count/roles, and multi-output
  descriptor relationships beyond the occurrence-count checks already owned by
  `OperationSignature` and `CompiledNode`.
- Represent the current dimension/Shape obligations listed in
  [Deferred constraint coverage](#deferred-constraint-coverage) through a small typed internal
  predicate model. Prove or disprove them only from current immutable model facts.
- Use static extents, structural equality of named and canonical expression dimensions, and
  conservative bounds already carried by constrained unknown expressions. Do not bind a symbol,
  guess a value, or introduce an algebra system.
- Reject a statically disproven obligation. Omit a proven obligation from the result. Retain an
  undecidable obligation once, in deterministic node/rule/axis order, with its owning `NodeId`
  and a concise semantic subject.
- Fail closed for an unrecognized `OperationKind` implementation or a current production
  kind/attributes variant that has no compiler inference rule.
- Keep validation phase-neutral: the descriptor contract of an operation is the same for a
  `FORWARD` or future `BACKWARD` node. This task does not create or reinterpret backward work.
- Add focused compiler tests covering every current operation family and attributes variant,
  descriptor mismatch category, constraint proof outcome, failure-order contract, immutability,
  deterministic ordering, unsupported kinds, and valid captured multi-output/state graphs.
- Update the Compile API and targeted Tensor API status statements after implementation. Add the
  reusable deferred-graph-constraint distinction to the glossary.
- Complete the required separate documentation-focused review in the same overall change.

### Validation boundary after capture

`CompiledGraphModel` construction and `GraphCapture` already prove structure: value and node ID
uniqueness, resolvable references, producer closure, topological availability, graph boundaries,
phase coverage, deterministic capture order, and local operation occurrence cardinality. This
task consumes those invariants and must not add a second structural graph validator.

The new pass proves semantic agreement:

- each ordered input descriptor is legal for its operation role;
- the attributes are consistent with the input ranks and Shapes;
- every stored output descriptor equals the independently inferred descriptor for that output
  position; and
- every compile-visible symbolic obligation is proven, rejected, or retained explicitly.

The pass builds only construction-local `ValueId` lookup data. It stores no producer/use index,
mutates no graph collection, allocates no new graph ID, and changes no descriptor reference.

### Required operation-family coverage

The implementation may group closely related rules into cohesive package-private classes, but it
must cover the complete production inventory below. Family dispatch uses typed `instanceof` or
enum switches; do not use class names, reflection, annotations, a service loader, or a mutable
registry.

| Group | Required inference and revalidation |
|---|---|
| Binary, scalar, unary, comparison, logical, classification, selection, and cast | Exact operand domains; same-category promotion; integral-operation subset; scalar attribute/data-type equality; ordered broadcasting; fixed BOOL results; Shape retention; cast target and gradient policy; unresolved layout and gradient propagation. |
| Ordinary, multi-axis, masked, statistical, arg-extrema, and sum-to-Shape reductions | Exact kind/attributes variants; numeric/BOOL domains; normalized distinct axes; mask type and broadcast-to-input rule; reduced Shape; retained type or fixed INT64; correction/domain constraint; non-empty selected arg-extrema extent; target-one-or-input-equal constraints; unresolved layout and gradient policy. |
| Cumulative scan, softmax, layer normalization, RMS normalization, and batch normalization | Floating/integral family domains as applicable; normalized axes and trailing Shapes; exact epsilon/scalar typing; affine/statistic vector roles; all one- and five-output descriptor relationships; channel/trailing equality and batch-statistic constraints; unresolved layout and exact gradient propagation. |
| Sort, argsort, and top-K | Accepted input domains; normalized axis; Shape retention or selected-static-`k` replacement; values/indices output order and types; selected-extent-at-least-`k` constraint; layout and gradient policy. |
| Scalar select, Gather, Gather Elements, one-hot, Gather-ND, Scatter Elements, Gather-compatible Scatter Add, and Scatter-ND | Index data type; normalized axes; tuple depth and batch prefix; exact data/update types; family-specific Shape formulas; fixed BOOL one-hot result; functional data-shaped scatter results; layout and gradient policy. |
| Contiguous, reshape, expand, permute, expand-dimensions, squeeze, slice, slice update, target-relative crop, pad, tile, concat, stack, unfold/fold axis, and unfold/fold 2D | Rank/axis/attribute relationships; exact Shape arithmetic; element-count, singleton/expansion, slice-region, crop-region, and window constraints; exact resolved-layout derivation where the current model contract produces geometry; otherwise unresolved layout; type and gradient retention. |
| Matrix multiplication and scaled dot-product attention | Numeric/floating promotion; ranks and ordered roles; contraction/embedding/sequence equality; singleton-or-equal batch and mask broadcast obligations; optional mask and scale rules; exact output/weights descriptor order and gradient policy. |
| Grouped NCHW convolution and NCHW maximum/average pooling | Floating promotion/domain; rank and optional bias roles; grouped channel divisibility/equality; exact spatial symbolic formulas; non-negative numerator obligations; exact result Shape/type/layout/gradient metadata. |
| Mean-squared error and dense/index categorical cross entropy with logits | Floating promotion; index versus dense target domains; class-axis and reduction Shape rules; mapped descriptor equalities and compile-visible extent constraints; exact result type/layout/gradient policy. |
| Initial graph RNG state and dropout | Exact zero-input state descriptor; exact input/state roles; probability/scalar typing; ordered public output, BOOL mask, and next-state descriptors; unresolved layouts and gradient policy. |

The implementation must derive these facts directly from descriptors, typed attributes, and
public model Shape/layout operations. It must not create temporary Tensor expressions as an
inference technique because that would consume public Tensor identities, create new provenance,
and make compiler validation depend on mutable API state.

Deferral must not broaden a current model contract. A relation becomes a deferred constraint only
where the current Tensor construction contract deliberately permits that unresolved relation
while still selecting one exact output descriptor. Current contracts that reject an unprovable
relation because no exact output Dimension can be selected remain rejecting compiler rules. This
includes general `ShapeBroadcast` pairs, directional expand pairs, composition equalities, and
unequal unresolved MATMUL or attention batch candidates for which the model cannot choose one
result Dimension. By contrast, the explicitly retained unresolved/static MATMUL and attention
broadcast cases, output-descriptor-independent mask alternatives, and the other obligations below
are deferred because their exact result descriptor is already determined.

### Deferred constraint coverage

Create one small package-private typed predicate vocabulary sufficient for current compile-visible
Shape obligations. It must support exactly these atomic relations plus typed conjunction and
disjunction:

- two Dimensions are equal;
- one Dimension is at least a non-negative constant;
- one Dimension is divisible by a positive constant;
- two Shapes have equal logical element counts;
- one Shape has a logical element count equal to, or at least, a non-negative constant; and
- a non-negative start plus a Shape/Dimension extent fits within a containing Dimension.

This vocabulary must express the current obligations, including:

- reshape element-count equality;
- sum-to-Shape target-one-or-input-equal alternatives;
- non-empty arg-extrema and top-K selected extents;
- statistical reduction domain count greater than correction;
- scalar-select and slice/crop/update upper bounds when they depend on an unresolved extent;
- MATMUL contraction equality and unresolved-versus-static batch singleton-or-equal rules;
- attention embedding/sequence equality, batch singleton-or-equal, mask singleton-or-equal, and
  positive embedding requirements;
- layer/RMS/batch-normalization extent equalities and the documented empty-channel or batch-count
  alternative for batch-normalization training;
- convolution group divisibility, channel/bias equalities, and spatial numerator non-negativity;
- pooling and window-transform numerator non-negativity and column/target compatibility; and
- topologically encountered loss Shape/extent relations that depend only on descriptors and
  typed attributes.

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
  non-negative domain;
- interval bounds use checked `long` arithmetic. Overflow, an absent upper bound, or a bound that
  depends on an unavailable opposing value makes that proof unavailable rather than false;
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
  equivalence beyond existing model-value equality is permitted.

Different unresolved symbols are not equal merely because one operation requires them to be.
The requirement itself is retained as a constraint. No binding map, symbol assignment, union-find
solver, substitution, serialization, cache, or public evaluation API is added.

Value-dependent obligations are not graph descriptor inference. Index values and bounds,
duplicate scatter targets, dense categorical target normalization, ignore-index contents, host
storage contents, NaN/infinity behavior, and numerical result validation cannot be proved from
`CompiledGraphModel` because graph values retain descriptors rather than element data. This task
must not pretend to represent or discharge them. They remain explicit future constant-analysis,
binding, preparation, or execution validation work in the lifecycle owner that has the required
values.

### Validation and failure order

The entry point validates in this deterministic order:

1. reject a null graph with `NullPointerException("graph")`;
2. construct a temporary value lookup from the already validated `graph.values()` order;
3. visit nodes in stored topological order;
4. resolve that node's input and output descriptors in position order;
5. select the typed family rule and validate operands/attributes in documented role order;
6. derive the complete expected output descriptor list;
7. compare expected and stored descriptors in output-position order; and
8. evaluate generated predicates in rule then axis order, rejecting the first disproven predicate
   and appending each deferred predicate.

An unsupported kind failure must identify `nodes[index]`, its `NodeId`, the concrete kind type,
and `kind.name()`. Every semantic failure must be prefixed with the zero-based node position,
`NodeId`, and typed operation-kind diagnostic. A descriptor mismatch must additionally identify
the zero-based output position, `ValueId`, expected descriptor, and stored descriptor. A
disproven constraint must identify its semantic subject and typed predicate. Exact family-detail
wording is not a serialization or public diagnostic schema, but the common context is exact:

```text
nodes[<index>] NodeId[value=<id>] <kind-class-name>.<kind.name()>: <detail>
```

Unsupported dispatch uses detail `unsupported operation kind`. Descriptor mismatch detail begins
`output[<position>] ValueId[value=<id>] expected=<descriptor>, stored=<descriptor>`. A disproven
predicate begins `constraint <subject> failed: <predicate>`. Checked descriptor-derivation
arithmetic uses detail `descriptor derivation failed: <cause-message>` and retains the arithmetic
exception as its cause. Family-specific detail after the common prefix may remain concise and is
not a serialization or public diagnostics contract. Tests must lock this prefix, the specialized
detail above, and the ordering needed to distinguish the first failure.

All unsupported-kind, operand, attribute-to-descriptor, output-descriptor, and disproven-
constraint failures throw `IllegalArgumentException`. Checked arithmetic failure while deriving
an expected descriptor is wrapped as an `IllegalArgumentException` with the same node context and
the arithmetic failure as its cause. This internal exception behavior is not a new public compile
exception taxonomy.

The result preserves one deferred constraint per owning occurrence and semantic role. Do not
deduplicate equal-looking constraints from distinct nodes, because occurrence context is needed
for later diagnostics and graph transformation.

### Relationship to the architecture compile order

This task supplies a reusable correctness pass and applies it to raw captured graphs as a safe
compiler ingress boundary. It does not claim that raw-capture validation replaces the
architecture's canonicalization-then-inference-and-validation sequence.

Task 0003 must consume only a successful `ValidatedGraph`, perform its separately specified
canonicalization/forward transformations, and invoke this same inference/validation operation on
every resulting graph candidate before that candidate can advance. Thus the eventual transformed
pipeline still performs canonicalization before the authoritative inference/validation of the
transformed graph, while malformed captured metadata cannot enter transformation code unchecked.

## Out of scope

- graph capture changes, public capture access, a public compiler facade, `GraphCompiler`,
  `CompiledGraph`, engine orchestration, or a compile configuration aggregate
- changing, replacing, normalizing, canonicalizing, or copying the accepted graph, nodes, values,
  IDs, Operations, attributes, descriptors, layouts, or phases
- persistent producer/use indexes or a new graph container component
- dead-code elimination, common-subexpression elimination, constant folding, algebraic
  simplification, same-type cast removal, view folding, layout materialization, graph candidate
  generation, optional optimization, or any task-0003 transformation
- autograd, adjoint rules, saved-value selection, gradient accumulation, backward graph
  construction, phase assignment, compile-mode interpretation, or task 0004
- publication bindings/plans, backend intent, capability queries, owner selection, partitions,
  logical memory orchestration, diagnostics aggregation, trace emission/payloads, or
  `CompileArtifacts`
- public or runtime dimension binding, concrete input-Shape binding, symbol substitution,
  constraint serialization, a general constraint solver, or embedding unresolved constraints in
  a public artifact before its consumer is designed
- inspection of Tensor labels, host storage, element values, constant propagation, index values,
  duplicate targets, numerical results, or runtime failure policy
- physical buffers, materialization decisions, preparation, backend lowering, kernel/route
  selection, schedules, executables, runtime residency, publication delivery, or execution
- changes to model/config/planning/trace/runtime/prepare/engine/backend source or tests, module
  dependencies, compiler/root/shared Gradle configuration, architecture rules/docs/tests,
  backend conformance tests, or integration tests
- a detailed task 0003, 0004, or 0005 specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially core invariants, compiler
  responsibilities, dependency rules, and compile lifecycle
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Compiler task 0001](0001-tensor-expression-graph-capture.md)
- [Model master plan](../../model/master-plan.md)
- [Model capability contract closure audit](../../model/model-capability-contract-closure-audit.md)
- [Model task 0009: Compiled graph model](../../model/tasks/0009-compiled-graph-model.md)
- [Model task 0018K: Operation signature and construction hardening](../../model/tasks/0018k-operation-signature-and-construction-hardening.md)
- [Model task 0018M: Symbolic extent expressions](../../model/tasks/0018m-symbolic-extent-expressions.md)
- [Model task 0018M1: Dynamic extent adoption](../../model/tasks/0018m1-dynamic-extent-adoption.md)
- [Model task 0023A: Binding-aware sum-to-Shape](../../model/tasks/0023a-binding-aware-sum-to-shape.md)
- [Planning contract closure audit](../../planning/planning-contract-closure-audit.md)
- [Config master plan](../../config/master-plan.md)
- [Trace master plan](../../trace/master-plan.md)
- [Runtime master plan](../../runtime/master-plan.md)
- [Prepare master plan](../../prepare/master-plan.md)
- [Training master plan](../../../extensions/training/master-plan.md)
- [Compile API](../../../../api/compile-api.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `Tensor` remains public mutable API state and is not graph IR. Inference consumes only immutable
  graph, operation, descriptor, Shape, Dimension, layout, and attributes values.
- `Operation` and its typed attributes remain model-owned semantic descriptions. Compiler applies
  those semantics to graph occurrences; it does not move semantic kinds or attributes into the
  compiler and does not add backend support to `Operation`.
- `CompiledGraphModel` remains model-owned immutable graph state. Validation does not add a field
  or mutate it. `ValidatedGraph` is package-private compiler pass state, not a second public graph
  model or compile artifact.
- Dynamic and expression Dimensions remain model values without runtime bindings. The compiler
  may represent and conservatively reason about obligations; it must not assign concrete sizes.
- Compiler output remains compile-time state only. No physical, prepared, executable, backend,
  or runtime object may appear in inference or constraint types.
- No dependency or module-boundary change is authorized. Compiler continues to avoid runtime,
  prepare, engine, and concrete-backend dependencies.
- No trace payload or public diagnostics taxonomy is selected. Exceptions from this internal pass
  are focused validation failures until task 0005 owns compilation diagnostics.
- `ARCHITECTURE.md` remains unchanged. Stop if implementation needs a public binding contract,
  another module's behavior, an artifact component, a dependency change, or a different lifecycle
  owner.

## Package impact

Existing package used:

- `io.github.pho001.synaptik.compiler` — remains the cohesive package-private front-end boundary
  for structural capture and captured-graph semantic inference/validation. No public compiler
  declaration is added.

Packages added or changed:

- No Java package is added. The compiler root gains package-private pass/result/constraint types
  and cohesive family inference helpers. A subpackage would require widening Java visibility
  before a cross-package consumer exists; task 0005 may refine the public/internal package map
  when concrete orchestration requires it.

Type placement:

- `io.github.pho001.synaptik.compiler.CapturedGraphInference` — package-private stateless entry
  point, graph/value lookup, typed family dispatch, deterministic node failure context, descriptor
  comparison, and constraint collection.
- `io.github.pho001.synaptik.compiler.ValidatedGraph` — package-private immutable pair of the exact
  accepted `CompiledGraphModel` reference and ordered unresolved constraints.
- `io.github.pho001.synaptik.compiler.DeferredGraphConstraint` — package-private immutable
  occurrence context plus the closed internal predicate vocabulary required above; companion
  package-private proof status/evaluator implementation may live in the same source file.
- `io.github.pho001.synaptik.compiler.ElementwiseInference` — package-private cohesive rules for
  elementwise, scalar, comparison, logical, selection, classification, and cast families.
- `io.github.pho001.synaptik.compiler.ReductionNormalizationInference` — package-private cohesive
  rules for reductions, scans, normalization, and ordering.
- `io.github.pho001.synaptik.compiler.IndexingInference` — package-private cohesive rules for
  scalar/tensor indexing, gathers, scatters, and one-hot.
- `io.github.pho001.synaptik.compiler.LayoutInference` — package-private cohesive rules for
  layout/Shape transformations, slices, composition, and windows.
- `io.github.pho001.synaptik.compiler.StructuredOperationInference` — package-private cohesive
  rules for matrix multiplication, attention, convolution, pooling, losses, and explicit graph
  RNG/dropout.

Tests mirror the production package. `CapturedGraphInferenceTest` owns entry/result/failure/proof
contracts and inventory coverage; the five family-focused test files mirror the inference groups
and use both valid task-0001 capture and directly constructed invalid graph occurrences.

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

Expected documentation/planning paths:

- update [Compile API](../../../../api/compile-api.md)
- update targeted current/planned boundaries in [Tensor API](../../../../api/tensor-api.md)
- update [Glossary](../../../../glossary.md)
- create and finalize this task specification
- update [Compiler master plan](../master-plan.md)
- update [Roadmap](../../../roadmap.md)

Review without modification unless a documented conflict requires stopping: `GraphCapture`, its
tests, all model source/tests, config, planning, trace, runtime, prepare, engine, training,
backend, build/Gradle, architecture, conformance, and integration paths.

## Maximum scope

This task may create or modify at most the exact twenty paths listed under
[Affected files](#affected-files): eight production files, six test files, and six documentation
or planning files.

The larger-than-normal file count is justified by one atomic correctness boundary: a graph cannot
be called semantically validated while silently trusting some current operation families. The
five family helpers and mirrored tests keep that complete inventory out of one god class. Do not
split semantic families into later partially validated compiler states.

No existing production or test source, model contract, build file, or architecture document is
authorized. If complete validation needs a twenty-first path or another module change, stop and
report the concrete missing contract instead of expanding scope.

## Acceptance criteria

- The compiler exposes exactly one new package-private static
  `CapturedGraphInference.inferAndValidate(CompiledGraphModel)` entry point and no public compiler
  API.
- A null graph fails first with `NullPointerException("graph")`.
- A successful result retains the exact graph reference, owns an immutable ordered constraint
  snapshot, and contains no Tensor, producer, provenance, backend, prepared, runtime, or physical
  state.
- The input graph, nodes, values, IDs, operations, attributes, descriptors, layouts, phases, and
  collections are never mutated, replaced, normalized, or reallocated as graph state.
- Every current production operation kind and accepted attributes variant is handled by typed
  family inference. Unsupported kind implementations fail closed with the required node context.
- Each operation revalidates ordered operand roles/domains and independently derives the complete
  output descriptor list. The first mismatching output reports node position/ID, kind, output
  position/ID, expected descriptor, and stored descriptor.
- Data-type promotion, Shape derivation, layout derivation, gradient eligibility, multi-output
  order, state edges, and hidden outputs match current model contracts for every family.
- The typed internal constraint vocabulary contains only the required atomic and boolean forms,
  uses immutable model values, and exposes no public binding/evaluation/serialization surface.
- Static, structural, and bounded proofs follow the conservative rules above. Proven predicates
  disappear, disproven predicates fail in deterministic order, and undecidable predicates remain
  ordered in the result.
- Equal-looking constraints from distinct node occurrences remain distinct and retain their own
  `NodeId` and subject.
- Different unresolved symbols are never unified or assigned values. Arithmetic proof attempts
  never wrap.
- Value-dependent index, duplicate-target, target-content, numerical-result, and storage
  obligations are neither claimed as proven nor disguised as descriptor constraints.
- The pass accepts valid task-0001 captures including pass-through graphs, multi-output top-K and
  attention, five-output batch normalization training, and hidden dropout mask/state slots.
- Focused tests include at least one valid occurrence for every current enum constant and every
  accepted attributes variant, plus manually constructed invalid graphs for each rule category.
- Tests lock deterministic validation order, descriptor mismatch context, constraint order,
  three proof outcomes, constraint immutability, fail-closed dispatch, and no public declarations.
- No test uses reflection to corrupt model objects; invalid semantic graphs are constructed only
  through current public immutable model constructors.
- Compile API marks package-private captured-graph verification inference and typed deferred
  constraints current without claiming a runnable compiler, transformation, binding, autograd,
  orchestration, diagnostics, planning, prepare, backend, runtime, or execution.
- Tensor API current/planned statements no longer say that all compiler operand revalidation and
  constraint representation are unimplemented. They continue to distinguish unresolved
  compile-time constraints from concrete binding and value-dependent execution validation.
- The glossary defines the reusable deferred graph constraint distinction and keeps current
  internal status separate from a public binding contract.
- `GraphCapture`, Public API, architecture docs/tests, other modules, Gradle, backend conformance,
  and integration tests remain unchanged for the recorded reasons.
- Exactly the twenty authorized paths change. Task 0002, its master-plan row, and the roadmap are
  `Ready` before implementation and become `Complete` only after all implementation,
  documentation, and validation evidence is final. Tasks 0003–0005 remain Draft rows without
  detailed specifications.
- A separate documentation-focused agent pass has finalized affected Javadocs, API references,
  glossary impact, terminology, links, status, and evidence in this same overall change.

## Tests / validation

During implementation, run family-focused tests as needed, for example:

```bash
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.CapturedGraphInferenceTest
./gradlew :modules:compiler:test --tests 'io.github.pho001.synaptik.compiler.*InferenceTest'
```

After executable code stabilizes, run one final compiler module suite:

```bash
./gradlew :modules:compiler:test
```

This task closes the compiler capture-and-validation milestone. The implementation context must
therefore run one capability-checkpoint validation after the module suite:

```bash
./gradlew test :testing:architecture-tests:test
```

No architecture-test source change is expected because dependencies and architecture rules do
not change. The checkpoint verifies that the complete new compiler inventory remains inside the
existing boundaries. Record XML test counts and Gradle outcomes; do not infer counts from console
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
and expected `@throws` coverage for every new contract-relevant declaration, generated Javadoc,
local Markdown links/anchors, balanced fences, final newlines, glossary terminology, exact
twenty-path scope, operation-family inventory coverage, no model/build/dependency/architecture
change, synchronized task/master/roadmap status, and absence of a task 0003-or-later detailed
specification.

Repository-wide validation after this capability checkpoint is deferred to continuous
integration unless executable code changes after the recorded checkpoint or a concrete
repository-wide risk is found.

## Dependencies

- Compiler task 0001, deterministic package-private structural graph capture — Complete.
- Model task 0009, immutable structurally closed compiled graph — Complete.
- Model task 0018K, family-owned operation signatures and occurrence cardinality — Complete.
- Model tasks 0018M/0018M1, canonical symbolic extent values and current operation adoption —
  Complete.
- Model task 0023A and subsequent model capability closure, current deferred-shape obligations
  and operation inventory — Complete.
- Planning task 0006 and its `CLOSED` contract audit — Complete, but not consumed by this task.
- Current compiler dependency on model — already present and sufficient.

Config 0004+, Trace 0003+, Runtime, Prepare, Training, and concrete backends are not dependencies.
This task consumes no config, planning evaluator, trace DTO, prepared/runtime contract, backend
capability, or execution state despite those existing compiler Gradle dependencies.

## Follow-up tasks

Future Draft compiler rows, in order:

- 0003 — consume a successful `ValidatedGraph`, add bounded canonicalization and forward
  optimization, and re-run this pass on every transformed candidate before it advances.
- 0004 — construct backward graph state and reuse inference/validation for generated operation
  occurrences; gradient rules and saved values remain wholly deferred to that task.
- 0005 — define publication/planning/diagnostic orchestration and immutable compile artifacts,
  including the eventual owning boundary for unresolved constraints only when its concrete
  downstream consumer is known.

Concrete dynamic binding, value-dependent validation, prepare/runtime input Shape contracts, and
backend execution checks remain later lifecycle work. Do not create speculative detailed tasks or
public artifacts for them here.

## Architecture impact

Expected impact: None.

This task implements the existing compiler-owned inference and validation responsibilities with
package-private compile-time state. It changes no module ownership, dependency direction, public
lifecycle, or artifact shape. If correct implementation requires public dimension binding,
another module's source, a new dependency, a `CompileArtifacts` component, or a different compile
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

- Family dispatch remains one closed typed `instanceof` chain in `CapturedGraphInference`; each
  cohesive helper uses enum switches or typed attributes and no registry, reflection, class-name
  dispatch, or temporary Tensor construction.
- `ValidatedGraph` snapshots constraints but retains the exact accepted graph reference.
- Deferred predicates and their three-valued evaluator remain package-private in the constraint
  source. Exact static element-count proof uses `BigInteger`; non-static interval arithmetic uses
  checked `long` operations and returns deferred when a bound is unavailable or overflows.
- Attention inference receives the structurally validated stored output count only to select its
  accepted one- or two-output signature; it derives both descriptors independently and does not
  trust stored descriptor contents.

## Known limitations

- No concrete dynamic binding is supplied. An accepted `ValidatedGraph` may therefore retain
  unresolved constraints and is not by itself executable.
- Value-dependent validity cannot be decided from descriptor-only graph values and remains later
  lifecycle work.
- The pass validates current production operation families only. An unrecognized custom kind
  fails closed until a separately specified compiler extension boundary exists.
- Raw captured graphs receive ingress validation; task 0003 still owns canonicalization and the
  authoritative post-transformation inference/validation invocation.
- No public compile exception taxonomy or trace payload exists; task 0005 owns that later
  aggregation boundary.

## Validation evidence

- Final implementation-focused run: `./gradlew :modules:compiler:test --tests
  'io.github.pho001.synaptik.compiler.*InferenceTest'` — `BUILD SUCCESSFUL`; JUnit XML records 30
  tests with zero skipped, failures, or errors across the six new inference test classes.
- Final implementation module run: `./gradlew :modules:compiler:test` — `BUILD SUCCESSFUL`.
  Compiler JUnit XML records 42 tests with zero skipped, failures, or errors: 12 existing graph-
  capture tests and 30 inference/validation tests.
- Final capture-and-validation capability checkpoint: `./gradlew test
  :testing:architecture-tests:test` — `BUILD SUCCESSFUL`. Architecture-test XML records 3 tests
  and the repository XML aggregate records 1,179 tests, all with zero skipped, failures, or errors.
- No Java source changed after the recorded final compiler suite and checkpoint.
- Documentation-focused context `/root/docs_compiler_0002` independently reviewed the final
  implementation, all six focused test files, generated compiler Javadoc, Compile API, targeted
  Tensor API boundaries, glossary terminology, planning state, and the directly relevant model
  contracts. It applied the General, API/Javadoc, Planning, and Example profiles.
- The documentation context changed Javadoc only in the eight already authorized production
  paths; no executable Java changed, so it reused the successful implementation test and
  capability-checkpoint evidence instead of repeating Java tests.
- Final resumed documentation Javadoc run: `./gradlew :modules:compiler:javadoc` —
  `BUILD SUCCESSFUL`; seven tasks up to date with no warnings. Generated output
  includes every package-private entry/result, family helper, predicate, proof status/evaluator,
  parameter, result, and documented expected failure.
- Markdown validation: `python3 /tmp/validate_synaptik_markdown.py` — passed for 233 Markdown
  files, 4,199 local links, 255 local anchors, and 2,946 fence markers; final-newline and trailing-
  whitespace checks passed. The command was run once before final planning-status edits and again
  on the final combined documentation state, with the final counts recorded below.
- Final documentation/scope checks: `git diff --check`, exact changed-path inventory,
  package-private declaration scan, operation-kind dispatch inventory scan, changed-area scan, and
  task-spec inventory scan passed. Exactly the twenty authorized paths changed; tasks 0003–0005
  remain Draft master-plan rows and no task 0003-or-later detailed specification exists.
- Package visibility cross-check: `javap -classpath modules/compiler/build/classes/java/main -p`
  over all eight new production types showed package-private enclosing declarations and the sole
  package-private `inferAndValidate` entry point.
- The implementation follow-up closed every issue from the independent acceptance audit within the
  existing six authorized test paths. Coverage now includes hand-built invalid graphs across every
  inference rule category; distinct equal-looking constraints retained per `NodeId`; deterministic
  constraint, node, first-disproof, and descriptor-before-constraint order; descriptor mismatches
  for data type, Shape, layout, gradient eligibility, and output position one; zero-node
  pass-through capture; and all five batch-normalization-training producer outputs.
- No-change conclusions: `ARCHITECTURE.md` and architecture source/tests remain unchanged because
  ownership, dependency direction, and lifecycle order did not change; Gradle/build and dependency
  files remain unchanged because the existing compiler-to-model boundary is sufficient; Public
  API remains unchanged because every new declaration is enclosed by package-private types;
  model/config/planning/trace/runtime/prepare/engine/training/backend source and tests remain
  unchanged because this pass consumes immutable model data only; backend conformance and
  integration tests remain unchanged because no backend or end-to-end executable behavior exists;
  task 0003+ specifications remain absent because transformation, autograd, publication, and
  artifact design are still Draft future work.

## Implementation notes

- Added the package-private captured-graph entry/result boundary, closed typed deferred-predicate
  vocabulary, conservative proof evaluator, and five cohesive family inference helpers.
- Typed coverage includes every production operation-kind enum and every accepted attributes
  variant across elementwise, reduction/normalization/ordering, indexing, layout/window,
  structured numeric/loss, RNG, and dropout families. Tests exercise enum inventories and
  attributes variants directly and validate representative public captured graphs, multi-output
  attention, top-K, and hidden dropout state.
- The implementation derives complete descriptor lists, validates ordered roles and attributes,
  evaluates constraints after descriptor comparison, retains unresolved occurrence context, and
  fails closed for a custom `OperationKind`.
- Graph capture, model/config/planning/trace/runtime/prepare/engine/training/backend source and
  tests, Gradle/build configuration, architecture documents/tests, conformance tests, and
  integration tests were reviewed as out of implementation scope and were not modified.
- The separate documentation-focused pass expanded and finalized package-private Javadocs,
  documented current binding-free verification in the Compile API, corrected targeted Tensor API
  current/planned boundaries, and defined the reusable deferred-graph-constraint distinction in
  the glossary.
- The final shared change contains exactly twenty paths: all 14 authorized production/test paths
  and all six authorized documentation/planning paths.

## Completion summary

- Completed changes: executable captured-graph semantic verification, typed deferred constraints,
  conservative proof evaluation, complete typed family dispatch, and focused inventory tests.
- Files changed or created: all eight authorized compiler production paths and all six authorized
  compiler test paths, plus the six authorized documentation/planning paths.
- Tests and validation: the final focused inference suite, compiler module suite, and capture-and-
  validation capability checkpoint passed with the XML evidence above; final compiler Javadoc,
  Markdown, link/anchor, whitespace, exact-scope, status, inventory, and diff checks passed.
- Documentation-agent review: clean-context `/root/docs_compiler_0002` completed the independent
  targeted review and finalization.
- Documentation impact: Compile API, targeted Tensor API boundaries, glossary, package-private
  Javadocs, compiler master plan, roadmap, and this task now describe the current internal pass
  without claiming concrete binding, transformation, public compilation, or execution.
- Javadoc review: every new package-private contract-relevant declaration has meaningful generated
  documentation with complete parameter/result and expected-failure coverage.
- Glossary impact: added the deferred graph constraint definition and distinguished internal
  three-valued proof from public/concrete binding and value-dependent validation.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
