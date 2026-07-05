# Task 0016D: Boolean All and Any Tensor Expressions

## Status

Complete

## Goal

Expose the existing aggregate `ALL` and `ANY` semantics through public, backend-independent BOOL
Tensor expressions. Each family supports full reduction, one axis with that axis removed, and one
axis retained with extent one. Every successful call returns a fresh storage-free,
non-differentiable BOOL Tensor with a locally derived Shape, exact operation attributes, and one-
input provenance.

This task extends the shared aggregate-reduction construction boundary completed by tasks 0016B
and 0016C. It constructs expression metadata only. It does not inspect truth values, define the
logical identity of an empty reduction domain, create gradient rules, capture a graph, or report
backend support.

## Scope

- Add exactly six public instance methods to `Tensor`: full, axis, and axis-with-
  `keepDimensions` overloads for reduction `all` and `any`.
- Extend the existing package-private final `TensorReductionExpressions` helper to accept
  `AggregateReductionKind.ALL` and `AggregateReductionKind.ANY` without adding a field, method,
  constructor, nested type, or second helper.
- Replace the helper's floating-only private validator with one kind-aware private validator while
  preserving the exact total six-method helper surface.
- Continue accepting only `FLOAT64`, `FLOAT32`, and `BFLOAT16` for `SUM`, `MEAN`, `PROD`, `MIN`,
  and `MAX`, with their existing failure behavior.
- Accept exactly `DataType.BOOL` for `ALL` and `ANY`; reject every floating and integral type
  without numeric truthiness or conversion.
- Reuse the existing single `Shape.normalizeAxis` call and structural result-shape derivation.
- Produce exact BOOL results with `requiresGrad=false`, unresolved layout, no label, and no host
  storage.
- Use `NoOperationAttrs.INSTANCE` for full forms and
  `AxisReductionAttrs(normalizedAxis, keepDimensions)` for axis forms.
- Record exact one-input provenance and delegate identity allocation exactly once to
  `TensorFactory.createDerived` through the existing helper.
- Update the exact Tensor reflection test, update the existing numeric-reduction test only for the
  intentionally generalized helper contract, and add one focused boolean-reduction test.
- Finalize Javadocs, Tensor API, Compile API, glossary, task evidence, master plan, and roadmap
  through the required independent documentation pass.

## Out of scope

- elementwise `logicalAnd`, `logicalOr`, and `logicalNot` behavior or their
  `BooleanLogicalKind.AND`, `OR`, and `NOT` identities
- changes to `sum`, `mean`, `prod`, reduction `min`, or reduction `max` public behavior, results,
  validation, provenance, or typed identities
- `argMax`, masked reductions, `cumSum`, softmax, loss reduction, pooling, normalization, or
  another operation family
- multiple axes, an axis collection, empty axis selection, full-reduction retained dimensions,
  named axes, or a reduction-options object
- truth-value or storage access, short-circuiting, aggregation, allocation, copying,
  materialization, aliasing, mutation, or output storage
- the empty-domain identities normally associated with conjunction/disjunction, errors for empty
  domains, initial values, or zero-element execution
- floating/integral truthiness, raw non-zero byte interpretation, nullable values, three-valued
  logic, implicit cast, promotion, or backend capability lookup
- gradient eligibility propagation, gradient values/rules, straight-through estimators, autograd,
  optimizer, or training execution
- changes to reduction semantic enums/attributes, boolean logical enums, Operation foundations,
  Shape/Dimension, descriptor, provenance, factory, or another existing contract
- aliases, public helpers/builders, static Tensor forms, labels, or overloads beyond the exact six
- graph capture, compiler inference/optimization, planning, prepare, runtime, backend, tracing,
  ONNX, dependencies, Gradle, architecture, another module, or a task-0016E specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0006](0006-operation-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0015C](0015c-boolean-logical-semantic-kinds.md)
- [Task 0015D](0015d-boolean-logical-tensor-expressions.md)
- [Task 0016A](0016a-reduction-semantic-kinds-and-attributes.md)
- [Task 0016B](0016b-sum-mean-and-product-tensor-expressions.md)
- [Task 0016C](0016c-min-and-max-tensor-reduction-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes exactly three fluent reduction forms per boolean
family: full reduction, one axis with removal, and one axis with explicit retained-dimension
choice. Axes may be negative. Legacy requires BOOL input, always produces a non-gradient BOOL
result, supports strided storage, and has CPU execution evidence for full and axis forms.

The new model deliberately uses rank-zero scalar full results instead of legacy `[1]`. Legacy
`-1` all-axis sentinels, operation traits, raw-byte truth interpretation, short-circuit loops,
storage handling, runtime graph state, lowering, and kernels are not copied into model
construction. The legacy kernel reads the first element and therefore does not establish a safe
empty-domain contract; this task leaves empty-domain logical identities to later executable and
conformance contracts.

## Architecture constraints

- `Tensor` remains public mutable API state, not IR or an executable value.
- The existing helper performs local kind/type validation and Shape derivation only; it never
  reads values/storage, traverses provenance, captures a graph, chooses truth behavior, or queries
  backends.
- Full forms pair with `NoOperationAttrs.INSTANCE`; axis forms normalize against the exact input
  Shape before creating `AxisReductionAttrs`. Attributes never receive a negative axis.
- Shape derivation creates no symbolic constraint, broadcast plan, stride, layout,
  materialization, or compiler state.
- Unaffected Dimension references are retained. A retained selected axis becomes a new
  `StaticDimension(1)`.
- Full output and rank-one removal use canonical `Shape.scalar()`. Axis reduction of a scalar is
  invalid through existing Shape behavior.
- Static zero extents and dynamic dimensions remain structurally valid. No logical empty-domain
  result is defined by model construction.
- Numeric aggregate kinds retain their existing floating-only validation, exact type, and input
  gradient eligibility. BOOL `ALL/ANY` require exact BOOL and consequently produce exact BOOL with
  false gradient eligibility through the existing descriptor contract.
- Every result is fresh, unlabeled, storage-free, unresolved-layout, and allocated only through
  `TensorFactory.createDerived` with exact one-input provenance.
- Aggregate `ALL/ANY` remain distinct from elementwise `BooleanLogicalKind.AND/OR`; method names,
  operation kinds, arity, and provenance distinguish the semantic families.
- Package direction remains `model.tensor -> model.operation.reduction`, operation, datatype, and
  shape. Neither reduction nor boolean-logical operation packages import Tensor.
- Stop if another production helper/type, existing public-contract change, empty-domain policy,
  storage access, gradient rule, graph behavior, dependency, or architecture decision is needed.

## Package impact

Existing packages used:

- `model.tensor` owns the public surface, local type/Shape/descriptor construction, provenance,
  and derived factory seam.
- `model.operation` supplies `Operation`, `OperationAttrs`, and `NoOperationAttrs`.
- `model.operation.reduction` supplies aggregate kinds and axis attributes.
- `model.operation.elementwise.logical` remains the separate identity family for elementwise
  AND/OR/NOT and is not modified.
- `model.datatype` supplies exact BOOL identity and floating-category validation.
- `model.shape` supplies Shape, Dimension, StaticDimension, scalar creation, and axis normalization.

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — exact public fluent boolean-reduction
  overloads alongside the existing aggregate methods.
- `io.github.pho001.synaptik.model.tensor.TensorReductionExpressions` — existing package-private
  kind/type validation, Shape derivation, and construction boundary shared by ordinary aggregate
  families.
- `TensorNumericReductionTest` — existing same-package regression test for numeric helper behavior
  and the generalized private helper shape.
- `TensorBooleanReductionTest` — new same-package focused BOOL/helper/public-behavior test.

## Required contract

### Public Tensor surface

Add exactly:

```java
public Tensor all()
public Tensor all(int axis)
public Tensor all(int axis, boolean keepDimensions)
public Tensor any()
public Tensor any(int axis)
public Tensor any(int axis, boolean keepDimensions)
```

Zero-argument methods delegate once to `TensorReductionExpressions.applyFull`. Axis-only methods
delegate once to `applyAxis` with false and do not call another public overload. Two-argument
methods delegate once with exact caller values. All are public instance, non-static, and non-
synchronized.

The number of declared public Tensor methods becomes exactly 67. The public method-name set gains
exactly `all` and `any`. Existing numeric aggregate and elementwise logical methods remain
unchanged.

### Generalized existing helper shape

Modify, but do not expand, the existing final package-private helper. It retains zero fields/nested
types, one private constructor, exactly two package-private static entries, and four private static
methods:

```java
static Tensor applyFull(Tensor input, AggregateReductionKind kind)
static Tensor applyAxis(Tensor input, AggregateReductionKind kind,
        int axis, boolean keepDimensions)
private static void validateKind(AggregateReductionKind kind)
private static void validateInput(Tensor input, AggregateReductionKind kind)
private static Shape reduceShape(Shape inputShape, int normalizedAxis,
        boolean keepDimensions)
private static Tensor create(Tensor input, AggregateReductionKind kind,
        OperationAttrs attrs, Shape shape)
```

Rename and replace only private `validateFloatingInput(Tensor)` with
`validateInput(Tensor, AggregateReductionKind)`. Update both entries to call it once after
`validateKind`. Add no other method, overload, field, nested type, public/protected API, cache,
registry, strategy, service, second helper, or test hook. Preserve the existing implementation
paths for `SUM`, `MEAN`, `PROD`, `MIN`, and `MAX`.

### Validation order

Both helper entries perform:

1. `Objects.requireNonNull(input, "input")`.
2. `Objects.requireNonNull(kind, "kind")`.
3. Accept exactly `SUM`, `MEAN`, `PROD`, `MIN`, `MAX`, `ALL`, or `ANY`; otherwise throw
   `IllegalArgumentException` with exact message
   `kind must be SUM, MEAN, PROD, MIN, MAX, ALL, or ANY, but was <kind>`.
4. Call `validateInput(input, kind)` exactly once.

For `ALL` or `ANY`, `validateInput` requires `input.descriptor().dataType() == DataType.BOOL`.
Otherwise it throws `IllegalArgumentException` with exact message
`input must have BOOL data type for <kind>, but was <dataType>`.

For `SUM`, `MEAN`, `PROD`, `MIN`, or `MAX`, `validateInput` preserves the completed numeric
contract: require `dataType.isFloating()`, otherwise throw `IllegalArgumentException` with exact
message `input must be a floating data type, but was <dataType>`.

`ARG_MAX` fails kind validation before input type or axis inspection. `applyAxis` normalizes the
axis exactly once only after kind and type succeed. Existing Shape exception types/messages remain
unchanged. Pre-factory failures consume no Tensor identity.

### Full boolean reduction

After validation, full forms reuse common construction with canonical `Shape.scalar()`, exact
`Operation(kind, NoOperationAttrs.INSTANCE)`, exact one-input provenance, and one createDerived
call with empty label. Scalar, static, zero-element, and dynamic input Shapes are accepted without
element-count or truth-value inspection. A scalar full reduction is fresh, not the input.

The result descriptor is exact BOOL, `Optional.empty()` layout, and `requiresGrad=false`. These
facts follow from exact BOOL input plus the existing common construction and TensorDescriptor
contract; do not add a second result-construction branch.

### Axis Shape derivation

Reuse the existing `reduceShape` implementation unchanged unless a Javadoc-only wording update is
required. Removal allocates one rank-minus-one `Dimension[]`, retains every nonselected Dimension
reference in order, and calls `Shape.ofDimensions` once. Rank one becomes canonical scalar.

Retention copies dimensions to one same-rank array, replaces exactly the selected entry with
`new StaticDimension(1)`, preserves every other reference, and calls `Shape.ofDimensions` once.

Do not require static Shape, inspect element count, read truth values, mutate the dimension list,
or create a public Shape utility. Construct one `AxisReductionAttrs` from the normalized axis and
exact retention flag.

### Common construction and semantic distinction

Reuse the existing `create` method. It constructs in order: one unresolved descriptor from exact
input type/eligibility and supplied Shape, one Operation with exact kind/attributes reference, one
provenance with `List.of(input)`, and one `TensorFactory.createDerived` call with empty label. It
performs no additional semantic validation.

Repeated and nested calls are fresh and never simplified. For example,
`left.logicalAnd(right)` records `BooleanLogicalKind.AND` and two ordered provenance inputs, while
`left.all()` or `left.all(axis)` records `AggregateReductionKind.ALL` and one input. Tests must
demonstrate this typed identity and arity distinction without executing either operation.

The input Tensor, descriptor, Shape, dimensions, label, provenance, storage association, and
contents remain unchanged. Output storage is always absent.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorReductionExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorNumericReductionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBooleanReductionTest.java`

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistent: Training API, capabilities, related foundational,
boolean-logical, and reduction Javadocs/tests, focused architecture, ADRs/tests, conformance/
integration tests, and Gradle configuration.

## Maximum scope

At most two production files, three tests, and six documentation/planning files: eleven paths
total. The fifth Java/test path is required because the completed numeric test owns exact shared-
helper regression checks while BOOL behavior needs its own focused suite. Tensor and TensorTest
may change only for the exact six overloads, imports/Javadocs, reflection expectations, signatures,
and non-synchronization assertions. TensorNumericReductionTest may change only for the intentional
six-method helper generalization, updated unsupported-kind contract, and preservation of every
numeric assertion. Stop beyond this scope or if another helper/type, Shape change, empty-domain
policy, storage access, graph/compiler behavior, or gradient rule is needed. Do not create task
0016E.

## Javadoc requirements

- Update Tensor type Javadoc only as needed for boolean aggregate expression construction.
- Document every new public method with full/axis scope, exact BOOL eligibility, normalization,
  result Shape, BOOL/false-gradient result, unresolved layout, freshness, storage absence,
  provenance, deferred empty-domain/truth/gradient/execution behavior, all parameters, return, and
  failures.
- Explain rank-zero full/rank-one results and retained extent one.
- Generalize helper type, entry, `validateKind`, `validateInput`, and common-construction Javadocs
  while preserving precise numeric behavior and documenting kind-specific type validation order,
  Shape references, identity effects, and failures.
- Explain acceptance of zero/dynamic extents without claiming `all(empty)` or `any(empty)` values.
- Clearly distinguish aggregate `ALL/ANY` from elementwise `AND/OR/NOT`.
- Review related Javadocs and record reasoned no-change conclusions or stop.

## Acceptance criteria

- Exactly six public overloads are added, Tensor has exactly 67 declared public methods, and the
  public name set gains only `all` and `any`.
- The helper retains exactly six declared nonsynthetic methods and no state/nested type; only its
  private type validator is intentionally generalized.
- Every public method is one exact delegation with correct aggregate kind/full/axis mapping.
- Existing numeric aggregate and elementwise logical methods retain behavior and typed identity.
- Exact null/kind/type/axis validation order and messages hold for numeric and BOOL families.
- Exact BOOL input succeeds for `ALL/ANY`; every floating/integral type fails without conversion.
- Full results are canonical scalar for scalar/static/zero/dynamic input.
- Axis results normalize once, remove/retain correctly, and preserve unaffected static/dynamic
  Dimension references.
- Results are exact BOOL with false eligibility, empty layout/label/storage, fresh identity, exact
  attributes, and one-input provenance.
- Aggregate ALL/ANY and elementwise AND/OR are demonstrably distinct typed operation families and
  provenance arities.
- Every completed 0016B/0016C numeric test behavior remains covered; inputs remain unchanged; no
  truth evaluation, empty-domain identity, gradient rule, graph/backend/build, or architecture
  behavior is added.
- Focused/aggregate tests, Javadoc, root tests, reflection/javap/import/bytecode/scope and docs
  validation pass.
- A separate clean-context docs agent finalizes permitted Javadocs/APIs/glossary/planning and
  records no-change conclusions.
- 0016D becomes Complete only after both passes; 0016E remains Draft without a specification.

## Tests / validation

Run before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorBooleanReductionTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorNumericReductionTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The new focused test covers exact public mapping; BOOL-only behavior; exact kind/attributes and
one-input provenance; scalar/static/zero/dynamic full forms; positive/negative and invalid axes;
removal, retention, rank-one scalar, static/dynamic reference retention; aggregate-versus-
elementwise typed identity/arity; freshness/nesting; unchanged input descriptor/Shape/layout/
label/provenance/storage/contents; null/kind/type/axis precedence; ID effects; and identifier
exhaustion. Existing numeric tests continue covering all five numeric kinds, all floating types,
non-floating rejection, helper shape, and every task-0016B/0016C invariant.

Manually inspect `javap -p -c -s`, reflection, imports, and source for exact overload descriptors,
six-method helper shape, one delegation per new overload, one kind-aware validation call, unchanged
numeric messages, one axis normalization, reused structural Shape derivation, exact attributes,
one createDerived call, and absence of truth/storage access, short-circuiting, empty-domain policy,
sentinel, multi-axis, element-count, cast, gradient, cross-layer type, registry/service, dependency,
or build change. Validate generated Javadoc, APIs/glossary, examples, links/anchors/fences/
whitespace, exact eleven paths, synchronized statuses, and absence of task 0016E.

## Dependencies

- 0001 supplies exact BOOL/floating DataTypes and non-differentiable BOOL metadata.
- 0002 supplies Dimensions, scalar Shape, Shape creation, and axis normalization.
- 0006 supplies Operation; 0007 supplies TensorDescriptor.
- 0011–0013 supply Tensor, centralized identity allocation, provenance, and createDerived.
- 0015C–0015D supply the distinct elementwise boolean kinds and Tensor methods that must remain
  unchanged.
- 0016A supplies ALL/ANY aggregate kinds, axis attributes, and full-form pairing.
- 0016B–0016C supply the shared helper and tested numeric full/axis construction behavior.

## Follow-up tasks

- 0016E remains Draft for arg-max and INT64 results.
- 0016F remains Draft for masked sum/mean; detailed planning decides composition versus a
  dedicated semantic form.
- Compiler tasks own capture/canonicalization/autograd; backend/conformance tasks own truth-byte
  interpretation, short-circuit or parallel evaluation, empty-domain identities/errors, lowering,
  storage, kernels, and cross-backend guarantees.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. This task composes model-owned Tensor, operation, descriptor, Shape, and
provenance contracts without executable or cross-layer state. Stop if architecture change is
required.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0006/0007/0011/0012/0013/0015C/0015D/0016A/
0016B/0016C/0016D, Tensor API, Compile API, Training API, glossary, current DataType/Dimension/
StaticDimension/Shape/TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/
NoOperationAttrs/AggregateReductionKind/AxisReductionAttrs/BooleanLogicalKind contracts and tests,
and Java 26 Gradle configuration.

Implement task 0016D exactly. Modify only Tensor.java and TensorReductionExpressions.java for
production. Update only TensorTest and TensorNumericReductionTest, and add
TensorBooleanReductionTest. Add exactly full, axis, and axis-with-keepDimensions all/any overloads;
each delegates once to the existing helper and exact aggregate kind. Preserve all numeric
aggregate and elementwise logical behavior.

Keep the helper at exactly six methods. Extend kind validation to exactly SUM, MEAN, PROD, MIN,
MAX, ALL, and ANY with the task's exact message. Replace private validateFloatingInput with exact
kind-aware validateInput: numeric kinds retain floating validation/message; ALL/ANY require BOOL
with the exact kind-specific message. Reuse canonical full scalar Shape, single-axis normalization/
removal/retention, unaffected Dimension references, common unresolved descriptor construction,
full/axis attributes, one-input provenance, and one createDerived call. BOOL results are exact BOOL
and non-differentiable. Every call is fresh. Keep aggregate ALL/ANY and elementwise AND/OR typed
identities distinct.

Do not inspect truth/storage values, define empty-domain identity, insert casts, add other
reductions, define gradients, capture graphs, change foundational contracts, add helpers, or
introduce cross-layer behavior. Stop beyond eleven paths or on architecture uncertainty.

Run all task validation, then hand the actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/Compile API/glossary/planning, record related-contract/capability/Training API/
architecture no-change conclusions, and rerun validation.

Update task 0016D, model master plan, and roadmap only for planning status/evidence. Do not mark
0016D Complete until both passes succeed. Leave 0016E Draft without a specification. Do not commit
or push.
```

## Local decisions

- Reuse and generalize the existing helper because all seven ordinary aggregate kinds share exact
  Shape derivation, attributes, descriptor/provenance construction, and factory boundaries. A
  second boolean helper would duplicate those invariants.
- Replace one private floating validator with one private kind-aware validator. The helper still
  has exactly six methods, while numeric failure behavior remains unchanged.
- Keep the legacy-compatible fluent names and three overload forms. Aggregate `all`/`any` remain
  separate from elementwise `logicalAnd`/`logicalOr` by typed operation identity and provenance
  arity.
- Full results use rank-zero scalar, deliberately replacing legacy `[1]`.
- Caller axes may be negative, but attributes store only normalized non-negative values.
- Zero and dynamic extents are structurally accepted. The model does not decide whether empty
  conjunction is true, empty disjunction is false, or a backend should reject an empty domain.
- BOOL input already cannot request gradients through TensorDescriptor, so common construction
  naturally produces `requiresGrad=false`; no special gradient branch is added.

## Known limitations

- No truth values/storage, only semantics and provenance.
- Exact BOOL input and full/one-axis forms only.
- Empty-domain logical identities/errors and execution order are unimplemented.
- No capture, canonicalization, ONNX/backend support, or execution.

## Validation evidence

Planning reviewed the architecture contract and focused architecture documentation;
documentation/planning rules; roadmap; model capabilities/master plan; completed tasks 0001, 0002,
0006, 0007, 0011, 0012, 0013, 0015C, 0015D, 0016A, 0016B, and 0016C; current Tensor, datatype,
operation, boolean-logical, reduction, descriptor, Shape, provenance, helper, and focused test
contracts; Tensor/Compile/Training APIs; glossary; and Java 26 Gradle configuration.

The legacy branch was read directly. It confirms full, axis-removing, and retained-axis all/any;
negative axes; exact BOOL input/output; non-gradient results; strided CPU execution; and typed
boolean result descriptors. Legacy `[1]`, negative all-axis sentinel, raw-byte truth evaluation,
short-circuit loops, operation traits, storage, runtime graph state, lowering, and kernels are
excluded or reassigned. Its first-element execution does not define a safe empty-domain result.

Planning selected six overloads, an in-place six-method helper generalization, one numeric helper-
contract test update, and one new BOOL-focused test. Existing public/foundational contracts
suffice; no package, dependency, foundation, or architecture change is required.

Planning validation:

- `git diff --check` passed. A targeted trailing-whitespace scan found no matches in the three
  changed planning paths, including this untracked task file.
- The canonical section scan found every required task section. The specification contains exact
  public/helper shapes, numeric and BOOL validation messages/order, package placement, eleven-path
  limit, acceptance criteria, validation, clean-context implementation/documentation handoff,
  decisions, limitations, evidence, and completion placeholders.
- The local Markdown target checker resolved 162 links across this task, the model master plan,
  and roadmap with zero missing files. Fence validation found balanced Markdown fences in all
  three paths; none of the changed links uses a heading anchor.
- Status inspection found task 0016D `Ready` in this specification, its linked master-plan row,
  roadmap frontier, and roadmap row. Task 0016E remains `Draft`, and no task-0016E specification
  exists.
- Package/scope review found no new package and exactly three planning paths changed. No Java,
  test, API, glossary, Gradle, architecture, AGENTS, or other-module path changed.

Implementation and independent documentation validation:

- Implementation context `/root/implement_model_0010` changed exactly the five authorized Java/
  test paths: two production files, two existing tests, and the new same-package
  `TensorBooleanReductionTest`. It added the six public overloads, generalized only the existing
  helper's kind/type validation, preserved its six-method/state-free shape, updated exact Tensor
  reflection expectations and numeric helper regression assertions, and added the focused BOOL
  suite. No package, production type, dependency, build file, or cross-layer behavior was added.
- Clean documentation context `/root/review_model_0016c_docs` independently reread the architecture
  contract; focused overview, lifecycle, module-boundary, and dependency explanations;
  documentation workflow and General, API/Javadoc, Planning, and Example profiles; planning guide
  and roadmap; model capabilities/master plan; tasks 0001, 0002, 0006, 0007, 0011, 0012, 0013,
  0015C, 0015D, 0016A, 0016B, 0016C, and 0016D; Tensor, Compile, and Training API references;
  glossary; actual final source/tests, related model contracts and tests, generated Javadoc/test
  reports, Java 26 Gradle configuration, and the complete workspace diff. It inspected behavior
  and artifacts directly instead of relying on the implementation handoff.
- The documentation pass finalized the four axis/retained-axis `Tensor.all`/`any` Javadocs so every
  new overload now states positive/negative normalization, removed/retained Shape behavior, exact
  BOOL result, false gradient eligibility, unresolved layout, freshness, storage absence,
  one-input provenance, kind-specific type failure, and deferred truth, empty-domain, gradient,
  compiler, backend, and execution behavior. It also clarified the full-form backend boundary and
  finalized `TensorReductionExpressions` validation-message and axis-normalization wording. The
  Tensor type Javadoc already described the new boolean aggregate family accurately.
- `docs/api/tensor-api.md` now documents the six-method boolean aggregate surface in the purpose,
  mental model, current expression summary, semantic-family status, planned boundary, and a focused
  full/axis/retained comparison table. It explains exact BOOL-only input/result, false eligibility,
  canonical scalar and structural axis shapes, exact one-input provenance, aggregate ALL/ANY versus
  elementwise AND/OR identity/arity, failures, freshness, and deferred truth evaluation,
  empty-domain identity, compiler/backend behavior, and execution.
- `docs/api/compile-api.md` now includes current boolean aggregate expression inputs while keeping
  compiler entry, capture, reduction inference/canonicalization, artifacts, backend ownership, and
  execution planned. `docs/glossary.md` synchronizes aggregate-reduction, normalized-axis, Tensor,
  and operation-family status and records the aggregate-versus-elementwise boolean distinction.
  No new reusable project term was required. No existing example code changed, so Example-format
  compile/run validation was not applicable.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorBooleanReductionTest --rerun-tasks` — `BUILD
  SUCCESSFUL`; the XML report contains 7 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorNumericReductionTest --rerun-tasks` — `BUILD
  SUCCESSFUL`; the XML report contains 9 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest --rerun-tasks` — `BUILD SUCCESSFUL`; the XML
  report contains 14 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test --rerun-tasks` — `BUILD SUCCESSFUL`; 51 XML suites contain 389
  tests with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc --rerun-tasks` — `BUILD SUCCESSFUL`. Generated public Javadoc
  contains all six overloads with rendered parameter, result, failure, Shape, provenance, typed-
  distinction, and deferred-behavior contracts. The package-private helper is intentionally absent
  from public generated pages and was reviewed completely in source.
- `./gradlew test --rerun-tasks` — `BUILD SUCCESSFUL`; the repository lifecycle completed 36
  actionable tasks with no failing task.
- `javap -p -c -s` confirmed the exact six public overload descriptors and one direct helper
  delegation per overload, including false in both axis-only methods and exact ALL/ANY kinds. It
  also confirmed the helper's private zero-argument constructor, exactly two package-private
  entries and four private methods, exact null/kind/type/axis validation order, one kind-aware
  validation call, unchanged numeric message path, BOOL messages, one axis normalization,
  structural removal/retention, exact attributes, one-input provenance, and one `createDerived`
  call. Verbose bytecode inspection found no synthetic helper member.
- Reflection tests confirmed 67 declared public Tensor methods, exact new method-name set and
  overload modifiers/signatures, helper final/package-private shape, zero fields/nested types, and
  six declared nonsynthetic methods. Focused behavior covers both boolean kinds, every non-BOOL
  rejection, full scalar/static/zero/dynamic inputs, positive/negative/invalid axes, removal/
  retention and Dimension identity, exact descriptor/attributes/provenance, aggregate-versus-
  elementwise identity/arity, freshness/nesting, input immutability, validation precedence/
  messages, ID effects, and exhaustion. The numeric suite retains every 0016B/0016C assertion.
- Production import/source/bytecode inspection found only the permitted tensor, datatype,
  operation, reduction, shape, and JDK contracts. It found no truth/storage access, short-
  circuiting, element-count or static-shape demand, empty-domain identity, cast, negative all-axis
  sentinel, multiple-axis API, gradient rule, graph/compiler/planning/runtime/prepare/backend type,
  registry, service, dependency, or build behavior. Existing numeric aggregate and elementwise
  logical public methods are unchanged.
- The local Markdown target-and-heading checker resolved all 247 links in the six changed
  documentation/planning paths with zero errors. Markdown fences are balanced; terminology and
  current/planned status scans found no stale boolean-reduction claim; targeted whitespace
  inspection and `git diff --check` passed.
- Final scope is exactly the authorized eleven paths: two production files, three tests, Tensor
  API, Compile API, glossary, this task, model master plan, and roadmap. Task 0016D is synchronized
  as Complete. Task 0016E remains Draft, and no task-0016E specification exists. No commit or push
  occurred.
- `docs/api/training-api.md` remains accurate unchanged because BOOL results are necessarily
  non-differentiable and this task adds no gradient value/rule, autograd, optimizer, session, or
  executable training behavior. The capability baseline already inventories full and retained-
  dimension boolean all/any and distinguishes model/public construction from later executable
  support, so it required no status edit.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests, backend-
  conformance tests, and integration tests remain accurate unchanged because the work stays inside
  model-owned Tensor expression metadata and changes no module boundary, dependency, lifecycle,
  backend behavior, or end-to-end execution. Java 26 Gradle configuration remains accurate
  unchanged because no build, dependency, preview, incubator, or toolchain behavior changed.
- `DataType`, `Dimension`, `StaticDimension`, `Shape`, `TensorDescriptor`, `TensorFactory`,
  `TensorProvenance`, `Operation`, `NoOperationAttrs`, `AggregateReductionKind`,
  `AxisReductionAttrs`, and `BooleanLogicalKind` contracts and Javadocs remain accurate unchanged.
  The generalized helper composes their existing exact-BOOL, non-differentiable descriptor,
  immutable-Shape, central-ID, typed-operation, attribute, and provenance contracts without
  changing them. Existing elementwise logical and numeric aggregate Javadocs/tests remain accurate
  and unchanged apart from the numeric test's intentional helper-name/unsupported-kind assertions.

## Implementation notes

- Added exactly six fluent full/axis/retained-axis `all` and `any` methods as direct delegations to
  the existing aggregate-reduction helper.
- Generalized the helper's accepted kinds and renamed its one private type validator to preserve
  numeric floating behavior while enforcing exact BOOL for ALL/ANY, without changing its six-
  method structure or common Shape/provenance/factory construction.
- Updated exact Tensor/helper regression assertions and added one focused seven-test BOOL suite.
- Finalized affected Tensor/helper Javadocs, Tensor API, Compile API, glossary, task evidence,
  model master plan, and roadmap without adding executable truth behavior.

## Completion summary

- Completed changes: Implemented and documented exact-BOOL full and one-axis all/any Tensor
  expression construction with fixed false eligibility, exact reduced shapes, and typed one-input
  provenance.
- Files changed or created: Exactly two production Java files, three tests, Tensor API, Compile
  API, glossary, this task, model master plan, and roadmap.
- Tests and validation: Focused suites 7/7, 9/9, and 14/14; all 389 model tests across 51 suites;
  model Javadoc; root tests; bytecode/reflection/import/source checks; 247 Markdown link/anchor
  checks; fence/terminology/whitespace checks; exact scope/status checks; and `git diff --check`
  passed.
- Documentation-agent review: Clean context `/root/review_model_0016c_docs` completed the
  independent pass using General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API, Compile API, and glossary now describe current boolean
  aggregate construction and its distinction from elementwise boolean logic. Training API,
  capabilities, architecture/ADRs/tests, conformance/integration tests, and build configuration
  remain accurate unchanged for the recorded reasons.
- Javadoc review: The new Tensor axis/retained overloads and affected helper contracts were
  finalized; full overloads and the Tensor type were already complete except for the added backend-
  boundary clarification. Related foundational, operation, descriptor, factory, provenance,
  boolean-logical, and reduction contracts remain accurate.
- Glossary impact: Synchronized existing aggregate-reduction, normalized-axis, Tensor, and
  operation-family entries; no new reusable term was necessary.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0016D. Task 0016E remains Draft without a detailed
  specification.

Status: Complete
