# Task 0016B: Sum, Mean, and Product Tensor Expressions

## Status

Complete

## Goal

Expose the implemented aggregate semantics through public, backend-independent floating Tensor
expressions for `sum`, `mean`, and `prod`. Each family supports full reduction, one axis with that
axis removed, and one axis retained with extent one. Every successful call returns a fresh
storage-free Tensor with a locally derived Shape, unchanged floating data type and gradient
eligibility, exact operation attributes, and one-input provenance.

This task constructs expression metadata only. It does not aggregate values, define numerical or
empty-domain behavior, create gradient rules, capture a graph, or report backend support.

## Scope

- Add exactly nine public instance methods to `Tensor`: full, axis, and axis-with-
  `keepDimensions` overloads for `sum`, `mean`, and `prod`.
- Add one package-private final `TensorReductionExpressions` helper with full and single-axis
  construction entries.
- Accept only `FLOAT64`, `FLOAT32`, and `BFLOAT16`; reject `INT32`, `INT64`, and `BOOL` without
  conversion.
- Normalize each positive or negative caller axis exactly once through `Shape.normalizeAxis`.
- Remove the normalized axis when `keepDimensions` is false. When true, replace only that
  dimension with `StaticDimension(1)`. Retain every unaffected Dimension reference.
- Use canonical rank-zero `Shape.scalar()` for full reduction and rank-one axis removal.
- Preserve exact input DataType and `requiresGrad` for all three kinds.
- Leave layout unresolved and attach no label or host storage.
- Use `NoOperationAttrs.INSTANCE` for full forms and a new
  `AxisReductionAttrs(normalizedAxis, keepDimensions)` for axis forms.
- Record exact one-input provenance and delegate identity allocation exactly once to
  `TensorFactory.createDerived`.
- Update the exact Tensor reflection test and add one focused numeric-reduction test.
- Finalize Javadocs, Tensor API, Compile API, glossary, task evidence, master plan, and roadmap
  through the required independent documentation pass.

## Out of scope

- reduction `min`/`max`, `all`/`any`, `argMax`, masked reductions, `cumSum`, softmax, loss
  reduction, pooling, normalization, or another operation family
- multiple axes, an axis collection, empty axis selection, full-reduction retained dimensions,
  named axes, or a reduction-options object
- value or storage access, aggregation, allocation, copying, materialization, aliasing, mutation,
  or output storage
- accumulation order, parallelism, vectorization, FAST/Kahan/Neumaier modes, intermediate
  precision, overflow, underflow, or product ordering
- empty-domain identities/errors, mean division by zero, NaN, infinity, or signed-zero policy
- integral/BOOL input, implicit cast, promotion, accumulation/output dtype override, or backend
  capability lookup
- gradient values or rules, zero-product policy, autograd, optimizer, or training execution
- changes to reduction semantics, Operation foundations, Shape/Dimension, descriptor, provenance,
  factory, or another existing contract
- aliases, public helpers/builders, static Tensor forms, labels, or overloads beyond the exact nine
- graph capture, compiler inference/optimization, planning, prepare, runtime, backend, tracing,
  ONNX, dependencies, Gradle, architecture, another module, or a task-0016C specification

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
- [Task 0016A](0016a-reduction-semantic-kinds-and-attributes.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes exactly three fluent forms per family: full
reduction, one axis with removal, and one axis with explicit retained-dimension choice. Axes may be
negative. Legacy retains floating input type, supports strided inputs, and installs Tensor-local
gradient callbacks for sum and mean while product has no callback. Tests cover full/axis/retained
forms, retained-result broadcasting, gradients, ONNX, and CPU/Metal routes.

The new model deliberately uses rank-zero scalar full results instead of legacy `[1]`. Product
preserves a true input `requiresGrad` request because this is eligibility metadata, not proof of an
installed rule. Legacy `-1` all-axis sentinels, operation traits, accumulation configuration,
callbacks, storage, lowering, and kernels are not copied into model construction.

## Architecture constraints

- `Tensor` remains public mutable API state, not IR or an executable value.
- The helper performs local validation and Shape derivation only; it never reads values/storage,
  traverses provenance, captures a graph, chooses accumulation, or queries backends.
- Full forms pair with `NoOperationAttrs.INSTANCE`; axis forms normalize against exact input Shape
  before creating `AxisReductionAttrs`. Attributes never receive a negative axis.
- Shape derivation creates no symbolic constraint, broadcast plan, stride, layout, materialization,
  or compiler state.
- Unaffected Dimension references are retained. A retained selected axis becomes a new
  `StaticDimension(1)`.
- Full output and rank-one removal use canonical `Shape.scalar()`. Axis reduction of a scalar is
  invalid through existing Shape behavior.
- Static zero extents and dynamic dimensions remain structurally valid. No empty-domain numerical
  result is defined.
- Result type and gradient eligibility exactly match the floating input. Eligibility does not
  promise a compiled gradient rule, including for product.
- Every result is fresh, unlabeled, storage-free, unresolved-layout, and allocated only through
  `TensorFactory.createDerived` with exact one-input provenance.
- Package direction is `model.tensor -> model.operation.reduction`, operation, datatype, and shape.
  The reduction package must not import Tensor.
- Stop if another type/file, existing-contract change, numerical policy, storage access, gradient
  rule, graph behavior, dependency, or architecture decision is required.

## Package impact

Existing packages used:

- `model.tensor` owns the public surface, local Shape/descriptor construction, provenance, and
  derived factory seam.
- `model.operation` supplies `Operation`, `OperationAttrs`, and `NoOperationAttrs`.
- `model.operation.reduction` supplies aggregate kinds and axis attributes.
- `model.datatype` supplies floating-category validation.
- `model.shape` supplies Shape, Dimension, StaticDimension, scalar creation, and axis normalization.

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — exact public fluent overloads.
- `io.github.pho001.synaptik.model.tensor.TensorReductionExpressions` — package-private local
  validation, Shape derivation, and construction boundary.
- `TensorNumericReductionTest` — same-package focused helper and public-behavior test.

## Required contract

### Public Tensor surface

Add exactly:

```java
public Tensor sum()
public Tensor sum(int axis)
public Tensor sum(int axis, boolean keepDimensions)
public Tensor mean()
public Tensor mean(int axis)
public Tensor mean(int axis, boolean keepDimensions)
public Tensor prod()
public Tensor prod(int axis)
public Tensor prod(int axis, boolean keepDimensions)
```

Zero-argument methods delegate once to `applyFull`. Axis-only methods delegate once to `applyAxis`
with false and do not call another public overload. Two-argument methods delegate once with exact
caller values. All are public instance, non-static, and non-synchronized.

### Helper shape

Create one final package-private helper with zero fields/nested types, one private constructor,
exactly two package-private static entries, and four private static methods:

```java
static Tensor applyFull(Tensor input, AggregateReductionKind kind)
static Tensor applyAxis(Tensor input, AggregateReductionKind kind,
        int axis, boolean keepDimensions)
private static void validateKind(AggregateReductionKind kind)
private static void validateFloatingInput(Tensor input)
private static Shape reduceShape(Shape inputShape, int normalizedAxis,
        boolean keepDimensions)
private static Tensor create(Tensor input, AggregateReductionKind kind,
        OperationAttrs attrs, Shape shape)
```

Add no overload, public/protected API, cache, registry, strategy, service, or test hook.

### Validation order

Both entries perform:

1. `Objects.requireNonNull(input, "input")`.
2. `Objects.requireNonNull(kind, "kind")`.
3. Accept exactly `SUM`, `MEAN`, or `PROD`; otherwise throw `IllegalArgumentException` with exact
   message `kind must be SUM, MEAN, or PROD, but was <kind>`.
4. Require floating input; otherwise throw `IllegalArgumentException` with exact message
   `input must be a floating data type, but was <dataType>`.

`applyAxis` then calls `inputShape.normalizeAxis(axis)` exactly once. Existing Shape exception type
and message remain unchanged, so kind/type failures precede axis failure. Pre-factory failures
consume no Tensor identity.

### Full reduction

After validation, full forms construct canonical `Shape.scalar()`, an unresolved descriptor with
exact input type and `requiresGrad`, exact `Operation(kind, NoOperationAttrs.INSTANCE)`, exact
one-input provenance, and one `createDerived` call with empty label. Scalar, static, zero-element,
and dynamic input Shapes are accepted without element-count inspection. A scalar full reduction is
fresh, not the input.

### Axis Shape derivation

For removal, allocate one `Dimension[]` of rank minus one, copy every nonselected Dimension in
order by exact reference, and call `Shape.ofDimensions` once. Rank one becomes canonical scalar.

For retention, copy dimensions to one same-rank array, replace exactly the selected entry with
`new StaticDimension(1)`, preserve every other reference, and call `Shape.ofDimensions` once.

Do not use `toLongArray`, require static Shape, inspect element count, mutate the dimension list,
or create a public Shape utility. Construct one `AxisReductionAttrs` from the normalized axis and
exact retention flag.

### Common construction

`create` receives validated values and constructs in order: one unresolved descriptor, one
Operation with exact kind/attributes reference, one provenance with `List.of(input)`, and one
`TensorFactory.createDerived` call with empty label. It performs no additional semantic validation.

All floating types preserve exact type. Non-floating types fail without cast. All three operations
preserve exact input `requiresGrad`; no gradient rule is created. Repeated and nested calls are
fresh and never simplified.

The input Tensor, descriptor, Shape, dimensions, label, provenance, storage association, and
contents remain unchanged. Output storage is always absent.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorReductionExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorNumericReductionTest.java`

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistent: Training API, capabilities, related foundational
and semantic Javadocs/tests, focused architecture, ADRs/tests, conformance/integration tests, and
Gradle configuration.

## Maximum scope

At most two production files, two tests, and six documentation/planning files: ten paths total.
Tensor and TensorTest may change only for the exact nine methods, imports/Javadocs, reflection
expectations, signatures, and non-synchronization assertions. Stop beyond this scope or if another
type, Shape change, numerical policy, storage access, graph/compiler behavior, or gradient rule is
needed. Do not create task 0016C.

## Javadoc requirements

- Update Tensor type Javadoc only as needed for numeric aggregate expression construction.
- Document every public method with full/axis scope, floating eligibility, normalization,
  result Shape, same type, unresolved layout, unchanged eligibility, freshness, storage absence,
  provenance, deferred numerics/gradients/execution, all parameters, return, and failures.
- Explain rank-zero full/rank-one results and retained extent one.
- Document helper type, constructor, entries, and private methods with validation/construction
  order, ownership, Shape references, ID effects, and failures.
- Explain acceptance of zero/dynamic extents without numerical policy and why product preserves
  eligibility without adding a rule.
- Review related Javadocs and record reasoned no-change conclusions or stop.

## Acceptance criteria

- Exactly nine public methods and the exact helper surface are added; no other API.
- Every public method is one exact delegation with correct kind/full/axis mapping.
- Exact null/kind/type/axis validation order and messages hold.
- All three floating types and kinds succeed; other kinds/types fail without conversion.
- Full results are canonical scalar for scalar/static/zero/dynamic input.
- Axis results normalize once, remove/retain correctly, and preserve unaffected static/dynamic
  Dimension references.
- Results have exact type, unchanged eligibility, empty layout/label/storage, fresh identity,
  exact attributes and one-input provenance.
- Inputs remain unchanged; no aggregation, numerical mode, gradient rule, graph/backend/build or
  architecture behavior is added.
- Focused/aggregate tests, Javadoc, root tests, reflection/javap/import/bytecode/scope and docs
  validation pass.
- A separate clean-context docs agent finalizes permitted Javadocs/APIs/glossary/planning and
  records no-change conclusions.
- 0016B becomes Complete only after both passes; 0016C remains Draft without a specification.

## Tests / validation

Run before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorNumericReductionTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact helper/public shapes and bytecode delegation; all kind/type/eligibility
combinations; full scalar/static/zero/dynamic Shapes; positive/negative and invalid axes; remove,
retain, rank-one scalar, static/dynamic reference retention; descriptor/attributes/provenance;
freshness and nesting; null/kind/type/axis precedence and ID effects; and unchanged input metadata,
Shape, provenance, storage, and contents.

Manually inspect `javap -p -c -s`, reflection, imports, and source for exact descriptors, helper
shape, one delegation, one axis normalization, structural Shape derivation, attributes, one
createDerived call, and absence of numeric/storage access, sentinel, multi-axis, element-count,
accumulation mode, cast, gradient, cross-layer type, registry/service, dependency, or build change.
Validate generated Javadoc, APIs/glossary, links/anchors/fences/whitespace, exact ten paths,
statuses, and absence of task 0016C.

## Dependencies

- 0001 supplies floating DataTypes.
- 0002 supplies Dimensions, scalar Shape, Shape creation, and axis normalization.
- 0006 supplies Operation; 0007 supplies TensorDescriptor.
- 0011–0013 supply Tensor, centralized identity allocation, provenance, and createDerived.
- 0016A supplies aggregate kinds, axis attributes, and full-form pairing.

## Follow-up tasks

- 0016C remains Draft for floating reduction min/max.
- 0016D remains Draft for BOOL all/any.
- 0016E remains Draft for arg-max and INT64 results.
- 0016F remains Draft for masked sum/mean; detailed planning decides composition versus a
  dedicated semantic form.
- Compiler tasks own capture/canonicalization/autograd; backend/config/conformance tasks own
  accumulation accuracy, empty domains, parallelism, lowering, storage, and kernels.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. This task composes model-owned Tensor, operation, descriptor, Shape, and
provenance contracts without executable or cross-layer state. Stop if architecture change is
required.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0006/0007/0011/0012/0013/0016A/0016B, Tensor API,
Compile API, Training API, glossary, current DataType/Dimension/StaticDimension/Shape/
TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/NoOperationAttrs/
AggregateReductionKind/AxisReductionAttrs contracts and tests, and Java 26 Gradle configuration.

Implement task 0016B exactly. Modify Tensor.java and add package-private final
TensorReductionExpressions.java. Update TensorTest only for the exact nine-method API and add
TensorNumericReductionTest. Add full, axis, and axis-with-keepDimensions sum/mean/prod overloads;
each delegates once to the exact helper entry and kind.

The helper has exactly applyFull, applyAxis, validateKind, validateFloatingInput, reduceShape, and
create. Accept only floating input. Full forms create canonical scalar Shape and NoOperationAttrs.
Axis forms normalize once, remove or replace the selected dimension, preserve unaffected Dimension
references, and use AxisReductionAttrs. Retain input type/requiresGrad, unresolved layout, exact
one-input provenance, and one createDerived call with no label/storage. Every call is fresh.

Do not aggregate values, define empty-domain/numerical/accuracy behavior, insert casts, add other
reductions, define gradients, capture graphs, change contracts, or add cross-layer behavior. Stop
beyond ten paths or on architecture uncertainty.

Run all task validation, then hand actual diff/evidence to a separate clean-context docs agent to
finalize permitted Javadocs/Tensor API/Compile API/glossary/planning and rerun validation. Update
0016B/master/roadmap only after both passes. Leave 0016C Draft without a spec. Do not commit/push.
```

## Local decisions

- Keep the legacy-compatible fluent names and three overload forms; use descriptive parameter name
  `keepDimensions` while preserving the `(int, boolean)` signature.
- Full results use rank-zero scalar, deliberately replacing legacy `[1]`.
- Caller axes may be negative, but attributes store only normalized non-negative values.
- Shape derivation stays in the package-private helper; one consumer does not justify a new public
  Shape API.
- Zero and dynamic extents are structurally accepted; numerical behavior is deferred.
- All kinds preserve input type and eligibility. Product eligibility is metadata, not a promise of
  an implemented gradient.
- Layout stays unresolved and one helper is shared because all three local contracts are equal;
  kind validation prevents accidental use by later reduction families.

## Known limitations

- No values/storage, only semantics and provenance.
- Floating input and full/one-axis forms only.
- Empty-domain numerics, accumulation accuracy, and gradient graphs are unimplemented.
- No masked reduction, capture, canonicalization, ONNX/backend support, or execution.

## Validation evidence

Planning reviewed architecture and focused docs; documentation/planning rules; roadmap; model
capabilities/master plan; tasks 0001, 0002, 0006, 0007, 0011, 0012, 0013, 0016A; current model
contracts/helpers/tests; APIs/glossary; and Java 26 Gradle configuration.

The legacy branch was read directly and confirms the nine overload forms, floating input, negative
axes, Shape behavior, type retention, strided input, retained-result broadcasting, sum/mean
callbacks, product without callback, accumulation configuration, ONNX, and backend evidence.
Legacy `[1]`, `-1` sentinel, callbacks, traits, runtime configuration, storage, and execution are
excluded or reassigned.

Planning selected nine methods and one helper. Existing contracts suffice; no package, dependency,
foundation, or architecture change is required.

Planning validation:

- `git diff --check` passed and targeted trailing-whitespace inspection found no matches in the
  three changed planning paths.
- The canonical section scan found every required task section, including exact API/helper shape,
  package impact, bounded scope, validation, handoff, decisions, limitations, and evidence.
- Every local Markdown file linked from this task, the master plan, and roadmap resolves; fences
  are balanced.
- Status inspection found 0016B `Ready` in this specification, its linked master-plan row, and its
  linked roadmap row/frontier text. Task 0016C remains `Draft`, and no task-0016C spec exists.
- Package review found no new package and preserves the one-way Tensor-to-reduction/shape
  dependency.
- Scope inspection found exactly this task, master plan, and roadmap changed. No Java, test, API,
  glossary, Gradle, architecture, AGENTS, or other-module path changed.

Implementation and independent documentation validation:

- The implementation pass changed exactly the authorized four Java/test paths: nine public
  `Tensor` methods, one final package-private helper, the exact Tensor reflection expectations,
  and one focused eight-test suite. A synthetic helper method caused by an array-constructor
  lambda was found and removed before final validation; the final helper has no synthetic method,
  field, nested type, cache, or test hook.
- Clean documentation context `/root/review_model_0016b_docs` independently reread the complete
  architecture contract; focused overview, lifecycle, boundary, and dependency explanations;
  documentation workflow and General, API/Javadoc, Planning, and Example profiles; planning guide
  and roadmap; model capabilities/master plan; tasks 0001, 0002, 0006, 0007, 0011, 0012, 0013,
  0016A, and 0016B; Tensor, Compile, and Training API references; glossary; actual final source,
  tests, generated Javadoc/reports, related foundational and reduction contracts, Java 26 Gradle
  configuration, and the complete workspace diff. It inspected behavior and artifacts rather than
  relying on the implementation handoff.
- The documentation pass found the new public `Tensor` methods and package-private
  `TensorReductionExpressions` type/member Javadocs complete without revision. They cover purpose,
  floating eligibility, full/axis/retained shape semantics, normalization and Dimension-reference
  behavior, exact descriptor/provenance ownership, freshness, null/kind/type/axis failures and
  ordering, identity effects, zero/dynamic extents, product eligibility, and deferred numerical,
  gradient, compiler, runtime, and backend behavior. Existing Javadocs for DataType, Dimension,
  StaticDimension, Shape, TensorDescriptor, TensorFactory, TensorProvenance, Operation,
  NoOperationAttrs, AggregateReductionKind, and AxisReductionAttrs remain accurate because this
  task composes their existing contracts without changing them.
- `docs/api/tensor-api.md` now documents the exact nine-method floating aggregate surface,
  full-versus-axis attributes, canonical scalar and structural axis shapes, unchanged type and
  gradient eligibility, unresolved storage-free results, one-input provenance, failure boundary,
  and deferred numerics/execution. Its complete newcomer-oriented dynamic-shape example compiled
  and ran against the built model classes, printing the documented scalar, removed-axis,
  retained-axis, normalized-attribute, exact-reference, and metadata results.
- `docs/api/compile-api.md` now includes current sum/mean/product expression construction while
  keeping capture, reduction inference/canonicalization, compile artifacts, and executable
  behavior planned. `docs/glossary.md` synchronizes aggregate reduction, normalized axis, Tensor,
  operation-family status, and Tensor-versus-graph distinctions with the implemented surface.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorNumericReductionTest` — `BUILD SUCCESSFUL`; 8 tests,
  zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; 14 tests, zero failures,
  errors, or skips.
- The first aggregate implementation attempt could not open the restricted Gradle-distribution
  cache lock and stopped before task execution. The approved rerun passed. The final
  post-documentation `./gradlew :modules:model:test` also reported `BUILD SUCCESSFUL`; 50 XML
  suites contain 381 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated public Javadoc contains all
  nine Tensor overloads and their parameter, result, failure, ownership, shape, provenance, and
  deferred-behavior contracts. The package-private helper is intentionally absent from public
  generated pages and was reviewed completely in source.
- `./gradlew test` — `BUILD SUCCESSFUL`; the repository lifecycle completed 36 actionable tasks
  with no failing task.
- `javap -p -c -s` confirmed exact public descriptors and one direct delegation per overload; the
  final helper has one private constructor, exactly two package-private entries and four private
  methods, no synthetic method, exact null/kind/type/axis order, one `normalizeAxis` call, one
  array and `Shape.ofDimensions` call per shape branch, exact attributes, one provenance input,
  and one `createDerived` call. Reflection tests independently confirm modifiers, field/nested-type
  absence, freshness, ID effects, shape/reference behavior, and exclusions.
- Import/source/bytecode scans found only the permitted datatype, operation, reduction, shape,
  Tensor, and JDK values. There is no value/storage access, element-count/static-shape demand,
  numeric loop, cast, negative sentinel, multiple-axis API, gradient rule, graph/compiler,
  planning, runtime, prepare, backend, registry, service, dependency, or build behavior.
- The local Markdown target-and-heading checker resolved 236 links in the six changed
  documentation/planning files with zero errors. Markdown fences are balanced; terminology and
  current/planned status scans found no stale reduction-expression claim; targeted trailing-
  whitespace scans found no matches; and `git diff --check` passed.
- Final scope is exactly ten paths: two production files, two tests, Tensor API, Compile API,
  glossary, this task, model master plan, and roadmap. Task 0016B is synchronized as Complete.
  Task 0016C remains Draft, and no task-0016C specification exists. No commit or push occurred.
- `docs/api/training-api.md` remains accurate unchanged because no gradient rule, gradient value,
  autograd, optimizer, training-session, or executable training behavior changed. The model
  capability baseline already inventories sum/mean/product full, axis, and retained forms while
  distinguishing model/public construction from executable support, so it needs no status edit.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance tests, and integration tests remain accurate unchanged because the work
  stays inside model-owned Tensor expression metadata and changes no module boundary, dependency,
  lifecycle, backend behavior, or end-to-end execution. Java 26 Gradle configuration remains
  accurate unchanged because no build, dependency, preview, incubator, or toolchain behavior
  changed.

## Implementation notes

- Added the exact nine fluent full/axis/retained-axis `sum`, `mean`, and `prod` methods as direct
  delegations to one bounded package-private helper.
- Added floating validation, canonical full scalar shape, structural one-axis removal/retention,
  exact type and eligibility retention, unresolved descriptors, full/axis attributes, and exact
  one-input provenance without value aggregation or cross-layer behavior.
- Added focused tests for exact surface/bytecode, all types/kinds/eligibility states, shapes and
  Dimension identity, freshness and unchanged inputs, validation precedence/messages, ID effects,
  and identifier exhaustion.
- Finalized Tensor API, Compile API, glossary, evidence, model master plan, and roadmap without
  changing the already-complete Javadocs or any executable behavior.

## Completion summary

- Completed changes: Implemented and documented floating full and one-axis sum, mean, and product
  Tensor expression construction with exact reduced shapes and provenance.
- Files changed or created: Exactly two production Java files, two tests, Tensor API, Compile API,
  glossary, this task, model master plan, and roadmap.
- Tests and validation: Focused suites 8/8 and 14/14, all 381 model tests across 50 suites, model
  Javadoc, root tests, compiled documentation example, bytecode/reflection/import/dependency
  checks, 236 Markdown link/anchor checks, fence/terminology/whitespace checks, exact scope/status
  checks, and `git diff --check` passed. The initial restricted-cache lock denial and approved
  passing rerun are recorded above.
- Documentation-agent review: Clean context `/root/review_model_0016b_docs` completed the
  independent pass using General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API and glossary now describe the current sum/mean/product surface;
  Compile API includes it as current expression input without claiming compiler behavior. Training
  API, capabilities, architecture/ADRs/tests, conformance/integration tests, and build
  configuration remain accurate unchanged for the recorded reasons.
- Javadoc review: New Tensor and helper Javadocs are complete unchanged; related foundational,
  descriptor, factory, provenance, operation, and reduction contracts remain accurate.
- Glossary impact: Aggregate reduction, normalized axis, Tensor status, operation-family status,
  and Tensor-versus-graph distinctions now include current numeric aggregate expressions.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0016B. Task 0016C remains Draft without a detailed
  specification.

Status: Complete
