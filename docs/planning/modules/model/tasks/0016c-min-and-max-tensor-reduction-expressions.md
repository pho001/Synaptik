# Task 0016C: Min and Max Tensor Reduction Expressions

## Status

Complete

## Goal

Expose the existing aggregate `MIN` and `MAX` semantics through public, backend-independent
floating Tensor expressions. Each family supports full reduction, one axis with that axis removed,
and one axis retained with extent one. Every successful call returns a fresh storage-free Tensor
with a locally derived Shape, unchanged floating data type and gradient eligibility, exact
operation attributes, and one-input provenance.

This task extends the shared numeric-reduction construction boundary completed by task 0016B. It
constructs expression metadata only. It does not compare values, define numerical or empty-domain
behavior, create gradient rules, capture a graph, or report backend support.

## Scope

- Add exactly six public instance methods to `Tensor`: full, axis, and axis-with-
  `keepDimensions` overloads for reduction `min` and `max`.
- Extend the existing package-private final `TensorReductionExpressions` helper to accept
  `AggregateReductionKind.MIN` and `AggregateReductionKind.MAX` without changing its declared
  method, field, constructor, or nested-type surface.
- Accept only `FLOAT64`, `FLOAT32`, and `BFLOAT16`; reject `INT32`, `INT64`, and `BOOL` without
  conversion.
- Reuse the existing single `Shape.normalizeAxis` call and structural result-shape derivation.
- Preserve exact input DataType and `requiresGrad` for both extrema kinds.
- Leave layout unresolved and attach no label or host storage.
- Use `NoOperationAttrs.INSTANCE` for full forms and
  `AxisReductionAttrs(normalizedAxis, keepDimensions)` for axis forms.
- Record exact one-input provenance and delegate identity allocation exactly once to
  `TensorFactory.createDerived` through the existing helper.
- Extend the exact Tensor reflection test and existing focused numeric-reduction test rather than
  adding a second helper or duplicate test family.
- Finalize Javadocs, Tensor API, Compile API, glossary, task evidence, master plan, and roadmap
  through the required independent documentation pass.

## Out of scope

- binary elementwise `min(Tensor)` and `max(Tensor)` behavior or their
  `BinaryArithmeticKind.MIN` and `BinaryArithmeticKind.MAX` identities
- `sum`, `mean`, or `prod` behavior changes; `all`/`any`, `argMax`, masked reductions, `cumSum`,
  softmax, loss reduction, pooling, normalization, or another operation family
- multiple axes, an axis collection, empty axis selection, full-reduction retained dimensions,
  named axes, or a reduction-options object
- value or storage access, comparison, aggregation, allocation, copying, materialization,
  aliasing, mutation, or output storage
- minimum/maximum comparison rules for NaN, signed zero, infinities, equal values, or backend
  differences
- empty-domain values/errors, initial values, reduction identities, or zero-element execution
- integral/BOOL input, implicit cast, promotion, output dtype override, or backend capability lookup
- gradient values or rules, extrema tie-gradient distribution, autograd, optimizer, or training
  execution
- changes to reduction semantic enums/attributes, Operation foundations, Shape/Dimension,
  descriptor, provenance, factory, or another existing contract
- aliases, public helpers/builders, static Tensor forms, labels, or overloads beyond the exact six
- graph capture, compiler inference/optimization, planning, prepare, runtime, backend, tracing,
  ONNX, dependencies, Gradle, architecture, another module, or a task-0016D specification

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
- [Task 0014A](0014a-binary-arithmetic-semantic-kinds.md)
- [Task 0014B](0014b-binary-arithmetic-tensor-expressions.md)
- [Task 0016A](0016a-reduction-semantic-kinds-and-attributes.md)
- [Task 0016B](0016b-sum-mean-and-product-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes exactly three fluent reduction forms per extrema
family: full reduction, one axis with removal, and one axis with explicit retained-dimension
choice. Axes may be negative. Legacy accepts floating input, retains its data type, supports
non-contiguous storage, and has CPU execution evidence. Its local backward callbacks distribute an
incoming gradient equally across values tied for the selected minimum or maximum.

The new model deliberately uses rank-zero scalar full results instead of legacy `[1]`. It preserves
a true input `requiresGrad` request as eligibility metadata but does not copy the legacy Tensor-
local gradient callback or define tie behavior. Legacy `-1` all-axis sentinels, operation traits,
storage handling, numerical loops, runtime configuration, lowering, and kernels are not copied
into model construction.

## Architecture constraints

- `Tensor` remains public mutable API state, not IR or an executable value.
- The existing helper performs local validation and Shape derivation only; it never reads
  values/storage, traverses provenance, captures a graph, chooses comparison behavior, or queries
  backends.
- Full forms pair with `NoOperationAttrs.INSTANCE`; axis forms normalize against the exact input
  Shape before creating `AxisReductionAttrs`. Attributes never receive a negative axis.
- Shape derivation creates no symbolic constraint, broadcast plan, stride, layout,
  materialization, or compiler state.
- Unaffected Dimension references are retained. A retained selected axis becomes a new
  `StaticDimension(1)`.
- Full output and rank-one removal use canonical `Shape.scalar()`. Axis reduction of a scalar is
  invalid through existing Shape behavior.
- Static zero extents and dynamic dimensions remain structurally valid. No empty-domain numerical
  result is defined.
- Result type and gradient eligibility exactly match the floating input. Eligibility does not
  promise an extrema gradient rule or any tie-distribution policy.
- Every result is fresh, unlabeled, storage-free, unresolved-layout, and allocated only through
  `TensorFactory.createDerived` with exact one-input provenance.
- Reduction `AggregateReductionKind.MIN/MAX` remain distinct from elementwise
  `BinaryArithmeticKind.MIN/MAX`; method overloads select the semantic family by their signature.
- Package direction remains `model.tensor -> model.operation.reduction`, operation, datatype, and
  shape. The reduction package must not import Tensor.
- Stop if another type/file, existing-contract change, numerical policy, storage access, gradient
  rule, graph behavior, dependency, or architecture decision is required.

## Package impact

Existing packages used:

- `model.tensor` owns the public surface, local Shape/descriptor construction, provenance, and
  derived factory seam.
- `model.operation` supplies `Operation`, `OperationAttrs`, and `NoOperationAttrs`.
- `model.operation.reduction` supplies aggregate kinds and axis attributes.
- `model.operation.elementwise.binary` remains the separate identity family for existing binary
  elementwise min/max and is not modified.
- `model.datatype` supplies floating-category validation.
- `model.shape` supplies Shape, Dimension, StaticDimension, scalar creation, and axis normalization.

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — exact public fluent overloads alongside the
  existing numeric aggregate methods.
- `io.github.pho001.synaptik.model.tensor.TensorReductionExpressions` — existing package-private
  local validation, Shape derivation, and construction boundary.
- `TensorNumericReductionTest` — existing same-package focused helper and public-behavior test.

## Required contract

### Public Tensor surface

Add exactly:

```java
public Tensor min()
public Tensor min(int axis)
public Tensor min(int axis, boolean keepDimensions)
public Tensor max()
public Tensor max(int axis)
public Tensor max(int axis, boolean keepDimensions)
```

Zero-argument methods delegate once to `TensorReductionExpressions.applyFull`. Axis-only methods
delegate once to `applyAxis` with false and do not call another public overload. Two-argument
methods delegate once with exact caller values. All are public instance, non-static, and non-
synchronized.

The overloads coexist with existing `min(Tensor)` and `max(Tensor)` binary expressions without
changing them. The number of declared public Tensor methods becomes exactly 61. The existing set
of public method names does not gain a name because `min` and `max` already exist.

### Existing helper shape

Modify, but do not structurally expand, the existing final package-private helper. It retains zero
fields/nested types, one private constructor, exactly two package-private static entries, and four
private static methods:

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

Add no method, overload, field, nested type, public/protected API, cache, registry, strategy,
service, or test hook. Preserve the existing implementation paths for `SUM`, `MEAN`, and `PROD`.

### Validation order

Both helper entries continue to perform:

1. `Objects.requireNonNull(input, "input")`.
2. `Objects.requireNonNull(kind, "kind")`.
3. Accept exactly `SUM`, `MEAN`, `PROD`, `MIN`, or `MAX`; otherwise throw
   `IllegalArgumentException` with exact message
   `kind must be SUM, MEAN, PROD, MIN, or MAX, but was <kind>`.
4. Require floating input; otherwise throw `IllegalArgumentException` with exact message
   `input must be a floating data type, but was <dataType>`.

`applyAxis` then calls `inputShape.normalizeAxis(axis)` exactly once. Existing Shape exception type
and message remain unchanged, so kind/type failures precede axis failure. Pre-factory failures
consume no Tensor identity.

The unsupported-kind tests must now reject exactly `ALL`, `ANY`, and `ARG_MAX`. Existing
`SUM`/`MEAN`/`PROD` behavior must remain unchanged.

### Full extrema reduction

After validation, full forms construct canonical `Shape.scalar()`, an unresolved descriptor with
exact input type and `requiresGrad`, exact `Operation(kind, NoOperationAttrs.INSTANCE)`, exact one-
input provenance, and one `createDerived` call with empty label. Scalar, static, zero-element, and
dynamic input Shapes are accepted without element-count inspection. A scalar full reduction is
fresh, not the input.

### Axis Shape derivation

Reuse the existing `reduceShape` method unchanged unless a Javadoc-only wording update is required.
For removal, it allocates one `Dimension[]` of rank minus one, copies every nonselected Dimension
in order by exact reference, and calls `Shape.ofDimensions` once. Rank one becomes canonical
scalar.

For retention, it copies dimensions to one same-rank array, replaces exactly the selected entry
with `new StaticDimension(1)`, preserves every other reference, and calls `Shape.ofDimensions`
once.

Do not use `toLongArray`, require static Shape, inspect element count, mutate the dimension list,
or create a public Shape utility. Construct one `AxisReductionAttrs` from the normalized axis and
exact retention flag.

### Common construction and semantic distinction

Reuse the existing `create` method. It receives validated values and constructs in order: one
unresolved descriptor, one Operation with exact kind/attributes reference, one provenance with
`List.of(input)`, and one `TensorFactory.createDerived` call with empty label. It performs no
additional semantic validation.

All floating types preserve exact type. Non-floating types fail without cast. Both extrema
operations preserve exact input `requiresGrad`; no gradient rule is created. Repeated and nested
calls are fresh and never simplified.

For example, `left.min(right)` records `BinaryArithmeticKind.MIN` and two ordered provenance
inputs, while `left.min()` or `left.min(axis)` records `AggregateReductionKind.MIN` and one input.
Tests must demonstrate that these identities remain distinct without executing either operation.

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

Review without modification unless inconsistent: Training API, capabilities, related foundational,
binary-arithmetic, and reduction Javadocs/tests, focused architecture, ADRs/tests, conformance/
integration tests, and Gradle configuration.

## Maximum scope

At most two production files, two tests, and six documentation/planning files: ten paths total.
Tensor and TensorTest may change only for the exact six overloads, imports/Javadocs, reflection
expectations, signatures, and non-synchronization assertions. TensorNumericReductionTest may be
extended in place but must retain coverage of all task-0016B behavior. Stop beyond this scope or if
another type, helper, test file, Shape change, numerical policy, storage access, graph/compiler
behavior, or gradient rule is needed. Do not create task 0016D.

## Javadoc requirements

- Update Tensor type Javadoc only as needed for numeric aggregate expression construction.
- Document every new public method with full/axis scope, floating eligibility, normalization,
  result Shape, same type, unresolved layout, unchanged eligibility, freshness, storage absence,
  provenance, deferred comparison/empty-domain/gradient/execution behavior, all parameters,
  return, and failures.
- Explain rank-zero full/rank-one results and retained extent one.
- Update helper type, entries, and `validateKind` Javadocs to cover all five supported kinds and
  exact validation order/message. Review the constructor and unchanged private methods.
- Explain acceptance of zero/dynamic extents without numerical policy and why preserving
  eligibility does not add an extrema gradient or tie-distribution rule.
- Clearly distinguish aggregate reduction `MIN/MAX` from binary elementwise `MIN/MAX`.
- Review related Javadocs and record reasoned no-change conclusions or stop.

## Acceptance criteria

- Exactly six public overloads are added, Tensor has exactly 61 declared public methods, and the
  helper retains its exact existing six-method surface with no additional state or type.
- Every public method is one exact delegation with correct reduction kind/full/axis mapping.
- Existing binary `min(Tensor)`/`max(Tensor)` and sum/mean/prod methods remain unchanged in
  behavior and typed semantic identity.
- Exact null/kind/type/axis validation order and updated messages hold.
- All three floating types and both extrema kinds succeed; other kinds/types fail without
  conversion.
- Full results are canonical scalar for scalar/static/zero/dynamic input.
- Axis results normalize once, remove/retain correctly, and preserve unaffected static/dynamic
  Dimension references.
- Results have exact type, unchanged eligibility, empty layout/label/storage, fresh identity,
  exact attributes and one-input provenance.
- Aggregate and binary MIN/MAX are demonstrably distinct typed operation families.
- Inputs remain unchanged; no comparison, numerical mode, tie policy, gradient rule, graph/backend/
  build or architecture behavior is added.
- Focused/aggregate tests, Javadoc, root tests, reflection/javap/import/bytecode/scope and docs
  validation pass.
- A separate clean-context docs agent finalizes permitted Javadocs/APIs/glossary/planning and
  records no-change conclusions.
- 0016C becomes Complete only after both passes; 0016D remains Draft without a specification.

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

Extend the existing focused tests to cover exact helper/public shapes and bytecode delegation; all
five supported kinds without weakening sum/mean/prod coverage; both extrema kinds across all
floating types and eligibility states; full scalar/static/zero/dynamic Shapes; positive/negative
and invalid axes; removal, retention, rank-one scalar, static/dynamic reference retention;
descriptor/attributes/provenance; aggregate-versus-binary typed identity; freshness and nesting;
null/kind/type/axis precedence and ID effects; and unchanged input metadata, Shape, provenance,
storage, and contents.

Manually inspect `javap -p -c -s`, reflection, imports, and source for exact overload descriptors,
unchanged helper shape, one delegation per new overload, one axis normalization, reused structural
Shape derivation, attributes, one createDerived call, and absence of numeric/storage access,
sentinel, multi-axis, element-count, comparison/NaN/tie policy, cast, gradient, cross-layer type,
registry/service, dependency, or build change. Validate generated Javadoc, APIs/glossary, links/
anchors/fences/whitespace, exact ten paths, synchronized statuses, and absence of task 0016D.

## Dependencies

- 0001 supplies floating DataTypes.
- 0002 supplies Dimensions, scalar Shape, Shape creation, and axis normalization.
- 0006 supplies Operation; 0007 supplies TensorDescriptor.
- 0011–0013 supply Tensor, centralized identity allocation, provenance, and createDerived.
- 0014A–0014B supply the distinct binary elementwise MIN/MAX identity and Tensor overloads that
  must remain unchanged.
- 0016A supplies aggregate MIN/MAX kinds, axis attributes, and full-form pairing.
- 0016B supplies the shared numeric-reduction helper and tested full/axis construction behavior.

## Follow-up tasks

- 0016D remains Draft for BOOL all/any.
- 0016E remains Draft for arg-max and INT64 results.
- 0016F remains Draft for masked sum/mean; detailed planning decides composition versus a
  dedicated semantic form.
- Compiler tasks own capture/canonicalization/autograd, including extrema tie-gradient semantics;
  backend/config/conformance tasks own comparison and NaN behavior, empty domains, lowering,
  storage, kernels, and cross-backend numerical guarantees.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. This task composes model-owned Tensor, operation, descriptor, Shape, and
provenance contracts without executable or cross-layer state. Stop if architecture change is
required.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0006/0007/0011/0012/0013/0014A/0014B/0016A/
0016B/0016C, Tensor API, Compile API, Training API, glossary, current DataType/Dimension/
StaticDimension/Shape/TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/
NoOperationAttrs/AggregateReductionKind/AxisReductionAttrs/BinaryArithmeticKind contracts and
tests, and Java 26 Gradle configuration.

Implement task 0016C exactly. Modify only Tensor.java and TensorReductionExpressions.java for
production. Update only TensorTest and TensorNumericReductionTest for tests. Add exactly full,
axis, and axis-with-keepDimensions min/max overloads; each delegates once to the existing helper
and exact aggregate kind. Preserve binary min/max and every sum/mean/prod behavior.

Do not change the helper surface. Extend its kind validation to exactly SUM, MEAN, PROD, MIN, and
MAX with the task's exact message. Reuse floating validation, canonical full scalar Shape,
single-axis normalization/removal/retention, unaffected Dimension references, unresolved result,
exact type/requiresGrad, full/axis attributes, one-input provenance, and one createDerived call.
Every call is fresh. Keep aggregate and binary MIN/MAX typed identities distinct.

Do not compare values, define empty-domain/NaN/signed-zero/tie behavior, insert casts, add other
reductions, define gradients, capture graphs, change contracts, add files/helpers, or introduce
cross-layer behavior. Stop beyond ten paths or on architecture uncertainty.

Run all task validation, then hand the actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/Compile API/glossary/planning, record related-contract/capability/Training API/
architecture no-change conclusions, and rerun validation.

Update task 0016C, model master plan, and roadmap only for planning status/evidence. Do not mark
0016C Complete until both passes succeed. Leave 0016D Draft without a specification. Do not commit
or push.
```

## Local decisions

- Reuse the task-0016B helper because all five floating aggregate families share the exact local
  validation, Shape derivation, descriptor, provenance, and factory boundary. A second extrema
  helper would duplicate invariants and create divergent validation risk.
- Keep the legacy-compatible fluent names and three overload forms; `min()`/`max()` are reduction
  overloads while `min(Tensor)`/`max(Tensor)` remain binary elementwise expressions.
- Full results use rank-zero scalar, deliberately replacing legacy `[1]`.
- Caller axes may be negative, but attributes store only normalized non-negative values.
- Zero and dynamic extents are structurally accepted; numerical behavior is deferred.
- Both kinds preserve input type and eligibility. Eligibility is metadata, not an implemented
  gradient or promise that equal extrema will receive any particular gradient distribution.
- Forward equality/ties need no model attribute because the selected reduced value is identical;
  comparison/NaN/signed-zero and backward tie behavior remain explicit later-layer contracts.

## Known limitations

- No values/storage, only semantics and provenance.
- Floating input and full/one-axis forms only.
- Empty-domain, NaN, signed-zero, comparison, and tie-gradient behavior are unimplemented.
- No masked reduction, capture, canonicalization, ONNX/backend support, or execution.

## Validation evidence

Planning reviewed the architecture contract and focused architecture documentation;
documentation/planning rules; roadmap; model capabilities/master plan; completed tasks 0001, 0002,
0006, 0007, 0011, 0012, 0013, 0014A, 0014B, 0016A, and 0016B; current Tensor, operation,
reduction, descriptor, Shape, provenance, helper, and focused test contracts; Tensor/Compile/
Training APIs; glossary; and Java 26 Gradle configuration.

The legacy branch was read directly. It confirms full, axis-removing, and retained-axis min/max;
negative axes; floating type retention; non-contiguous execution; and equal tie splitting in its
Tensor-local backward callbacks. Legacy `[1]`, negative all-axis sentinel, callbacks, traits,
storage, runtime configuration, lowering, and kernels are excluded or reassigned.

Planning selected six overloads and an in-place helper/test extension. Existing contracts suffice;
no package, production/test file, dependency, foundation, or architecture change is required.

Planning validation:

- `git diff --check` passed. A targeted trailing-whitespace scan found no matches in the three
  changed planning paths, including this untracked task file.
- The canonical section scan found every required task section. The specification contains exact
  public/helper shapes, validation messages and order, package placement, ten-path limit,
  acceptance criteria, validation, clean-context implementation/documentation handoff, decisions,
  limitations, evidence, and completion placeholders.
- The local Markdown target checker resolved 158 links across this task, the model master plan,
  and roadmap with zero missing files. Fence validation found balanced Markdown fences in all
  three paths.
- Status inspection found task 0016C `Ready` in this specification, its linked master-plan row,
  roadmap frontier, and roadmap row. Task 0016D remains `Draft`, and no task-0016D specification
  exists.
- Package/scope review found no new package and exactly three planning paths changed. No Java,
  test, API, glossary, Gradle, architecture, AGENTS, or other-module path changed.

Implementation and independent documentation validation:

- The implementation pass changed exactly the four authorized Java/test paths. It added the six
  public `Tensor` overloads, extended only the existing helper's accepted kind set and rejection
  message, updated the exact Tensor reflection expectations, and extended the focused numeric-
  reduction suite. No production type, package, field, nested type, dependency, build file, or
  cross-layer behavior was added.
- Clean documentation context `/root/review_model_0016c_docs` independently reread the architecture
  contract; focused overview, lifecycle, module-boundary, and dependency explanations;
  documentation workflow and General, API/Javadoc, Planning, and Example profiles; planning guide
  and roadmap; model capabilities/master plan; tasks 0001, 0002, 0006, 0007, 0011, 0012, 0013,
  0014A, 0014B, 0016A, 0016B, and 0016C; Tensor, Compile, and Training API references; glossary;
  actual final source/tests, related model contracts and tests, generated Javadoc/reports, Java 26
  Gradle configuration, and the complete workspace diff. It inspected the implementation and test
  evidence directly rather than relying on the implementation handoff.
- The independent review found one test-regression gap before closure: the first implementation
  pass had replaced task-0016B's explicit repeated `sum` and nested `prod` freshness coverage with
  extrema-only coverage. A separate constrained implementation turn in `/root/implement_model_0010`
  modified only the already-authorized `TensorNumericReductionTest` path to restore the SUM/PROD
  assertions while retaining repeated MIN and nested MAX coverage. The documentation context then
  independently inspected the corrected diff and reran every required validation.
- The documentation pass found all six new public `Tensor` method Javadocs and every affected
  `TensorReductionExpressions` type/member Javadoc complete without revision. They document
  floating eligibility, full/axis/retained Shape semantics, exact normalization and Dimension-
  reference behavior, result type and eligibility, unresolved layout, freshness, storage absence,
  exact attributes and provenance, validation ordering and failures, aggregate-versus-binary typed
  identity, and deferred value, empty-domain, comparison, NaN, signed-zero, tie-gradient, compiler,
  backend, and execution behavior. The Tensor type Javadoc already covers the expanded numeric-
  aggregate family accurately.
- `docs/api/tensor-api.md` now documents the fifteen-method numeric aggregate surface, adds MIN and
  MAX to the full/axis/retained table, explains canonical scalar and structural single-axis shapes,
  exact type/eligibility/provenance, and distinguishes one-input aggregate MIN/MAX from two-input
  binary elementwise MIN/MAX. Its existing complete numeric-aggregate example was not changed: it
  continues to demonstrate the common helper's normalization, Shape, attributes, and provenance
  contract for all five families, so no new example compile/run was required.
- `docs/api/compile-api.md` now includes current aggregate min/max expression inputs without
  claiming capture, reduction inference/canonicalization, autograd, artifacts, backend ownership,
  or execution. `docs/glossary.md` synchronizes aggregate-reduction, normalized-axis, Tensor, and
  operation-family status and records the typed aggregate-versus-binary extrema distinction; no
  new reusable glossary term was introduced.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorNumericReductionTest --rerun-tasks` — `BUILD
  SUCCESSFUL`; the XML report contains 9 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest --rerun-tasks` — the first documentation-pass
  attempt could not open the restricted Gradle distribution lock and stopped before task
  execution. The approved rerun reported `BUILD SUCCESSFUL`; the XML report contains 14 tests with
  zero failures, errors, or skips.
- `./gradlew :modules:model:test --rerun-tasks` — `BUILD SUCCESSFUL`; 50 XML suites contain 382
  tests with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc --rerun-tasks` — `BUILD SUCCESSFUL`. Generated public Javadoc
  contains all six overloads with rendered parameter, result, failure, shape, provenance, and
  deferred-behavior contracts. The package-private helper is intentionally absent from public
  generated pages and was reviewed completely in source.
- `./gradlew test --rerun-tasks` — `BUILD SUCCESSFUL`; the repository lifecycle completed 36
  actionable tasks with no failing task.
- `javap -p -c -s` confirmed the exact six public overload descriptors and one direct helper
  delegation per overload, including false in both axis-only methods and exact aggregate kinds.
  It also confirmed the final helper's private zero-argument constructor, exactly two package-
  private entries and four private methods, exact validation order, five accepted kinds, one axis
  normalization, structural removal/retention, exact attributes, one-input provenance, and one
  `createDerived` call. Verbose bytecode inspection found no synthetic helper member.
- Reflection tests confirmed 61 declared public Tensor methods, exact overload modifiers and
  signatures, helper final/package-private shape, zero fields/nested types, and six declared
  nonsynthetic methods. Focused behavior covers all five kinds, all three floating types and both
  eligibility states, full scalar/static/zero/dynamic inputs, positive/negative/invalid axes,
  removal/retention and Dimension identity, exact descriptor/attributes/provenance, aggregate-
  versus-binary identity, freshness/nesting, input immutability, validation precedence/messages,
  ID effects, and exhaustion.
- Production import/source/bytecode inspection found only the permitted tensor, datatype,
  operation, reduction, shape, and JDK contracts. It found no value/storage access, element-count
  or static-shape demand, comparison loop, cast, negative all-axis sentinel, multiple-axis API,
  gradient rule, graph/compiler/planning/runtime/prepare/backend type, registry, service,
  dependency, or build behavior. Existing binary min/max and sum/mean/prod methods are unchanged.
- The local Markdown target-and-heading checker resolved all 242 links in the six changed
  documentation/planning paths with zero errors. Markdown fences are balanced; terminology and
  current/planned status scans found no stale aggregate-expression claim; targeted whitespace
  inspection and `git diff --check` passed.
- Final scope is exactly the authorized ten paths: two production files, two tests, Tensor API,
  Compile API, glossary, this task, model master plan, and roadmap. Task 0016C is synchronized as
  Complete. Task 0016D remains Draft, and no task-0016D specification exists. No commit or push
  occurred.
- `docs/api/training-api.md` remains accurate unchanged because this task preserves only an
  eligibility request and adds no gradient value/rule, extrema tie distribution, autograd,
  optimizer, session, or executable training behavior. The capability baseline already inventories
  full and retained-dimension reduction min/max and distinguishes model/public construction from
  later executable support, so it required no status edit.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests, backend-
  conformance tests, and integration tests remain accurate unchanged because the work stays inside
  model-owned Tensor expression metadata and changes no module boundary, dependency, lifecycle,
  backend behavior, or end-to-end execution. Java 26 Gradle configuration remains accurate
  unchanged because no build, dependency, preview, incubator, or toolchain behavior changed.
- `DataType`, `Dimension`, `StaticDimension`, `Shape`, `TensorDescriptor`, `TensorFactory`,
  `TensorProvenance`, `Operation`, `NoOperationAttrs`, `AggregateReductionKind`,
  `AxisReductionAttrs`, and `BinaryArithmeticKind` contracts and Javadocs remain accurate
  unchanged. The helper composes their existing floating, immutable-Shape, unresolved-descriptor,
  central-ID, typed-operation, attribute, and provenance contracts without changing them.

## Implementation notes

- Added exactly six fluent full/axis/retained-axis reduction `min` and `max` methods as direct
  delegations to the existing bounded numeric-reduction helper.
- Extended helper kind validation to exactly SUM, MEAN, PROD, MIN, and MAX with the specified
  message while preserving all common construction paths and helper structure.
- Extended focused tests for every shared family contract and typed aggregate-versus-binary
  distinction. After independent review, restored the prior explicit SUM/PROD freshness/nesting
  assertions alongside the new MIN/MAX assertions.
- Finalized Tensor API, Compile API, glossary, evidence, model master plan, and roadmap without
  changing the already-complete Javadocs or adding executable behavior.

## Completion summary

- Completed changes: Implemented and documented floating full and one-axis minimum and maximum
  Tensor expression construction with exact reduced shapes and typed one-input provenance.
- Files changed or created: Exactly two production Java files, two tests, Tensor API, Compile API,
  glossary, this task, model master plan, and roadmap.
- Tests and validation: Focused suites 9/9 and 14/14, all 382 model tests across 50 suites, model
  Javadoc, root tests, bytecode/reflection/import/source checks, 242 Markdown link/anchor checks,
  fence/terminology/whitespace checks, exact scope/status checks, and `git diff --check` passed.
  The one restricted Gradle-lock denial and approved passing rerun are recorded above.
- Documentation-agent review: Clean context `/root/review_model_0016c_docs` completed the
  independent pass using General, API/Javadoc, Planning, and Example profiles and caught the
  focused regression-coverage gap before closure.
- Documentation impact: Tensor API, Compile API, and glossary now describe current numeric
  aggregate min/max construction and its distinction from binary elementwise min/max. Training
  API, capabilities, architecture/ADRs/tests, conformance/integration tests, and build configuration
  remain accurate unchanged for the recorded reasons.
- Javadoc review: New Tensor and affected helper Javadocs are complete unchanged; related
  foundational, operation, descriptor, factory, provenance, binary, and reduction contracts remain
  accurate.
- Glossary impact: Synchronized existing aggregate-reduction, normalized-axis, Tensor, and
  operation-family entries; no new reusable term was necessary.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0016C. Task 0016D remains Draft without a detailed
  specification.

Status: Complete
