# Task 0016C: Min and Max [Tensor](../../../../glossary.md#tensor) Reduction Expressions

**Editorial reading key (later context, not a status update).**

The zero-argument or axis min/max overload reduces a domain to one extreme value per remaining coordinate. It differs from pairwise min/max, which compares corresponding positions of two Tensors. The application supplies the receiver and optional axis/retention choice; the helper derives a descriptor and one-input provenance, not the selected numeric winner or a storage copy. Equal extreme values need not imply the same derivative allocation, so the forward-value and backward-tie questions remain distinct as recorded here.

Focused current reading: [Current aggregate extrema reference](../../../../reference/tensor/reductions.md#numeric-aggregate-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Expose the existing aggregate `MIN` and `MAX` semantics through public, [backend](../../../../glossary.md#backend)-independent
floating [Tensor](../../../../glossary.md#tensor) expressions. Each family supports full reduction, one [axis](../../../../glossary.md#term-axis) with that axis removed,
and one axis retained with [extent](../../../../glossary.md#term-extent) one. Every successful call returns a fresh storage-free Tensor
with a locally derived [Shape](../../../../glossary.md#shape), unchanged floating [data type](../../../../glossary.md#data-type--datatype) and [gradient](../../../../glossary.md#term-gradient) eligibility, exact
[operation](../../../../glossary.md#operation) attributes, and one-input [provenance](../../../../glossary.md#provenance).

This task extends the shared numeric-reduction construction boundary completed by task 0016B. It
constructs expression [metadata](../../../../glossary.md#term-metadata) only. It does not compare values, define numerical or empty-domain
behavior, create gradient rules, [capture](../../../../glossary.md#term-capture-meanings) a [graph](../../../../glossary.md#graph), or report backend support.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly six public instance methods to [`Tensor`](../../../../glossary.md#tensor): full, [axis](../../../../glossary.md#term-axis), and axis-with-
  `keepDimensions` overloads for reduction `min` and `max`.
- Extend the existing package-private final `TensorReductionExpressions` helper to accept
  `AggregateReductionKind.MIN` and `AggregateReductionKind.MAX` without changing its declared
  method, field, constructor, or nested-type surface.
- Accept only [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), and [`BFLOAT16`](../../../../glossary.md#term-bfloat16); reject [`INT32`](../../../../glossary.md#data-type--datatype), [`INT64`](../../../../glossary.md#data-type--datatype), and [`BOOL`](../../../../glossary.md#data-type--datatype) without
  conversion.
- Reuse the existing single `Shape.normalizeAxis` call and structural result-[shape](../../../../glossary.md#shape) derivation.
- Preserve exact input [DataType](../../../../glossary.md#data-type--datatype) and `requiresGrad` for both [extrema](../../../../glossary.md#saturation-pole-and-extremum) kinds.
- Leave [layout](../../../../glossary.md#layout) unresolved and attach no label or [host storage](../../../../glossary.md#host-storage).
- Use `NoOperationAttrs.INSTANCE` for full forms and
  `AxisReductionAttrs(normalizedAxis, keepDimensions)` for axis forms.
- Record exact one-input [provenance](../../../../glossary.md#provenance) and delegate [identity](../../../../glossary.md#term-identity-meanings) allocation exactly once to
  `TensorFactory.createDerived` through the existing helper.
- Extend the exact Tensor reflection test and existing focused numeric-reduction test rather than
  adding a second helper or duplicate test family.
- Finalize Javadocs, Tensor API, Compile API, glossary, task evidence, master plan, and roadmap
  through the required independent documentation pass.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) [elementwise](../../../../glossary.md#term-elementwise) `min(Tensor)` and `max(Tensor)` behavior or their
  `BinaryArithmeticKind.MIN` and `BinaryArithmeticKind.MAX` [identities](../../../../glossary.md#term-identity-meanings)
- `sum`, [`mean`](../../../../glossary.md#term-mean), or `prod` behavior changes; `all`/`any`, `argMax`, masked reductions, `cumSum`,
  softmax, loss reduction, pooling, normalization, or another operation family<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax); [loss reduction glossary](../../../../glossary.md#loss-reduction); [pooling glossary](../../../../glossary.md#term-pooling); [normalization glossary](../../../../glossary.md#term-normalization); [operation glossary](../../../../glossary.md#operation))</span>
- multiple axes, an axis collection, empty axis selection, full-reduction retained dimensions,
  named axes, or a reduction-options object<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [dimensions glossary](../../../../glossary.md#dimension))</span>
- value or storage access, comparison, aggregation, allocation, copying, materialization,
  aliasing, mutation, or output storage<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- minimum/maximum comparison rules for NaN, signed zero, infinities, equal values, or backend
  differences<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [signed zero glossary](../../../../glossary.md#term-signed-zero); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [backend glossary](../../../../glossary.md#backend))</span>
- empty-domain values/errors, initial values, reduction identities, or zero-[element](../../../../glossary.md#term-element) execution
- integral/[BOOL](../../../../glossary.md#data-type--datatype) input, implicit cast, promotion, output [dtype](../../../../glossary.md#data-type--datatype) override, or [backend capability](../../../../glossary.md#backend-capability) [lookup](../../../../glossary.md#term-lookup)
- gradient values or rules, extrema tie-gradient distribution, autograd, optimizer, or training
  execution<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [distribution glossary](../../../../glossary.md#term-distribution); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- changes to reduction semantic enums/attributes, Operation foundations, Shape/Dimension,
  descriptor, provenance, factory, or another existing contract<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- [aliases](../../../../glossary.md#aliasing-and-arena), public helpers/builders, static [Tensor](../../../../glossary.md#tensor) forms, labels, or overloads beyond the exact six
- graph capture, compiler inference/optimization, planning, prepare, runtime, backend, tracing,
  ONNX, dependencies, Gradle, architecture, another module, or a task-0016D specification<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [ONNX glossary](../../../../glossary.md#term-onnx); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
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
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes exactly three fluent reduction forms per [extrema](../../../../glossary.md#saturation-pole-and-extremum)
family: full reduction, one [axis](../../../../glossary.md#term-axis) with removal, and one axis with explicit retained-[dimension](../../../../glossary.md#dimension)
choice. Axes may be negative. Legacy accepts floating input, retains its [data type](../../../../glossary.md#data-type--datatype), supports
non-[contiguous storage](../../../../glossary.md#contiguous-request), and has [CPU](../../../../glossary.md#term-cpu) execution evidence. Its local backward callbacks distribute an
incoming [gradient](../../../../glossary.md#term-gradient) equally across values tied for the selected minimum or maximum.

The new [model](../../../../glossary.md#term-model) deliberately uses [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [scalar](../../../../glossary.md#term-scalar) full results instead of legacy `[1]`. It preserves
a true input `requiresGrad` request as eligibility [metadata](../../../../glossary.md#term-metadata) but does not copy the legacy [Tensor](../../../../glossary.md#tensor)-
local gradient [callback](../../../../glossary.md#term-callback) or define tie behavior. Legacy `-1` all-axis sentinels, [operation](../../../../glossary.md#operation) traits,
storage handling, numerical loops, [runtime](../../../../glossary.md#term-runtime) configuration, [lowering](../../../../glossary.md#lowering), and [kernels](../../../../glossary.md#kernel) are not copied
into model construction.

## Architecture constraints

- [`Tensor`](../../../../glossary.md#tensor) remains public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state, not [IR](../../../../glossary.md#term-intermediate-representation-ir) or an executable value.
- The existing helper performs local validation and Shape derivation only; it never reads
  values/storage, traverses provenance, captures a graph, chooses comparison behavior, or queries
  backends.<span data-glossary-navigation="true"> ([captures glossary](../../../../glossary.md#term-capture-meanings))</span><span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [graph glossary](../../../../glossary.md#graph); [backends glossary](../../../../glossary.md#backend))</span>
- Full forms pair with `NoOperationAttrs.INSTANCE`; axis forms normalize against the exact input
  Shape before creating `AxisReductionAttrs`. Attributes never receive a negative [axis](../../../../glossary.md#term-axis).
- Shape derivation creates no symbolic constraint, broadcast plan, stride, layout,
  materialization, or compiler state.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [materialization glossary](../../../../glossary.md#materialization); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Unaffected [Dimension](../../../../glossary.md#dimension) references are retained. A retained selected axis becomes a new
  `StaticDimension(1)`.
- Full output and [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) removal use [canonical](../../../../glossary.md#term-canonical-form) `Shape.scalar()`. Axis reduction of a scalar is
  invalid through existing Shape behavior.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Static zero extents and dynamic dimensions remain structurally valid. No empty-domain numerical
  result is defined.
- Result type and gradient eligibility exactly match the floating input. Eligibility does not
  promise an extrema gradient rule or any tie-distribution policy.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [distribution glossary](../../../../glossary.md#term-distribution))</span>
- Every result is fresh, unlabeled, storage-free, unresolved-[layout](../../../../glossary.md#layout), and allocated only through
  `TensorFactory.createDerived` with exact one-input [provenance](../../../../glossary.md#provenance).
- Reduction `AggregateReductionKind.MIN/MAX` remain distinct from [elementwise](../../../../glossary.md#term-elementwise)
  `BinaryArithmeticKind.MIN/MAX`; method overloads select the semantic family by their [signature](../../../../glossary.md#term-signature).
- Package direction remains `model.tensor -> model.operation.reduction`, operation, datatype, and
  shape. The reduction package must not import Tensor.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Stop if another type/file, existing-contract change, numerical policy, storage access, gradient
  rule, graph behavior, dependency, or architecture decision is required.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `model.tensor` owns the public surface, local Shape/descriptor construction, provenance, and
  derived factory seam.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- `model.operation` supplies [`Operation`](../../../../glossary.md#operation), [`OperationAttrs`](../../../../glossary.md#operationattrs), and [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs).
- `model.operation.reduction` supplies aggregate kinds and [axis](../../../../glossary.md#term-axis) attributes.
- `model.operation.elementwise.binary` remains the separate identity family for existing binary
  elementwise min/max and is not modified.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [elementwise glossary](../../../../glossary.md#term-elementwise))</span>
- `model.datatype` supplies floating-category [validation](../../../../glossary.md#term-validation).
- `model.shape` supplies [Shape](../../../../glossary.md#shape), [Dimension](../../../../glossary.md#dimension), StaticDimension, [scalar](../../../../glossary.md#term-scalar) creation, and axis [normalization](../../../../glossary.md#term-normalization).

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — exact public fluent overloads alongside the
  existing numeric aggregate methods.
- `io.github.pho001.synaptik.model.tensor.TensorReductionExpressions` — existing package-private
  local validation, Shape derivation, and construction boundary.
- `TensorNumericReductionTest` — existing same-package focused helper and public-behavior test.

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) surface

Add exactly:

```java
public Tensor min()
public Tensor min(int axis)
public Tensor min(int axis, boolean keepDimensions)
public Tensor max()
public Tensor max(int axis)
public Tensor max(int axis, boolean keepDimensions)
```

Zero-[argument](../../../../glossary.md#term-argument) methods delegate once to `TensorReductionExpressions.applyFull`. [Axis](../../../../glossary.md#term-axis)-only methods
delegate once to `applyAxis` with false and do not call another public overload. Two-argument
methods delegate once with exact caller values. All are public instance, non-static, and non-
synchronized.

The overloads coexist with existing `min(Tensor)` and `max(Tensor)` [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) expressions without
changing them. The number of declared public Tensor methods becomes exactly 61. The existing set
of public method names does not gain a name because `min` and `max` already exist.

### Existing helper [shape](../../../../glossary.md#shape)

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

Add no method, overload, field, nested type, public/protected [API](../../../../glossary.md#term-application-programming-interface-api), [cache](../../../../glossary.md#term-cache), [registry](../../../../glossary.md#term-registry), strategy,
service, or test hook. Preserve the existing implementation paths for `SUM`, [`MEAN`](../../../../glossary.md#term-mean), and `PROD`.

### [Validation](../../../../glossary.md#term-validation) order

Both helper entries continue to perform:

1. `Objects.requireNonNull(input, "input")`.
2. `Objects.requireNonNull(kind, "kind")`.
3. Accept exactly `SUM`, [`MEAN`](../../../../glossary.md#term-mean), `PROD`, `MIN`, or `MAX`; otherwise throw
   `IllegalArgumentException` with exact message
   `kind must be SUM, MEAN, PROD, MIN, or MAX, but was <kind>`.
4. Require floating input; otherwise throw `IllegalArgumentException` with exact message
   `input must be a floating data type, but was <dataType>`.

`applyAxis` then calls `inputShape.normalizeAxis(axis)` exactly once. Existing [Shape](../../../../glossary.md#shape) exception type
and message remain unchanged, so kind/type failures precede [axis](../../../../glossary.md#term-axis) failure. Pre-factory failures
consume no [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings).

The unsupported-kind tests must now reject exactly `ALL`, `ANY`, and `ARG_MAX`. Existing
`SUM`/`MEAN`/`PROD` behavior must remain unchanged.

### Full [extrema](../../../../glossary.md#saturation-pole-and-extremum) reduction

After [validation](../../../../glossary.md#term-validation), full forms construct [canonical](../../../../glossary.md#term-canonical-form) `Shape.scalar()`, an unresolved [descriptor](../../../../glossary.md#term-descriptor) with
exact input type and `requiresGrad`, exact `Operation(kind, NoOperationAttrs.INSTANCE)`, exact one-
input [provenance](../../../../glossary.md#provenance), and one `createDerived` call with empty label. [Scalar](../../../../glossary.md#term-scalar), static, zero-[element](../../../../glossary.md#term-element), and
dynamic input [Shapes](../../../../glossary.md#shape) are accepted without element-count inspection. A scalar full reduction is
fresh, not the input.

### [Axis](../../../../glossary.md#term-axis) [Shape](../../../../glossary.md#shape) derivation

Reuse the existing `reduceShape` method unchanged unless a [Javadoc](../../../../glossary.md#term-javadoc)-only wording update is required.
For removal, it allocates one `Dimension[]` of [rank](../../../../glossary.md#term-rank) minus one, copies every nonselected [Dimension](../../../../glossary.md#dimension)
in order by exact reference, and calls `Shape.ofDimensions` once. Rank one becomes [canonical](../../../../glossary.md#term-canonical-form)
[scalar](../../../../glossary.md#term-scalar).

For retention, it copies dimensions to one same-rank array, replaces exactly the selected entry
with `new StaticDimension(1)`, preserves every other reference, and calls `Shape.ofDimensions`
once.

Do not use `toLongArray`, require static Shape, inspect [element](../../../../glossary.md#term-element) count, mutate the dimension list,
or create a public Shape utility. Construct one `AxisReductionAttrs` from the [normalized axis](../../../../glossary.md#normalized-axis) and
exact retention flag.

### Common construction and semantic distinction

Reuse the existing `create` method. It receives validated values and constructs in order: one
unresolved [descriptor](../../../../glossary.md#term-descriptor), one [Operation](../../../../glossary.md#operation) with exact kind/attributes reference, one [provenance](../../../../glossary.md#provenance) with
`List.of(input)`, and one `TensorFactory.createDerived` call with empty label. It performs no
additional semantic [validation](../../../../glossary.md#term-validation).

All floating types preserve exact type. Non-floating types fail without cast. Both [extrema](../../../../glossary.md#saturation-pole-and-extremum)
operations preserve exact input `requiresGrad`; no [gradient](../../../../glossary.md#term-gradient) rule is created. Repeated and nested
calls are fresh and never simplified.

For example, `left.min(right)` records `BinaryArithmeticKind.MIN` and two ordered provenance
inputs, while `left.min()` or `left.min(axis)` records `AggregateReductionKind.MIN` and one input.
Tests must demonstrate that these [identities](../../../../glossary.md#term-identity-meanings) remain distinct without executing either operation.

The input [Tensor](../../../../glossary.md#tensor), descriptor, [Shape](../../../../glossary.md#shape), [dimensions](../../../../glossary.md#dimension), label, provenance, storage association, and
contents remain unchanged. Output storage is always absent.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorReductionExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorNumericReductionTest.java`

Documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistent: Training [API](../../../../glossary.md#term-application-programming-interface-api), capabilities, related foundational,
[binary](../../../../glossary.md#term-binary-descriptor-and-metadata)-arithmetic, and reduction Javadocs/tests, focused architecture, ADRs/tests, [conformance](../../../../glossary.md#term-conformance)/
integration tests, and [Gradle](../../../../glossary.md#term-gradle) configuration.

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, two tests, and six documentation/[planning](../../../../glossary.md#term-planning) files: ten paths total.
[Tensor](../../../../glossary.md#tensor) and TensorTest may change only for the exact six overloads, imports/Javadocs, reflection
expectations, [signatures](../../../../glossary.md#term-signature), and non-[synchronization](../../../../glossary.md#term-synchronization) assertions. TensorNumericReductionTest may be
extended in place but must retain coverage of all task-0016B behavior. Stop beyond this scope or if
another type, helper, test file, [Shape](../../../../glossary.md#shape) change, numerical policy, storage access, [graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler)
behavior, or [gradient](../../../../glossary.md#term-gradient) rule is needed. Do not create task 0016D.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Update [Tensor](../../../../glossary.md#tensor) type Javadoc only as needed for numeric aggregate expression construction.
- Document every new public method with full/axis scope, floating eligibility, normalization,
  result Shape, same type, unresolved layout, unchanged eligibility, freshness, storage absence,
  provenance, deferred comparison/empty-domain/gradient/execution behavior, all parameters,
  return, and failures.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [scope glossary](../../../../glossary.md#term-scope); [normalization glossary](../../../../glossary.md#term-normalization); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- Explain [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) full/[rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) results and retained [extent](../../../../glossary.md#term-extent) one.
- Update helper type, entries, and `validateKind` Javadocs to cover all five supported kinds and
  exact validation order/message. Review the constructor and unchanged private methods.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Explain acceptance of zero/dynamic extents without numerical policy and why preserving
  eligibility does not add an extrema gradient or tie-distribution rule.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [distribution glossary](../../../../glossary.md#term-distribution))</span>
- Clearly distinguish [aggregate reduction](../../../../glossary.md#aggregate-reduction) `MIN/MAX` from [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) [elementwise](../../../../glossary.md#term-elementwise) `MIN/MAX`.
- Review related Javadocs and record reasoned no-change conclusions or stop.

## Acceptance criteria

- Exactly six public overloads are added, Tensor has exactly 61 declared public methods, and the
  helper retains its exact existing six-method surface with no additional state or type.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Every public method is one exact delegation with correct reduction kind/full/[axis](../../../../glossary.md#term-axis) mapping.
- Existing [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) `min(Tensor)`/`max(Tensor)` and sum/mean/prod methods remain unchanged in
  behavior and typed semantic identity.<span data-glossary-navigation="true"> ([mean glossary](../../../../glossary.md#term-mean); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Exact null/kind/type/axis [validation](../../../../glossary.md#term-validation) order and updated messages hold.
- All three floating types and both extrema kinds succeed; other kinds/types fail without
  conversion.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- Full results are [canonical](../../../../glossary.md#term-canonical-form) [scalar](../../../../glossary.md#term-scalar) for scalar/static/zero/dynamic input.
- Axis results normalize once, remove/retain correctly, and preserve unaffected static/dynamic
  Dimension references.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension))</span>
- Results have exact type, unchanged eligibility, empty layout/label/storage, fresh identity,
  exact attributes and one-input provenance.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>
- Aggregate and binary MIN/MAX are demonstrably distinct typed [operation](../../../../glossary.md#operation) families.
- Inputs remain unchanged; no comparison, numerical mode, tie policy, gradient rule, graph/backend/
  build or architecture behavior is added.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend))</span>
- Focused/aggregate tests, Javadoc, root tests, reflection/javap/import/bytecode/scope and docs
  validation pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>
- A separate clean-context docs agent finalizes permitted Javadocs/APIs/glossary/planning and
  records no-change conclusions.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- 0016C becomes Complete only after both passes; 0016D remains Draft without a specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorNumericReductionTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Extend the existing focused tests to cover exact helper/public [shapes](../../../../glossary.md#shape) and [bytecode](../../../../glossary.md#term-bytecode) delegation; all
five supported kinds without weakening sum/[mean](../../../../glossary.md#term-mean)/prod coverage; both [extrema](../../../../glossary.md#saturation-pole-and-extremum) kinds across all
floating types and eligibility states; full [scalar](../../../../glossary.md#term-scalar)/static/zero/dynamic Shapes; positive/negative
and invalid [axes](../../../../glossary.md#term-axis); removal, retention, [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) scalar, static/dynamic reference retention;
[descriptor](../../../../glossary.md#term-descriptor)/attributes/[provenance](../../../../glossary.md#provenance); aggregate-versus-[binary](../../../../glossary.md#term-binary-descriptor-and-metadata) typed [identity](../../../../glossary.md#term-identity-meanings); freshness and nesting;
null/kind/type/axis precedence and ID effects; and unchanged input [metadata](../../../../glossary.md#term-metadata), Shape, provenance,
storage, and contents.

Manually inspect `javap -p -c -s`, reflection, imports, and source for exact overload descriptors,
unchanged helper shape, one delegation per new overload, one axis [normalization](../../../../glossary.md#term-normalization), reused structural
Shape derivation, attributes, one createDerived call, and absence of numeric/storage access,
sentinel, multi-axis, [element](../../../../glossary.md#term-element)-count, comparison/[NaN](../../../../glossary.md#term-nan)/tie policy, cast, [gradient](../../../../glossary.md#term-gradient), cross-layer type,
[registry](../../../../glossary.md#term-registry)/service, [dependency](../../../../glossary.md#term-dependency), or build change. Validate generated [Javadoc](../../../../glossary.md#term-javadoc), APIs/glossary, links/
anchors/fences/whitespace, exact ten paths, synchronized statuses, and absence of task 0016D.

## Dependencies

- 0001 supplies floating DataTypes.
- 0002 supplies [Dimensions](../../../../glossary.md#dimension), [scalar](../../../../glossary.md#term-scalar) [Shape](../../../../glossary.md#shape), Shape creation, and [axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization).
- 0006 supplies [Operation](../../../../glossary.md#operation); 0007 supplies TensorDescriptor.
- 0011–0013 supply [Tensor](../../../../glossary.md#tensor), centralized [identity](../../../../glossary.md#term-identity-meanings) allocation, [provenance](../../../../glossary.md#provenance), and createDerived.
- 0014A–0014B supply the distinct binary elementwise MIN/MAX identity and Tensor overloads that
  must remain unchanged.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [elementwise glossary](../../../../glossary.md#term-elementwise))</span>
- 0016A supplies aggregate MIN/MAX kinds, axis attributes, and full-form pairing.
- 0016B supplies the shared numeric-reduction helper and tested full/axis construction behavior.

## Follow-up tasks

- 0016D remains Draft for [BOOL](../../../../glossary.md#data-type--datatype) all/any.
- 0016E remains Draft for arg-max and [INT64](../../../../glossary.md#data-type--datatype) results.
- 0016F remains Draft for masked sum/mean; detailed planning decides composition versus a
  dedicated semantic form.<span data-glossary-navigation="true"> ([mean glossary](../../../../glossary.md#term-mean); [planning glossary](../../../../glossary.md#term-planning))</span>
- Compiler tasks own capture/canonicalization/autograd, including extrema tie-gradient semantics;
  backend/config/conformance tasks own comparison and NaN behavior, empty domains, lowering,
  storage, kernels, and cross-backend numerical guarantees.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [autograd glossary](../../../../glossary.md#autograd); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [NaN glossary](../../../../glossary.md#term-nan); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel))</span>

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. This task composes [model](../../../../glossary.md#term-model)-owned [Tensor](../../../../glossary.md#tensor), [operation](../../../../glossary.md#operation), [descriptor](../../../../glossary.md#term-descriptor), [Shape](../../../../glossary.md#shape), and
[provenance](../../../../glossary.md#provenance) contracts without executable or cross-layer state. Stop if architecture change is
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
  helper would duplicate invariants and create divergent validation risk.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- Keep the legacy-compatible fluent names and three overload forms; `min()`/`max()` are reduction
  overloads while `min(Tensor)`/`max(Tensor)` remain [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) [elementwise](../../../../glossary.md#term-elementwise) expressions.
- Full results use [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [scalar](../../../../glossary.md#term-scalar), deliberately replacing legacy `[1]`.
- Caller [axes](../../../../glossary.md#term-axis) may be negative, but attributes store only normalized non-negative values.
- Zero and dynamic extents are structurally accepted; numerical behavior is deferred.
- Both kinds preserve input type and eligibility. Eligibility is metadata, not an implemented
  gradient or promise that equal extrema will receive any particular gradient distribution.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [gradient glossary](../../../../glossary.md#term-gradient); [distribution glossary](../../../../glossary.md#term-distribution))</span>
- Forward equality/ties need no model attribute because the selected reduced value is identical;
  comparison/NaN/signed-zero and backward tie behavior remain explicit later-layer contracts.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>

## Known limitations

- No values/storage, only semantics and [provenance](../../../../glossary.md#provenance).
- Floating input and full/one-[axis](../../../../glossary.md#term-axis) forms only.
- Empty-domain, [NaN](../../../../glossary.md#term-nan), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), comparison, and tie-[gradient](../../../../glossary.md#term-gradient) behavior are unimplemented.
- No [masked reduction](../../../../glossary.md#masked-reduction), [capture](../../../../glossary.md#term-capture-meanings), [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), [ONNX](../../../../glossary.md#term-onnx)/[backend](../../../../glossary.md#backend) support, or execution.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed the [architecture contract](../../../../glossary.md#architecture-contract) and focused architecture documentation;
documentation/planning rules; roadmap; [model](../../../../glossary.md#term-model) capabilities/master plan; completed tasks 0001, 0002,
0006, 0007, 0011, 0012, 0013, 0014A, 0014B, 0016A, and 0016B; current [Tensor](../../../../glossary.md#tensor), [operation](../../../../glossary.md#operation),
reduction, [descriptor](../../../../glossary.md#term-descriptor), [Shape](../../../../glossary.md#shape), [provenance](../../../../glossary.md#provenance), helper, and focused test contracts; Tensor/[Compile](../../../../glossary.md#compile)/
Training APIs; glossary; and Java 26 [Gradle](../../../../glossary.md#term-gradle) configuration.

The legacy branch was read directly. It confirms full, [axis](../../../../glossary.md#term-axis)-removing, and retained-axis min/max;
negative axes; floating type retention; non-[contiguous](../../../../glossary.md#term-contiguous) execution; and equal tie splitting in its
Tensor-local backward callbacks. Legacy `[1]`, negative all-axis sentinel, callbacks, traits,
storage, [runtime](../../../../glossary.md#term-runtime) configuration, [lowering](../../../../glossary.md#lowering), and [kernels](../../../../glossary.md#kernel) are excluded or reassigned.

Planning selected six overloads and an in-place helper/test extension. Existing contracts suffice;
no package, production/test file, [dependency](../../../../glossary.md#term-dependency), foundation, or architecture change is required.

Planning validation:

- `git diff --check` passed. A targeted trailing-whitespace scan found no matches in the three
  changed planning paths, including this untracked task file.
- The canonical section scan found every required task section. The specification contains exact
  public/helper shapes, validation messages and order, package placement, ten-path limit,
  acceptance criteria, validation, clean-context implementation/documentation handoff, decisions,
  limitations, evidence, and completion placeholders.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- The local Markdown target checker resolved 158 links across this task, the model master plan,
  and roadmap with zero missing files. Fence validation found balanced Markdown fences in all
  three paths.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Status inspection found task 0016C [`Ready`](../../../../glossary.md#term-ready) in this specification, its linked master-plan row,
  roadmap frontier, and roadmap row. Task 0016D remains `Draft`, and no task-0016D specification
  exists.
- Package/scope review found no new package and exactly three planning paths changed. No Java,
  test, API, glossary, Gradle, architecture, AGENTS, or other-module path changed.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [module path glossary](../../../../glossary.md#term-module-path))</span>

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
  evidence directly rather than relying on the implementation handoff.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [module glossary](../../../../glossary.md#term-module); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- The independent review found one test-regression gap before closure: the first implementation
  pass had replaced task-0016B's explicit repeated <span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression))</span>`sum` and nested `prod` freshness coverage with
  extrema-only coverage. A separate constrained implementation turn in <span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>`/root/implement_model_0010`
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
  aggregate family accurately.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [layout glossary](../../../../glossary.md#layout); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [identity glossary](../../../../glossary.md#term-identity-meanings); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- `docs/api/tensor-api.md` now documents the fifteen-method numeric aggregate surface, adds MIN and
  MAX to the full/axis/retained table, explains canonical scalar and structural single-axis shapes,
  exact type/eligibility/provenance, and distinguishes one-input aggregate MIN/MAX from two-input
  binary elementwise MIN/MAX. Its existing complete numeric-aggregate example was not changed: it
  continues to demonstrate the common helper's normalization, Shape, attributes, and provenance
  contract for all five families, so no new example compile/run was required.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [elementwise glossary](../../../../glossary.md#term-elementwise); [run glossary](../../../../glossary.md#term-run))</span>
- `docs/api/compile-api.md` now includes current aggregate min/max expression inputs without
  claiming capture, reduction inference/canonicalization, autograd, artifacts, backend ownership,
  or execution. <span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [autograd glossary](../../../../glossary.md#autograd); [artifacts glossary](../../../../glossary.md#term-artifact); [backend ownership glossary](../../../../glossary.md#backend-ownership))</span>`docs/glossary.md` synchronizes aggregate-reduction, normalized-axis, Tensor, and
  operation-family status and records the typed aggregate-versus-binary extrema distinction; no
  new reusable glossary term was introduced.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorNumericReductionTest --rerun-tasks` — `BUILD
  SUCCESSFUL`; the XML report contains 9 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest --rerun-tasks` — the first documentation-pass
  attempt could not open the restricted Gradle distribution lock and stopped before task
  execution. The approved rerun reported <span data-glossary-navigation="true"> ([distribution glossary](../../../../glossary.md#term-distribution))</span>`BUILD SUCCESSFUL`; the XML report contains 14 tests with
  zero failures, errors, or skips.
- `./gradlew :modules:model:test --rerun-tasks` — `BUILD SUCCESSFUL`; 50 XML suites contain 382
  tests with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc --rerun-tasks` — `BUILD SUCCESSFUL`. Generated public Javadoc
  contains all six overloads with rendered parameter, result, failure, shape, provenance, and
  deferred-behavior contracts. The package-private helper is intentionally absent from public
  generated pages and was reviewed completely in source.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>
- `./gradlew test --rerun-tasks` — `BUILD SUCCESSFUL`; the repository lifecycle completed 36
  actionable tasks with no failing task.
- `javap -p -c -s` confirmed the exact six public overload descriptors and one direct helper
  delegation per overload, including false in both axis-only methods and exact aggregate kinds.
  It also confirmed the final helper's private zero-argument constructor, exactly two package-
  private entries and four private methods, exact validation order, five accepted kinds, one axis
  normalization, structural removal/retention, exact attributes, one-input provenance, and one<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>
  `createDerived` call. Verbose [bytecode](../../../../glossary.md#term-bytecode) inspection found no synthetic helper member.
- Reflection tests confirmed 61 declared public Tensor methods, exact overload modifiers and
  signatures, helper final/package-private shape, zero fields/nested types, and six declared
  nonsynthetic methods. Focused behavior covers all five kinds, all three floating types and both
  eligibility states, full scalar/static/zero/dynamic inputs, positive/negative/invalid axes,
  removal/retention and Dimension identity, exact descriptor/attributes/provenance, aggregate-
  versus-binary identity, freshness/nesting, input immutability, validation precedence/messages,
  ID effects, and exhaustion.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- Production import/source/bytecode inspection found only the permitted tensor, datatype,
  operation, reduction, shape, and JDK contracts. It found no value/storage access, element-count
  or static-shape demand, comparison loop, cast, negative all-axis sentinel, multiple-axis API,
  gradient rule, graph/compiler/planning/runtime/prepare/backend type, registry, service,
  dependency, or build behavior. Existing binary min/max and sum/mean/prod methods are unchanged.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [element glossary](../../../../glossary.md#term-element); [graph glossary](../../../../glossary.md#graph); [prepare glossary](../../../../glossary.md#prepare); [registry glossary](../../../../glossary.md#term-registry); [mean glossary](../../../../glossary.md#term-mean))</span>
- The local Markdown target-and-heading checker resolved all 242 links in the six changed
  documentation/planning paths with zero errors. Markdown fences are balanced; terminology and
  current/planned status scans found no stale aggregate-expression claim; targeted whitespace
  inspection and `git diff --check` passed.
- Final scope is exactly the authorized ten paths: two production files, two tests, Tensor API,
  Compile API, glossary, this task, model master plan, and roadmap. Task 0016C is synchronized as
  Complete. Task 0016D remains Draft, and no task-0016D specification exists. No commit or push
  occurred.<span data-glossary-navigation="true"> ([commit glossary](../../../../glossary.md#term-commit))</span>
- `docs/api/training-api.md` remains accurate unchanged because this task preserves only an
  eligibility request and adds no gradient value/rule, extrema tie distribution, autograd,
  optimizer, session, or executable training behavior. The capability baseline already inventories
  full and retained-dimension reduction min/max and distinguishes model/public construction from
  later executable support, so it required no status edit.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests, backend-
  conformance tests, and integration tests remain accurate unchanged because the work stays inside
  model-owned Tensor expression metadata and changes no module boundary, dependency, lifecycle,
  backend behavior, or end-to-end execution. Java 26 Gradle configuration remains accurate
  unchanged because no build, dependency, preview, incubator, or toolchain behavior changed.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [metadata glossary](../../../../glossary.md#term-metadata); [toolchain glossary](../../../../glossary.md#term-toolchain))</span>
- [`DataType`](../../../../glossary.md#data-type--datatype), [`Dimension`](../../../../glossary.md#dimension), `StaticDimension`, `Shape`, `TensorDescriptor`, `TensorFactory`,
  `TensorProvenance`, `Operation`, [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), `AggregateReductionKind`,
  `AxisReductionAttrs`, and `BinaryArithmeticKind` contracts and Javadocs remain accurate
  unchanged. The helper composes their existing floating, immutable-Shape, unresolved-descriptor,
  central-ID, typed-operation, attribute, and provenance contracts without changing them.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>

## Implementation notes

- Added exactly six fluent full/[axis](../../../../glossary.md#term-axis)/retained-axis reduction `min` and `max` methods as direct
  delegations to the existing bounded numeric-reduction helper.
- Extended helper kind validation to exactly SUM, MEAN, PROD, MIN, and MAX with the specified
  message while preserving all common construction paths and helper structure.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [MEAN glossary](../../../../glossary.md#term-mean))</span>
- Extended focused tests for every shared family contract and typed aggregate-versus-binary
  distinction. After independent review, restored the prior explicit SUM/PROD freshness/nesting
  assertions alongside the new MIN/MAX assertions.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- Finalized Tensor API, Compile API, glossary, evidence, model master plan, and roadmap without
  changing the already-complete Javadocs or adding executable behavior.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>

## Completion summary

- Completed changes: Implemented and documented floating full and one-axis minimum and maximum
  Tensor expression construction with exact reduced shapes and typed one-input provenance.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [Tensor glossary](../../../../glossary.md#tensor); [shapes glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance))</span>
- Files changed or created: Exactly two production Java files, two tests, Tensor API, Compile API,
  glossary, this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused suites 9/9 and 14/14, all 382 model tests across 50 suites, model
  Javadoc, root tests, bytecode/reflection/import/source checks, 242 Markdown link/anchor checks,
  fence/terminology/whitespace checks, exact scope/status checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
  The one restricted Gradle-lock denial and approved passing rerun are recorded above.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Documentation-agent review: Clean context `/root/review_model_0016c_docs` completed the
  independent pass using General, API/Javadoc, Planning, and Example profiles and caught the
  focused regression-coverage gap before closure.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [regression glossary](../../../../glossary.md#term-regression))</span>
- Documentation impact: Tensor API, Compile API, and glossary now describe current numeric
  aggregate min/max construction and its distinction from binary elementwise min/max. Training
  API, capabilities, architecture/ADRs/tests, conformance/integration tests, and build configuration
  remain accurate unchanged for the recorded reasons.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [elementwise glossary](../../../../glossary.md#term-elementwise); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Javadoc review: New Tensor and affected helper Javadocs are complete unchanged; related
  foundational, operation, descriptor, factory, provenance, binary, and reduction contracts remain
  accurate.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Glossary impact: Synchronized existing aggregate-reduction, normalized-axis, Tensor, and
  operation-family entries; no new reusable term was necessary.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0016C. Task 0016D remains Draft without a detailed
  specification.

Status: Complete
