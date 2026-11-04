# Task 0016B: Sum, [Mean](../../../../glossary.md#term-mean), and Product [Tensor](../../../../glossary.md#tensor) Expressions

**Editorial reading key (later context, not a status update).**

The application calls sum, mean or prod on a Tensor. For input shape [2, 3], reducing axis one removes it to produce [2], or retains extent one to produce [2, 1]; neither form constructs stored totals now. Full reduction instead describes a rank-zero scalar. The model helper normalizes the axis and records one input plus reduction attributes, without reading values or copying them. Model's descriptor work is distinct from Compiler derivative rules and backend accumulation; zero/dynamic metadata acceptance here does not settle empty-domain numerical policy.

**Reduction reading example.** In exact real arithmetic, let $x\in\mathbb{R}^{2\times3}$ and $s_i=\sum_{j=0}^{2}x_{i,j}$ for row index $i\in\{0,1\}$. For rows $(1,2,3)$ and $(4,5,6)$, $s=(6,15)$. Keeping the reduced axis represents these same totals with shape $[2,1]$ rather than $[2]$. This interprets the operation, not a recorded execution or a permission to reorder rounded sums.

Focused current reading: [Current numeric aggregate reference](../../../../reference/tensor/reductions.md#numeric-aggregate-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Expose the implemented aggregate semantics through public, [backend](../../../../glossary.md#backend)-independent floating [Tensor](../../../../glossary.md#tensor)
expressions for `sum`, [`mean`](../../../../glossary.md#term-mean), and `prod`. Each family supports full reduction, one [axis](../../../../glossary.md#term-axis) with that
axis removed, and one axis retained with [extent](../../../../glossary.md#term-extent) one. Every successful call returns a fresh
storage-free Tensor with a locally derived [Shape](../../../../glossary.md#shape), unchanged floating [data type](../../../../glossary.md#data-type--datatype) and [gradient](../../../../glossary.md#term-gradient)
eligibility, exact [operation](../../../../glossary.md#operation) attributes, and one-input [provenance](../../../../glossary.md#provenance).

This task constructs expression [metadata](../../../../glossary.md#term-metadata) only. It does not aggregate values, define numerical or
empty-domain behavior, create gradient rules, [capture](../../../../glossary.md#term-capture-meanings) a [graph](../../../../glossary.md#graph), or report backend support.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly nine public instance methods to [`Tensor`](../../../../glossary.md#tensor): full, [axis](../../../../glossary.md#term-axis), and axis-with-
  `keepDimensions` overloads for `sum`, [`mean`](../../../../glossary.md#term-mean), and `prod`.
- Add one package-private final `TensorReductionExpressions` helper with full and single-axis
  construction entries.
- Accept only [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), and [`BFLOAT16`](../../../../glossary.md#term-bfloat16); reject [`INT32`](../../../../glossary.md#data-type--datatype), [`INT64`](../../../../glossary.md#data-type--datatype), and [`BOOL`](../../../../glossary.md#data-type--datatype) without
  conversion.
- Normalize each positive or negative caller axis exactly once through `Shape.normalizeAxis`.
- Remove the [normalized axis](../../../../glossary.md#normalized-axis) when `keepDimensions` is false. When true, replace only that
  dimension with `StaticDimension(1)`. Retain every unaffected [Dimension](../../../../glossary.md#dimension) reference.
- Use [canonical](../../../../glossary.md#term-canonical-form) [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `Shape.scalar()` for full reduction and [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) axis removal.
- Preserve exact input [DataType](../../../../glossary.md#data-type--datatype) and `requiresGrad` for all three kinds.
- Leave [layout](../../../../glossary.md#layout) unresolved and attach no label or [host storage](../../../../glossary.md#host-storage).
- Use `NoOperationAttrs.INSTANCE` for full forms and a new
  `AxisReductionAttrs(normalizedAxis, keepDimensions)` for axis forms.
- Record exact one-input [provenance](../../../../glossary.md#provenance) and delegate [identity](../../../../glossary.md#term-identity-meanings) allocation exactly once to
  `TensorFactory.createDerived`.
- Update the exact Tensor reflection test and add one focused numeric-reduction test.
- Finalize Javadocs, Tensor API, Compile API, glossary, task evidence, master plan, and roadmap
  through the required independent documentation pass.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- reduction `min`/`max`, `all`/`any`, `argMax`, masked reductions, `cumSum`, softmax, loss
  reduction, pooling, normalization, or another operation family<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax); [loss reduction glossary](../../../../glossary.md#loss-reduction); [pooling glossary](../../../../glossary.md#term-pooling); [normalization glossary](../../../../glossary.md#term-normalization); [operation glossary](../../../../glossary.md#operation))</span>
- multiple axes, an axis collection, empty axis selection, full-reduction retained dimensions,
  named axes, or a reduction-options object<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [dimensions glossary](../../../../glossary.md#dimension))</span>
- value or storage access, aggregation, allocation, copying, materialization, aliasing, mutation,
  or output storage<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- accumulation order, parallelism, vectorization, FAST/Kahan/Neumaier modes, intermediate
  precision, overflow, underflow, or product ordering<span data-glossary-navigation="true"> ([precision glossary](../../../../glossary.md#term-precision); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow))</span>
- empty-domain [identities](../../../../glossary.md#term-identity-meanings)/errors, [mean](../../../../glossary.md#term-mean) division by zero, [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), or [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) policy
- integral/BOOL input, implicit cast, promotion, accumulation/output dtype override, or backend
  capability lookup<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [dtype glossary](../../../../glossary.md#data-type--datatype); [backend capability glossary](../../../../glossary.md#backend-capability); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- [gradient](../../../../glossary.md#term-gradient) values or rules, zero-product policy, [autograd](../../../../glossary.md#autograd), [optimizer](../../../../glossary.md#term-optimizer), or training execution
- changes to reduction semantics, Operation foundations, Shape/Dimension, descriptor, provenance,
  factory, or another existing contract<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- [aliases](../../../../glossary.md#aliasing-and-arena), public helpers/builders, static [Tensor](../../../../glossary.md#tensor) forms, labels, or overloads beyond the exact nine
- graph capture, compiler inference/optimization, planning, prepare, runtime, backend, tracing,
  ONNX, dependencies, Gradle, architecture, another module, or a task-0016C specification<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [ONNX glossary](../../../../glossary.md#term-onnx); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

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
- [Task 0016A](0016a-reduction-semantic-kinds-and-attributes.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes exactly three fluent forms per family: full
reduction, one [axis](../../../../glossary.md#term-axis) with removal, and one axis with explicit retained-[dimension](../../../../glossary.md#dimension) choice. Axes may be
negative. Legacy retains floating input type, supports strided inputs, and installs [Tensor](../../../../glossary.md#tensor)-local
[gradient](../../../../glossary.md#term-gradient) callbacks for sum and [mean](../../../../glossary.md#term-mean) while product has no [callback](../../../../glossary.md#term-callback). Tests cover full/axis/retained
forms, retained-result [broadcasting](../../../../glossary.md#broadcasting), gradients, [ONNX](../../../../glossary.md#term-onnx), and [CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) routes.

The new [model](../../../../glossary.md#term-model) deliberately uses [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [scalar](../../../../glossary.md#term-scalar) full results instead of legacy `[1]`. Product
preserves a true input `requiresGrad` request because this is eligibility [metadata](../../../../glossary.md#term-metadata), not proof of an
installed rule. Legacy `-1` all-axis sentinels, [operation](../../../../glossary.md#operation) traits, accumulation configuration,
callbacks, storage, [lowering](../../../../glossary.md#lowering), and [kernels](../../../../glossary.md#kernel) are not copied into model construction.

## Architecture constraints

- [`Tensor`](../../../../glossary.md#tensor) remains public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state, not [IR](../../../../glossary.md#term-intermediate-representation-ir) or an executable value.
- The helper performs local validation and Shape derivation only; it never reads values/storage,
  traverses provenance, captures a graph, chooses accumulation, or queries backends.<span data-glossary-navigation="true"> ([captures glossary](../../../../glossary.md#term-capture-meanings))</span><span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [graph glossary](../../../../glossary.md#graph); [backends glossary](../../../../glossary.md#backend))</span>
- Full forms pair with `NoOperationAttrs.INSTANCE`; axis forms normalize against exact input Shape
  before creating `AxisReductionAttrs`. Attributes never receive a negative [axis](../../../../glossary.md#term-axis).
- Shape derivation creates no symbolic constraint, broadcast plan, stride, layout, materialization,
  or compiler state.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [materialization glossary](../../../../glossary.md#materialization); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Unaffected [Dimension](../../../../glossary.md#dimension) references are retained. A retained selected axis becomes a new
  `StaticDimension(1)`.
- Full output and [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) removal use [canonical](../../../../glossary.md#term-canonical-form) `Shape.scalar()`. Axis reduction of a scalar is
  invalid through existing Shape behavior.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Static zero extents and dynamic dimensions remain structurally valid. No empty-domain numerical
  result is defined.
- Result type and gradient eligibility exactly match the floating input. Eligibility does not
  promise a compiled gradient rule, including for product.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
- Every result is fresh, unlabeled, storage-free, unresolved-[layout](../../../../glossary.md#layout), and allocated only through
  `TensorFactory.createDerived` with exact one-input [provenance](../../../../glossary.md#provenance).
- Package direction is `model.tensor -> model.operation.reduction`, operation, datatype, and shape.
  The reduction package must not import Tensor.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Stop if another type/file, existing-contract change, numerical policy, storage access, gradient
  rule, graph behavior, dependency, or architecture decision is required.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `model.tensor` owns the public surface, local Shape/descriptor construction, provenance, and
  derived factory seam.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- `model.operation` supplies [`Operation`](../../../../glossary.md#operation), [`OperationAttrs`](../../../../glossary.md#operationattrs), and [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs).
- `model.operation.reduction` supplies aggregate kinds and [axis](../../../../glossary.md#term-axis) attributes.
- `model.datatype` supplies floating-category [validation](../../../../glossary.md#term-validation).
- `model.shape` supplies [Shape](../../../../glossary.md#shape), [Dimension](../../../../glossary.md#dimension), StaticDimension, [scalar](../../../../glossary.md#term-scalar) creation, and axis [normalization](../../../../glossary.md#term-normalization).

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — exact public fluent overloads.
- `io.github.pho001.synaptik.model.tensor.TensorReductionExpressions` — package-private local
  validation, Shape derivation, and construction boundary.
- `TensorNumericReductionTest` — same-package focused helper and public-behavior test.

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) surface

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

Zero-[argument](../../../../glossary.md#term-argument) methods delegate once to `applyFull`. [Axis](../../../../glossary.md#term-axis)-only methods delegate once to `applyAxis`
with false and do not call another public overload. Two-argument methods delegate once with exact
caller values. All are public instance, non-static, and non-synchronized.

### Helper [shape](../../../../glossary.md#shape)

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

Add no overload, public/protected [API](../../../../glossary.md#term-application-programming-interface-api), [cache](../../../../glossary.md#term-cache), [registry](../../../../glossary.md#term-registry), strategy, service, or test hook.

### [Validation](../../../../glossary.md#term-validation) order

Both entries perform:

1. `Objects.requireNonNull(input, "input")`.
2. `Objects.requireNonNull(kind, "kind")`.
3. Accept exactly `SUM`, [`MEAN`](../../../../glossary.md#term-mean), or `PROD`; otherwise throw `IllegalArgumentException` with exact
   message `kind must be SUM, MEAN, or PROD, but was <kind>`.
4. Require floating input; otherwise throw `IllegalArgumentException` with exact message
   `input must be a floating data type, but was <dataType>`.

`applyAxis` then calls `inputShape.normalizeAxis(axis)` exactly once. Existing [Shape](../../../../glossary.md#shape) exception type
and message remain unchanged, so kind/type failures precede [axis](../../../../glossary.md#term-axis) failure. Pre-factory failures
consume no [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings).

### Full reduction

After [validation](../../../../glossary.md#term-validation), full forms construct [canonical](../../../../glossary.md#term-canonical-form) `Shape.scalar()`, an unresolved [descriptor](../../../../glossary.md#term-descriptor) with
exact input type and `requiresGrad`, exact `Operation(kind, NoOperationAttrs.INSTANCE)`, exact
one-input [provenance](../../../../glossary.md#provenance), and one `createDerived` call with empty label. [Scalar](../../../../glossary.md#term-scalar), static, zero-[element](../../../../glossary.md#term-element),
and dynamic input [Shapes](../../../../glossary.md#shape) are accepted without element-count inspection. A scalar full reduction is
fresh, not the input.

### [Axis](../../../../glossary.md#term-axis) [Shape](../../../../glossary.md#shape) derivation

For removal, allocate one `Dimension[]` of [rank](../../../../glossary.md#term-rank) minus one, copy every nonselected [Dimension](../../../../glossary.md#dimension) in
order by exact reference, and call `Shape.ofDimensions` once. Rank one becomes [canonical](../../../../glossary.md#term-canonical-form) [scalar](../../../../glossary.md#term-scalar).

For retention, copy dimensions to one same-rank array, replace exactly the selected entry with
`new StaticDimension(1)`, preserve every other reference, and call `Shape.ofDimensions` once.

Do not use `toLongArray`, require static Shape, inspect [element](../../../../glossary.md#term-element) count, mutate the dimension list,
or create a public Shape utility. Construct one `AxisReductionAttrs` from the [normalized axis](../../../../glossary.md#normalized-axis) and
exact retention flag.

### Common construction

`create` receives validated values and constructs in order: one unresolved [descriptor](../../../../glossary.md#term-descriptor), one
[Operation](../../../../glossary.md#operation) with exact kind/attributes reference, one [provenance](../../../../glossary.md#provenance) with `List.of(input)`, and one
`TensorFactory.createDerived` call with empty label. It performs no additional semantic [validation](../../../../glossary.md#term-validation).

All floating types preserve exact type. Non-floating types fail without cast. All three operations
preserve exact input `requiresGrad`; no [gradient](../../../../glossary.md#term-gradient) rule is created. Repeated and nested calls are
fresh and never simplified.

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

Review without modification unless inconsistent: Training [API](../../../../glossary.md#term-application-programming-interface-api), capabilities, related foundational
and semantic Javadocs/tests, focused architecture, ADRs/tests, [conformance](../../../../glossary.md#term-conformance)/integration tests, and
[Gradle](../../../../glossary.md#term-gradle) configuration.

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, two tests, and six documentation/[planning](../../../../glossary.md#term-planning) files: ten paths total.
[Tensor](../../../../glossary.md#tensor) and TensorTest may change only for the exact nine methods, imports/Javadocs, reflection
expectations, [signatures](../../../../glossary.md#term-signature), and non-[synchronization](../../../../glossary.md#term-synchronization) assertions. Stop beyond this scope or if another
type, [Shape](../../../../glossary.md#shape) change, numerical policy, storage access, [graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler) behavior, or [gradient](../../../../glossary.md#term-gradient) rule is
needed. Do not create task 0016C.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Update [Tensor](../../../../glossary.md#tensor) type Javadoc only as needed for numeric aggregate expression construction.
- Document every public method with full/axis scope, floating eligibility, normalization,
  result Shape, same type, unresolved layout, unchanged eligibility, freshness, storage absence,
  provenance, deferred numerics/gradients/execution, all parameters, return, and failures.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [scope glossary](../../../../glossary.md#term-scope); [normalization glossary](../../../../glossary.md#term-normalization); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [gradients glossary](../../../../glossary.md#term-gradient); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- Explain [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) full/[rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) results and retained [extent](../../../../glossary.md#term-extent) one.
- Document helper type, constructor, entries, and private methods with validation/construction
  order, ownership, Shape references, ID effects, and failures.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Explain acceptance of zero/dynamic extents without numerical policy and why product preserves
  eligibility without adding a rule.
- Review related Javadocs and record reasoned no-change conclusions or stop.

## Acceptance criteria

- Exactly nine public methods and the exact helper surface are added; no other [API](../../../../glossary.md#term-application-programming-interface-api).
- Every public method is one exact delegation with correct kind/full/[axis](../../../../glossary.md#term-axis) mapping.
- Exact null/kind/type/axis [validation](../../../../glossary.md#term-validation) order and messages hold.
- All three floating types and kinds succeed; other kinds/types fail without conversion.
- Full results are [canonical](../../../../glossary.md#term-canonical-form) [scalar](../../../../glossary.md#term-scalar) for scalar/static/zero/dynamic input.
- Axis results normalize once, remove/retain correctly, and preserve unaffected static/dynamic
  Dimension references.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension))</span>
- Results have exact type, unchanged eligibility, empty layout/label/storage, fresh identity,
  exact attributes and one-input provenance.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
- Inputs remain unchanged; no aggregation, numerical mode, gradient rule, graph/backend/build or
  architecture behavior is added.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend))</span>
- Focused/aggregate tests, Javadoc, root tests, reflection/javap/import/bytecode/scope and docs
  validation pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>
- A separate clean-context docs agent finalizes permitted Javadocs/APIs/glossary/planning and
  records no-change conclusions.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- 0016B becomes Complete only after both passes; 0016C remains Draft without a specification.

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

Focused tests cover exact helper/public [shapes](../../../../glossary.md#shape) and [bytecode](../../../../glossary.md#term-bytecode) delegation; all kind/type/eligibility
combinations; full [scalar](../../../../glossary.md#term-scalar)/static/zero/dynamic Shapes; positive/negative and invalid [axes](../../../../glossary.md#term-axis); remove,
retain, [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) scalar, static/dynamic reference retention; [descriptor](../../../../glossary.md#term-descriptor)/attributes/[provenance](../../../../glossary.md#provenance);
freshness and nesting; null/kind/type/axis precedence and ID effects; and unchanged input [metadata](../../../../glossary.md#term-metadata),
Shape, provenance, storage, and contents.

Manually inspect `javap -p -c -s`, reflection, imports, and source for exact descriptors, helper
shape, one delegation, one axis [normalization](../../../../glossary.md#term-normalization), structural Shape derivation, attributes, one
createDerived call, and absence of numeric/storage access, sentinel, multi-axis, [element](../../../../glossary.md#term-element)-count,
accumulation mode, cast, [gradient](../../../../glossary.md#term-gradient), cross-layer type, [registry](../../../../glossary.md#term-registry)/service, [dependency](../../../../glossary.md#term-dependency), or build change.
Validate generated [Javadoc](../../../../glossary.md#term-javadoc), APIs/glossary, links/anchors/fences/whitespace, exact ten paths,
statuses, and absence of task 0016C.

## Dependencies

- 0001 supplies floating DataTypes.
- 0002 supplies [Dimensions](../../../../glossary.md#dimension), [scalar](../../../../glossary.md#term-scalar) [Shape](../../../../glossary.md#shape), Shape creation, and [axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization).
- 0006 supplies [Operation](../../../../glossary.md#operation); 0007 supplies TensorDescriptor.
- 0011–0013 supply [Tensor](../../../../glossary.md#tensor), centralized [identity](../../../../glossary.md#term-identity-meanings) allocation, [provenance](../../../../glossary.md#provenance), and createDerived.
- 0016A supplies aggregate kinds, axis attributes, and full-form pairing.

## Follow-up tasks

- 0016C remains Draft for floating reduction min/max.
- 0016D remains Draft for [BOOL](../../../../glossary.md#data-type--datatype) all/any.
- 0016E remains Draft for arg-max and [INT64](../../../../glossary.md#data-type--datatype) results.
- 0016F remains Draft for masked sum/mean; detailed planning decides composition versus a
  dedicated semantic form.<span data-glossary-navigation="true"> ([mean glossary](../../../../glossary.md#term-mean); [planning glossary](../../../../glossary.md#term-planning))</span>
- Compiler tasks own capture/canonicalization/autograd; backend/config/conformance tasks own
  accumulation accuracy, empty domains, parallelism, lowering, storage, and kernels.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [autograd glossary](../../../../glossary.md#autograd); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel))</span>

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. This task composes [model](../../../../glossary.md#term-model)-owned [Tensor](../../../../glossary.md#tensor), [operation](../../../../glossary.md#operation), [descriptor](../../../../glossary.md#term-descriptor), [Shape](../../../../glossary.md#shape), and
[provenance](../../../../glossary.md#provenance) contracts without executable or cross-layer state. Stop if architecture change is
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

- Keep the legacy-compatible fluent names and three overload forms; use descriptive [parameter](../../../../glossary.md#term-parameter) name
  `keepDimensions` while preserving the `(int, boolean)` [signature](../../../../glossary.md#term-signature).
- Full results use [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [scalar](../../../../glossary.md#term-scalar), deliberately replacing legacy `[1]`.
- Caller [axes](../../../../glossary.md#term-axis) may be negative, but attributes store only normalized non-negative values.
- Shape derivation stays in the package-private helper; one consumer does not justify a new public
  Shape API.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Zero and dynamic extents are structurally accepted; numerical behavior is deferred.
- All kinds preserve input type and eligibility. Product eligibility is metadata, not a promise of
  an implemented gradient.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Layout stays unresolved and one helper is shared because all three local contracts are equal;
  kind validation prevents accidental use by later reduction families.<span data-glossary-navigation="true"> ([Layout glossary](../../../../glossary.md#layout); [validation glossary](../../../../glossary.md#term-validation))</span>

## Known limitations

- No values/storage, only semantics and [provenance](../../../../glossary.md#provenance).
- Floating input and full/one-[axis](../../../../glossary.md#term-axis) forms only.
- Empty-domain numerics, accumulation accuracy, and [gradient](../../../../glossary.md#term-gradient) [graphs](../../../../glossary.md#graph) are unimplemented.
- No [masked reduction](../../../../glossary.md#masked-reduction), [capture](../../../../glossary.md#term-capture-meanings), [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), [ONNX](../../../../glossary.md#term-onnx)/[backend](../../../../glossary.md#backend) support, or execution.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed architecture and focused docs; documentation/planning rules; roadmap; [model](../../../../glossary.md#term-model)
capabilities/master plan; tasks 0001, 0002, 0006, 0007, 0011, 0012, 0013, 0016A; current model
contracts/helpers/tests; APIs/glossary; and Java 26 [Gradle](../../../../glossary.md#term-gradle) configuration.

The legacy branch was read directly and confirms the nine overload forms, floating input, negative
[axes](../../../../glossary.md#term-axis), [Shape](../../../../glossary.md#shape) behavior, type retention, strided input, retained-result [broadcasting](../../../../glossary.md#broadcasting), sum/[mean](../../../../glossary.md#term-mean)
callbacks, product without [callback](../../../../glossary.md#term-callback), accumulation configuration, [ONNX](../../../../glossary.md#term-onnx), and [backend](../../../../glossary.md#backend) evidence.
Legacy `[1]`, `-1` sentinel, callbacks, traits, [runtime](../../../../glossary.md#term-runtime) configuration, storage, and execution are
excluded or reassigned.

Planning selected nine methods and one helper. Existing contracts suffice; no package, [dependency](../../../../glossary.md#term-dependency),
foundation, or architecture change is required.

Planning validation:

- `git diff --check` passed and targeted trailing-whitespace inspection found no matches in the
  three changed planning paths.
- The canonical section scan found every required task section, including exact API/helper shape,
  package impact, bounded scope, validation, handoff, decisions, limitations, and evidence.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [scan glossary](../../../../glossary.md#term-cumulative-scan); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [scope glossary](../../../../glossary.md#term-scope); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- Every local Markdown file linked from this task, the master plan, and roadmap resolves; fences
  are balanced.
- Status inspection found 0016B [`Ready`](../../../../glossary.md#term-ready) in this specification, its linked master-plan row, and its
  linked roadmap row/frontier text. Task 0016C remains `Draft`, and no task-0016C spec exists.
- Package review found no new package and preserves the one-way Tensor-to-reduction/shape
  dependency.
- Scope inspection found exactly this task, master plan, and roadmap changed. No Java, test, API,
  glossary, Gradle, architecture, AGENTS, or other-module path changed.<span data-glossary-navigation="true"> ([module path glossary](../../../../glossary.md#term-module-path))</span>

Implementation and independent documentation validation:

- The implementation pass changed exactly the authorized four Java/test paths: nine public
  [`Tensor`](../../../../glossary.md#tensor) methods, one final package-private helper, the exact Tensor reflection expectations,
  and one focused eight-test suite. A synthetic helper method caused by an array-constructor
  lambda was found and removed before final validation; the final helper has no synthetic method,
  field, nested type, cache, or test hook.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- Clean documentation context `/root/review_model_0016b_docs` independently reread the complete
  architecture contract; focused overview, lifecycle, boundary, and dependency explanations;
  documentation workflow and General, API/Javadoc, Planning, and Example profiles; planning guide
  and roadmap; model capabilities/master plan; tasks 0001, 0002, 0006, 0007, 0011, 0012, 0013,
  0016A, and 0016B; Tensor, Compile, and Training API references; glossary; actual final source,
  tests, generated Javadoc/reports, related foundational and reduction contracts, Java 26 Gradle
  configuration, and the complete workspace diff. It inspected behavior and artifacts rather than
  relying on the implementation handoff.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [lifecycle glossary](../../../../glossary.md#lifecycle); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Compile glossary](../../../../glossary.md#compile); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- The documentation pass found the new public `Tensor` methods and package-private
  `TensorReductionExpressions` type/member Javadocs complete without revision. They cover purpose,
  floating eligibility, full/axis/retained shape semantics, normalization and Dimension-reference
  behavior, exact descriptor/provenance ownership, freshness, null/kind/type/axis failures and
  ordering, identity effects, zero/dynamic extents, product eligibility, and deferred numerical,
  gradient, compiler, runtime, and backend behavior. Existing Javadocs for DataType, Dimension,
  StaticDimension, Shape, TensorDescriptor, TensorFactory, TensorProvenance, Operation,
  NoOperationAttrs, AggregateReductionKind, and AxisReductionAttrs remain accurate because this
  task composes their existing contracts without changing them.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [Dimension glossary](../../../../glossary.md#dimension); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [identity glossary](../../../../glossary.md#term-identity-meanings); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [DataType glossary](../../../../glossary.md#data-type--datatype); [Operation glossary](../../../../glossary.md#operation); [NoOperationAttrs glossary](../../../../glossary.md#nooperationattrs))</span>
- `docs/api/tensor-api.md` now documents the exact nine-method floating aggregate surface,
  full-versus-axis attributes, canonical scalar and structural axis shapes, unchanged type and
  gradient eligibility, unresolved storage-free results, one-input provenance, failure boundary,
  and deferred numerics/execution. Its complete newcomer-oriented dynamic-shape example compiled
  and ran against the built model classes, printing the documented scalar, removed-axis,
  retained-axis, normalized-attribute, exact-reference, and metadata results.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- `docs/api/compile-api.md` now includes current sum/mean/product expression construction while
  keeping capture, reduction inference/canonicalization, compile artifacts, and executable
  behavior planned. <span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>`docs/glossary.md` synchronizes aggregate reduction, normalized axis, Tensor,
  operation-family status, and Tensor-versus-graph distinctions with the implemented surface.<span data-glossary-navigation="true"> ([aggregate reduction glossary](../../../../glossary.md#aggregate-reduction); [normalized axis glossary](../../../../glossary.md#normalized-axis); [graph glossary](../../../../glossary.md#graph))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorNumericReductionTest` — `BUILD SUCCESSFUL`; 8 tests,
  zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; 14 tests, zero failures,
  errors, or skips.
- The first aggregate implementation attempt could not open the restricted Gradle-distribution
  cache lock and stopped before task execution. The approved rerun passed. The final
  post-documentation <span data-glossary-navigation="true"> ([distribution glossary](../../../../glossary.md#term-distribution))</span>`./gradlew :modules:model:test` also reported `BUILD SUCCESSFUL`; 50 XML
  suites contain 381 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated public Javadoc contains all
  nine Tensor overloads and their parameter, result, failure, ownership, shape, provenance, and
  deferred-behavior contracts. The package-private helper is intentionally absent from public
  generated pages and was reviewed completely in source.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>
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
  planning, runtime, prepare, backend, registry, service, dependency, or build behavior.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [element glossary](../../../../glossary.md#term-element); [prepare glossary](../../../../glossary.md#prepare); [registry glossary](../../../../glossary.md#term-registry))</span>
- The local Markdown target-and-heading checker resolved 236 links in the six changed
  documentation/planning files with zero errors. Markdown fences are balanced; terminology and
  current/planned status scans found no stale reduction-expression claim; targeted trailing-
  whitespace scans found no matches; and <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>`git diff --check` passed.
- Final scope is exactly ten paths: two production files, two tests, Tensor API, Compile API,
  glossary, this task, model master plan, and roadmap. Task 0016B is synchronized as Complete.
  Task 0016C remains Draft, and no task-0016C specification exists. No commit or push occurred.<span data-glossary-navigation="true"> ([commit glossary](../../../../glossary.md#term-commit))</span>
- `docs/api/training-api.md` remains accurate unchanged because no gradient rule, gradient value,
  autograd, optimizer, training-session, or executable training behavior changed. The model
  capability baseline already inventories sum/mean/product full, axis, and retained forms while
  distinguishing model/public construction from executable support, so it needs no status edit.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance tests, and integration tests remain accurate unchanged because the work
  stays inside model-owned Tensor expression metadata and changes no module boundary, dependency,
  lifecycle, backend behavior, or end-to-end execution. Java 26 Gradle configuration remains
  accurate unchanged because no build, dependency, preview, incubator, or toolchain behavior
  changed.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module); [toolchain glossary](../../../../glossary.md#term-toolchain))</span>

## Implementation notes

- Added the exact nine fluent full/[axis](../../../../glossary.md#term-axis)/retained-axis `sum`, [`mean`](../../../../glossary.md#term-mean), and `prod` methods as direct
  delegations to one bounded package-private helper.
- Added floating validation, canonical full scalar shape, structural one-axis removal/retention,
  exact type and eligibility retention, unresolved descriptors, full/axis attributes, and exact
  one-input provenance without value aggregation or cross-layer behavior.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [canonical glossary](../../../../glossary.md#term-canonical-form); [scalar glossary](../../../../glossary.md#term-scalar); [shape glossary](../../../../glossary.md#shape); [descriptors glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- Added focused tests for exact surface/bytecode, all types/kinds/eligibility states, shapes and
  Dimension identity, freshness and unchanged inputs, validation precedence/messages, ID effects,
  and identifier exhaustion.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [Dimension glossary](../../../../glossary.md#dimension); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Finalized Tensor API, Compile API, glossary, evidence, model master plan, and roadmap without
  changing the already-complete Javadocs or any executable behavior.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>

## Completion summary

- Completed changes: Implemented and documented floating full and one-axis sum, mean, and product
  Tensor expression construction with exact reduced shapes and provenance.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [mean glossary](../../../../glossary.md#term-mean); [Tensor glossary](../../../../glossary.md#tensor); [shapes glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance))</span>
- Files changed or created: Exactly two production Java files, two tests, Tensor API, Compile API,
  glossary, this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused suites 8/8 and 14/14, all 381 model tests across 50 suites, model
  Javadoc, root tests, compiled documentation example, bytecode/reflection/import/dependency
  checks, 236 Markdown link/anchor checks, fence/terminology/whitespace checks, exact scope/status
  checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed. The initial restricted-cache lock denial and approved
  passing rerun are recorded above.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- Documentation-agent review: Clean context `/root/review_model_0016b_docs` completed the
  independent pass using General, API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and glossary now describe the current sum/mean/product surface;
  Compile API includes it as current expression input without claiming compiler behavior. Training
  API, capabilities, architecture/ADRs/tests, conformance/integration tests, and build
  configuration remain accurate unchanged for the recorded reasons.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Javadoc review: New Tensor and helper Javadocs are complete unchanged; related foundational,
  descriptor, factory, provenance, operation, and reduction contracts remain accurate.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation))</span>
- Glossary impact: Aggregate reduction, normalized axis, Tensor status, operation-family status,
  and Tensor-versus-graph distinctions now include current numeric aggregate expressions.<span data-glossary-navigation="true"> ([Aggregate reduction glossary](../../../../glossary.md#aggregate-reduction); [normalized axis glossary](../../../../glossary.md#normalized-axis); [graph glossary](../../../../glossary.md#graph))</span>
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0016B. Task 0016C remains Draft without a detailed
  specification.

Status: Complete
