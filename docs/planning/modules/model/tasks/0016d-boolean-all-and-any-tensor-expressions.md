# Task 0016D: [Boolean](../../../../glossary.md#data-type--datatype) All and Any [Tensor](../../../../glossary.md#tensor) Expressions

**Editorial reading key (later context, not a status update).**

The application calls all/any to describe combining an entire truth domain, not elementwise logicalAnd/logicalOr with a second Tensor. Axis removal/retention changes result shape in the same way as other aggregates, but the result remains BOOL and cannot request gradients. The model helper reads the input descriptor, not truth bytes, and allocates only expression metadata. A structurally accepted zero extent does not itself decide the truth value of an empty conjunction or disjunction in this historical record.

Focused current reading: [Current Boolean-reduction reference](../../../../reference/tensor/masked-index-reductions.md#boolean-aggregate-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Expose the existing aggregate `ALL` and `ANY` semantics through public, [backend](../../../../glossary.md#backend)-independent [BOOL](../../../../glossary.md#data-type--datatype)
[Tensor](../../../../glossary.md#tensor) expressions. Each family supports full reduction, one [axis](../../../../glossary.md#term-axis) with that axis removed, and one
axis retained with [extent](../../../../glossary.md#term-extent) one. Every successful call returns a fresh storage-free,
non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) BOOL Tensor with a locally derived [Shape](../../../../glossary.md#shape), exact [operation](../../../../glossary.md#operation) attributes, and one-
input [provenance](../../../../glossary.md#provenance).

This task extends the shared aggregate-reduction construction boundary completed by tasks 0016B
and 0016C. It constructs expression [metadata](../../../../glossary.md#term-metadata) only. It does not inspect truth values, define the
logical [identity](../../../../glossary.md#term-identity-meanings) of an empty reduction domain, create [gradient](../../../../glossary.md#term-gradient) rules, [capture](../../../../glossary.md#term-capture-meanings) a [graph](../../../../glossary.md#graph), or report
backend support.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly six public instance methods to [`Tensor`](../../../../glossary.md#tensor): full, [axis](../../../../glossary.md#term-axis), and axis-with-
  `keepDimensions` overloads for reduction `all` and `any`.
- Extend the existing package-private final `TensorReductionExpressions` helper to accept
  `AggregateReductionKind.ALL` and `AggregateReductionKind.ANY` without adding a field, method,
  constructor, nested type, or second helper.
- Replace the helper's floating-only private validator with one kind-aware private validator while
  preserving the exact total six-method helper surface.
- Continue accepting only [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), and [`BFLOAT16`](../../../../glossary.md#term-bfloat16) for `SUM`, [`MEAN`](../../../../glossary.md#term-mean), `PROD`, `MIN`,
  and `MAX`, with their existing failure behavior.
- Accept exactly `DataType.BOOL` for `ALL` and `ANY`; reject every floating and integral type
  without numeric truthiness or conversion.
- Reuse the existing single `Shape.normalizeAxis` call and structural result-[shape](../../../../glossary.md#shape) derivation.
- Produce exact [BOOL](../../../../glossary.md#data-type--datatype) results with `requiresGrad=false`, unresolved layout, no label, and no host
  storage.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [host storage glossary](../../../../glossary.md#host-storage))</span>
- Use `NoOperationAttrs.INSTANCE` for full forms and
  `AxisReductionAttrs(normalizedAxis, keepDimensions)` for axis forms.
- Record exact one-input [provenance](../../../../glossary.md#provenance) and delegate [identity](../../../../glossary.md#term-identity-meanings) allocation exactly once to
  `TensorFactory.createDerived` through the existing helper.
- Update the exact Tensor reflection test, update the existing numeric-reduction test only for the
  intentionally generalized helper contract, and add one focused boolean-reduction test.<span data-glossary-navigation="true"> ([boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- Finalize Javadocs, Tensor API, Compile API, glossary, task evidence, master plan, and roadmap
  through the required independent documentation pass.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- [elementwise](../../../../glossary.md#term-elementwise) `logicalAnd`, `logicalOr`, and `logicalNot` behavior or their
  `BooleanLogicalKind.AND`, `OR`, and `NOT` [identities](../../../../glossary.md#term-identity-meanings)
- changes to `sum`, [`mean`](../../../../glossary.md#term-mean), `prod`, reduction `min`, or reduction `max` public behavior, results,
  validation, provenance, or typed identities<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [provenance glossary](../../../../glossary.md#provenance))</span>
- `argMax`, masked reductions, `cumSum`, softmax, loss reduction, pooling, normalization, or
  another operation family<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax); [loss reduction glossary](../../../../glossary.md#loss-reduction); [pooling glossary](../../../../glossary.md#term-pooling); [normalization glossary](../../../../glossary.md#term-normalization); [operation glossary](../../../../glossary.md#operation))</span>
- multiple axes, an axis collection, empty axis selection, full-reduction retained dimensions,
  named axes, or a reduction-options object<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [dimensions glossary](../../../../glossary.md#dimension))</span>
- truth-value or storage access, short-circuiting, aggregation, allocation, copying,
  materialization, aliasing, mutation, or output storage<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- the empty-domain identities normally associated with conjunction/disjunction, errors for empty
  domains, initial values, or zero-element execution<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
- floating/integral truthiness, raw non-zero byte interpretation, nullable values, three-valued
  logic, implicit cast, promotion, or backend capability lookup<span data-glossary-navigation="true"> ([backend capability glossary](../../../../glossary.md#backend-capability); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- gradient eligibility propagation, gradient values/rules, straight-through estimators, autograd,
  optimizer, or training execution<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- changes to reduction semantic enums/attributes, boolean logical enums, Operation foundations,
  Shape/Dimension, descriptor, provenance, factory, or another existing contract<span data-glossary-navigation="true"> ([boolean glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- [aliases](../../../../glossary.md#aliasing-and-arena), public helpers/builders, static [Tensor](../../../../glossary.md#tensor) forms, labels, or overloads beyond the exact six
- graph capture, compiler inference/optimization, planning, prepare, runtime, backend, tracing,
  ONNX, dependencies, Gradle, architecture, another module, or a task-0016E specification<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [ONNX glossary](../../../../glossary.md#term-onnx); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

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
- [Task 0015C](0015c-boolean-logical-semantic-kinds.md)
- [Task 0015D](0015d-boolean-logical-tensor-expressions.md)
- [Task 0016A](0016a-reduction-semantic-kinds-and-attributes.md)
- [Task 0016B](0016b-sum-mean-and-product-tensor-expressions.md)
- [Task 0016C](0016c-min-and-max-tensor-reduction-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes exactly three fluent reduction forms per [boolean](../../../../glossary.md#data-type--datatype)
family: full reduction, one [axis](../../../../glossary.md#term-axis) with removal, and one axis with explicit retained-[dimension](../../../../glossary.md#dimension)
choice. Axes may be negative. Legacy requires [BOOL](../../../../glossary.md#data-type--datatype) input, always produces a non-[gradient](../../../../glossary.md#term-gradient) BOOL
result, supports strided storage, and has [CPU](../../../../glossary.md#term-cpu) execution evidence for full and axis forms.

The new [model](../../../../glossary.md#term-model) deliberately uses [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [scalar](../../../../glossary.md#term-scalar) full results instead of legacy `[1]`. Legacy
`-1` all-axis sentinels, [operation](../../../../glossary.md#operation) traits, raw-byte truth interpretation, short-circuit loops,
storage handling, [runtime](../../../../glossary.md#term-runtime) [graph](../../../../glossary.md#graph) state, [lowering](../../../../glossary.md#lowering), and [kernels](../../../../glossary.md#kernel) are not copied into model
construction. The legacy [kernel](../../../../glossary.md#kernel) reads the first [element](../../../../glossary.md#term-element) and therefore does not establish a safe
empty-domain contract; this task leaves empty-domain logical [identities](../../../../glossary.md#term-identity-meanings) to later executable and
[conformance](../../../../glossary.md#term-conformance) contracts.

## Architecture constraints

- [`Tensor`](../../../../glossary.md#tensor) remains public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state, not [IR](../../../../glossary.md#term-intermediate-representation-ir) or an executable value.
- The existing helper performs local kind/type validation and Shape derivation only; it never
  reads values/storage, traverses provenance, captures a graph, chooses truth behavior, or queries
  backends.<span data-glossary-navigation="true"> ([captures glossary](../../../../glossary.md#term-capture-meanings))</span><span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [graph glossary](../../../../glossary.md#graph); [backends glossary](../../../../glossary.md#backend))</span>
- Full forms pair with `NoOperationAttrs.INSTANCE`; axis forms normalize against the exact input
  Shape before creating `AxisReductionAttrs`. Attributes never receive a negative [axis](../../../../glossary.md#term-axis).
- Shape derivation creates no symbolic constraint, broadcast plan, stride, layout,
  materialization, or compiler state.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [materialization glossary](../../../../glossary.md#materialization); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Unaffected [Dimension](../../../../glossary.md#dimension) references are retained. A retained selected axis becomes a new
  `StaticDimension(1)`.
- Full output and [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) removal use [canonical](../../../../glossary.md#term-canonical-form) `Shape.scalar()`. Axis reduction of a scalar is
  invalid through existing Shape behavior.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Static zero extents and dynamic dimensions remain structurally valid. No logical empty-domain
  result is defined by model construction.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Numeric aggregate kinds retain their existing floating-only validation, exact type, and input
  gradient eligibility. BOOL <span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>`ALL/ANY` require exact BOOL and consequently produce exact BOOL with
  false gradient eligibility through the existing descriptor contract.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Every result is fresh, unlabeled, storage-free, unresolved-[layout](../../../../glossary.md#layout), and allocated only through
  `TensorFactory.createDerived` with exact one-input [provenance](../../../../glossary.md#provenance).
- Aggregate `ALL/ANY` remain distinct from [elementwise](../../../../glossary.md#term-elementwise) `BooleanLogicalKind.AND/OR`; method names,
  operation kinds, arity, and provenance distinguish the semantic families.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [arity glossary](../../../../glossary.md#term-arity))</span>
- Package direction remains `model.tensor -> model.operation.reduction`, operation, datatype, and
  shape. Neither reduction nor boolean-logical operation packages import Tensor.<span data-glossary-navigation="true"> ([boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- Stop if another production helper/type, existing public-contract change, empty-domain policy,
  storage access, gradient rule, graph behavior, dependency, or architecture decision is needed.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `model.tensor` owns the public surface, local type/Shape/descriptor construction, provenance,
  and derived factory seam.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- `model.operation` supplies [`Operation`](../../../../glossary.md#operation), [`OperationAttrs`](../../../../glossary.md#operationattrs), and [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs).
- `model.operation.reduction` supplies aggregate kinds and [axis](../../../../glossary.md#term-axis) attributes.
- `model.operation.elementwise.logical` remains the separate identity family for elementwise
  AND/OR/NOT and is not modified.<span data-glossary-navigation="true"> ([elementwise glossary](../../../../glossary.md#term-elementwise))</span>
- `model.datatype` supplies exact [BOOL](../../../../glossary.md#data-type--datatype) [identity](../../../../glossary.md#term-identity-meanings) and floating-category [validation](../../../../glossary.md#term-validation).
- `model.shape` supplies [Shape](../../../../glossary.md#shape), [Dimension](../../../../glossary.md#dimension), StaticDimension, [scalar](../../../../glossary.md#term-scalar) creation, and axis [normalization](../../../../glossary.md#term-normalization).

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — exact public fluent boolean-reduction
  overloads alongside the existing aggregate methods.<span data-glossary-navigation="true"> ([boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- `io.github.pho001.synaptik.model.tensor.TensorReductionExpressions` — existing package-private
  kind/type validation, Shape derivation, and construction boundary shared by ordinary aggregate
  families.
- `TensorNumericReductionTest` — existing same-package regression test for numeric helper behavior
  and the generalized private helper shape.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression))</span>
- `TensorBooleanReductionTest` — new same-package focused BOOL/helper/public-behavior test.

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) surface

Add exactly:

```java
public Tensor all()
public Tensor all(int axis)
public Tensor all(int axis, boolean keepDimensions)
public Tensor any()
public Tensor any(int axis)
public Tensor any(int axis, boolean keepDimensions)
```

Zero-[argument](../../../../glossary.md#term-argument) methods delegate once to `TensorReductionExpressions.applyFull`. [Axis](../../../../glossary.md#term-axis)-only methods
delegate once to `applyAxis` with false and do not call another public overload. Two-argument
methods delegate once with exact caller values. All are public instance, non-static, and non-
synchronized.

The number of declared public Tensor methods becomes exactly 67. The public method-name set gains
exactly `all` and `any`. Existing numeric aggregate and [elementwise](../../../../glossary.md#term-elementwise) logical methods remain
unchanged.

### Generalized existing helper [shape](../../../../glossary.md#shape)

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
`validateKind`. Add no other method, overload, field, nested type, public/protected [API](../../../../glossary.md#term-application-programming-interface-api), [cache](../../../../glossary.md#term-cache),
[registry](../../../../glossary.md#term-registry), strategy, service, second helper, or test hook. Preserve the existing implementation
paths for `SUM`, [`MEAN`](../../../../glossary.md#term-mean), `PROD`, `MIN`, and `MAX`.

### [Validation](../../../../glossary.md#term-validation) order

Both helper entries perform:

1. `Objects.requireNonNull(input, "input")`.
2. `Objects.requireNonNull(kind, "kind")`.
3. Accept exactly `SUM`, [`MEAN`](../../../../glossary.md#term-mean), `PROD`, `MIN`, `MAX`, `ALL`, or `ANY`; otherwise throw
   `IllegalArgumentException` with exact message
   `kind must be SUM, MEAN, PROD, MIN, MAX, ALL, or ANY, but was <kind>`.
4. Call `validateInput(input, kind)` exactly once.

For `ALL` or `ANY`, `validateInput` requires `input.descriptor().dataType() == DataType.BOOL`.
Otherwise it throws `IllegalArgumentException` with exact message
`input must have BOOL data type for <kind>, but was <dataType>`.

For `SUM`, `MEAN`, `PROD`, `MIN`, or `MAX`, `validateInput` preserves the completed numeric
contract: require `dataType.isFloating()`, otherwise throw `IllegalArgumentException` with exact
message `input must be a floating data type, but was <dataType>`.

`ARG_MAX` fails kind validation before input type or [axis](../../../../glossary.md#term-axis) inspection. `applyAxis` normalizes the
axis exactly once only after kind and type succeed. Existing [Shape](../../../../glossary.md#shape) exception types/messages remain
unchanged. Pre-factory failures consume no [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings).

### Full [boolean](../../../../glossary.md#data-type--datatype) reduction

After [validation](../../../../glossary.md#term-validation), full forms reuse common construction with [canonical](../../../../glossary.md#term-canonical-form) `Shape.scalar()`, exact
`Operation(kind, NoOperationAttrs.INSTANCE)`, exact one-input [provenance](../../../../glossary.md#provenance), and one createDerived
call with empty label. [Scalar](../../../../glossary.md#term-scalar), static, zero-[element](../../../../glossary.md#term-element), and dynamic input [Shapes](../../../../glossary.md#shape) are accepted without
element-count or truth-value inspection. A scalar full reduction is fresh, not the input.

The result [descriptor](../../../../glossary.md#term-descriptor) is exact [BOOL](../../../../glossary.md#data-type--datatype), `Optional.empty()` [layout](../../../../glossary.md#layout), and `requiresGrad=false`. These
facts follow from exact BOOL input plus the existing common construction and TensorDescriptor
contract; do not add a second result-construction branch.

### [Axis](../../../../glossary.md#term-axis) [Shape](../../../../glossary.md#shape) derivation

Reuse the existing `reduceShape` implementation unchanged unless a [Javadoc](../../../../glossary.md#term-javadoc)-only wording update is
required. Removal allocates one [rank](../../../../glossary.md#term-rank)-minus-one `Dimension[]`, retains every nonselected [Dimension](../../../../glossary.md#dimension)
reference in order, and calls `Shape.ofDimensions` once. Rank one becomes [canonical](../../../../glossary.md#term-canonical-form) [scalar](../../../../glossary.md#term-scalar).

Retention copies dimensions to one same-rank array, replaces exactly the selected entry with
`new StaticDimension(1)`, preserves every other reference, and calls `Shape.ofDimensions` once.

Do not require static Shape, inspect [element](../../../../glossary.md#term-element) count, read truth values, mutate the dimension list,
or create a public Shape utility. Construct one `AxisReductionAttrs` from the [normalized axis](../../../../glossary.md#normalized-axis) and
exact retention flag.

### Common construction and semantic distinction

Reuse the existing `create` method. It constructs in order: one unresolved [descriptor](../../../../glossary.md#term-descriptor) from exact
input type/eligibility and supplied [Shape](../../../../glossary.md#shape), one [Operation](../../../../glossary.md#operation) with exact kind/attributes reference, one
[provenance](../../../../glossary.md#provenance) with `List.of(input)`, and one `TensorFactory.createDerived` call with empty label. It
performs no additional semantic [validation](../../../../glossary.md#term-validation).

Repeated and nested calls are fresh and never simplified. For example,
`left.logicalAnd(right)` records `BooleanLogicalKind.AND` and two ordered provenance inputs, while
`left.all()` or `left.all(axis)` records `AggregateReductionKind.ALL` and one input. Tests must
demonstrate this typed [identity](../../../../glossary.md#term-identity-meanings) and [arity](../../../../glossary.md#term-arity) distinction without executing either operation.

The input [Tensor](../../../../glossary.md#tensor), descriptor, Shape, [dimensions](../../../../glossary.md#dimension), label, provenance, storage association, and
contents remain unchanged. Output storage is always absent.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorReductionExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorNumericReductionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBooleanReductionTest.java`

Documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistent: Training [API](../../../../glossary.md#term-application-programming-interface-api), capabilities, related foundational,
[boolean](../../../../glossary.md#data-type--datatype)-logical, and reduction Javadocs/tests, focused architecture, ADRs/tests, [conformance](../../../../glossary.md#term-conformance)/
integration tests, and [Gradle](../../../../glossary.md#term-gradle) configuration.

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, three tests, and six documentation/[planning](../../../../glossary.md#term-planning) files: eleven paths
total. The fifth Java/test path is required because the completed numeric test owns exact shared-
helper [regression](../../../../glossary.md#term-regression) checks while [BOOL](../../../../glossary.md#data-type--datatype) behavior needs its own focused suite. [Tensor](../../../../glossary.md#tensor) and TensorTest
may change only for the exact six overloads, imports/Javadocs, reflection expectations, [signatures](../../../../glossary.md#term-signature),
and non-[synchronization](../../../../glossary.md#term-synchronization) assertions. TensorNumericReductionTest may change only for the intentional
six-method helper [generalization](../../../../glossary.md#term-generalization), updated unsupported-kind contract, and preservation of every
numeric assertion. Stop beyond this scope or if another helper/type, [Shape](../../../../glossary.md#shape) change, empty-domain
policy, storage access, [graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler) behavior, or [gradient](../../../../glossary.md#term-gradient) rule is needed. Do not create task
0016E.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Update [Tensor](../../../../glossary.md#tensor) type Javadoc only as needed for [boolean](../../../../glossary.md#data-type--datatype) aggregate expression construction.
- Document every new public method with full/axis scope, exact BOOL eligibility, normalization,
  result Shape, BOOL/false-gradient result, unresolved layout, freshness, storage absence,
  provenance, deferred empty-domain/truth/gradient/execution behavior, all parameters, return, and
  failures.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [scope glossary](../../../../glossary.md#term-scope); [BOOL glossary](../../../../glossary.md#data-type--datatype); [normalization glossary](../../../../glossary.md#term-normalization); [Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- Explain [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) full/[rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) results and retained [extent](../../../../glossary.md#term-extent) one.
- Generalize helper type, entry, `validateKind`, `validateInput`, and common-construction Javadocs
  while preserving precise numeric behavior and documenting kind-specific type validation order,
  Shape references, identity effects, and failures.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Explain acceptance of zero/dynamic extents without claiming `all(empty)` or `any(empty)` values.
- Clearly distinguish aggregate `ALL/ANY` from [elementwise](../../../../glossary.md#term-elementwise) `AND/OR/NOT`.
- Review related Javadocs and record reasoned no-change conclusions or stop.

## Acceptance criteria

- Exactly six public overloads are added, Tensor has exactly 67 declared public methods, and the
  public name set gains only <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>`all` and `any`.
- The helper retains exactly six declared nonsynthetic methods and no state/nested type; only its
  private type validator is intentionally generalized.
- Every public method is one exact delegation with correct aggregate kind/full/[axis](../../../../glossary.md#term-axis) mapping.
- Existing numeric aggregate and [elementwise](../../../../glossary.md#term-elementwise) logical methods retain behavior and typed [identity](../../../../glossary.md#term-identity-meanings).
- Exact null/kind/type/axis [validation](../../../../glossary.md#term-validation) order and messages hold for numeric and [BOOL](../../../../glossary.md#data-type--datatype) families.
- Exact BOOL input succeeds for `ALL/ANY`; every floating/integral type fails without conversion.
- Full results are [canonical](../../../../glossary.md#term-canonical-form) [scalar](../../../../glossary.md#term-scalar) for scalar/static/zero/dynamic input.
- Axis results normalize once, remove/retain correctly, and preserve unaffected static/dynamic
  Dimension references.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension))</span>
- Results are exact BOOL with false eligibility, empty layout/label/storage, fresh identity, exact
  attributes, and one-input provenance.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>
- Aggregate ALL/ANY and elementwise AND/OR are demonstrably distinct typed operation families and
  provenance arities.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Every completed 0016B/0016C numeric test behavior remains covered; inputs remain unchanged; no
  truth evaluation, empty-domain identity, gradient rule, graph/backend/build, or architecture
  behavior is added.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [gradient glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend))</span>
- Focused/aggregate tests, Javadoc, root tests, reflection/javap/import/bytecode/scope and docs
  validation pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>
- A separate clean-context docs agent finalizes permitted Javadocs/APIs/glossary/planning and
  records no-change conclusions.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- 0016D becomes Complete only after both passes; 0016E remains Draft without a specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorBooleanReductionTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorNumericReductionTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The new focused test covers exact public mapping; [BOOL](../../../../glossary.md#data-type--datatype)-only behavior; exact kind/attributes and
one-input [provenance](../../../../glossary.md#provenance); [scalar](../../../../glossary.md#term-scalar)/static/zero/dynamic full forms; positive/negative and invalid [axes](../../../../glossary.md#term-axis);
removal, retention, [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) scalar, static/dynamic reference retention; aggregate-versus-
[elementwise](../../../../glossary.md#term-elementwise) typed [identity](../../../../glossary.md#term-identity-meanings)/[arity](../../../../glossary.md#term-arity); freshness/nesting; unchanged input [descriptor](../../../../glossary.md#term-descriptor)/[Shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout)/
label/provenance/storage/contents; null/kind/type/axis precedence; ID effects; and identifier
exhaustion. Existing numeric tests continue covering all five numeric kinds, all floating types,
non-floating rejection, helper shape, and every task-0016B/0016C [invariant](../../../../glossary.md#term-invariant).

Manually inspect `javap -p -c -s`, reflection, imports, and source for exact overload descriptors,
six-method helper shape, one delegation per new overload, one kind-aware validation call, unchanged
numeric messages, one axis [normalization](../../../../glossary.md#term-normalization), reused structural Shape derivation, exact attributes,
one createDerived call, and absence of truth/storage access, short-circuiting, empty-domain policy,
sentinel, multi-axis, [element](../../../../glossary.md#term-element)-count, cast, [gradient](../../../../glossary.md#term-gradient), cross-layer type, [registry](../../../../glossary.md#term-registry)/service, [dependency](../../../../glossary.md#term-dependency),
or build change. Validate generated [Javadoc](../../../../glossary.md#term-javadoc), APIs/glossary, examples, links/anchors/fences/
whitespace, exact eleven paths, synchronized statuses, and absence of task 0016E.

## Dependencies

- 0001 supplies exact [BOOL](../../../../glossary.md#data-type--datatype)/floating DataTypes and non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) BOOL [metadata](../../../../glossary.md#term-metadata).
- 0002 supplies [Dimensions](../../../../glossary.md#dimension), [scalar](../../../../glossary.md#term-scalar) [Shape](../../../../glossary.md#shape), Shape creation, and [axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization).
- 0006 supplies [Operation](../../../../glossary.md#operation); 0007 supplies TensorDescriptor.
- 0011–0013 supply [Tensor](../../../../glossary.md#tensor), centralized [identity](../../../../glossary.md#term-identity-meanings) allocation, [provenance](../../../../glossary.md#provenance), and createDerived.
- 0015C–0015D supply the distinct elementwise boolean kinds and Tensor methods that must remain
  unchanged.<span data-glossary-navigation="true"> ([elementwise glossary](../../../../glossary.md#term-elementwise); [boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- 0016A supplies ALL/ANY aggregate kinds, axis attributes, and full-form pairing.
- 0016B–0016C supply the shared helper and tested numeric full/axis construction behavior.

## Follow-up tasks

- 0016E remains Draft for arg-max and [INT64](../../../../glossary.md#data-type--datatype) results.
- 0016F remains Draft for masked sum/mean; detailed planning decides composition versus a
  dedicated semantic form.<span data-glossary-navigation="true"> ([mean glossary](../../../../glossary.md#term-mean); [planning glossary](../../../../glossary.md#term-planning))</span>
- Compiler tasks own capture/canonicalization/autograd; backend/conformance tasks own truth-byte
  interpretation, short-circuit or parallel evaluation, empty-domain identities/errors, lowering,
  storage, kernels, and cross-backend guarantees.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [autograd glossary](../../../../glossary.md#autograd); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [evaluation glossary](../../../../glossary.md#term-evaluation); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel))</span>

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. This task composes [model](../../../../glossary.md#term-model)-owned [Tensor](../../../../glossary.md#tensor), [operation](../../../../glossary.md#operation), [descriptor](../../../../glossary.md#term-descriptor), [Shape](../../../../glossary.md#shape), and
[provenance](../../../../glossary.md#provenance) contracts without executable or cross-layer state. Stop if architecture change is
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
  second boolean helper would duplicate those invariants.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- Replace one private floating validator with one private kind-aware validator. The helper still
  has exactly six methods, while numeric failure behavior remains unchanged.
- Keep the legacy-compatible fluent names and three overload forms. Aggregate `all`/`any` remain
  separate from elementwise <span data-glossary-navigation="true"> ([elementwise glossary](../../../../glossary.md#term-elementwise))</span>`logicalAnd`/`logicalOr` by typed operation identity and provenance
  arity.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [identity glossary](../../../../glossary.md#term-identity-meanings); [arity glossary](../../../../glossary.md#term-arity))</span>
- Full results use [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [scalar](../../../../glossary.md#term-scalar), deliberately replacing legacy `[1]`.
- Caller [axes](../../../../glossary.md#term-axis) may be negative, but attributes store only normalized non-negative values.
- Zero and dynamic extents are structurally accepted. The model does not decide whether empty
  conjunction is true, empty disjunction is false, or a backend should reject an empty domain.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [backend glossary](../../../../glossary.md#backend))</span>
- BOOL input already cannot request gradients through TensorDescriptor, so common construction
  naturally produces <span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype))</span>`requiresGrad=false`; no special [gradient](../../../../glossary.md#term-gradient) branch is added.

## Known limitations

- No truth values/storage, only semantics and [provenance](../../../../glossary.md#provenance).
- Exact [BOOL](../../../../glossary.md#data-type--datatype) input and full/one-[axis](../../../../glossary.md#term-axis) forms only.
- Empty-domain logical [identities](../../../../glossary.md#term-identity-meanings)/errors and execution order are unimplemented.
- No [capture](../../../../glossary.md#term-capture-meanings), [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), [ONNX](../../../../glossary.md#term-onnx)/[backend](../../../../glossary.md#backend) support, or execution.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed the [architecture contract](../../../../glossary.md#architecture-contract) and focused architecture documentation;
documentation/planning rules; roadmap; [model](../../../../glossary.md#term-model) capabilities/master plan; completed tasks 0001, 0002,
0006, 0007, 0011, 0012, 0013, 0015C, 0015D, 0016A, 0016B, and 0016C; current [Tensor](../../../../glossary.md#tensor), datatype,
[operation](../../../../glossary.md#operation), [boolean](../../../../glossary.md#data-type--datatype)-logical, reduction, [descriptor](../../../../glossary.md#term-descriptor), [Shape](../../../../glossary.md#shape), [provenance](../../../../glossary.md#provenance), helper, and focused test
contracts; Tensor/[Compile](../../../../glossary.md#compile)/Training APIs; glossary; and Java 26 [Gradle](../../../../glossary.md#term-gradle) configuration.

The legacy branch was read directly. It confirms full, [axis](../../../../glossary.md#term-axis)-removing, and retained-axis all/any;
negative axes; exact [BOOL](../../../../glossary.md#data-type--datatype) input/output; non-[gradient](../../../../glossary.md#term-gradient) results; strided [CPU](../../../../glossary.md#term-cpu) execution; and typed
boolean result descriptors. Legacy `[1]`, negative all-axis sentinel, raw-byte truth [evaluation](../../../../glossary.md#term-evaluation),
short-circuit loops, operation traits, storage, [runtime](../../../../glossary.md#term-runtime) [graph](../../../../glossary.md#graph) state, [lowering](../../../../glossary.md#lowering), and [kernels](../../../../glossary.md#kernel) are
excluded or reassigned. Its first-[element](../../../../glossary.md#term-element) execution does not define a safe empty-domain result.

Planning selected six overloads, an in-place six-method helper [generalization](../../../../glossary.md#term-generalization), one numeric helper-
contract test update, and one new BOOL-focused test. Existing public/foundational contracts
suffice; no package, [dependency](../../../../glossary.md#term-dependency), foundation, or architecture change is required.

Planning validation:

- `git diff --check` passed. A targeted trailing-whitespace scan found no matches in the three
  changed planning paths, including this untracked task file.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- The canonical section scan found every required task section. The specification contains exact
  public/helper shapes, numeric and BOOL validation messages/order, package placement, eleven-path
  limit, acceptance criteria, validation, clean-context implementation/documentation handoff,
  decisions, limitations, evidence, and completion placeholders.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- The local Markdown target checker resolved 162 links across this task, the model master plan,
  and roadmap with zero missing files. Fence validation found balanced Markdown fences in all
  three paths; none of the changed links uses a heading anchor.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Status inspection found task 0016D [`Ready`](../../../../glossary.md#term-ready) in this specification, its linked master-plan row,
  roadmap frontier, and roadmap row. Task 0016E remains `Draft`, and no task-0016E specification
  exists.
- Package/scope review found no new package and exactly three planning paths changed. No Java,
  test, API, glossary, Gradle, architecture, AGENTS, or other-module path changed.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [module path glossary](../../../../glossary.md#term-module-path))</span>

Implementation and independent documentation validation:

- Implementation context `/root/implement_model_0010` changed exactly the five authorized Java/
  test paths: two production files, two existing tests, and the new same-package
  `TensorBooleanReductionTest`. It added the six public overloads, generalized only the existing
  helper's kind/type validation, preserved its six-method/state-free shape, updated exact Tensor
  reflection expectations and numeric helper regression assertions, and added the focused BOOL
  suite. No package, production type, dependency, build file, or cross-layer behavior was added.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression))</span>
- Clean documentation context `/root/review_model_0016c_docs` independently reread the architecture
  contract; focused overview, lifecycle, module-boundary, and dependency explanations;
  documentation workflow and General, API/Javadoc, Planning, and Example profiles; planning guide
  and roadmap; model capabilities/master plan; tasks 0001, 0002, 0006, 0007, 0011, 0012, 0013,
  0015C, 0015D, 0016A, 0016B, 0016C, and 0016D; Tensor, Compile, and Training API references;
  glossary; actual final source/tests, related model contracts and tests, generated Javadoc/test
  reports, Java 26 Gradle configuration, and the complete workspace diff. It inspected behavior
  and artifacts directly instead of relying on the implementation handoff.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [module glossary](../../../../glossary.md#term-module); [Javadoc glossary](../../../../glossary.md#term-javadoc); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- The documentation pass finalized the four axis/retained-axis `Tensor.all`/`any` Javadocs so every
  new overload now states positive/negative normalization, removed/retained Shape behavior, exact
  BOOL result, false gradient eligibility, unresolved layout, freshness, storage absence,
  one-input provenance, kind-specific type failure, and deferred truth, empty-domain, gradient,
  compiler, backend, and execution behavior. It also clarified the full-form backend boundary and
  finalized <span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [layout glossary](../../../../glossary.md#layout); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>`TensorReductionExpressions` validation-message and axis-normalization wording. The
  Tensor type Javadoc already described the new boolean aggregate family accurately.
- `docs/api/tensor-api.md` now documents the six-method boolean aggregate surface in the purpose,
  mental model, current expression summary, semantic-family status, planned boundary, and a focused
  full/axis/retained comparison table. It explains exact BOOL-only input/result, false eligibility,
  canonical scalar and structural axis shapes, exact one-input provenance, aggregate ALL/ANY versus
  elementwise AND/OR identity/arity, failures, freshness, and deferred truth evaluation,
  empty-domain identity, compiler/backend behavior, and execution.<span data-glossary-navigation="true"> ([mental model glossary](../../../../glossary.md#term-mental-model))</span><span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [elementwise glossary](../../../../glossary.md#term-elementwise); [identity glossary](../../../../glossary.md#term-identity-meanings); [arity glossary](../../../../glossary.md#term-arity))</span>
- `docs/api/compile-api.md` now includes current boolean aggregate expression inputs while keeping
  compiler entry, capture, reduction inference/canonicalization, artifacts, backend ownership, and
  execution planned. <span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [backend ownership glossary](../../../../glossary.md#backend-ownership))</span>`docs/glossary.md` synchronizes aggregate-reduction, normalized-axis, Tensor,
  and operation-family status and records the aggregate-versus-elementwise boolean distinction.
  No new reusable project term was required. No existing example code changed, so Example-format
  compile/run validation was not applicable.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
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
  from public generated pages and was reviewed completely in source.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>
- `./gradlew test --rerun-tasks` — `BUILD SUCCESSFUL`; the repository lifecycle completed 36
  actionable tasks with no failing task.
- `javap -p -c -s` confirmed the exact six public overload descriptors and one direct helper
  delegation per overload, including false in both axis-only methods and exact ALL/ANY kinds. It
  also confirmed the helper's private zero-argument constructor, exactly two package-private
  entries and four private methods, exact null/kind/type/axis validation order, one kind-aware
  validation call, unchanged numeric message path, BOOL messages, one axis normalization,
  structural removal/retention, exact attributes, one-input provenance, and one <span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>`createDerived`
  call. Verbose bytecode inspection found no synthetic helper member.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Reflection tests confirmed 67 declared public Tensor methods, exact new method-name set and
  overload modifiers/signatures, helper final/package-private shape, zero fields/nested types, and
  six declared nonsynthetic methods. Focused behavior covers both boolean kinds, every non-BOOL
  rejection, full scalar/static/zero/dynamic inputs, positive/negative/invalid axes, removal/
  retention and Dimension identity, exact descriptor/attributes/provenance, aggregate-versus-
  elementwise identity/arity, freshness/nesting, input immutability, validation precedence/
  messages, ID effects, and exhaustion. The numeric suite retains every 0016B/0016C assertion.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- Production import/source/bytecode inspection found only the permitted tensor, datatype,
  operation, reduction, shape, and JDK contracts. It found no truth/storage access, short-
  circuiting, element-count or static-shape demand, empty-domain identity, cast, negative all-axis
  sentinel, multiple-axis API, gradient rule, graph/compiler/planning/runtime/prepare/backend type,
  registry, service, dependency, or build behavior. Existing numeric aggregate and elementwise
  logical public methods are unchanged.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [prepare glossary](../../../../glossary.md#prepare); [registry glossary](../../../../glossary.md#term-registry))</span>
- The local Markdown target-and-heading checker resolved all 247 links in the six changed
  documentation/planning paths with zero errors. Markdown fences are balanced; terminology and
  current/planned status scans found no stale boolean-reduction claim; targeted whitespace
  inspection and `git diff --check` passed.
- Final scope is exactly the authorized eleven paths: two production files, three tests, Tensor
  API, Compile API, glossary, this task, model master plan, and roadmap. Task 0016D is synchronized
  as Complete. Task 0016E remains Draft, and no task-0016E specification exists. No commit or push
  occurred.<span data-glossary-navigation="true"> ([commit glossary](../../../../glossary.md#term-commit))</span>
- `docs/api/training-api.md` remains accurate unchanged because BOOL results are necessarily
  non-differentiable and this task adds no gradient value/rule, autograd, optimizer, session, or
  executable training behavior. The capability baseline already inventories full and retained-
  dimension boolean all/any and distinguishes model/public construction from later executable
  support, so it required no status edit.<span data-glossary-navigation="true"> ([differentiable glossary](../../../../glossary.md#derivative-and-chain-rule); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests, backend-
  conformance tests, and integration tests remain accurate unchanged because the work stays inside
  model-owned Tensor expression metadata and changes no module boundary, dependency, lifecycle,
  backend behavior, or end-to-end execution. Java 26 Gradle configuration remains accurate
  unchanged because no build, dependency, preview, incubator, or toolchain behavior changed.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [metadata glossary](../../../../glossary.md#term-metadata); [toolchain glossary](../../../../glossary.md#term-toolchain))</span>
- [`DataType`](../../../../glossary.md#data-type--datatype), [`Dimension`](../../../../glossary.md#dimension), `StaticDimension`, `Shape`, `TensorDescriptor`, `TensorFactory`,
  `TensorProvenance`, `Operation`, [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), `AggregateReductionKind`,
  `AxisReductionAttrs`, and `BooleanLogicalKind` contracts and Javadocs remain accurate unchanged.
  The generalized helper composes their existing exact-BOOL, non-differentiable descriptor,
  immutable-Shape, central-ID, typed-operation, attribute, and provenance contracts without
  changing them. Existing elementwise logical and numeric aggregate Javadocs/tests remain accurate
  and unchanged apart from the numeric test's intentional helper-name/unsupported-kind assertions.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>

## Implementation notes

- Added exactly six fluent full/[axis](../../../../glossary.md#term-axis)/retained-axis `all` and `any` methods as direct delegations to
  the existing aggregate-reduction helper.
- Generalized the helper's accepted kinds and renamed its one private type validator to preserve
  numeric floating behavior while enforcing exact BOOL for ALL/ANY, without changing its six-
  method structure or common Shape/provenance/factory construction.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance))</span>
- Updated exact [Tensor](../../../../glossary.md#tensor)/helper [regression](../../../../glossary.md#term-regression) assertions and added one focused seven-test [BOOL](../../../../glossary.md#data-type--datatype) suite.
- Finalized affected Tensor/helper Javadocs, Tensor API, Compile API, glossary, task evidence,
  model master plan, and roadmap without adding executable truth behavior.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>

## Completion summary

- Completed changes: Implemented and documented exact-BOOL full and one-axis all/any Tensor
  expression construction with fixed false eligibility, exact reduced shapes, and typed one-input
  provenance.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [axis glossary](../../../../glossary.md#term-axis); [Tensor glossary](../../../../glossary.md#tensor); [shapes glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance))</span>
- Files changed or created: Exactly two production Java files, three tests, Tensor API, Compile
  API, glossary, this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused suites 7/7, 9/9, and 14/14; all 389 model tests across 51 suites;
  model Javadoc; root tests; bytecode/reflection/import/source checks; 247 Markdown link/anchor
  checks; fence/terminology/whitespace checks; exact scope/status checks; and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`
  passed.
- Documentation-agent review: Clean context `/root/review_model_0016c_docs` completed the
  independent pass using General, API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API, Compile API, and glossary now describe current boolean
  aggregate construction and its distinction from elementwise boolean logic. Training API,
  capabilities, architecture/ADRs/tests, conformance/integration tests, and build configuration
  remain accurate unchanged for the recorded reasons.<span data-glossary-navigation="true"> ([boolean glossary](../../../../glossary.md#data-type--datatype); [elementwise glossary](../../../../glossary.md#term-elementwise); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Javadoc review: The new Tensor axis/retained overloads and affected helper contracts were
  finalized; full overloads and the Tensor type were already complete except for the added backend-
  boundary clarification. Related foundational, operation, descriptor, factory, provenance,
  boolean-logical, and reduction contracts remain accurate.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Glossary impact: Synchronized existing aggregate-reduction, normalized-axis, Tensor, and
  operation-family entries; no new reusable term was necessary.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0016D. Task 0016E remains Draft without a detailed
  specification.

Status: Complete
