# Task 0017J: Pad and Tile [Tensor](../../../../glossary.md#tensor) Expressions

## Status

Complete

## Goal

Add public [constant](../../../../glossary.md#term-constant)-[padding](../../../../glossary.md#padding) and per-[axis](../../../../glossary.md#term-axis) [tiling](../../../../glossary.md#tiling) expression construction to [`Tensor`](../../../../glossary.md#tensor).

Both methods must validate complete [rank](../../../../glossary.md#term-rank)-aligned requests, derive every locally provable result
[Shape](../../../../glossary.md#shape), retain exact [data type](../../../../glossary.md#data-type--datatype) and [gradient](../../../../glossary.md#term-gradient) eligibility, record typed pad/tile semantics with
one-input [provenance](../../../../glossary.md#provenance), and return fresh storage-free expressions with unresolved [layout](../../../../glossary.md#layout). This task
does not allocate or populate output values, convert padding constants, define gradients, choose
[materialization](../../../../glossary.md#materialization), lower operations, or execute work.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly these public instance methods to [Tensor](../../../../glossary.md#tensor):
  - `pad(long[] before, long[] after, double constantValue)`
  - `tile(long... repeats)`
- Add one field-free package-private final `TensorPadTileExpressions` helper in `model.tensor`.
- Give the helper exactly two package-private entry methods and three private shared/[Shape](../../../../glossary.md#shape) methods.
- Require non-null request arrays whose length equals input rank; defensively clone before
  constructing semantic attributes.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>
- Use PadAttrs and TileAttrs as the sole width/repeat [validation](../../../../glossary.md#term-validation) and [immutable](../../../../glossary.md#term-immutable) [snapshot](../../../../glossary.md#term-snapshot) boundary.
- Preserve every raw double [padding](../../../../glossary.md#padding) [constant](../../../../glossary.md#term-constant) without [eager](../../../../glossary.md#term-eager-evaluation) conversion or validation.
- Derive padding extents with checked addition and tile extents with checked multiplication.
- Preserve an exact dynamic Dimension reference only when padding is zero on both sides or repeat
  count is one; reject transformations of dynamic dimensions that cannot be represented locally.
- Accept [scalar](../../../../glossary.md#term-scalar) empty requests and zero-[extent](../../../../glossary.md#term-extent) static [dimensions](../../../../glossary.md#dimension).
- Preserve input [DataType](../../../../glossary.md#data-type--datatype) and requiresGrad for all six current data types.
- Keep result [layout](../../../../glossary.md#layout) unresolved for every pad/tile expression, including [identity](../../../../glossary.md#term-identity-meanings) requests.
- Create exact [PAD](../../../../glossary.md#padding)/PadAttrs or TILE/TileAttrs [operation](../../../../glossary.md#operation), ordered [provenance](../../../../glossary.md#provenance) `[input]`, absent label,
  absent storage, and fresh identity.
- Update TensorTest only for exact public [API](../../../../glossary.md#term-application-programming-interface-api) inventory and add one focused expression suite.
- Finalize Javadocs, Tensor/Compile API, glossary, task evidence, master plan, and roadmap through
  the mandatory independent documentation pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- another public overload, int-array compatibility overload, List-based API, per-axis padding
  constants, default padding value, default repeats, or partial-rank requests<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [axis glossary](../../../../glossary.md#term-axis); [constants glossary](../../../../glossary.md#term-constant); [rank glossary](../../../../glossary.md#term-rank))</span>
- reflect, edge, wrap, symmetric, negative/cropping, [tensor](../../../../glossary.md#tensor)-valued, or another [padding](../../../../glossary.md#padding) mode
- zero/negative repeats, implicit leading axes, rank promotion, element-wise repeat, interleave,
  broadcast expand, or a scalar-repeat overload<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- eagerly converting constantValue to FLOAT64/FLOAT32/BFLOAT16/INT32/INT64/BOOL, rejecting NaN or
  infinity, applying range checks, saturation, truncation, BFLOAT16 rounding, or BOOL truthiness<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [saturation glossary](../../../../glossary.md#term-saturation); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- changing PadKind, PadAttrs, TileKind, TileAttrs, DataType, Shape, Dimension, LayoutDescriptor,
  TensorDescriptor, TensorFactory, TensorProvenance, Operation, or completed tests/contracts<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [Operation glossary](../../../../glossary.md#operation))</span>
- deriving resolved output layout, claiming a view/alias, attaching or inspecting host storage,
  allocating output, copying/repeating/filling values, or choosing physical materialization<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena); [host storage glossary](../../../../glossary.md#host-storage); [materialization glossary](../../../../glossary.md#materialization))</span>
- gradients, pad-backward slice, tile-backward reduction, autograd, graph capture, compiler passes,
  planning, prepare, runtime, backend lowering/kernels, engine, trace, ONNX, or training behavior<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [slice glossary](../../../../glossary.md#slice); [autograd glossary](../../../../glossary.md#autograd); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- another helper/type/test, dependency, Gradle/build option, architecture change, another module,
  or task-0017K specification<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0017I](0017i-pad-and-tile-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The selected baseline exposes [constant](../../../../glossary.md#term-constant) `pad(int[] before, int[] after, double constantValue)` and
`tile(int... repeats)`. Legacy code requires one width/repeat per [axis](../../../../glossary.md#term-axis), non-negative [padding](../../../../glossary.md#padding),
positive repeats, checked int [Shape](../../../../glossary.md#shape) arithmetic, and preserves input [DataType](../../../../glossary.md#data-type--datatype). Its [CPU](../../../../glossary.md#term-cpu) route later
interprets the stored double as exact [FLOAT64](../../../../glossary.md#term-float64), Java-narrowed [FLOAT32](../../../../glossary.md#term-float32)/integral, [binary32](../../../../glossary.md#floating-point-precision-and-rounding)-to-[BFLOAT16](../../../../glossary.md#term-bfloat16),
or zero/non-zero [BOOL](../../../../glossary.md#data-type--datatype). It also couples [model](../../../../glossary.md#term-model) construction to [eager](../../../../glossary.md#term-eager-evaluation) storage, [gradients](../../../../glossary.md#term-gradient), [compiler](../../../../glossary.md#term-compiler),
[kernels](../../../../glossary.md#kernel), and [runtime](../../../../glossary.md#term-runtime) behavior.

The new public [API](../../../../glossary.md#term-application-programming-interface-api) widens [dimensions](../../../../glossary.md#dimension) to long. It retains raw double semantics and all current data
types without doing conversion in model expression construction. Concrete [lowering](../../../../glossary.md#lowering)/execution must
eventually implement a documented cross-[backend](../../../../glossary.md#backend) conversion policy, but this task neither chooses a
[kernel](../../../../glossary.md#kernel) nor materializes a converted [scalar](../../../../glossary.md#term-scalar). Shape arithmetic remains local and checked.

Current Dimension supports static sizes or a [symbol](../../../../glossary.md#term-symbol), not affine expressions such as `N + 2` or
`N * 3`. Therefore zero-width pad and repeat-one tile preserve a dynamic symbol, while non-[identity](../../../../glossary.md#term-identity-meanings)
transformations of a dynamic dimension are rejected rather than hidden as an unsound constraint.

## Architecture constraints

- Tensor remains public mutable API state, not graph IR. New methods create expression metadata
  only through the existing derived factory seam.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Semantic [identity](../../../../glossary.md#term-identity-meanings) and [immutable](../../../../glossary.md#term-immutable) [parameters](../../../../glossary.md#term-parameter) come only from completed task 0017I.
- Input descriptor/Shape may be inspected; values, storage, residency, backend capability, and
  previous provenance must not be inspected.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape); [residency glossary](../../../../glossary.md#residency); [backend capability glossary](../../../../glossary.md#backend-capability); [provenance glossary](../../../../glossary.md#provenance))</span>
- Raw arrays are caller syntax only. PadAttrs/TileAttrs hold immutable copied lists.
- Result rank equals input rank. Static result extents are checked long arithmetic; locally
  unchanged dynamic dimensions retain exact references.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [dimensions glossary](../../../../glossary.md#dimension))</span>
- All result layouts are Optional.empty. Pad introduces new positions and tile repeats values, so
  physical geometry/materialization cannot be represented as an input alias here. Even semantic
  identity requests remain explicit for later compiler canonicalization.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization); [alias glossary](../../../../glossary.md#aliasing-and-arena); [compiler glossary](../../../../glossary.md#term-compiler); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- Raw constantValue remains inside PadAttrs. Result descriptor preserves input DataType; no model-
  layer value conversion or validation is performed.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [model glossary](../../../../glossary.md#term-model); [validation glossary](../../../../glossary.md#term-validation))</span>
- Compiler owns capture/canonicalization/backward graph; planning/backend prepare own
  materialization/lowering; runtime executes only prepared work.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [backward graph glossary](../../../../glossary.md#backward-graph); [planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- No [dependency](../../../../glossary.md#term-dependency) or [module](../../../../glossary.md#term-module) boundary changes are authorized.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.tensor.Tensor` receives exactly two public methods.
- `io.github.pho001.synaptik.model.tensor.TensorPadTileExpressions` owns request validation, Shape
  derivation, typed operation/provenance construction, and unresolved result descriptors.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- `TensorPadTileExpressionTest` mirrors the package for focused helper/[API](../../../../glossary.md#term-application-programming-interface-api) [validation](../../../../glossary.md#term-validation).
- `TensorTest` changes only its exact public API inventory/reflection assertions.

PadKind/PadAttrs/TileKind/TileAttrs remain in `model.operation.layout` unchanged.

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) methods

Add exactly:

```java
public Tensor pad(long[] before, long[] after, double constantValue) {
    return TensorPadTileExpressions.pad(this, before, after, constantValue);
}

public Tensor tile(long... repeats) {
    return TensorPadTileExpressions.tile(this, repeats);
}
```

Each method contains exactly one return and one matching helper invocation. Both are non-static,
non-synchronized, and perform no direct [validation](../../../../glossary.md#term-validation), allocation, field access, or delegation to
another public Tensor method.

### Helper [shape](../../../../glossary.md#shape)

Create one package-private final, field-free class with one private zero-[argument](../../../../glossary.md#term-argument) constructor and
exactly these five static methods:

```java
static Tensor pad(Tensor input, long[] before, long[] after, double constantValue)
static Tensor tile(Tensor input, long[] repeats)
private static Shape paddedShape(Shape inputShape, PadAttrs attrs)
private static Shape tiledShape(Shape inputShape, TileAttrs attrs)
private static Tensor create(
        Tensor input,
        TensorDescriptor inputDescriptor,
        Shape resultShape,
        Operation operation)
```

Add no fields, nested type, overload, alternate constructor, [cache](../../../../glossary.md#term-cache), [mutable](../../../../glossary.md#term-mutable) state, or extra method.

### [Padding](../../../../glossary.md#padding) construction

`pad` performs this exact order:

1. null-check input, before, and after, in order, with exact messages `input`, `before`, `after`;
2. read exact input [descriptor](../../../../glossary.md#term-descriptor) and [Shape](../../../../glossary.md#shape) once;
3. require before length equal [rank](../../../../glossary.md#term-rank), otherwise throw IllegalArgumentException with exact message
   `padding before length <length> must equal input rank <rank>`;
4. require after length equal rank, otherwise throw with exact message
   `padding after length <length> must equal input rank <rank>`;
5. clone before then after exactly once;
6. box each private clone in order and construct exactly one PadAttrs with unchanged constantValue;
7. call paddedShape once;
8. construct exactly one [Operation](../../../../glossary.md#operation)(PadKind.[PAD](../../../../glossary.md#padding), attrs);
9. call create once and return.

PadAttrs supplies indexed null-free/non-negative [validation](../../../../glossary.md#term-validation) and [immutable](../../../../glossary.md#term-immutable) list snapshots. Primitive
arrays cannot contain null. No constantValue check occurs. Caller arrays are never retained or
mutated; post-clone caller mutation cannot affect attributes or Shape.

### [Tiling](../../../../glossary.md#tiling) construction

`tile` performs this exact order:

1. null-check input then repeats, with exact messages `input` and `repeats`;
2. read exact [descriptor](../../../../glossary.md#term-descriptor) and [Shape](../../../../glossary.md#shape) once;
3. require repeat length equal [rank](../../../../glossary.md#term-rank), otherwise throw IllegalArgumentException with exact message
   `tile repeats length <length> must equal input rank <rank>`;
4. clone repeats exactly once;
5. box the private clone in order and construct exactly one TileAttrs;
6. call tiledShape once;
7. construct exactly one [Operation](../../../../glossary.md#operation)(TileKind.TILE, attrs);
8. call create once and return.

TileAttrs supplies positive-repeat [validation](../../../../glossary.md#term-validation) and [immutable](../../../../glossary.md#term-immutable) [snapshot](../../../../glossary.md#term-snapshot). Empty repeats are valid only
for [scalar](../../../../glossary.md#term-scalar) rank zero because exact rank equality is required.

### Padded [Shape](../../../../glossary.md#shape)

`paddedShape` visits [axes](../../../../glossary.md#term-axis) in increasing order and creates one [rank](../../../../glossary.md#term-rank)-sized [Dimension](../../../../glossary.md#dimension) array.

- StaticDimension size `s` produces new StaticDimension exactly
  `Math.addExact(Math.addExact(s, before), after)`.
- DynamicDimension is retained by exact reference only when before and after are both zero.
- A dynamic dimension with non-zero [padding](../../../../glossary.md#padding) throws IllegalArgumentException with exact message
  `cannot pad dynamic axis <axis> with before=<before> and after=<after>`.

Return Shape.ofDimensions once. [Scalar](../../../../glossary.md#term-scalar) empty lists return [canonical](../../../../glossary.md#term-canonical-form) scalar Shape. Static zero
extents are accepted; zero/zero pad preserves logical size zero through a new StaticDimension.
Checked [overflow](../../../../glossary.md#term-overflow) propagates as ArithmeticException before [Tensor](../../../../glossary.md#tensor) ID allocation.

### Tiled [Shape](../../../../glossary.md#shape)

`tiledShape` also creates one [rank](../../../../glossary.md#term-rank)-sized [Dimension](../../../../glossary.md#dimension) array in [axis](../../../../glossary.md#term-axis) order.

- StaticDimension size `s` produces new StaticDimension `Math.multiplyExact(s, repeat)`.
- DynamicDimension is retained by exact reference only when repeat equals one.
- A dynamic dimension with another repeat throws IllegalArgumentException with exact message
  `cannot tile dynamic axis <axis> with repeat=<repeat>`.

Return Shape.ofDimensions once. [Scalar](../../../../glossary.md#term-scalar) empty repeats return [canonical](../../../../glossary.md#term-canonical-form) scalar Shape. Static zero
extents remain zero for every positive repeat. [Overflow](../../../../glossary.md#term-overflow) propagates before ID allocation.

### [Constant](../../../../glossary.md#term-constant) and data-type boundary

Every current [DataType](../../../../glossary.md#data-type--datatype) is accepted. Result DataType exactly equals input DataType and requiresGrad
is unchanged. PadAttrs retains constantValue as the exact supplied double primitive; [model](../../../../glossary.md#term-model) code
does not narrow, round, saturate, normalize, or reject it.

This task documents only that later execution must interpret the constant according to the
prepared output DataType consistently across [backends](../../../../glossary.md#backend). Detailed [finite](../../../../glossary.md#term-finite)-[precision](../../../../glossary.md#term-precision) conversion and
[kernel](../../../../glossary.md#kernel) [conformance](../../../../glossary.md#term-conformance) belong to backend contracts/tests, not this expression constructor. No [eager](../../../../glossary.md#term-eager-evaluation)
converted [scalar](../../../../glossary.md#term-scalar), [carrier](../../../../glossary.md#term-storage-carrier), storage, or secondary attribute is created.

### Common result construction

`create` constructs exactly:

```java
TensorDescriptor descriptor = new TensorDescriptor(
        inputDescriptor.dataType(),
        resultShape,
        Optional.empty(),
        inputDescriptor.requiresGrad());
TensorProvenance provenance = new TensorProvenance(operation, List.of(input));
return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
```

The supplied [Operation](../../../../glossary.md#operation) is exact [PAD](../../../../glossary.md#padding)/PadAttrs or TILE/TileAttrs. Every success consumes one fresh ID
and has absent label/storage. [Validation](../../../../glossary.md#term-validation), attribute, [Shape](../../../../glossary.md#shape), arithmetic, and operation failures
consume no ID. Identifier exhaustion is final. No input label, old [provenance](../../../../glossary.md#provenance), [layout](../../../../glossary.md#layout), values, or
storage is retained except the exact input reference inside new provenance.

## Affected files

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorPadTileExpressions.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorPadTileExpressionTest.java`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

At most these ten paths may change. Production is limited to [Tensor](../../../../glossary.md#tensor) plus one package-private
helper; tests to Tensor inventory plus one focused suite; documentation to public [API](../../../../glossary.md#term-application-programming-interface-api)/status and
[planning](../../../../glossary.md#term-planning) evidence.

If another helper/type/test, overload, semantic/foundational edit, [dependency](../../../../glossary.md#term-dependency), build/architecture
change, or eleventh path is needed, stop and report. Do not create task 0017K.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Fully document both public methods and helper type/constructor/all five methods.
- Explain exact rank requests, primitive-array ownership, validation order, scalar empty requests,
  static zero extents, and dynamic identity-only rules.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [validation glossary](../../../../glossary.md#term-validation); [scalar glossary](../../../../glossary.md#term-scalar); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Explain checked [Shape](../../../../glossary.md#shape) arithmetic with numeric examples for pad and complete-pattern tile.
- Document raw double retention, acceptance of all DataTypes, unchanged eligibility, and explicit
  deferral of concrete conversion/materialization.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- Explain why every result [layout](../../../../glossary.md#layout) is unresolved and why this differs from [view](../../../../glossary.md#term-view-meanings) expressions.
- Document result Shape/type/layout/label/storage/operation/attrs/provenance/freshness and every
  null/rank/width/repeat/dynamic/arithmetic/ID failure.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance))</span>
- Explain no [eager](../../../../glossary.md#term-eager-evaluation) values, physical [alias](../../../../glossary.md#aliasing-and-arena), [gradient](../../../../glossary.md#term-gradient), [compiler](../../../../glossary.md#term-compiler), [backend](../../../../glossary.md#backend), [ONNX](../../../../glossary.md#term-onnx), or execution work.
- Independently review Pad/Tile semantics, DataType, Shape/Dimension, TensorDescriptor,
  TensorFactory, TensorProvenance, and adjacent view-expression Javadocs; record unchanged reasons
  or stop on discrepancy.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [Dimension glossary](../../../../glossary.md#dimension))</span>

## Acceptance criteria

- Tensor exposes exactly pad(long[],long[],double) and tile(long...); public method count rises 87
  to 89.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Each public method delegates once; helper is exact final/package-private/field-free with private
  constructor and five methods.
- Null/[rank](../../../../glossary.md#term-rank)/order, defensive clone, PadAttrs/TileAttrs [validation](../../../../glossary.md#term-validation), messages, and [ownership](../../../../glossary.md#term-ownership-meanings) match.
- Scalar, static, zero-extent, and locally unchanged dynamic Shapes behave exactly as specified;
  non-identity dynamic transforms and checked overflow fail before ID allocation.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [extent glossary](../../../../glossary.md#term-extent); [Shapes glossary](../../../../glossary.md#shape); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- All six DataTypes and valid eligibility states are retained exactly; every raw double is accepted
  without conversion.
- Every result [layout](../../../../glossary.md#layout) is unresolved, including [identity](../../../../glossary.md#term-identity-meanings) and fully static requests.
- Result records exact typed [operation](../../../../glossary.md#operation)/attributes/[input], absent label/storage, fresh identity.
- Repeated, nested, zero-width, repeat-one, and identity requests stay explicit.
- No values/storage/[gradient](../../../../glossary.md#term-gradient)/[compiler](../../../../glossary.md#term-compiler)/[planning](../../../../glossary.md#term-planning)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) behavior or architecture change.
- Independent documentation review and all status/evidence synchronization complete before
  marking Complete.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorPadTileExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact public/helper [API](../../../../glossary.md#term-application-programming-interface-api); all data types/eligibility; raw [constant](../../../../glossary.md#term-constant) categories;
null/[rank](../../../../glossary.md#term-rank)/negative/repeat validation and precedence; caller-array [ownership](../../../../glossary.md#term-ownership-meanings); [scalar](../../../../glossary.md#term-scalar)/static/zero/
dynamic [Shape](../../../../glossary.md#shape) rules; exact unaffected dynamic [identity](../../../../glossary.md#term-identity-meanings); checked [overflow](../../../../glossary.md#term-overflow); unresolved [layout](../../../../glossary.md#layout) for
every input layout state; exact [operation](../../../../glossary.md#operation)/attrs/[provenance](../../../../glossary.md#provenance); absent label/storage; dead-storage
non-interference; freshness/nesting/identity; ID side effects; and exhaustion.

Inspect javap, reflection, imports, [bytecode](../../../../glossary.md#term-bytecode), and source. Confirm two one-call public delegates,
exact five-method helper, one clone per array, one attribute/Shape/Operation path per family, one
common createDerived, and no value/storage/cross-layer behavior. Validate generated [Javadoc](../../../../glossary.md#term-javadoc),
executable examples, [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) API/glossary, links/anchors/fences/whitespace, exact ten paths,
synchronized statuses, and no task-0017K specification.

## Dependencies

- 0001/0002 supply [DataType](../../../../glossary.md#data-type--datatype) and static/dynamic [Shape](../../../../glossary.md#shape) contracts.
- 0003/0007 supply unresolved-[layout](../../../../glossary.md#layout) TensorDescriptor construction.
- 0011–0013 supply [Tensor](../../../../glossary.md#tensor), derived [identity](../../../../glossary.md#term-identity-meanings), and [provenance](../../../../glossary.md#provenance).
- 0017I supplies [PAD](../../../../glossary.md#padding)/TILE and [immutable](../../../../glossary.md#term-immutable) attributes.
- Neighboring expressions provide patterns but no extra production [dependency](../../../../glossary.md#term-dependency).

## Follow-up tasks

- 0017K remains Draft for concat/stack/unstack semantic [identities](../../../../glossary.md#term-identity-meanings) and [parameters](../../../../glossary.md#term-parameter).
- [Compiler](../../../../glossary.md#term-compiler) later owns pad/tile [capture](../../../../glossary.md#term-capture-meanings), [identity](../../../../glossary.md#term-identity-meanings) [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), and [backward graph](../../../../glossary.md#backward-graph) construction.
- Planning/backend prepare own allocation, materialization, lowering, constant conversion, and
  kernels; backend conformance must enforce common observable semantics.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare); [materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering); [constant glossary](../../../../glossary.md#term-constant); [kernels glossary](../../../../glossary.md#kernel); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- [ONNX](../../../../glossary.md#term-onnx) extension later owns Pad/Tile mappings.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. [Model](../../../../glossary.md#term-model) already owns [Tensor](../../../../glossary.md#tensor) expression construction, [Shape](../../../../glossary.md#shape) [metadata](../../../../glossary.md#term-metadata),
[operation](../../../../glossary.md#operation) semantics, [descriptors](../../../../glossary.md#term-descriptor), and [provenance](../../../../glossary.md#provenance). [Lifecycle](../../../../glossary.md#lifecycle) [ownership](../../../../glossary.md#term-ownership-meanings) remains unchanged.

Stop if implementation needs physical output storage, converted [constants](../../../../glossary.md#term-constant), [gradients](../../../../glossary.md#term-gradient),
[compiler](../../../../glossary.md#term-compiler)/[planning](../../../../glossary.md#term-planning)/[backend](../../../../glossary.md#backend) behavior, another [dependency](../../../../glossary.md#term-dependency), or architecture changes.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0003/0007/0011/0012/0013/0017I/0017J, Tensor API,
Compile API, Training API, glossary, current DataType/Dimension/Shape/TensorDescriptor/Tensor/
TensorFactory/TensorProvenance/Operation/PadKind/PadAttrs/TileKind/TileAttrs contracts and focused
expression tests, and Java 26 Gradle configuration.

Implement task 0017J exactly. Modify Tensor.java and add package-private final
TensorPadTileExpressions.java. Update TensorTest only for exact two-method API expansion and add
TensorPadTileExpressionTest. Add exactly pad(long[],long[],double) and tile(long...).

The field-free helper has exactly five specified methods. Both paths null/rank validate and clone
complete arrays, construct exact immutable attrs, derive checked static Shapes, preserve only
identity-transformed dynamic Dimension references, retain exact type/eligibility, and create exact
PAD or TILE one-input provenance through one common createDerived call. Every result layout is
unresolved and every result is fresh. Preserve raw double without conversion and accept all data
types.

Do not modify semantics/foundations, add overloads/types/helpers, convert constants, support other
padding/repeat modes, inspect/copy values/storage, derive physical layout, define gradients,
capture/canonicalize graphs, or add compiler/planning/prepare/runtime/backend/ONNX behavior,
dependencies, build/architecture changes, or later specs. Stop beyond ten paths or on architecture
uncertainty.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/Compile API/glossary/planning, record no-change conclusions, and rerun validation.

Update 0017J, model master plan, and roadmap only for status/evidence. Do not mark Complete until
both passes succeed. Leave 0017K Draft without a specification. Do not commit/push.
```

## Local decisions

- Public width/repeat arrays use long geometry; tile remains varargs for the one-value-per-axis
  ergonomic form.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- Every current [DataType](../../../../glossary.md#data-type--datatype) is accepted and raw double is retained without [model](../../../../glossary.md#term-model)-layer conversion.
- Dynamic dimensions support only transformations whose extent is provably unchanged: zero/zero
  pad or repeat one.<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension); [extent glossary](../../../../glossary.md#term-extent))</span>
- All results keep layout unresolved because pad/tile require output materialization and cannot be
  represented as ordinary input view geometry.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Scalar empty requests, static zero extents, zero-width pad, and repeat-one tile are valid fresh
  explicit operations; compiler owns identity elimination.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [compiler glossary](../../../../glossary.md#term-compiler); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Known limitations

- No non-constant padding, negative crop, zero repeats, dynamic affine Shape, converted constant,
  resolved output layout, values, storage, gradients, compiler capture, materialization, lowering,
  execution, or ONNX mapping exists.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [padding glossary](../../../../glossary.md#padding); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering); [ONNX glossary](../../../../glossary.md#term-onnx))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorPadTileExpressionTest --tests
  io.github.pho001.synaptik.model.tensor.TensorTest --rerun-tasks` passed: 11 focused pad/tile tests
  plus 14 exact Tensor API tests, with zero failures, errors, or skips.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `./gradlew :modules:model:test --rerun-tasks` passed: 70 suites and 577 tests, with zero failures,
  errors, or skips. `./gradlew test --rerun-tasks` also passed with 36 executed actionable tasks.
- `./gradlew :modules:model:javadoc --rerun-tasks` passed. Generated [`Tensor`](../../../../glossary.md#tensor) Javadoc contains the
  complete <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`pad(long[], long[], double)` and `tile(long...)` contracts, parameters, results, and
  expected failures.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter))</span>
- `javap -p -s -c`, reflection assertions, import/source inspection, and focused tests confirm 89
  public Tensor methods; two one-call public delegates; one final package-private, field-free
  helper with a private constructor and exactly five methods; one clone per request array; one
  attributes/Shape/Operation path per family; and one shared <span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Operation glossary](../../../../glossary.md#operation))</span>`TensorFactory.createDerived` call.
- All six data types and valid eligibility states, raw constant categories, scalar/static/zero/
  dynamic Shapes, overflow and validation precedence, array ownership, every input layout state,
  provenance, freshness, nesting, absent label/storage, ID side effects, and exhaustion passed the
  focused test matrix.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [scalar glossary](../../../../glossary.md#term-scalar); [overflow glossary](../../../../glossary.md#term-overflow); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>
- The documented `PadTileExpressionExample` compiled against current model classes and printed
  Shapes <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>`[5, 7]` and `[4, 12]`, [immutable](../../../../glossary.md#term-immutable) normalized attributes, [`PAD`](../../../../glossary.md#padding)/`TILE`, exact one-input
  provenance, unresolved layouts, and absent storage exactly as documented.
- Tensor API, Compile API, and glossary local-link targets and new anchors resolve; code fences are
  balanced; changed Markdown has no trailing whitespace and ends with newlines; <span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [targets glossary](../../../../glossary.md#term-target))</span>`git diff --check`
  passes.
- The changed-path inventory is exactly the authorized ten paths: two production sources, two
  tests, Tensor API, Compile API, glossary, this task, model master plan, and roadmap. Task 0017K
  remains Draft at row 64 and no task-0017K specification exists.
- Training API remains accurate unchanged because no gradient or training contract was added.
  Capabilities remain accurate because they already select constant pad and complete-pattern tile.
  Pad/Tile semantics and DataType, Dimension, Shape, TensorDescriptor, TensorFactory,
  TensorProvenance, Operation, and adjacent expression contracts remain accurate because this task
  composes their existing APIs without changing them.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [DataType glossary](../../../../glossary.md#data-type--datatype); [Dimension glossary](../../../../glossary.md#dimension))</span>
- Architecture, ADRs, architecture tests, backend conformance, integration tests, Gradle/Java 26,
  dependencies, and other modules remain accurate unchanged because this is model-only expression
  metadata without a boundary, backend, execution, or end-to-end behavior change.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [metadata glossary](../../../../glossary.md#term-metadata))</span>

## Implementation notes

- `Tensor.pad` and `Tensor.tile` delegate to one focused package-private helper. The helper validates
  references and exact rank before cloning complete arrays, then relies on <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>`PadAttrs` or `TileAttrs`
  for ordered element validation and immutable snapshots.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [validation glossary](../../../../glossary.md#term-validation); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Static pad and tile extents use `Math.addExact` and `Math.multiplyExact`. Dynamic Dimensions are
  retained by exact reference only for zero/zero padding or repeat one; unrepresentable symbolic
  transforms fail before identity allocation.<span data-glossary-navigation="true"> ([Dimensions glossary](../../../../glossary.md#dimension); [padding glossary](../../../../glossary.md#padding))</span>
- The common construction path preserves exact data type and gradient eligibility, deliberately
  supplies unresolved layout and absent label, records exact typed operation plus provenance<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient); [layout glossary](../../../../glossary.md#layout); [operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance))</span>
  `[input]`, and delegates final [identity](../../../../glossary.md#term-identity-meanings) creation to `TensorFactory.createDerived`.
- The independent documentation pass finalized public and helper Javadocs, Tensor API, Compile API,
  glossary, task evidence, model master plan, and roadmap. It found no source/test inconsistency
  requiring an implementation-owned Java edit.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>

## Completion summary

- Completed changes: added public constant-pad and complete-pattern tile Tensor expression
  construction with checked Shape validation, exact metadata retention, unresolved result layout,
  and fresh one-input provenance; added focused and exact API tests; finalized all required
  documentation and planning evidence.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: exactly the ten authorized Tensor/helper/test/API/glossary/task/master-
  plan/roadmap paths.
- Validation: focused 25 tests, full model 577 tests, model Javadoc, root test, bytecode/reflection/
  source/import inspection, executable documentation example, links/anchors/fences/whitespace,
  exact-path/status/no-follow-up-spec checks, and <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>`git diff --check` all passed.
- Unresolved issues: None for task 0017J.
- Required follow-up: None. Task 0017K remains Draft without a detailed specification.

Status: Complete
