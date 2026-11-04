# Task 0017F1: Expand-[Dimensions](../../../../glossary.md#dimension) and Squeeze [Tensor](../../../../glossary.md#tensor) Expressions

## Status

Complete

## Goal

Add public, storage-free [Tensor](../../../../glossary.md#tensor) expressions that insert one [singleton](../../../../glossary.md#term-singleton) [axis](../../../../glossary.md#term-axis) or remove one statically
known singleton axis using the completed `EXPAND_DIMS` and `SQUEEZE` semantics.

Both transformations preserve logical values and the exact order of unaffected [dimensions](../../../../glossary.md#dimension).
Resolved input geometry produces a new same-offset [view](../../../../glossary.md#term-view-meanings) [layout](../../../../glossary.md#layout) with one [stride](../../../../glossary.md#term-stride) inserted or removed;
unresolved geometry remains unresolved. This task constructs [model](../../../../glossary.md#term-model) [metadata](../../../../glossary.md#term-metadata) only and does not
attach storage, define [gradients](../../../../glossary.md#term-gradient), [capture](../../../../glossary.md#term-capture-meanings) a [graph](../../../../glossary.md#graph), choose [materialization](../../../../glossary.md#materialization), lower, or execute work.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly `Tensor.expandDims(int axis)` and `Tensor.squeeze(int axis)`.
- Make each public method delegate exactly once to its matching package-private helper entry.
- Add one package-private final field-free `TensorRankEditingExpressions` helper with exactly the
  nine methods specified under Required contract and one private constructor.
- Normalize expand-[dimensions](../../../../glossary.md#dimension) insertion [axes](../../../../glossary.md#term-axis) over the output positions `[0, inputRank]`, accepting
  negative positions relative to `inputRank + 1`.
- Normalize squeeze axes through the existing input-[Shape](../../../../glossary.md#shape) axis contract.
- Require the selected squeeze dimension to be statically known with [extent](../../../../glossary.md#term-extent) exactly one.
- Reject a dynamic selected squeeze dimension because singleton compatibility is not locally
  provable.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton))</span>
- Build inserted result Shape with one new `StaticDimension(1)` and exact unaffected input
  Dimension references.
- Build squeezed result Shape by removing one Dimension and preserving exact unaffected references.
- For resolved input layout, insert one deterministic stride for expand-dimensions while preserving
  existing strides and offset.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride))</span>
- For resolved input layout, remove the selected stride for squeeze while preserving remaining
  strides and offset.
- Support every resolved input [layout](../../../../glossary.md#layout) kind and leave layout unresolved when input layout is absent.
- Preserve exact input [data type](../../../../glossary.md#data-type--datatype) and [gradient](../../../../glossary.md#term-gradient) eligibility.
- Construct exact `AxisTransformKind.EXPAND_DIMS` or `SQUEEZE`,
  `AxisTransformAttrs(normalizedAxis)`, and ordered [provenance](../../../../glossary.md#provenance) `[input]`.
- Call `TensorFactory.createDerived` once with no label or [host storage](../../../../glossary.md#host-storage).
- Return a fresh [Tensor](../../../../glossary.md#tensor) for every valid request, including inverse-like, repeated, and nested calls.
- Add one focused same-package test and update Tensor's exact public method inventory.
- Finalize Javadocs, Tensor API, Compile API, glossary, task evidence, master plan, and roadmap
  through the mandatory independent documentation pass.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- permute/transpose changes, permutation attributes, another operation kind/attributes type, or
  modification of completed task 0017F<span data-glossary-navigation="true"> ([transpose glossary](../../../../glossary.md#term-transpose); [permutation glossary](../../../../glossary.md#permutation); [operation glossary](../../../../glossary.md#operation))</span>
- automatic squeeze of all [singleton](../../../../glossary.md#term-singleton) [axes](../../../../glossary.md#term-axis), multi-axis insertion/removal, axis arrays/lists,
  `unsqueeze` [alias](../../../../glossary.md#aliasing-and-arena), [variadic](../../../../glossary.md#term-variadic) methods, or another public [Tensor](../../../../glossary.md#tensor) overload
- squeezing a dynamic dimension by assumption, binding a symbol to one, graph-wide constraint
  solving, or runtime-dependent result rank<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension); [binding glossary](../../../../glossary.md#term-binding); [symbol glossary](../../../../glossary.md#term-symbol); [graph glossary](../../../../glossary.md#graph); [runtime glossary](../../../../glossary.md#term-runtime); [rank glossary](../../../../glossary.md#term-rank))</span>
- changing Shape, Dimension, StaticDimension, DynamicDimension, LayoutDescriptor, LayoutKind,
  TensorDescriptor, TensorFactory, TensorProvenance, AxisTransformKind, AxisTransformAttrs,
  Operation, or their completed tests<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- reading/copying values, storage lookup/attachment, physical aliasing, allocation, materialization,
  mutation, capacity/lifetime checks, or zero-copy execution guarantees<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [aliasing glossary](../../../../glossary.md#term-aliasing); [materialization glossary](../../../../glossary.md#materialization); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- returning input, cancelling expand/squeeze pairs, canonicalizing nested edits, caching, interning,
  or CSE<span data-glossary-navigation="true"> ([CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>
- gradients, inverse rank-edit operations, autograd, training, graph capture, compiler passes,
  planning, prepare, backend lowering, runtime, execution, ONNX mapping, or conformance<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- another module, dependency, Gradle/build change, preview feature, architecture change, or task-
  0017G implementation/specification<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [preview feature glossary](../../../../glossary.md#term-preview-feature))</span>

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
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0017E](0017e-axis-transform-semantics.md)
- [Task 0017F](0017f-permute-and-transpose-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes `Tensor.expandDims(int)` and
`Tensor.squeeze(int)`. Insertion accepts positions through [rank](../../../../glossary.md#term-rank) inclusive, normalizes negative [axes](../../../../glossary.md#term-axis)
against rank plus one, inserts static [extent](../../../../glossary.md#term-extent) one, and derives a [view](../../../../glossary.md#term-view-meanings) [stride](../../../../glossary.md#term-stride) from the following input
axis or uses one at the end. Squeeze normalizes an existing axis, requires extent one, removes its
[Shape](../../../../glossary.md#shape)/stride entries, and preserves [storage offset](../../../../glossary.md#term-storage-offset). Legacy [gradients](../../../../glossary.md#term-gradient) use the inverse rank edit.

This task preserves [model](../../../../glossary.md#term-model)-level request [normalization](../../../../glossary.md#term-normalization), Shape/[layout](../../../../glossary.md#layout) [metadata](../../../../glossary.md#term-metadata), exact [operation](../../../../glossary.md#operation)
attributes, data-type/eligibility retention, and [provenance](../../../../glossary.md#provenance). It excludes immediate storage [aliasing](../../../../glossary.md#term-aliasing),
[mutable](../../../../glossary.md#term-mutable) [graph](../../../../glossary.md#graph) builders, gradient callbacks, operation traits, [compiler](../../../../glossary.md#term-compiler)/[lowering](../../../../glossary.md#lowering), [kernels](../../../../glossary.md#kernel), [runtime](../../../../glossary.md#term-runtime),
and [backend](../../../../glossary.md#backend) behavior.

## Architecture constraints

- [Tensor](../../../../glossary.md#tensor) remains public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state and is not an [IR](../../../../glossary.md#term-intermediate-representation-ir) [node](../../../../glossary.md#node).
- [Operation](../../../../glossary.md#operation) stores [backend](../../../../glossary.md#backend)-neutral meaning and no support, cost, route, or executable state.
- Shape and LayoutDescriptor are logical immutable metadata; resolved view geometry neither
  attaches physical storage nor guarantees zero-copy execution.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [metadata glossary](../../../../glossary.md#term-metadata); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- [Model](../../../../glossary.md#term-model) validates only locally provable insertion/removal facts.
- Dynamic [singleton](../../../../glossary.md#term-singleton) [binding](../../../../glossary.md#term-binding) and [graph](../../../../glossary.md#graph)-wide constraints belong to [compiler](../../../../glossary.md#term-compiler) [shape](../../../../glossary.md#shape) [inference](../../../../glossary.md#term-inference).
- Compiler owns graph capture/canonicalization; planning/backend prepare own materialization and
  concrete lowering; runtime executes prepared schedules.<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- No [service locator](../../../../glossary.md#term-service-locator), backend [lookup](../../../../glossary.md#term-lookup), new [dependency](../../../../glossary.md#term-dependency), or architecture change is authorized.

## Package impact

No package is added or moved.

- Public methods remain in `io.github.pho001.synaptik.model.tensor.Tensor`.
- Package-private `TensorRankEditingExpressions` lives in `model.tensor` beside other expression
  helpers.
- Focused `TensorRankEditingExpressionTest` mirrors that package.
- Existing shape, layout, operation-layout, descriptor, provenance, and factory packages are only
  consumed and remain unchanged.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) surface

Add exactly:

```java
public Tensor expandDims(int axis) {
    return TensorRankEditingExpressions.expandDims(this, axis);
}

public Tensor squeeze(int axis) {
    return TensorRankEditingExpressions.squeeze(this, axis);
}
```

Do not add [aliases](../../../../glossary.md#aliasing-and-arena), overloads, arrays, collections, or automatic all-[axis](../../../../glossary.md#term-axis) squeeze.

### Helper structure

`TensorRankEditingExpressions` is package-private, final, field-free, and non-instantiable. Apart
from one private zero-[argument](../../../../glossary.md#term-argument) constructor, it declares exactly:

```java
static Tensor expandDims(Tensor input, int axis)
static Tensor squeeze(Tensor input, int axis)
private static int normalizeInsertionAxis(int axis, int rank)
private static void validateSqueezableDimension(
        Shape inputShape, int normalizedAxis)
private static Shape insertSingleton(Shape inputShape, int normalizedAxis)
private static Shape removeSingleton(Shape inputShape, int normalizedAxis)
private static Optional<LayoutDescriptor> resolveInsertedLayout(
        TensorDescriptor inputDescriptor,
        Shape inputShape,
        Shape resultShape,
        int normalizedAxis)
private static Optional<LayoutDescriptor> resolveSqueezedLayout(
        TensorDescriptor inputDescriptor, Shape resultShape, int normalizedAxis)
private static Tensor create(
        Tensor input,
        TensorDescriptor inputDescriptor,
        Shape resultShape,
        Optional<LayoutDescriptor> resultLayout,
        AxisTransformKind kind,
        int normalizedAxis)
```

Do not add fields, nested types, another constructor, helper file, or hidden overload.

### Expand-[dimensions](../../../../glossary.md#dimension) [validation](../../../../glossary.md#term-validation) and order

`expandDims` performs exactly:

1. null-check input with exact message `input`;
2. read exact input [descriptor](../../../../glossary.md#term-descriptor) once and exact [Shape](../../../../glossary.md#shape) once;
3. normalize insertion [axis](../../../../glossary.md#term-axis);
4. derive result Shape;
5. resolve or defer inserted [layout](../../../../glossary.md#layout);
6. construct the result once with `EXPAND_DIMS`.

Normalize in `long`. If raw axis is negative, add `rank + 1` once; otherwise retain it. Valid
normalized positions are inclusive `[0, rank]`. Invalid values throw `IndexOutOfBoundsException`
with exact message:

```text
Axis <axis> is outside insertion range for shape rank <rank>
```

For [rank](../../../../glossary.md#term-rank) two, valid raw axes are `-3` through `2`; `-3` and `0` select the start, while `-1` and
`2` select the end. For [scalar](../../../../glossary.md#term-scalar) rank zero, both `-1` and `0` select its only insertion position.

### Squeeze [validation](../../../../glossary.md#term-validation) and order

`squeeze` performs exactly:

1. null-check input with exact message `input`;
2. read exact input [descriptor](../../../../glossary.md#term-descriptor) once and exact [Shape](../../../../glossary.md#shape) once;
3. normalize through `inputShape.normalizeAxis(axis)` exactly once;
4. validate the selected [Dimension](../../../../glossary.md#dimension) as a static [singleton](../../../../glossary.md#term-singleton);
5. derive result Shape;
6. resolve or defer squeezed [layout](../../../../glossary.md#layout);
7. construct the result once with `SQUEEZE`.

Invalid existing [axes](../../../../glossary.md#term-axis) retain Shape's `IndexOutOfBoundsException` and exact message
`Axis <axis> is outside shape rank <rank>`. A selected dimension other than
`StaticDimension(1)`, including zero, another static [extent](../../../../glossary.md#term-extent), or DynamicDimension, throws
`IllegalArgumentException` with exact message:

```text
cannot squeeze axis <normalizedAxis> of <inputShape>: dimension must be statically known as 1
```

[Scalar](../../../../glossary.md#term-scalar) squeeze always fails axis [normalization](../../../../glossary.md#term-normalization). Do not assume a dynamic [symbol](../../../../glossary.md#term-symbol) equals one.

### Result [Shapes](../../../../glossary.md#shape)

Insertion allocates one `Dimension[]` of length `inputRank + 1`, inserts exactly one new
`StaticDimension(1)` at normalizedAxis, and preserves exact input [Dimension](../../../../glossary.md#dimension) references before and
after it. Removal allocates one array of length `inputRank - 1`, skips the selected dimension, and
preserves every unaffected exact reference. Construct both through `Shape.ofDimensions`.

[Rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) squeeze may produce [canonical](../../../../glossary.md#term-canonical-form) [scalar](../../../../glossary.md#term-scalar) Shape. Do not mutate or retain arrays.

### Inserted [layout](../../../../glossary.md#layout)

Read `inputDescriptor.layout()` exactly once. If absent, return `Optional.empty()`.

For a present layout, allocate one result-[rank](../../../../glossary.md#term-rank) [stride](../../../../glossary.md#term-stride) array. Copy exact input strides around the
inserted position. The inserted stride is:

```text
1                                                     when normalizedAxis == inputRank
Math.multiplyExact(inputLayout.stride(normalizedAxis),
                   inputShape dimension size)         otherwise
```

The input [Shape](../../../../glossary.md#shape) is fully static when layout is present. The multiplication uses checked long
arithmetic and may fail before [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings) allocation. This deterministic [singleton](../../../../glossary.md#term-singleton) stride
preserves [canonical](../../../../glossary.md#term-canonical-form) dense classification when the input is canonical while remaining address-
equivalent for any resolved strided or broadcast [view](../../../../glossary.md#term-view-meanings).

Create one `LayoutDescriptor.of(resultShape, resultStrides, inputOffset, true)`. Preserve offset,
all existing raw strides, and explicit view [metadata](../../../../glossary.md#term-metadata). Let LayoutDescriptor derive kind/span.

### Squeezed [layout](../../../../glossary.md#layout)

Read `inputDescriptor.layout()` exactly once. If absent, return `Optional.empty()`.

For a present layout, allocate one result-[rank](../../../../glossary.md#term-rank) [stride](../../../../glossary.md#term-stride) array, omit the selected stride, and preserve
all other exact strides in order. Create one
`LayoutDescriptor.of(resultShape, resultStrides, inputOffset, true)`. Preserve exact offset and let
LayoutDescriptor derive kind/span. Never reuse input layout.

### Result construction

Create exactly:

```java
AxisTransformAttrs attrs = new AxisTransformAttrs(normalizedAxis);
TensorDescriptor descriptor = new TensorDescriptor(
        inputDescriptor.dataType(),
        resultShape,
        resultLayout,
        inputDescriptor.requiresGrad());
Operation operation = new Operation(kind, attrs);
TensorProvenance provenance = new TensorProvenance(operation, List.of(input));
return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
```

`kind` must be exactly EXPAND_DIMS or SQUEEZE according to the public entry. Every successful call
consumes one fresh [Tensor](../../../../glossary.md#tensor) ID and has absent label/storage. Failures before `createDerived` consume no
ID; identifier exhaustion occurs only at final delegation.

## Affected files

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorRankEditingExpressions.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRankEditingExpressionTest.java`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/tasks/0017f1-expand-dimensions-and-squeeze-tensor-expressions.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

At most these ten paths may change. Production is limited to [Tensor](../../../../glossary.md#tensor) and one package-private helper;
tests to Tensor inventory and one focused suite; documentation to public [API](../../../../glossary.md#term-application-programming-interface-api)/status and [planning](../../../../glossary.md#term-planning).

If another type, helper, overload, existing contract edit, [dependency](../../../../glossary.md#term-dependency), build/architecture change,
or eleventh path is required, stop and report. Do not create task 0017G.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Fully document both public methods and the helper type, constructor, and all nine methods.
- Explain insertion-axis versus existing-axis normalization, including negative boundaries and
  scalar cases.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Explain static [singleton](../../../../glossary.md#term-singleton) proof and why dynamic selected [dimensions](../../../../glossary.md#dimension) are rejected.
- Document exact Dimension-reference preservation and [rank](../../../../glossary.md#term-rank) changes.
- Explain inserted/removed stride geometry, checked multiplication, offset/view retention, all
  resolved layout kinds, and unresolved cases.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [layout glossary](../../../../glossary.md#layout))</span>
- Document result type, Shape, layout, eligibility, label, storage, operation, attributes,
  provenance, freshness, every failure, and identifier exhaustion.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance))</span>
- Explain that [view](../../../../glossary.md#term-view-meanings) [metadata](../../../../glossary.md#term-metadata) is not attached storage or execution proof.
- Independently review AxisTransformKind/Attrs, Shape/Dimension, layout, descriptor, factory,
  provenance, and neighboring view-expression Javadocs; stop on an out-of-scope discrepancy.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- [Tensor](../../../../glossary.md#tensor) exposes exactly `expandDims(int)` and `squeeze(int)`; public method inventory rises from
  83 to 85.
- Each method delegates once; helper has exact final/package-private/field-free/nine-method [shape](../../../../glossary.md#shape).
- Insertion and squeeze validation order, normalization, exception types/messages, scalar and
  negative boundaries match the specification.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [normalization glossary](../../../../glossary.md#term-normalization); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Squeeze accepts only exact static [singleton](../../../../glossary.md#term-singleton) and rejects zero/non-one/dynamic [dimensions](../../../../glossary.md#dimension).
- Result Shapes preserve exact unaffected Dimension references and insert/remove exactly one [axis](../../../../glossary.md#term-axis).
- Every resolved layout kind derives exact inserted/removed strides, offset, view flag, kind, and
  span; unresolved input remains unresolved.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Checked inserted-[stride](../../../../glossary.md#term-stride) [overflow](../../../../glossary.md#term-overflow) fails before ID consumption.
- All six data types and valid eligibility states retain exact [metadata](../../../../glossary.md#term-metadata).
- Results have exact kind/AxisTransformAttrs/[input], absent label/storage, and fresh [identity](../../../../glossary.md#term-identity-meanings).
- Repeated, nested, and inverse-like calls remain explicit and uncanonicalized.
- No values/storage/cross-layer behavior, [dependency](../../../../glossary.md#term-dependency), or architecture change.
- Independent documentation review and all validation/status synchronization complete before
  marking Complete.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorRankEditingExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact public/helper [API](../../../../glossary.md#term-application-programming-interface-api); all data types/eligibility; insertion positive/
negative/boundary/[scalar](../../../../glossary.md#term-scalar) [axes](../../../../glossary.md#term-axis); squeeze positive/negative/[rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)/scalar cases; static [singleton](../../../../glossary.md#term-singleton),
zero/non-one/dynamic rejection; exact [Dimension](../../../../glossary.md#dimension) [identity](../../../../glossary.md#term-identity-meanings); dense/offset/strided/broadcast [layouts](../../../../glossary.md#layout);
inserted/removed [strides](../../../../glossary.md#term-stride), offsets, kinds, [view](../../../../glossary.md#term-view-meanings) flags, spans; unresolved geometry; checked [overflow](../../../../glossary.md#term-overflow);
[operation](../../../../glossary.md#operation)/attributes/[provenance](../../../../glossary.md#provenance); absent label/storage; dead-storage non-interference; freshness; ID
side effects; and exhaustion.

Inspect `javap -p -c -s`, reflection, imports, and source. Confirm two one-call public methods,
exact nine-method helper, one insertion [normalization](../../../../glossary.md#term-normalization), one squeeze normalization/proof, two [Shape](../../../../glossary.md#shape)
paths, two layout paths, one shared construction, and no permute/[transpose](../../../../glossary.md#term-transpose)/value/storage/cross-layer
behavior. Validate generated [Javadoc](../../../../glossary.md#term-javadoc), executable examples, [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) API, glossary, links/
anchors/fences/whitespace, exact ten paths, synchronized status, and no task-0017G spec.

## Dependencies

- 0002 supplies [immutable](../../../../glossary.md#term-immutable) [Dimension](../../../../glossary.md#dimension)/[Shape](../../../../glossary.md#shape), [axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization), [scalar](../../../../glossary.md#term-scalar)/zero/dynamic forms.
- 0003 supplies resolved [stride](../../../../glossary.md#term-stride)/offset/[view](../../../../glossary.md#term-view-meanings) geometry and classification.
- 0007 supplies resolved-or-unresolved TensorDescriptor.
- 0011–0013 supply [Tensor](../../../../glossary.md#tensor), centralized derived [identity](../../../../glossary.md#term-identity-meanings), and [provenance](../../../../glossary.md#provenance).
- 0017E supplies EXPAND_DIMS/SQUEEZE and AxisTransformAttrs.
- 0017F provides adjacent axis-transform expression patterns but no hard production [dependency](../../../../glossary.md#term-dependency).

## Follow-up tasks

- 0017G remains Draft for [slice](../../../../glossary.md#slice) semantics.
- [Compiler](../../../../glossary.md#term-compiler) later owns inverse-pair/nested [rank](../../../../glossary.md#term-rank)-edit [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) and dynamic constraints.
- [Planning](../../../../glossary.md#term-planning)/[backend](../../../../glossary.md#backend) [prepare](../../../../glossary.md#prepare) later own [materialization](../../../../glossary.md#materialization) and [view](../../../../glossary.md#term-view-meanings)/copy [lowering](../../../../glossary.md#lowering).
- Training/compiler-generated semantics later own inverse rank-edit [gradients](../../../../glossary.md#term-gradient).

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. [Model](../../../../glossary.md#term-model) owns every composed contract; [compiler](../../../../glossary.md#term-compiler), [planning](../../../../glossary.md#term-planning), [prepare](../../../../glossary.md#prepare), [backend](../../../../glossary.md#backend),
[runtime](../../../../glossary.md#term-runtime), and training [ownership](../../../../glossary.md#term-ownership-meanings) remains unchanged.

Stop if implementation needs symbolic [binding](../../../../glossary.md#term-binding), storage [aliasing](../../../../glossary.md#term-aliasing), [gradient](../../../../glossary.md#term-gradient) rules, cross-layer
behavior, another [dependency](../../../../glossary.md#term-dependency), or architecture change.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0003/0007/0011/0012/0013/0017E/0017F/0017F1, Tensor
API, Compile API, Training API, glossary, current Dimension/Shape/LayoutDescriptor/LayoutKind/
TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/AxisTransformKind/
AxisTransformAttrs and focused expression contracts/tests, and Java 26 Gradle configuration.

Implement task 0017F1 exactly. Modify Tensor.java and add package-private final
TensorRankEditingExpressions.java. Update TensorTest only for exact two-method API expansion and add
TensorRankEditingExpressionTest. Add exactly expandDims(int) and squeeze(int).

The field-free helper has exactly nine specified methods. Normalize insertion axes against rank+1;
normalize squeeze through Shape and accept only a statically known singleton. Preserve exact
unaffected Dimension references. For any resolved layout, insert the checked following-axis
stride (or one at the end) or remove the selected stride, preserving offset in a new view;
unresolved input stays unresolved. Preserve type/eligibility, create exact kind/
AxisTransformAttrs/[input], and call createDerived once with no label/storage. Every call is fresh.

Do not modify semantic/foundational/completed permutation contracts, assume dynamic singleton,
inspect/copy values or storage, attach aliases, canonicalize, add APIs/helpers/types, define
gradients, capture graphs, or add cross-layer behavior, dependencies, build/architecture changes,
or later specs. Stop beyond ten paths or on architecture uncertainty.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/Compile API/glossary/planning, record no-change conclusions, and rerun validation.

Update task 0017F1, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0017G Draft without a specification. Do not commit/push.
```

## Local decisions

- The former combined expression frontier remains split because singleton rank editing has
  different normalization and layout algebra from permutation.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [normalization glossary](../../../../glossary.md#term-normalization); [layout glossary](../../../../glossary.md#layout); [permutation glossary](../../../../glossary.md#permutation))</span>
- Expand-[dimensions](../../../../glossary.md#dimension) negative [axes](../../../../glossary.md#term-axis) normalize against [rank](../../../../glossary.md#term-rank) plus one, so `-1` inserts at the end.
- Squeeze uses Shape's established existing-axis normalization and requires local static proof of
  extent one; dynamic dimensions are not guessed or constrained here.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [extent glossary](../../../../glossary.md#term-extent))</span>
- Insertion stride follows legacy-compatible deterministic view geometry: following stride times
  following extent, or one at the end, with checked arithmetic.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Squeeze removes only the selected stride. Both paths preserve offset and mark resolved results as
  logical views without storage attachment or execution guarantees.
- Every valid call is fresh; [compiler](../../../../glossary.md#term-compiler) owns inverse-pair and nested [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form).

## Known limitations

- Dynamic [dimensions](../../../../glossary.md#dimension) cannot be squeezed even if a future [binding](../../../../glossary.md#term-binding) could equal one.
- No all-axis squeeze, multi-axis edit, host-storage alias, gradient inverse, compiler capture,
  materialization, backend lowering, runtime execution, ONNX mapping, or conformance behavior.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [host glossary](../../../../glossary.md#term-host); [alias glossary](../../../../glossary.md#aliasing-and-arena); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [materialization glossary](../../../../glossary.md#materialization); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed architecture/focused boundary docs; documentation/planning rules; roadmap; [model](../../../../glossary.md#term-model)
capabilities/master plan; prerequisite tasks; current [Dimension](../../../../glossary.md#dimension)/[Shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout), [descriptor](../../../../glossary.md#term-descriptor),
[Tensor](../../../../glossary.md#tensor)/factory/[provenance](../../../../glossary.md#provenance), [Operation](../../../../glossary.md#operation), AxisTransform semantics, completed [permutation](../../../../glossary.md#permutation) and neighboring
[view](../../../../glossary.md#term-view-meanings)-expression source/tests; Tensor/[Compile](../../../../glossary.md#compile)/Training APIs, glossary, and Java 26 [Gradle](../../../../glossary.md#term-gradle).

The read-only legacy branch confirms insertion-[axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization) against [rank](../../../../glossary.md#term-rank) plus one, selected-
axis squeeze, [singleton](../../../../glossary.md#term-singleton) proof, Shape/[stride](../../../../glossary.md#term-stride) insertion/removal, offset-preserving views, and inverse
[gradient](../../../../glossary.md#term-gradient) capability. Coupled storage, [graph](../../../../glossary.md#graph), gradient, [compiler](../../../../glossary.md#term-compiler), [kernel](../../../../glossary.md#kernel), [runtime](../../../../glossary.md#term-runtime), and [backend](../../../../glossary.md#backend) design
is excluded.

Existing contracts support the exact methods/helper without another public type, package,
[dependency](../../../../glossary.md#term-dependency), foundational edit, or architecture decision. Initial planning validation is recorded
below; implementation and independent documentation evidence follow it.

Planning validation after [synchronization](../../../../glossary.md#term-synchronization):

- `git diff --check` passed, and targeted trailing-whitespace scans found no matches.
- Exact planning scope is three paths: this task, model master plan, and roadmap. No Java, Gradle,
  architecture, API, glossary, completed-task, or other-module file changed during planning.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module))</span>
- All 20 [canonical](../../../../glossary.md#term-canonical-form) task sections are present.
- Markdown backtick fences are balanced: sixteen in this task, two in the master plan, and zero in
  the roadmap.
- All 211 local Markdown file links across the three changed planning files resolve.
- Every changed file ends with a newline.
- At initial planning time, task, master plan, and roadmap consistently queued 0017F1 for execution
  while leaving 0017G Draft; no detailed 0017G specification existed or has since been created.
- The model task sequence contains 74 ordered rows with no duplicate order number.
- Package review confirms no new package: public methods and helper stay in `model.tensor`, and the
  focused test mirrors that package.
- Scope review confirms exactly ten permitted implementation paths and no need for another public
  type, dependency, foundational edit, or architecture decision.
- Granularity review confirms singleton insertion/removal is one cohesive rank-editing task and
  remains separate from completed permutation and future slice semantics.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>

## Implementation notes

- [`Tensor`](../../../../glossary.md#tensor) now exposes exactly `expandDims(int)` and `squeeze(int)`, each as one direct delegation
  to the matching package-private helper entry.
- New final, field-free `TensorRankEditingExpressions` has one private constructor and exactly the
  specified nine methods. It performs one insertion normalization or one existing-axis
  normalization/static-singleton proof, builds one rank-edited Shape, conditionally builds one
  same-offset logical view layout, and delegates once to shared derived construction.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [singleton glossary](../../../../glossary.md#term-singleton); [rank glossary](../../../../glossary.md#term-rank); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Inserted [Shapes](../../../../glossary.md#shape) retain every unaffected [Dimension](../../../../glossary.md#dimension) reference and add one new
  `StaticDimension(1)`. Squeezed Shapes retain every unaffected reference and reject zero,
  non-one, and dynamic selected dimensions. Resolved scalar, empty, dense, offset, strided, and
  broadcast geometry follows the required checked stride rules; unresolved geometry remains
  unresolved.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [stride glossary](../../../../glossary.md#term-stride))</span>
- Both paths retain exact input [data type](../../../../glossary.md#data-type--datatype) and [gradient](../../../../glossary.md#term-gradient) eligibility and record exact
  `AxisTransformKind`, `AxisTransformAttrs(normalizedAxis)`, and ordered `[input]` provenance.
  Every valid result is fresh, unlabeled, and storage-free.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- `TensorTest` now fixes the public method inventory at 85. The focused rank-editing suite covers
  both normalization domains, scalar and empty geometry, all data types/eligibility, exact
  Dimension identity, every resolved layout category, unresolved layout, overflow and validation
  side effects, provenance, freshness, storage non-interference, and identifier exhaustion.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- The independent documentation context was
  `/root/implement_model_0017f1/review_model_0017f1_docs`. It applied the General,
  API/Javadoc, Planning, and Example profiles; independently reviewed final source, tests,
  generated Javadoc, neighboring contracts, and the exact diff; and retained the complete
  production Javadocs unchanged. It finalized the Tensor API, Compile API, glossary, task, master
  plan, and roadmap, including one executable Java 26 rank-editing example.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning); [Compile glossary](../../../../glossary.md#compile))</span>
- No Training API or capability-baseline update was needed: rank editing changes neither training
  state nor the already listed model capability. No semantic/foundational/completed permutation
  contract changed because this task composes them through separate rank-editing algebra. No
  architecture document, ADR, architecture test, backend-conformance test, integration test,
  Gradle file, dependency, or other module changed because no boundary, dependency, executable,
  backend, or end-to-end behavior changed.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [permutation glossary](../../../../glossary.md#permutation); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [architecture test glossary](../../../../glossary.md#term-architecture-test); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>

Final [validation](../../../../glossary.md#term-validation) evidence:

- The focused [Gradle](../../../../glossary.md#term-gradle) command passed 11 `TensorRankEditingExpressionTest` and 14 `TensorTest` cases
  with zero failures, errors, or skips. Full model tests passed 529 cases with zero failures,
  errors, or skips.
- `./gradlew :modules:model:javadoc` and `./gradlew test` passed; the root run reported 36
  actionable tasks. Generated Javadoc contains both public methods and their complete result,
  failure, axis, layout, storage, provenance, and freshness contracts.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- The documented Java 26 `RankEditingExpressionExample` compiled and ran. It printed input Shape
  `[2, 1, 3]`, expanded Shape `[2, 1, 3, 1]`, normalized insertion [axis](../../../../glossary.md#term-axis) `3`, squeezed Shape
  `[2, 3]`, normalized removal axis `1`, exact kinds, unresolved [layouts](../../../../glossary.md#layout), and absent storage.
- `javap -p -c -s`, reflection tests, import scans, and source inspection confirm two one-call
  public delegates, exactly 85 public Tensor methods, the final/package-private/field-free helper
  with one private constructor and exactly nine methods, one normalization/proof path per
  operation, two Shape paths, two layout paths, and one shared construction path.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Local Markdown links and anchors, balanced fences, trailing whitespace, final newlines,
  generated Javadoc text, and `git diff --check` passed. Exactly the ten permitted paths changed.
  Task/master/roadmap status is synchronized at Complete, 0017G remains Draft, and no 0017G task
  specification exists.

## Completion summary

- Completed changes: Added exact singleton-axis insertion/removal Tensor expressions, their bounded
  helper, focused tests, the 85-method public inventory, and synchronized status/evidence.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [axis glossary](../../../../glossary.md#term-axis); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Files changed/created: Exactly the ten paths listed under Affected files.
- Tests/validation: Focused 25-test selection, 529-test model suite, model Javadoc, root test
  lifecycle, Java 26 example, bytecode/reflection/import/source checks, 264 local links/anchors,
  fences/whitespace/newlines, exact scope/status, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [lifecycle glossary](../../../../glossary.md#lifecycle); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0017f1/review_model_0017f1_docs` independently reviewed final source,
  tests, generated Javadoc, related contracts, and the actual diff under the General,
  API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API gained the complete current rank-editing contract and runnable
  example; Compile API gained current expression inventory without compiler claims; task, master
  plan, and roadmap now record completion. Training API, capabilities, architecture/ADRs/tests,
  conformance/integration, Gradle/dependencies, and other modules remain accurate unchanged.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [Compile glossary](../../../../glossary.md#compile); [compiler glossary](../../../../glossary.md#term-compiler); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>
- Javadoc review: New Tensor and helper Javadocs are complete and accurate; related
  AxisTransformKind/Attrs, Shape/Dimension, layout, descriptor, factory, provenance, and neighboring
  view-expression Javadocs remain accurate unchanged.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Glossary impact: Existing implementation-status, axis-transform, normalized-axis, provenance,
  Tensor, and common-distinction text now records current expand-dimensions/squeeze construction;
  no new glossary term was needed.
- Unresolved issues: None within task scope; the documented Known limitations are intentional
  layer boundaries.
- Follow-up required: None for task 0017F1. Task 0017G remains Draft without a specification.

Status: Complete
