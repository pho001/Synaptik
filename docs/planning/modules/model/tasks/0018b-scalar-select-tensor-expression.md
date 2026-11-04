# Task 0018B: [Scalar Select](../../../../glossary.md#scalar-select) [Tensor](../../../../glossary.md#tensor) Expression

## Status

Complete

## Goal

Add the public [model](../../../../glossary.md#term-model)-level expression that selects one [scalar](../../../../glossary.md#term-scalar) [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) from one [Tensor](../../../../glossary.md#tensor) [axis](../../../../glossary.md#term-axis).

The expression normalizes a positive or negative axis, normalizes and checks the scalar index when
the selected [extent](../../../../glossary.md#term-extent) is statically known, removes the selected axis from the result [Shape](../../../../glossary.md#shape), derives
[view](../../../../glossary.md#term-view-meanings) geometry only when it is locally provable, and records exact `SELECT` semantics with
one-input [provenance](../../../../glossary.md#provenance). It constructs [immutable](../../../../glossary.md#term-immutable) expression [metadata](../../../../glossary.md#term-metadata) only: it neither reads values
nor promises that a physical storage [alias](../../../../glossary.md#aliasing-and-arena) will be used during execution.

For input Shape `[2, 3, 4]`, `select(1, 2)` produces Shape `[2, 4]`. The same selection can be
written `select(-2, -1)` because the selected extent is statically known as `3`.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly one public instance method to [`Tensor`](../../../../glossary.md#tensor):
  `Tensor select(int axis, long index)`.
- Add one field-free package-private final `TensorSelectExpressions` helper in `model.tensor`.
- Give the helper exactly one package-private entry and four private methods specified below.
- Null-check the helper input, read its immutable descriptor and Shape, and normalize the raw axis
  exactly once through <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [descriptor glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape); [axis glossary](../../../../glossary.md#term-axis))</span>`Shape.normalizeAxis`.
- For a static selected extent, normalize one negative index by adding the extent once and require
  the normalized coordinate to be in bounds.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- For a dynamic selected extent, accept a non-negative index unchanged with deferred bounds
  validation and reject a negative index because local normalization is impossible.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Remove the selected [Dimension](../../../../glossary.md#dimension) while preserving every unaffected Dimension reference exactly.
- For resolved non-empty input geometry, remove the selected stride and advance the storage offset
  by <span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [storage offset glossary](../../../../glossary.md#term-storage-offset))</span>`normalizedIndex * selectedStride` using checked arithmetic, then construct one new [view](../../../../glossary.md#term-view-meanings)
  `LayoutDescriptor`.
- Leave result [layout](../../../../glossary.md#layout) unresolved when input layout is unresolved or result [element](../../../../glossary.md#term-element) count is zero.
- Preserve exact input [data type](../../../../glossary.md#data-type--datatype) and [gradient](../../../../glossary.md#term-gradient) eligibility for every current data type.
- Construct exact `SelectKind.SELECT`, `SelectAttrs`, and ordered [provenance](../../../../glossary.md#provenance) `[input]` and delegate
  once to `TensorFactory.createDerived` with no label or storage.
- Keep every valid request explicit and fresh, including repeated, nested, same-coordinate, and
  rank-one-to-scalar selection.<span data-glossary-navigation="true"> ([rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Update `TensorTest` only for the deliberate public API expansion and add one focused expression
  test.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Finalize Javadocs, Tensor API, Compile API, glossary, task evidence, master plan, and roadmap
  through the mandatory independent documentation pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- another public overload, static [facade](../../../../glossary.md#term-facade), `int`-specific overload, default axis/index, index Tensor,
  collection of indices, mask, range, slice, gather, take, or scatter behavior<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [Tensor glossary](../../../../glossary.md#tensor); [mask glossary](../../../../glossary.md#term-boolean-mask); [gather glossary](../../../../glossary.md#gather))</span>
- accepting a negative index for a dynamic selected extent, inventing a symbolic extent, adding a
  runtime-bound index expression, or silently retaining an unnormalized negative sentinel<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- eagerly rejecting every non-negative index on a dynamic extent; its upper-bound validation is
  intentionally deferred until that extent is bound<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- changing `SelectKind`, `SelectAttrs`, [`Shape`](../../../../glossary.md#shape), [`Dimension`](../../../../glossary.md#dimension), `LayoutDescriptor`,
  `TensorDescriptor`, `TensorFactory`, `TensorProvenance`, [`Operation`](../../../../glossary.md#operation), or any completed Java
  contract/test other than the exact Tensor API inventory<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- reading, copying, selecting, or materializing values; attaching or observing host storage;
  asserting a physical alias, zero-copy execution, or backend view support<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage); [alias glossary](../../../../glossary.md#aliasing-and-arena); [backend glossary](../../../../glossary.md#backend); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- returning the input, rewriting select as [`SLICE`](../../../../glossary.md#slice), `UNSTACK`, or gather, folding nested selects,
  canonicalizing coordinates, or eliminating expressions<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- defining select backward/scatter-add, gradient rules, autograd, graph capture, compiler passes,
  planning requirements, prepare, runtime, backend lowering/kernels, engine, trace, ONNX, training,
  or execution behavior<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- another production helper/type, dependency, Gradle/build option, architecture change, another
  module, or task-0018C specification<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Documentation profile index](../../../../developer-guide/documentation/README.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0015E](0015e-where-selection-semantic-kind.md)
- [Task 0017G](0017g-slice-semantics.md)
- [Task 0017H](0017h-slice-tensor-expressions.md)
- [Task 0017K](0017k-tensor-composition-semantics.md)
- [Task 0017L](0017l-tensor-composition-expressions.md)
- [Task 0018A](0018a-scalar-select-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The selected baseline requires public [scalar](../../../../glossary.md#term-scalar)-[axis](../../../../glossary.md#term-axis) selection. The read-only legacy branch exposes
an input, one raw axis, and one raw scalar index. It normalizes both against a concrete static
[shape](../../../../glossary.md#shape), removes the selected axis, removes its [stride](../../../../glossary.md#term-stride), advances the [storage offset](../../../../glossary.md#term-storage-offset) by index times
that stride, records `SELECT`, and defines backward through scatter-add.

The new [model](../../../../glossary.md#term-model) preserves the public forward-expression capability while using current contracts:
long-valued Shape [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), explicit static/dynamic [Dimension](../../../../glossary.md#dimension) values, [immutable](../../../../glossary.md#term-immutable) optional [layout](../../../../glossary.md#layout),
typed `SELECT` semantics, and immutable [Tensor](../../../../glossary.md#tensor) [provenance](../../../../glossary.md#provenance). It deliberately broadens the locally
representable dynamic case: a non-negative scalar coordinate can be retained when the selected
[extent](../../../../glossary.md#term-extent) is symbolic because axis removal does not require its size. A negative index still needs a
known extent and is therefore rejected for a dynamic selected Dimension.

Legacy storage [aliases](../../../../glossary.md#aliasing-and-arena), [eager](../../../../glossary.md#term-eager-evaluation) values, [gradient](../../../../glossary.md#term-gradient) callbacks, [graph](../../../../glossary.md#graph) builders, [operation](../../../../glossary.md#operation) traits,
[compiler](../../../../glossary.md#term-compiler)/[lowering](../../../../glossary.md#lowering) code, [kernels](../../../../glossary.md#kernel), and [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) behavior are capability evidence only and are
not copied.

## Architecture constraints

- [`Tensor`](../../../../glossary.md#tensor) remains public mutable API state, not compiled IR. The method constructs a fresh
  storage-free expression through the existing package-private derived-factory seam.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Semantic [identity](../../../../glossary.md#term-identity-meanings) and normalized attributes come only from completed task 0018A.
- The helper may inspect immutable descriptor, Shape, Dimension, and optional layout metadata, but
  never values, host storage, runtime residency, device state, or backend capability.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [descriptor glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [layout glossary](../../../../glossary.md#layout); [metadata glossary](../../../../glossary.md#term-metadata); [host storage glossary](../../../../glossary.md#host-storage); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [device glossary](../../../../glossary.md#term-device); [backend capability glossary](../../../../glossary.md#backend-capability))</span>
- Axis and static-index normalization are local request validation. A dynamic non-negative index
  is normalized already but has deferred upper-bound validation; no hidden symbolic constraint is
  added to Shape.<span data-glossary-navigation="true"> ([Axis glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization); [validation glossary](../../../../glossary.md#term-validation))</span>
- Result Shape removal is locally exact even when the selected Dimension is dynamic. Every
  unaffected Dimension reference is retained exactly.
- Resolved layout is logical view metadata only. It neither attaches storage nor promises physical
  aliasing or zero-copy execution.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- A result with zero known elements has unresolved layout because it references no storage element
  and needs no arbitrary offset geometry.<span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element))</span>
- Compiler owns capture/canonicalization and later dynamic-bound validation placement; compiler-
  generated/training work owns select backward; planning/prepare/backend own materialization,
  lowering, and execution.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering))</span>
- No [dependency](../../../../glossary.md#term-dependency), package [ownership](../../../../glossary.md#term-ownership-meanings), or [module](../../../../glossary.md#term-module) boundary changes are authorized.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.tensor.Tensor` — receives the one public expression surface.
- `io.github.pho001.synaptik.model.tensor.TensorSelectExpressions` — owns local normalization,
  Shape/layout derivation, and semantic construction.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout))</span>
- `TensorSelectExpressionTest` — mirrors `model.tensor` for focused helper/[API](../../../../glossary.md#term-application-programming-interface-api) [validation](../../../../glossary.md#term-validation).
- `TensorTest` — changes only its exact public API inventory and reflection assertions.

`SelectKind` and `SelectAttrs` remain in `model.operation.index`; no semantic contract moves into
the [Tensor](../../../../glossary.md#tensor) package.

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) method

Add exactly:

```java
public Tensor select(int axis, long index) {
    return TensorSelectExpressions.apply(this, axis, index);
}
```

The method contains one return statement and exactly one helper call. It is non-static,
non-synchronized, performs no direct [validation](../../../../glossary.md#term-validation) or field access, and adds no overload.

### Helper [shape](../../../../glossary.md#shape)

Create one package-private final, field-free class with one private zero-[argument](../../../../glossary.md#term-argument) constructor and
exactly these five static methods:

```java
static Tensor apply(Tensor input, int axis, long index)
private static long normalizeIndex(Dimension selected, int normalizedAxis, long rawIndex)
private static Shape removeAxis(Shape inputShape, int normalizedAxis)
private static Optional<LayoutDescriptor> resolveViewLayout(
        TensorDescriptor inputDescriptor,
        Shape resultShape,
        int normalizedAxis,
        long normalizedIndex)
private static Tensor create(
        Tensor input,
        TensorDescriptor inputDescriptor,
        Shape resultShape,
        Optional<LayoutDescriptor> resultLayout,
        SelectAttrs attrs)
```

Add no field, nested type, alternate constructor, overload, [cache](../../../../glossary.md#term-cache), [mutable](../../../../glossary.md#term-mutable) state, or extra method.

### Entry [validation](../../../../glossary.md#term-validation) and construction order

`apply` performs exactly this order:

1. null-check `input` with `Objects.requireNonNull(input, "input")`;
2. read the exact input `TensorDescriptor` once;
3. read its exact [`Shape`](../../../../glossary.md#shape) once;
4. normalize [`axis`](../../../../glossary.md#term-axis) exactly once through `inputShape.normalizeAxis(axis)`;
5. read the exact selected [`Dimension`](../../../../glossary.md#dimension) once;
6. call `normalizeIndex` once;
7. construct one `SelectAttrs(normalizedAxis, normalizedIndex)`;
8. call `removeAxis` once;
9. call `resolveViewLayout` once;
10. call `create` once and return its result.

All local validation, Shape construction, [layout](../../../../glossary.md#layout) arithmetic, [descriptor](../../../../glossary.md#term-descriptor) construction, [Operation](../../../../glossary.md#operation),
and [provenance](../../../../glossary.md#provenance) creation occur before identifier allocation. A local failure consumes no [Tensor](../../../../glossary.md#tensor)
identifier. Identifier exhaustion occurs only at the final `createDerived` call.

### [Axis](../../../../glossary.md#term-axis) and index [normalization](../../../../glossary.md#term-normalization)

Axis behavior is exactly the current `Shape.normalizeAxis(int)` contract:

- non-negative axes address the input from the front;
- negative axes add [rank](../../../../glossary.md#term-rank) once;
- invalid axes, including every axis for a [scalar](../../../../glossary.md#term-scalar) input, throw `IndexOutOfBoundsException` with
  the existing Shape message <span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>`Axis <raw> is outside shape rank <rank>`.

`normalizeIndex` distinguishes the selected [Dimension](../../../../glossary.md#dimension) category.

For `StaticDimension(size)`:

1. copy `rawIndex` into a local `long normalized`;
2. if negative, add `size` once;
3. if the result is negative or at least `size`, throw `IndexOutOfBoundsException` with exact
   message `select index <rawIndex> is outside axis <normalizedAxis> extent <size>`;
4. return the normalized non-negative [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix).

This naturally rejects every index for a zero-sized selected [extent](../../../../glossary.md#term-extent). Adding a non-negative static
extent once to a negative long cannot [overflow](../../../../glossary.md#term-overflow).

For a dynamic selected Dimension:

- reject a negative raw index with `IllegalArgumentException` and exact message
  `select index <rawIndex> cannot be normalized against dynamic axis <normalizedAxis>`;
- return a non-negative raw index unchanged without claiming that it is in bounds.

Do not use `-1` or another sentinel in `SelectAttrs`.

### Result [Shape](../../../../glossary.md#shape)

`removeAxis` allocates one `Dimension[]` of length `inputRank - 1`, copies every unaffected exact
[Dimension](../../../../glossary.md#dimension) reference in original order, and calls `Shape.ofDimensions` once. Selecting the only
[axis](../../../../glossary.md#term-axis) of a [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [Tensor](../../../../glossary.md#tensor) returns [canonical](../../../../glossary.md#term-canonical-form) [scalar](../../../../glossary.md#term-scalar) Shape. A selected dynamic Dimension disappears;
unselected dynamic Dimensions remain exact references.

Examples:

| Input Shape | Request | Normalized attrs | Result Shape |
|---|---|---|---|
| `[2, 3, 4]` | `select(1, 2)` | axis `1`, index `2` | `[2, 4]` |
| `[2, 3, 4]` | `select(-2, -1)` | axis `1`, index `2` | `[2, 4]` |
| `[N, 4]` | `select(0, 7)` | axis `0`, index `7`, bounds deferred | `[4]` |
| `[5]` | `select(0, 0)` | axis `0`, index `0` | scalar `[]` |

### Result [layout](../../../../glossary.md#layout)

`resolveViewLayout` reads the input layout Optional once.

- If input layout is unresolved, return `Optional.empty()`.
- If the fully static result [Shape](../../../../glossary.md#shape) has known [element](../../../../glossary.md#term-element) count zero, return `Optional.empty()`.
- Otherwise copy the input strides once, remove the selected stride while preserving order, and
  calculate the result offset exactly as:

```java
long resultOffset = Math.addExact(
        inputLayout.storageOffset(),
        Math.multiplyExact(normalizedIndex, inputStrides[normalizedAxis]));
```

- Construct exactly one
  `LayoutDescriptor.of(resultShape, resultStrides, resultOffset, true)` and return it present.

Every resolved input layout is accepted, including dense, offset, strided, and zero-[stride](../../../../glossary.md#term-stride) [view](../../../../glossary.md#term-view-meanings)
geometry. The [descriptor](../../../../glossary.md#term-descriptor) reclassifies the reduced geometry. A dynamic input cannot have resolved
layout under current `TensorDescriptor`, so a dynamic selected [extent](../../../../glossary.md#term-extent) always produces unresolved
result geometry even if removing that [axis](../../../../glossary.md#term-axis) makes the result Shape fully static.

For [contiguous](../../../../glossary.md#term-contiguous) `[2, 3, 4]` with strides `[12, 4, 1]`, selecting axis `1`, index `2` produces
Shape `[2, 4]`, strides `[12, 1]`, and [storage offset](../../../../glossary.md#term-storage-offset) `8`, marked as a view. For input `[2, 3, 0]`,
selecting axis `1`, index `1` produces empty Shape `[2, 0]` with unresolved layout.

### [Descriptor](../../../../glossary.md#term-descriptor), semantics, [provenance](../../../../glossary.md#provenance), and [identity](../../../../glossary.md#term-identity-meanings)

`create` constructs exactly:

```java
TensorDescriptor descriptor = new TensorDescriptor(
        inputDescriptor.dataType(),
        resultShape,
        resultLayout,
        inputDescriptor.requiresGrad());
Operation operation = new Operation(SelectKind.SELECT, attrs);
TensorProvenance provenance = new TensorProvenance(operation, List.of(input));
return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
```

Every current [data type](../../../../glossary.md#data-type--datatype) is accepted. Exact type and `requiresGrad` are retained. Every valid call
returns a fresh [Tensor](../../../../glossary.md#tensor) with no label, no [host storage](../../../../glossary.md#host-storage), exact one-input provenance, and the exact
normalized `SelectAttrs` reference. Fresh identity does not imply [eager](../../../../glossary.md#term-eager-evaluation) values or execution.

## Affected files

Expected implementation change:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorSelectExpressions.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSelectExpressionTest.java`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the ten paths listed above.

If implementation requires another production type, semantic-contract change, another test,
capability-baseline edit, architecture document, [dependency](../../../../glossary.md#term-dependency), build change, another [module](../../../../glossary.md#term-module), or more
than ten paths, stop and propose a follow-up task. Do not trade away [validation](../../../../glossary.md#term-validation) or documentation
to fit the limit.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document the new public [Tensor](../../../../glossary.md#tensor) method and every helper method, including the private constructor.
- Explain [axis](../../../../glossary.md#term-axis) and index [normalization](../../../../glossary.md#term-normalization) separately, including the dynamic selected-[extent](../../../../glossary.md#term-extent) policy.
- Include the `[2, 3, 4]` positive and negative request examples with concrete normalized/result
  values.
- Explain exact [Shape](../../../../glossary.md#shape) reference retention and [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)-to-[scalar](../../../../glossary.md#term-scalar) behavior.
- Explain conditional [view](../../../../glossary.md#term-view-meanings) geometry with the concrete [strides](../../../../glossary.md#term-stride) `[12, 4, 1]` to `[12, 1]` and
  offset `8` calculation.
- Explain empty-result and unresolved-[layout](../../../../glossary.md#layout) behavior without promising a physical [alias](../../../../glossary.md#aliasing-and-arena).
- Explain exact type/eligibility retention, one-input provenance, fresh identity, and absence of
  label/storage/value execution.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Distinguish [scalar select](../../../../glossary.md#scalar-select) from `WHERE`, `UNSTACK`, [`SLICE`](../../../../glossary.md#slice), and tensor-index [gather](../../../../glossary.md#gather).
- Do not promise [gradient](../../../../glossary.md#term-gradient) rules, [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings), [backend](../../../../glossary.md#backend) support, [materialization](../../../../glossary.md#materialization), or execution.

## Acceptance criteria

- [`Tensor`](../../../../glossary.md#tensor) has exactly one new public non-static, non-synchronized
  `select(int, long): Tensor` method and no overload.
- The public method delegates once to the exact helper and performs no other work.
- The helper is package-private, final, field-free, has one private constructor, and exactly the
  five specified static methods.
- Input null, axis normalization, static index normalization/bounds, and dynamic negative-index
  behavior follow the exact order, exception types, and messages.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>
- Dynamic non-negative selected indices remain representable with unresolved layout and deferred
  upper-bound validation.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- Result [Shape](../../../../glossary.md#shape) removes exactly one [axis](../../../../glossary.md#term-axis) and preserves every unaffected [Dimension](../../../../glossary.md#dimension) reference.
- Resolved non-empty layout removes the selected stride, advances offset with checked arithmetic,
  and is marked as a new view; unresolved and empty cases remain unresolved.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Exact input [data type](../../../../glossary.md#data-type--datatype) and [gradient](../../../../glossary.md#term-gradient) eligibility are retained for all six current types.
- Every valid call creates exact `SELECT`/`SelectAttrs`, ordered `[input]` provenance, no label or
  storage, and fresh identity.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- No value access, storage access, gradient rule, graph capture, compiler/runtime/backend behavior,
  semantic-contract modification, dependency, build change, or architecture change is introduced.<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Tensor API, Compile API, glossary, task evidence, master plan, and roadmap are independently
  reviewed and synchronized; Training API and capability/architecture docs record reasoned
  no-change conclusions.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>
- All [validation](../../../../glossary.md#term-validation) passes and the final diff contains exactly the ten permitted paths.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run):

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.tensor.TensorSelectExpressionTest
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused tests must verify:

- exact public [Tensor](../../../../glossary.md#tensor) [signature](../../../../glossary.md#term-signature), modifiers, single helper delegation, and public [API](../../../../glossary.md#term-application-programming-interface-api) count;
- exact helper package, visibility, final/field-free [shape](../../../../glossary.md#shape), private constructor, and five methods;
- input-null failure through direct package-private helper invocation;
- positive/negative [axis](../../../../glossary.md#term-axis) and positive/negative static-index [normalization](../../../../glossary.md#term-normalization);
- exact static out-of-bounds, zero-[extent](../../../../glossary.md#term-extent), dynamic-negative, and [scalar](../../../../glossary.md#term-scalar)-axis failures;
- accepted dynamic non-negative index, selected dynamic-axis removal, unaffected reference
  retention, and unresolved result layout;
- [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)-to-scalar and all six data types/eligibility values;
- contiguous, offset, strided, and zero-stride resolved input geometry; selected-stride removal,
  checked offset, new view classification, and exact result layout values;<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [stride glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- unresolved input and empty-result unresolved [layout](../../../../glossary.md#layout);
- exact Operation kind/attributes, normalized values, provenance input reference/order, absent
  label/storage, and fresh identity across repeated/nested calls;<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- no [eager](../../../../glossary.md#term-eager-evaluation) value/storage access and no [compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) imports.

Manual validation must inspect `javap -p -c -s`, reflection/source/imports, generated [Javadoc](../../../../glossary.md#term-javadoc),
the executable documentation example when changed, Markdown links/anchors/fences/whitespace,
exact ten-path [scope](../../../../glossary.md#term-scope), synchronized task/master/roadmap status, and absence of a task-0018C
specification.

## Dependencies

- Task 0002 provides [Shape](../../../../glossary.md#shape), [Dimension](../../../../glossary.md#dimension), static/dynamic, and [axis](../../../../glossary.md#term-axis)-[normalization](../../../../glossary.md#term-normalization) contracts.
- Task 0003 provides resolved logical [layout](../../../../glossary.md#layout) geometry and checked [descriptor](../../../../glossary.md#term-descriptor) construction.
- Task 0007 provides the resolved/unresolved TensorDescriptor boundary.
- Tasks 0011–0013 provide [Tensor](../../../../glossary.md#tensor), derived [identity](../../../../glossary.md#term-identity-meanings) allocation, and [immutable](../../../../glossary.md#term-immutable) [provenance](../../../../glossary.md#provenance).
- Task 0018A provides exact SELECT semantics and normalized attributes.

## Follow-up tasks

- 0018C: [axis](../../../../glossary.md#term-axis) [gather](../../../../glossary.md#gather) semantic [identities](../../../../glossary.md#term-identity-meanings) and normalized attributes.
- 0018D: public axis gather/take [Tensor](../../../../glossary.md#tensor) expressions.

Do not create either follow-up specification during this task.

## Architecture impact

Expected impact: None.

This task fills the existing [model](../../../../glossary.md#term-model)-owned public expression surface. If implementation requires a
new architecture rule, dynamic-[shape](../../../../glossary.md#shape) contract change, [runtime](../../../../glossary.md#term-runtime) service, or cross-[module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency),
stop and report the issue.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0003/0007/0011/0012/0013/0015E/0017G/0017H/0017K/
0017L/0018A/0018B, Tensor API, Compile API, Training API, glossary, current Shape/Dimension/
LayoutDescriptor/TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/SelectKind/
SelectAttrs contracts and focused expression tests, and Java 26 Gradle configuration.

Implement task 0018B exactly. Modify Tensor.java and add package-private final
TensorSelectExpressions.java. Update TensorTest only for the exact one-method API expansion and add
TensorSelectExpressionTest. Add exactly select(int axis, long index), delegating once to the
helper.

The field-free helper has exactly five specified methods. Null-check input, normalize the axis
once, normalize/check a static selected extent, accept a non-negative dynamic index with deferred
bounds, and reject a negative dynamic index with the exact message. Remove the selected axis while
preserving unaffected Dimension references. For resolved non-empty geometry, remove the selected
stride and checked-advance the offset in one new view; unresolved or empty results stay unresolved.
Preserve exact type/eligibility, create exact SELECT/SelectAttrs/[input], and call createDerived
once with no label/storage. Every request is fresh.

Do not modify semantic/foundational contracts, add overloads/types/helpers, inspect/copy values or
storage, attach physical aliases, define gradients, capture/canonicalize graphs, or add compiler/
planning/prepare/runtime/backend behavior, dependencies, build/architecture changes, or later
specs. Stop beyond ten paths or on architecture uncertainty.

Run all specified validation, then hand the actual diff/evidence to a separate clean-context docs
agent in the same change. It must inspect source/tests/generated Javadoc, finalize permitted
Javadocs/Tensor API/Compile API/glossary/planning, record Training API/capability/architecture and
related-contract no-change conclusions, and rerun validation.

Update task 0018B, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0018C Draft without a specification. Do not commit/push.
```

## Local decisions

- A negative public index is normalized only when the selected Dimension is static. A dynamic
  selected Dimension accepts every non-negative <span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension))</span>`long` unchanged and defers only its upper-bound
  check; it rejects a negative index rather than storing an unnormalized sentinel.
- Result Shape construction removes exactly one Dimension and retains every unaffected exact
  reference. Rank-one selection uses the canonical scalar Shape returned by<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [canonical glossary](../../../../glossary.md#term-canonical-form); [scalar glossary](../../../../glossary.md#term-scalar))</span>
  `Shape.ofDimensions()`.
- Resolved input layout produces resolved result geometry only when the result is non-empty. The
  selected stride is removed and the offset is checked-advanced; an unresolved input or empty
  result stays unresolved.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [stride glossary](../../../../glossary.md#term-stride))</span>
- Resolved layout is logical view metadata only. The expression remains storage-free and makes no
  physical-alias or zero-copy promise.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [metadata glossary](../../../../glossary.md#term-metadata); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Repeated, nested, and same-coordinate requests remain fresh explicit expressions. Compiler-owned
  capture and canonicalization are not performed by the model helper.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [model glossary](../../../../glossary.md#term-model))</span>

## Known limitations

- A negative index on a dynamic selected extent is intentionally unsupported because no numeric
  extent exists for local normalization. The upper bound of an accepted non-negative dynamic
  index remains deferred.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- The task defines no selected values, physical storage alias, gradient or backward rule, graph
  capture, canonicalization, materialization, backend lowering, ONNX mapping, or execution.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [gradient glossary](../../../../glossary.md#term-gradient); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [materialization glossary](../../../../glossary.md#materialization); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [ONNX glossary](../../../../glossary.md#term-onnx))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Clean implementation context `/root/implement_model_0018b` added the exact public method,
  field-free five-method helper, Tensor API inventory expansion, and focused ten-test suite before
  handing the actual shared-tree diff to independent documentation context<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
  `/root/implement_model_0018b/review_model_0018b_docs`.
- The documentation context applied General plus API/Javadoc style to production Javadocs, Tensor
  API, Compile API, and glossary; Planning style to this task, the model master plan, and roadmap;
  and Example format to the executable Tensor API example. It inspected the final source, tests,
  generated Javadoc, and actual diff rather than relying on the implementation summary.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Compile glossary](../../../../glossary.md#compile); [Planning glossary](../../../../glossary.md#term-planning))</span>
- Reviewed architecture and process material: `AGENTS.md`, `ARCHITECTURE.md`, current focused
  architecture navigation/boundaries, documentation rules and General/API-Javadoc/Planning/
  Example profiles, planning guide and roadmap, model capabilities/master plan, and tasks 0002,
  0003, 0007, 0011, 0012, 0013, 0015E, 0017G, 0017H, 0017K, 0017L, 0018A, and 0018B.
- Reviewed API and implementation material: Tensor, Compile, and Training API references; glossary;
  final [`Tensor`](../../../../glossary.md#tensor), `TensorSelectExpressions`, `TensorTest`, and `TensorSelectExpressionTest`;
  Shape/Dimension, LayoutDescriptor, TensorDescriptor, TensorFactory, TensorProvenance, Operation,
  SelectKind, and SelectAttrs contracts; related WHERE, SLICE, and UNSTACK contracts/tests; model
  generated Javadoc; and Java 26 root/model Gradle configuration.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [Operation glossary](../../../../glossary.md#operation); [SLICE glossary](../../../../glossary.md#slice); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- The Javadoc review corrected the distinction between resolved input geometry and a non-empty
  result and completed result-element-count overflow wording. The public method and helper type,
  private constructor, and all five methods document parameters, returns, failures, axis/index
  normalization, exact Shape references, scalar output, conditional logical view geometry,
  provenance, freshness, and cross-layer exclusions.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [overflow glossary](../../../../glossary.md#term-overflow); [parameters glossary](../../../../glossary.md#term-parameter); [axis glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization); [scalar glossary](../../../../glossary.md#term-scalar); [view glossary](../../../../glossary.md#term-view-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
- Tensor API now documents scalar SELECT versus WHERE/UNSTACK/SLICE/tensor-index gather; raw and
  normalized axes/indexes; static versus dynamic selected extents; exact unaffected Dimension
  references; rank-one scalar output; <span data-glossary-navigation="true"> ([scalar SELECT glossary](../../../../glossary.md#scalar-select); [gather glossary](../../../../glossary.md#gather); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`[2, 3, 4]` selection; `[12, 4, 1]` to `[12, 1]` strides and
  offset `8`; empty/unresolved layout; logical-view versus physical-alias status; exact result
  metadata/provenance/freshness; and value/gradient/compiler/backend/execution boundaries.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [alias glossary](../../../../glossary.md#aliasing-and-arena); [metadata glossary](../../../../glossary.md#term-metadata); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- Compile API now lists scalar select as current model expression construction while graph capture,
  canonicalization, dynamic upper-bound validation placement, materialization, gradients,
  lowering, and execution remain planned. Glossary wording now makes the same reusable semantic,
  expression, and ownership distinctions.<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorSelectExpressionTest` — `BUILD SUCCESSFUL`; the XML
  report contains 10 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; the XML report contains
  14 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 76 XML suites contain 648 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated `Tensor.html` contains the
  exact `select(int, long)` signature, positive/negative normalization example, dynamic selected-
  extent policy, Shape/view result contract, complete parameter/return/failure documentation,
  exact SELECT semantics, and physical-alias/compiler/backend/execution boundaries. The package-
  private helper Javadocs were reviewed in source because standard public Javadoc omits it.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [extent glossary](../../../../glossary.md#term-extent))</span>
- `./gradlew test` — `BUILD SUCCESSFUL`; all 36 actionable tasks completed without a failing task
  in the final repository lifecycle run.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [run glossary](../../../../glossary.md#term-run))</span>
- `javap -classpath modules/model/build/classes/java/main -p -c -s` for `Tensor` and
  `TensorSelectExpressions`, focused reflection tests, and source inspection confirm exactly 97
  declared public Tensor methods, the one exact public one-call delegation, a private zero-argument
  helper constructor, exactly five static helper methods and no fields/nested types, one Shape axis
  normalization, static/dynamic index branches, exact Shape/layout construction, checked offset,
  exact descriptor/SELECT/attrs/<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>`[input]` construction, and one final `createDerived` call.
- Production import inspection found only model-owned semantic/layout/shape/tensor contracts and
  JDK collections/null checking. Source, bytecode, and focused tests show no compiler, planning,
  prepare, runtime, backend, engine, trace, training, value-read, or storage-read behavior.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [bytecode glossary](../../../../glossary.md#term-bytecode); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace))</span>
- The documented Java 26 `ScalarSelectExpressionExample` compiled with
  `javac -cp modules/model/build/classes/java/main -d /tmp/synaptik-select-doc-example
  /tmp/ScalarSelectExpressionExample.java` and ran with the [model](../../../../glossary.md#term-model) classes. It printed exact [Shape](../../../../glossary.md#shape)
  `[2, 4]`, `SelectAttrs[axis=1, index=2]`, [strides](../../../../glossary.md#term-stride) `[12, 1]`, offset `8`, view/provenance/
  eligibility/storage facts exactly as documented.
- The targeted Markdown path-and-heading validator resolved all 388 local links, including 110
  heading anchors, across the six changed documentation/planning files. Code fences are balanced,
  trailing-whitespace scans found no matches, and all ten authorized paths have final newlines.
- Final changed-path inventory contains exactly the ten authorized paths: Tensor and its select
  helper, TensorTest and the focused select test, Tensor API, Compile API, glossary, this task,
  model master plan, and roadmap. Task/master-plan/roadmap status is synchronized as Complete;
  task 0018C remains Draft and no task-0018C specification exists. `git diff --check` passes.
- Training API remains accurate unchanged because the task adds no gradient object/rule, autograd,
  parameter, optimizer, session, publication, or training execution behavior. Capabilities remain
  accurate unchanged because they already list scalar-index select and distinguish model/public-
  API support from compiler, planning, backend-prepare, runtime, and end-to-end support.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication))</span>
- Shape/Dimension, LayoutDescriptor, TensorDescriptor, TensorFactory, TensorProvenance, Operation,
  SelectKind, SelectAttrs, and WHERE/SLICE/UNSTACK contracts remain accurate unchanged because
  this task composes their existing normalization, geometry, identity, semantic, and provenance
  rules without changing them. Completed task specifications remain historical evidence.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests, backend-
  conformance tests, integration tests, Gradle/dependencies, and other modules remain unchanged
  because the task stays inside existing model ownership and adds no dependency rule, backend
  behavior, executable end-to-end behavior, or build requirement.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Implementation notes

- Added exactly `Tensor.select(int axis, long index)` as one helper delegation.
- Added package-private final, field-free `TensorSelectExpressions` with the exact five methods for
  deterministic validation, normalization, Shape/layout derivation, and fresh storage-free SELECT
  provenance.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [normalization glossary](../../../../glossary.md#term-normalization); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>
- Expanded the exact Tensor reflection inventory and added ten focused tests covering API/helper
  shape, validation/messages, static/dynamic normalization, Shape references, all data types and
  valid eligibility choices, resolved/unresolved/empty layout, provenance/freshness, storage
  noninterference, and identity timing.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Finalized production/helper Javadocs, Tensor and Compile API references, glossary terminology,
  executable example, and synchronized planning status/evidence without architecture changes.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: Implemented and documented public scalar-index selection with static/dynamic
  normalization, exact axis removal, conditional checked logical-view geometry, and fresh exact
  SELECT provenance.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [normalization glossary](../../../../glossary.md#term-normalization); [axis glossary](../../../../glossary.md#term-axis); [view glossary](../../../../glossary.md#term-view-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
- Files changed or created: Exactly the ten authorized production, test, API, glossary, task,
  master-plan, and roadmap paths.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Tests and validation: Focused 10-test and 14-test suites, all 648 model tests across 76 suites,
  model Javadoc, root tests, javap/reflection/import/source/generated-page review, executable Java
  26 example, 388 link/110-anchor checks, fence/whitespace/newline checks, exact scope/status and
  no-0018C-spec checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0018b/review_model_0018b_docs` completed the required independent pass
  with General, API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API, Compile API, and glossary now describe current scalar-select
  expression construction while gradients, compiler behavior, physical materialization, backend
  behavior, and execution remain planned or separately owned.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [materialization glossary](../../../../glossary.md#materialization); [backend glossary](../../../../glossary.md#backend))</span>
- Javadoc review: Public Tensor method and helper type/constructor/five methods are final; related
  semantic, Shape/layout, descriptor/factory/provenance/operation, and adjacent expression
  contracts remain accurate unchanged for the recorded reasons.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation))</span>
- Glossary impact: Scalar-select terminology now distinguishes normalized semantic attributes,
  public input-aware expression construction, logical view metadata, and physical/executable
  behavior.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0018B. Task 0018C remains Draft without a detailed
  specification.

Status: Complete
