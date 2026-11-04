# Task 0017H: [Slice](../../../../glossary.md#slice) [Tensor](../../../../glossary.md#tensor) Expressions

**Editorial reading key (later context, not a status update).**

A slice selects coordinate ranges; its step advances logical coordinates, whereas a layout stride measures storage-element spacing. The application passes starts, exclusive ends, axes and steps. The helper normalizes the historical positive-step request, derives lengths and possible logical view geometry, and records one input. The linked current signed/length-based contract is later context, not a change to these archived bounds or restrictions. The application calls the public Tensor method; its model helper validates available descriptions and returns a fresh storage-free expression with exact input references. It neither copies input values nor calculates the result. Compiler consumes this provenance later; actual storage and execution are separate downstream concerns.

Focused current reading: [Current slicing reference](../../../../reference/tensor/slicing.md#slice-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Add public general and single-[axis](../../../../glossary.md#term-axis) positive-step [slice](../../../../glossary.md#slice) expression construction to [`Tensor`](../../../../glossary.md#tensor).

Both forms must normalize caller coordinates against locally known static [dimensions](../../../../glossary.md#dimension), preserve
[rank](../../../../glossary.md#term-rank) and unaffected Dimension references, derive resolved strided-[view](../../../../glossary.md#term-view-meanings) geometry when it is safe
and meaningful, record exact `SLICE` semantics with one-input [provenance](../../../../glossary.md#provenance), and return fresh
storage-free Tensor expressions. This task constructs [model](../../../../glossary.md#term-model) [metadata](../../../../glossary.md#term-metadata) only; it does not read values,
attach storage [aliases](../../../../glossary.md#aliasing-and-arena), execute slicing, define [gradients](../../../../glossary.md#term-gradient), or choose [materialization](../../../../glossary.md#materialization).

## [Scope](../../../../glossary.md#term-scope)

- Add exactly these public instance methods to [`Tensor`](../../../../glossary.md#tensor):
  - `slice(long[] starts, long[] ends, int[] axes, long[] steps)`
  - `sliceAxis(int axis, long fromInclusive, long toExclusive)`
- Add one field-free package-private final `TensorSliceExpressions` helper in `model.tensor`.
- Give the helper exactly one general entry, one single-axis entry, and six private methods defined
  by this task.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- Require non-null general-request arrays with equal lengths and defensively clone all four before
  inspecting their elements.<span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element))</span>
- Normalize positive or negative raw axes exactly once against input rank and reject duplicates
  after normalization.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Require strictly positive steps; do not support reverse slicing.
- Require every selected input dimension to be statically known. Preserve unselected static or
  dynamic Dimension references exactly.<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension))</span>
- Normalize negative starts/ends by adding selected dimension size once, then clamp every bound
  into inclusive range `[0, dimensionSize]`.
- Use [half-open](../../../../glossary.md#term-half-open-interval) `[start, end)` meaning and checked [overflow](../../../../glossary.md#term-overflow)-safe positive-step [extent](../../../../glossary.md#term-extent) calculation.
- Permit empty results when normalized start is at least normalized end or the selected/input
  shape already contains a zero extent.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- Construct normalized `SliceAttrs` in original request-entry order.
- Derive one new view-marked LayoutDescriptor for non-empty results when input layout is resolved;
  leave layout unresolved when input layout is unresolved or result element count is zero.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [layout glossary](../../../../glossary.md#layout))</span>
- Preserve input [data type](../../../../glossary.md#data-type--datatype) and [gradient](../../../../glossary.md#term-gradient) eligibility; create exact [`SLICE`](../../../../glossary.md#slice), `SliceAttrs`, and
  ordered provenance <span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>`[input]`; attach no label or storage.
- Keep every valid request explicit and fresh, including identity, empty-entry, unit-step, repeated,
  and nested slices.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Update only Tensor [API](../../../../glossary.md#term-application-programming-interface-api) inventory plus one focused expression test.
- Finalize Javadocs, Tensor/Compile API, glossary, task evidence, master plan, and roadmap through
  the mandatory independent documentation pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- another public overload, List-based API, nullable/default axes or steps, omitted bounds,
  ellipsis, open-ended slice object, scalar indexing, <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [axes glossary](../../../../glossary.md#term-axis); [slice glossary](../../../../glossary.md#slice); [scalar glossary](../../../../glossary.md#term-scalar))</span>`select`, or task-0018 indexing work
- `int[]` bound/step overloads retained solely for legacy [signature](../../../../glossary.md#term-signature) compatibility
- zero or negative steps, reverse slicing, axis repetition, axis sorting, slice-chain
  canonicalization, identity elimination, or bounds reordering<span data-glossary-navigation="true"> ([canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- slicing a selected dynamic dimension, symbolic bound expressions, constraint generation, graph-
  wide Shape inference, or runtime dimension binding<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension); [graph glossary](../../../../glossary.md#graph); [Shape glossary](../../../../glossary.md#shape); [inference glossary](../../../../glossary.md#term-inference); [runtime glossary](../../../../glossary.md#term-runtime); [binding glossary](../../../../glossary.md#term-binding))</span>
- rejecting empty slices merely because legacy rejected them; zero-extent Shape is a supported
  current model value<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [model glossary](../../../../glossary.md#term-model))</span>
- reading, copying, scattering, or eagerly materializing values; attaching, retaining, replacing,
  or observing host storage; asserting a physical alias or zero-copy execution<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- modifying SliceKind, SliceAttrs, Shape, Dimension, LayoutDescriptor, TensorDescriptor,
  TensorFactory, TensorProvenance, Operation, or any completed Java contract/test<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation))</span>
- slice backward, gradients, autograd, graph capture, compiler passes, planning requirements,
  prepare, runtime, backend lowering/execution, engine, trace, ONNX, or training behavior<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- another production helper/type, dependency, Gradle/build option, architecture change, another
  module, or task-0017I specification<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

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
- [Task 0017G](0017g-slice-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The selected baseline includes general strided [`slice`](../../../../glossary.md#slice) and one-[axis](../../../../glossary.md#term-axis) `sliceAxis`. The read-only
legacy branch exposes four parallel `int[]` [parameters](../../../../glossary.md#term-parameter), accepts negative axes/bounds, clamps bounds
to a concrete [dimension](../../../../glossary.md#dimension), supports positive steps, rejects duplicate axes and empty dimensions,
and derives offset/[stride](../../../../glossary.md#term-stride) [view](../../../../glossary.md#term-view-meanings) geometry. Its one-axis convenience delegates to one general entry
with step one.

The new [API](../../../../glossary.md#term-application-programming-interface-api) retains the capability but uses `long[]` bounds/steps to match current long [Shape](../../../../glossary.md#shape) and
[layout](../../../../glossary.md#layout) geometry. It requires all four arrays explicitly rather than preserving nullable legacy
defaults. It permits zero-[extent](../../../../glossary.md#term-extent) results because current Shape, LayoutDescriptor, TensorDescriptor,
storage allocation, and factory contracts deliberately support empty [tensors](../../../../glossary.md#tensor). It rejects selected
dynamic dimensions because local [normalization](../../../../glossary.md#term-normalization) and clamping cannot prove their result extent.

Legacy storage [aliases](../../../../glossary.md#aliasing-and-arena), [eager](../../../../glossary.md#term-eager-evaluation) value access, [graph](../../../../glossary.md#graph) builders, [gradient](../../../../glossary.md#term-gradient) callbacks, traits, [compiler](../../../../glossary.md#term-compiler),
[lowering](../../../../glossary.md#lowering), [kernels](../../../../glossary.md#kernel), and [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) behavior are not copied.

## Architecture constraints

- [`Tensor`](../../../../glossary.md#tensor) remains public mutable API state, not IR. The new methods build fresh expression
  metadata through the existing package-private derived factory seam.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- [Slice](../../../../glossary.md#slice) semantic [identity](../../../../glossary.md#term-identity-meanings)/attributes come only from completed task 0017G.
- The helper may inspect immutable descriptor, Shape, and optional layout metadata, but never
  values, host storage, runtime residency, device state, or backend capability.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [descriptor glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [host storage glossary](../../../../glossary.md#host-storage); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [device glossary](../../../../glossary.md#term-device); [backend capability glossary](../../../../glossary.md#backend-capability))</span>
- Raw caller arrays are request syntax only. Normalized/clamped values alone enter SliceAttrs.
- A selected dynamic Dimension is rejected locally instead of becoming a hidden symbolic
  constraint. Unselected dynamic Dimensions retain exact references.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension))</span>
- Result rank equals input rank. Selected dimensions become new StaticDimension extents;
  unselected dimensions retain exact references.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>
- Resolved layout is logical view metadata only. It does not attach storage or promise physical
  aliasing or zero-copy execution.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- Empty results use unresolved layout because no storage element is referenced and one-past-end
  offsets/strides would be arbitrary or could overflow despite having no observable geometry.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [strides glossary](../../../../glossary.md#term-stride); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- Compiler owns canonicalization and capture; planning/backend prepare own materialization and
  lowering; compiler-generated/training work owns slice backward.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [capture glossary](../../../../glossary.md#term-capture-meanings); [planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare); [materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering))</span>
- No [dependency](../../../../glossary.md#term-dependency) or [module](../../../../glossary.md#term-module) boundary changes are authorized.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.tensor.Tensor` — receives the two public convenience surfaces.
- `io.github.pho001.synaptik.model.tensor.TensorSliceExpressions` — owns local validation,
  normalization, Shape/layout derivation, and semantic construction.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout))</span>
- `TensorSliceExpressionTest` — mirrors the package for focused helper/[API](../../../../glossary.md#term-application-programming-interface-api) [validation](../../../../glossary.md#term-validation).
- `TensorTest` — changes only its exact public API inventory and reflection assertions.

SliceKind and SliceAttrs remain in `model.operation.layout`; no semantic contract moves into the
[Tensor](../../../../glossary.md#tensor) package.

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) methods

Add exactly:

```java
public Tensor slice(long[] starts, long[] ends, int[] axes, long[] steps) {
    return TensorSliceExpressions.apply(this, starts, ends, axes, steps);
}

public Tensor sliceAxis(int axis, long fromInclusive, long toExclusive) {
    return TensorSliceExpressions.applyAxis(this, axis, fromInclusive, toExclusive);
}
```

Each public method contains one return statement and exactly one matching helper call. Neither
method performs [validation](../../../../glossary.md#term-validation), allocates another request representation, reads fields directly, or
calls the other public method. Both are non-static and non-synchronized.

[`slice`](../../../../glossary.md#slice) accepts arrays of equal length in caller entry order. Empty arrays are valid and produce a
fresh explicit [identity](../../../../glossary.md#term-identity-meanings) slice. `sliceAxis` is equivalent to one entry with the supplied [axis](../../../../glossary.md#term-axis) and
bounds plus step `1L`; it has no distinct [operation](../../../../glossary.md#operation) kind.

### Helper [shape](../../../../glossary.md#shape)

Create one package-private final, field-free class with one private zero-[argument](../../../../glossary.md#term-argument) constructor and
exactly these eight static methods:

```java
static Tensor apply(Tensor input, long[] starts, long[] ends, int[] axes, long[] steps)
static Tensor applyAxis(Tensor input, int axis, long fromInclusive, long toExclusive)
private static SliceAttrs normalize(
        Shape inputShape, long[] starts, long[] ends, int[] axes, long[] steps)
private static long normalizeBound(long rawBound, long dimensionSize)
private static long sliceExtent(long start, long end, long step)
private static Shape deriveShape(Shape inputShape, SliceAttrs attrs)
private static Optional<LayoutDescriptor> resolveViewLayout(
        TensorDescriptor inputDescriptor, Shape resultShape, SliceAttrs attrs)
private static Tensor create(
        Tensor input,
        TensorDescriptor inputDescriptor,
        Shape resultShape,
        Optional<LayoutDescriptor> resultLayout,
        SliceAttrs attrs)
```

Add no field, nested type, alternate constructor, overload, [cache](../../../../glossary.md#term-cache), [mutable](../../../../glossary.md#term-mutable) state, or extra method.

### General [validation](../../../../glossary.md#term-validation) and [normalization](../../../../glossary.md#term-normalization)

`apply` performs this exact order:

1. null-check `input`, `starts`, `ends`, [`axes`](../../../../glossary.md#term-axis), and `steps`, in order, using those exact messages;
2. require equal array lengths, otherwise throw `IllegalArgumentException` with exact message
   `starts, ends, axes, and steps must have matching lengths`;
3. clone starts, ends, axes, and steps once each, in that order;
4. read the exact input [descriptor](../../../../glossary.md#term-descriptor) and [Shape](../../../../glossary.md#shape) once;
5. call `normalize` once with only the private copies;
6. call `deriveShape` once;
7. call `resolveViewLayout` once;
8. call `create` once and return its result.

Null/length failures occur before cloning, descriptor inspection, allocation of semantic lists, or
[Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings) consumption. Caller arrays are never retained or mutated, and concurrent caller
mutation after the clones cannot change the expression.

`normalize` processes entries in ascending index order. It creates one [rank](../../../../glossary.md#term-rank)-sized `boolean[]` for
normalized-axis duplicate detection and four [mutable](../../../../glossary.md#term-mutable) constructor-local lists. At each index:

1. normalize raw `int axis` once using `long`; a negative value adds rank once;
2. reject an out-of-range result with exact message
   `slice axis <raw> at index <index> is outside rank <rank>`;
3. reject the first repeated [normalized axis](../../../../glossary.md#normalized-axis) with exact message
   `slice contains duplicate normalized axis <axis> at index <index>`;
4. reject a non-positive step with exact message `steps[<index>] must be positive: <step>`;
5. require `inputShape.dimensions().get(axis)` to be StaticDimension, otherwise throw
   `IllegalArgumentException` with exact message
   `slice axis <axis> at index <index> must have a statically known dimension`;
6. normalize and clamp the copied start/end against that [dimension](../../../../glossary.md#dimension);
7. append normalized start, end, axis, and unchanged positive step to the four lists.

After all entries pass, construct exactly one `SliceAttrs`, which performs the final [immutable](../../../../glossary.md#term-immutable)
snapshots. Do not sort entries or normalize their order.

`normalizeBound` adds `dimensionSize` once when the raw bound is negative, using long arithmetic,
then clamps below zero to zero and above the dimension to the dimension. Every raw long is valid.

`sliceExtent` returns zero when start is at least end. Otherwise it calculates exactly:

```java
1L + (end - 1L - start) / step
```

This form is safe for normalized `0 <= start < end <= dimensionSize <= Long.MAX_VALUE` and avoids
[overflow](../../../../glossary.md#term-overflow) from `end - start + step - 1`.

### Single-[axis](../../../../glossary.md#term-axis) delegation

`applyAxis` creates exactly four private one-[element](../../../../glossary.md#term-element) arrays and calls `apply` once:

```java
return apply(
        input,
        new long[] {fromInclusive},
        new long[] {toExclusive},
        new int[] {axis},
        new long[] {1L});
```

It performs no independent [normalization](../../../../glossary.md#term-normalization) or semantic construction. Therefore both public forms
share identical errors, clamping, geometry, [provenance](../../../../glossary.md#provenance), and ID side effects.

### Result [Shape](../../../../glossary.md#shape)

`deriveShape` copies exact input [Dimension](../../../../glossary.md#dimension) references into a [rank](../../../../glossary.md#term-rank)-sized array. For each SliceAttrs
entry in order, replace the selected [axis](../../../../glossary.md#term-axis) with one new `StaticDimension(sliceExtent(...))`. Return
`Shape.ofDimensions` once. [Scalar](../../../../glossary.md#term-scalar) plus empty entries remains the [canonical](../../../../glossary.md#term-canonical-form) scalar Shape.

Unselected dynamic, static, and zero dimensions preserve exact references. Selected dimensions
are always static because normalize rejected dynamic selection. Result rank never changes.

Examples for input Shape `[3, 6]`:

| Request | Normalized attrs | Result Shape |
|---|---|---|
| starts `[0,1]`, ends `[3,6]`, axes `[0,1]`, steps `[1,2]` | unchanged | `[3,3]` |
| starts `[-2]`, ends `[-1]`, axes `[-1]`, steps `[1]` | starts `[4]`, ends `[5]`, axes `[1]` | `[3,1]` |
| starts `[5]`, ends `[2]`, axes `[1]`, steps `[1]` | unchanged bounds | `[3,0]` |
| all arrays empty | empty attrs | `[3,6]` with exact Dimension references |

### Result [layout](../../../../glossary.md#layout)

`resolveViewLayout` reads the input layout Optional once.

- If input layout is unresolved, return `Optional.empty()`.
- If `resultShape.knownElementCount()` is zero, return `Optional.empty()` even when input layout is
  resolved.
- Otherwise copy input [strides](../../../../glossary.md#term-stride) once and start from exact input [storage offset](../../../../glossary.md#term-storage-offset).
- For each attrs entry in order, update offset with checked
  `offset + start * originalInputStride(axis)` and replace that [axis](../../../../glossary.md#term-axis) stride with checked
  `originalInputStride(axis) * step`.
- Use the original input stride for both calculations so request-entry processing cannot compound
  a stride accidentally.
- Create exactly one `LayoutDescriptor.of(resultShape, resultStrides, resultOffset, true)` and
  return it in Optional.

Every resolved input layout kind is accepted, including dense, offset, strided, and broadcast.
LayoutDescriptor reclassifies result kind/span. Arithmetic [overflow](../../../../glossary.md#term-overflow) propagates as
`ArithmeticException`. [View](../../../../glossary.md#term-view-meanings) [metadata](../../../../glossary.md#term-metadata) does not attach storage or guarantee executable [aliasing](../../../../glossary.md#term-aliasing).

### Result construction

`create` constructs exactly:

```java
TensorDescriptor descriptor = new TensorDescriptor(
        inputDescriptor.dataType(),
        resultShape,
        resultLayout,
        inputDescriptor.requiresGrad());
Operation operation = new Operation(SliceKind.SLICE, attrs);
TensorProvenance provenance = new TensorProvenance(operation, List.of(input));
return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
```

Every success consumes one fresh [Tensor](../../../../glossary.md#tensor) ID, retains exact input [data type](../../../../glossary.md#data-type--datatype) and eligibility, and has
absent label/storage. Failures before `createDerived` consume no ID. Factory exhaustion remains
the final failure. No storage, previous [provenance](../../../../glossary.md#provenance), or input label is inspected or retained except
the exact input reference in new provenance.

## Affected files

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorSliceExpressions.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSliceExpressionTest.java`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

At most the ten paths above may change. Production is limited to [Tensor](../../../../glossary.md#tensor) plus one package-private
helper; tests to Tensor inventory plus one focused suite; documentation to public [API](../../../../glossary.md#term-application-programming-interface-api)/status and
[planning](../../../../glossary.md#term-planning) evidence.

If another helper/type/test, overload, semantic/foundational edit, [dependency](../../../../glossary.md#term-dependency), build/architecture
change, or eleventh path is needed, stop and report. Do not create task 0017I.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Fully document both public methods and helper type/constructor/all eight methods.
- Explain four-array parallel ownership, half-open bounds, positive steps, negative axis/bound
  normalization, clamping, entry order, duplicate rules, and static selected dimensions.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [half-open glossary](../../../../glossary.md#term-half-open-interval); [normalization glossary](../../../../glossary.md#term-normalization); [dimensions glossary](../../../../glossary.md#dimension))</span>
- Explain empty arrays, empty results, zero extents, and why empty [layout](../../../../glossary.md#layout) remains unresolved.
- Document Shape Dimension-reference retention and selected static extent calculation with
  concrete numeric examples.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>
- Document resolved stride/offset calculations, checked arithmetic, every accepted input layout
  kind, view flag, unresolved cases, and lack of physical alias guarantee.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Document result type, Shape, layout, eligibility, label, storage, Operation, SliceAttrs,
  provenance, freshness, failure order, and identifier side effects.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Explain single-[axis](../../../../glossary.md#term-axis) step-one delegation and why no distinct [operation](../../../../glossary.md#operation) kind exists.
- Independently review SliceKind/SliceAttrs, Shape/Dimension, LayoutDescriptor, TensorDescriptor,
  TensorFactory, TensorProvenance, and adjacent view-expression Javadocs; record why unchanged
  contracts remain accurate or stop on discrepancy.

## Acceptance criteria

- [Tensor](../../../../glossary.md#tensor) exposes exactly the two specified methods; declared public method count rises 85 to 87.
- Each public method delegates once; helper is final/package-private/field-free with exact private
  constructor and eight methods.
- Exact null/length/order messages, defensive clones, axis/step/static-dimension validation,
  negative normalization, clamping, duplicate handling, and extent arithmetic match this task.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>
- General empty arrays and [scalar](../../../../glossary.md#term-scalar) [identity](../../../../glossary.md#term-identity-meanings) are valid and fresh.
- Empty slices produce zero-[extent](../../../../glossary.md#term-extent) [Shape](../../../../glossary.md#shape) and unresolved [layout](../../../../glossary.md#layout) rather than legacy rejection.
- Shape preserves exact unaffected [Dimension](../../../../glossary.md#dimension) references and [rank](../../../../glossary.md#term-rank).
- Resolved non-empty dense/offset/strided/broadcast layouts derive exact checked offset/strides,
  view flag, kind, and span; unresolved/empty results stay unresolved.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- All six data types and valid eligibility states retain exact [metadata](../../../../glossary.md#term-metadata).
- Result records exact [`SLICE`](../../../../glossary.md#slice), normalized SliceAttrs, and `[input]`, with absent label/storage and
  fresh identity.
- Single-[axis](../../../../glossary.md#term-axis) form uses exact one-entry step-one general path.
- Early [validation](../../../../glossary.md#term-validation)/layout failures consume no ID; final exhaustion behavior is preserved.
- No values/storage/[gradient](../../../../glossary.md#term-gradient)/[compiler](../../../../glossary.md#term-compiler)/[planning](../../../../glossary.md#term-planning)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) behavior or architecture change.
- Independent documentation review and all status/evidence synchronization complete before
  marking Complete.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorSliceExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact public/helper [shape](../../../../glossary.md#shape); all data types/eligibility; null/length validation
precedence; caller-array [ownership](../../../../glossary.md#term-ownership-meanings); positive/negative/extreme [axes](../../../../glossary.md#term-axis) and bounds; clamping; positive
steps; normalized duplicate axes; selected dynamic rejection; unselected dynamic retention;
[scalar](../../../../glossary.md#term-scalar)/empty/zero/static/mixed-dynamic Shapes; exact [Dimension](../../../../glossary.md#dimension) [identity](../../../../glossary.md#term-identity-meanings); [extent](../../../../glossary.md#term-extent) examples and
[overflow](../../../../glossary.md#term-overflow)-safe large values; dense/offset/strided/broadcast/unresolved/empty [layout](../../../../glossary.md#layout) outcomes;
checked offset/[stride](../../../../glossary.md#term-stride) overflow; exact [operation](../../../../glossary.md#operation)/attrs/[provenance](../../../../glossary.md#provenance); absent label/storage; dead-storage
non-interference; freshness; nested/identity calls; ID side effects; and exhaustion.

Inspect `javap -p -c -s`, reflection, imports, and source. Confirm two one-call public methods,
exact eight-method helper, four defensive clones, one [normalization](../../../../glossary.md#term-normalization) pass, one Shape derivation,
one layout path, one common construction, and no forbidden behavior. Validate generated [Javadoc](../../../../glossary.md#term-javadoc),
executable examples, [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api)/glossary, links/anchors/fences/whitespace, exact ten paths,
synchronized statuses, and no task-0017I specification.

## Dependencies

- 0002 supplies [immutable](../../../../glossary.md#term-immutable) static/dynamic [Dimensions](../../../../glossary.md#dimension), [Shape](../../../../glossary.md#shape), [scalar](../../../../glossary.md#term-scalar)/zero forms, and [axis](../../../../glossary.md#term-axis) vocabulary.
- 0003 supplies resolved [strides](../../../../glossary.md#term-stride), offsets, [view](../../../../glossary.md#term-view-meanings) [metadata](../../../../glossary.md#term-metadata), classification, and spans.
- 0007 supplies resolved-or-unresolved TensorDescriptor.
- 0011–0013 supply [Tensor](../../../../glossary.md#tensor), centralized derived [identity](../../../../glossary.md#term-identity-meanings), and immutable [provenance](../../../../glossary.md#provenance).
- 0017G supplies [SLICE](../../../../glossary.md#slice) and immutable normalized SliceAttrs.
- Completed neighboring view-expression tasks provide patterns but no extra production [dependency](../../../../glossary.md#term-dependency).

## Follow-up tasks

- 0017I remains Draft for pad/tile semantic [identities](../../../../glossary.md#term-identity-meanings) and [immutable](../../../../glossary.md#term-immutable) [parameters](../../../../glossary.md#term-parameter).
- [Compiler](../../../../glossary.md#term-compiler) later owns [slice](../../../../glossary.md#slice) [capture](../../../../glossary.md#term-capture-meanings) and [identity](../../../../glossary.md#term-identity-meanings)/nested-slice [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form).
- Compiler-generated/training tasks later own slice-backward scatter semantics.
- [Planning](../../../../glossary.md#term-planning)/[backend](../../../../glossary.md#backend) [prepare](../../../../glossary.md#prepare) later own [materialization](../../../../glossary.md#materialization) and concrete [view](../../../../glossary.md#term-view-meanings)/copy [lowering](../../../../glossary.md#lowering).
- [ONNX](../../../../glossary.md#term-onnx) extension later owns Slice mapping without entering the [model](../../../../glossary.md#term-model) or [runtime](../../../../glossary.md#term-runtime) [hot path](../../../../glossary.md#term-hot-path).

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. [Model](../../../../glossary.md#term-model) already owns [Tensor](../../../../glossary.md#tensor) expression construction, [Shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout) [metadata](../../../../glossary.md#term-metadata),
[operation](../../../../glossary.md#operation) semantics, [descriptors](../../../../glossary.md#term-descriptor), and [provenance](../../../../glossary.md#provenance). [Lifecycle](../../../../glossary.md#lifecycle) [ownership](../../../../glossary.md#term-ownership-meanings) remains unchanged.

Stop if implementation needs symbolic [binding](../../../../glossary.md#term-binding), physical storage [aliasing](../../../../glossary.md#term-aliasing), [gradients](../../../../glossary.md#term-gradient),
[compiler](../../../../glossary.md#term-compiler)/[planning](../../../../glossary.md#term-planning)/[backend](../../../../glossary.md#backend) behavior, another [dependency](../../../../glossary.md#term-dependency), or architecture changes.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0003/0007/0011/0012/0013/0017G/0017H, Tensor API,
Compile API, Training API, glossary, current Dimension/Shape/LayoutDescriptor/LayoutKind/
TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/SliceKind/SliceAttrs contracts and
focused view-expression tests, and Java 26 Gradle configuration.

Implement task 0017H exactly. Modify Tensor.java and add package-private final
TensorSliceExpressions.java. Update TensorTest only for exact two-method API expansion and add
TensorSliceExpressionTest. Add exactly slice(long[],long[],int[],long[]) and
sliceAxis(int,long,long).

The field-free helper has exactly eight specified methods. General slicing null-checks and clones
all four equal-length arrays, normalizes negative axes/bounds once, clamps bounds, rejects duplicate
axes/non-positive steps/selected dynamic dimensions, permits empty results, and constructs
normalized SliceAttrs. Preserve rank and unaffected Dimension references. For non-empty resolved
input geometry, derive checked start-adjusted offset and step-multiplied strides in a new view;
unresolved or empty results stay unresolved. Preserve type/eligibility, create exact SLICE/attrs/
[input], and call createDerived once with no label/storage. Single-axis delegates through one
step-one general entry. Every request is fresh.

Do not modify semantic/foundational contracts, add overloads/types/helpers, support reverse steps
or selected dynamic axes, inspect/copy values/storage, attach physical aliases, define gradients,
capture/canonicalize graphs, or add compiler/planning/prepare/runtime/backend/ONNX behavior,
dependencies, build/architecture changes, or later specs. Stop beyond ten paths or on architecture
uncertainty.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/Compile API/glossary/planning, record no-change conclusions, and rerun validation.

Update 0017H, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0017I Draft without a specification. Do not commit/push.
```

## Local decisions

- Public bounds/steps use `long[]` to match current [Shape](../../../../glossary.md#shape)/LayoutDescriptor geometry; [axes](../../../../glossary.md#term-axis) use
  `int[]` and retain familiar axis addressing.
- All four arrays are explicit and non-null. Legacy nullable/default arrays are not retained
  because the full capability and one-axis convenience do not require ambiguous defaults.
- Selected dynamic [dimensions](../../../../glossary.md#dimension) are rejected; local code does not invent symbolic constraints.
- Raw negative bounds add dimension size once and clamp, preserving useful legacy behavior.
- Empty slices are accepted because zero extents are first-class in the current model; the legacy
  rejection is not copied.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Empty results keep [layout](../../../../glossary.md#layout) unresolved to avoid arbitrary/no-[element](../../../../glossary.md#term-element) offset and [stride](../../../../glossary.md#term-stride) facts.
- Resolved non-empty results are [view](../../../../glossary.md#term-view-meanings) [metadata](../../../../glossary.md#term-metadata) only, with no attached storage or execution promise.
- [Identity](../../../../glossary.md#term-identity-meanings), repeated, and nested requests remain explicit; [compiler](../../../../glossary.md#term-compiler) owns [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form).

## Known limitations

- Reverse/negative-step and selected-dynamic slicing are unsupported.
- No open-ended bounds/default arrays, physical storage alias, eager value access, gradients,
  compiler capture/canonicalization, materialization, lowering, execution, or ONNX mapping exists.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [eager glossary](../../../../glossary.md#term-eager-evaluation); [gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering); [ONNX glossary](../../../../glossary.md#term-onnx))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Architecture, focused boundary documentation, planning/documentation rules, roadmap, model
  capabilities/master plan, completed semantic task 0017G, neighboring view-expression tasks,
  current Shape/Dimension/LayoutDescriptor/Tensor contracts/tests, Tensor/Compile/Training APIs,
  glossary, Java 26 Gradle configuration, and read-only legacy slice implementation/tests were
  reviewed before specifying behavior.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Dimension glossary](../../../../glossary.md#dimension); [Compile glossary](../../../../glossary.md#compile); [slice glossary](../../../../glossary.md#slice))</span>
- Legacy evidence confirms general parallel-parameter slicing, negative normalization/clamping,
  positive steps, duplicate-axis rejection, strided offset views, and one-axis step-one delegation.
  Coupled legacy storage, graph, gradient, compiler, kernel, and runtime design is excluded.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [normalization glossary](../../../../glossary.md#term-normalization); [graph glossary](../../../../glossary.md#graph); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [kernel glossary](../../../../glossary.md#kernel); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- `git diff --check` passed with no whitespace errors.
- Changed-path inventory contains exactly three planning paths: this task, the model master plan,
  and roadmap. No Java, API, glossary, Gradle, AGENTS, ARCHITECTURE, focused architecture, completed
  task, or other-module file changed during planning.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module))</span>
- Markdown structure check passed: 23 level-two sections, 14 balanced code-fence markers, no
  trailing whitespace, and final newline present.
- Every local Markdown link in all three changed files resolves.
- Roadmap contains all 74 ordered unique task rows; task 0017H is linked and Ready at row 61, task
  0017I remains Draft at row 62, and no task-0017I specification exists.
- Task, master plan, and roadmap consistently identify 0017H as the next [Ready](../../../../glossary.md#term-ready) frontier.
- Package/scope review confirms no package addition: two public methods and the package-private
  helper remain in <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`model.tensor`, the focused test mirrors that package, and implementation is
  bounded to the ten authorized paths without a dependency or architecture change.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- No [Gradle](../../../../glossary.md#term-gradle) test was [run](../../../../glossary.md#term-run) because this [planning](../../../../glossary.md#term-planning)-only change modifies no production or test code.
- Clean implementation context `/root/implement_model_0017h` added the two public delegating
  methods, exact field-free eight-method helper, Tensor API inventory updates, and focused
  expression test before handing the actual shared-tree diff to the independent documentation
  context.
- Clean documentation context
  `/root/implement_model_0017h/review_model_0017h_docs` applied the General, API/Javadoc,
  Planning, and Example profiles. It read the architecture and focused boundary documents,
  documentation/planning rules, model capabilities/master plan, prerequisite tasks, Tensor/
  Compile/Training APIs, glossary, final production/tests, generated Javadoc, Java 26 Gradle
  configuration, and actual diff rather than relying on the implementation handoff.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- The documentation pass corrected the public Javadoc distinction between resolved input geometry
  and a non-empty result, completed element-count/layout arithmetic failure wording, and confirmed
  the helper type, constructor, and all eight methods document parameters, results, ownership,
  normalization, Shape/layout derivation, provenance, freshness, and cross-layer exclusions.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [layout glossary](../../../../glossary.md#layout); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Tensor API now documents four-array ownership/order, half-open positive-step normalization,
  static selected dimensions, empty requests/results, exact unaffected Dimension retention,
  checked offset/stride derivation for every resolved input layout kind, logical view status,
  single-axis delegation, result metadata, failure/ID behavior, and current limitations. Compile
  API now lists slice expressions as current model inputs while capture, canonicalization,
  gradient scatter, materialization, backend/ONNX lowering, and execution remain planned. The
  glossary aligns the reusable slice and provenance terminology with those boundaries.<span data-glossary-navigation="true"> ([half-open glossary](../../../../glossary.md#term-half-open-interval); [metadata glossary](../../../../glossary.md#term-metadata); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [materialization glossary](../../../../glossary.md#materialization); [backend glossary](../../../../glossary.md#backend); [ONNX glossary](../../../../glossary.md#term-onnx); [lowering glossary](../../../../glossary.md#lowering))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorSliceExpressionTest --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; XML reports contain 11
  slice-expression tests and 14 Tensor tests, with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 68 XML suites contain 554 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated `Tensor.html` contains both
  public methods, non-empty-result/unresolved-layout rules, complete parameter/return/failure
  documentation, exact SLICE semantics, and physical-alias/compiler/backend/execution boundaries.
  Package-private helper Javadocs were reviewed in source because standard public Javadoc omits
  the helper.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- `./gradlew test` — `BUILD SUCCESSFUL`; all 36 actionable tasks completed without a failing task
  in the final repository lifecycle run.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `javap -classpath modules/model/build/classes/java/main -p -c -s` for [`Tensor`](../../../../glossary.md#tensor) and
  `TensorSliceExpressions`, focused reflection tests, import scans, and source inspection confirmed
  exactly 87 declared public Tensor methods; the two exact one-call slice methods; one private
  zero-argument helper constructor; exactly eight static helper methods and no fields; four clones
  in order; one normalization, Shape, layout, and creation path; checked arithmetic; exact
  descriptor/SLICE/attrs/<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>`[input]` construction; and no forbidden layer import or behavior.
- The documented Java 26 `SliceExpressionExample` compiled and ran against model classes. It
  printed Shape `[3, 3]`, normalized attributes, [strides](../../../../glossary.md#term-stride) `[6, 2]`, offset `1`, `STRIDED`, span
  `18`, [view](../../../../glossary.md#term-view-meanings)/[provenance](../../../../glossary.md#provenance)/storage facts, and normalized one-[axis](../../../../glossary.md#term-axis) [Shape](../../../../glossary.md#shape) `[3, 1]` exactly as shown.
- A targeted Markdown path-and-heading validator resolved all 332 local links and anchors across
  the six changed documentation/planning files. Fence counts are balanced, trailing-whitespace
  checks found no matches, and every changed file has a final newline.
- Final scope review contains exactly the ten authorized paths: Tensor and its slice helper, the
  two Tensor tests, Tensor API, Compile API, glossary, this task, model master plan, and roadmap.
  Task/master-plan/roadmap status is synchronized as `Complete`; task 0017I and later rows remain
  Draft, and no task-0017I specification exists. `git diff --check` passes.
- `SliceKind`/`SliceAttrs`, Dimension/Shape, LayoutKind/LayoutDescriptor, TensorDescriptor,
  TensorFactory, TensorProvenance, Operation, and adjacent view-expression contracts remain
  accurate unchanged: this task composes their normalized semantics, immutable geometry,
  eligibility, central identity, and provenance contracts without changing them. Completed task
  specifications likewise remain historical contracts and required no edits.<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [immutable glossary](../../../../glossary.md#term-immutable); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Training API remains unchanged because no gradient object/rule, autograd, parameter, optimizer,
  session, or training execution behavior was added. Capabilities remain unchanged because they
  already select general/single-axis slicing and logical sliced views without serving as task-
  completion evidence.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- `ARCHITECTURE.md`, focused architecture/ADRs/tests, backend-conformance and integration tests,
  Gradle/dependencies, and other modules remain unchanged because this task stays within existing
  model ownership and adds no dependency rule, compiler/planning/prepare/runtime/backend behavior,
  executable alias, end-to-end behavior, or build requirement.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [prepare glossary](../../../../glossary.md#prepare))</span>

## Implementation notes

- Added exactly `Tensor.slice(long[], long[], int[], long[])` and
  `Tensor.sliceAxis(int, long, long)` as one-call delegations.
- Added the package-private field-free `TensorSliceExpressions` helper with the exact eight methods
  for deterministic request ownership, normalization/clamping, same-rank Shape derivation,
  conditional checked view geometry, and fresh storage-free SLICE provenance.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [normalization glossary](../../../../glossary.md#term-normalization); [rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape); [view glossary](../../../../glossary.md#term-view-meanings); [SLICE glossary](../../../../glossary.md#slice); [provenance glossary](../../../../glossary.md#provenance))</span>
- Expanded Tensor reflection inventory and added the focused 11-test slice suite covering all
  required API, validation, Shape, layout, metadata, freshness, storage-noninterference, and ID
  cases.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [validation glossary](../../../../glossary.md#term-validation); [layout glossary](../../../../glossary.md#layout); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Finalized public/helper Javadocs, Tensor and Compile API references, glossary terminology, and
  planning status/evidence without changing architecture or another contract.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: Implemented and documented general and single-axis positive-step slice Tensor
  expression construction with defensive request ownership, selected-static normalization,
  zero-extent support, conditional checked view geometry, and exact fresh provenance.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [slice glossary](../../../../glossary.md#slice); [Tensor glossary](../../../../glossary.md#tensor); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [normalization glossary](../../../../glossary.md#term-normalization); [extent glossary](../../../../glossary.md#term-extent); [view glossary](../../../../glossary.md#term-view-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
- Files changed or created: Exactly the ten authorized production, test, API, glossary, task,
  master-plan, and roadmap paths.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Tests and validation: Focused 25 tests, all 554 model tests, model Javadoc, root tests, compiled
  Java 26 example, bytecode/reflection/import/source checks, generated-Javadoc review, 332 local
  link/anchor checks, fence/whitespace/final-newline checks, exact scope/status review, no-0017I-
  spec check, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0017h/review_model_0017h_docs` completed the independent pass using
  General, API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Slice expression construction is current in Tensor API, Compile API, and
  glossary; compiler/training/planning/prepare/runtime/backend/ONNX execution boundaries remain
  explicitly planned or separately owned.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- Javadoc review: Both public methods and helper type/constructor/eight methods are final; related
  semantic, Shape/layout, descriptor/factory/provenance/operation, and adjacent view contracts
  remain accurate unchanged for the reasons recorded above.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation))</span>
- Glossary impact: Slice and provenance entries now distinguish semantic attributes, public
  expression construction, logical view metadata, and physical/executable behavior.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- Architecture impact: None; no architecture contract, ADR, architecture test, dependency, or
  module boundary changed.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [architecture test glossary](../../../../glossary.md#term-architecture-test); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0017H. Task 0017I remains Draft without a detailed
  specification.

Status: Complete
