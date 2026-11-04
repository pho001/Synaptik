# Task 0017D1: Expand [Tensor](../../../../glossary.md#tensor) Expressions

**Editorial reading key (later context, not a status update).**

The application supplies a target Shape, and expansion compares source and target axes from the right. It may repeat source singletons but may not shrink an ordinary non-singleton. A zero stride describes repeated use of one source position; deriving it does not attach shared memory or duplicate bytes. The result is a storage-free expression with exact input provenance. This historical task rejects unresolved unequal pairs; the linked current reference describes later retained binding obligations, not an original capability of this checkpoint.

Focused current reading: [Current expansion and retained obligations](../../../../reference/tensor/reshape-expand.md#expand-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Add public, storage-free [Tensor](../../../../glossary.md#tensor) expressions for the completed `EXPAND` semantic meaning.

Expansion right-aligns the input [Shape](../../../../glossary.md#shape) with an exact [target](../../../../glossary.md#term-target) Shape. Existing input [dimensions](../../../../glossary.md#dimension) must
equal their aligned target dimensions or be statically known singletons. Additional leading target
[axes](../../../../glossary.md#term-axis) are permitted because missing input axes behave as logical singletons. When input [layout](../../../../glossary.md#layout) and
target geometry are resolved locally, construction derives zero-[stride](../../../../glossary.md#term-stride) [alias](../../../../glossary.md#aliasing-and-arena)-[view](../../../../glossary.md#term-view-meanings) geometry.

This task creates [immutable](../../../../glossary.md#term-immutable) [model](../../../../glossary.md#term-model) [metadata](../../../../glossary.md#term-metadata) only. It does not repeat values, attach storage, define
[gradients](../../../../glossary.md#term-gradient), [capture](../../../../glossary.md#term-capture-meanings) a [graph](../../../../glossary.md#graph), choose [materialization](../../../../glossary.md#materialization), lower the [operation](../../../../glossary.md#operation), or execute it.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly `Tensor.expand(long...)` and `Tensor.expand(Shape)`.
- Make each public method delegate exactly once to its matching helper overload.
- Add one package-private final stateless `TensorExpandExpressions` helper in `model.tensor`.
- Give the helper exactly the six methods specified under Required contract and one private
  zero-argument constructor.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>
- Normalize raw [dimensions](../../../../glossary.md#dimension) through `Shape.of` without retaining or mutating the caller array.
- Treat empty raw dimensions as the [canonical](../../../../glossary.md#term-canonical-form) [scalar](../../../../glossary.md#term-scalar) [target](../../../../glossary.md#term-target) [Shape](../../../../glossary.md#shape).
- Accept non-negative static extents, including zero; numeric `-1` has no expand meaning.
- Retain the exact scalar, zero-[extent](../../../../glossary.md#term-extent), static, mixed dynamic, or fully dynamic target `Shape`
  supplied to the Shape overload.
- Require target rank to be at least input rank and validate aligned axes from right to left by
  structural equality or an input-side static singleton.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [structural equality glossary](../../../../glossary.md#term-structural-equality); [singleton glossary](../../../../glossary.md#term-singleton))</span>
- Accept additional leading target [axes](../../../../glossary.md#term-axis) of every valid Dimension category.
- Reject locally unprovable dynamic compatibility instead of binding symbols or creating hidden
  constraints.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- For a fully static target and any resolved input layout, derive one new view layout by preserving
  input offset and unchanged aligned strides while assigning zero strides to additional leading
  axes and expanded singleton axes.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [strides glossary](../../../../glossary.md#term-stride))</span>
- Derive geometry from dense, offset-dense, strided, and already-broadcast input [layouts](../../../../glossary.md#layout).
- Leave result layout unresolved for a dynamic target or unresolved input layout.
- Preserve exact input [data type](../../../../glossary.md#data-type--datatype) and [gradient](../../../../glossary.md#term-gradient) eligibility.
- Construct exact `ShapeTransformKind.EXPAND`, `TargetShapeAttrs(targetShape)`, and [provenance](../../../../glossary.md#provenance)
  `[input]`.
- Call `TensorFactory.createDerived` exactly once with no label or [host storage](../../../../glossary.md#host-storage).
- Return a fresh Tensor for every valid call, including identity-like, scalar, repeated, and nested
  requests.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Add focused tests and update [Tensor](../../../../glossary.md#tensor)'s exact public method inventory.
- Complete the mandatory independent documentation pass before marking the task Complete.

## Out of [scope](../../../../glossary.md#term-scope)

- reshape changes, reshape [inference](../../../../glossary.md#term-inference), numeric `-1`, [element](../../../../glossary.md#term-element)-count comparison, or modification of
  `TensorReshapeExpressions`
- `expandDims`, squeeze, permute, transpose, tile, repeat, broadcast-to aliases, or another public
  Tensor method or overload<span data-glossary-navigation="true"> ([transpose glossary](../../../../glossary.md#term-transpose); [Tensor glossary](../../../../glossary.md#tensor))</span>
- changes to Shape, Dimension, ShapeBroadcast, LayoutDescriptor, LayoutKind, TensorDescriptor,
  TensorFactory, TensorProvenance, Operation, ShapeTransformKind, or TargetShapeAttrs<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [Operation glossary](../../../../glossary.md#operation))</span>
- target rank below input rank, shrinking non-singleton dimensions, binding dynamic dimensions,
  resolving unequal symbols, or recording deferred symbolic constraints<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [rank glossary](../../../../glossary.md#term-rank); [singleton glossary](../../../../glossary.md#term-singleton); [binding glossary](../../../../glossary.md#term-binding))</span>
- value reads or repetition, storage lookup or alias attachment, allocation, copy, capacity or
  lifetime validation, mutation, or eager materialization<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [lifetime glossary](../../../../glossary.md#term-lifetime); [validation glossary](../../../../glossary.md#term-validation); [eager glossary](../../../../glossary.md#term-eager-evaluation); [materialization glossary](../../../../glossary.md#materialization))</span>
- forcing [contiguous](../../../../glossary.md#term-contiguous) input, choosing [alias](../../../../glossary.md#aliasing-and-arena) versus copy, or promising zero-copy execution
- returning the input, expansion-chain [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), caching, interning, or deduplication
- [gradient](../../../../glossary.md#term-gradient) formulas, broadcast-gradient reduction, [autograd](../../../../glossary.md#autograd), or training behavior
- graph capture, compiler passes, planning requirements, prepare lowering, backend support, runtime
  behavior, execution, or ONNX mapping<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- another [module](../../../../glossary.md#term-module), [dependency](../../../../glossary.md#term-dependency), [Gradle](../../../../glossary.md#term-gradle)/build change, architecture change, or task-0017E specification

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) — authoritative model ownership and layer
  boundaries.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [Architecture overview](../../../../architecture/overview.md) — [compile](../../../../glossary.md#compile)/[prepare](../../../../glossary.md#prepare)/[run](../../../../glossary.md#term-run) separation.
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span> — [Tensor](../../../../glossary.md#tensor) expressions precede [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names).
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span> — model contains no runtime or
  concrete backend behavior.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span> — model remains independent of
  compiler, runtime, prepare, engine, and backends.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [engine glossary](../../../../glossary.md#term-engine); [backends glossary](../../../../glossary.md#backend))</span>
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
  — later layers own materialization, lowering, storage, and execution.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span> — task [scope](../../../../glossary.md#term-scope), evidence, and clean-context rules.
- [Model capabilities](../capabilities.md) — selected `expand` and broadcast-[view](../../../../glossary.md#term-view-meanings) capability.

## Capability origin

The selected capability baseline requires `expand` and expanded/broadcast [views](../../../../glossary.md#term-view-meanings). The read-only
`legacy/pre-rewrite` implementation exposes `Tensor.expand(int...)`, right-aligns source and
[target](../../../../glossary.md#term-target), permits leading-[axis](../../../../glossary.md#term-axis) and [singleton](../../../../glossary.md#term-singleton) expansion, preserves source [strides](../../../../glossary.md#term-stride) for equal axes, and
uses zero strides for repeated axes. Tests cover leading-[rank](../../../../glossary.md#term-rank) and singleton expansion, [BOOL](../../../../glossary.md#data-type--datatype) and
floating data, view behavior, later [contiguous](../../../../glossary.md#term-contiguous) [materialization](../../../../glossary.md#materialization), and backward reduction.

This task preserves only [model](../../../../glossary.md#term-model)-level capability: target [validation](../../../../glossary.md#term-validation), logical semantics, [descriptor](../../../../glossary.md#term-descriptor)
geometry, [metadata](../../../../glossary.md#term-metadata) retention, and [provenance](../../../../glossary.md#provenance). It adapts legacy positive-only int [Shapes](../../../../glossary.md#shape) to the
completed long-sized [scalar](../../../../glossary.md#term-scalar)/zero/dynamic model and excludes immediate storage [aliasing](../../../../glossary.md#term-aliasing), [mutable](../../../../glossary.md#term-mutable)
[graph](../../../../glossary.md#graph) [nodes](../../../../glossary.md#node), [gradient](../../../../glossary.md#term-gradient) callbacks, [compiler](../../../../glossary.md#term-compiler)/[lowering](../../../../glossary.md#lowering) code, [kernels](../../../../glossary.md#kernel), [runtime](../../../../glossary.md#term-runtime), and [backends](../../../../glossary.md#backend).

## Architecture constraints

- [Tensor](../../../../glossary.md#tensor) remains public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state and is not an [IR](../../../../glossary.md#term-intermediate-representation-ir) [node](../../../../glossary.md#node).
- [Operation](../../../../glossary.md#operation) records [backend](../../../../glossary.md#backend)-neutral semantics and no backend support or route.
- LayoutDescriptor records logical geometry, not physical storage or executable [aliasing](../../../../glossary.md#term-aliasing).
- [Model](../../../../glossary.md#term-model) may validate only compatibility provable from current [immutable](../../../../glossary.md#term-immutable) [Shape](../../../../glossary.md#shape) [metadata](../../../../glossary.md#term-metadata).
- Model must not bind symbols or create [graph](../../../../glossary.md#graph)-wide constraints.
- [Compiler](../../../../glossary.md#term-compiler) owns [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction), constraints, and [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form).
- [Planning](../../../../glossary.md#term-planning) owns logical [materialization](../../../../glossary.md#materialization) requirements; backend [prepare](../../../../glossary.md#prepare) owns concrete [lowering](../../../../glossary.md#lowering).
- [Runtime](../../../../glossary.md#term-runtime) owns prepared storage/[residency](../../../../glossary.md#residency) and execution.
- No [service locator](../../../../glossary.md#term-service-locator), backend [lookup](../../../../glossary.md#term-lookup), new [dependency](../../../../glossary.md#term-dependency), or architecture change is authorized.

## Package impact

No new package is introduced.

- Public overloads remain in `io.github.pho001.synaptik.model.tensor.Tensor`.
- Package-private `TensorExpandExpressions` lives beside [Tensor](../../../../glossary.md#tensor) and existing expression helpers.
- Focused tests mirror `model.tensor` for package-private structural inspection.
- Existing shape, layout, operation-layout, descriptor, provenance, and factory packages remain
  unchanged.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) surface

Add exactly:

```java
public Tensor expand(long... requestedShape) {
    return TensorExpandExpressions.apply(this, requestedShape);
}

public Tensor expand(Shape targetShape) {
    return TensorExpandExpressions.apply(this, targetShape);
}
```

Do not add a static form, [alias](../../../../glossary.md#aliasing-and-arena), [descriptor](../../../../glossary.md#term-descriptor) overload, factory entry, or default [target](../../../../glossary.md#term-target).

### Helper structure

`TensorExpandExpressions` is package-private, final, field-free, and non-instantiable. Apart from
one private constructor, it declares exactly:

```java
static Tensor apply(Tensor input, long[] requestedShape)
static Tensor apply(Tensor input, Shape targetShape)
private static void validateExpansion(Shape inputShape, Shape targetShape)
private static Optional<LayoutDescriptor> resolveViewLayout(
        TensorDescriptor inputDescriptor, Shape targetShape)
private static long[] deriveExpandedStrides(
        Shape inputShape, LayoutDescriptor inputLayout, Shape targetShape)
private static Tensor create(
        Tensor input,
        TensorDescriptor inputDescriptor,
        Shape targetShape,
        Optional<LayoutDescriptor> resultLayout)
```

Do not add fields, nested types, another constructor, helper file, or hidden overload.

### [Validation](../../../../glossary.md#term-validation) and [normalization](../../../../glossary.md#term-normalization) order

For `apply(Tensor, long[])`:

1. null-check input with exact message `input`;
2. null-check requestedShape with exact message `requestedShape`;
3. read the exact input [descriptor](../../../../glossary.md#term-descriptor) once and its [Shape](../../../../glossary.md#shape) once;
4. construct [target](../../../../glossary.md#term-target) through `Shape.of(requestedShape)`;
5. validate expansion;
6. resolve or defer [layout](../../../../glossary.md#layout);
7. construct the result once.

`Shape.of` supplies existing static-[dimension](../../../../glossary.md#dimension) validation. The first negative [extent](../../../../glossary.md#term-extent), including
`-1`, fails with `IllegalArgumentException` message
`Static dimension size must be non-negative: <value>`. The array is never retained or mutated.
Empty varargs produce `Shape.scalar()`.

For `apply(Tensor, Shape)`:

1. null-check input with exact message `input`;
2. null-check targetShape with exact message `targetShape`;
3. read the exact input descriptor once and its Shape once;
4. validate expansion;
5. resolve or defer layout;
6. construct the result once.

The exact supplied target Shape reference becomes both descriptor Shape and attribute target.

### Directional right-aligned compatibility

If [target](../../../../glossary.md#term-target) [rank](../../../../glossary.md#term-rank) is below input rank, throw `IllegalArgumentException` with exact message:

```text
expand target rank <targetRank> must be at least input rank <inputRank>
```

Otherwise compute `rankOffset = targetRank - inputRank` and inspect aligned input [axes](../../../../glossary.md#term-axis) in ascending
order. An aligned pair is valid only when its [Dimensions](../../../../glossary.md#dimension) are structurally equal or the input
Dimension is `StaticDimension(1)`. For the first invalid pair throw:

```text
cannot expand input shape <inputShape> to target shape <targetShape> at target axis <targetAxis>
```

Axes before `rankOffset` are new leading axes and are valid. Consequences include:

- [scalar](../../../../glossary.md#term-scalar) input may expand to any target [Shape](../../../../glossary.md#shape);
- `[3]` may expand to `[2, 3]`;
- `[1, 3]` may expand to `[2, 3]`;
- `[N, 1]` may expand to `[N, 4]`;
- `[1]` may expand to `[N]`;
- `[N]` may expand to `[N]`, but not locally to `[1]`, `[2]`, or `[M]`;
- `[2, 3]` may not expand to `[2, 4]` or a lower-rank target.

Target [singleton](../../../../glossary.md#term-singleton) is not permission to shrink an input. Do not use symmetric [broadcasting](../../../../glossary.md#broadcasting) as a
constraint solver or replace the caller's exact target Shape.

### Resolved and unresolved [layout](../../../../glossary.md#layout)

Read `inputDescriptor.layout()` exactly once. Return `Optional.empty()` when [target](../../../../glossary.md#term-target) is dynamic or
input layout is absent. Do not guess [strides](../../../../glossary.md#term-stride) or synthesize a [materialization](../../../../glossary.md#materialization) request.

For a static target and present input layout, allocate one target-[rank](../../../../glossary.md#term-rank) `long[]` and derive:

- zero for every new leading [axis](../../../../glossary.md#term-axis);
- zero for an aligned source [extent](../../../../glossary.md#term-extent) one whose target extent differs from one;
- otherwise the exact `inputLayout.stride(sourceAxis)`, including existing zero/non-canonical
  strides.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>

Create exactly:

```java
LayoutDescriptor.of(
        targetShape,
        derivedStrides,
        inputLayout.storageOffset(),
        true)
```

The result is always marked as a [view](../../../../glossary.md#term-view-meanings). Repetition of target extents greater than one through zero
stride classifies as `BROADCAST_ZERO_STRIDE`; [identity](../../../../glossary.md#term-identity-meanings)-like, [singleton](../../../../glossary.md#term-singleton), or empty geometry may
classify as dense, offset-dense, or strided. Preserve the exact offset and never reuse the input
layout object. This is logical [metadata](../../../../glossary.md#term-metadata) only; no storage is attached and zero-copy execution is
not promised.

### Result construction

Common construction creates exactly:

```java
TensorDescriptor descriptor = new TensorDescriptor(
        inputDescriptor.dataType(),
        targetShape,
        resultLayout,
        inputDescriptor.requiresGrad());
TargetShapeAttrs attrs = new TargetShapeAttrs(targetShape);
Operation operation = new Operation(ShapeTransformKind.EXPAND, attrs);
TensorProvenance provenance = new TensorProvenance(operation, List.of(input));
return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
```

Every success consumes one fresh [Tensor](../../../../glossary.md#tensor) ID and has absent label/storage. Failures before
`createDerived` consume no ID; exhaustion remains the final factory failure.

## Affected files

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorExpandExpressions.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorExpandExpressionTest.java`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/tasks/0017d1-expand-tensor-expressions.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

At most the ten paths above may change. Production is limited to [Tensor](../../../../glossary.md#tensor) plus one new
package-private helper; tests to inventory plus one focused suite; documentation to the public
[model](../../../../glossary.md#term-model) [API](../../../../glossary.md#term-application-programming-interface-api)/status and [planning](../../../../glossary.md#term-planning) evidence.

If another production type, helper, overload, foundational edit, [dependency](../../../../glossary.md#term-dependency), build/architecture
change, or eleventh path is needed, stop and report. Do not create task 0017E.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Fully document both public overloads and every helper contract.
- Explain logical repetition without claiming [eager](../../../../glossary.md#term-eager-evaluation) values or storage.
- Document raw-array [ownership](../../../../glossary.md#term-ownership-meanings), [scalar](../../../../glossary.md#term-scalar) empty request, non-negative [dimensions](../../../../glossary.md#dimension), no `-1`, exact Shape
  retention, right alignment, leading axes, singleton expansion, and dynamic compatibility.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [axes glossary](../../../../glossary.md#term-axis); [singleton glossary](../../../../glossary.md#term-singleton))</span>
- Document result type, [Shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout), eligibility, label, storage, [provenance](../../../../glossary.md#provenance), and freshness.
- Document every parameter, result, null failure, compatibility/static-dimension/arithmetic
  failure, and identifier exhaustion.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>
- Explain zero-stride derivation, offset retention, all resolved input layout kinds, unresolved
  dynamic geometry, and why view metadata is not attached storage or execution proof.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Independently review related shape/layout/descriptor/factory/provenance/operation/reshape/
  contiguous Javadocs and stop on any out-of-scope contradiction.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation); [contiguous glossary](../../../../glossary.md#term-contiguous); [scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- Exactly two expand overloads exist; [Tensor](../../../../glossary.md#tensor)'s declared public method inventory rises 79 to 81.
- Each overload delegates once; helper is exact: final, package-private, field-free, one private
  constructor, six methods.
- Null order/messages, raw ownership, negative handling, scalar empty request, and exact Shape
  retention match this specification.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Rank, equality, input-singleton, leading-axis, zero-extent, scalar, and dynamic rules match this
  specification.<span data-glossary-navigation="true"> ([Rank glossary](../../../../glossary.md#term-rank); [singleton glossary](../../../../glossary.md#term-singleton); [axis glossary](../../../../glossary.md#term-axis); [extent glossary](../../../../glossary.md#term-extent))</span>
- All six data types and valid eligibility states retain exact [metadata](../../../../glossary.md#term-metadata).
- Dense, offset, strided, and broadcast resolved inputs derive correct strides, offset, view flag,
  kind, and span; dynamic/unresolved cases stay unresolved.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Result records exact EXPAND/[target](../../../../glossary.md#term-target)/[input], absent label/storage, and fresh [identity](../../../../glossary.md#term-identity-meanings).
- Same-[Shape](../../../../glossary.md#shape), repeated, and nested calls remain explicit.
- Early failures consume no ID; final factory exhaustion is preserved.
- No values/storage or cross-layer behavior is introduced.
- Independent documentation review and status/evidence synchronization complete before status
  becomes Complete.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation pass:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorExpandExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover [API](../../../../glossary.md#term-application-programming-interface-api)/helper [shape](../../../../glossary.md#shape); all data types and eligibility; array [ownership](../../../../glossary.md#term-ownership-meanings); [scalar](../../../../glossary.md#term-scalar),
zero, static, and dynamic [targets](../../../../glossary.md#term-target); leading/equal/[singleton](../../../../glossary.md#term-singleton) cases; exact failures; dense, offset,
strided, and broadcast [layouts](../../../../glossary.md#layout); preserved/zero [strides](../../../../glossary.md#term-stride), offset/kind/[view](../../../../glossary.md#term-view-meanings)/span; unresolved geometry;
[operation](../../../../glossary.md#operation)/attributes/[provenance](../../../../glossary.md#provenance); absent label/storage; dead-storage non-interference; freshness;
[identity](../../../../glossary.md#term-identity-meanings) side effects; and exhaustion.

Inspect `javap -p -c -s`, reflection, imports, and source. Confirm two one-call delegations, exact
six-method helper, one compatibility path, one layout decision, one stride derivation, one shared
construction path, and no value/storage/cross-layer behavior. Validate generated [Javadoc](../../../../glossary.md#term-javadoc),
executable examples, API/glossary status, links, anchors, fences, whitespace, exact ten paths,
synchronized statuses, and no task-0017E specification.

## Dependencies

- 0002: [Shape](../../../../glossary.md#shape), [Dimension](../../../../glossary.md#dimension), [scalar](../../../../glossary.md#term-scalar)/zero/dynamic forms, [structural equality](../../../../glossary.md#term-structural-equality), defensive construction.
- 0003: resolved [strides](../../../../glossary.md#term-stride), offsets, [views](../../../../glossary.md#term-view-meanings), [layout](../../../../glossary.md#layout) classification, and referenced spans.
- 0007: resolved-or-unresolved TensorDescriptor.
- 0011–0013: [Tensor](../../../../glossary.md#tensor), centralized derived [identity](../../../../glossary.md#term-identity-meanings), and [immutable](../../../../glossary.md#term-immutable) [provenance](../../../../glossary.md#provenance).
- 0017C: exact EXPAND semantics and TargetShapeAttrs.
- 0017D: adjacent target-Shape expression pattern and reshape/expand boundary; contextual, not a
  production dependency.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Follow-up tasks

- 0017E remains Draft for [axis](../../../../glossary.md#term-axis)-transform semantics.
- 0017F remains Draft for permute, [transpose](../../../../glossary.md#term-transpose), expand-[dimensions](../../../../glossary.md#dimension), and squeeze expressions.
- [Compiler](../../../../glossary.md#term-compiler) later owns dynamic constraints and expansion [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form).
- [Planning](../../../../glossary.md#term-planning)/[backend](../../../../glossary.md#backend) [prepare](../../../../glossary.md#prepare) later own [materialization](../../../../glossary.md#materialization) and [alias](../../../../glossary.md#aliasing-and-arena)/copy [lowering](../../../../glossary.md#lowering).
- [Runtime](../../../../glossary.md#term-runtime) later executes prepared work; training/compiler-generated tasks own backward reduction.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. [Model](../../../../glossary.md#term-model) already owns all composed contracts. [Compiler](../../../../glossary.md#term-compiler), [planning](../../../../glossary.md#term-planning), [prepare](../../../../glossary.md#prepare),
[backend](../../../../glossary.md#backend), [runtime](../../../../glossary.md#term-runtime), and training responsibilities remain unchanged. Stop if symbolic [binding](../../../../glossary.md#term-binding),
storage [aliasing](../../../../glossary.md#term-aliasing), [materialization](../../../../glossary.md#materialization) policy, [gradient](../../../../glossary.md#term-gradient) rules, another [dependency](../../../../glossary.md#term-dependency), or architecture
change becomes necessary.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0003/0007/0011/0012/0013/0017C/0017D/0017D1, Tensor
API, Compile API, Training API, glossary, current Shape/Dimension/LayoutDescriptor/LayoutKind/
TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/ShapeTransformKind/
TargetShapeAttrs/reshape contracts and tests, and Java 26 Gradle configuration.

Implement task 0017D1 exactly. Modify Tensor.java and add package-private final
TensorExpandExpressions.java. Update TensorTest only for the exact two-overload API expansion and
add TensorExpandExpressionTest. Add exactly expand(long...) and expand(Shape), each delegating once
to the matching helper overload.

The stateless helper has exactly six specified methods. Raw requests use Shape.of, accept
non-negative dimensions including zero, treat empty as scalar, and give -1 no special meaning.
Retain exact Shape-overload references. Require target rank at least input rank; accept aligned
pairs only when equal or input is a static singleton; accept new leading axes; reject unprovable
dynamic combinations.

For static target plus any resolved input layout, derive a new view layout with exact input offset,
preserved aligned strides, and zero strides for leading or expanded singleton axes. Otherwise leave
layout unresolved. Retain exact type/target/eligibility, create exact EXPAND + TargetShapeAttrs and
[input] provenance, and call createDerived once with no label/storage. Every request is fresh.

Do not modify reshape/foundational contracts, bind symbols, inspect/repeat/copy values or storage,
attach aliases, force materialization, return input, canonicalize, add APIs/helpers/types, define
gradients, capture graphs, or add cross-layer behavior, dependencies, build/architecture changes,
or later specs. Stop beyond ten paths or on architecture uncertainty.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/Compile API/glossary/planning, record no-change conclusions, and rerun validation.

Update 0017D1, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0017E Draft without a specification. Do not commit/push.
```

## Local decisions

- Expand mirrors reshape with raw `long...` and exact [`Shape`](../../../../glossary.md#shape), but has no [inference](../../../../glossary.md#term-inference) sentinel.
- Empty raw [dimensions](../../../../glossary.md#dimension) mean [scalar](../../../../glossary.md#term-scalar), consistent with Shape.of.
- [Validation](../../../../glossary.md#term-validation) is directional: [target](../../../../glossary.md#term-target) [singleton](../../../../glossary.md#term-singleton) cannot shrink an input.
- Equal dynamic symbols and input static singleton to dynamic target are locally provable; other
  dynamic combinations are rejected rather than hidden as constraints.
- Leading [axes](../../../../glossary.md#term-axis) are expansion from implicit singleton axes.
- Every resolved input layout supports stride derivation; contiguity is unnecessary because expand
  preserves aligned source strides and inserts zero strides.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [stride glossary](../../../../glossary.md#term-stride); [contiguity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- Resolved results are [view](../../../../glossary.md#term-view-meanings) [metadata](../../../../glossary.md#term-metadata) without storage attachment or zero-copy guarantee.
- Every call is fresh; [compiler](../../../../glossary.md#term-compiler) later owns [identity](../../../../glossary.md#term-identity-meanings)/nested [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form).

## Known limitations

- Unprovable dynamic expansion is rejected and no [graph](../../../../glossary.md#graph)-wide constraint is recorded.
- Dynamic [target](../../../../glossary.md#term-target) or unresolved input [layout](../../../../glossary.md#layout) stays unresolved.
- No host-storage alias, gradient, compiler capture, planning requirement, backend lowering,
  runtime execution, ONNX mapping, or conformance behavior exists yet.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [alias glossary](../../../../glossary.md#aliasing-and-arena); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed the architecture and focused [lifecycle](../../../../glossary.md#lifecycle)/[module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency)/[runtime](../../../../glossary.md#term-runtime)-boundary docs;
documentation/planning rules; roadmap; [model](../../../../glossary.md#term-model) capabilities/master plan; prerequisite tasks; current
[Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension)/ShapeBroadcast, [layout](../../../../glossary.md#layout), [descriptor](../../../../glossary.md#term-descriptor), [Tensor](../../../../glossary.md#tensor)/factory/[provenance](../../../../glossary.md#provenance), [operation](../../../../glossary.md#operation), and
shape-transform source/tests; Tensor/[Compile](../../../../glossary.md#compile)/Training APIs, glossary, and Java 26 [Gradle](../../../../glossary.md#term-gradle) baseline.

The read-only legacy branch confirms varargs expand, right [alignment](../../../../glossary.md#term-alignment), [singleton](../../../../glossary.md#term-singleton)/leading expansion,
preserved and zero [strides](../../../../glossary.md#term-stride), [view](../../../../glossary.md#term-view-meanings) semantics, all selected data categories, later [materialization](../../../../glossary.md#materialization),
and backward reduction. Coupled legacy storage, [graph](../../../../glossary.md#graph), [gradient](../../../../glossary.md#term-gradient), [compiler](../../../../glossary.md#term-compiler), [kernel](../../../../glossary.md#kernel), and runtime
design is excluded.

Current contracts support this task without a new public type, package, dependency, foundational
edit, or architecture decision.

Planning validation after synchronizing this task, the model master plan, and roadmap:

- `git diff --check` passed, and targeted trailing-whitespace scans found no matches.
- Exact planning scope is three paths: this task, model master plan, and roadmap. No Java, Gradle,
  architecture, API, glossary, completed-task, or other-module file changed during planning.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- All 20 [canonical](../../../../glossary.md#term-canonical-form) task sections are present.
- Markdown backtick fences are balanced: sixteen in this task, two in the master plan, and zero in
  the roadmap.
- All 183 local Markdown file links across the three changed planning files resolve.
- Every changed file ends with a newline.
- Task, master plan, and roadmap consistently identify 0017D1 as Ready and 0017E as Draft; no
  detailed 0017E specification exists.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- The model task sequence contains 73 ordered rows with no duplicate order number.
- Package review confirms no new package: public methods and the package-private helper stay in
  `model.tensor`, and tests mirror that package.
- Scope and dependency review confirms the implementation can stay within ten authorized paths
  and compose only completed model contracts.
- Granularity review confirms expansion compatibility/stride algebra is one isolated concept and
  remains separate from task 0017E axis-transform semantics.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>

Implementation and independent documentation validation:

- Implementation context `/root/implement_model_0017d1` added the two public overloads, exact
  six-method helper, and focused tests. Clean documentation context
  `/root/implement_model_0017d1/docs_review_0017d1` independently inspected the final diff,
  source, tests, generated Javadoc, bytecode, imports, Java 26 build configuration, API pages,
  glossary, and planning state. It applied General plus API/Javadoc style to Java and API work,
  Planning style to evidence/status, and Example format to the executable expand example.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Source and test review confirmed exact raw `Shape.of` ownership, literal non-negative dimensions
  with no <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>`-1` inference, scalar empty request, exact Shape retention, directional right alignment,
  input-singleton and leading-axis expansion, rejection of unprovable dynamics, all six data
  types, unchanged valid gradient eligibility, and fresh unlabeled storage-free results.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Resolved static geometry preserves input offset and unchanged aligned strides and inserts zero
  strides for leading and expanded-singleton axes across dense, offset, strided, and already-
  broadcast layouts. Dynamic target or unresolved input geometry remains unresolved. Construction
  records exact <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>`EXPAND`, `TargetShapeAttrs(targetShape)`, and `[input]` provenance and calls
  `createDerived` once without inspecting or attaching storage.
- The implementation context's first focused command, `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorExpandExpressionTest --tests
  io.github.pho001.synaptik.model.tensor.TensorTest`, compiled and ran 27 tests but failed one
  test-only assertion in `earlyValidationAndArithmeticFailuresConsumeNoTensorIdentity`. The
  assertion incorrectly expected valid expansion geometry to produce a new `ArithmeticException`;
  valid source layouts combined with zero-stride expansion cannot create that claimed case. The
  invalid assertion was removed and the test renamed to describe the remaining early-validation
  contract. Production source did not change, and every subsequent focused, model, and root run
  passed as recorded below.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorExpandExpressionTest --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; XML records 27 tests
  across the two suites (13 expand and 14 Tensor), with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 63 XML suites record 497 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated `Tensor.html` contains both
  overloads, complete parameters/results/failures, raw ownership, directional compatibility,
  dynamic rejection, resolved/unresolved view geometry, storage/materialization boundaries, and
  identifier exhaustion. Package-private helper Javadocs were reviewed in source.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter))</span>
- `./gradlew test` — `BUILD SUCCESSFUL`; all 36 repository test tasks completed without failure.
  No architecture, backend-conformance, or integration test changed because the task adds model
  expression metadata without dependency, backend, or end-to-end behavior.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- `javap -p -c -s` and reflection-backed focused tests confirmed each public overload is one
  matching helper delegation; Tensor exposes exactly 81 declared public methods; and the helper is
  final, package-private, field-free, has one private constructor and exact six methods. Bytecode
  confirms one compatibility path, one layout decision, one stride derivation, one common
  descriptor/attributes/Operation/provenance path, and exactly one `createDerived` call.
- Production helper imports are limited to model and JDK contracts. Source, bytecode, and tests
  found no value repetition, host-storage access or alias attachment, allocation, gradient rule,
  graph/compiler/planning/prepare/runtime/backend dependency, cache, registry, service, implicit
  materialization, or hidden canonicalization.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [host glossary](../../../../glossary.md#term-host); [alias glossary](../../../../glossary.md#aliasing-and-arena); [prepare glossary](../../../../glossary.md#prepare); [cache glossary](../../../../glossary.md#term-cache); [registry glossary](../../../../glossary.md#term-registry); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- The Tensor API example was compiled and executed by Java 26 JShell against the built model
  classes and printed the documented Shape `[2, 4, 3]`, strides `[0, 0, 1]`, offset five,
  `BROADCAST_ZERO_STRIDE`, span eight, view flag, `EXPAND` kind, exact input identity, exact
  dynamic-Shape retention, unresolved dynamic layout, and absent storage. JShell reported a
  macOS preferences-history flush exception only after all statements and output completed; this
  host-only exit limitation did not affect compilation or execution evidence.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- The targeted Markdown validator resolved 232 local file targets and 85 heading anchors across
  the six changed documentation/planning files. Backtick fences are balanced (136 Tensor API,
  four Compile API, sixteen task, two master-plan markers; zero in glossary and roadmap), no
  trailing whitespace was found, every changed file ends with a newline, and `git diff --check`
  passed.
- Exact [scope](../../../../glossary.md#term-scope) is the authorized ten paths: `Tensor.java`, `TensorExpandExpressions.java`,
  `TensorTest.java`, `TensorExpandExpressionTest.java`, Tensor API, Compile API, glossary, this
  task, model master plan, and roadmap. Task 0017D1, the master-plan row/current status/notes, and
  roadmap frontier/table are synchronized as Complete. Task 0017E remains Draft, and no detailed
  0017E specification exists. No commit or push occurred.<span data-glossary-navigation="true"> ([commit glossary](../../../../glossary.md#term-commit))</span>
- Training API remains accurate unchanged because no gradient, autograd, parameter, optimizer,
  publication, or session behavior was added. <span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication))</span>`capabilities.md` already records expand and the
  layer split. Shape/Dimension, LayoutDescriptor/LayoutKind, TensorDescriptor, TensorFactory,
  TensorProvenance, Operation, ShapeTransformKind, TargetShapeAttrs, reshape, and contiguous
  contracts remain accurate because expand composes their existing immutable semantics without
  changing them.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests, backend-
  conformance and integration material, Java 26 Gradle configuration, dependencies, other modules,
  and unrelated tests remain accurate unchanged because ownership, dependency direction,
  lifecycle, build structure, backend behavior, and executable behavior did not change.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>

## Implementation notes

- Added exactly `Tensor.expand(long...)` and `Tensor.expand(Shape)` as one-call delegations to a
  package-private final stateless helper with the six required methods.
- Added focused coverage for API shape, all data types and valid gradient choices, scalar/zero/
  static/dynamic Shapes, exact compatibility failures, every resolved input layout family,
  unresolved geometry, provenance/freshness, storage non-interference, identity consumption, and
  exhaustion.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [scalar glossary](../../../../glossary.md#term-scalar); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Finalized the Tensor/helper Javadocs, Tensor/Compile APIs, glossary, task evidence, model master
  plan, and roadmap in the mandatory independent documentation context.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>
- Added no value work, storage attachment, gradient rule, compiler/planning/prepare/runtime/backend
  behavior, dependency, build change, or architecture change.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Completion summary

- Completed changes: Implemented and documented raw and exact-Shape public expand expressions with
  directional compatibility, conditional same-offset zero-stride view geometry, and exact one-
  input provenance.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [stride glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
- Files changed or created: Exactly two production files, two focused test files, Tensor API,
  Compile API, glossary, this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused tests passed 27/27; all 497 model tests across 63 suites, model
  Javadoc, root tests, Java 26 example execution, bytecode/reflection/import/source/generated-
  Javadoc, link/anchor/fence/terminology/whitespace, exact-scope/status, and diff checks passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: Clean context
  `/root/implement_model_0017d1/docs_review_0017d1` completed the independent pass using General,
  API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and glossary now present expand construction as current;
  Compile API preserves the boundary between current expression metadata and planned capture,
  dynamic constraints, canonicalization, materialization, lowering, and execution.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering))</span>
- Javadoc review: Tensor and helper expand contracts are complete. All reviewed foundational,
  layout, descriptor, factory, provenance, operation, reshape, and contiguous contracts remain
  accurate unchanged.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation); [contiguous glossary](../../../../glossary.md#term-contiguous))</span>
- Glossary impact: Reshape/expand semantics and Tensor status now cover raw ownership, exact Shape
  retention, directional compatibility, conditional zero-stride view metadata, freshness, and
  cross-layer boundaries.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0017D1. Task 0017E remains Draft without a detailed
  specification.

Status: Complete
