# Task 0017D: Reshape [Tensor](../../../../glossary.md#tensor) Expressions

**Editorial reading key (later context, not a status update).**

The application supplies raw extents or an exact Shape to reshape. The helper normalizes request-only -1, checks provable element counts and records a new expression; it does not shuffle or copy stored values. A resolved contiguous input can justify logical view geometry, but the returned Tensor still has no attached storage alias. Unknown count equality is a later Compiler obligation, not proof that arbitrary dynamic sizes fit. Model owns that semantic boundary; physical view-versus-copy decisions stay downstream.

Focused current reading: [Current reshape explanation](../../../../reference/tensor/reshape-expand.md#reshape-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Add public `Tensor.reshape` expressions that preserve the ordered logical [element](../../../../glossary.md#term-element) sequence while
changing its [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [Shape](../../../../glossary.md#shape).

The [API](../../../../glossary.md#term-application-programming-interface-api) supports a legacy-compatible `long...` request with at most one numeric `-1` [inference](../../../../glossary.md#term-inference)
sentinel and an exact `Shape` overload for normalized static or dynamic [model](../../../../glossary.md#term-model) Shapes. Locally
provable element counts must match. A resolved [contiguous](../../../../glossary.md#term-contiguous) input may produce resolved [alias](../../../../glossary.md#aliasing-and-arena)-[view](../../../../glossary.md#term-view-meanings)
geometry for a fully static [target](../../../../glossary.md#term-target); otherwise [layout](../../../../glossary.md#layout) remains unresolved for later [compiler](../../../../glossary.md#term-compiler),
[planning](../../../../glossary.md#term-planning), and [backend](../../../../glossary.md#backend) work.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly `Tensor.reshape(long... requestedShape)` and `Tensor.reshape(Shape targetShape)`.
- Add one package-private final stateless `TensorReshapeExpressions` helper.
- Normalize a [defensive copy](../../../../glossary.md#term-defensive-copy) of the raw request without retaining or mutating caller arrays.
- Accept non-negative requested [dimensions](../../../../glossary.md#dimension) and at most one exact `-1` sentinel; reject every other
  negative value.
- Treat an empty raw request as [canonical](../../../../glossary.md#term-canonical-form) [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [scalar](../../../../glossary.md#term-scalar) [Shape](../../../../glossary.md#shape).
- Infer `-1` only from a fully known input element count and a non-zero checked product of the
  other requested dimensions.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
- Accept zero extents, including inferred zero when the other requested product is non-zero; reject
  ambiguous inference when the known requested product is zero.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference))</span>
- Reject unequal element counts when both input and target counts are known; defer the equality
  constraint when either Shape is dynamic.
- Accept an exact non-null [target](../../../../glossary.md#term-target) Shape reference without copying or [normalization](../../../../glossary.md#term-normalization).
- Retain exact input [DataType](../../../../glossary.md#data-type--datatype), exact normalized target Shape, and unchanged `requiresGrad`.
- Resolve a result alias-view layout only when the input has resolved contiguous geometry and the
  target Shape is fully static.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [view glossary](../../../../glossary.md#term-view-meanings); [layout glossary](../../../../glossary.md#layout); [contiguous glossary](../../../../glossary.md#term-contiguous))</span>
- Preserve the input layout's element offset, use target canonical row-major strides, and mark the
  resolved result layout as a view.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [strides glossary](../../../../glossary.md#term-stride))</span>
- Leave result layout unresolved for absent, strided, broadcast, or dynamic geometry instead of
  choosing materialization.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- Construct exact `ShapeTransformKind.RESHAPE` plus `TargetShapeAttrs(targetShape)` semantics,
  exact one-input provenance, no label, and no storage.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Return a fresh expression for same-shape, repeated, and nested reshape requests.
- Accept all six current DataTypes, valid gradient eligibility, scalar/zero/static/dynamic Shapes,
  and resolved or unresolved input layouts.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
- Add one focused same-package expression test and update `TensorTest` only for the two deliberate
  public overloads.
- Finalize affected Javadocs, Tensor API, Compile API, glossary, task evidence, model master plan,
  and roadmap through the required independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public `Tensor.expand`, expand [broadcasting](../../../../glossary.md#broadcasting), zero-[stride](../../../../glossary.md#term-stride) derivation, or task-0017D1 implementation
- `int...` overload, generic collection/array overload, multiple inferred dimensions, named
  dimensions, flatten convenience, memory-order option, copy/view flag, destination storage, or
  another public type<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension))</span>
- storing raw request arrays, numeric `-1` in [Shape](../../../../glossary.md#shape) or `TargetShapeAttrs`, an inferred-axis field,
  an element-count proof object, or a symbolic equation object<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- [binding](../../../../glossary.md#term-binding) dynamic symbols, solving [graph](../../../../glossary.md#graph)-wide [element](../../../../glossary.md#term-element)-count constraints, accepting `-1` for a
  dynamic input, or silently inventing a dynamic symbol for the sentinel<span data-glossary-navigation="true"> ([symbol glossary](../../../../glossary.md#term-symbol))</span>
- reading or copying tensor values, allocating or attaching host storage, mutating input metadata,
  reusing input layout objects, or claiming eager materialization<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor); [host storage glossary](../../../../glossary.md#host-storage); [metadata glossary](../../../../glossary.md#term-metadata); [layout glossary](../../../../glossary.md#layout); [eager glossary](../../../../glossary.md#term-eager-evaluation); [materialization glossary](../../../../glossary.md#materialization))</span>
- treating a strided or broadcast input as a reshape [view](../../../../glossary.md#term-view-meanings), forcing a [contiguous](../../../../glossary.md#term-contiguous) result, inserting
  `contiguous()`, or selecting [alias](../../../../glossary.md#aliasing-and-arena)/copy policy when view geometry cannot be proven locally
- returning the input, eliminating same-shape or nested reshapes, merging provenance, constant
  folding, or compiler canonicalization<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [constant folding glossary](../../../../glossary.md#term-constant-folding); [compiler glossary](../../../../glossary.md#term-compiler); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- data-type conversion or promotion, new gradient rules, backward operations, saved values,
  autograd expansion, optimizer, or training execution<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- graph capture/traversal, compiler pass implementation, planning ownership/materialization,
  prepare/runtime/backend behavior, engine, tracing, ONNX mapping, or conformance<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [engine glossary](../../../../glossary.md#term-engine); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- changes to Shape/Dimension, LayoutDescriptor/LayoutKind, TensorDescriptor, ShapeTransformKind,
  TargetShapeAttrs, TensorFactory, TensorProvenance, Operation, storage or graph contracts,
  existing expression helpers, or their tests<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation))</span>
- dependencies, Gradle, Java version, AGENTS, ARCHITECTURE, focused architecture documentation,
  architecture tests, capabilities, another module, completed tasks, or unrelated documentation<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- a detailed task-0017D1 or task-0017E specification

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
- [Task 0017A](0017a-contiguous-semantic-kind.md)
- [Task 0017B](0017b-contiguous-tensor-expression.md)
- [Task 0017C](0017c-reshape-and-expand-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes:

```java
Tensor reshape(int... requestedShape)
```

Legacy behavior accepts one `-1`, infers it from a known flat size, rejects multiple sentinels or
incompatible counts, preserves logical [element](../../../../glossary.md#term-element) order and [DataType](../../../../glossary.md#data-type--datatype), creates a [view](../../../../glossary.md#term-view-meanings) for [contiguous](../../../../glossary.md#term-contiguous)
input, and otherwise selects a materialized result. Legacy [gradient](../../../../glossary.md#term-gradient) construction reshapes the
output gradient back to the source [Shape](../../../../glossary.md#shape).

The new [API](../../../../glossary.md#term-application-programming-interface-api) widens raw [dimensions](../../../../glossary.md#dimension) to `long`, permits empty varargs as [scalar](../../../../glossary.md#term-scalar) Shape, supports zero
extents, and adds an exact `Shape` overload so explicit dynamic [model](../../../../glossary.md#term-model) Shapes are representable.
Those differences follow the completed Shape contract rather than legacy int-array, positive-only,
[rank](../../../../glossary.md#term-rank)-at-least-one limitations. Locally unprovable dynamic count equality is deferred; a raw `-1`
cannot be inferred from an unknown input count.

Legacy immediate storage [views](../../../../glossary.md#view), automatic materialized output, [mutable](../../../../glossary.md#term-mutable) arrays, [graph](../../../../glossary.md#graph) builders,
gradient callbacks, [operation](../../../../glossary.md#operation) traits, [layout](../../../../glossary.md#layout) planners, physical [buffers](../../../../glossary.md#term-buffer), [kernels](../../../../glossary.md#kernel), [lowering](../../../../glossary.md#lowering),
fallback, and [runtime](../../../../glossary.md#term-runtime) state are not copied. This task creates only model expression [metadata](../../../../glossary.md#term-metadata) and
locally provable view geometry.

## Architecture constraints

- [`Tensor`](../../../../glossary.md#tensor) remains public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state and is not an intermediate-representation [node](../../../../glossary.md#node).
- The result is a fresh storage-free expression with immutable provenance, not an eagerly reshaped
  buffer, compiled node, runtime value, prepared unit, or executable copy.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [provenance glossary](../../../../glossary.md#provenance); [buffer glossary](../../../../glossary.md#term-buffer); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- `ShapeTransformKind.RESHAPE` plus `TargetShapeAttrs` are the exact semantic pair. Raw `-1` request
  syntax is normalized before attributes construction and is never stored in model Shape state.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Reshape preserves the ordered logical element sequence. When both element counts are known they
  must be equal; unknown dynamic equality becomes a later compiler constraint.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- A resolved reshape view is locally provable only when input geometry is contiguous and target
  geometry is fully static. It uses canonical target strides, the exact input element offset, and
  explicit view metadata.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [contiguous glossary](../../../../glossary.md#term-contiguous); [target glossary](../../../../glossary.md#term-target); [canonical glossary](../../../../glossary.md#term-canonical-form); [strides glossary](../../../../glossary.md#term-stride); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- A resolved input [layout](../../../../glossary.md#layout) is compatible only with a fully static input [Shape](../../../../glossary.md#shape) by existing
  `TensorDescriptor` invariants. No additional storage or backend fact is needed to describe the
  logical view.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Absent/non-contiguous input layout or dynamic target Shape yields unresolved result layout. The
  model must not force materialization, insert another operation, or guess physical geometry.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [operation glossary](../../../../glossary.md#operation))</span>
- Resolved alias-view metadata does not attach host storage, allocate a buffer, promise zero-copy
  execution, or require one backend route. Planning and backend prepare retain those decisions.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [host storage glossary](../../../../glossary.md#host-storage); [backend route glossary](../../../../glossary.md#backend-route); [Planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare))</span>
- Every valid request remains explicit and fresh. Compiler later owns redundant reshape
  elimination, reshape-chain canonicalization, and graph-wide symbolic reasoning.<span data-glossary-navigation="true"> ([canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [graph glossary](../../../../glossary.md#graph))</span>
- All current DataTypes are accepted because reshape changes coordinates, not logical values.
  Gradient eligibility is retained unchanged but no gradient rule is added.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [Gradient glossary](../../../../glossary.md#term-gradient))</span>
- The helper remains in `model.tensor`; semantics remain in `model.operation.layout`; Shape and
  LayoutDescriptor remain foundational immutable values. No project dependency changes.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Stop if implementation requires new public value types, dynamic binding, storage access,
  materialization policy, cross-module state, or architecture changes.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor entries, the package-private helper,
  descriptor/provenance construction, and the existing derived factory seam.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- `io.github.pho001.synaptik.model.operation.layout` — supplies `ShapeTransformKind.RESHAPE` and
  `TargetShapeAttrs`.
- `io.github.pho001.synaptik.model.operation` — supplies [immutable](../../../../glossary.md#term-immutable) [`Operation`](../../../../glossary.md#operation).
- `io.github.pho001.synaptik.model.shape` — supplies normalized static/dynamic/scalar Shape and
  checked element counts.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [Shape glossary](../../../../glossary.md#shape); [element glossary](../../../../glossary.md#term-element))</span>
- `io.github.pho001.synaptik.model.layout` — supplies canonical target strides and explicit
  resolved view geometry.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [target glossary](../../../../glossary.md#term-target); [strides glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings))</span>

No package is added or moved.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — gains the two public reshape overloads.
- `io.github.pho001.synaptik.model.tensor.TensorReshapeExpressions` — package-private stateless
  coordinator colocated with Tensor and `TensorFactory.createDerived`.
- `TensorReshapeExpressionTest` — same-package focused validation of public overloads, private
  normalization rules, layout/provenance construction, and side-effect boundaries.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [normalization glossary](../../../../glossary.md#term-normalization); [layout glossary](../../../../glossary.md#layout))</span>
- `TensorTest` — changes only its exact public-method inventory from 77 to 79 and adds both reshape
  signatures.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) methods

Add exactly:

```java
public Tensor reshape(long... requestedShape) {
    return TensorReshapeExpressions.apply(this, requestedShape);
}

public Tensor reshape(Shape targetShape) {
    return TensorReshapeExpressions.apply(this, targetShape);
}
```

Both methods are public, final by virtue of Tensor, non-static, and non-synchronized. Each body
contains exactly one delegation and no inline [validation](../../../../glossary.md#term-validation), [inference](../../../../glossary.md#term-inference), [layout](../../../../glossary.md#layout) logic, storage access,
or optimization. Java widens individual int literals to the long varargs [element](../../../../glossary.md#term-element) type; an existing
`int[]` is not accepted as a hidden compatibility overload.

### Helper [shape](../../../../glossary.md#shape)

Create one package-private final class with no fields or nested types, one private zero-[argument](../../../../glossary.md#term-argument)
constructor, and exactly these six methods:

```java
static Tensor apply(Tensor input, long[] requestedShape)
static Tensor apply(Tensor input, Shape targetShape)
private static Shape normalizeRequestedShape(Shape inputShape, long[] requestedShape)
private static void validateTargetShape(Shape inputShape, Shape targetShape)
private static Optional<LayoutDescriptor> resolveViewLayout(
        TensorDescriptor inputDescriptor, Shape targetShape)
private static Tensor create(
        Tensor input,
        TensorDescriptor inputDescriptor,
        Shape targetShape,
        Optional<LayoutDescriptor> resultLayout)
```

Add no overload, product helper, validator type, builder, [cache](../../../../glossary.md#term-cache), [registry](../../../../glossary.md#term-registry), service, policy object,
or other production file. Product and sentinel handling remain local to
`normalizeRequestedShape`.

### Raw request [validation](../../../../glossary.md#term-validation) and [inference](../../../../glossary.md#term-inference)

The raw-array entry follows this order:

1. Null-check `input` with message `input`.
2. Null-check `requestedShape` with message `requestedShape`.
3. Read the exact input [descriptor](../../../../glossary.md#term-descriptor) once and its exact [Shape](../../../../glossary.md#shape) once.
4. Pass the Shape and caller array to `normalizeRequestedShape`.
5. Resolve [view](../../../../glossary.md#term-view-meanings) [layout](../../../../glossary.md#layout) and perform common construction.

`normalizeRequestedShape` never mutates or retains the caller array. It first scans every [element](../../../../glossary.md#term-element)
in ascending index order:

- non-negative values are ordinary static [dimensions](../../../../glossary.md#dimension), including zero;
- exact `-1` marks the sole inferred dimension;
- any value below `-1` fails with
  `requestedShape[<index>] must be non-negative or -1: <value>`; and
- a second `-1` fails with `requestedShape must contain at most one -1`.

With no sentinel, create `Shape.of(requestedShape)` and apply the common [target](../../../../glossary.md#term-target) validation. Empty
varargs therefore produce `Shape.scalar()`.

With one sentinel:

- require `inputShape.knownElementCount()` to be present, otherwise fail with
  `cannot infer -1 from dynamic input shape <inputShape>`;
- compute the checked product of all non-sentinel dimensions after the full value/sentinel scan;
  if any such dimension is zero, the known product is zero without multiplying other values;<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- reject zero known product with
  `cannot infer -1 when known requested dimensions have product zero`;
- require exact divisibility, otherwise fail with
  `cannot infer reshape dimension: input element count <count> is not divisible by known requested product <product>`;
- replace the sentinel in a fresh cloned array with `count / product`, create `Shape.of(copy)`, and
  run common target validation.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>

Checked multiplication [overflow](../../../../glossary.md#term-overflow) propagates as `ArithmeticException`. Zero-aware product handling
does not report irrelevant overflow when the mathematical product is already zero.

### Exact [Shape](../../../../glossary.md#shape) [validation](../../../../glossary.md#term-validation)

The Shape overload follows this order:

1. Null-check `input` with message `input`.
2. Null-check `targetShape` with message `targetShape`.
3. Read the exact input [descriptor](../../../../glossary.md#term-descriptor) once and its exact Shape once.
4. Call `validateTargetShape(inputShape, targetShape)`.
5. Resolve [view](../../../../glossary.md#term-view-meanings) [layout](../../../../glossary.md#layout) and perform common construction.

`validateTargetShape` calls `knownElementCount()` once for each Shape. If both are present and
unequal, fail with:

```text
reshape element count mismatch: input=<inputCount>, target=<targetCount>
```

If either count is unknown, accept the [target](../../../../glossary.md#term-target) and defer the equality constraint. [Element](../../../../glossary.md#term-element)-count
[overflow](../../../../glossary.md#term-overflow) propagates. The exact target Shape reference supplied to the overload is retained in
attributes and result descriptor.

### [View](../../../../glossary.md#term-view-meanings)-[layout](../../../../glossary.md#layout) derivation

`resolveViewLayout` reads the input [descriptor](../../../../glossary.md#term-descriptor)'s layout optional exactly once.

Return `Optional.empty()` when:

- [target](../../../../glossary.md#term-target) [Shape](../../../../glossary.md#shape) is dynamic;
- input layout is unresolved; or
- resolved input layout is not [contiguous](../../../../glossary.md#term-contiguous), including strided and broadcast geometry.

Otherwise:

1. Create one temporary `LayoutDescriptor.contiguous(targetShape)` to obtain canonical target
   strides.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [strides glossary](../../../../glossary.md#term-stride))</span>
2. Create the exact stored result with `LayoutDescriptor.of(targetShape, canonical.strides(),
   inputLayout.storageOffset(), true)`.
3. Return that exact new view descriptor in `Optional.of`.

The result is `DENSE_CONTIGUOUS` when offset is zero or `DENSE_WITH_OFFSET` when it is non-zero,
and `isView()` is true in both cases. [Scalar](../../../../glossary.md#term-scalar) and zero-[extent](../../../../glossary.md#term-extent) targets follow existing layout
geometry rules. No input layout object is reused.

### Common result construction

`create` constructs exactly:

1. one `TensorDescriptor` with input DataType, exact target Shape, supplied layout optional, and
   unchanged input <span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout))</span>`requiresGrad`;
2. one `TargetShapeAttrs` retaining the exact [target](../../../../glossary.md#term-target) [Shape](../../../../glossary.md#shape);
3. one `Operation(ShapeTransformKind.RESHAPE, attrs)`;
4. one `TensorProvenance(operation, List.of(input))`; and
5. one `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` call.

Every result is fresh, unlabeled, and storage-free. Input label, [provenance](../../../../glossary.md#provenance), storage, [liveness](../../../../glossary.md#term-liveness),
values, and bytes are not read or changed. The result's resolved [view](../../../../glossary.md#term-view-meanings) [metadata](../../../../glossary.md#term-metadata) is a logical fact,
not an attached [host](../../../../glossary.md#term-host)-storage [alias](../../../../glossary.md#aliasing-and-arena) or an executable guarantee.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorReshapeExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorReshapeExpressionTest.java`

Documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `docs/api/training-api.md`
- `docs/planning/modules/model/capabilities.md`
- Shape/Dimension, LayoutDescriptor/LayoutKind, TensorDescriptor, ShapeTransformKind,
  TargetShapeAttrs, TensorFactory, TensorProvenance, Operation, storage/graph contracts, existing
  expression helpers, and their Javadocs/tests<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [Operation glossary](../../../../glossary.md#operation); [graph glossary](../../../../glossary.md#graph))</span>
- focused architecture documents, ADRs, architecture tests, backend-conformance material,
  integration tests, Gradle configuration, and every other module<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, two test files, three [API](../../../../glossary.md#term-application-programming-interface-api)/glossary files, and three [planning](../../../../glossary.md#term-planning) files:
ten paths total.

Do not modify ShapeTransformKind, TargetShapeAttrs, another existing Java source/test,
capabilities, Training API, [Gradle](../../../../glossary.md#term-gradle), AGENTS, architecture documents/tests, another [module](../../../../glossary.md#term-module), completed
task, or unrelated documentation. Stop if another production type/file, public overload, helper
method, [dependency](../../../../glossary.md#term-dependency), cross-layer behavior, or eleventh path is needed. Do not create task 0017D1 or
0017E.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Update the Tensor type overview to include reshape and distinguish resolved alias-view geometry
  from contiguous expression geometry and unresolved ordinary expression layouts.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [alias glossary](../../../../glossary.md#aliasing-and-arena); [view glossary](../../../../glossary.md#term-view-meanings); [contiguous glossary](../../../../glossary.md#term-contiguous); [layouts glossary](../../../../glossary.md#layout))</span>
- Document both public overloads completely, including array [ownership](../../../../glossary.md#term-ownership-meanings), `-1`, scalar empty request,
  zero extents, dynamic deferral, element-count failures, exact Shape retention, and overflow.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [element glossary](../../../../glossary.md#term-element); [Shape glossary](../../../../glossary.md#shape); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- Document result DataType/Shape/eligibility/layout/provenance/label/storage/freshness for every
  branch.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [provenance glossary](../../../../glossary.md#provenance))</span>
- Explain that resolved view layout preserves canonical target strides and input offset but does
  not attach storage or promise zero-copy execution.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [target glossary](../../../../glossary.md#term-target); [strides glossary](../../../../glossary.md#term-stride))</span>
- Explain why absent/non-contiguous/dynamic geometry remains unresolved and why the model does not
  insert contiguous materialization.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [materialization glossary](../../../../glossary.md#materialization))</span>
- Document helper class, private constructor, all six methods, parameters, ownership, mutation,
  exact return semantics, validation order, and every failure.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [validation glossary](../../../../glossary.md#term-validation))</span>
- Document `NullPointerException`, `IllegalArgumentException`, `ArithmeticException`, and
  `IllegalStateException` on the applicable public/helper methods.
- Review ShapeTransformKind, TargetShapeAttrs, Shape, LayoutDescriptor, LayoutKind,
  TensorDescriptor, TensorFactory, TensorProvenance, Operation, and Contiguous Javadocs and record
  why they remain accurate, or stop on inconsistency.<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation))</span>

## Acceptance criteria

- Tensor adds exactly two reshape overloads and its declared-public-method count changes from 77
  to 79 without another public API change.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Each public overload delegates exactly once to the matching helper overload.
- The helper is package-private final/stateless with one private constructor and exactly the six
  specified methods, visibilities, parameters, and return types.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter))</span>
- Null checks, raw-request [scan](../../../../glossary.md#term-cumulative-scan), `-1` inference, count validation, view resolution, and construction
  follow the exact documented order and messages.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Raw arrays are neither retained nor mutated; exact [Shape](../../../../glossary.md#shape) overload references are retained.
- Empty varargs produce [scalar](../../../../glossary.md#term-scalar) Shape; zero extents are accepted; one inferable `-1` works; invalid
  negatives, duplicate sentinels, dynamic inference, zero-product ambiguity, non-divisibility,
  and known count mismatch fail exactly.
- When either exact Shape count is dynamic, construction succeeds without inventing a binding or
  suppressing later compiler validation.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- All six DataTypes and valid [gradient](../../../../glossary.md#term-gradient) flags are retained unchanged.
- Resolved contiguous input plus static target produces a new canonical-stride, same-offset,
  view-marked resolved layout. Offset zero and non-zero cases classify correctly.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [target glossary](../../../../glossary.md#term-target); [canonical glossary](../../../../glossary.md#term-canonical-form); [stride glossary](../../../../glossary.md#term-stride); [layout glossary](../../../../glossary.md#layout))</span>
- Unresolved, strided, broadcast, or dynamic cases produce unresolved result layout without
  materialization or layout reuse.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- Exact RESHAPE kind, exact TargetShapeAttrs/target Shape, one-input provenance, fresh ID, absent
  label, and absent storage are recorded.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Same-shape, repeated, and nested reshapes remain fresh explicit expressions.
- Early validation/layout arithmetic failures consume no Tensor ID; exhaustion propagates only at
  final factory delegation after valid model construction.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Input descriptor/layout/label/provenance/storage/liveness/values remain unchanged; no eager copy,
  allocation, gradient, compiler, planning, prepare, runtime, backend, graph, ONNX, dependency, or
  build behavior is added.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [liveness glossary](../../../../glossary.md#term-liveness); [eager glossary](../../../../glossary.md#term-eager-evaluation); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [graph glossary](../../../../glossary.md#graph); [ONNX glossary](../../../../glossary.md#term-onnx); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused/aggregate tests, model Javadoc, root tests, reflection/javap/bytecode/import/source/scope
  checks, documentation examples/links/formatting, and synchronized statuses pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>
- A separate clean-context documentation agent finalizes affected Javadocs, Tensor API, Compile
  API, glossary, task evidence, master plan, and roadmap and records related no-change conclusions.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- Task 0017D becomes Complete only after both passes. Task 0017D1 remains Draft without a detailed
  specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorReshapeExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover exact public/helper [API](../../../../glossary.md#term-application-programming-interface-api); all DataTypes/valid eligibility; [scalar](../../../../glossary.md#term-scalar) empty
request; zero extents; exact static/dynamic [Shape](../../../../glossary.md#shape) overloads; valid `-1` positions; defensive array
behavior; every exact validation message and precedence; checked arithmetic; resolved dense and
offset [view](../../../../glossary.md#term-view-meanings) [layouts](../../../../glossary.md#layout); unresolved/strided/broadcast/dynamic layouts; exact [operation](../../../../glossary.md#operation)/attributes/
[provenance](../../../../glossary.md#provenance); absent label/storage; dead-storage non-interference; freshness; no [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form);
early failure ID non-consumption; and final identifier exhaustion.

Manually inspect `javap -p -c -s`, reflection, imports, and source. Confirm two one-call public
delegations; exact six-method stateless helper; no hidden overload; one request [scan](../../../../glossary.md#term-cumulative-scan); one count
validation path; one view-layout decision; one [descriptor](../../../../glossary.md#term-descriptor)/attributes/Operation/provenance/factory
path; and no value copy, [host](../../../../glossary.md#term-host) allocation, storage access, [graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler)/[planning](../../../../glossary.md#term-planning)/[prepare](../../../../glossary.md#prepare)/[runtime](../../../../glossary.md#term-runtime)/
[backend](../../../../glossary.md#backend) type, [gradient](../../../../glossary.md#term-gradient) rule, [registry](../../../../glossary.md#term-registry), [cache](../../../../glossary.md#term-cache), service, or hidden [materialization](../../../../glossary.md#materialization). Validate generated
[Javadoc](../../../../glossary.md#term-javadoc), executable documentation examples, [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) API status, glossary, links, anchors,
fences, whitespace, exact ten-path [scope](../../../../glossary.md#term-scope), synchronized statuses, and absence of a task-0017D1
specification.

## Dependencies

- Task 0002 supplies Shape construction, static/dynamic dimensions, and checked known element
  counts.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [dimensions glossary](../../../../glossary.md#dimension); [element glossary](../../../../glossary.md#term-element))</span>
- Task 0003 supplies [canonical](../../../../glossary.md#term-canonical-form) and explicit resolved [layout](../../../../glossary.md#layout) geometry.
- Task 0007 supplies resolved-or-unresolved TensorDescriptor state.
- Tasks 0011–0013 supply public Tensor metadata, centralized derived identity allocation, and
  immutable provenance.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [metadata glossary](../../../../glossary.md#term-metadata); [identity glossary](../../../../glossary.md#term-identity-meanings); [immutable glossary](../../../../glossary.md#term-immutable); [provenance glossary](../../../../glossary.md#provenance))</span>
- Task 0017C supplies exact RESHAPE semantics and normalized TargetShapeAttrs.
- Tasks 0017A–0017B establish the adjacent contiguous semantic/materialization boundary used when
  documenting why reshape does not insert contiguous automatically.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [materialization glossary](../../../../glossary.md#materialization))</span>

## Follow-up tasks

- 0017D1 remains Draft for public expand overloads, right-aligned equal/singleton/leading-axis
  validation, zero-stride alias-view geometry, target Shape retention, and provenance.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [axis glossary](../../../../glossary.md#term-axis); [validation glossary](../../../../glossary.md#term-validation); [stride glossary](../../../../glossary.md#term-stride); [target glossary](../../../../glossary.md#term-target); [Shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance))</span>
- 0017E–0017F remain Draft for [permutation](../../../../glossary.md#permutation), [transpose](../../../../glossary.md#term-transpose), expand-[dimensions](../../../../glossary.md#dimension), and squeeze.
- [Compiler](../../../../glossary.md#term-compiler) later owns deferred dynamic [element](../../../../glossary.md#term-element)-count constraints and reshape-chain [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form).
- [Planning](../../../../glossary.md#term-planning) later derives [materialization](../../../../glossary.md#materialization) requirements for unresolved/non-[view](../../../../glossary.md#term-view-meanings)-compatible reshape.
- [Backend](../../../../glossary.md#backend) [prepare](../../../../glossary.md#prepare) later chooses [alias](../../../../glossary.md#aliasing-and-arena)/copy/native [lowering](../../../../glossary.md#lowering); [runtime](../../../../glossary.md#term-runtime) executes prepared work.
- Training and compiler-generated semantic tasks later own reshape backward behavior.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. [Model](../../../../glossary.md#term-model) already owns public [Tensor](../../../../glossary.md#tensor) expression construction, [operation](../../../../glossary.md#operation)
semantics, [Shape](../../../../glossary.md#shape), and logical [layout](../../../../glossary.md#layout) [descriptors](../../../../glossary.md#term-descriptor). [Compiler](../../../../glossary.md#term-compiler) owns [graph](../../../../glossary.md#graph)-wide [inference](../../../../glossary.md#term-inference) and
[canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form); [planning](../../../../glossary.md#term-planning) owns logical [materialization](../../../../glossary.md#materialization) requirements; [backend](../../../../glossary.md#backend) [prepare](../../../../glossary.md#prepare) owns
concrete [alias](../../../../glossary.md#aliasing-and-arena)/copy [lowering](../../../../glossary.md#lowering) and storage; [runtime](../../../../glossary.md#term-runtime) executes prepared schedules.

If implementation requires dynamic [binding](../../../../glossary.md#term-binding), [eager](../../../../glossary.md#term-eager-evaluation) storage, materialization policy, cross-[module](../../../../glossary.md#term-module)
state, another [dependency](../../../../glossary.md#term-dependency), or architecture change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0003/0007/0011/0012/0013/0017A/0017B/0017C/0017D,
Tensor API, Compile API, Training API, glossary, current Shape/Dimension/LayoutDescriptor/
LayoutKind/TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/ShapeTransformKind/
TargetShapeAttrs contracts and focused tests, and Java 26 Gradle configuration.

Implement task 0017D exactly. Modify Tensor.java and add package-private final
TensorReshapeExpressions.java. Update TensorTest only for the exact two-overload API expansion and
add TensorReshapeExpressionTest. Add exactly reshape(long...) and reshape(Shape), each delegating
once to the matching helper overload.

The stateless helper has exactly the six specified methods. Normalize a copied raw request, accept
non-negative dimensions and one -1, infer only from known input count and non-zero checked known
product, validate known counts, and defer unknown dynamic equality. Resolve a new same-offset,
canonical-stride view layout only for resolved contiguous input plus static target; otherwise use
unresolved layout. Retain exact type/target Shape/eligibility, create exact RESHAPE +
TargetShapeAttrs and [input] provenance, and call createDerived once with no label/storage. Every
valid request is fresh.

Do not implement expand, bind symbols, inspect/copy values or storage, force contiguous or
materialization, return input, canonicalize, add APIs/helpers/types, change existing contracts,
define gradients, capture graphs, or add compiler/planning/prepare/runtime/backend behavior,
dependencies, build/architecture changes, or later specs. Stop beyond ten paths or on architecture
uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/bytecode/import/manual,
documentation/example/link/whitespace/scope/status check. Then hand the actual diff/evidence to a
separate clean-context documentation agent in the same change. It must inspect source/tests/
generated Javadoc, finalize permitted Javadocs/Tensor API/Compile API/glossary/planning, record
related-contract/capability/Training API/architecture no-change conclusions, and rerun validation.

Update task 0017D, model master plan, and roadmap only for planning status/evidence. Do not mark
0017D Complete until both passes succeed. Leave 0017D1 Draft without a specification. Do not
commit or push.
```

## Local decisions

- The former combined reshape/expand expression row is split into 0017D and 0017D1. Each operation
  has independent validation and layout algebra; combining them would exceed one isolated task's
  conceptual and test scope.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [validation glossary](../../../../glossary.md#term-validation); [layout glossary](../../../../glossary.md#layout); [scope glossary](../../../../glossary.md#term-scope))</span>
- Public raw [dimensions](../../../../glossary.md#dimension) use `long...`, matching the model's long static sizes while preserving
  ordinary calls such as `reshape(2, 3)` through primitive [widening](../../../../glossary.md#term-widening).
- The exact [`Shape`](../../../../glossary.md#shape) overload exposes normalized dynamic Shapes without inventing a numeric sentinel
  or a second symbolic-shape representation.
- Empty varargs means [scalar](../../../../glossary.md#term-scalar) Shape, consistent with `Shape.of()` and the [model](../../../../glossary.md#term-model)'s [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) scalar.
- One `-1` is request syntax only. Dynamic input cannot infer it because no unique numeric extent
  exists; callers can use the Shape overload for explicit symbolic targets.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [targets glossary](../../../../glossary.md#term-target))</span>
- Zero-product inference is rejected as ambiguous, while zero may be inferred when the other
  requested product is non-zero and input count is zero.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference))</span>
- Unknown dynamic count equality is deferred rather than rejected or assumed. Compiler later owns
  the constraint.
- Resolved reshape view geometry is published only when contiguity and static target geometry prove
  it locally. It retains input offset, uses canonical target strides, and marks the new descriptor
  as a view.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [contiguity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [canonical glossary](../../../../glossary.md#term-canonical-form); [strides glossary](../../../../glossary.md#term-stride); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Non-contiguous/unresolved/dynamic cases remain unresolved; reshape does not implicitly request or
  insert materialization.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [materialization glossary](../../../../glossary.md#materialization))</span>
- Every call is fresh. Same-shape and nested elimination belong to [compiler](../../../../glossary.md#term-compiler) [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form).

## Known limitations

- Expand remains unavailable until task 0017D1.
- Dynamic count compatibility is recorded but not solved; invalid bindings must fail in compiler
  inference later.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding); [compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference))</span>
- Non-[contiguous](../../../../glossary.md#term-contiguous) reshape result geometry is unresolved and no executable [materialization](../../../../glossary.md#materialization) exists.
- No [host](../../../../glossary.md#term-host)-storage [alias](../../../../glossary.md#aliasing-and-arena) is attached even when resolved [layout](../../../../glossary.md#layout) describes a logical [view](../../../../glossary.md#term-view-meanings).
- No gradient rule, compiler capture, planning requirement, backend lowering, runtime execution,
  ONNX mapping, or conformance behavior is implemented.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [capture glossary](../../../../glossary.md#term-capture-meanings); [planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) read the [architecture contract](../../../../glossary.md#architecture-contract) and focused [lifecycle](../../../../glossary.md#lifecycle)/[module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency)/[runtime](../../../../glossary.md#term-runtime)-boundary
explanations; documentation and planning rules; roadmap; [model](../../../../glossary.md#term-model) capabilities/master plan; tasks
0002, 0003, 0007, 0011, 0012, 0013, 0017A, 0017B, and 0017C; current [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension),
LayoutDescriptor/LayoutKind, TensorDescriptor/[Tensor](../../../../glossary.md#tensor)/factory/[provenance](../../../../glossary.md#provenance), [Operation](../../../../glossary.md#operation), and
ShapeTransform semantic source/tests; Tensor/[Compile](../../../../glossary.md#compile)/Training APIs and glossary; and Java 26
[Gradle](../../../../glossary.md#term-gradle) configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms public varargs
reshape, one `-1`, count [inference](../../../../glossary.md#term-inference) and validation, logical order/type preservation, [contiguous](../../../../glossary.md#term-contiguous)
[view](../../../../glossary.md#term-view-meanings) behavior, non-contiguous [materialization](../../../../glossary.md#materialization), [gradient](../../../../glossary.md#term-gradient) reconstruction, and [backend](../../../../glossary.md#backend) execution.
Legacy int limits, positive-only/[rank](../../../../glossary.md#term-rank)-at-least-one restrictions, immediate storage, [graph](../../../../glossary.md#graph) builders,
operation traits, planners, [kernels](../../../../glossary.md#kernel), [lowering](../../../../glossary.md#lowering), and runtime behavior are excluded or deliberately
adapted to completed new-model contracts.

Planning found reshape and expand expression construction too broad for one task and split the
former row into 0017D reshape and 0017D1 expand. Existing public contracts support both planned
reshape overloads, local count validation, conditional view [layout](../../../../glossary.md#layout), [immutable](../../../../glossary.md#term-immutable) provenance, and
central [identity](../../../../glossary.md#term-identity-meanings) allocation without a new type, dependency, foundational edit, or architecture
decision.

Planning validation after synchronizing this task, the model master plan, and roadmap:

- `git diff --check` passed.
- The targeted trailing-whitespace scan returned no matches across the three changed planning
  files.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- All 196 local Markdown file links across the three planning files resolve.
- Markdown code-fence counts are balanced: twelve in this task, two in the master plan, and zero
  in the roadmap.
- All 20 canonical task-specification headings are present, together with focused Capability
  origin, Required contract, and Javadoc requirements sections.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- At the planning stage, task, model master plan, and roadmap consistently identified 0017D as
  Ready. Task 0017D1 remained Draft, and no task-0017D1 specification existed.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Package review confirms no new package. Public entries and the helper remain in `model.tensor`,
  and completed Shape/layout/operation values are consumed through their existing packages.
- Scope review confirms exactly ten permitted implementation paths and exactly three planning
  paths in the current diff. No Java, API, glossary, architecture, Gradle, AGENTS, completed-task,
  or other-module file changed during planning.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Granularity review split the former combined reshape/expand expression row into 0017D and
  0017D1. Reshape inference/count/view rules and expand broadcast/zero-stride rules now have
  independent implementation and validation sessions.
- Contract review confirms the planned raw `long...` and exact Shape overloads reuse the completed
  long-dimension, scalar, zero-extent, dynamic, known-count, resolved-layout, descriptor,
  provenance, and identity contracts without a new abstraction.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [extent glossary](../../../../glossary.md#term-extent); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- The model-task sequence was renumbered after inserting 0017D1 and contains no duplicate order
  number within that table.

Implementation and independent documentation validation:

- Implementation context `/root/implement_model_0017d` added the two public overloads, the exact
  six-method package-private helper, and focused tests. Canonical clean documentation context
  `/root/implement_model_0017d/review_model_0017d_docs` independently inspected the actual diff,
  source, tests, generated Javadoc, bytecode, imports, build configuration, and documentation. It
  applied General and API/Javadoc style to Java and API work, Planning style to status/evidence,
  and Example format to the new reshape example.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Independent source and test inspection found no implementation defect. Raw requests are scanned
  before inference, caller arrays are never retained or mutated, empty varargs produce scalar
  Shape, one `-1` uses a known input count and non-zero checked product, inferred zero is valid,
  zero-product ambiguity is rejected, known mismatches fail, and dynamic equality is deferred.
  The exact Shape overload retains its target reference. All six data types and valid gradient
  choices retain exact metadata.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Resolved contiguous input plus static target creates one new view-marked layout with canonical
  target strides and the same input element offset. Offset zero and non-zero classify as<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
  `DENSE_CONTIGUOUS` and `DENSE_WITH_OFFSET`; unresolved, strided, broadcast, or dynamic geometry
  remains unresolved. Construction records exact `RESHAPE`, `TargetShapeAttrs(targetShape)`, and
  `[input]` provenance and attaches no label or storage.
- The documentation pass finalized Tensor API reshape behavior, the Compile API current-versus-
  planned boundary, glossary status and terminology, and synchronized planning. It corrected the
  Tensor type overview to link `ShapeTransformKind`; the submitted public overload and helper
  Javadocs were otherwise complete and required no behavioral-source change.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorReshapeExpressionTest --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; 28 tests across two
  suites, specifically 14 reshape tests and 14 Tensor tests, with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 62 suites and 484 tests, with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated `Tensor.html` contains both
  public overloads, their parameters/results/four failure categories, raw ownership and inference,
  dynamic deferral, conditional view geometry, storage/materialization boundaries, and the<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
  `ShapeTransformKind` type-overview link. Package-private helper Javadocs were reviewed in source.
- `./gradlew test` — `BUILD SUCCESSFUL`; all 36 repository test tasks completed without failure.
  No architecture, backend-conformance, or integration test changed because the task adds only
  model expression metadata and no dependency, backend, or end-to-end behavior.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test))</span>
- `javap -p -c -s` confirms each public overload contains one matching helper call. The final
  stateless helper has one private constructor and exactly six methods; bytecode shows one raw
  validation/inference path, one exact-Shape validation path, one layout decision, and one shared
  descriptor/attributes/Operation/provenance/`createDerived` construction path.
- Import and source scans found only the permitted model/JDK contracts. There is no storage/value
  access, allocation, copy, gradient rule, graph/compiler/planning/prepare/runtime/backend type,
  cache, registry, service, implicit contiguous operation, or hidden canonicalization.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare); [cache glossary](../../../../glossary.md#term-cache); [registry glossary](../../../../glossary.md#term-registry); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- The Tensor API reshape example compiled with Java 26 against built model classes and printed the
  documented normalized Shape <span data-glossary-navigation="true"> ([normalized Shape glossary](../../../../glossary.md#normalized-shape))</span>`[3, 2]`, [canonical](../../../../glossary.md#term-canonical-form) [strides](../../../../glossary.md#term-stride) `[2, 1]`, offset five, view flag,
  `RESHAPE` kind, input identity, exact dynamic-Shape retention, unresolved dynamic layout, and
  absent storage.
- The targeted Markdown validator resolved 298 local file links and heading anchors across the six
  changed documentation/planning files. Backtick fences are balanced (132 Tensor API, four Compile
  API, twelve task, two master-plan markers; zero in glossary and roadmap), no tilde fence was
  introduced, trailing-whitespace scans found no matches, every changed file ends with a newline,
  and `git diff --check` plus untracked-file whitespace checks passed.
- Exact [scope](../../../../glossary.md#term-scope) is the authorized ten paths: `Tensor.java`, `TensorReshapeExpressions.java`,
  `TensorTest.java`, `TensorReshapeExpressionTest.java`, Tensor API, Compile API, glossary, this
  task, model master plan, and roadmap. Task 0017D1 remains Draft, and no detailed 0017D1 task
  specification exists. No commit or push occurred.<span data-glossary-navigation="true"> ([commit glossary](../../../../glossary.md#term-commit))</span>
- Shape and Dimension Javadocs remain accurate because they already define scalar/zero/dynamic
  Shapes, checked known counts, defensive ownership, and exclusion of numeric dynamic sentinels.
  LayoutDescriptor/LayoutKind remain accurate because they already define canonical and offset
  contiguous geometry, independent view metadata, and no storage/materialization policy.
  TensorDescriptor, TensorFactory, TensorProvenance, Operation, ShapeTransformKind,
  TargetShapeAttrs, and Contiguous contracts remain accurate because reshape composes their
  existing immutable descriptor, centralized identity, origin, semantic-pairing, normalized-Shape,
  and materialization boundaries without changing them.
- Training API remains unchanged because no gradient, autograd, parameter, optimizer, publication,
  or session behavior was added. <span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication))</span>`capabilities.md` already records reshape and the layer split.
  `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance and integration material, Java 26 Gradle configuration, dependencies, other
  modules, and unrelated tests remain accurate because ownership, dependency direction, lifecycle,
  build structure, backend behavior, and executable behavior did not change.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>

## Implementation notes

- Added exactly `Tensor.reshape(long...)` and `Tensor.reshape(Shape)` as one-call delegations to a
  package-private final stateless helper with the six required methods.
- Added focused coverage for API shape, all data types and valid gradient choices, scalar/zero/
  static/dynamic Shapes, every inference and validation edge, view and unresolved geometry,
  provenance/freshness, storage non-interference, identity consumption, and exhaustion.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [scalar glossary](../../../../glossary.md#term-scalar); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [view glossary](../../../../glossary.md#term-view-meanings); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Finalized the Tensor overview and reshape Javadocs, Tensor/Compile APIs, glossary, task evidence,
  model master plan, and roadmap in the mandatory independent documentation context.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>
- Added no expand behavior, value movement, storage attachment, gradient rule, compiler/planning/
  prepare/runtime/backend behavior, dependency, build change, or architecture change.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Completion summary

- Completed changes: Implemented and documented raw-inferred and exact-Shape public reshape
  expressions with local count validation, conditional same-offset canonical view geometry, and
  exact one-input provenance.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation); [view glossary](../../../../glossary.md#term-view-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
- Files changed or created: Exactly two production files, two focused test files, Tensor API,
  Compile API, glossary, this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused tests passed 28/28; all 484 model tests across 62 suites, model
  Javadoc, root tests, Java 26 example, bytecode/reflection/import/source/generated-Javadoc,
  link/anchor/fence/terminology/whitespace, exact-scope/status, and diff checks passed.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: [Canonical](../../../../glossary.md#term-canonical-form) clean context
  `/root/implement_model_0017d/review_model_0017d_docs` completed the independent pass using
  General, API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and glossary now present reshape construction as current;
  Compile API preserves the boundary between current expression metadata and planned capture,
  dynamic constraints, canonicalization, materialization, lowering, and execution.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: [Tensor](../../../../glossary.md#tensor) and helper reshape contracts are complete; the Tensor overview now links
  `ShapeTransformKind`. All reviewed foundational, layout, descriptor, factory, provenance,
  operation, shape-transform, and contiguous contracts remain accurate unchanged.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation); [contiguous glossary](../../../../glossary.md#term-contiguous))</span>
- Glossary impact: Reshape semantics and Tensor status now cover raw inference, exact Shape
  retention, conditional view metadata, unresolved cases, freshness, and cross-layer boundaries.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference))</span>
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0017D. Task 0017D1 remains Draft without a detailed
  specification.

Status: Complete
