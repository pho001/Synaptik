# Task 0017C: [Reshape and Expand Semantics](../../../../glossary.md#reshape-and-expand-semantics)

## Status

Complete

## Goal

Define typed, [backend](../../../../glossary.md#backend)-independent semantic [identities](../../../../glossary.md#term-identity-meanings) and [immutable](../../../../glossary.md#term-immutable) [target](../../../../glossary.md#term-target)-[shape](../../../../glossary.md#shape) attributes for
`RESHAPE` and `EXPAND`.

Both operations have one logical input and one exact target `Shape`, but they mean different
things. `RESHAPE` preserves the logical [element](../../../../glossary.md#term-element) sequence while changing its [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) shape.
`EXPAND` preserves existing values while logically repeating compatible [singleton](../../../../glossary.md#term-singleton) [dimensions](../../../../glossary.md#dimension) or
adding leading dimensions. This task defines those meanings only; public [Tensor](../../../../glossary.md#tensor) requests,
compatibility [validation](../../../../glossary.md#term-validation), [layout](../../../../glossary.md#layout) derivation, [provenance](../../../../glossary.md#provenance), [gradients](../../../../glossary.md#term-gradient), [materialization](../../../../glossary.md#materialization), and execution
remain outside its [scope](../../../../glossary.md#term-scope).

## [Scope](../../../../glossary.md#term-scope)

- Add one public `ShapeTransformKind` enum implementing [`OperationKind`](../../../../glossary.md#operationkind).
- Define exactly `RESHAPE` and `EXPAND`, in that order.
- Add one public `TargetShapeAttrs` record implementing [`OperationAttrs`](../../../../glossary.md#operationattrs) with exactly one
  `Shape targetShape` component.
- Reject null target Shape with exact parameter-name failure and retain every non-null immutable
  Shape reference unchanged.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Accept [scalar](../../../../glossary.md#term-scalar), zero-[extent](../../../../glossary.md#term-extent), ordinary static, and explicit dynamic [target](../../../../glossary.md#term-target) [Shapes](../../../../glossary.md#shape) structurally.
- Document that stored target Shape is normalized model semantics and never a raw public
  reshape-request array containing a numeric <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>`-1` [inference](../../../../glossary.md#term-inference) sentinel.
- Pair both kinds explicitly with `TargetShapeAttrs` without changing generic [`Operation`](../../../../glossary.md#operation)
  validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Document one logical input, result-shape meaning, logical element-order behavior, and the
  semantic difference between shape reinterpretation and broadcast repetition.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
- Add one focused same-package semantic-contract test for both cohesive types.
- Keep both types in the existing `model.operation.layout` package.
- Finalize Javadocs, Tensor API semantic reference, glossary, task evidence, model master plan, and
  roadmap through the required independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public `Tensor.reshape`, `Tensor.expand`, another Tensor method, overload, static facade, factory,
  builder, expression helper, or task-0017D implementation<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>
- raw requested-[shape](../../../../glossary.md#shape) arrays, varargs, a public numeric `-1` sentinel, inferred-dimension
  calculation, element-count comparison, divisibility, overflow policy, or request normalization<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension); [element glossary](../../../../glossary.md#term-element); [overflow glossary](../../../../glossary.md#term-overflow); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- input [Tensor](../../../../glossary.md#tensor), input Shape, input/output [DataType](../../../../glossary.md#data-type--datatype), `requiresGrad`, descriptor construction, label,
  identity, provenance, host storage, or <span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance); [host storage glossary](../../../../glossary.md#host-storage))</span>`TensorFactory.createDerived`
- reshape view eligibility, input contiguity, effective strides, storage offset, view flag,
  broadcast zero strides, result <span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [contiguity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [strides glossary](../../../../glossary.md#term-stride); [storage offset glossary](../../../../glossary.md#term-storage-offset))</span>`LayoutDescriptor`, [aliasing](../../../../glossary.md#term-aliasing), copying, or [materialization](../../../../glossary.md#materialization) policy
- right-aligned expand compatibility validation, target-rank validation, singleton-axis checks,
  symbolic constraint solving, dynamic-dimension binding, or graph-wide inference<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [target glossary](../../../../glossary.md#term-target); [rank glossary](../../../../glossary.md#term-rank); [singleton glossary](../../../../glossary.md#term-singleton); [axis glossary](../../../../glossary.md#term-axis); [binding glossary](../../../../glossary.md#term-binding); [graph glossary](../../../../glossary.md#graph); [inference glossary](../../../../glossary.md#term-inference))</span>
- contiguous, permute, transpose, expand-dimensions, squeeze, slice, pad, tile, concat, stack,
  unstack, unfold, fold, select, gather, or scatter semantics<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [transpose glossary](../../../../glossary.md#term-transpose); [slice glossary](../../../../glossary.md#slice); [gather glossary](../../../../glossary.md#gather))</span>
- gradients, reshape backward, broadcast reduction, autograd, compiler-generated operations,
  optimizer, or training behavior<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [compiler glossary](../../../../glossary.md#term-compiler); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- operation factories, registries, parsers, visitors, aliases, string dispatch, reflection
  discovery, maps, services, arity fields, result-kind fields, cost, fusion, capability, backend
  support, lowering, route, kernel, executable, or physical-buffer metadata<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [dispatch glossary](../../../../glossary.md#term-dispatch); [arity glossary](../../../../glossary.md#term-arity); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [buffer glossary](../../../../glossary.md#term-buffer); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- compiler capture/canonicalization, planning ownership/materialization, prepare, runtime residency,
  backend storage/execution, engine, trace, ONNX mapping, or conformance<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- changes to `ContiguousKind`, Operation foundations, Shape/Dimension, LayoutDescriptor/LayoutKind,
  Tensor contracts, graph records, existing Java tests, dependencies, Gradle, architecture, or
  another module<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- a detailed task-0017D specification or any later task implementation

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
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0017A](0017a-contiguous-semantic-kind.md)
- [Task 0017B](0017b-contiguous-tensor-expression.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes:

```java
Tensor reshape(int... requestedShape)
Tensor expand(int... requestedShape)
```

Legacy reshape accepts one `-1` request [dimension](../../../../glossary.md#dimension), infers it from the input [element](../../../../glossary.md#term-element) count, rejects
multiple sentinels or incompatible counts, preserves [row-major](../../../../glossary.md#term-row-major) logical element order, and chooses
between a [view](../../../../glossary.md#term-view-meanings) and a materialized result according to input [layout](../../../../glossary.md#layout). Legacy expand accepts a [target](../../../../glossary.md#term-target)
[rank](../../../../glossary.md#term-rank) at least as large as the source rank, aligns source [axes](../../../../glossary.md#term-axis) from the right, retains equal axes,
repeats [singleton](../../../../glossary.md#term-singleton) axes, permits new leading axes, and represents repetition through zero [strides](../../../../glossary.md#term-stride).

The new [model](../../../../glossary.md#term-model) deliberately supports [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [scalar](../../../../glossary.md#term-scalar) [Shapes](../../../../glossary.md#shape), zero extents, long static dimensions,
and explicit dynamic dimensions. Therefore legacy positive-int restrictions and empty-shape
rejection are not copied into the [immutable](../../../../glossary.md#term-immutable) semantic attributes. Task 0017D will specify the new
public request forms, local [inference](../../../../glossary.md#term-inference)/compatibility rules, result [descriptor](../../../../glossary.md#term-descriptor), and [provenance](../../../../glossary.md#provenance).

Legacy [mutable](../../../../glossary.md#term-mutable) arrays, immediate layout/storage [views](../../../../glossary.md#view), [graph](../../../../glossary.md#graph) builders, [gradient](../../../../glossary.md#term-gradient) callbacks,
[operation](../../../../glossary.md#operation) traits, cost/[fusion](../../../../glossary.md#term-fusion)/result tags, [materialization](../../../../glossary.md#materialization) selection, [kernels](../../../../glossary.md#kernel), [lowering](../../../../glossary.md#lowering), fallback,
and [runtime](../../../../glossary.md#term-runtime) state are not copied. They are either implementation details or responsibilities of
later [compiler](../../../../glossary.md#term-compiler), [planning](../../../../glossary.md#term-planning), [prepare](../../../../glossary.md#prepare), runtime, training, and [backend](../../../../glossary.md#backend) tasks.

## Architecture constraints

- Production remains in `modules/model`, which owns backend-independent operation semantics and
  immutable Shape values.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- `ShapeTransformKind` identifies semantic meaning only. It is not a Tensor, graph occurrence,
  result descriptor, layout, materialization requirement, executable operation, or backend route.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [occurrence glossary](../../../../glossary.md#term-occurrence); [descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization); [backend route glossary](../../../../glossary.md#backend-route))</span>
- `RESHAPE` means a one-input coordinate-shape change that preserves the ordered logical element
  sequence and therefore requires compatible element counts after local or graph-wide binding.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [element glossary](../../../../glossary.md#term-element); [binding glossary](../../../../glossary.md#term-binding))</span>
- `EXPAND` means a one-input logical broadcast to a target Shape through compatible singleton or
  leading dimensions; repeated logical positions may later use zero-stride view geometry.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [singleton glossary](../../../../glossary.md#term-singleton); [dimensions glossary](../../../../glossary.md#dimension); [stride glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- `TargetShapeAttrs` stores the exact semantic result Shape shared by both kinds. It does not store
  the input Shape, raw request syntax, inferred-axis index, input/output strides, layout, or
  compatibility proof.<span data-glossary-navigation="true"> ([axis index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- A stored [`Shape`](../../../../glossary.md#shape) has already passed its own model invariants. Static dimensions are non-negative,
  dynamic dimensions use explicit symbols, scalar Shape has rank zero, and zero extents are valid.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [scalar glossary](../../../../glossary.md#term-scalar); [rank glossary](../../../../glossary.md#term-rank))</span>
- Numeric `-1` is public reshape request syntax only. It cannot appear inside `Shape` and is never
  stored by these attributes. Task 0017D owns request normalization before Operation construction.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>
- The attributes constructor validates only non-nullity. It cannot compare element counts, decide
  reshape view eligibility, validate expand compatibility, bind symbols, or derive layout.
- Generic [`Operation`](../../../../glossary.md#operation) remains an open kind/attributes pair and does not enforce family
  compatibility, arity, target-shape rules, descriptor inference, gradients, or backend support.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [inference glossary](../../../../glossary.md#term-inference); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Package direction is `model.operation.layout -> model.operation + model.shape`. It must not
  depend on tensor, layout-value, storage, graph, compiler, planning, prepare, runtime, backend,
  engine, trace, or training packages.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace))</span>
- Stop if implementation needs input state, arrays, another attributes type, result inference,
  layout/materialization behavior, dependency, or architecture change.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation.layout` — owns cohesive layout/view operation
  semantics and receives both new public contracts.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- `io.github.pho001.synaptik.model.operation` — supplies [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), and
  generic immutable [`Operation`](../../../../glossary.md#operation) composition.
- `io.github.pho001.synaptik.model.shape` — supplies the [immutable](../../../../glossary.md#term-immutable) [`Shape`](../../../../glossary.md#shape) [target](../../../../glossary.md#term-target) value.

No package is added or moved.

Type placement:

- `io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind` — public enum for the two
  target-shape-based semantic identities.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>
- `io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs` — public immutable exact
  target-Shape parameter shared by both kinds.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>
- `ShapeTransformSemanticsTest` — same-package focused test for vocabulary, record invariants,
  typed composition, and dependency boundaries.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

## Required contract

### Semantic kind vocabulary

Create exactly:

```java
public enum ShapeTransformKind implements OperationKind {
    RESHAPE,
    EXPAND
}
```

The enum declares no project field, explicit constructor, method, nested type, per-[constant](../../../../glossary.md#term-constant) class
body, [alias](../../../../glossary.md#aliasing-and-arena), [symbol](../../../../glossary.md#term-symbol), or [metadata](../../../../glossary.md#term-metadata). [Compiler](../../../../glossary.md#term-compiler)-generated enum machinery is not additional project [API](../../../../glossary.md#term-application-programming-interface-api).
Inherited `Enum.name()` satisfies `OperationKind.name()` and returns the exact constant text.

| Kind | One-input semantic meaning | Not decided here |
|---|---|---|
| `RESHAPE` | preserve the ordered logical [element](../../../../glossary.md#term-element) sequence while interpreting it through `targetShape` [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) | count compatibility, [view](../../../../glossary.md#term-view-meanings)/copy choice, [layout](../../../../glossary.md#layout), [gradients](../../../../glossary.md#term-gradient), execution |
| `EXPAND` | logically repeat compatible [singleton](../../../../glossary.md#term-singleton) or new leading [dimensions](../../../../glossary.md#dimension) to `targetShape` | broadcast [validation](../../../../glossary.md#term-validation), zero [strides](../../../../glossary.md#term-stride), [aliasing](../../../../glossary.md#term-aliasing), gradients, execution |

Neither kind stores [arity](../../../../glossary.md#term-arity), [target](../../../../glossary.md#term-target) [Shape](../../../../glossary.md#shape), input facts, output facts, layout, or [backend](../../../../glossary.md#backend) metadata.

### [Target](../../../../glossary.md#term-target)-[shape](../../../../glossary.md#shape) attributes

Create exactly:

```java
public record TargetShapeAttrs(Shape targetShape) implements OperationAttrs
```

The record has exactly one component, one public [canonical](../../../../glossary.md#term-canonical-form) constructor, one explicit documented
`targetShape()` accessor, and record-generated `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), and `toString`. Add no
overload, factory, builder, array accessor, inferred-[axis](../../../../glossary.md#term-axis) field, input Shape, [layout](../../../../glossary.md#layout), [cache](../../../../glossary.md#term-cache), nested
type, or helper [API](../../../../glossary.md#term-application-programming-interface-api).

The canonical constructor performs exactly:

```java
targetShape = Objects.requireNonNull(targetShape, "targetShape");
```

It retains the exact [immutable](../../../../glossary.md#term-immutable) Shape reference. Null fails with `NullPointerException` and exact
message `targetShape`. [Scalar](../../../../glossary.md#term-scalar), zero-[extent](../../../../glossary.md#term-extent), ordinary static, mixed static/dynamic, and fully dynamic
Shapes are accepted structurally without [element](../../../../glossary.md#term-element)-count or broadcast [validation](../../../../glossary.md#term-validation).

Record equality and hashing use structural Shape equality. Generated text is diagnostic only and
is not [serialization](../../../../glossary.md#term-serialization), public request syntax, parser input, [ONNX](../../../../glossary.md#term-onnx) mapping, [backend](../../../../glossary.md#backend) [dispatch](../../../../glossary.md#term-dispatch), or a
layout plan.

### Typed composition

Document these exact valid pairings:

```java
new Operation(ShapeTransformKind.RESHAPE, attrs)
new Operation(ShapeTransformKind.EXPAND, attrs)
```

Both operations retain the exact `TargetShapeAttrs` reference. Do not use
`NoOperationAttrs.INSTANCE`, add a family factory, expose an enum `operation()` method, or change
generic [`Operation`](../../../../glossary.md#operation) to validate pairings. A [target](../../../../glossary.md#term-target) [Shape](../../../../glossary.md#shape) is intrinsic semantic data for both
operations even when it equals the eventual input Shape.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/ShapeTransformKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/TargetShapeAttrs.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/layout/ShapeTransformSemanticsTest.java`

Documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `docs/api/compile-api.md`
- `docs/api/training-api.md`
- `docs/planning/modules/model/capabilities.md`
- `ContiguousKind`, Operation foundations, Shape/Dimension, LayoutDescriptor/LayoutKind, Tensor
  contracts and expressions, graph contracts, and their Javadocs/tests<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph))</span>
- focused architecture documents, ADRs, architecture tests, backend-conformance material,
  integration tests, Gradle configuration, and every other module<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, one focused test, and five documentation/[planning](../../../../glossary.md#term-planning) files: eight paths
total.

Do not modify existing Java source/tests, capabilities, [Compile](../../../../glossary.md#compile)/Training [API](../../../../glossary.md#term-application-programming-interface-api), [Gradle](../../../../glossary.md#term-gradle), AGENTS,
architecture documents/tests, another [module](../../../../glossary.md#term-module), completed tasks, or unrelated documentation. Stop if
another production concept, input [validation](../../../../glossary.md#term-validation), [Tensor](../../../../glossary.md#tensor) expression, [layout](../../../../glossary.md#layout) rule, [dependency](../../../../glossary.md#term-dependency), or ninth
path is needed. Do not create task 0017D.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document `ShapeTransformKind` as typed one-input target-shape transformation semantics rather
  than Tensor behavior, resolved layout, graph occurrence, or executable support.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [layout glossary](../../../../glossary.md#layout); [graph glossary](../../../../glossary.md#graph); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Document `RESHAPE` as ordered logical [element](../../../../glossary.md#term-element)-sequence preservation under new [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) and
  `EXPAND` as logical [singleton](../../../../glossary.md#term-singleton)/leading-[axis](../../../../glossary.md#term-axis) repetition.
- Explain that [arity](../../../../glossary.md#term-arity) and [target](../../../../glossary.md#term-target)-[shape](../../../../glossary.md#shape) compatibility are family context, not enum [metadata](../../../../glossary.md#term-metadata).
- Document `TargetShapeAttrs` construction, exact reference retention, nullability, accepted Shape
  categories, record value semantics, diagnostic text, and explicit accessor.<span data-glossary-navigation="true"> ([nullability glossary](../../../../glossary.md#term-nullability))</span>
- Explain why stored `Shape` is normalized [model](../../../../glossary.md#term-model) semantics and cannot contain a raw numeric `-1`
  inference sentinel.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference))</span>
- Document exact [Operation](../../../../glossary.md#operation) pairings and absence of generic compatibility [validation](../../../../glossary.md#term-validation).
- Defer input/target validation, count inference, broadcasting, layout/view/copy decisions,
  provenance, gradients, compiler behavior, materialization, backend support, and execution.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [view glossary](../../../../glossary.md#term-view-meanings); [provenance glossary](../../../../glossary.md#provenance); [gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [materialization glossary](../../../../glossary.md#materialization); [backend glossary](../../../../glossary.md#backend))</span>
- Review ContiguousKind, Shape, Dimension, Operation/OperationAttrs/OperationKind, LayoutDescriptor,
  LayoutKind, TensorDescriptor, and Tensor Javadocs and record why they remain accurate, or stop on
  an inconsistency.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [OperationAttrs glossary](../../../../glossary.md#operationattrs); [OperationKind glossary](../../../../glossary.md#operationkind))</span>

## Acceptance criteria

- Exactly one public `ShapeTransformKind` enum and one public `TargetShapeAttrs` record are added in
  the existing operation-layout package.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- The enum implements [`OperationKind`](../../../../glossary.md#operationkind) and declares exactly `RESHAPE`, then `EXPAND`.
- The enum adds no project field, method, constructor, nested type, constant body, alias, arity,
  target, layout, cost, fusion, backend, inference, materialization, or execution metadata.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [alias glossary](../../../../glossary.md#aliasing-and-arena); [arity glossary](../../../../glossary.md#term-arity); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [inference glossary](../../../../glossary.md#term-inference); [materialization glossary](../../../../glossary.md#materialization); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- The record implements [`OperationAttrs`](../../../../glossary.md#operationattrs) and has exactly one `Shape targetShape` component, one
  canonical constructor, one explicit accessor, and no additional state or API.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Null [target](../../../../glossary.md#term-target) [Shape](../../../../glossary.md#shape) fails with exact `NullPointerException` message `targetShape`; every non-null
  Shape is retained by exact reference.
- Scalar, zero-extent, static, and dynamic target Shapes are accepted without input-dependent
  validation.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [extent glossary](../../../../glossary.md#term-extent); [validation glossary](../../../../glossary.md#term-validation))</span>
- Record equality/hash/text remain ordinary value/diagnostic behavior; the accessor returns the
  exact stored Shape reference.<span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash))</span>
- Both kinds compose with exact `TargetShapeAttrs`; neither composes with no-attributes in the
  documented family contract, and no compatibility validator is added.
- Production imports only [Operation](../../../../glossary.md#operation) contracts, Shape, and `Objects`. No Tensor, layout-value,
  storage, graph, compiler, planning, prepare, runtime, backend, engine, trace, training,
  dependency, or executable behavior is added.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- No public [Tensor](../../../../glossary.md#tensor) method, raw request array, `-1` inference, element-count/broadcast validation,
  result descriptor, layout geometry, provenance, gradient, or execution is implemented.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Focused/aggregate tests, model Javadoc, root tests, reflection/javap/import/source/scope checks,
  documentation links/formatting, and synchronized statuses pass.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- A separate clean-context documentation agent finalizes both Javadocs, Tensor API, glossary,
  task evidence, master plan, and roadmap and records related no-change conclusions.
- Task 0017C becomes Complete only after both passes. Task 0017D remains Draft without a detailed
  specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.layout.ShapeTransformSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover exact package/visibility/interfaces; enum [constant](../../../../glossary.md#term-constant) names and order;
absence of project enum state, behavior, nested types, [aliases](../../../../glossary.md#aliasing-and-arena), and constant bodies; record status,
component order/type, constructor/accessor surface, absence of extra fields/methods/nested types;
null failure; exact reference retention for [scalar](../../../../glossary.md#term-scalar)/static/zero/dynamic [Shapes](../../../../glossary.md#shape); structural record
equality; typed distinction from a private test-local equal-name kind; and explicit [Operation](../../../../glossary.md#operation)
composition retaining exact kind and attributes references.

Manually inspect `javap -p -c -s`, reflection, imports, and [Gradle](../../../../glossary.md#term-gradle) dependencies. Confirm no [Tensor](../../../../glossary.md#tensor),
[layout](../../../../glossary.md#layout)-value, storage, [provenance](../../../../glossary.md#provenance), [graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler)/[planning](../../../../glossary.md#term-planning)/[prepare](../../../../glossary.md#prepare)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) type, raw shape
array, sentinel, count/broadcast [inference](../../../../glossary.md#term-inference), [gradient](../../../../glossary.md#term-gradient), cost, [fusion](../../../../glossary.md#term-fusion), route, [registry](../../../../glossary.md#term-registry), map, or service
appears. Validate generated [Javadoc](../../../../glossary.md#term-javadoc), Tensor [API](../../../../glossary.md#term-application-programming-interface-api) semantic status, glossary terms, links, anchors,
fences, whitespace, exact eight-path [scope](../../../../glossary.md#term-scope), synchronized statuses, and absence of a task-0017D
specification.

## Dependencies

- Task 0002 supplies [immutable](../../../../glossary.md#term-immutable) static/dynamic/[scalar](../../../../glossary.md#term-scalar)/zero-[extent](../../../../glossary.md#term-extent) [`Shape`](../../../../glossary.md#shape) values.
- Task 0005 supplies [`OperationKind`](../../../../glossary.md#operationkind) and [`OperationAttrs`](../../../../glossary.md#operationattrs).
- Task 0006 supplies immutable generic [`Operation`](../../../../glossary.md#operation) composition and reference retention.
- Completed tasks 0017A–0017B establish the operation-layout package and semantic/expression split;
  they are implementation-order context rather than Java dependencies of these two types.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>

## Follow-up tasks

- 0017D remains Draft for exact public reshape/expand request forms, `-1` inference, local
  element-count and right-aligned expansion validation, result descriptor/layout construction,
  provenance, and storage-free derived Tensor creation.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [element glossary](../../../../glossary.md#term-element); [validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [Tensor glossary](../../../../glossary.md#tensor))</span>
- 0017E–0017F remain Draft for permutation, transpose, expand-dimensions, and squeeze semantics and
  expressions.<span data-glossary-navigation="true"> ([permutation glossary](../../../../glossary.md#permutation); [transpose glossary](../../../../glossary.md#term-transpose); [dimensions glossary](../../../../glossary.md#dimension))</span>
- [Compiler](../../../../glossary.md#term-compiler) later owns [graph](../../../../glossary.md#graph)-wide symbolic constraints and [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form).
- [Planning](../../../../glossary.md#term-planning) later derives logical [materialization](../../../../glossary.md#materialization) requirements without selecting a concrete route.
- Backend prepare later chooses view, alias, copy, or backend-native lowering; runtime executes the
  prepared schedule.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare); [view glossary](../../../../glossary.md#term-view-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime); [prepared schedule glossary](../../../../glossary.md#prepared-schedule--preparedschedule))</span>
- Training and compiler-generated semantic tasks later own differentiation and backward forms.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns [backend](../../../../glossary.md#backend)-independent [operation](../../../../glossary.md#operation) semantics,
[Shape](../../../../glossary.md#shape), and logical [layout](../../../../glossary.md#layout) facts to `modules/model`; [compiler](../../../../glossary.md#term-compiler) [inference](../../../../glossary.md#term-inference) to compiler; [materialization](../../../../glossary.md#materialization)
requirements to [planning](../../../../glossary.md#term-planning); concrete [lowering](../../../../glossary.md#lowering)/storage to backend [prepare](../../../../glossary.md#prepare); and [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution) to
[runtime](../../../../glossary.md#term-runtime).

If implementation requires [Tensor](../../../../glossary.md#tensor) behavior, result inference, layout/materialization policy,
backend [metadata](../../../../glossary.md#term-metadata), another [dependency](../../../../glossary.md#term-dependency), or architecture change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0003/0005/0006/0013/0017A/0017B/0017C, Tensor API,
Compile API, Training API, glossary, current Shape/Dimension/OperationKind/OperationAttrs/
Operation/ContiguousKind contracts and focused tests, and Java 26 Gradle configuration.

Implement task 0017C exactly. Add only ShapeTransformKind.java, TargetShapeAttrs.java, and
ShapeTransformSemanticsTest.java under io.github.pho001.synaptik.model.operation.layout for Java
code and tests.

The public enum contains exactly RESHAPE and EXPAND in that order and adds no project state,
methods, aliases, arity, target, layout, or metadata. The public record implements OperationAttrs
with exactly non-null Shape targetShape, exact NullPointerException message targetShape, explicit
documented accessor, exact reference retention, and no other API/state. Accept every current
Shape category. Compose each kind explicitly with TargetShapeAttrs.

Document RESHAPE as preserving ordered logical elements under target coordinates and EXPAND as
logical singleton/leading-axis repetition. The stored Shape is normalized semantics, never raw
public request syntax or a numeric -1 sentinel. Do not implement Tensor methods, raw arrays,
inference, element-count/broadcast validation, descriptors, layouts, provenance, gradients,
compiler/planning/prepare/runtime/backend behavior, dependencies, build/architecture changes,
existing Java edits, or later specs. Stop beyond eight paths or on architecture uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff and evidence to a
separate clean-context documentation agent in the same change. It must inspect source/tests/
generated Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record related-
contract/capability/Compile API/Training API/architecture no-change conclusions, and rerun
validation.

Update task 0017C, model master plan, and roadmap only for planning status/evidence. Do not mark
0017C Complete until both passes succeed. Leave 0017D Draft without a specification. Do not commit
or push.
```

## Local decisions

- `ShapeTransformKind` groups exactly the two transformations whose intrinsic parameter is a
  complete target Shape. Contiguous has no parameter and remains in <span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [target glossary](../../../../glossary.md#term-target); [Contiguous glossary](../../../../glossary.md#term-contiguous))</span>`ContiguousKind`; later axis,
  slice, pad/tile, composition, and window families need different typed attributes.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [slice glossary](../../../../glossary.md#slice))</span>
- `TargetShapeAttrs` is shared because both operations store the same semantic fact. Separate
  reshape/expand records would duplicate an identical immutable contract without adding meaning.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- The record stores [`Shape`](../../../../glossary.md#shape), not `long[]`, `int[]`, or `List<Long>`, so semantic attributes reuse
  canonical static/dynamic/scalar/zero-extent invariants and require no defensive array copy.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [scalar glossary](../../../../glossary.md#term-scalar); [extent glossary](../../../../glossary.md#term-extent))</span>
- Every non-null Shape is accepted structurally. Input-dependent count and broadcast validation
  belongs to task 0017D, which has both input and target context.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Numeric `-1` belongs only to the future public reshape request boundary. It is normalized before
  `TargetShapeAttrs` construction and never becomes [model](../../../../glossary.md#term-model) Shape state.
- RESHAPE and EXPAND remain distinct kinds despite sharing attributes because one reinterprets an
  ordered element sequence while the other introduces logical repetition.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
- No alias/view/materialization flag is stored. Logical geometry and executable realization are
  derived later by their owning layers.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [view glossary](../../../../glossary.md#term-view-meanings); [materialization glossary](../../../../glossary.md#materialization))</span>

## Known limitations

- No public reshape or expand expression exists until task 0017D.
- The attributes cannot prove element-count compatibility, singleton expansion compatibility,
  target rank, symbolic constraints, view eligibility, or layout geometry.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [singleton glossary](../../../../glossary.md#term-singleton); [target glossary](../../../../glossary.md#term-target); [rank glossary](../../../../glossary.md#term-rank); [view glossary](../../../../glossary.md#term-view-meanings); [layout glossary](../../../../glossary.md#layout))</span>
- Raw public `-1` [inference](../../../../glossary.md#term-inference) syntax and exact failure messages remain unspecified until task 0017D.
- No provenance, gradient rule, compiler capture, materialization plan, backend lowering, runtime
  execution, ONNX mapping, or conformance behavior is implied.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [materialization glossary](../../../../glossary.md#materialization); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) read the [architecture contract](../../../../glossary.md#architecture-contract) and focused [lifecycle](../../../../glossary.md#lifecycle)/[module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency)/[runtime](../../../../glossary.md#term-runtime)-boundary
explanations; documentation and planning rules; roadmap; [model](../../../../glossary.md#term-model) capabilities and master plan;
tasks 0002, 0003, 0005, 0006, 0013, 0017A, and 0017B; current [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension), [Operation](../../../../glossary.md#operation), [layout](../../../../glossary.md#layout)
semantic/value, [Tensor](../../../../glossary.md#tensor), and [descriptor](../../../../glossary.md#term-descriptor) contracts/tests; Tensor/[Compile](../../../../glossary.md#compile)/Training APIs and glossary;
and Java 26 [Gradle](../../../../glossary.md#term-gradle) configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms public reshape and
expand capabilities, a single raw reshape `-1` request sentinel, [element](../../../../glossary.md#term-element)-count [inference](../../../../glossary.md#term-inference),
right-aligned [singleton](../../../../glossary.md#term-singleton)/leading-[axis](../../../../glossary.md#term-axis) expansion, [immutable](../../../../glossary.md#term-immutable) operation [target](../../../../glossary.md#term-target)-shape meaning, [view](../../../../glossary.md#term-view-meanings) or
[materialization](../../../../glossary.md#materialization) decisions, [gradient](../../../../glossary.md#term-gradient) construction, and [CPU](../../../../glossary.md#term-cpu)/[accelerator](../../../../glossary.md#term-accelerator) execution evidence. Legacy
arrays, positive-int restrictions, [scalar](../../../../glossary.md#term-scalar)/zero rejection, [graph](../../../../glossary.md#graph) builders, storage [views](../../../../glossary.md#view), operation
traits, planners, [kernels](../../../../glossary.md#kernel), [lowering](../../../../glossary.md#lowering), and runtime code are excluded or assigned to later [owners](../../../../glossary.md#term-owner).

Planning selected one two-value enum, one shared exact-Shape attributes record, and one focused
test. Existing Shape invariants provide normalized semantic target values, while generic Operation
already supplies typed composition. No Tensor behavior, result inference, layout contract,
dependency, foundational change, or architecture decision is required.

Planning validation after synchronizing this task, the model master plan, and roadmap:

- `git diff --check` passed.
- The targeted trailing-whitespace scan returned no matches across the three changed planning
  files.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- All 190 local Markdown file links across the three planning files resolve.
- Markdown code-fence counts are balanced: fourteen in this task, two in the master plan, and zero
  in the roadmap.
- All 20 canonical task-specification headings are present, together with focused Capability
  origin, Required contract, and Javadoc requirements sections.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Task, model master plan, and roadmap consistently identify 0017C as Ready. Task 0017D remains
  Draft, and no task-0017D specification exists.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Package review confirms both planned public types remain in the existing
  `model.operation.layout` package and introduce only the allowed dependency on foundational
  `model.shape` plus Operation contracts.
- Scope review confirms exactly eight permitted implementation paths and exactly three planning
  paths in the current diff. No Java, API, glossary, architecture, Gradle, AGENTS, completed-task,
  or other-module file changed during planning.<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope))</span>
- Contract review confirms `Shape` already excludes negative static dimensions, represents scalar
  and zero extents explicitly, and models dynamic dimensions through symbols. Therefore the stored
  target Shape cannot contain a raw numeric `-1` sentinel and needs no duplicate validation.
- Dependency review confirms 0002 and 0005–0006 are the real Java prerequisites. Tasks 0017A–0017B
  provide package and sequencing context without creating false source dependencies.

Implementation and independent documentation validation:

- Implementation context `/root/implement_model_0017c` added exactly
  `ShapeTransformKind.java`, `TargetShapeAttrs.java`, and `ShapeTransformSemanticsTest.java`.
  Clean documentation context
  `/root/implement_model_0017c/review_model_0017c_docs` independently read the required
  architecture, documentation and planning profiles, plans and historical tasks, APIs, glossary,
  Java 26 build configuration, final source/test, generated Javadoc, XML reports, and complete
  workspace diff. It applied General and API/Javadoc style to Java, Tensor API, and glossary;
  Planning style to this task, the model master plan, and roadmap; and Example format to the
  conceptual reshape/expand comparison.
- Independent source and generated-Javadoc review found both submitted production Javadocs
  complete unchanged. They document one-input family context, exact RESHAPE and EXPAND meaning,
  explicit `TargetShapeAttrs` pairings, accepted Shape categories, exact reference retention,
  null failure, normalized semantic Shape versus raw `-1` request syntax, record value semantics,
  diagnostic text, and the required inference/layout/provenance/gradient/compiler/planning/
  backend/execution boundaries.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.layout.ShapeTransformSemanticsTest` —
  `BUILD SUCCESSFUL`; XML records 8 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 61 XML suites record 470 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated pages contain the enum type,
  both constants, record type/component, canonical constructor, explicit accessor, exception,
  ownership/value semantics, exact pairings, normalized-Shape boundary, and cross-layer
  exclusions.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `./gradlew test` — `BUILD SUCCESSFUL`; all repository test tasks completed without failure.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` confirms the enum has exactly
  `RESHAPE`, then `EXPAND`, plus compiler-generated enum machinery and no project field or method.
  It confirms the record has exactly one private final `Shape targetShape` field, a canonical
  constructor whose only semantic validation is
  `Objects.requireNonNull(targetShape, "targetShape")`, a direct accessor, and record-generated
  equality, hashing, and diagnostic text. The eight focused reflection tests independently cover
  the same surface and exact Operation composition.
- Production imports are exactly [`OperationKind`](../../../../glossary.md#operationkind) for the enum and [`OperationAttrs`](../../../../glossary.md#operationattrs), `Shape`, and
  `Objects` for the record. Source and bytecode inspection found no Tensor, raw array, inferred
  axis, count/broadcast validation, descriptor, layout-value, provenance, gradient, graph,
  compiler, planning, prepare, runtime, backend, engine, trace, training, registry, map, service,
  or executable state or behavior.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace); [registry glossary](../../../../glossary.md#term-registry))</span>
- Tensor [API](../../../../glossary.md#term-application-programming-interface-api) now presents `ShapeTransformKind` and `TargetShapeAttrs` as current semantic
  contracts, explains normalized target-Shape ownership and the RESHAPE-versus-EXPAND distinction,
  and keeps public Tensor construction and cross-layer behavior planned. The glossary synchronizes
  implementation status, the OperationKind/OperationAttrs distinction, and reusable reshape/
  expand terminology.
- A targeted Markdown check validated 285 local links and heading anchors across the five changed
  documentation/planning files. Backtick fences are balanced, no tilde fence is present, every
  changed file ends with a newline, targeted trailing-whitespace scans return no matches, and
  `git diff --check` passes.
- Exact scope is the eight authorized paths: two production files, one focused test, Tensor API,
  glossary, this task, model master plan, and roadmap. Task 0017C, the master-plan row/current
  status/notes, and roadmap frontier/table are synchronized as Complete. Task 0017D remains Draft,
  and no task-0017D specification exists. No commit or push was performed.<span data-glossary-navigation="true"> ([commit glossary](../../../../glossary.md#term-commit))</span>
- `Operation`, `OperationAttrs`, and `OperationKind` remain accurate unchanged because the new
  values conform to their existing open typed-pairing contracts without changing compatibility
  validation. `Shape` and `Dimension` remain accurate because the record only retains an already
  valid immutable Shape. `ContiguousKind` remains accurate because its parameterless canonical-
  geometry request is distinct from target-shape transformations. `LayoutDescriptor`,
  `LayoutKind`, `TensorDescriptor`, and `Tensor` remain accurate because no resolved geometry,
  result descriptor, public method, provenance, or storage behavior was added; their focused tests
  therefore require no change.
- `capabilities.md` already inventories reshape/expand capability and correctly separates model
  semantics, public Tensor construction, compiler, planning, backend prepare, runtime, and tests.
  Compile API remains accurate because no public expression, capture, inference, canonicalization,
  artifact, or materialization behavior was added. Training API remains accurate because no
  gradient, autograd, optimizer, parameter, publication, or session behavior changed.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [artifact glossary](../../../../glossary.md#term-artifact); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer); [parameter glossary](../../../../glossary.md#term-parameter); [publication glossary](../../../../glossary.md#term-publication))</span>
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests, backend-
  conformance material, and integration tests remain accurate unchanged because module ownership,
  dependency direction, lifecycle, backend behavior, and end-to-end behavior did not change.
  Root/model Java 26 Gradle configuration, dependencies, other modules, and unrelated tests also
  remain unchanged because the task adds only model-owned semantic vocabulary.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>

## Implementation notes

- Added the exact two-value `ShapeTransformKind` enum and one-component `TargetShapeAttrs` record in
  the existing operation-layout package, with no additional project API or state.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [layout glossary](../../../../glossary.md#layout); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Added the focused eight-test suite for vocabulary/order, exact enum and record reflection shape,
  every current Shape category, null failure, reference retention, record value semantics, typed
  identity, exact Operation composition, and cross-layer state exclusion.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Finalized Tensor API, glossary, task evidence, model master plan, and roadmap in the mandatory
  independent documentation context. The submitted production Javadocs required no correction.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [model glossary](../../../../glossary.md#term-model))</span>
- Added no Tensor expression, request array, inference, compatibility validation, descriptor,
  layout, provenance, gradient, compiler, planning, backend, execution, dependency, build, or
  architecture behavior.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Completion summary

- Completed changes: Implemented and documented exact RESHAPE and EXPAND semantic identities with
  one shared immutable normalized target-Shape attributes value.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [target glossary](../../../../glossary.md#term-target); [Shape glossary](../../../../glossary.md#shape))</span>
- Files changed or created: Exactly two production Java files, one focused test, Tensor API,
  glossary, this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused tests passed 8/8; all 470 model tests across 61 suites, generated
  model Javadoc, root tests, javap/reflection/source/import/generated-documentation checks, 285
  Markdown link/anchor checks, fence/terminology/whitespace checks, exact scope/status checks, and<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0017c/review_model_0017c_docs` completed the independent pass using
  General, API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and glossary now present target-shape semantics as current while
  keeping public request normalization, compatibility validation, expressions, descriptors,
  provenance, and cross-layer behavior planned.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [descriptors glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- Javadoc review: Both new production Javadocs are complete unchanged; related operation, Shape,
  contiguous, resolved-layout, Tensor descriptor/expression, and cross-layer contracts remain
  accurate for the reasons recorded above.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [contiguous glossary](../../../../glossary.md#term-contiguous); [layout glossary](../../../../glossary.md#layout); [Tensor descriptor glossary](../../../../glossary.md#tensor-descriptor))</span>
- Glossary impact: Implementation status, target-shape attributes, and the reusable reshape versus
  expand distinction now reflect the implemented contract.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0017C. Task 0017D remains Draft without a detailed
  specification.

Status: Complete
