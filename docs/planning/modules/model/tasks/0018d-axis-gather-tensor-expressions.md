# Task 0018D: [Axis](../../../../glossary.md#term-axis) [Gather](../../../../glossary.md#gather) [Tensor](../../../../glossary.md#tensor) Expressions

**Editorial reading key (later context, not a status update).**

Axis gather consumes ordered data and integer-index inputs; index positions select source coordinates, not Boolean eligibility. The application is the caller of the data Tensor method and supplies an indices Tensor plus axis. The helper inspects descriptors to derive the selected shape rule but never reads index values or checks their numerical bounds. The old gather/gatherAxis/take spellings below remain historical evidence. The application calls the public Tensor method; its model helper validates available descriptions and returns a fresh storage-free expression with exact input references. It neither copies input values nor calculates the result. Compiler consumes this provenance later; actual storage and execution are separate downstream concerns.

Focused current reading: [Current gather/gatherElements taxonomy](../../../../reference/tensor/gather.md#axis-gather-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Add public [model](../../../../glossary.md#term-model)-level [Tensor](../../../../glossary.md#tensor) expressions for the three completed [axis](../../../../glossary.md#term-axis)-[gather](../../../../glossary.md#gather) meanings:
[shape](../../../../glossary.md#shape)-reducing `gather`, [ONNX](../../../../glossary.md#term-onnx)-style `gatherAxis` with its `take` [alias](../../../../glossary.md#aliasing-and-arena), and shape-aligned
`takeAlongAxis`.

Every expression consumes ordered logical inputs `[data, indices]`, requires an [`INT32`](../../../../glossary.md#data-type--datatype) or [`INT64`](../../../../glossary.md#data-type--datatype)
indices Tensor, normalizes one data axis, derives the [operation](../../../../glossary.md#operation)-specific result Shape, preserves
the data Tensor's type and [gradient](../../../../glossary.md#term-gradient) eligibility, leaves [layout](../../../../glossary.md#layout) unresolved, and records exact
two-input [provenance](../../../../glossary.md#provenance). This task constructs [metadata](../../../../glossary.md#term-metadata) only; it never reads index values or data.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly these public instance methods to [`Tensor`](../../../../glossary.md#tensor):
  - `Tensor gather(Tensor indices, int axis)`
  - `Tensor gatherAxis(Tensor indices, int axis)`
  - `Tensor take(int axis, Tensor indices)`
  - `Tensor takeAlongAxis(Tensor indices, int axis)`
- Add one field-free package-private final `TensorAxisGatherExpressions` helper in `model.tensor`.
- Give the helper exactly four package-private entries and five private methods specified below.
- Null-check data and indices in deterministic order and accept only exact index data types
  [`INT32`](../../../../glossary.md#data-type--datatype) and [`INT64`](../../../../glossary.md#data-type--datatype).
- Normalize each raw positive or negative [axis](../../../../glossary.md#term-axis) exactly once through the data [Shape](../../../../glossary.md#shape).
- For [`GATHER`](../../../../glossary.md#gather), require indices Shape equal to data Shape with the selected axis removed and use
  that derived reduced Shape as the result.
- For `GATHER_AXIS`, insert every exact indices Dimension at the selected data-axis position while
  removing that selected data Dimension.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension))</span>
- Make tensor-index `take` delegate exactly to the helper's `gatherAxis` entry and therefore create
  exact `GATHER_AXIS` semantics rather than a separate [operation](../../../../glossary.md#operation).
- For `TAKE_ALONG_AXIS`, require equal ranks and equal non-axis Dimensions, then retain the exact
  indices Shape as the result Shape.
- Preserve exact [data type](../../../../glossary.md#data-type--datatype) and `requiresGrad`; always use unresolved result [layout](../../../../glossary.md#layout).
- Create exact `AxisGatherKind`, `IndexAxisAttrs`, ordered `[data, indices]` provenance, no label or
  storage, and one final <span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>`TensorFactory.createDerived` call per valid request.
- Keep every valid request fresh, including repeated and nested expressions.
- Update `TensorTest` only for the deliberate four-method public API expansion and add one focused
  expression test.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Finalize Javadocs, Tensor API, Compile API, glossary, task evidence, master plan, and roadmap
  through the mandatory independent documentation pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- Apply the explicitly authorized [Javadoc](../../../../glossary.md#term-javadoc)-only timing and bounds-[ownership](../../../../glossary.md#term-ownership-meanings) correction to
  `AxisGatherKind` and `IndexAxisAttrs`; declarations, validation, semantics, and tests remain
  unchanged.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- `take(int, int[])`, `take(int, long[])`, another primitive-array or collection convenience,
  eager index-Tensor construction, or task-0018D1 implementation<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [Tensor glossary](../../../../glossary.md#tensor))</span>
- gather-ND, scalar select, scatter, masks, slices, ranges, index conversion, or another operation
  family<span data-glossary-navigation="true"> ([gather-ND glossary](../../../../glossary.md#gather-nd); [scalar select glossary](../../../../glossary.md#scalar-select); [masks glossary](../../../../glossary.md#term-boolean-mask); [operation glossary](../../../../glossary.md#operation))</span>
- accepting floating or BOOL index tensors, implicitly casting indices, inspecting whether index
  values are integral, or converting <span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype))</span>[`INT64`](../../../../glossary.md#data-type--datatype) to [`INT32`](../../../../glossary.md#data-type--datatype)
- reading, normalizing, clamping, or bounds-checking actual index values; negative index-value
  policy remains an execution/input-value concern outside metadata construction<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- broadcasting gather inputs, accepting mismatched dynamic symbols as equal, inventing symbolic
  constraints, or binding dynamic extents<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [gather glossary](../../../../glossary.md#gather); [binding glossary](../../../../glossary.md#term-binding))</span>
- deriving or preserving resolved layout, view/alias geometry, storage offsets, materialization,
  host storage, device state, or backend routes<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena); [materialization glossary](../../../../glossary.md#materialization); [host storage glossary](../../../../glossary.md#host-storage); [device glossary](../../../../glossary.md#term-device); [backend glossary](../../../../glossary.md#backend))</span>
- modifying `AxisGatherKind` or `IndexAxisAttrs` declarations or behavior, or modifying Shape,
  Dimension, DataType, TensorDescriptor, TensorFactory, TensorProvenance, Operation, or any
  completed Java contract/test other than the exact Tensor API inventory; the two semantic files
  have one explicit Javadoc-only exception for stale task timing and index-bounds ownership<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [DataType glossary](../../../../glossary.md#data-type--datatype); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- gradient rules, repeated-index accumulation, scatter backward, graph capture, canonicalization,
  compiler passes, planning, prepare, runtime, backend, engine, trace, ONNX implementation,
  training, or execution behavior<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- another production helper/type, dependency, Gradle/build option, architecture change, another
  module, or task-0018D1 specification<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

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
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0018A](0018a-scalar-select-semantics.md)
- [Task 0018B](0018b-scalar-select-tensor-expression.md)
- [Task 0018C](0018c-axis-gather-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only legacy branch exposes:

```java
Tensor gather(Tensor indices, int dimension)
Tensor gatherAxis(Tensor indices, int axis)
Tensor take(int axis, Tensor indices)
Tensor take(int axis, int[] indices)
Tensor takeAlongAxis(Tensor indices, int dimension)
```

Legacy [`gather`](../../../../glossary.md#gather) removes the selected data [axis](../../../../glossary.md#term-axis) and requires one index for every [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) of that
reduced [Shape](../../../../glossary.md#shape). Legacy `gatherAxis` uses [ONNX](../../../../glossary.md#term-onnx)-style Shape insertion, and [tensor](../../../../glossary.md#tensor)-index `take`
delegates to it. Legacy `takeAlongAxis` requires indices [rank](../../../../glossary.md#term-rank) equal to data rank and identical
non-axis extents, then returns the indices Shape.

This task preserves the four public methods that consume an existing index Tensor. The primitive
`int[]` convenience additionally owns [eager](../../../../glossary.md#term-eager-evaluation) copied index-Tensor creation and therefore remains the
separate task 0018D1. The selected new baseline tightens legacy index typing to exact [`INT32`](../../../../glossary.md#data-type--datatype) or
[`INT64`](../../../../glossary.md#data-type--datatype); inconsistent acceptance of other numeric data types is not preserved.

Legacy [graph](../../../../glossary.md#graph) builders, direct data access, [gradient](../../../../glossary.md#term-gradient) callbacks, scatter implementations, traits,
[compiler](../../../../glossary.md#term-compiler)/[lowering](../../../../glossary.md#lowering) code, [kernels](../../../../glossary.md#kernel), and [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) behavior are capability evidence only and are
not copied.

## Architecture constraints

- [`Tensor`](../../../../glossary.md#tensor) remains public mutable API state, not IR. Each method creates a fresh storage-free
  expression through the existing package-private derived-factory seam.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Semantic [identities](../../../../glossary.md#term-identity-meanings) and [normalized axis](../../../../glossary.md#normalized-axis) attributes come only from completed task 0018C.
- The helper may inspect immutable data/index descriptors and Shapes, but never values, host
  storage, runtime residency, device state, or backend capability.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [descriptors glossary](../../../../glossary.md#term-descriptor); [Shapes glossary](../../../../glossary.md#shape); [host storage glossary](../../../../glossary.md#host-storage); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [device glossary](../../../../glossary.md#term-device); [backend capability glossary](../../../../glossary.md#backend-capability))</span>
- Index type validation is a model API contract because the index representation is explicit
  metadata. Index value bounds cannot be checked without reading values and remain deferred.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Shape compatibility is structural and conservative: equal static sizes or equal dynamic symbols
  pass through existing Dimension/Shape equality; no new symbolic constraint is invented.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension))</span>
- Gather results are value-reordered/materialized semantics rather than logical storage views, so
  every result layout is unresolved even when all input geometry is resolved.<span data-glossary-navigation="true"> ([views glossary](../../../../glossary.md#view))</span><span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [layout glossary](../../../../glossary.md#layout))</span>
- [Data type](../../../../glossary.md#data-type--datatype) and [gradient](../../../../glossary.md#term-gradient) eligibility come only from `data`; indices never contribute gradient
  eligibility. Current TensorDescriptor already prevents gradients for integral/BOOL data.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Compiler owns graph capture and canonicalization; compiler-generated/training tasks own gather
  backward/scatter; planning/prepare/backend own materialization, lowering, and execution.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering))</span>
- No [dependency](../../../../glossary.md#term-dependency), package [ownership](../../../../glossary.md#term-ownership-meanings), or [module](../../../../glossary.md#term-module) boundary changes are authorized.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.tensor.Tensor` — receives four [Tensor](../../../../glossary.md#tensor)-index expression methods.
- `io.github.pho001.synaptik.model.tensor.TensorAxisGatherExpressions` — owns local type/axis/Shape
  validation and semantic construction.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [Shape glossary](../../../../glossary.md#shape))</span>
- `TensorAxisGatherExpressionTest` — mirrors `model.tensor` for focused helper/[API](../../../../glossary.md#term-application-programming-interface-api) [validation](../../../../glossary.md#term-validation).
- `TensorTest` — changes only its exact public API inventory and reflection assertions.

`AxisGatherKind` and `IndexAxisAttrs` remain in `model.operation.index`; no semantic contract moves
into the Tensor package.

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) methods

Add exactly:

```java
public Tensor gather(Tensor indices, int axis) {
    return TensorAxisGatherExpressions.gather(this, indices, axis);
}

public Tensor gatherAxis(Tensor indices, int axis) {
    return TensorAxisGatherExpressions.gatherAxis(this, indices, axis);
}

public Tensor take(int axis, Tensor indices) {
    return TensorAxisGatherExpressions.take(this, axis, indices);
}

public Tensor takeAlongAxis(Tensor indices, int axis) {
    return TensorAxisGatherExpressions.takeAlongAxis(this, indices, axis);
}
```

Each public method contains one return statement and exactly one matching helper call. The methods
are non-static and non-synchronized, perform no direct [validation](../../../../glossary.md#term-validation) or field access, and add no
overloads in this task.

### Helper [shape](../../../../glossary.md#shape)

Create one package-private final, field-free class with one private zero-[argument](../../../../glossary.md#term-argument) constructor and
exactly these nine static methods:

```java
static Tensor gather(Tensor data, Tensor indices, int axis)
static Tensor gatherAxis(Tensor data, Tensor indices, int axis)
static Tensor take(Tensor data, int axis, Tensor indices)
static Tensor takeAlongAxis(Tensor data, Tensor indices, int axis)
private static void validateIndexType(String operation, TensorDescriptor indicesDescriptor)
private static Shape removeAxis(Shape dataShape, int normalizedAxis)
private static Shape gatherAxisShape(
        Shape dataShape, Shape indicesShape, int normalizedAxis)
private static void validateTakeAlongAxis(
        Shape dataShape, Shape indicesShape, int normalizedAxis)
private static Tensor create(
        Tensor data,
        Tensor indices,
        TensorDescriptor dataDescriptor,
        Shape resultShape,
        AxisGatherKind kind,
        IndexAxisAttrs attrs)
```

Add no field, nested type, alternate constructor, overload, [cache](../../../../glossary.md#term-cache), [mutable](../../../../glossary.md#term-mutable) state, or extra method.

### Shared [validation](../../../../glossary.md#term-validation) order

The [`gather`](../../../../glossary.md#gather), `gatherAxis`, and `takeAlongAxis` entries each perform this common prefix in exact
order:

1. null-check `data` with message `data`;
2. null-check `indices` with message `indices`;
3. read the exact data [descriptor](../../../../glossary.md#term-descriptor) once;
4. read the exact indices descriptor once;
5. call `validateIndexType` once with [operation](../../../../glossary.md#operation) name `gather`, `gatherAxis`, or
   `takeAlongAxis`;
6. read the exact data [Shape](../../../../glossary.md#shape) once;
7. normalize the raw [axis](../../../../glossary.md#term-axis) exactly once through `dataShape.normalizeAxis(axis)`;
8. construct one `IndexAxisAttrs(normalizedAxis)` after operation-specific Shape validation;
9. call `create` once.

Null and index-type failures occur before axis [normalization](../../../../glossary.md#term-normalization). All type/axis/Shape validation and
[metadata](../../../../glossary.md#term-metadata) construction occur before identifier allocation, so local failures consume no [Tensor](../../../../glossary.md#tensor) ID.

`validateIndexType` accepts exactly `DataType.INT32` and `DataType.INT64`. Every other type throws
`IllegalArgumentException` with exact message:

```text
<operation> indices data type must be INT32 or INT64: <actual>
```

No method inspects index values or storage.

### [GATHER](../../../../glossary.md#gather) [Shape](../../../../glossary.md#shape) rule

After the shared prefix, `gather` calls `removeAxis(dataShape, normalizedAxis)` once. The helper
creates one [rank](../../../../glossary.md#term-rank)-minus-one Shape by preserving every unaffected exact data [Dimension](../../../../glossary.md#dimension) reference.
[Rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) data produces [canonical](../../../../glossary.md#term-canonical-form) [scalar](../../../../glossary.md#term-scalar) Shape.

Require `indicesDescriptor.shape().equals(resultShape)`. Otherwise throw
`IllegalArgumentException` with exact message:

```text
gather indices shape must equal data shape without gathered axis: expected=<expected>, actual=<actual>
```

Use the derived reduced data Shape as the result, not the indices Shape reference. Then construct
`AxisGatherKind.GATHER`.

Examples:

- data `[2, 3, 4]`, [axis](../../../../glossary.md#term-axis) `1`, indices `[2, 4]` produces `[2, 4]`;
- data `[N, 3]`, axis `1`, indices `[N]` passes only when the same symbolic Dimension is used;
- rank-one data `[5]`, axis `0`, scalar indices produces scalar result.

### GATHER_AXIS and take [Shape](../../../../glossary.md#shape) rule

After the shared prefix, `gatherAxis` calls `gatherAxisShape` once. The result [Dimension](../../../../glossary.md#dimension) order is:

1. exact data Dimensions before the selected [axis](../../../../glossary.md#term-axis);
2. every exact indices Dimension in order;
3. exact data Dimensions after the selected axis.

The selected data Dimension is omitted. No other compatibility check is required. Examples:

- data `[2, 3, 4]`, axis `1`, indices `[5, 6]` produces `[2, 5, 6, 4]`;
- [scalar](../../../../glossary.md#term-scalar) indices replace the selected axis with no Dimensions, so data `[2, 3, 4]`, axis `1`
  produces `[2, 4]`;
- data `[2, N]`, axis `0`, indices `[K]` produces `[K, N]` with exact `K` and `N` references.

Construct `AxisGatherKind.GATHER_AXIS`.

The helper's `take` method contains exactly:

```java
return gatherAxis(data, indices, axis);
```

It performs no independent [validation](../../../../glossary.md#term-validation) or construction. Therefore [tensor](../../../../glossary.md#tensor)-index take shares exact
errors, Shape, kind, attributes, [provenance](../../../../glossary.md#provenance), [layout](../../../../glossary.md#layout), and identifier side effects with gatherAxis.

### TAKE_ALONG_AXIS [Shape](../../../../glossary.md#shape) rule

After the shared prefix, read the exact indices Shape once and call `validateTakeAlongAxis` once.

- If ranks differ, throw `IllegalArgumentException` with exact message
  `takeAlongAxis indices rank must match data rank: expected=<dataRank>, actual=<indicesRank>`.
- Compare every non-selected [axis](../../../../glossary.md#term-axis) in increasing order with `Dimension.equals`.
- On the first mismatch, throw `IllegalArgumentException` with exact message
  `takeAlongAxis indices dimension at axis <axis> must match data: expected=<expected>, actual=<actual>`.
- Do not compare the selected-axis extents.

Use the exact indices Shape reference as result Shape and construct
`AxisGatherKind.TAKE_ALONG_AXIS`.

Examples:

- data `[2, 3, 4]`, indices `[2, 7, 4]`, axis `1` produces exact indices Shape `[2, 7, 4]`;
- data `[N, 3]`, indices `[N, K]`, axis `1` passes with the same `N` reference/meaning;
- data `[N, 3]`, indices `[M, 2]`, axis `1` fails when `N` and `M` differ.

### Result [descriptor](../../../../glossary.md#term-descriptor), semantics, [provenance](../../../../glossary.md#provenance), and [identity](../../../../glossary.md#term-identity-meanings)

`create` constructs exactly:

```java
TensorDescriptor descriptor = new TensorDescriptor(
        dataDescriptor.dataType(),
        resultShape,
        Optional.empty(),
        dataDescriptor.requiresGrad());
Operation operation = new Operation(kind, attrs);
TensorProvenance provenance = new TensorProvenance(operation, List.of(data, indices));
return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
```

Every current [data type](../../../../glossary.md#data-type--datatype) is accepted for data. Exact data type and `requiresGrad` are retained;
indices never affect either. Every result [layout](../../../../glossary.md#layout) is unresolved. Every valid call returns a fresh
[Tensor](../../../../glossary.md#tensor) with no label or storage, exact ordered `[data, indices]` provenance, exact kind, and exact
normalized attributes.

## Affected files

Expected implementation change:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorAxisGatherExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/AxisGatherKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/IndexAxisAttrs.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorAxisGatherExpressionTest.java`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the twelve paths listed above. The two semantic-contract
paths are an explicitly user-authorized documentation-review exception and may change only
[Javadoc](../../../../glossary.md#term-javadoc) wording; their declarations and executable behavior must remain unchanged.

If implementation requires primitive take convenience, another production type/helper, semantic
or foundational declaration/behavior change, another test, capability-baseline edit, architecture
document, [dependency](../../../../glossary.md#term-dependency), build change, another [module](../../../../glossary.md#term-module), or more than twelve paths, stop and propose a
follow-up task.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document all four public Tensor methods and every helper method, including the private
  constructor.
- Define ordered `[data, indices]` roles and exact [`INT32`](../../../../glossary.md#data-type--datatype)/[`INT64`](../../../../glossary.md#data-type--datatype) index requirement.
- Explain each [Shape](../../../../glossary.md#shape) rule with the concrete examples above, including dynamic [Dimension](../../../../glossary.md#dimension) behavior.
- Explain `take` as an exact [alias](../../../../glossary.md#aliasing-and-arena) for [tensor](../../../../glossary.md#tensor)-index `gatherAxis` and task 0018D1 as the deferred
  primitive convenience.
- Explain exact type/eligibility retention, unresolved layout, provenance, fresh identity, and
  absence of label/storage/value execution.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- State that no index values or bounds are inspected and no [gradient](../../../../glossary.md#term-gradient) rule is defined.
- Distinguish [scalar](../../../../glossary.md#term-scalar) `SELECT`, [gather-ND](../../../../glossary.md#gather-nd), and scatter.
- Do not promise [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings), [backend](../../../../glossary.md#backend) support, [materialization](../../../../glossary.md#materialization), [lowering](../../../../glossary.md#lowering), or execution.

## Acceptance criteria

- Tensor adds exactly the four specified public, non-static, non-synchronized methods and no
  primitive overload.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Every public method delegates once to its exact helper entry and performs no other work.
- The helper is package-private, final, field-free, has one private constructor, and exactly nine
  specified static methods.
- Null, index-type, axis, and family-specific Shape validation follows the exact order, exception
  types, and messages.
- [`GATHER`](../../../../glossary.md#gather) derives and retains the reduced data [Shape](../../../../glossary.md#shape) and requires structurally equal indices Shape.
- `GATHER_AXIS` inserts exact indices [Dimensions](../../../../glossary.md#dimension); `take` delegates to it and records the same kind.
- `TAKE_ALONG_AXIS` requires equal [rank](../../../../glossary.md#term-rank)/non-[axis](../../../../glossary.md#term-axis) Dimensions and retains the exact indices Shape.
- Dynamic Dimensions are accepted only through existing structural equality; no constraints are
  invented.<span data-glossary-navigation="true"> ([structural equality glossary](../../../../glossary.md#term-structural-equality))</span>
- All results preserve exact data type/eligibility, use unresolved layout, exact normalized attrs,
  ordered <span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout))</span>`[data, indices]` [provenance](../../../../glossary.md#provenance), no label/storage, and fresh [identity](../../../../glossary.md#term-identity-meanings).
- No values/storage, primitive index creation, gradients, graph/compiler/runtime/backend behavior,
  dependencies, build, semantic declarations/behavior, or architecture are changed; the two
  authorized semantic Javadocs only correct current-versus-planned and bounds-ownership wording.<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Tensor API, Compile API, glossary, task evidence, master plan, and roadmap are independently
  reviewed and synchronized; Training API, capabilities, architecture, and related contracts
  receive reasoned no-change conclusions.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>
- All [validation](../../../../glossary.md#term-validation) passes and the final diff contains exactly the twelve permitted paths.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run):

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.tensor.TensorAxisGatherExpressionTest
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests must verify exact [API](../../../../glossary.md#term-application-programming-interface-api)/helper [shape](../../../../glossary.md#shape), null/type/[axis](../../../../glossary.md#term-axis) validation order and messages,
all six data types for data, both index types, every static/dynamic Shape rule and failure,
[rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)/[scalar](../../../../glossary.md#term-scalar) indices, take [alias](../../../../glossary.md#aliasing-and-arena) behavior, exact kinds/attrs/[provenance](../../../../glossary.md#provenance), unresolved [layout](../../../../glossary.md#layout),
absent label/storage, fresh [identities](../../../../glossary.md#term-identity-meanings), and lack of value/storage/cross-layer access.

Manual validation must inspect `javap -p -c -s`, one-call delegation [bytecode](../../../../glossary.md#term-bytecode), reflection, source,
imports, generated [Javadoc](../../../../glossary.md#term-javadoc), the executable documentation example when changed, Markdown links,
anchors, fences, whitespace, exact twelve-path [scope](../../../../glossary.md#term-scope), synchronized status, and absence of a
task-0018D1 specification.

## Dependencies

- Task 0001 provides exact [DataType](../../../../glossary.md#data-type--datatype) categories and [`INT32`](../../../../glossary.md#data-type--datatype)/[`INT64`](../../../../glossary.md#data-type--datatype) [identities](../../../../glossary.md#term-identity-meanings).
- Task 0002 provides [Shape](../../../../glossary.md#shape), [Dimension](../../../../glossary.md#dimension), dynamic-[symbol](../../../../glossary.md#term-symbol) equality, and [axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization).
- Tasks 0011–0013 provide [Tensor](../../../../glossary.md#tensor), derived [identity](../../../../glossary.md#term-identity-meanings) allocation, and [immutable](../../../../glossary.md#term-immutable) [provenance](../../../../glossary.md#provenance).
- Task 0018C provides exact axis-[gather](../../../../glossary.md#gather) kinds and normalized attributes.

## Follow-up tasks

- 0018D1: legacy primitive `take(int, int[])` convenience with copied [eager](../../../../glossary.md#term-eager-evaluation) [INT32](../../../../glossary.md#data-type--datatype) indices.
- 0018E: [gather-ND](../../../../glossary.md#gather-nd) semantics and [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[dimension](../../../../glossary.md#dimension) attributes.

Do not create either follow-up specification during this task.

## Architecture impact

Expected impact: None.

This task fills the existing [model](../../../../glossary.md#term-model)-owned public expression surface. If implementation requires a
new architecture rule, symbolic constraint system, value access, or cross-[module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), stop
and report the issue.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0007/0011/0012/0013/0018A/0018B/0018C/0018D,
Tensor API, Compile API, Training API, glossary, current DataType/Dimension/Shape/TensorDescriptor/
Tensor/TensorFactory/TensorProvenance/Operation/AxisGatherKind/IndexAxisAttrs contracts and focused
expression tests, and Java 26 Gradle configuration.

Implement task 0018D exactly. Modify Tensor.java and add package-private final
TensorAxisGatherExpressions.java. Update TensorTest only for the exact four-method API expansion
and add TensorAxisGatherExpressionTest. Add exactly gather(Tensor,int), gatherAxis(Tensor,int),
take(int,Tensor), and takeAlongAxis(Tensor,int).

The field-free helper has exactly nine specified methods. Follow exact null/index-type/axis/Shape
validation order and messages. GATHER requires indices Shape equal to data Shape without the axis.
GATHER_AXIS inserts the full indices Shape; take delegates exactly to it. TAKE_ALONG_AXIS requires
equal ranks/non-axis Dimensions and returns the exact indices Shape. Accept only INT32/INT64
indices. Preserve exact data type/eligibility, leave layout unresolved, create exact kind/attrs and
ordered [data, indices] provenance, and call createDerived once. Every request is fresh.

Do not add primitive take, gather-ND/scatter, value/bounds/storage access, resolved layout,
gradients, graph/compiler/planning/runtime/backend behavior, semantic/foundational declaration or
behavior changes, dependencies, build/architecture changes, or later specs. The documentation pass
may apply only the explicitly authorized stale timing/bounds Javadoc corrections to
`AxisGatherKind` and `IndexAxisAttrs`. Stop beyond twelve paths or on uncertainty.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs
agent in the same change. It must inspect source/tests/generated Javadoc, finalize permitted
Javadocs/Tensor API/Compile API/glossary/planning, record Training API/capability/architecture and
related-contract no-change conclusions, and rerun validation.

Update task 0018D, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0018D1 Draft without a specification. Do not commit/push.
```

## Local decisions

- One field-free helper owns all local validation and construction. The four public methods remain
  one-call delegations, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`take` delegates inside the helper to the exact `gatherAxis` entry so
  there is one GATHER_AXIS validation and construction path.
- [`GATHER`](../../../../glossary.md#gather) always derives a new reduced data Shape and compares the indices Shape structurally;
  even when equal, the result retains the derived Shape rather than the indices Shape reference.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- `GATHER_AXIS` retains exact unaffected data Dimensions and exact inserted indices Dimensions.
  Scalar indices insert no Dimensions and therefore remove the selected data axis.<span data-glossary-navigation="true"> ([Dimensions glossary](../../../../glossary.md#dimension); [Scalar glossary](../../../../glossary.md#term-scalar); [axis glossary](../../../../glossary.md#term-axis))</span>
- `TAKE_ALONG_AXIS` validates rank before increasing-order non-axis Dimension equality and retains
  the exact indices Shape reference after validation. The selected extents are deliberately not
  compared.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>
- Index type is input metadata and is validated locally as exact INT32 or INT64. Index values and
  their bounds require value access and remain outside expression construction.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype))</span>
- An explicit user authorization expanded the task from ten to twelve paths solely to correct
  stale timing and bounds-ownership Javadoc in <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`AxisGatherKind` and `IndexAxisAttrs`. Their
  declarations, bytecode behavior, and focused semantic tests remain unchanged.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>

## Known limitations

- No method reads index values, normalizes negative index values, or checks index-value bounds.
- Dynamic compatibility is structural only: equal symbols pass and different symbols fail; no
  symbolic equality constraint or runtime binding is created.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [binding glossary](../../../../glossary.md#term-binding))</span>
- Every result layout is unresolved. The expressions create no values, physical aliases, storage,
  gradient or repeated-index rule, graph/compiler behavior, materialization, backend lowering,
  ONNX mapping, or execution.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [gradient glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [materialization glossary](../../../../glossary.md#materialization); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- Primitive `take(int, int[])` remains the Draft task 0018D1 without a detailed specification.

## [Validation](../../../../glossary.md#term-validation) evidence

- Clean implementation context `/root/implement_model_0018d` added the exact public methods,
  helper, focused tests, and initial planning updates. Independent documentation context<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
  `/root/implement_model_0018d/review_model_0018d_docs` inspected the actual shared-tree diff,
  source, tests, generated Javadoc, bytecode, APIs, glossary, planning state, and build
  configuration before finalizing documentation in the same overall change.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- The documentation pass applied General plus API/Javadoc style to the four affected production
  Javadocs, Tensor API, Compile API, and glossary; Planning style to this task, the model master
  plan, and roadmap; and Example format to the executable axis-gather example.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model); [axis glossary](../../../../glossary.md#term-axis); [gather glossary](../../../../glossary.md#gather))</span>
- Reviewed architecture and process material included `AGENTS.md`, `ARCHITECTURE.md`, the current
  architecture index, overview, lifecycle, module boundaries, and dependency rules; documentation
  rules and General/API-Javadoc/Planning/Example profiles; planning guide and roadmap; model
  capabilities/master plan; tasks 0001, 0002, 0007, 0011, 0012, 0013, and 0018A–0018D; Tensor,
  Compile, and Training API references; glossary; Java 26 root/model Gradle configuration; final
  implementation/tests; and generated model Javadoc.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Related-contract review covered DataType, Dimension, Shape, TensorDescriptor, Tensor,
  TensorFactory, TensorProvenance, Operation, AxisGatherKind, IndexAxisAttrs, SelectKind,
  SelectAttrs, and adjacent select/gather contracts/tests. <span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [Dimension glossary](../../../../glossary.md#dimension); [Operation glossary](../../../../glossary.md#operation))</span>`AxisGatherKind` and `IndexAxisAttrs`
  contained stale “later task” and index-bounds ownership wording. Explicit user authorization
  expanded the task from ten to twelve paths solely for Javadoc corrections; declarations,
  constructor validation, record/enum structure, generated methods, and semantics remain
  unchanged.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorAxisGatherExpressionTest` — `BUILD SUCCESSFUL`; XML
  reports 10 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; XML reports 14 tests
  with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; XML aggregation reports 667 tests across
  78 suites with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL` after the final [Javadoc](../../../../glossary.md#term-javadoc) edits. Generated
  `Tensor.html`, `AxisGatherKind.html`, and `IndexAxisAttrs.html` contain the exact [signatures](../../../../glossary.md#term-signature),
  [`INT32`](../../../../glossary.md#data-type--datatype)/[`INT64`](../../../../glossary.md#data-type--datatype) requirement, three Shape rules, alias, current input-aware boundary, normalized
  axis, ordered provenance, and explicit no-value/no-bounds/no-gradient/no-compiler/no-backend/
  no-execution boundaries. The package-private helper Javadocs were reviewed in source.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [normalized axis glossary](../../../../glossary.md#normalized-axis); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- `./gradlew test` — `BUILD SUCCESSFUL`; all 36 root lifecycle tasks completed or were up-to-date
  with no failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` plus focused reflection tests
  confirmed exactly 101 declared public Tensor methods, four new non-static/non-synchronized
  one-call delegations, one final package-private field-free helper with one private constructor
  and exactly nine static methods, exact validation order, one-axis normalization per canonical
  entry, one-call <span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>`take` delegation, exact [Shape](../../../../glossary.md#shape) construction, and one final `createDerived` call.
- Source, imports, tests, and bytecode confirm exact INT32/INT64 validation messages; GATHER's new
  reduced Shape and equality check; GATHER_AXIS Dimension insertion; TAKE_ALONG_AXIS rank-first and
  increasing non-axis checks with exact indices Shape retention; exact data type/eligibility;
  unresolved layout; exact kind/attrs/<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [data type glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout))</span>`[data, indices]` provenance; no label/storage; and fresh
  repeated/nested identities. No value, index-bound, storage, gradient, compiler, planning,
  prepare, runtime, backend, engine, trace, training, or ONNX behavior was added.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- The documented `AxisGatherExpressionExample` compiled with
  `javac -cp modules/model/build/classes/java/main -d
  /tmp/synaptik-axis-gather-doc-example /tmp/AxisGatherExpressionExample.java` and ran with model
  classes. It printed exact Shapes `[2, 4]`, `[2, 5, 6, 4]`, `[2, 5, 6, 4]`, and `[2, 7, 4]`,
  `GATHER_AXIS`, `IndexAxisAttrs[axis=1]`, and four expected `true` [metadata](../../../../glossary.md#term-metadata)/[provenance](../../../../glossary.md#provenance) facts.
- The targeted Markdown validator resolved 399 local links, including 115 heading anchors, across
  the six changed documentation/planning files. All Markdown fences balance, all twelve paths end
  with a newline, targeted trailing-whitespace scans found no matches, and `git diff --check`
  passes.
- Final changed-path inventory contains exactly the twelve authorized paths: Tensor, the new
  helper, two Javadoc-only semantic contracts, TensorTest, the new focused test, Tensor API,
  Compile API, glossary, this task, model master plan, and roadmap. Task/master/roadmap status is
  synchronized as Complete. Task 0018D1 remains Draft and no task-0018D1 specification exists.
- Training API remains accurate unchanged because the task adds no gradient object/rule, autograd,
  parameter, optimizer, publication, session, or training execution behavior. The capability
  baseline remains accurate unchanged because it already lists gather/gather-axis/take-along-axis,
  exact integral index types, and the separate support layers.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [parameter glossary](../../../../glossary.md#term-parameter); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication))</span>
- DataType, Dimension/Shape, TensorDescriptor, TensorFactory, TensorProvenance, Operation,
  SelectKind/SelectAttrs, scalar-select construction, and other operation-family contracts remain
  accurate unchanged because this task composes their existing metadata, structural equality,
  derived identity, and provenance rules without modifying them. Only the two authorized
  axis-gather semantic Javadocs required timing/bounds wording corrections.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [structural equality glossary](../../../../glossary.md#term-structural-equality); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance tests, integration tests, Gradle/dependencies, and other modules remain
  accurate unchanged because the task stays within model ownership and changes no dependency
  rule, backend behavior, executable end-to-end behavior, or build requirement.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>

## Implementation notes

- Added the exact four public [Tensor](../../../../glossary.md#tensor)-index methods and the exact field-free nine-method helper.
- Added focused ten-test coverage and expanded the exact Tensor public-method inventory by four.
- Finalized public/helper/semantic Javadocs, Tensor and Compile API references, the glossary,
  executable example, and synchronized planning status without declaration or behavior changes to
  completed semantic contracts.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: Implemented and documented public axis gather, gather-axis/take, and
  take-along-axis expression construction with exact type/axis/Shape validation, unresolved result
  layout, retained data metadata, and fresh ordered provenance.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [gather glossary](../../../../glossary.md#gather); [Shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation); [layout glossary](../../../../glossary.md#layout); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- Files changed or created: Exactly the twelve authorized production, test, API, glossary, task,
  master-plan, and roadmap paths, including the two explicitly authorized Javadoc-only semantic
  corrections.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Tests and validation: Focused 10-test and 14-test suites, all 667 model tests across 78 suites,
  model Javadoc, root tests, javap/reflection/import/source/generated-page review, executable Java
  26 example, 399-link/115-anchor checks, fence/whitespace/newline checks, exact scope/status and
  no-0018D1-spec checks, and <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0018d/review_model_0018d_docs` completed the required independent pass
  with General, API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API, Compile API, and glossary now describe current axis-gather
  expression construction while index-value bounds, gradients, compiler behavior, lowering, and
  execution remain planned or separately owned.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [lowering glossary](../../../../glossary.md#lowering))</span>
- Javadoc review: Tensor's four methods, the helper type/constructor/nine methods, and the two
  authorized semantic timing/bounds corrections are final. Related foundational and adjacent
  contracts remain accurate for the reasons recorded above.
- Glossary impact: Axis-gather terminology now distinguishes semantic values, current input-aware
  expression construction, structural dynamic compatibility, exact take aliasing, and deferred
  value/executable behavior.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0018D. Task 0018D1 remains Draft without a detailed
  specification.

Status: Complete
