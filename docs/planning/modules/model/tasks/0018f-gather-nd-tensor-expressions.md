# Task 0018F: [Gather-ND](../../../../glossary.md#gather-nd) [Tensor](../../../../glossary.md#tensor) Expressions

## Status

Complete

## Goal

Add public [model](../../../../glossary.md#term-model)-level [Gather-ND](../../../../glossary.md#gather-nd) expressions with exact index-type, [rank](../../../../glossary.md#term-rank), [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-prefix,
tuple-depth, and result-[Shape](../../../../glossary.md#shape) [validation](../../../../glossary.md#term-validation).

The zero-batch convenience and explicit-batch method both consume ordered logical inputs
`[data, indices]`, preserve data [metadata](../../../../glossary.md#term-metadata), leave result [layout](../../../../glossary.md#layout) unresolved, and record exact
`GATHER_ND` [provenance](../../../../glossary.md#provenance). They construct expression metadata only and never read index values.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly two public instance methods to [`Tensor`](../../../../glossary.md#tensor):
  - `Tensor gatherNd(Tensor indices)`
  - `Tensor gatherNd(Tensor indices, int batchDimensions)`
- Add one field-free package-private final `TensorGatherNdExpressions` helper.
- Give the helper exactly two package-private entries and six private methods specified below.
- Make the no-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) entry delegate exactly to the explicit entry with `batchDimensions = 0`.
- Require indices [data type](../../../../glossary.md#data-type--datatype) to be exactly [`INT32`](../../../../glossary.md#data-type--datatype) or [`INT64`](../../../../glossary.md#data-type--datatype).
- Require indices [rank](../../../../glossary.md#term-rank) at least one and batch count smaller than both indices and data rank.
- Require every shared leading [batch Dimension](../../../../glossary.md#batch-dimension) to be structurally equal.
- Require the final indices Dimension to be statically known and use its positive extent as tuple
  depth.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [extent glossary](../../../../glossary.md#term-extent))</span>
- Require [tuple depth](../../../../glossary.md#term-tuple-depth) no greater than `dataRank - batchDimensions`.
- Derive exact result Shape as indices prefix without its final tuple-depth Dimension followed by
  the unindexed data suffix.
- Produce [canonical](../../../../glossary.md#term-canonical-form) [scalar](../../../../glossary.md#term-scalar) [Shape](../../../../glossary.md#shape) when both result parts are empty.
- Preserve exact data type and [gradient](../../../../glossary.md#term-gradient) eligibility; always leave result [layout](../../../../glossary.md#layout) unresolved.
- Construct exact `GatherNdKind.GATHER_ND`, `GatherNdAttrs`, ordered `[data, indices]` provenance,
  no label or storage, and one final <span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>`TensorFactory.createDerived` call.
- Keep every valid request explicit and fresh.
- Update `TensorTest` only for the two-method public API expansion and add one focused expression
  test.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Apply [Javadoc](../../../../glossary.md#term-javadoc)-only current-status corrections to `GatherNdKind` and `GatherNdAttrs`; do not
  change their declarations or behavior.
- Finalize Tensor API, Compile API, glossary, task evidence, master plan, and roadmap through the
  mandatory independent documentation pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- [gather](../../../../glossary.md#gather) [axis](../../../../glossary.md#term-axis), [scalar select](../../../../glossary.md#scalar-select), [scatter-ND](../../../../glossary.md#scatter-nd), axis scatter, [masks](../../../../glossary.md#term-boolean-mask), slices, or another [operation](../../../../glossary.md#operation) family
- primitive-array Gather-ND conveniences, tuple objects, collection inputs, default indices, or
  another overload<span data-glossary-navigation="true"> ([Gather-ND glossary](../../../../glossary.md#gather-nd))</span>
- accepting floating/BOOL indices, implicit conversion, inspecting whether numeric values are
  integral, or converting <span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype))</span>[`INT64`](../../../../glossary.md#data-type--datatype) to [`INT32`](../../../../glossary.md#data-type--datatype)
- reading, normalizing, clamping, or bounds-checking index values; negative index-value policy is
  not model metadata construction<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- dynamic tuple depth, symbolic tuple-length constraints, mismatched dynamic batch symbols,
  broadcasting batch Dimensions, or graph-wide Shape inference<span data-glossary-navigation="true"> ([tuple depth glossary](../../../../glossary.md#term-tuple-depth); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [broadcasting glossary](../../../../glossary.md#broadcasting); [Dimensions glossary](../../../../glossary.md#dimension); [graph glossary](../../../../glossary.md#graph); [Shape glossary](../../../../glossary.md#shape); [inference glossary](../../../../glossary.md#term-inference))</span>
- resolved result layout, view/alias geometry, storage access, materialization, device state, or
  backend route selection<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena); [materialization glossary](../../../../glossary.md#materialization); [device glossary](../../../../glossary.md#term-device); [backend route glossary](../../../../glossary.md#backend-route))</span>
- modifying Gather-ND semantic declarations/behavior, Shape, Dimension, DataType,
  TensorDescriptor, TensorFactory, TensorProvenance, Operation, or unrelated completed contracts<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype))</span>
- gradients, repeated-index accumulation, scatter-ND backward, graph capture, canonicalization,
  compiler, planning, prepare, runtime, backend, engine, trace, ONNX implementation, training, or
  execution behavior<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- another production helper/type, dependency, Gradle/build option, architecture change, another
  module, or task-0018G specification<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

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
- [Task 0018C](0018c-axis-gather-semantics.md)
- [Task 0018D](0018d-axis-gather-tensor-expressions.md)
- [Task 0018E](0018e-gather-nd-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only legacy branch exposes zero-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) and explicit-batch [Gather-ND](../../../../glossary.md#gather-nd). It requires numeric
indices, validates shared batch [Shapes](../../../../glossary.md#shape), reads [tuple depth](../../../../glossary.md#term-tuple-depth) from the final indices [extent](../../../../glossary.md#term-extent), and derives
the result from the indices prefix plus unindexed data suffix.

The new [model](../../../../glossary.md#term-model) tightens index [data type](../../../../glossary.md#data-type--datatype) to exact [`INT32`](../../../../glossary.md#data-type--datatype) or [`INT64`](../../../../glossary.md#data-type--datatype), supports current static/dynamic
[Dimension](../../../../glossary.md#dimension) values conservatively, and uses [canonical](../../../../glossary.md#term-canonical-form) [scalar](../../../../glossary.md#term-scalar) Shape rather than legacy `[1]` when the
formula produces [rank](../../../../glossary.md#term-rank) zero. It does not copy legacy value access, [graph](../../../../glossary.md#graph) builders, [gradient](../../../../glossary.md#term-gradient)
callbacks, scatter implementations, traits, [lowering](../../../../glossary.md#lowering), [kernels](../../../../glossary.md#kernel), or [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) behavior.

## Architecture constraints

- [`Tensor`](../../../../glossary.md#tensor) remains public mutable API state, not IR. The methods create fresh storage-free
  expressions through the existing derived-factory seam.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Kind and attributes come only from completed task 0018E.
- The helper may inspect [immutable](../../../../glossary.md#term-immutable) [descriptors](../../../../glossary.md#term-descriptor) and [Shapes](../../../../glossary.md#shape), but never values or [host](../../../../glossary.md#term-host)/[device](../../../../glossary.md#term-device) storage.
- Index type is explicit metadata and is validated locally. Index bounds require values and remain
  deferred.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- Dynamic Dimensions pass only through existing structural equality. Tuple depth must be static
  because it controls which axes exist in the result Shape.<span data-glossary-navigation="true"> ([Dimensions glossary](../../../../glossary.md#dimension); [structural equality glossary](../../../../glossary.md#term-structural-equality); [Tuple depth glossary](../../../../glossary.md#term-tuple-depth); [axes glossary](../../../../glossary.md#term-axis))</span>
- [Gather-ND](../../../../glossary.md#gather-nd) is value-reordering/[materialization](../../../../glossary.md#materialization) semantics, so result [layout](../../../../glossary.md#layout) is always unresolved.
- Type and [gradient](../../../../glossary.md#term-gradient) eligibility come only from data; indices do not contribute eligibility.
- Compiler owns capture/canonicalization; compiler/training owns backward scatter; later layers own
  lowering, materialization, and execution.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [lowering glossary](../../../../glossary.md#lowering))</span>
- No [dependency](../../../../glossary.md#term-dependency), package [ownership](../../../../glossary.md#term-ownership-meanings), or [module](../../../../glossary.md#term-module) boundary changes are authorized.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.tensor.Tensor` — receives two [Gather-ND](../../../../glossary.md#gather-nd) methods.
- `io.github.pho001.synaptik.model.tensor.TensorGatherNdExpressions` — owns local validation,
  Shape derivation, and semantic construction.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- `TensorGatherNdExpressionTest` — mirrors `model.tensor` for focused [validation](../../../../glossary.md#term-validation).
- `TensorTest` — changes only exact public [API](../../../../glossary.md#term-application-programming-interface-api) inventory/reflection assertions.
- `GatherNdKind` and `GatherNdAttrs` remain in `model.operation.index`; only stale temporal and
  validation-boundary Javadoc wording may change.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) methods

Add exactly:

```java
public Tensor gatherNd(Tensor indices) {
    return TensorGatherNdExpressions.gatherNd(this, indices);
}

public Tensor gatherNd(Tensor indices, int batchDimensions) {
    return TensorGatherNdExpressions.gatherNd(this, indices, batchDimensions);
}
```

Each method contains one return statement and one matching helper call, is non-static and
non-synchronized, and performs no direct [validation](../../../../glossary.md#term-validation) or field access.

### Helper [shape](../../../../glossary.md#shape)

Create one package-private final, field-free class with one private zero-[argument](../../../../glossary.md#term-argument) constructor and
exactly these eight static methods:

```java
static Tensor gatherNd(Tensor data, Tensor indices)
static Tensor gatherNd(Tensor data, Tensor indices, int batchDimensions)
private static void validateIndexType(TensorDescriptor indicesDescriptor)
private static void validateBatchDimensions(
        Shape dataShape, Shape indicesShape, int batchDimensions)
private static void validateBatchPrefix(
        Shape dataShape, Shape indicesShape, int batchDimensions)
private static int tupleDepth(
        Shape dataShape, Shape indicesShape, int batchDimensions)
private static Shape resultShape(
        Shape dataShape, Shape indicesShape, int batchDimensions, int tupleDepth)
private static Tensor create(
        Tensor data,
        Tensor indices,
        TensorDescriptor dataDescriptor,
        Shape resultShape,
        GatherNdAttrs attrs)
```

Add no field, nested type, alternate constructor, overload, [cache](../../../../glossary.md#term-cache), [mutable](../../../../glossary.md#term-mutable) state, or extra method.

### Default delegation

The two-[argument](../../../../glossary.md#term-argument) helper contains exactly:

```java
return gatherNd(data, indices, 0);
```

It performs no independent [validation](../../../../glossary.md#term-validation). Both public forms therefore share errors, [Shape](../../../../glossary.md#shape),
[provenance](../../../../glossary.md#provenance), [layout](../../../../glossary.md#layout), [identity](../../../../glossary.md#term-identity-meanings), and side effects.

### Explicit [validation](../../../../glossary.md#term-validation) order

The explicit helper performs exactly:

1. null-check `data` with message `data`;
2. null-check `indices` with message `indices`;
3. read exact data [descriptor](../../../../glossary.md#term-descriptor) once;
4. read exact indices descriptor once;
5. call `validateIndexType` once;
6. read exact data [Shape](../../../../glossary.md#shape) once;
7. read exact indices Shape once;
8. reject indices [rank](../../../../glossary.md#term-rank) zero with `IllegalArgumentException` and exact message
   `gatherNd indices rank must be at least 1`;
9. construct one `GatherNdAttrs(batchDimensions)`, preserving its exact negative-value failure;
10. call `validateBatchDimensions` once;
11. call `validateBatchPrefix` once;
12. call `tupleDepth` once;
13. call `resultShape` once;
14. call `create` once.

All local failures precede result identifier allocation.

`validateIndexType` accepts only [`INT32`](../../../../glossary.md#data-type--datatype) and [`INT64`](../../../../glossary.md#data-type--datatype); otherwise it throws
`IllegalArgumentException` with exact message:

```text
gatherNd indices data type must be INT32 or INT64: <actual>
```

### [Batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [validation](../../../../glossary.md#term-validation)

After non-negative validation by `GatherNdAttrs`, `validateBatchDimensions` requires:

- `batchDimensions < indicesShape.rank()`, otherwise exact message
  `gatherNd batchDimensions must be less than indices rank: batchDimensions=<B>, indicesRank=<Q>`;
- `batchDimensions < dataShape.rank()`, otherwise exact message
  `gatherNd batchDimensions must be less than data rank: batchDimensions=<B>, dataRank=<R>`.

`validateBatchPrefix` compares [Dimensions](../../../../glossary.md#dimension) at [axes](../../../../glossary.md#term-axis) `[0, B)` in increasing order. On first mismatch,
throw exact message:

```text
gatherNd batch dimension at axis <axis> must match data: expected=<dataDimension>, actual=<indicesDimension>
```

[Structural equality](../../../../glossary.md#term-structural-equality) supports identical static sizes and equal dynamic symbols. No [broadcasting](../../../../glossary.md#broadcasting) or
constraint generation is allowed.

### [Tuple depth](../../../../glossary.md#term-tuple-depth)

`tupleDepth` reads the final indices [Dimension](../../../../glossary.md#dimension) exactly once.

- If it is dynamic, throw `IllegalArgumentException` with exact message
  `gatherNd tuple depth must be statically known`.
- Let its static long [extent](../../../../glossary.md#term-extent) be `depth` and `maximum = dataRank - batchDimensions`.
- If `depth < 1` or `depth > maximum`, throw exact message
  `gatherNd tuple depth must be in [1, data rank - batchDimensions]: depth=<depth>, maximum=<maximum>`.
- Return the validated depth as `int`; the upper bound proves conversion safe.

### Result [Shape](../../../../glossary.md#shape)

`resultShape` creates one [Dimension](../../../../glossary.md#dimension) array ordered as:

1. exact indices Dimensions `[0, indicesRank - 1)`, excluding only tuple-depth Dimension;
2. exact data Dimensions `[batchDimensions + tupleDepth, dataRank)`.

Use checked [rank](../../../../glossary.md#term-rank) addition when allocating. Call `Shape.ofDimensions` once. Empty parts produce
[canonical](../../../../glossary.md#term-canonical-form) [scalar](../../../../glossary.md#term-scalar) Shape.

Examples:

- data `[2, 3, 4]`, indices `[5, 2]`, `B=0`, `K=2` gives `[5, 4]`;
- data `[2, 3, 4]`, indices `[2, 5, 1]`, `B=1`, `K=1` gives `[2, 5, 4]`;
- data `[2, 3]`, indices `[2]`, `B=0`, `K=2` gives scalar `[]`;
- data `[N, 3, 4]`, indices `[N, M, 1]`, `B=1`, `K=1` gives `[N, M, 4]` while retaining exact
  `N`, `M`, and suffix Dimension references.

### Result [metadata](../../../../glossary.md#term-metadata)

`create` constructs exactly:

```java
TensorDescriptor descriptor = new TensorDescriptor(
        dataDescriptor.dataType(), resultShape, Optional.empty(), dataDescriptor.requiresGrad());
Operation operation = new Operation(GatherNdKind.GATHER_ND, attrs);
TensorProvenance provenance = new TensorProvenance(operation, List.of(data, indices));
return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
```

Every current [data type](../../../../glossary.md#data-type--datatype) is accepted for data. Results preserve exact data type/eligibility, have
unresolved [layout](../../../../glossary.md#layout), absent label/storage, exact attrs, ordered [provenance](../../../../glossary.md#provenance), and fresh [identity](../../../../glossary.md#term-identity-meanings).

## Affected files

Expected implementation change:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorGatherNdExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/GatherNdKind.java`
  — Javadoc only<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/GatherNdAttrs.java`
  — Javadoc only
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorGatherNdExpressionTest.java`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the twelve paths listed above. The two semantic files allow
[Javadoc](../../../../glossary.md#term-javadoc)-only current-status corrections; declarations and behavior must remain [bytecode](../../../../glossary.md#term-bytecode)-equivalent.

If implementation requires another production type/helper, semantic behavior change, another
test, capability baseline, architecture, [dependency](../../../../glossary.md#term-dependency), build, another [module](../../../../glossary.md#term-module), or more than twelve
paths, stop and propose a follow-up task.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document both public methods and all helper methods/constructor.
- Explain validation order, index type, ranks, batch prefix, static tuple depth, result formula,
  dynamic equality, and all examples.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [tuple depth glossary](../../../../glossary.md#term-tuple-depth))</span>
- Explain unresolved [layout](../../../../glossary.md#layout), exact type/eligibility, [provenance](../../../../glossary.md#provenance), fresh [identity](../../../../glossary.md#term-identity-meanings), and no storage.
- State that index values/bounds are never inspected.
- Correct GatherNdKind/GatherNdAttrs wording from future input-aware boundary to current public
  boundary, without declaration or behavior changes.
- Distinguish [axis](../../../../glossary.md#term-axis) [gather](../../../../glossary.md#gather) and [scatter-ND](../../../../glossary.md#scatter-nd) and avoid promising [gradients](../../../../glossary.md#term-gradient)/execution/[backend](../../../../glossary.md#backend) support.

## Acceptance criteria

- [Tensor](../../../../glossary.md#tensor) adds exactly two public non-static, non-synchronized [Gather-ND](../../../../glossary.md#gather-nd) methods.
- Each delegates once; default uses exact [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) count zero through helper delegation.
- Helper is package-private/final/field-free with one private constructor and exactly eight methods.
- [Validation](../../../../glossary.md#term-validation) follows exact order, exception types, and messages.
- Only [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) indices pass; values are not read.
- Batch prefix, static positive [tuple depth](../../../../glossary.md#term-tuple-depth), and result [Shape](../../../../glossary.md#shape) rules match specification.
- [Canonical](../../../../glossary.md#term-canonical-form) [scalar](../../../../glossary.md#term-scalar) and exact [Dimension](../../../../glossary.md#dimension) reference retention are verified.
- Results preserve data [metadata](../../../../glossary.md#term-metadata), unresolved [layout](../../../../glossary.md#layout), exact GATHER_ND/attrs, `[data, indices]`, no
  label/storage, and fresh identity.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Semantic declaration/behavior remains unchanged; only authorized [Javadoc](../../../../glossary.md#term-javadoc) wording changes.
- Documentation/planning are independently reviewed and synchronized; unrelated docs receive
  reasoned no-change conclusions.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- All validation passes with exactly twelve paths.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run):

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.tensor.TensorGatherNdExpressionTest
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Manually verify `javap`, reflection, one-call delegation, helper [shape](../../../../glossary.md#shape),
semantic [bytecode](../../../../glossary.md#term-bytecode) equivalence apart from documentation, imports, generated [Javadoc](../../../../glossary.md#term-javadoc), executable
example, Markdown links/anchors/fences/whitespace, exact twelve paths, synchronized status, and no
task-0018G specification.

## Dependencies

- Tasks 0001/0002 provide [DataType](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape), [Dimension](../../../../glossary.md#dimension), and [structural equality](../../../../glossary.md#term-structural-equality).
- Tasks 0011–0013 provide [Tensor](../../../../glossary.md#tensor), derived [identity](../../../../glossary.md#term-identity-meanings), and [provenance](../../../../glossary.md#provenance).
- Task 0018E provides exact [Gather-ND](../../../../glossary.md#gather-nd) semantics and normalized [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) attributes.

## Follow-up tasks

- 0018G: [axis](../../../../glossary.md#term-axis) scatter semantics and reduction policy.
- 0018I: [Scatter-ND](../../../../glossary.md#scatter-nd) semantics and reduction policy.

Do not create follow-up specifications during this task.

## Architecture impact

Expected impact: None.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0007/0011/0012/0013/0018C/0018D/0018E/0018F,
Tensor API, Compile API, Training API, glossary, current DataType/Dimension/Shape/TensorDescriptor/
Tensor/TensorFactory/TensorProvenance/Operation/GatherNdKind/GatherNdAttrs contracts and tests, and
Java 26 Gradle configuration.

Implement task 0018F exactly. Modify Tensor.java, add package-private final
TensorGatherNdExpressions.java, update TensorTest only for two methods, and add
TensorGatherNdExpressionTest. Add exactly gatherNd(Tensor) and gatherNd(Tensor,int).

The field-free helper has exactly eight methods. Follow exact null/type/rank/batch-prefix/static-
tuple-depth validation order/messages. Derive indices-prefix plus data-suffix Shape, including
canonical scalar. Preserve data type/eligibility, unresolved layout, exact GATHER_ND/attrs and
[data, indices] provenance, and one createDerived call. Never inspect index values.

Permit Javadoc-only current-status corrections in GatherNdKind/GatherNdAttrs; do not change their
declarations or behavior. Do not add scatter, gradients, compiler/runtime/backend behavior,
dependencies, build/architecture changes, or later specs. Stop beyond twelve paths.

Run all validation, then hand actual diff/evidence to a separate clean-context docs agent to
finalize permitted Javadocs/Tensor API/Compile API/glossary/planning and rerun validation.
Update task/master/roadmap only after both passes. Leave 0018G Draft without a spec. Do not commit.
```

## Local decisions

- The public no-batch method remains a one-call Tensor delegation, and the helper's two-argument
  entry delegates exactly once to the explicit entry with batch count zero. This keeps one
  validation and construction path without inventing a default attributes singleton.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [argument glossary](../../../../glossary.md#term-argument); [validation glossary](../../../../glossary.md#term-validation); [singleton glossary](../../../../glossary.md#term-singleton))</span>
- The exact eight-method, field-free helper owns only local descriptor and Shape work. It reads
  each descriptor once, validates metadata in the specified order, calls <span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata))</span>`Shape.ofDimensions`
  once for the result, and performs one final `TensorFactory.createDerived` call.
- Shared [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) compatibility uses existing structural `Dimension.equals` semantics. Equal dynamic
  symbols pass even when represented by distinct immutable objects; different symbols fail rather
  than creating a compiler-style constraint.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Static positive tuple depth remains an occurrence-specific final indices Dimension. The result
  preserves exact Dimension references from the indices prefix and data suffix, and the existing
  empty <span data-glossary-navigation="true"> ([tuple depth glossary](../../../../glossary.md#term-tuple-depth); [occurrence glossary](../../../../glossary.md#term-occurrence); [Dimension glossary](../../../../glossary.md#dimension))</span>`Shape.ofDimensions` path supplies [canonical](../../../../glossary.md#term-canonical-form) [scalar](../../../../glossary.md#term-scalar) [identity](../../../../glossary.md#term-identity-meanings).
- `GatherNdKind` and `GatherNdAttrs` received only temporal Javadoc corrections from “later” or
  “future” input-aware construction to the current public Tensor boundary. Comparison against the
  pre-documentation <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`javap` baselines proves declaration and executable-[bytecode](../../../../glossary.md#term-bytecode) equivalence.
- The Tensor API uses one complete executable example covering zero-batch, batched, scalar, and
  dynamic-Dimension cases. It exposes metadata and provenance only and makes no value, bounds,
  gradient, compiler, backend, materialization, or execution claim.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend); [materialization glossary](../../../../glossary.md#materialization))</span>

## Known limitations

- Gather-ND construction does not read, normalize, clamp, or bounds-check index values, including
  negative or out-of-range coordinates.<span data-glossary-navigation="true"> ([Gather-ND glossary](../../../../glossary.md#gather-nd); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Tuple depth must be statically known and positive. The model creates no symbolic tuple-length or
  dynamic-rank constraint.<span data-glossary-navigation="true"> ([Tuple depth glossary](../../../../glossary.md#term-tuple-depth); [model glossary](../../../../glossary.md#term-model); [rank glossary](../../../../glossary.md#term-rank))</span>
- Shared batch Dimensions must already be structurally equal. Broadcasting and graph-wide symbol
  equivalence remain outside this local model boundary.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [Dimensions glossary](../../../../glossary.md#dimension); [Broadcasting glossary](../../../../glossary.md#broadcasting); [graph glossary](../../../../glossary.md#graph); [symbol glossary](../../../../glossary.md#term-symbol))</span>
- Every result layout is unresolved and no storage is attached. Values, physical aliases,
  materialization, gradients or scatter-ND backward, graph capture/canonicalization, ONNX mapping,
  backend lowering, and execution remain owned by later layers and tasks.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [materialization glossary](../../../../glossary.md#materialization); [gradients glossary](../../../../glossary.md#term-gradient); [scatter-ND glossary](../../../../glossary.md#scatter-nd); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [ONNX glossary](../../../../glossary.md#term-onnx); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Clean implementation context `/root/implement_model_0018f` added the two public Tensor methods,
  exact helper, and focused tests, then handed the actual shared-tree diff and passing pre-handoff
  evidence to independent documentation context<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
  `/root/implement_model_0018f/review_model_0018f_docs`. The documentation pass used General plus
  API/Javadoc style for Java and API contracts, Planning style for this task/master/roadmap, and
  Example format for the complete Gather-ND example.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning); [Gather-ND glossary](../../../../glossary.md#gather-nd))</span>
- The documentation context read `AGENTS.md`, the complete architecture contract and focused
  current/overview/lifecycle/module-boundary/dependency explanations, documentation workflow and
  selected profiles, planning guide and roadmap, model capabilities/master plan, tasks
  0001/0002/0007/0011/0012/0013/0018C/0018D/0018E/0018F, Tensor/Compile/Training API references,
  glossary, Java 26 root/model Gradle configuration, final implementation/tests, related model
  contracts, generated Javadoc, and the complete diff before editing.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [lifecycle glossary](../../../../glossary.md#lifecycle); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [model glossary](../../../../glossary.md#term-model); [Compile glossary](../../../../glossary.md#compile); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Related-contract review covered DataType, Dimension variants, Shape, TensorDescriptor, Tensor,
  TensorFactory, TensorProvenance, Operation, GatherNdKind, GatherNdAttrs, axis-gather semantics,
  and focused tests. Their behavior remains unchanged; only Tensor's new API/Javadocs and the two
  explicitly authorized semantic temporal Javadocs changed.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [Dimension glossary](../../../../glossary.md#dimension); [Shape glossary](../../../../glossary.md#shape); [Operation glossary](../../../../glossary.md#operation); [axis glossary](../../../../glossary.md#term-axis); [gather glossary](../../../../glossary.md#gather))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorGatherNdExpressionTest` — `BUILD SUCCESSFUL`; XML
  reports 10 tests, zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; XML reports 14 tests,
  zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; XML aggregation reports 694 tests across
  81 suites with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated [`Tensor`](../../../../glossary.md#tensor), `GatherNdKind`, and
  `GatherNdAttrs` pages contain both overloads, exact parameters/results/failures, validation and
  Shape rules, current semantic boundary, provenance/metadata ownership, and explicit deferred
  value/gradient/compiler/backend/execution behavior. Package-private helper Javadocs were
  reviewed in source.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [provenance glossary](../../../../glossary.md#provenance); [metadata glossary](../../../../glossary.md#term-metadata); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- `./gradlew test` — `BUILD SUCCESSFUL`; the repository lifecycle reports 36 actionable tasks
  with no failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` confirmed two one-call public
  delegates; a package-private final helper with no fields, one private constructor, and exactly
  eight methods; default-to-zero delegation; exact null/type/rank/attrs/batch-prefix/tuple/result/
  create order; <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>`Math.addExact`; one result array; one `Shape.ofDimensions`; exact descriptor,
  operation, provenance, and final <span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>`createDerived` construction. Focused reflection tests confirm
  the same API and modifier shape.
- `cmp -s` of fresh `javap -p -c -s` output against `/tmp/gather-kind-after.txt` and
  `/tmp/gather-attrs-after.txt` passed, proving the Javadoc-only semantic edits leave declarations
  and executable behavior bytecode-equivalent.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- The documented `GatherNdExpressionExample` compiled with Java 26 against model main classes and
  ran successfully. It printed `[5, 4]`, `[2, 5, 4]`, [canonical](../../../../glossary.md#term-canonical-form) [scalar](../../../../glossary.md#term-scalar) [identity](../../../../glossary.md#term-identity-meanings), `[N, M, 4]`,
  `GatherNdAttrs[batchDimensions=1]`, exact dynamic/suffix Dimension retention, ordered exact
  provenance, and combined type/eligibility/unresolved-layout/no-label/no-storage facts.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- Source/import/manual inspection found only the expected local model/JDK dependencies and no
  value or storage read, graph/compiler/planning/prepare/runtime/backend/training dependency, new
  package, build option, reflection, registry, or service state. The focused storage test proves
  input arrays and attached storage remain untouched while the result is storage-free.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [graph glossary](../../../../glossary.md#graph); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [registry glossary](../../../../glossary.md#term-registry))</span>
- The targeted Markdown validator resolved 417 local links, including 121 heading anchors, across
  Tensor API, Compile API, glossary, this task, model master plan, and roadmap with zero errors.
  Backtick fence counts were 218, 4, 4, 16, 2, and 0 respectively and therefore balanced; no
  tilde fences or trailing whitespace were found, and every changed path has a final newline.
- Final scope review found exactly the twelve authorized paths: Tensor, the new helper, the two
  Javadoc-only semantic contracts, TensorTest, the new focused test, Tensor API, Compile API,
  glossary, this task, model master plan, and roadmap. Task 0018G remains a Draft row and no
  task-0018G specification exists. <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Compile API changed only to list the current model expression input and explicitly retains graph
  capture, inference, optimization, artifacts, materialization planning, and engine behavior as
  planned. Training API remains accurate unchanged because no gradient rule/object, autograd,
  parameter, optimizer, publication, session, or training execution behavior changed.<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [inference glossary](../../../../glossary.md#term-inference); [artifacts glossary](../../../../glossary.md#term-artifact); [materialization glossary](../../../../glossary.md#materialization); [engine glossary](../../../../glossary.md#term-engine); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication))</span>
- `capabilities.md` remains accurate unchanged because it already inventories Gather-ND with batch
  dimensions, exact integral index tensors, and separate model/public/compiler/backend/runtime
  support layers. `ARCHITECTURE.md`, focused architecture docs, ADRs, architecture tests,
  backend-conformance tests, integration tests, Gradle/dependencies, other modules, and later task
  specs remain unchanged because this task changes no ownership, dependency rule, backend
  behavior, executable end-to-end behavior, or build requirement.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>

## Implementation notes

- Added exactly `Tensor.gatherNd(Tensor)` and `Tensor.gatherNd(Tensor, int)` as one-call public
  instance delegations.
- Added the field-free package-private `TensorGatherNdExpressions` with exact metadata validation,
  structural batch-prefix checks, static tuple-depth validation, Shape derivation, and one final
  storage-free derived construction.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [validation glossary](../../../../glossary.md#term-validation); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [Shape glossary](../../../../glossary.md#shape))</span>
- Added the ten-test focused suite and expanded [Tensor](../../../../glossary.md#tensor)'s exact public [API](../../../../glossary.md#term-application-programming-interface-api) assertions by two.
- Finalized Tensor/helper and semantic Javadocs, Tensor and Compile API current-status references,
  glossary terminology, the complete executable example, and synchronized planning evidence.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: Implemented and documented public zero-batch and explicit-batch Gather-ND
  metadata construction with exact type/rank/batch/tuple/Shape validation and ordered provenance.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [Gather-ND glossary](../../../../glossary.md#gather-nd); [metadata glossary](../../../../glossary.md#term-metadata); [rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation); [provenance glossary](../../../../glossary.md#provenance))</span>
- Files changed or created: Exactly the twelve authorized production, test, API, glossary, task,
  master-plan, and roadmap paths.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Tests and validation: Focused 10-test and 14-test suites, all 694 model tests across 81 suites,
  model Javadoc, root tests, bytecode/reflection/import/source/generated-page review, executable
  Java 26 example, 417-link/121-anchor checks, fence/whitespace/newline checks, exact scope/status
  and no-0018G-spec checks, semantic bytecode equivalence, and <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0018f/review_model_0018f_docs` completed the mandatory independent pass
  with General, API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API, Compile API, and glossary now describe current Gather-ND
  expression construction while index-value bounds, gradients, compiler behavior, materialization,
  lowering, backend behavior, and execution remain separately owned.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering); [backend glossary](../../../../glossary.md#backend))</span>
- Javadoc review: Both public methods, the helper type/constructor/eight methods, Tensor type, and
  two authorized semantic timing corrections are final. Related model contracts remain accurate
  unchanged for the reasons recorded above.
- Glossary impact: Gather-ND now distinguishes semantic values from current input-aware Tensor
  construction, structural dynamic equality, static tuple depth, exact Dimension retention,
  canonical scalar results, and deferred value/executable behavior.<span data-glossary-navigation="true"> ([tuple depth glossary](../../../../glossary.md#term-tuple-depth); [Dimension glossary](../../../../glossary.md#dimension); [canonical glossary](../../../../glossary.md#term-canonical-form); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0018F. Task 0018G remains the next Draft frontier without a
  detailed specification.

Status: Complete
