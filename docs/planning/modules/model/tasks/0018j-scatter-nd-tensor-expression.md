# Task 0018J: [Scatter-ND](../../../../glossary.md#scatter-nd) [Tensor](../../../../glossary.md#tensor) Expression

**Editorial reading key (current context, not a status update).**

This completed task records metadata-only construction, not execution support. The caller invokes
`data.scatterNd`; `Tensor` delegates to `TensorScatterNdExpressions`, which validates descriptors
and records ordered references to data, indices, and updates—not their values. Data is the unchanged
base; indices describe coordinate tuples; updates supply replacement values or suffix slices. Tuple
depth counts addressed axes; the batch prefix identifies shared leading dimensions. For data
`[2,3,4]`, zero-batch indices `[5,2]` require updates `[5,4]`, while the result remains `[2,3,4]`.

Follow the current [Scatter-ND
contract](../../../../reference/tensor/scatter.md#scatter-nd-expressions) for overloads and
validation, and [Scatter Elements
semantics](../../../../reference/tensor/scatter.md#scatter-elements-expressions) for the reduction
rules now shared by Scatter-ND. Those current numerical descriptions extend beyond this task’s
historical scope. Duplicate targets mean multiple tuples address the same destination; `NONE`
requires uniqueness, not checked here. Model semantics defines meaning; execution providers realize
supported routes rather than redefine it. Semantic responsibility is not buffer ownership: this
storage-free expression neither executes nor owns result buffers. Gradient eligibility does not
promise a backward rule.

## Status

Complete

## Goal

Add public, locally validated [Tensor](../../../../glossary.md#tensor)-expression construction for functional tuple-index
`SCATTER_ND`.

The three public overloads consume ordered `[data, indices, updates]`, share one explicit
[validation](../../../../glossary.md#term-validation) path, preserve exact data [Shape](../../../../glossary.md#shape)/type, leave [layout](../../../../glossary.md#layout) unresolved, and record exact
[Scatter-ND](../../../../glossary.md#scatter-nd) semantics without reading index or update values.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly three public instance methods to [`Tensor`](../../../../glossary.md#tensor):
  - `Tensor scatterNd(Tensor indices, Tensor updates)`
  - `Tensor scatterNd(Tensor indices, Tensor updates, ScatterReduction reduction)`
  - `Tensor scatterNd(Tensor indices, Tensor updates, ScatterReduction reduction,
    int batchDimensions)`
- Add one field-free package-private final `TensorScatterNdExpressions` helper with exactly the
  eleven methods specified below.
- Delegate the shortest overload with `ScatterReduction.NONE` and zero [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [Dimensions](../../../../glossary.md#dimension).
- Delegate the reduction overload with the exact supplied reduction and zero batch Dimensions.
- Require exact [`INT32`](../../../../glossary.md#data-type--datatype)/[`INT64`](../../../../glossary.md#data-type--datatype) indices and exact matching data/update types.
- Permit `NONE` for every current data type; permit arithmetic reductions for floating/integral
  data and reject them for BOOL.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Require indices rank at least one, valid normalized batch count, structurally equal batch prefix,
  static positive tuple depth, and exact updates Shape.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [tuple depth glossary](../../../../glossary.md#term-tuple-depth))</span>
- Preserve exact data [Shape](../../../../glossary.md#shape)/type and data/update [gradient](../../../../glossary.md#term-gradient)-eligibility OR in one unresolved result.
- Construct exact `SCATTER_ND`/attributes and ordered `[data, indices, updates]` provenance, then
  call <span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>`TensorFactory.createDerived` once with no label/storage.
- Keep every valid request fresh and [metadata](../../../../glossary.md#term-metadata)-only.
- Update `TensorTest` only for the three-method [API](../../../../glossary.md#term-application-programming-interface-api) expansion and add one focused expression test.
- Permit [Javadoc](../../../../glossary.md#term-javadoc)-only current-status corrections to `ScatterNdKind` and `ScatterNdAttrs`; their
  declarations and behavior must not change.
- Finalize Tensor API, Compile API, glossary, task/master/roadmap through the mandatory independent
  documentation pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- another Scatter-ND overload, primitive/collection indices, default indices, factories, static
  methods, or caller-supplied result Shape/type<span data-glossary-navigation="true"> ([Scatter-ND glossary](../../../../glossary.md#scatter-nd); [Shape glossary](../../../../glossary.md#shape))</span>
- [axis](../../../../glossary.md#term-axis) scatter changes, [Gather-ND](../../../../glossary.md#gather-nd) changes, [gather](../../../../glossary.md#gather)/scatter backward, [masks](../../../../glossary.md#term-boolean-mask), slices, or other families
- floating/[BOOL](../../../../glossary.md#data-type--datatype) indices, conversion, promotion, mixed data/update types, or output type selection
- dynamic tuple depth, symbolic tuple constraints, broadcast batch Dimensions, or graph-wide
  inference<span data-glossary-navigation="true"> ([tuple depth glossary](../../../../glossary.md#term-tuple-depth); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [Dimensions glossary](../../../../glossary.md#dimension); [graph glossary](../../../../glossary.md#graph); [inference glossary](../../../../glossary.md#term-inference))</span>
- reading/normalizing/clamping/bounds-checking index values or detecting `NONE` duplicate [targets](../../../../glossary.md#term-target)
- applying writes/reductions, in-place mutation, numerical order, overflow, NaN/signed-zero,
  reproducibility, atomics, or backend numeric policy<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [backend glossary](../../../../glossary.md#backend))</span>
- resolved [layout](../../../../glossary.md#layout), [aliases](../../../../glossary.md#aliasing-and-arena), storage access, [materialization](../../../../glossary.md#materialization), [device](../../../../glossary.md#term-device) state, or [backend route](../../../../glossary.md#backend-route)
- rejecting floating gradient eligibility for reductions without a current backward rule;
  eligibility metadata is not a backward-support promise<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- modifying foundational/semantic behavior, dependencies, Gradle, architecture, another module,
  or creating task 0019 specifications<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- gradients, graph capture, compiler, planning, prepare, runtime, backend, engine, trace, ONNX,
  training, or execution behavior<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace); [ONNX glossary](../../../../glossary.md#term-onnx))</span>

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
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0018E](0018e-gather-nd-semantics.md)
- [Task 0018F](0018f-gather-nd-tensor-expressions.md)
- [Task 0018G](0018g-axis-scatter-semantics.md)
- [Task 0018H](0018h-axis-scatter-tensor-expressions.md)
- [Task 0018I](0018i-scatter-nd-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only legacy branch exposes replacement with zero [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) count, explicit zero-batch
reduction, and explicit reduction plus batch count. It validates tuple-index [Shape](../../../../glossary.md#shape) relationships,
matching types, and arithmetic-reduction restrictions before [graph](../../../../glossary.md#graph) construction.

The new boundary narrows indices consistently to [`INT32`](../../../../glossary.md#data-type--datatype)/[`INT64`](../../../../glossary.md#data-type--datatype), uses [canonical](../../../../glossary.md#term-canonical-form) [scalar](../../../../glossary.md#term-scalar) updates
instead of legacy `[1]`, preserves [immutable](../../../../glossary.md#term-immutable) [Dimension](../../../../glossary.md#dimension) references, and never treats null as
`NONE`. Legacy value access, [mutable](../../../../glossary.md#term-mutable) Shapes, graph builders, [gradient](../../../../glossary.md#term-gradient) callbacks, traits, [lowering](../../../../glossary.md#lowering),
[kernels](../../../../glossary.md#kernel), and [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) behavior are not copied.

## Architecture constraints

- [`Tensor`](../../../../glossary.md#tensor) remains public mutable API state, not IR. Methods create fresh storage-free expressions
  through the existing derived-factory seam.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Kind, attributes, and reductions come only from completed tasks 0018G and 0018I.
- The helper may inspect [immutable](../../../../glossary.md#term-immutable) [descriptors](../../../../glossary.md#term-descriptor)/[Shapes](../../../../glossary.md#shape), never values or storage.
- Index type, matching update type, ranks, batch prefix, tuple depth, and updates Shape are locally
  decidable. Index bounds and duplicates are not.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [tuple depth glossary](../../../../glossary.md#term-tuple-depth))</span>
- Result always preserves exact data Shape/type and has unresolved [layout](../../../../glossary.md#layout).
- [Gradient](../../../../glossary.md#term-gradient) eligibility is data/update OR; indices do not contribute it.
- Compiler/training owns backward construction. Later layers own value-aware validation,
  lowering, materialization, and execution.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [validation glossary](../../../../glossary.md#term-validation); [lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization))</span>
- No [dependency](../../../../glossary.md#term-dependency), package, or [module](../../../../glossary.md#term-module)-boundary change is authorized.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.tensor.Tensor` — three public [Scatter-ND](../../../../glossary.md#scatter-nd) overloads.
- `io.github.pho001.synaptik.model.tensor.TensorScatterNdExpressions` — local validation, expected
  updates Shape, semantics, and provenance.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance))</span>
- `TensorScatterNdExpressionTest` — same-package focused coverage.
- `TensorTest` — exact public [API](../../../../glossary.md#term-application-programming-interface-api) inventory/reflection changes only.
- `ScatterNdKind` and `ScatterNdAttrs` — [Javadoc](../../../../glossary.md#term-javadoc)-only temporal updates.

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) methods

Add exactly:

```java
public Tensor scatterNd(Tensor indices, Tensor updates) {
    return TensorScatterNdExpressions.scatterNd(this, indices, updates);
}

public Tensor scatterNd(
        Tensor indices, Tensor updates, ScatterReduction reduction) {
    return TensorScatterNdExpressions.scatterNd(this, indices, updates, reduction);
}

public Tensor scatterNd(
        Tensor indices,
        Tensor updates,
        ScatterReduction reduction,
        int batchDimensions) {
    return TensorScatterNdExpressions.scatterNd(
            this, indices, updates, reduction, batchDimensions);
}
```

Each is one matching return/delegate, non-static, non-synchronized, with no field access or direct
[validation](../../../../glossary.md#term-validation).

### Helper [shape](../../../../glossary.md#shape)

Create one package-private final field-free class, one private zero-[argument](../../../../glossary.md#term-argument) constructor, and
exactly these eleven static methods:

```java
static Tensor scatterNd(Tensor data, Tensor indices, Tensor updates)
static Tensor scatterNd(
        Tensor data, Tensor indices, Tensor updates, ScatterReduction reduction)
static Tensor scatterNd(
        Tensor data,
        Tensor indices,
        Tensor updates,
        ScatterReduction reduction,
        int batchDimensions)
private static void validateIndexType(TensorDescriptor indicesDescriptor)
private static void validateMatchingDataType(
        TensorDescriptor dataDescriptor, TensorDescriptor updatesDescriptor)
private static void validateReductionDataType(
        TensorDescriptor dataDescriptor, ScatterReduction reduction)
private static void validateBatchDimensions(
        Shape dataShape, Shape indicesShape, int batchDimensions)
private static void validateBatchPrefix(
        Shape dataShape, Shape indicesShape, int batchDimensions)
private static int tupleDepth(
        Shape dataShape, Shape indicesShape, int batchDimensions)
private static Shape expectedUpdatesShape(
        Shape dataShape, Shape indicesShape, int batchDimensions, int tupleDepth)
private static Tensor create(
        Tensor data,
        Tensor indices,
        Tensor updates,
        TensorDescriptor dataDescriptor,
        TensorDescriptor updatesDescriptor,
        ScatterNdAttrs attrs)
```

Add no field, nested type, alternate constructor, overload, [cache](../../../../glossary.md#term-cache), [mutable](../../../../glossary.md#term-mutable) state, or extra method.

### Delegation

The shortest helper contains exactly:

```java
return scatterNd(data, indices, updates, ScatterReduction.NONE, 0);
```

The reduction helper contains exactly:

```java
return scatterNd(data, indices, updates, reduction, 0);
```

Neither performs independent [validation](../../../../glossary.md#term-validation).

### Explicit [validation](../../../../glossary.md#term-validation) order

The five-[argument](../../../../glossary.md#term-argument) helper performs exactly:

1. null-check data, indices, updates, reduction in order with [parameter](../../../../glossary.md#term-parameter)-name messages;
2. read data, indices, updates [descriptors](../../../../glossary.md#term-descriptor) once each in order;
3. validate index type;
4. validate exact matching data/update type;
5. validate reduction/data-type compatibility;
6. read exact data, indices, updates [Shapes](../../../../glossary.md#shape) once each in order;
7. reject [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) indices with
   `scatterNd indices rank must be at least 1`;
8. create one `ScatterNdAttrs(batchDimensions, reduction)`;
9. validate [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) count fit;
10. validate shared batch prefix;
11. read/validate [tuple depth](../../../../glossary.md#term-tuple-depth) once;
12. derive expected updates Shape once;
13. require exact updates equality or throw
    `scatterNd updates shape must equal indices prefix plus data suffix: expected=<expected>, actual=<actual>`;
14. call create once.

All failures precede result [identity](../../../../glossary.md#term-identity-meanings) allocation.

### Type [validation](../../../../glossary.md#term-validation)

Indices accept only [`INT32`](../../../../glossary.md#data-type--datatype)/[`INT64`](../../../../glossary.md#data-type--datatype); otherwise:

```text
scatterNd indices data type must be INT32 or INT64: <actual>
```

Update mismatch is:

```text
scatterNd updates data type must match data: expected=<dataType>, actual=<updatesType>
```

[BOOL](../../../../glossary.md#data-type--datatype) with non-`NONE` reduction is:

```text
scatterNd BOOL data supports only NONE reduction: <reduction>
```

All floating/integral types accept all reductions; BOOL accepts `NONE` only.

### [Batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) and tuple [validation](../../../../glossary.md#term-validation)

Validate batch count against indices [rank](../../../../glossary.md#term-rank) first, then data rank:

```text
scatterNd batchDimensions must be less than indices rank: batchDimensions=<B>, indicesRank=<Q>
scatterNd batchDimensions must be less than data rank: batchDimensions=<B>, dataRank=<R>
```

Compare each shared [batch Dimension](../../../../glossary.md#batch-dimension) in increasing order; mismatch is:

```text
scatterNd batch dimension at axis <axis> must match data: expected=<dataDimension>, actual=<indicesDimension>
```

The final indices [Dimension](../../../../glossary.md#dimension) must be static. Dynamic failure is:

```text
scatterNd tuple depth must be statically known
```

Static depth must lie in `[1, data rank - batchDimensions]`; otherwise:

```text
scatterNd tuple depth must be in [1, data rank - batchDimensions]: depth=<depth>, maximum=<maximum>
```

### Expected updates [Shape](../../../../glossary.md#shape)

Derive exact Shape as indices prefix without final [tuple depth](../../../../glossary.md#term-tuple-depth) followed by data suffix beginning at
`batchDimensions + tupleDepth`. Preserve every exact [Dimension](../../../../glossary.md#dimension) reference and use `Math.addExact`
for [rank](../../../../glossary.md#term-rank) addition. Empty prefix and suffix produce [canonical](../../../../glossary.md#term-canonical-form) [scalar](../../../../glossary.md#term-scalar) Shape.

Examples inherited from task 0018I:

- data `[2,3,4]`, indices `[5,2]`, `B=0`, updates `[5,4]`;
- data `[2,3,4]`, indices `[2,5,1]`, `B=1`, updates `[2,5,4]`;
- data `[2,3]`, indices `[2]`, `B=0`, scalar updates `[]`.

### Common result construction

Create exactly one [descriptor](../../../../glossary.md#term-descriptor) with exact [data type](../../../../glossary.md#data-type--datatype), exact data [Shape](../../../../glossary.md#shape) reference, empty [layout](../../../../glossary.md#layout), and
data/update eligibility OR; one [Operation](../../../../glossary.md#operation) with `ScatterNdKind.SCATTER_ND` and exact attributes;
one [provenance](../../../../glossary.md#provenance) with exact ordered `[data, indices, updates]`; and one final
`TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` call. Add no label or
storage. Every success is fresh.

## Affected files

Expected implementation change:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorScatterNdExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterNdKind.java`
  — Javadoc only<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterNdAttrs.java`
  — Javadoc only
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScatterNdExpressionTest.java`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

At most the twelve paths above. The bound combines one public expression family, two semantic
[Javadoc](../../../../glossary.md#term-javadoc) temporal updates, public [API](../../../../glossary.md#term-application-programming-interface-api) documentation, and synchronized [planning](../../../../glossary.md#term-planning). Stop if another
production type/test, semantic behavior change, capability edit, [dependency](../../../../glossary.md#term-dependency), build/architecture
change, another [module](../../../../glossary.md#term-module), or a thirteenth path is required.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document all public/helper methods with parameters, results, constraints, ownership, identity
  side effects, and failures.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Define data, indices, updates, [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) prefix, [tuple depth](../../../../glossary.md#term-tuple-depth), [target](../../../../glossary.md#term-target), reduction, and duplicates.
- Explain delegation defaults, exact [validation](../../../../glossary.md#term-validation) order, formula, and all three examples.
- Explain type/reduction rules, exact data [Shape](../../../../glossary.md#shape)/type retention, unresolved [layout](../../../../glossary.md#layout), and eligibility.
- Explain that values/storage/bounds/duplicates/writes/reductions are not inspected/executed.
- Correct stale task-0018J temporal wording in semantic Javadocs only; behavior stays unchanged.
- Promise no [gradients](../../../../glossary.md#term-gradient), [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings), numeric order, [backend](../../../../glossary.md#backend) support, or execution.

## Acceptance criteria

- Exactly three public [Tensor](../../../../glossary.md#tensor) methods and exact one-call delegates are added.
- Helper has exact class/constructor/field/eleven-method surface and two exact delegation bodies.
- Explicit path follows exact [validation](../../../../glossary.md#term-validation)/construction order and messages.
- Type/reduction, [rank](../../../../glossary.md#term-rank), [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), prefix, tuple-depth, and updates-[Shape](../../../../glossary.md#shape) rules are exact.
- Valid result retains exact data Shape/type, eligibility OR, unresolved layout, no label/storage,
  exact semantics/provenance, and fresh identity.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- No values/storage/bounds/duplicates are inspected and no writes/reductions execute.
- Both semantic types remain behaviorally [bytecode](../../../../glossary.md#term-bytecode)-equivalent after [Javadoc](../../../../glossary.md#term-javadoc)-only changes.
- Tensor/Compile API, glossary, task/master/roadmap are independently finalized. Training API,
  capabilities, architecture, and related contracts receive reasoned no-change conclusions.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- All validation passes with exactly twelve changed paths.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run):

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.tensor.TensorScatterNdExpressionTest
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact [API](../../../../glossary.md#term-application-programming-interface-api)/helper/delegation; null/type/reduction/[rank](../../../../glossary.md#term-rank)/[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/prefix/tuple/[Shape](../../../../glossary.md#shape)
order and messages; all allowed/rejected data/index types and reductions; [scalar](../../../../glossary.md#term-scalar)/static/dynamic
Shapes; exact [descriptor](../../../../glossary.md#term-descriptor)/[provenance](../../../../glossary.md#provenance)/fresh [identity](../../../../glossary.md#term-identity-meanings); and absence of value/storage/bounds/duplicate
inspection.

Manually inspect `javap -p -c -s`, reflection/source/imports, helper method count, generated
[Javadoc](../../../../glossary.md#term-javadoc), runnable example, Markdown links/anchors/fences/whitespace, exact [scope](../../../../glossary.md#term-scope)/status, semantic
[bytecode](../../../../glossary.md#term-bytecode) equivalence, and absence of a task-0019 specification.

## Dependencies

- 0001/0002: [DataType](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape), [Dimension](../../../../glossary.md#dimension), static/dynamic facts.
- 0007/0011/0012/0013: [descriptor](../../../../glossary.md#term-descriptor), [Tensor](../../../../glossary.md#tensor), [identity](../../../../glossary.md#term-identity-meanings), [provenance](../../../../glossary.md#provenance).
- 0018F: completed [Gather-ND](../../../../glossary.md#gather-nd) [validation](../../../../glossary.md#term-validation)/Shape pattern.
- 0018G/0018H: reduction type rules and functional-scatter result boundary.
- 0018I: exact [Scatter-ND](../../../../glossary.md#scatter-nd) semantic kind and attributes.

## Follow-up tasks

- 0019: [linear](../../../../glossary.md#term-linear-meanings) algebra and attention operations, to be planned only after this task completes.
- 0023: [compiler](../../../../glossary.md#term-compiler)-generated backward semantics.

Do not create a follow-up specification here.

## Architecture impact

Expected impact: None. Stop if a new rule, [dependency](../../../../glossary.md#term-dependency), or cross-layer service is required.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0007/0011/0012/0013/0018F/0018G/0018H/0018I/
0018J, Tensor API, Compile API, Training API, glossary, current Tensor/Shape/Dimension/descriptor/
factory/provenance, Gather-ND and scatter contracts/helpers/tests, and Java 26 Gradle.

Implement task 0018J exactly. Modify Tensor.java and add package-private final
TensorScatterNdExpressions.java. Update TensorTest only for exact three-method API expansion and
add TensorScatterNdExpressionTest. Add exactly three scatterNd overloads.

The field-free helper has exactly eleven methods. Follow exact null/descriptor/index-type/
data-update-type/reduction/rank/batch-prefix/tuple-depth/updates-Shape/construction order and
messages. Default overloads delegate with NONE/B=0 or supplied reduction/B=0. Preserve exact data
Shape/type, data/update eligibility OR, unresolved layout, exact SCATTER_ND attributes, ordered
[data, indices, updates] provenance, and one createDerived call. Never inspect values.

Permit Javadoc-only current-status corrections in ScatterNdKind and ScatterNdAttrs; do not change
declarations/behavior. Do not add gradients, compiler/runtime/backend behavior, dependencies,
build/architecture changes, or later specs. Stop beyond twelve paths or on uncertainty.

Run all validation, then hand actual diff/evidence to a separate clean-context docs agent. It must
inspect source/tests/Javadoc, finalize permitted Javadocs/Tensor API/Compile API/glossary/planning,
record no-change conclusions, and rerun validation.

Update 0018J, master plan, and roadmap only for status/evidence. Do not mark Complete before both
passes. Leave 0019 Draft without a specification. Do not commit/push.
```

## Local decisions

- Three overloads preserve baseline capability without a separate [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-only overload.
- Defaults delegate directly to the full path; null never means `NONE`.
- [Validation](../../../../glossary.md#term-validation) mirrors [Gather-ND](../../../../glossary.md#gather-nd), then compares updates against the derived Gather-ND result [Shape](../../../../glossary.md#shape).
- Result keeps exact data Shape/type with unresolved [layout](../../../../glossary.md#layout) and eligibility OR.
- Bounds and duplicate validity remain value-aware responsibilities.

## Known limitations

- Index values, negative-index policy, bounds, and duplicates are not inspected.
- Numeric order, [overflow](../../../../glossary.md#term-overflow), [NaN](../../../../glossary.md#term-nan)/[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), atomics, [backend](../../../../glossary.md#backend) support, and execution are undefined.
- No [gradient](../../../../glossary.md#term-gradient) rule, [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction), [compiler](../../../../glossary.md#term-compiler)/[ONNX](../../../../glossary.md#term-onnx)/[planning](../../../../glossary.md#term-planning)/[prepare](../../../../glossary.md#prepare)/[runtime](../../../../glossary.md#term-runtime)/backend work is added.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) validation completed against repository rules, architecture, planning style, capability
baseline, completed [Gather-ND](../../../../glossary.md#gather-nd)/[axis](../../../../glossary.md#term-axis)-scatter/[Scatter-ND](../../../../glossary.md#scatter-nd) contracts, current [Tensor](../../../../glossary.md#tensor) patterns, and
read-only legacy capability/tests. Implementation validation then passed the focused Scatter-ND
suite with 10 tests, the Tensor [API](../../../../glossary.md#term-application-programming-interface-api) suite with 14 tests, and the full [model](../../../../glossary.md#term-model) suite with 735 tests
across 85 suites and no failures, errors, or skipped tests. Model [Javadoc](../../../../glossary.md#term-javadoc) and the root test task
also passed.

Independent documentation review ran in clean context
`/root/implement_model_0018j/review_model_0018j_docs`. It read the architecture and focused
architecture documents, documentation rules with the General, API/Javadoc, Planning, and Example
profiles, planning guide and roadmap, model capability/master plans, prerequisite and adjacent
tasks, current Tensor/[Compile](../../../../glossary.md#compile)/Training APIs, glossary, implementation, tests, generated Javadoc,
and Java 26 build configuration. The review finalized production Javadocs, Tensor and Compile API
status, glossary terminology, and synchronized planning evidence without changing semantic
behavior.

Post-documentation validation repeated the focused, Tensor API, full-model, model-Javadoc, and
root-test tasks. `javap` and reflection confirmed exactly three public methods, one final
package-private field-free helper, one private constructor, eleven static methods, exact delegate
[bytecode](../../../../glossary.md#term-bytecode), and the required explicit construction path. Semantic bytecode [hashes](../../../../glossary.md#term-hash) remained
`1189d3dce3752ddfd96f9f8f2205bb8a8ed78bc102b7aeb6de88732b31beca4c` for
`ScatterNdKind` and
`06eb287e0f6685e97bc42c2731cb77c2ac0f667f70a8e75839f293333eae0c22` for
`ScatterNdAttrs`. Import/source/generated-page checks and the Java 26 example passed with the
documented zero-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), batched, [scalar](../../../../glossary.md#term-scalar)-update, attribute, [metadata](../../../../glossary.md#term-metadata), [provenance](../../../../glossary.md#provenance), and freshness
output. Markdown 442-link/134-anchor and fence checks, whitespace/final-newline, exact twelve-path,
synchronized-status, and no-task-0019-specification checks also passed.

Training API needs no change because no [gradient](../../../../glossary.md#term-gradient) construction or training behavior was added.
Capabilities need no change because this task implements an already-recorded model capability.
Foundational, Gather-ND, axis-scatter, Scatter-ND semantic, and reduction contracts need no
behavioral change because this task consumes their current APIs; only the two explicitly permitted
semantic Javadocs needed temporal corrections. Architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/architecture-test changes are
unnecessary because [module](../../../../glossary.md#term-module) boundaries and dependencies did not change. [Backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) and
integration tests are unnecessary because no backend or executable behavior was added. Java 26
[Gradle](../../../../glossary.md#term-gradle)/dependencies, other modules, and later tasks remain unchanged because the implementation is
model-local and task 0019 stays Draft without a specification.

## Implementation notes

Implemented exactly three public overloads and the field-free eleven-method helper. The shared
path now performs the specified type, reduction, [rank](../../../../glossary.md#term-rank), [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-prefix, tuple-depth, and updates-[Shape](../../../../glossary.md#shape)
[validation](../../../../glossary.md#term-validation) before creating one fresh storage-free result with exact [metadata](../../../../glossary.md#term-metadata) and [provenance](../../../../glossary.md#provenance).
Focused tests cover public/helper surfaces, delegation, validation ordering/messages, accepted and
rejected domains, Shape formulas, result metadata, provenance, freshness, value/storage
non-inspection, and [identity](../../../../glossary.md#term-identity-meanings) exhaustion.

## Completion summary

Completed changes:

- added the three public `Tensor.scatterNd` overloads and one exact shared implementation helper;
- added focused expression tests and updated the exact public [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) inventory;
- corrected current semantic Javadocs without changing semantic declarations or [bytecode](../../../../glossary.md#term-bytecode); and
- finalized Tensor API, [Compile](../../../../glossary.md#compile) API, glossary, task, [model](../../../../glossary.md#term-model) master plan, and roadmap documentation.

Files changed or created: exactly the twelve paths listed under [Affected files](#affected-files).
[Validation](../../../../glossary.md#term-validation): all automated and manual checks recorded in [Validation evidence](#validation-evidence)
passed. Unresolved issues: none. Required follow-up: none for task 0018J; task 0019 remains Draft
and may be planned separately.

Status: Complete
