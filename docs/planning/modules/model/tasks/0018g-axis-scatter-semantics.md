# Task 0018G: [Axis](../../../../glossary.md#term-axis) Scatter Semantics

**Editorial reading key (later context, not a status update).**

Functional scatter describes a new data-shaped result, never in-place mutation of the receiver. Its ordered roles are base data, integer destination indices and update values; gather reads in the opposite direction. A reduction combines all addressed updates, whereas replacement NONE requires unambiguous targets. Attribute records describe axis/reduction choices but cannot inspect duplicate coordinates or index bounds without values. This historical three-kind taxonomy predates the current normalized gather/scatter names. Model owns those meanings; Compiler/backend integration and actual destination storage are separate responsibilities.

Focused current reading: [Current gather-compatible Scatter Add](../../../../reference/tensor/scatter.md#gather-compatible-scatter-add-expressions); [Current Scatter Elements reduction policy](../../../../reference/tensor/scatter.md#scatter-elements-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Define typed, [backend](../../../../glossary.md#backend)-independent semantic [identities](../../../../glossary.md#term-identity-meanings) for the three selected functional
[axis](../../../../glossary.md#term-axis)-scatter operations and one reusable [immutable](../../../../glossary.md#term-immutable) reduction vocabulary.

All three operations consume ordered logical inputs `[data, indices, updates]` and produce a new
result with the exact `data` [Shape](../../../../glossary.md#shape). They never mutate `data` in place:

- `SCATTER_ADD` adds reduced-[rank](../../../../glossary.md#term-rank) updates selected like the inverse of [`GATHER`](../../../../glossary.md#gather);
- `SCATTER_AXIS_ADD` adds rank-changing updates selected like the inverse of `GATHER_AXIS`; and
- `SCATTER_ELEMENTS` writes or reduces same-rank updates along one selected axis.

This task defines meanings and immutable [parameters](../../../../glossary.md#term-parameter) only. Public [Tensor](../../../../glossary.md#tensor) methods, input data-type
and Shape [validation](../../../../glossary.md#term-validation), [descriptors](../../../../glossary.md#term-descriptor), [provenance](../../../../glossary.md#provenance), index-value bounds and duplicate checks, [gradients](../../../../glossary.md#term-gradient),
[compiler](../../../../glossary.md#term-compiler) behavior, [lowering](../../../../glossary.md#lowering), and execution remain later responsibilities.

## [Scope](../../../../glossary.md#term-scope)

- Add one public `AxisScatterKind` enum implementing [`OperationKind`](../../../../glossary.md#operationkind) with exactly
  `SCATTER_ADD`, `SCATTER_AXIS_ADD`, and `SCATTER_ELEMENTS`, in that order.
- Add one public `ScatterReduction` enum with exactly `NONE`, `ADD`, `MUL`, `MAX`, and `MIN`, in
  that order.
- Add one public `ScatterElementsAttrs` record implementing [`OperationAttrs`](../../../../glossary.md#operationattrs) with exactly a
  non-negative normalized `int axis` and a non-null `ScatterReduction reduction`.
- Pair `SCATTER_ADD` and `SCATTER_AXIS_ADD` explicitly with the existing `IndexAxisAttrs`; their
  kind names define fixed addition and they do not carry a configurable reduction.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry))</span>
- Pair `SCATTER_ELEMENTS` explicitly with `ScatterElementsAttrs`.
- Document exact ordered `[data, indices, updates]` roles, functional result semantics, the three
  distinct Shape relationships, and reduction meanings.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- Document `NONE` as replacement semantics whose duplicate target coordinates are invalid rather
  than resolved by an unspecified update order.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Keep value-aware index bounds, duplicate detection, numerical accumulation order, and execution
  outside the semantic contracts.
- Add one focused same-package structural, validation, value-semantics, distinction, and
  composition test.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Permit [Javadoc](../../../../glossary.md#term-javadoc)-only wording changes to `IndexAxisAttrs` so its completed normalized-axis value
  accurately documents both gather and scatter pairings without changing declaration or behavior.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [gather glossary](../../../../glossary.md#gather))</span>
- Keep production in the existing `io.github.pho001.synaptik.model.operation.index` package.
- Finalize Javadocs, Tensor API semantic reference, glossary, task evidence, master plan, and
  roadmap through the mandatory independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public `Tensor.scatterAdd`, `scatterAxisAdd`, or `scatterElements` methods, overloads, factories,
  helpers, or task-0018H implementation
- scatter-ND, gather-ND backward, scalar-select backward, gather backward, take-along-axis
  backward, fold, masks, slices, or another operation family<span data-glossary-navigation="true"> ([scatter-ND glossary](../../../../glossary.md#scatter-nd); [gather-ND glossary](../../../../glossary.md#gather-nd); [scalar glossary](../../../../glossary.md#term-scalar); [gather glossary](../../../../glossary.md#gather); [masks glossary](../../../../glossary.md#term-boolean-mask); [operation glossary](../../../../glossary.md#operation))</span>
- storing data, indices, updates, input count, rank, Shape, selected extent, data type, result data
  type, descriptor, layout, requiresGrad, provenance, label, storage, or backend facts<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [extent glossary](../../../../glossary.md#term-extent); [data type glossary](../../../../glossary.md#data-type--datatype); [descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [backend glossary](../../../../glossary.md#backend))</span>
- validating [`INT32`](../../../../glossary.md#data-type--datatype) or [`INT64`](../../../../glossary.md#data-type--datatype) index tensors, matching data/update types, floating eligibility,
  BOOL restrictions, ranks, Shapes, broadcasting, bounds, duplicate indices, or result metadata<span data-glossary-navigation="true"> ([tensors glossary](../../../../glossary.md#tensor); [BOOL glossary](../../../../glossary.md#data-type--datatype); [broadcasting glossary](../../../../glossary.md#broadcasting); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- [axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization) from raw negative syntax or validating an axis against a concrete [rank](../../../../glossary.md#term-rank)
- applying a configurable reduction to `SCATTER_ADD` or `SCATTER_AXIS_ADD`; both are intrinsically
  additive semantic identities<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>
- treating null reduction as `NONE`, adding an implicit default, or using null as semantic state
- defining iteration order, floating-point accumulation order, NaN behavior, signed-zero behavior,
  overflow behavior, empty-domain behavior, atomicity, determinism across backends, or kernels<span data-glossary-navigation="true"> ([floating-point glossary](../../../../glossary.md#term-floating-point); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [atomicity glossary](../../../../glossary.md#term-atomicity); [kernels glossary](../../../../glossary.md#kernel))</span>
- adding operation arity, result-kind, costs, fusion, backend support, routes, traits, registries,
  factories, visitors, parsers, maps, aliases, or reflective discovery<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [fusion glossary](../../../../glossary.md#term-fusion); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- gradients, graph capture, canonicalization, compiler, planning, prepare, runtime, backend,
  engine, trace, ONNX implementation, training, or execution behavior<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- changing existing Java behavior/tests, dependencies, Gradle, architecture, another module, or
  creating a task-0018H specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

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
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0018C](0018c-axis-gather-semantics.md)
- [Task 0018D](0018d-axis-gather-tensor-expressions.md)
- [Task 0018E](0018e-gather-nd-semantics.md)
- [Task 0018F](0018f-gather-nd-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only legacy branch exposes three distinct public [axis](../../../../glossary.md#term-axis)-scatter capabilities:

```java
Tensor scatterAdd(Tensor indices, Tensor updates, int axis)
Tensor scatterAxisAdd(Tensor indices, Tensor updates, int axis)
Tensor scatterElements(Tensor indices, Tensor updates, int axis, ScatterReduction reduction)
```

Legacy `scatterAdd` is the fixed-add counterpart of reduced-[rank](../../../../glossary.md#term-rank) [`gather`](../../../../glossary.md#gather). Its indices and updates
have the data [Shape](../../../../glossary.md#shape) with the selected axis removed. One index value at each reduced [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)
chooses the destination coordinate on that axis, and repeated destination coordinates accumulate.

Legacy `scatterAxisAdd` is the fixed-add counterpart of [ONNX](../../../../glossary.md#term-onnx)-style `gatherAxis`. Its updates have
the Shape that `gatherAxis(data, indices, axis)` would produce: the complete indices Shape replaces
the selected data axis. Repeated destinations accumulate into the result.

Legacy `scatterElements` uses same-rank indices and updates with equal Shapes. Non-axis [Dimensions](../../../../glossary.md#dimension)
match `data`; the selected indices Dimension may differ from the selected data Dimension. It
supports replacement plus `ADD`, `MUL`, `MAX`, and `MIN` reductions. The selected baseline keeps
legacy rejection of duplicate [targets](../../../../glossary.md#term-target) for replacement semantics so result meaning does not depend
on an unspecified update order.

The new [model](../../../../glossary.md#term-model) preserves these public mathematical meanings while replacing nullable defaults and
legacy [operation](../../../../glossary.md#operation) classes with explicit [immutable](../../../../glossary.md#term-immutable) typed values. Legacy [mutable](../../../../glossary.md#term-mutable) Shapes, [graph](../../../../glossary.md#graph)
builders, [gradient](../../../../glossary.md#term-gradient) callbacks, traits, [lowering](../../../../glossary.md#lowering), [kernels](../../../../glossary.md#kernel), and [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) behavior are not
copied.

## Architecture constraints

- Production remains in `modules/model`, which owns [backend](../../../../glossary.md#backend)-neutral [operation](../../../../glossary.md#operation) meanings.
- `AxisScatterKind` identifies mathematical functional-scatter meaning only, not an occurrence,
  Tensor, graph node, descriptor, executable, kernel, or backend route.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [descriptor glossary](../../../../glossary.md#term-descriptor); [kernel glossary](../../../../glossary.md#kernel); [backend route glossary](../../../../glossary.md#backend-route))</span>
- Every kind has ordered logical inputs `[data, indices, updates]`; semantic attributes store none
  of those operands.
- Every conceptual result starts from `data`, applies writes or reductions, retains the exact data
  Shape, and is a new value. These contracts do not mutate data or execute that transformation.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- `IndexAxisAttrs.axis` and `ScatterElementsAttrs.axis` are already normalized and non-negative.
  Neither stores rank or proves that its axis exists for a particular input.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [axis glossary](../../../../glossary.md#term-axis))</span>
- `SCATTER_ADD` and `SCATTER_AXIS_ADD` use existing `IndexAxisAttrs`; addition is intrinsic to the
  semantic kind and must not be duplicated as configurable attribute state.
- `SCATTER_ELEMENTS` uses `ScatterElementsAttrs` because its reduction is caller-selected semantic
  state in addition to its normalized axis.<span data-glossary-navigation="true"> ([normalized axis glossary](../../../../glossary.md#normalized-axis))</span>
- `ScatterReduction` is reusable semantic vocabulary. Task 0018I may use it for scatter-ND without
  introducing another reduction enum.<span data-glossary-navigation="true"> ([scatter-ND glossary](../../../../glossary.md#scatter-nd))</span>
- Shape relationships and reduction descriptions are explanatory contracts used by task 0018H;
  these types perform no input-aware Shape or data-type validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Generic `Operation` remains an open kind/attributes pair and does not validate family pairing,
  arity, ranks, Shapes, data types, bounds, duplicates, gradients, or backend support.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Package direction is `model.operation.index -> model.operation + java.base` only.
- Stop if implementation requires Tensor, Shape, DataType, provenance, another production type,
  another test, dependency, architecture change, or cross-layer behavior.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [provenance glossary](../../../../glossary.md#provenance); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.operation.index.AxisScatterKind` — exact three-way semantic
  vocabulary for functional axis scatter.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- `io.github.pho001.synaptik.model.operation.index.ScatterReduction` — reusable immutable
  replacement/reduction vocabulary for functional scatter families.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- `io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs` — normalized axis and
  explicit reduction for <span data-glossary-navigation="true"> ([normalized axis glossary](../../../../glossary.md#normalized-axis))</span>`SCATTER_ELEMENTS` only.
- `io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs` — existing normalized axis
  contract reused unchanged by the two fixed-add kinds; only Javadoc wording may change.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `AxisScatterSemanticsTest` — same-package focused structural and semantic-composition test.

The existing index-[operation](../../../../glossary.md#operation) package remains cohesive and independent of public [Tensor](../../../../glossary.md#tensor) and [graph](../../../../glossary.md#graph)
packages.

## Required contract

### Semantic kind vocabulary

Create exactly:

```java
public enum AxisScatterKind implements OperationKind {
    SCATTER_ADD,
    SCATTER_AXIS_ADD,
    SCATTER_ELEMENTS
}
```

The enum declares no project field, explicit constructor, method, nested type, per-[constant](../../../../glossary.md#term-constant) body,
[alias](../../../../glossary.md#aliasing-and-arena), [arity](../../../../glossary.md#term-arity), [axis](../../../../glossary.md#term-axis), reduction, [Shape](../../../../glossary.md#shape), [data type](../../../../glossary.md#data-type--datatype), result, cost, [fusion](../../../../glossary.md#term-fusion), route, or [backend](../../../../glossary.md#backend) [metadata](../../../../glossary.md#term-metadata).
Inherited `Enum.name()` satisfies `OperationKind.name()`.

Document exact meanings:

| Kind | Ordered logical inputs | Conceptual Shape relationship | Attributes |
|---|---|---|---|
| `SCATTER_ADD` | `[data, indices, updates]` | `indices.shape == updates.shape == remove(data.shape, axis)`; fixed addition produces `data.shape` | `IndexAxisAttrs` |
| `SCATTER_AXIS_ADD` | `[data, indices, updates]` | `updates.shape == gatherAxisResultShape(data.shape, indices.shape, axis)`; fixed addition produces `data.shape` | `IndexAxisAttrs` |
| `SCATTER_ELEMENTS` | `[data, indices, updates]` | indices and updates have equal [rank](../../../../glossary.md#term-rank)/Shape, match data off-axis, and reduce into `data.shape` | `ScatterElementsAttrs` |

The Javadocs must state that the table explains meaning rather than performing [validation](../../../../glossary.md#term-validation). They
must distinguish the three kinds from axis [gather](../../../../glossary.md#gather), [Gather-ND](../../../../glossary.md#gather-nd), [scatter-ND](../../../../glossary.md#scatter-nd), and in-place mutation.

### Reduction vocabulary

Create exactly:

```java
public enum ScatterReduction {
    NONE,
    ADD,
    MUL,
    MAX,
    MIN
}
```

The enum declares no project field, explicit constructor, method, nested type, per-[constant](../../../../glossary.md#term-constant) body,
[alias](../../../../glossary.md#aliasing-and-arena), [identity](../../../../glossary.md#term-identity-meanings) value, operator function, numeric type, or [backend](../../../../glossary.md#backend) [metadata](../../../../glossary.md#term-metadata).

Document the conceptual role of each value:

- `NONE` replaces the base value at each addressed target and requires target coordinates within
  one operation to be unique; duplicate targets are invalid and are not resolved by update order.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [operation glossary](../../../../glossary.md#operation))</span>
- `ADD` combines the base value and all updates addressed to a [target](../../../../glossary.md#term-target) by addition.
- `MUL` combines the base value and all updates addressed to a target by multiplication.
- `MAX` combines the base value and all updates addressed to a target by maximum.
- `MIN` combines the base value and all updates addressed to a target by minimum.

These descriptions define semantic operators, not execution algorithms, accumulation order,
numeric edge behavior, supported DataTypes, or backend capabilities.

### Scatter-[elements](../../../../glossary.md#term-element) attributes

Create exactly:

```java
public record ScatterElementsAttrs(int axis, ScatterReduction reduction)
        implements OperationAttrs
```

The record has exactly two components in that order, one [canonical](../../../../glossary.md#term-canonical-form) constructor, two explicit
documented accessors, and record-generated `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), and `toString`. Add no [rank](../../../../glossary.md#term-rank),
selected [extent](../../../../glossary.md#term-extent), raw [axis](../../../../glossary.md#term-axis), [normalization](../../../../glossary.md#term-normalization) flag, input, [Shape](../../../../glossary.md#shape), result, default constructor, factory,
builder, nested type, or extra state/[API](../../../../glossary.md#term-application-programming-interface-api).

Constructor [validation](../../../../glossary.md#term-validation) order and behavior are exact:

1. if `axis < 0`, throw `IllegalArgumentException` with exact message
   `axis must be non-negative: <axis>`;
2. require non-null `reduction` with `Objects.requireNonNull(reduction, "reduction")`;
3. otherwise retain both values unchanged.

Zero and `Integer.MAX_VALUE` axes are structurally valid because no input rank is present. Every
enum reduction is structurally valid. Null never means `NONE`.

### Fixed-add attribute reuse

Compose both fixed-add kinds with the existing normalized-[axis](../../../../glossary.md#term-axis) record:

```java
IndexAxisAttrs axis = new IndexAxisAttrs(1);
Operation scatterAdd = new Operation(AxisScatterKind.SCATTER_ADD, axis);
Operation scatterAxisAdd = new Operation(AxisScatterKind.SCATTER_AXIS_ADD, axis);
```

Do not add reduction state to either composition. Update `IndexAxisAttrs` [Javadoc](../../../../glossary.md#term-javadoc) only so it names
the completed axis-[gather](../../../../glossary.md#gather) and fixed-add axis-scatter pairings. Its record declaration, constructor,
[validation](../../../../glossary.md#term-validation), accessor [bytecode](../../../../glossary.md#term-bytecode), value semantics, and behavior must remain unchanged.

### Configurable composition

Compose scatter-[elements](../../../../glossary.md#term-element) explicitly:

```java
ScatterElementsAttrs attrs =
        new ScatterElementsAttrs(1, ScatterReduction.ADD);
Operation scatterElements =
        new Operation(AxisScatterKind.SCATTER_ELEMENTS, attrs);
```

The exact attributes reference is retained. Generic [`Operation`](../../../../glossary.md#operation) does not enforce family pairing.
Add no operation factory, default reduction, compatibility validator, or [matrix](../../../../glossary.md#term-matrix-meanings).

### Concrete [Shape](../../../../glossary.md#shape) examples

Javadocs and focused test terminology must use these non-executable examples:

- `SCATTER_ADD`: data `[2, 3, 4]`, [axis](../../../../glossary.md#term-axis) `1`, indices `[2, 4]`, and updates `[2, 4]` produce result
  `[2, 3, 4]`. At reduced [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `[0, 2]`, index `1` adds update `[0, 2]` to data coordinate
  `[0, 1, 2]`.
- `SCATTER_AXIS_ADD`: data `[2, 3, 4]`, axis `1`, indices `[5, 6]`, and updates `[2, 5, 6, 4]`
  produce result `[2, 3, 4]`. Update `[0, i, j, 2]` [targets](../../../../glossary.md#term-target) data
  `[0, indices[i, j], 2]`.
- `SCATTER_ELEMENTS`: data `[2, 3, 4]`, axis `1`, and equal indices/updates Shapes `[2, 5, 4]`
  produce result `[2, 3, 4]`. At update coordinate `[0, 4, 2]`, the corresponding index chooses
  the middle coordinate of data target `[0, index, 2]`; the selected reduction determines how the
  update combines with the base and any repeated target.

No production code in this task stores operands, computes Shapes, reads indices, or executes these
examples.

## Affected files

Expected implementation change:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/AxisScatterKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterReduction.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterElementsAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/IndexAxisAttrs.java`
  — Javadoc only<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/index/AxisScatterSemanticsTest.java`
- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the ten paths listed above.

If implementation requires another production type, another test, behavioral changes to existing
Java, public [Tensor](../../../../glossary.md#tensor) behavior, [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) change, capability-baseline edit, [dependency](../../../../glossary.md#term-dependency), build
change, architecture document, another [module](../../../../glossary.md#term-module), or more than ten paths, stop and report the issue.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document every public type, enum [constant](../../../../glossary.md#term-constant), record component/accessor, and [canonical](../../../../glossary.md#term-canonical-form) constructor.
- Define functional scatter, base data, indices, updates, target coordinate, duplicate target, and
  reduction before relying on those terms.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Explain `[data, indices, updates]` order and exact result-[Shape](../../../../glossary.md#shape) preservation.
- Explain all three distinct Shape relationships with the concrete examples above.
- Explain [normalized axis](../../../../glossary.md#normalized-axis) meaning and why [rank](../../../../glossary.md#term-rank) [validation](../../../../glossary.md#term-validation) is deferred.
- Explain fixed-add kinds versus configurable scatter-[elements](../../../../glossary.md#term-element) reduction.
- Explain every reduction without promising an execution algorithm or [backend](../../../../glossary.md#backend) behavior.
- State that `NONE` duplicate targets are invalid but value-aware detection occurs after model
  metadata construction.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- State that only task 0018H will validate index types, input/update DataTypes, Shapes, and public
  caller axes.
- Distinguish [axis](../../../../glossary.md#term-axis) scatter from axis [gather](../../../../glossary.md#gather), [Gather-ND](../../../../glossary.md#gather-nd), [scatter-ND](../../../../glossary.md#scatter-nd), fold, and in-place mutation.
- Do not promise gradients, compiler capture, backend support, numerical execution, index bounds,
  materialization, or a particular accumulation order.<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [materialization glossary](../../../../glossary.md#materialization))</span>

## Acceptance criteria

- `AxisScatterKind` is a public enum implementing [`OperationKind`](../../../../glossary.md#operationkind) with exactly `SCATTER_ADD`,
  `SCATTER_AXIS_ADD`, and `SCATTER_ELEMENTS`, in that order.
- `ScatterReduction` is a public enum with exactly `NONE`, `ADD`, `MUL`, `MAX`, and `MIN`, in that
  order.
- Neither enum adds project-declared state, methods, constructors, nested types, aliases, or
  metadata.<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- `ScatterElementsAttrs` is a public record implementing [`OperationAttrs`](../../../../glossary.md#operationattrs) with exactly `int axis`
  and `ScatterReduction reduction`, in that order.
- Axis and null-reduction failures use the exact validation order, exception types, and messages;
  valid values are unchanged.<span data-glossary-navigation="true"> ([Axis glossary](../../../../glossary.md#term-axis))</span>
- Record-generated value semantics remain the object contract and both accessors are documented.
- Both fixed-add kinds compose with unchanged `IndexAxisAttrs`; scatter-[elements](../../../../glossary.md#term-element) composes with
  `ScatterElementsAttrs` for every exact reduction.
- `IndexAxisAttrs` declaration and behavior remain bytecode-equivalent; only its Javadoc may
  change.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Javadocs preserve ordered inputs, functional result meaning, Shape distinctions, examples,
  reduction meanings, duplicate boundary, and deferred validation.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- No Tensor, DataType, Shape, descriptor, provenance, graph, gradient, compiler, runtime, backend,
  or execution behavior is introduced.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [DataType glossary](../../../../glossary.md#data-type--datatype); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- Tensor API, glossary, task evidence, master plan, and roadmap are independently reviewed and
  synchronized. Compile API, Training API, capabilities, architecture, and related contracts
  receive reasoned no-change conclusions.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>
- All [validation](../../../../glossary.md#term-validation) passes and the final diff contains exactly the ten permitted paths.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run):

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.operation.index.AxisScatterSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must verify:

- exact kind-enum [constant](../../../../glossary.md#term-constant) count, names, order, interface, and absence of project [API](../../../../glossary.md#term-application-programming-interface-api)/state;
- exact reduction-enum constant count, names, order, and absence of project API/state;
- exact record status, component names/types/order, interface, accessors, and absence of extra
  state/API;
- zero, ordinary, and `Integer.MAX_VALUE` [axis](../../../../glossary.md#term-axis) retention with every reduction;
- exact negative-axis-first validation and message, followed by exact null-reduction failure and
  message;
- record-generated equality, hashing, and diagnostic text;
- exact composition of fixed-add kinds with one `IndexAxisAttrs` reference;
- exact composition of `SCATTER_ELEMENTS` with one `ScatterElementsAttrs` reference for every
  reduction;
- kind [identity](../../../../glossary.md#term-identity-meanings) distinctions from each other, axis-[gather](../../../../glossary.md#gather) kinds, and [Gather-ND](../../../../glossary.md#gather-nd);
- absence of [scatter-ND](../../../../glossary.md#scatter-nd), [gradient](../../../../glossary.md#term-gradient)-only, gather-gradient, [alias](../../../../glossary.md#aliasing-and-arena), or default enum constants;
- absence of [Tensor](../../../../glossary.md#tensor), [DataType](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout), [graph](../../../../glossary.md#graph), [compiler](../../../../glossary.md#term-compiler), [runtime](../../../../glossary.md#term-runtime), and [backend](../../../../glossary.md#backend) dependencies.

Manually inspect `javap -p -c -s`, reflection/source/imports, generated [Javadoc](../../../../glossary.md#term-javadoc), Markdown links,
anchors, fences, whitespace, exact ten-path [scope](../../../../glossary.md#term-scope), synchronized task/master/roadmap status,
[bytecode](../../../../glossary.md#term-bytecode)-equivalent `IndexAxisAttrs`, and absence of a task-0018H specification.

## Dependencies

- Task 0005 defines the minimal [operation](../../../../glossary.md#operation)-kind and typed-attributes contracts.
- Task 0006 defines the open [immutable](../../../../glossary.md#term-immutable) `Operation` pair.
- Task 0018C defines `IndexAxisAttrs` and the three gather meanings that the fixed-add scatter
  kinds invert at the semantic level.<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather))</span>
- Task 0018D supplies the completed public gather Shape terminology used to explain the inverse
  relationships without adding those public methods here.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>

## Follow-up tasks

- 0018H: public scatter-add, scatter-[axis](../../../../glossary.md#term-axis)-add, and scatter-[elements](../../../../glossary.md#term-element) [Tensor](../../../../glossary.md#tensor) expressions.
- 0018I: [scatter-ND](../../../../glossary.md#scatter-nd) semantics reusing `ScatterReduction` with [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[dimension](../../../../glossary.md#dimension) [parameters](../../../../glossary.md#term-parameter).

Do not create either follow-up specification during this task.

## Architecture impact

Expected impact: None.

The task fills the existing [model](../../../../glossary.md#term-model)-owned [operation](../../../../glossary.md#operation) vocabulary. If implementation requires a new
architecture rule or cross-[module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), stop and report the issue.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0005/0006/0018C/0018D/0018E/0018F/0018G, Tensor API,
Compile API, Training API, glossary, current OperationKind/OperationAttrs/Operation/IndexAxisAttrs/
AxisGatherKind/GatherNdKind and related index-family contracts/tests, and Java 26 Gradle.

Implement task 0018G exactly. Add only AxisScatterKind.java, ScatterReduction.java,
ScatterElementsAttrs.java, and AxisScatterSemanticsTest.java for new Java files under
io.github.pho001.synaptik.model.operation.index. Permit Javadoc-only changes to IndexAxisAttrs;
its declaration and behavior must remain bytecode-equivalent.

AxisScatterKind contains exactly SCATTER_ADD, SCATTER_AXIS_ADD, and SCATTER_ELEMENTS in order.
ScatterReduction contains exactly NONE, ADD, MUL, MAX, and MIN in order. ScatterElementsAttrs
contains exactly normalized non-negative int axis then non-null ScatterReduction reduction, with
the exact validation order/messages and explicit documented accessors. Fixed-add kinds pair with
IndexAxisAttrs; configurable scatter-elements pairs with ScatterElementsAttrs. Document ordered
[data, indices, updates], functional data-shaped results, three distinct Shape relationships,
fixed-add versus configurable reduction, all reduction meanings, and invalid NONE duplicates.

Do not add Tensor methods, Shape/DataType/result/provenance validation, scatter-ND/gradient types,
value execution, numeric policy, factories, graph/compiler/planning/runtime/backend behavior,
dependencies, build or architecture changes, behavioral existing-Java edits, or later specs. Stop
beyond ten paths or on architecture uncertainty.

Run all specified validation, then hand the actual diff/evidence to a separate clean-context docs
agent in the same change. It must inspect source/tests/generated Javadoc, finalize permitted
Javadocs/Tensor API/glossary/planning, record Compile API/Training API/capability/architecture and
related-contract no-change conclusions, and rerun validation.

Update task 0018G, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0018H Draft without a specification. Do not commit/push.
```

## Local decisions

- Three enum constants preserve distinct update-alignment contracts rather than treating every
  axis scatter as one operation plus a mode flag.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [alignment glossary](../../../../glossary.md#term-alignment); [axis glossary](../../../../glossary.md#term-axis))</span>
- `SCATTER_ADD` and `SCATTER_AXIS_ADD` reuse `IndexAxisAttrs` because both need only an already-
  normalized target axis; their fixed addition is intrinsic to their kind names.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- `SCATTER_ELEMENTS` receives `ScatterElementsAttrs` because its reduction is caller-selected
  semantic state. Null is rejected rather than silently becoming `NONE`.
- `ScatterReduction` is a separate reusable enum because both scatter-elements and later
  scatter-ND expose the same selected replacement/reduction vocabulary.<span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element); [scatter-ND glossary](../../../../glossary.md#scatter-nd))</span>
- `NONE` means unambiguous replacement. Duplicate target coordinates are invalid rather than
  inheriting traversal-order-dependent last-write behavior.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Ordered `[data, indices, updates]` roles and Shape relationships are semantic documentation, not
  stored arity, operand, or inference state. Generic <span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [arity glossary](../../../../glossary.md#term-arity); [inference glossary](../../../../glossary.md#term-inference))</span>[`Operation`](../../../../glossary.md#operation) remains open.

## Known limitations

- The semantic values do not normalize raw caller axes or validate data/indices/update ranks,
  types, Shapes, bounds, duplicate target coordinates, dynamic Dimensions, result descriptors, or
  gradient eligibility. Task 0018H owns the locally decidable public expression rules.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [Shapes glossary](../../../../glossary.md#shape); [target glossary](../../../../glossary.md#term-target); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [Dimensions glossary](../../../../glossary.md#dimension); [descriptors glossary](../../../../glossary.md#term-descriptor); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- The reduction vocabulary defines mathematical combination choices but not numerical evaluation
  order, floating-point reproducibility, overflow, NaN/signed-zero handling, atomic execution, or
  backend support.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [floating-point glossary](../../../../glossary.md#term-floating-point); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance); [overflow glossary](../../../../glossary.md#term-overflow); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [backend glossary](../../../../glossary.md#backend))</span>
- No Tensor method, provenance, storage, graph capture, compiler transformation, gradient rule,
  ONNX mapping, planning, materialization, lowering, runtime behavior, or execution is implemented.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiler glossary](../../../../glossary.md#term-compiler); [ONNX glossary](../../../../glossary.md#term-onnx); [planning glossary](../../../../glossary.md#term-planning); [materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Scatter-ND](../../../../glossary.md#scatter-nd) remains the separate later semantic/expression pair 0018I–0018J.

## [Validation](../../../../glossary.md#term-validation) evidence

- Clean implementation context `/root/implement_model_0018g` added exactly `AxisScatterKind`,
  `ScatterReduction`, `ScatterElementsAttrs`, and `AxisScatterSemanticsTest`, plus the authorized
  Javadoc-only <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`IndexAxisAttrs` update. Independent documentation context
  `/root/implement_model_0018g/review_model_0018g_docs` inspected the actual shared-tree diff,
  final source/test, generated Javadoc, related contracts, APIs, glossary, planning state, and Java
  26 build configuration before finalizing documentation in the same overall change.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- The documentation pass applied General plus API/Javadoc style to production Javadocs, Tensor API,
  and glossary; Planning style to this task, the model master plan, and roadmap; and Example format
  to the conceptual Shape/coordinate examples. It retained the three new production Javadocs as
  complete and refined only <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor); [model glossary](../../../../glossary.md#term-model); [Shape glossary](../../../../glossary.md#shape); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`IndexAxisAttrs` wording to distinguish the current gather boundary
  from task-0018H public scatter validation. No declaration, test, or executable behavior changed.<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather))</span>
- Reviewed architecture and process material included `AGENTS.md`, `ARCHITECTURE.md`, the focused
  current architecture, overview, lifecycle, module-boundary, dependency, and
  runtime/prepare/backend explanations; documentation rules and General/API-Javadoc/Planning/
  Example profiles; planning guide and roadmap; model capabilities/master plan; tasks 0005, 0006,
  and 0018C–0018G; Tensor, Compile, and Training API references; glossary; Java 26 root/model Gradle
  configuration; final implementation/tests; and generated model Javadoc.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [Compile glossary](../../../../glossary.md#compile); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.index.AxisScatterSemanticsTest` — `BUILD SUCCESSFUL`;
  XML reports 12 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; XML aggregation reports 706 tests across
  82 suites with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; generated public pages contain all three
  new types, every enum constant, the record components, canonical constructor, explicit accessors,
  exact failures, ordered roles, functional result meaning, Shape examples, reduction meanings,
  duplicate-target boundary, task-0018H ownership, family distinctions, and exclusions.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [canonical glossary](../../../../glossary.md#term-canonical-form); [target glossary](../../../../glossary.md#term-target); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `./gradlew test` — `BUILD SUCCESSFUL`; the repository lifecycle reports 36 actionable tasks with
  no failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` confirmed exact enum order and
  generated-only enum machinery. `ScatterElementsAttrs` has exactly the two specified fields,
  negative-axis-first then non-null-reduction validation, direct accessors, and generated record
  methods. Diffing current <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>`IndexAxisAttrs` output against `/tmp/index-axis-before.javap` produced
  no output, proving declaration and executable-bytecode equivalence after its Javadoc-only edit.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Focused reflection, source, and import inspection confirmed exact packages, modifiers,
  interfaces, constants, record components/methods, validation messages, value semantics,
  kind/attributes compositions, and only the permitted local operation/JDK imports. No Tensor,
  DataType, Shape, layout, provenance, graph, compiler, planning, prepare, runtime, backend,
  training, or execution dependency or behavior was introduced.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [DataType glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Generated-page inspection and a targeted Markdown validator passed for all changed documentation:
  392 local links, including 100 heading anchors, resolved with zero errors. Fences are balanced,
  trailing-whitespace scans found no matches, all ten paths have final newlines, and
  `git diff --check` passes.
- Final changed-path inventory contains exactly the ten authorized paths: three new production
  contracts, Javadoc-only `IndexAxisAttrs`, one focused test, Tensor API, glossary, this task,
  model master plan, and roadmap. Task/master-plan/roadmap status is synchronized as Complete.
  Task 0018H remains Draft, and no task-0018H specification exists.
- Compile API remains accurate unchanged because this task adds semantic vocabulary only: it adds
  no Tensor expression, graph capture, inference, validation, optimization, artifact, or engine
  behavior. Training API remains accurate unchanged because no gradient, autograd, parameter,
  optimizer, publication, session, or training execution behavior changed. The capability
  baseline already inventories all three axis-scatter operations, the five reductions, exact
  integral index types, and separate support layers, so it required no edit.<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [inference glossary](../../../../glossary.md#term-inference); [artifact glossary](../../../../glossary.md#term-artifact); [engine glossary](../../../../glossary.md#term-engine); [gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [parameter glossary](../../../../glossary.md#term-parameter); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication))</span>
- Existing Operation foundations, aggregate/scan reductions, fold, scalar select, axis gather,
  Gather-ND, Tensor, provenance, graph, and other operation-family contracts remain accurate
  unchanged because the new values compose the open semantic contracts without altering their
  declarations or behavior. Scatter-ND and later tasks remain deferred.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [scalar select glossary](../../../../glossary.md#scalar-select); [Gather-ND glossary](../../../../glossary.md#gather-nd); [Scatter-ND glossary](../../../../glossary.md#scatter-nd))</span>
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance tests, integration tests, Gradle/dependencies, other modules, and later task
  specifications remain accurate unchanged because this task stays within existing model
  ownership and changes no dependency rule, backend behavior, executable end-to-end behavior, or
  build requirement.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>

## Implementation notes

- Added `AxisScatterKind` with exact ordered `SCATTER_ADD`, `SCATTER_AXIS_ADD`, and
  `SCATTER_ELEMENTS` [constants](../../../../glossary.md#term-constant) and no project-declared behavior or [metadata](../../../../glossary.md#term-metadata).
- Added reusable `ScatterReduction` with exact ordered `NONE`, `ADD`, `MUL`, `MAX`, and `MIN`
  values, including unambiguous replacement semantics for `NONE`.
- Added `ScatterElementsAttrs(int axis, ScatterReduction reduction)` with exact validation order,
  direct accessors, and generated record value semantics. Both fixed-add kinds reuse unchanged<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
  `IndexAxisAttrs`.
- Added 12 focused tests covering exact structure, retained values, exact failures, value
  semantics, typed composition, family distinctions, and dependency boundaries.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Finalized production Javadocs, Tensor API semantic reference, glossary terminology/inventories,
  and synchronized planning evidence/status through the independent documentation pass.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: Implemented and documented three functional axis-scatter semantic identities,
  reusable five-way reduction meaning, and explicit scatter-elements attributes without public
  Tensor or cross-layer behavior.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [elements glossary](../../../../glossary.md#term-element); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Files changed or created: Exactly the ten authorized production, test, API, glossary, task,
  master-plan, and roadmap paths.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Tests and validation: Focused 12-test and all 706-model-test/82-suite runs, model Javadoc, root
  tests, javap/reflection/import/source/generated-page review, 392-link/100-anchor checks,
  fence/whitespace/newline checks, exact scope/status and no-0018H-spec checks,<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
  `IndexAxisAttrs` [bytecode](../../../../glossary.md#term-bytecode) equivalence, and `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0018g/review_model_0018g_docs` completed the mandatory independent pass
  with General, API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and glossary now define current functional axis-scatter
  vocabulary while public Tensor construction, type/Shape/axis validation, index bounds,
  duplicate detection, gradients, compiler behavior, lowering, backend behavior, and execution
  remain separately owned.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [lowering glossary](../../../../glossary.md#lowering); [backend glossary](../../../../glossary.md#backend))</span>
- Javadoc review: All three new public types, every constant, the attributes record components,
  canonical constructor, and accessors are complete. <span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>`IndexAxisAttrs` received only an ownership-
  boundary clarification and remains bytecode-equivalent.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Glossary impact: Added reusable axis-scatter, base/indices/updates, target, duplicate-target, and
  reduction terminology and synchronized current <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>[`OperationKind`](../../../../glossary.md#operationkind)/[`OperationAttrs`](../../../../glossary.md#operationattrs) inventories.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0018G. Task 0018H remains Draft without a detailed
  specification.

Status: Complete
