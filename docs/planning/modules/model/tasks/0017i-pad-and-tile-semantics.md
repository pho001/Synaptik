# Task 0017I: Pad and Tile Semantics

**Editorial reading key (later context, not a status update).**

Padding adds constant-filled border positions; tiling repeats the complete pattern along each axis, rather than repeating each individual element consecutively. Widths, repeat counts and the historical binary64 constant are semantic parameters, not already-populated output storage. Rank matching needs the later input-aware helper. Here Model defines a typed meaning for later construction code; this semantic-family task neither receives stored operands nor evaluates them. Its completion is not evidence of Compiler capture, derivatives or backend execution.

Focused current reading: [Current pad/tile reference](../../../../reference/tensor/composition.md#pad-and-tile-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Define typed, [backend](../../../../glossary.md#backend)-independent semantic [identities](../../../../glossary.md#term-identity-meanings) and [immutable](../../../../glossary.md#term-immutable) normalized attributes for
[constant](../../../../glossary.md#term-constant) [padding](../../../../glossary.md#padding) and per-[axis](../../../../glossary.md#term-axis) [tiling](../../../../glossary.md#tiling).

Padding adds constant-filled logical positions before and after each input axis. Tiling repeats the
complete input pattern a positive number of times along every axis. This task defines [operation](../../../../glossary.md#operation)
meaning and [parameter](../../../../glossary.md#term-parameter) invariants only. Public [Tensor](../../../../glossary.md#tensor) methods, [Shape](../../../../glossary.md#shape)/data-type [validation](../../../../glossary.md#term-validation), result
[descriptors](../../../../glossary.md#term-descriptor), [provenance](../../../../glossary.md#provenance), [gradients](../../../../glossary.md#term-gradient), [materialization](../../../../glossary.md#materialization), [compiler](../../../../glossary.md#term-compiler) behavior, and execution remain in
later tasks and layers.

## [Scope](../../../../glossary.md#term-scope)

- Add one public `PadKind` enum implementing [`OperationKind`](../../../../glossary.md#operationkind) with exactly [`PAD`](../../../../glossary.md#padding).
- Add one public `TileKind` enum implementing `OperationKind` with exactly `TILE`.
- Add one public `PadAttrs` record implementing [`OperationAttrs`](../../../../glossary.md#operationattrs) with exactly, in order:
  - `List<Long> before`
  - `List<Long> after`
  - `double constantValue`
- Require equal before/after sizes, non-null [elements](../../../../glossary.md#term-element), and non-negative [padding](../../../../glossary.md#padding) widths.
- Preserve every double constant unchanged without conversion, finiteness checks, or
  normalization.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Add one public `TileAttrs` record implementing `OperationAttrs` with exactly one
  `List<Long> repeats` component.
- Require non-null repeat elements and strictly positive repeat counts.
- Accept empty padding/repeat lists as [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [scalar](../../../../glossary.md#term-scalar) [identity](../../../../glossary.md#term-identity-meanings) [parameters](../../../../glossary.md#term-parameter).
- Preserve list order and store [immutable](../../../../glossary.md#term-immutable) snapshots after [validation](../../../../glossary.md#term-validation).
- Document exact kind/attribute pairings without adding generic compatibility validation.
- Add one focused same-package semantic test for all four cohesive production contracts.
- Keep production in the existing `model.operation.layout` package.
- Finalize Javadocs, Tensor API semantic reference, glossary, task evidence, model master plan, and
  roadmap through the mandatory independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public `Tensor.pad`, `Tensor.tile`, another Tensor method, overload, expression helper, factory,
  builder, or task-0017J implementation<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- input Tensor, input/output Shape or DataType, rank validation, result dimensions, overflow
  checks, TensorDescriptor, identity, label, provenance, storage, or gradient eligibility<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [DataType glossary](../../../../glossary.md#data-type--datatype); [rank glossary](../../../../glossary.md#term-rank); [validation glossary](../../../../glossary.md#term-validation); [dimensions glossary](../../../../glossary.md#dimension); [overflow glossary](../../../../glossary.md#term-overflow); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- converting the double padding constant into FLOAT64, FLOAT32, BFLOAT16, integral, or BOOL
  values; range checks, rounding, saturation, BOOL normalization, NaN policy, or infinity policy<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [constant glossary](../../../../glossary.md#term-constant); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [rounding glossary](../../../../glossary.md#term-rounding); [saturation glossary](../../../../glossary.md#term-saturation); [normalization glossary](../../../../glossary.md#term-normalization); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- negative padding as cropping, reflect/edge/wrap padding, per-axis constants, tensor-valued
  padding, or another padding mode<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- zero/negative tile repeats, implicit leading axes, NumPy-style repeat-rank promotion, partial
  repeat lists, or a separate scalar repeat overload<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- layout/view/alias derivation, zero strides, allocation, copying, materialization, storage access,
  value population, or execution<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena); [strides glossary](../../../../glossary.md#term-stride); [materialization glossary](../../../../glossary.md#materialization))</span>
- gradients, pad backward slicing, tile backward reduction, autograd, compiler-generated
  operations, optimizer, or training behavior<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [compiler glossary](../../../../glossary.md#term-compiler); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- graph capture/canonicalization, planning, prepare, runtime, backend lowering/kernels, engine,
  trace, ONNX mapping, or conformance behavior<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- factories, registries, parsers, visitors, maps, string dispatch, reflective discovery, arity,
  result, cost, fusion, backend-support, route, or kernel metadata<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [arity glossary](../../../../glossary.md#term-arity); [fusion glossary](../../../../glossary.md#term-fusion); [kernel glossary](../../../../glossary.md#kernel); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- changing existing Java/tests, dependencies, Gradle, architecture, another module, or task-0017J
  specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

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
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0017G](0017g-slice-semantics.md)
- [Task 0017H](0017h-slice-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The selected baseline requires [constant](../../../../glossary.md#term-constant) `pad` and per-[axis](../../../../glossary.md#term-axis) `tile`. The read-only
`legacy/pre-rewrite` branch exposes:

```java
Tensor pad(int[] before, int[] after, double constantValue)
Tensor tile(int... repeats)
```

Legacy [padding](../../../../glossary.md#padding) requires one non-negative before/after width per input axis, preserves [rank](../../../../glossary.md#term-rank), adds
the widths to each result [dimension](../../../../glossary.md#dimension), and fills added positions from one double constant. Legacy
[tiling](../../../../glossary.md#tiling) requires one positive repeat per input axis and multiplies each result dimension by its
repeat. Both preserve input [data type](../../../../glossary.md#data-type--datatype); their builders also couple [graph](../../../../glossary.md#graph) construction, [Shape](../../../../glossary.md#shape),
[gradients](../../../../glossary.md#term-gradient), storage/[materialization](../../../../glossary.md#materialization), and execution concerns.

The new [model](../../../../glossary.md#term-model) preserves the semantic capability while using [immutable](../../../../glossary.md#term-immutable) `List<Long>` [parameters](../../../../glossary.md#term-parameter)
compatible with long Shape dimensions. This task stores only normalized intrinsic values. Task
0017J will own public primitive-array/varargs syntax, exact rank, checked Shape arithmetic,
data-type compatibility for the constant, result [descriptors](../../../../glossary.md#term-descriptor), and [provenance](../../../../glossary.md#provenance).

Legacy [mutable](../../../../glossary.md#term-mutable) arrays, immediate graph/storage objects, gradient callbacks, traits,
materialization choices, [lowering](../../../../glossary.md#lowering), [kernels](../../../../glossary.md#kernel), and [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) behavior are not copied.

## Architecture constraints

- Production remains in `modules/model`, which owns [backend](../../../../glossary.md#backend)-neutral [operation](../../../../glossary.md#operation) semantics.
- `PadKind` and `TileKind` identify logical meaning only; they are not Tensors, graph occurrences,
  descriptors, layouts, executables, or backend routes.<span data-glossary-navigation="true"> ([Tensors glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [descriptors glossary](../../../../glossary.md#term-descriptor); [layouts glossary](../../../../glossary.md#layout))</span>
- Attributes store normalized intrinsic parameters only, never input rank, Shape, DataType,
  result Shape, layout, storage, provenance, or runtime facts.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape); [DataType glossary](../../../../glossary.md#data-type--datatype); [provenance glossary](../../../../glossary.md#provenance); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Pad list index `i` couples a before and after width for the same [normalized axis](../../../../glossary.md#normalized-axis) position.
- Padding widths are non-negative. The double constant is retained as supplied and receives no
  data-type interpretation in this task.<span data-glossary-navigation="true"> ([Padding glossary](../../../../glossary.md#padding); [constant glossary](../../../../glossary.md#term-constant))</span>
- Tile list index `i` gives the positive number of complete input-pattern repetitions along axis
  position <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>`i`.
- Empty lists are structurally valid scalar identity parameters. Eventual input-rank matching is
  a task-0017J responsibility.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Generic `Operation` remains an open kind/attributes pair and does not validate family pairing,
  arity, rank, Shape, data type, gradients, or backend support.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [data type glossary](../../../../glossary.md#data-type--datatype); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Package direction is `model.operation.layout -> model.operation + java.base` only.
- Stop if implementation needs DataType/Shape state, another semantic type, existing Java edits,
  a dependency, or an architecture decision.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.operation.layout.PadKind` — [constant](../../../../glossary.md#term-constant)-[padding](../../../../glossary.md#padding) [identity](../../../../glossary.md#term-identity-meanings).
- `io.github.pho001.synaptik.model.operation.layout.PadAttrs` — immutable normalized per-axis
  widths and raw semantic constant.
- `io.github.pho001.synaptik.model.operation.layout.TileKind` — per-[axis](../../../../glossary.md#term-axis) [tiling](../../../../glossary.md#tiling) identity.
- `io.github.pho001.synaptik.model.operation.layout.TileAttrs` — [immutable](../../../../glossary.md#term-immutable) positive repeat counts.
- `PadTileSemanticsTest` — same-package structural, [validation](../../../../glossary.md#term-validation), [ownership](../../../../glossary.md#term-ownership-meanings), and composition test.

Both operations transform logical [layout](../../../../glossary.md#layout)/[extent](../../../../glossary.md#term-extent) and therefore remain cohesive with existing
[contiguous](../../../../glossary.md#term-contiguous), [shape](../../../../glossary.md#shape)-transform, axis-transform, and [slice](../../../../glossary.md#slice) semantics in `model.operation.layout`.

## Required contract

### Semantic kind vocabularies

Create exactly:

```java
public enum PadKind implements OperationKind {
    PAD
}

public enum TileKind implements OperationKind {
    TILE
}
```

Each enum declares no project field, explicit constructor, method, nested type, per-[constant](../../../../glossary.md#term-constant) body,
[alias](../../../../glossary.md#aliasing-and-arena), [arity](../../../../glossary.md#term-arity), [parameter](../../../../glossary.md#term-parameter), [Shape](../../../../glossary.md#shape), [DataType](../../../../glossary.md#data-type--datatype), result, [layout](../../../../glossary.md#layout), or [backend](../../../../glossary.md#backend) [metadata](../../../../glossary.md#term-metadata). Inherited
`Enum.name()` satisfies `OperationKind.name()`.

[`PAD`](../../../../glossary.md#padding) means one logical input with constant-filled positions inserted before/after every [axis](../../../../glossary.md#term-axis)
without changing [rank](../../../../glossary.md#term-rank). `TILE` means one logical input whose complete pattern is repeated along
each axis without changing rank. Neither kind calculates output extents or values.

### [Padding](../../../../glossary.md#padding) attributes

Create exactly:

```java
public record PadAttrs(
        List<Long> before,
        List<Long> after,
        double constantValue) implements OperationAttrs
```

The record has exactly three components in that order, one public [canonical](../../../../glossary.md#term-canonical-form) constructor, three
explicit documented accessors, and record-generated object methods. Add no arrays, factory,
builder, [rank](../../../../glossary.md#term-rank), size method, padding mode, [DataType](../../../../glossary.md#data-type--datatype), converted [scalar](../../../../glossary.md#term-scalar), [Shape](../../../../glossary.md#shape), [cache](../../../../glossary.md#term-cache), nested type, or
extra [API](../../../../glossary.md#term-application-programming-interface-api)/state.

Constructor [validation](../../../../glossary.md#term-validation) order is exact:

1. null-check `before`, then `after`, with exact component-name messages;
2. reject unequal sizes with `IllegalArgumentException` and exact message
   `before and after must have matching sizes`;
3. inspect entries in ascending index order;
4. at each index null-check before then after with exact messages `before[<index>]` and
   `after[<index>]`;
5. reject negative before with exact message
   `before[<index>] must be non-negative: <value>`;
6. reject negative after with exact message
   `after[<index>] must be non-negative: <value>`;
7. after every entry passes, assign `before = List.copyOf(before)` then
   `after = List.copyOf(after)`, exactly once each.

Perform no validation or [normalization](../../../../glossary.md#term-normalization) of `constantValue`. Retain the supplied primitive exactly,
including [finite](../../../../glossary.md#term-finite) values, [signed zero](../../../../glossary.md#term-signed-zero), [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and [NaN](../../../../glossary.md#term-nan). The accessor must expose the retained
primitive; data-type conversion is deferred. Record-generated equality/[hash](../../../../glossary.md#term-hash) semantics remain
ordinary Java record/double semantics and are not redefined as a bitwise [serialization](../../../../glossary.md#term-serialization) contract.

The two lists preserve caller order but not [identity](../../../../glossary.md#term-identity-meanings). Caller mutation after construction cannot
affect stored values and accessor mutation fails. Empty lists are valid. `Long.MAX_VALUE` padding
is structurally valid because output Shape arithmetic is absent here.

### [Tiling](../../../../glossary.md#tiling) attributes

Create exactly:

```java
public record TileAttrs(List<Long> repeats) implements OperationAttrs
```

The record has one component, one public [canonical](../../../../glossary.md#term-canonical-form) constructor, one explicit documented accessor,
and generated object methods. Add no primitive-array overload, factory, builder, [rank](../../../../glossary.md#term-rank), [Shape](../../../../glossary.md#shape),
[cache](../../../../glossary.md#term-cache), nested type, or extra [API](../../../../glossary.md#term-application-programming-interface-api)/state.

Constructor [validation](../../../../glossary.md#term-validation) order is exact:

1. null-check `repeats` with exact message `repeats`;
2. inspect entries in ascending index order;
3. null-check each [element](../../../../glossary.md#term-element) with exact message `repeats[<index>]`;
4. reject zero or negative values with exact message
   `repeats[<index>] must be positive: <value>`;
5. after every entry passes, assign `repeats = List.copyOf(repeats)` exactly once.

Preserve order and values but not caller list [identity](../../../../glossary.md#term-identity-meanings). Empty list is valid [scalar](../../../../glossary.md#term-scalar) identity;
`Long.MAX_VALUE` is structurally valid. Result Shape multiplication and [overflow](../../../../glossary.md#term-overflow) are deferred.

### Typed composition

Document exactly these valid pairings:

```java
Operation padded = new Operation(PadKind.PAD, padAttrs);
Operation tiled = new Operation(TileKind.TILE, tileAttrs);
```

[Operation](../../../../glossary.md#operation) retains exact kind and attribute references. Do not use [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), pair a kind
with the other family attributes, add compatibility validators/factories, or change Operation.
The valid pairs are documented rather than generically enforced.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/PadKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/PadAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/TileKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/TileAttrs.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/layout/PadTileSemanticsTest.java`

Documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistency requires stopping: [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api), Training API,
capabilities, [DataType](../../../../glossary.md#data-type--datatype)/[Shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout)/[Tensor](../../../../glossary.md#tensor) contracts, [Operation](../../../../glossary.md#operation) foundations, existing semantic
families/tests, architecture/ADRs/tests, [conformance](../../../../glossary.md#term-conformance)/integration tests, [Gradle](../../../../glossary.md#term-gradle), and other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

At most four production files, one focused test, and five documentation/[planning](../../../../glossary.md#term-planning) files: ten paths.
If another Java type/test, existing Java edit, [dependency](../../../../glossary.md#term-dependency), build/architecture change, or eleventh
path is required, stop and report. Do not create task 0017J.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document both enums, every constant, both records, canonical constructors, components, and all
  explicit accessors.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Explain [constant](../../../../glossary.md#term-constant) [padding](../../../../glossary.md#padding) with a concrete one-dimensional example: input `[10, 20]`, before `1`,
  after `2`, constant `-1` semantically requests `[-1, 10, 20, -1, -1]`, without claiming
  execution exists.
- Explain two-dimensional [tiling](../../../../glossary.md#tiling) with input `[[1, 2], [3, 4]]` and repeats `[2, 3]`, distinguishing
  complete-pattern repetition from scalar element repeat, without claiming values are computed.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [element glossary](../../../../glossary.md#term-element))</span>
- Explain list ownership, ordering, empty scalar identity, extreme accepted values, and deferred
  rank/Shape arithmetic.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [identity glossary](../../../../glossary.md#term-identity-meanings); [rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape))</span>
- Explain raw double constant retention and deferred DataType conversion, including signed zero,
  NaN, and infinity.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [signed zero glossary](../../../../glossary.md#term-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- Document exact [validation](../../../../glossary.md#term-validation) order, [parameters](../../../../glossary.md#term-parameter), results, null failures, and value failures.
- Explain that no Tensor, Shape/result inference, layout, storage, materialization, provenance,
  gradient, compiler, backend, ONNX, or execution behavior is defined.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [inference glossary](../../../../glossary.md#term-inference); [layout glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization); [provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- Review related Operation, DataType, Shape, SliceAttrs, and layout-semantic Javadocs; record why
  unchanged contracts remain accurate or stop on discrepancy.<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation))</span>

## Acceptance criteria

- Both public enums have exactly their specified one [constant](../../../../glossary.md#term-constant) and no extra project [API](../../../../glossary.md#term-application-programming-interface-api)/state.
- PadAttrs is exactly the specified three-component record; TileAttrs is exactly the specified
  one-component record; neither adds extra API/state.
- Empty and ordinary lists are accepted; caller order/values are retained.
- Null containers/elements, mismatched padding sizes, negative widths, and non-positive repeats
  fail with exact type/message/precedence.<span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element); [padding glossary](../../../../glossary.md#padding))</span>
- Pad lists receive exactly two post-[validation](../../../../glossary.md#term-validation) snapshots; repeats receives exactly one.
- Caller/accessor mutation cannot alter stored values.
- Every double constant and extreme valid long is retained without [Shape](../../../../glossary.md#shape)/[DataType](../../../../glossary.md#data-type--datatype) interpretation.
- Exact typed [Operation](../../../../glossary.md#operation) compositions retain references; no validator/factory is added.
- No [Tensor](../../../../glossary.md#tensor)/Shape/[layout](../../../../glossary.md#layout)/[provenance](../../../../glossary.md#provenance)/[gradient](../../../../glossary.md#term-gradient)/cross-layer behavior or existing Java edit appears.
- Complete Javadocs, Tensor API, glossary, independent docs review, evidence, and statuses finish
  before marking Complete.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.layout.PadTileSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact enum order/[API](../../../../glossary.md#term-application-programming-interface-api); exact record components/order/types/API; empty/[scalar](../../../../glossary.md#term-scalar) and
ordinary examples; list [ownership](../../../../glossary.md#term-ownership-meanings)/immutability; value semantics; all [constant](../../../../glossary.md#term-constant) categories including
raw [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero)/[NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity) retention; extreme long values; every validation failure/precedence;
exact [Operation](../../../../glossary.md#operation) pairings/reference retention; absence of cross-family pairing in examples; and no
extra members or forbidden dependencies.

Inspect `javap -p -c -s`, reflection, imports, and source. Confirm four production types, exact
records/enums, two plus one post-validation `List.copyOf` calls, no retained helper state, and no
[DataType](../../../../glossary.md#data-type--datatype)/[Shape](../../../../glossary.md#shape)/[Tensor](../../../../glossary.md#tensor)/[layout](../../../../glossary.md#layout)/storage/[graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler)/[planning](../../../../glossary.md#term-planning)/[prepare](../../../../glossary.md#prepare)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend)/[ONNX](../../../../glossary.md#term-onnx)/training
types. Validate generated [Javadoc](../../../../glossary.md#term-javadoc), API/glossary terminology, links/anchors/fences/whitespace,
exact ten paths, synchronized statuses, and no task-0017J specification.

## Dependencies

- 0005 supplies [OperationKind](../../../../glossary.md#operationkind) and [OperationAttrs](../../../../glossary.md#operationattrs).
- 0006 supplies [immutable](../../../../glossary.md#term-immutable) generic [Operation](../../../../glossary.md#operation) composition.
- 0001/0002 supply deferred [DataType](../../../../glossary.md#data-type--datatype)/[Shape](../../../../glossary.md#shape) terminology but are not production dependencies.
- Completed [layout](../../../../glossary.md#layout)/[view](../../../../glossary.md#term-view-meanings) semantics establish package [ownership](../../../../glossary.md#term-ownership-meanings) and semantic/expression separation.

## Follow-up tasks

- 0017J remains Draft for public pad/tile requests, rank validation, array ownership, checked Shape
  arithmetic, constant/data-type policy, result descriptors, provenance, and local layout rules.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Shape glossary](../../../../glossary.md#shape); [constant glossary](../../../../glossary.md#term-constant); [descriptors glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [layout glossary](../../../../glossary.md#layout))</span>
- [Compiler](../../../../glossary.md#term-compiler) later owns [identity](../../../../glossary.md#term-identity-meanings)/[canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) and [backward graph](../../../../glossary.md#backward-graph) construction.
- [Planning](../../../../glossary.md#term-planning)/[backend](../../../../glossary.md#backend) [prepare](../../../../glossary.md#prepare) later own allocation, [materialization](../../../../glossary.md#materialization), [lowering](../../../../glossary.md#lowering), and [kernels](../../../../glossary.md#kernel).
- [ONNX](../../../../glossary.md#term-onnx) extension later owns Pad/Tile mappings without entering [model](../../../../glossary.md#term-model) or [runtime](../../../../glossary.md#term-runtime) [hot path](../../../../glossary.md#term-hot-path).

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. These [model](../../../../glossary.md#term-model)-owned semantic values remain in the existing package and
[dependency direction](../../../../glossary.md#java-platform-and-build-tools). Stop if implementation needs [Tensor](../../../../glossary.md#tensor)/[Shape](../../../../glossary.md#shape)/[DataType](../../../../glossary.md#data-type--datatype) state, dependencies,
generic [Operation](../../../../glossary.md#operation) changes, or architecture changes.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0005/0006/0017G/0017H/0017I, Tensor API, Compile
API, Training API, glossary, current OperationKind/OperationAttrs/Operation and layout-operation
semantic contracts/tests, DataType/Shape terminology, and Java 26 Gradle configuration.

Implement task 0017I exactly. Add only PadKind.java, PadAttrs.java, TileKind.java, TileAttrs.java,
and PadTileSemanticsTest.java under io.github.pho001.synaptik.model.operation.layout.

PadKind contains exactly PAD; TileKind contains exactly TILE. PadAttrs contains exactly
List<Long> before, List<Long> after, and double constantValue; validate equal sizes, indexed
non-null elements, and non-negative widths with exact order/messages, then snapshot both lists.
Retain every double unchanged. TileAttrs contains exactly List<Long> repeats; validate indexed
non-null strictly positive values and snapshot once. Empty lists and extreme valid longs are
accepted. Document exact typed pairings, scalar identity, constant-padding meaning, and complete-
pattern tiling meaning.

Do not add Tensor methods, result Shape/DataType/layout/provenance behavior, constant conversion,
other padding modes, zero/negative repeats, gradients, graph/compiler/planning/runtime/backend/
ONNX behavior, factories, dependencies, build/architecture changes, existing Java edits, or later
specs. Stop beyond ten paths or on architecture uncertainty.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/glossary/planning, record no-change conclusions, and rerun validation.

Update 0017I, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0017J Draft without a specification. Do not commit/push.
```

## Local decisions

- Pad and tile use separate one-constant enums because they have unrelated attribute shapes and
  typed pairings; neither is an alias or mode of the other.<span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Parameters use immutable long lists to align with current Shape geometry while avoiding mutable
  legacy arrays.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- [Padding](../../../../glossary.md#padding) [constant](../../../../glossary.md#term-constant) remains `double` to preserve the selected legacy capability. Semantic storage
  does not imply a conversion rule for any DataType.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype))</span>
- All double values are structurally accepted and retained; public/task-0017J policy owns any
  eventual data-type compatibility decision.
- Tile repeats remain strictly positive for parity; zero-repeat empty-result semantics are not
  introduced silently.
- Empty lists are valid [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [scalar](../../../../glossary.md#term-scalar) [identity](../../../../glossary.md#term-identity-meanings) [parameters](../../../../glossary.md#term-parameter); input-[rank](../../../../glossary.md#term-rank) matching is deferred.

## Known limitations

- No public Tensor methods, rank/Shape arithmetic, constant conversion, result descriptor,
  provenance, gradient, materialization, lowering, execution, or ONNX mapping exists.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient); [materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- Only [constant](../../../../glossary.md#term-constant) [padding](../../../../glossary.md#padding) and positive per-[axis](../../../../glossary.md#term-axis) complete-pattern [tiling](../../../../glossary.md#tiling) are represented.

## [Validation](../../../../glossary.md#term-validation) evidence

- Architecture/focused boundary docs, documentation/planning rules, roadmap, model capabilities and
  master plan, Operation foundations, current layout semantic contracts/tests, DataType/Shape
  terminology, Tensor/Compile/Training APIs, glossary, Java 26 build, and read-only legacy
  pad/tile source/tests were reviewed before specifying behavior.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [layout glossary](../../../../glossary.md#layout); [DataType glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile))</span>
- Legacy evidence confirms constant double padding with equal-rank non-negative before/after
  arrays and positive per-axis complete-pattern tiling. Coupled legacy Shape, graph, storage,
  gradient, compiler, kernel, ONNX, and runtime design is excluded.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [padding glossary](../../../../glossary.md#padding); [rank glossary](../../../../glossary.md#term-rank); [axis glossary](../../../../glossary.md#term-axis); [graph glossary](../../../../glossary.md#graph); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [kernel glossary](../../../../glossary.md#kernel); [ONNX glossary](../../../../glossary.md#term-onnx); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- `git diff --check` passed with no whitespace errors.
- Changed-path inventory contains exactly three planning paths: this task, model master plan, and
  roadmap. No Java, API, glossary, Gradle, AGENTS, ARCHITECTURE, focused architecture, completed
  task, or other-module file changed during planning.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module))</span>
- Markdown structure check passed: 23 level-two sections, 14 balanced code-fence markers, no
  trailing whitespace, and final newline present.
- Every local Markdown link in all three changed files resolves.
- Roadmap contains all 74 ordered task rows. Task 0017I is linked and Ready at row 62; task 0017J
  remains Draft at row 63; no task-0017J specification exists.
- Task, master plan, and roadmap consistently identify 0017I as the next [Ready](../../../../glossary.md#term-ready) frontier.
- Package/[scope](../../../../glossary.md#term-scope) review confirms four cohesive public semantic contracts remain in
  `model.operation.layout`, one focused test mirrors that package, and implementation is bounded
  to ten authorized paths without dependency or architecture changes.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- No [Gradle](../../../../glossary.md#term-gradle) test was [run](../../../../glossary.md#term-run) because this [planning](../../../../glossary.md#term-planning)-only change modifies no production or test code.
- Clean implementation context `/root/implement_model_0017i` added exactly the four production
  contracts and focused test specified by this task before handing the actual shared-tree diff to
  the independent documentation context.
- Clean documentation context
  `/root/implement_model_0017i/review_model_0017i_docs` applied the General, API/Javadoc,
  Planning, and Example profiles. It independently reviewed the architecture and focused boundary
  documents, documentation/planning rules, capability baseline, prerequisite and neighboring
  tasks, Tensor/Compile/Training APIs, glossary, final source/tests, generated Javadoc, Java 26
  build configuration, and actual diff.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- The documentation pass retained the complete `PadKind` and `TileKind` Javadocs unchanged,
  tightened `PadAttrs` and `TileAttrs` constructor Javadocs with exact indexed failure messages,
  and finalized Tensor API and glossary coverage. The result documents exact typed pairings,
  scalar identity, immutable ordered snapshots, constant padding <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [identity glossary](../../../../glossary.md#term-identity-meanings); [immutable glossary](../../../../glossary.md#term-immutable))</span>`[10, 20] ->
  [-1, 10, 20, -1, -1]`, two-dimensional complete-pattern [tiling](../../../../glossary.md#tiling), structurally valid
  `Long.MAX_VALUE`, raw signed-zero/NaN/infinity retention, deferred Shape/DataType policy, and
  every excluded layer.<span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.layout.PadTileSemanticsTest` — `BUILD SUCCESSFUL`; the
  focused XML report contains 12 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 69 XML reports contain 566 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; generated pages contain both enums and
  constants, both record components and canonical constructors, explicit accessors, parameters,
  results, exact failure conditions, ownership, both examples, scalar/extreme/raw-double cases,
  and cross-layer exclusions.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [parameters glossary](../../../../glossary.md#term-parameter); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `./gradlew test` — `BUILD SUCCESSFUL`; the root lifecycle reported 36 actionable tasks without a
  failing task.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `javap -classpath modules/model/build/classes/java/main -p -c -s` for all four production types,
  focused reflection tests, source inspection, and import scans confirm the exact one-constant
  enums; exact three- and one-component records; validation order; direct accessors; generated
  record methods; exactly two plus one post-validation `List.copyOf` calls; no retained helper
  state or extra API; and production imports only `model.operation` plus `java.base`.
- A targeted local Markdown file-and-GitHub-heading checker resolved all 334 links and anchors in
  the five changed documentation/planning files with zero errors. Fence counts are balanced
  (`162`, `0`, `14`, `2`, `0`), trailing-whitespace scans found no matches, and every changed file
  ends with a newline.
- Final path inventory contains exactly the permitted ten paths: four production sources, one
  focused test, Tensor API, glossary, this task, model master plan, and roadmap. Task, master plan,
  and roadmap identify 0017I as Complete; 0017J remains Draft and no task-0017J specification
  exists.
- Compile API remains accurate unchanged because there is no public pad/tile Tensor expression or
  compiler capture. Training API remains accurate unchanged because no gradient, autograd,
  optimizer, parameter, session, or training behavior is defined. Capabilities remains accurate
  unchanged because it already selects constant pad and tile and distinguishes model semantics
  from later public and executable layers.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), [`Operation`](../../../../glossary.md#operation), DataType, Shape/Dimension, SliceAttrs/SliceKind,
  adjacent layout-operation semantics, Tensor/layout/graph contracts, generated related Javadocs,
  and completed task specifications remain accurate unchanged because this family composes their
  existing ownership and value contracts without changing signatures or behavior.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [signatures glossary](../../../../glossary.md#term-signature))</span>
- No architecture document, ADR, architecture test, backend-conformance test, integration test,
  Gradle file, dependency, existing unrelated Java/test, other module, or later task specification
  changed because this task adds only backend-independent model semantic values and no boundary,
  executable, backend, or end-to-end behavior.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [architecture test glossary](../../../../glossary.md#term-architecture-test); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test))</span>
- `git diff --check` passed with no whitespace errors after final documentation and planning
  synchronization.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>

## Implementation notes

- Added separate `PadKind.PAD` and `TileKind.TILE` [identities](../../../../glossary.md#term-identity-meanings) plus exact [immutable](../../../../glossary.md#term-immutable) `PadAttrs` and
  `TileAttrs` records in the existing [layout](../../../../glossary.md#layout)-[operation](../../../../glossary.md#operation) package.
- `PadAttrs` validates containers, matching sizes, indexed elements, and non-negative widths in the
  specified precedence, then snapshots both lists. It retains every <span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element))</span>`double` bit pattern without
  DataType interpretation. <span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype))</span>`TileAttrs` validates indexed strictly positive repeats and snapshots
  once. Empty lists and extreme valid longs remain structurally valid.
- Added one focused same-package test covering exact API shape, validation messages and precedence,
  immutable ownership, value semantics, raw-double retention, scalar/extreme cases, typed
  Operation composition, and forbidden-state absence.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Finalized public Javadocs, Tensor API, glossary, task evidence, model master plan, and roadmap
  without creating the 0017J specification or changing architecture or another contract.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [model glossary](../../../../glossary.md#term-model))</span>

## Completion summary

- Completed changes: Added and documented exact backend-neutral constant-padding and
  complete-pattern tiling semantic identities and immutable normalized attributes, with focused
  semantic tests and synchronized planning evidence.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [constant glossary](../../../../glossary.md#term-constant); [padding glossary](../../../../glossary.md#padding); [tiling glossary](../../../../glossary.md#tiling); [immutable glossary](../../../../glossary.md#term-immutable); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: Exactly the ten paths listed under Affected files.
- Tests and validation: Focused 12-test suite, all 566 model tests, model Javadoc, root tests,
  bytecode/reflection/import/source checks, generated-Javadoc review, 334 local link/anchor checks,
  fence/whitespace/final-newline checks, exact scope/status review, no-0017J-spec check, and<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0017i/review_model_0017i_docs` independently finalized the change using
  the General, API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Documentation impact: Tensor API and glossary now define current pad/tile semantics, exact
  examples, ownership, validation, and deferred boundaries while public Tensor construction
  remains Draft task 0017J work.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Javadoc review: PadKind/TileKind type and constant Javadocs were complete unchanged; PadAttrs and
  TileAttrs constructor failure documentation was tightened, and all type/component/accessor/
  ownership/example/exclusion contracts are complete. Related Javadocs remain accurate unchanged.
- Glossary impact: Added reusable Padding and Tiling terms and updated implementation status,
  OperationKind, OperationAttrs, and the kind/attributes/operation distinction.<span data-glossary-navigation="true"> ([OperationKind glossary](../../../../glossary.md#operationkind); [OperationAttrs glossary](../../../../glossary.md#operationattrs); [operation glossary](../../../../glossary.md#operation))</span>
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0017I. Task 0017J remains Draft without a specification.

Status: Complete
