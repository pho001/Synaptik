# Task 0019C: Sort and Argsort

## Status

Complete

## Goal

Add deterministic, stable, [axis](../../../../glossary.md#term-axis)-wise `sort` and `argsort` [model](../../../../glossary.md#term-model) semantics plus their public
[`Tensor`](../../../../glossary.md#tensor) expression construction. `sort` returns values only and `argsort` returns indices only;
each call is one single-output [producer](../../../../glossary.md#term-producer) [occurrence](../../../../glossary.md#term-occurrence).

This is the first half of the former sorting/top-K frontier. Full sorting preserves the input
[Shape](../../../../glossary.md#shape), while top-K replaces one [extent](../../../../glossary.md#term-extent) with `k` and must return values and indices from one shared
two-output producer. Keeping those contracts in tasks 0019C and 0019C1 makes each capability fit
one isolated model session without splitting semantic values from its public [facade](../../../../glossary.md#term-facade).

For a row `[3.0, NaN, -0.0, +0.0, 3.0]`, ascending stable sort requests
`[-0.0, +0.0, 3.0, 3.0, NaN]` and argsort requests `[2, 3, 0, 4, 1]`. Descending stable sort
requests `[3.0, 3.0, +0.0, -0.0, NaN]` and indices `[0, 4, 3, 2, 1]`. Equal values, including
multiple NaNs, retain increasing logical input-index order.

## Rationale and selected convention

NumPy and JAX use the established public names `sort` and `argsort`, an [axis](../../../../glossary.md#term-axis), a descending choice,
and a stability concept. Current [NumPy sort](https://numpy.org/doc/stable/reference/generated/numpy.sort.html)
also places NaNs last for both directions, while [JAX sort](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sort.html)
defaults to stable ordering. Synaptik selects the small deterministic intersection: sorting is
always stable, NaNs are always last, and callers choose only axis and direction. No algorithm name
or unstable [backend](../../../../glossary.md#backend)-dependent tie order enters [model](../../../../glossary.md#term-model) semantics.

## [Scope](../../../../glossary.md#term-scope)

- Add one `operation.ordering` package for ordering [operation](../../../../glossary.md#operation) semantics.
- Add public `OrderingKind` with exactly `SORT` and `ARGSORT`, in that order.
- Add public [immutable](../../../../glossary.md#term-immutable) `SortAttrs(int axis, boolean descending)`.
- Give both kinds exact one-input, one-output `SortAttrs` [signatures](../../../../glossary.md#term-signature).
- Add package-private `TensorSortExpressions` for shared validation, Shape/result construction,
  and provenance construction.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [provenance glossary](../../../../glossary.md#provenance))</span>
- Add exactly these four public instance methods:

```java
Tensor sort(int axis)
Tensor sort(int axis, boolean descending)
Tensor argsort(int axis)
Tensor argsort(int axis, boolean descending)
```

- Make the one-[argument](../../../../glossary.md#term-argument) forms delegate semantically to ascending order.
- Accept all six current input data types. Floating and integral values use their documented
  numerical ordering; BOOL uses <span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype))</span>`false < true`.
- Normalize the [axis](../../../../glossary.md#term-axis) before attributes construction and preserve the exact input [Shape](../../../../glossary.md#shape) reference.
- Preserve input [data type](../../../../glossary.md#data-type--datatype) and `requiresGrad` on `sort`; use fixed [`INT64`](../../../../glossary.md#data-type--datatype) and false
  `requiresGrad` on `argsort`.
- Leave both result [layouts](../../../../glossary.md#layout) unresolved and results unlabeled, storage-free, and fresh.
- Record one exact input, one output descriptor, one producer, output index zero, and one new
  Tensor wrapper/ID per successful call.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer))</span>
- Define exact stable ordering, duplicates, NaNs, infinities, signed zero, scalar, empty-axis, and
  dynamic-extent contracts.<span data-glossary-navigation="true"> ([infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [signed zero glossary](../../../../glossary.md#term-signed-zero); [scalar glossary](../../../../glossary.md#term-scalar); [extent glossary](../../../../glossary.md#term-extent))</span>
- Update all global operation-signature and public-[Tensor](../../../../glossary.md#tensor) inventories in the same change.
- Finalize Javadoc, Tensor API, Compile API, glossary, capability/task/master/roadmap text, and
  validation evidence through a mandatory separate clean-context documentation pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- top-K, `TopKResult`, `TOP_K`, `TopKAttrs`, partial sorting, partition, kth-value, rank, search,
  lexicographic multi-key sort, or a combined values-and-indices sort result; task 0019C1 owns
  genuine two-output top-K<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [rank glossary](../../../../glossary.md#term-rank))</span>
- a multi-output `SORT` [occurrence](../../../../glossary.md#term-occurrence), a public `SortResult`, or making one `sort` call expose indices
- flattening when no axis is supplied, a parameterless final-axis overload, in-place sorting,
  output buffers, named axes, structured/complex values, or user-defined comparators<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- caller-selectable stability, an unstable mode, algorithm names, `kind`, quicksort, radix sort,
  backend route selection, or a stability boolean stored in attributes<span data-glossary-navigation="true"> ([backend route glossary](../../../../glossary.md#backend-route); [boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- caller-selectable NaN placement, NaN payload canonicalization, or a NaN-placement attribute;
  NaNs-last is fixed family semantics<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- changing arg-min/arg-max ordering or tie policies; those reductions retain their independently
  selected NaN-preference contract
- value evaluation, storage reads or writes, permutation materialization, compiler capture,
  constant folding, gradient rules, autograd, planning, lowering, kernels, prepare, runtime, or
  execution<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [permutation glossary](../../../../glossary.md#permutation); [materialization glossary](../../../../glossary.md#materialization); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [constant folding glossary](../../../../glossary.md#term-constant-folding); [gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [planning glossary](../../../../glossary.md#term-planning); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- changes to `TensorFactory`, [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), `TensorProvenance`, [`Operation`](../../../../glossary.md#operation),
  [`OperationKind`](../../../../glossary.md#operationkind), `OperationSignature`, [`DataType`](../../../../glossary.md#data-type--datatype), [`Shape`](../../../../glossary.md#shape), or [`Dimension`](../../../../glossary.md#dimension)
- dependencies, Gradle, architecture/focused-architecture documents, architecture tests,
  backend-conformance/integration tests, another module, or a detailed task-0019C1 specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially model ownership, public
  mutable <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [mutable glossary](../../../../glossary.md#term-mutable))</span>[`Tensor`](../../../../glossary.md#tensor), [backend](../../../../glossary.md#backend)-independent [`Operation`](../../../../glossary.md#operation), and [compiler](../../../../glossary.md#term-compiler)/backend/[runtime](../../../../glossary.md#term-runtime) boundaries
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Task 0018L](0018l-shared-multi-output-tensor-provenance.md)
- [Task 0018U](0018u-integral-elementwise-arithmetic-and-comparisons.md)
- [Task 0018U1](0018u1-integral-reductions-and-arg-min-normalization.md)
- [Task 0019B1](0019b1-explicit-graph-dropout-construction.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Runtime API](../../../../api/runtime-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns ordering semantics, descriptors, public Tensor construction, and immutable
  pre-capture producer provenance only.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [immutable glossary](../../../../glossary.md#term-immutable); [capture glossary](../../../../glossary.md#term-capture-meanings); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- [`Tensor`](../../../../glossary.md#tensor) remains public mutable API state, not an intermediate-representation node. Ordering
  provenance is occurrence metadata and adds no graph-local identity.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [node glossary](../../../../glossary.md#node); [occurrence glossary](../../../../glossary.md#term-occurrence); [metadata glossary](../../../../glossary.md#term-metadata); [graph glossary](../../../../glossary.md#graph); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `OrderingKind` and `SortAttrs` contain no backend support, algorithm, cost, kernel, storage,
  compiler service, gradient implementation, or runtime state.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [kernel glossary](../../../../glossary.md#kernel); [compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Stable means equal keys retain increasing logical coordinate order along the selected axis,
  independently of physical layout, strides, traversal, backend, or algorithm.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [axis glossary](../../../../glossary.md#term-axis); [layout glossary](../../../../glossary.md#layout); [strides glossary](../../../../glossary.md#term-stride))</span>
- NaN-last is a semantic output-order rule for both ascending and descending order. It is not a
  backend choice and is not inferred from an implementation comparator.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan))</span>
- `sort` and `argsort` are separate semantic requests. Calling each creates a distinct producer;
  they must not share a producer, cache, result registry, sibling lookup, or hidden public carrier.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [registry glossary](../../../../glossary.md#term-registry); [lookup glossary](../../../../glossary.md#term-lookup); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Construction must use the existing one-output `TensorFactory.createDerived(...)` seam exactly
  once after local validation. No factory/provenance foundation change is needed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Compiler work later owns capture, graph validation, canonicalization, gradients, and backward
  construction. Backend prepare later owns algorithm and kernel selection; runtime later executes
  prepared work.<span data-glossary-navigation="true"> ([canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [prepare glossary](../../../../glossary.md#prepare))</span>
- If implementation requires cross-module behavior, a producer registry, changes to shared
  factory/provenance contracts, or architecture changes, stop and report the conflict.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor methods and package-private
  expression construction.
- `io.github.pho001.synaptik.model.operation` — supplies existing typed operations and [signatures](../../../../glossary.md#term-signature).
- `io.github.pho001.synaptik.model.datatype` and `.shape` — supply exact type and [Shape](../../../../glossary.md#shape) facts.

Packages added or changed:

- `io.github.pho001.synaptik.model.operation.ordering` — new public semantic family for full sort,
  argsort, and later top-K ordering contracts. It owns no implementation algorithm.

Type placement:

- `io.github.pho001.synaptik.model.operation.ordering.OrderingKind` — owns the typed `SORT` and
  `ARGSORT` [identities](../../../../glossary.md#term-identity-meanings); task 0019C1 may later add `TOP_K` to this same cohesive family.
- `io.github.pho001.synaptik.model.operation.ordering.SortAttrs` — owns one normalized axis and
  descending choice shared by the two full-order kinds.<span data-glossary-navigation="true"> ([normalized axis glossary](../../../../glossary.md#normalized-axis))</span>
- `io.github.pho001.synaptik.model.tensor.TensorSortExpressions` — package-private stateless
  validation and construction helper colocated with Tensor/factory access. It is final, has one
  private zero-argument constructor, no fields/nested types/interfaces, and exactly one declared
  package-private static <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [argument glossary](../../../../glossary.md#term-argument))</span>`apply(Tensor, OrderingKind, int, boolean)` method returning [`Tensor`](../../../../glossary.md#tensor).
- `io.github.pho001.synaptik.model.tensor.Tensor` — owns the four public receiver methods.

Test placement:

- `io.github.pho001.synaptik.model.operation.ordering.OrderingSemanticsTest` — locks enum,
  attributes, signatures, stable value-order policy, and non-execution boundaries.
- `io.github.pho001.synaptik.model.tensor.TensorSortExpressionTest` — locks public/helper surface,
  validation, descriptors, Shape references, provenance, identities, and input immutability.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- Existing root operation-signature and Tensor-surface inventory tests remain global owners and
  are updated rather than duplicated.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>

## Required semantic contracts

### Kinds and [signatures](../../../../glossary.md#term-signature)

Create:

```java
public enum OrderingKind implements OperationKind {
    SORT,
    ARGSORT
}
```

Both [constants](../../../../glossary.md#term-constant) accept exactly:

```java
OperationSignature.fixed(SortAttrs.class, 1, 1)
```

The enum exposes only the family-owned stable [immutable](../../../../glossary.md#term-immutable) signature list required by the existing
[`OperationKind`](../../../../glossary.md#operationkind) contract. It must not expose a comparator, algorithm, supported-data-type table,
or top-K constant in this task.

### Attributes

Create:

```java
public record SortAttrs(int axis, boolean descending) implements OperationAttrs
```

The [canonical](../../../../glossary.md#term-canonical-form) constructor rejects a negative [axis](../../../../glossary.md#term-axis) with exact message:

```text
axis must be non-negative: <axis>
```

The record retains the [normalized axis](../../../../glossary.md#normalized-axis) and exact direction flag. Stability and [NaN](../../../../glossary.md#term-nan) placement are
fixed semantics, not configurable components. It performs no [rank](../../../../glossary.md#term-rank), [extent](../../../../glossary.md#term-extent), [Tensor](../../../../glossary.md#tensor), value,
[descriptor](../../../../glossary.md#term-descriptor), algorithm, or execution [validation](../../../../glossary.md#term-validation). It explicitly documents/overrides both generated
accessors but adds no other public behavior.

### Public [API](../../../../glossary.md#term-application-programming-interface-api) and defaults

Add only:

```java
public Tensor sort(int axis)
public Tensor sort(int axis, boolean descending)
public Tensor argsort(int axis)
public Tensor argsort(int axis, boolean descending)
```

The one-[argument](../../../../glossary.md#term-argument) forms are ascending (`descending == false`). There is no method that returns both
values and indices, no default-[axis](../../../../glossary.md#term-axis) overload, and no stability/[NaN](../../../../glossary.md#term-nan) [parameter](../../../../glossary.md#term-parameter).

### Eligibility and result [metadata](../../../../glossary.md#term-metadata)

All current types succeed:

- [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), and [BFLOAT16](../../../../glossary.md#term-bfloat16) use floating ordering below;
- [INT32](../../../../glossary.md#data-type--datatype) and [INT64](../../../../glossary.md#data-type--datatype) use signed mathematical order; and
- [BOOL](../../../../glossary.md#data-type--datatype) uses `false < true`.

`sort` result:

- exact input [data type](../../../../glossary.md#data-type--datatype);
- exact input [Shape](../../../../glossary.md#shape) reference;
- exact input `requiresGrad` value;
- unresolved [layout](../../../../glossary.md#layout), absent label/storage;
- fresh [Tensor](../../../../glossary.md#tensor) and ID; and
- one-output [provenance](../../../../glossary.md#provenance) for `OrderingKind.SORT` and exact `SortAttrs`.

`argsort` result:

- fixed `DataType.INT64`;
- exact input Shape reference;
- false `requiresGrad`;
- unresolved layout, absent label/storage;
- fresh Tensor and ID; and
- one-output provenance for `OrderingKind.ARGSORT` and exact `SortAttrs`.

[Gradient](../../../../glossary.md#term-gradient) eligibility is metadata only. This task does not define a sort gradient or claim that
one exists. Argsort is non-[differentiable](../../../../glossary.md#derivative-and-chain-rule).

### [Axis](../../../../glossary.md#term-axis), [scalar](../../../../glossary.md#term-scalar), empty, and dynamic [Shape](../../../../glossary.md#shape)

- Normalize positive or negative caller axes once with `Shape.normalizeAxis(int)` before creating
  `SortAttrs`.
- Scalar input rejects every axis with the existing Shape axis exception and consumes no ID.
- A selected static extent of zero is valid. The result is the same empty Shape and no value needs
  selection.
- [Extent](../../../../glossary.md#term-extent) one is valid and retains its sole value/index.
- A dynamic or expression extent is accepted without binding or proof because full sort preserves
  the exact Shape. Eventual execution sorts the bound finite axis domain.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [finite glossary](../../../../glossary.md#term-finite))</span>
- Empty unselected axes are also valid; they describe zero independent slices.

### Total output ordering and stability

Ordering is applied independently to each [slice](../../../../glossary.md#slice) along the normalized logical [axis](../../../../glossary.md#term-axis).

- Non-NaN floating values use numerical order. Negative infinity precedes finite values and
  positive infinity follows them in ascending order; descending reverses those non-NaN classes.<span data-glossary-navigation="true"> ([Negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [finite glossary](../../../../glossary.md#term-finite); [positive infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Negative zero is strictly below positive zero in ascending order and strictly above it after
  descending reversal. The two zeros are therefore not a stability tie.<span data-glossary-navigation="true"> ([Negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Every NaN compares in one final NaN class after all non-NaN values for both ascending and
  descending requests. No NaN payload, sign, or signaling/quiet distinction affects placement.<span data-glossary-navigation="true"> ([NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Equal finite/integral/BOOL keys and multiple NaNs retain increasing original logical coordinate
  order. <span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`argsort` therefore returns deterministic [INT64](../../../../glossary.md#data-type--datatype) logical indices.
- `sort` retains the exact selected input element representation, including NaN payload bits and
  signed zero; it does not canonicalize or numerically recreate values.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [signed zero glossary](../../../../glossary.md#term-signed-zero))</span>
- Duplicates are never deduplicated. Sorting changes only logical order along the selected axis.
- Stability is unconditional. No conforming backend may choose another tie order, even if its
  internal primitive is unstable; backend prepare must select or construct conforming behavior.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare))</span>

This ordering is deliberately distinct from arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum) reduction semantics, where [NaN](../../../../glossary.md#term-nan) is an
[extremum](../../../../glossary.md#term-extremum) candidate for both ARG_MIN and ARG_MAX. Neither family redefines the other.

## Construction, [provenance](../../../../glossary.md#provenance), [validation](../../../../glossary.md#term-validation), and ID contract

The helper validates in this exact order:

1. `input` non-null (`"input"`);
2. `kind` non-null (`"kind"`);
3. normalize [`axis`](../../../../glossary.md#term-axis) through the exact input [Shape](../../../../glossary.md#shape); and
4. derive attributes and the selected result [descriptor](../../../../glossary.md#term-descriptor) before factory delegation.

The helper [parameter](../../../../glossary.md#term-parameter) is typed as `OrderingKind`, whose current inventory contains only the two
accepted kinds, so there is no unreachable cross-family kind branch. Task 0019C1 must add explicit
sort-helper exclusion when it adds `TOP_K` to that enum. There is no data-type rejection because
every current type is selected. `descending` needs no validation. Every local failure and axis
failure consumes no [Tensor](../../../../glossary.md#tensor) ID.

For a valid call:

- construct one `Operation(kind, new SortAttrs(normalizedAxis, descending))`;
- call `TensorFactory.createDerived(...)` once with exact ordered input `[input]`;
- create exactly one output descriptor, wrapper, Tensor ID, [producer](../../../../glossary.md#term-producer), and provenance value;
- use producer output slot zero;
- retain no input storage and perform no storage access; and
- never mutate input descriptor, label, storage, provenance, or backing values.

Repeated equal calls create fresh Tensors, IDs, producers, operations, attributes, descriptors,
and provenance. The exact input Tensor and exact Shape reference are retained. Identifier
exhaustion occurs only after all local [metadata](../../../../glossary.md#term-metadata) validation, follows current factory behavior, and
is not translated or rolled back.

## Affected files

Expected production:

- new `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/ordering/OrderingKind.java`
- new `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/ordering/SortAttrs.java`
- new `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorSortExpressions.java`
- existing `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected tests:

- new `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/ordering/OrderingSemanticsTest.java`
- new `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSortExpressionTest.java`
- existing `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
  for the global production signature inventory
- existing `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
  for the authoritative public-method count/name inventory; exact count becomes 167
- existing `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
  for its intentionally global public-Tensor count; exact count becomes 167, with binary-
  arithmetic-specific assertions otherwise unchanged<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- existing `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
  for its intentionally global public-Tensor count; exact count becomes 167, with MATMUL-specific
  assertions otherwise unchanged<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>

Expected documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Mandatory inspected inventories/seams, changed only if an acceptance criterion proves the listed
[scope](../../../../glossary.md#term-scope) insufficient:

- `TensorFactory.createDerived(...)` and `createDerivedOutputs(...)`, [`TensorProducer`](../../../../glossary.md#producer--tensorproducer),
  `TensorProvenance`, and their tests
- [`Operation`](../../../../glossary.md#operation), [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), `OperationSignature`, and global [signature](../../../../glossary.md#term-signature) tests
- [`DataType`](../../../../glossary.md#data-type--datatype), [`Shape`](../../../../glossary.md#shape), [`Dimension`](../../../../glossary.md#dimension), `StaticDimension`, their focused tests, and dynamic-Shape tests
- comparison and arg-extrema kinds/attributes/helpers/tests for signed, NaN, zero, tie, and index
  terminology<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [NaN glossary](../../../../glossary.md#term-nan))</span>
- public result carriers, especially `DropoutResult`, only to confirm no sort [carrier](../../../../glossary.md#term-storage-carrier) is needed
- `docs/api/runtime-api.md` and `docs/api/training-api.md` for reasoned no-change conclusions
- architecture/focused-architecture documents, Gradle, architecture tests,
  backend-conformance/integration tests, and other modules for scope/no-change confirmation<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify exactly the 17 expected paths above: four production, six test,
and seven documentation/[planning](../../../../glossary.md#term-planning) paths. Do not modify an inspected seam merely to consume scope.

No `TensorFactory`, [producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance) foundation, result [carrier](../../../../glossary.md#term-storage-carrier), Java path outside
`modules/model`, [Gradle](../../../../glossary.md#term-gradle)/[dependency](../../../../glossary.md#term-dependency) file, architecture document, [architecture test](../../../../glossary.md#term-architecture-test),
[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance)/[integration test](../../../../glossary.md#term-integration-test), other [module](../../../../glossary.md#term-module), or later task specification may change. If
the capability needs more than 17 paths or another concept, stop and propose a focused follow-up.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Give `OrderingKind`, `SortAttrs`, `TensorSortExpressions`, and both overload families meaningful
  Javadoc covering purpose, exact inputs/results, nullability, normalized axes, stability, NaNs,
  signed zero, infinities, duplicates, empty/scalar/dynamic Shapes, metadata, provenance, ID
  effects, failure modes, and compiler/backend/execution boundaries.<span data-glossary-navigation="true"> ([nullability glossary](../../../../glossary.md#term-nullability); [axes glossary](../../../../glossary.md#term-axis); [signed zero glossary](../../../../glossary.md#term-signed-zero); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [scalar glossary](../../../../glossary.md#term-scalar); [Shapes glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- Update Tensor API with current construction examples for values and indices, ascending and
  descending output tables, all-type eligibility, Shape/metadata/provenance, and honest
  current-versus-planned behavior.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Update Compile API only to inventory sort/argsort as current model expressions available to a
  future compiler; do not claim capture, validation, canonicalization, gradients, or support.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model); [capture glossary](../../../../glossary.md#term-capture-meanings); [validation glossary](../../../../glossary.md#term-validation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Update the glossary with reusable stable sort/argsort and NaN-last terminology without
  duplicating the full API guide.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan))</span>
- Runtime API and Training API are expected no-change because there is no executable or training-
  extension API. Record the reasoned conclusion.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Architecture docs/ADRs/tests, conformance/integration tests, Gradle, and other modules are
  expected no-change because ownership and execution boundaries do not change. Record each
  conclusion.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Acceptance criteria

- The exact public surface is the four methods listed above; no carrier, default-axis,
  stability, NaN, algorithm, or in-place overload is added.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [NaN glossary](../../../../glossary.md#term-nan))</span>
- `TensorSortExpressions` has the exact field-free, one-method package-private helper surface
  specified under Package impact; no reusable comparator or utility abstraction is added.
- `OrderingKind` has exactly `SORT`, `ARGSORT` and exact fixed `SortAttrs` one-input/one-output
  signatures. <span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>`SortAttrs` retains a normalized non-negative [axis](../../../../glossary.md#term-axis) and descending flag.
- Every current [DataType](../../../../glossary.md#data-type--datatype) succeeds; [BOOL](../../../../glossary.md#data-type--datatype) uses false-before-true ascending order. `sort` retains
  exact input type/Shape reference/requiresGrad, while <span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>`argsort` is INT64/false-gradient with the
  same exact Shape reference.<span data-glossary-navigation="true"> ([INT64 glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Stable equal-key, duplicate, NaN-last-both-directions, NaN representation, signed-zero,
  infinity, scalar rejection, empty axis, singleton axis, dynamic extent, and logical-index
  semantics are explicitly documented and representation-tested where model construction allows.<span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity); [scalar glossary](../../../../glossary.md#term-scalar); [singleton glossary](../../../../glossary.md#term-singleton); [extent glossary](../../../../glossary.md#term-extent); [model glossary](../../../../glossary.md#term-model))</span>
- Each call is one independent one-output [producer](../../../../glossary.md#term-producer) with exact input `[input]`, output slot zero,
  one wrapper/ID, unresolved layout, absent label/storage, and no input mutation.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- Validation order, exact messages, and no-ID behavior match this specification. Identifier
  exhaustion propagates after valid local construction according to the factory contract.<span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation))</span>
- Sort and argsort remain distinct occurrences and do not use shared multi-output construction.
- No value evaluation, algorithm, gradient, compiler, backend, runtime, or execution support is
  implemented or claimed.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Global signature inventory and both global Tensor public-surface inventories are updated up
  front and lock exact count 167.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Exactly one final `./gradlew :modules:model:test` run occurs after executable Java stabilizes.
  The clean documentation pass reuses it and does not repeat it unless executable Java changes.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- The separate documentation-focused pass finalizes all affected Javadocs/docs/glossary/planning
  evidence, runs model Javadoc and documentation checks, and records all no-change conclusions.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Task/master/roadmap show 0019C Ready until implementation completes, 0019C1 Draft without a
  detailed specification, and 0019D/0019E unchanged Draft. No other model task is Ready.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) focused ordering semantic/expression tests plus directly affected
[signature](../../../../glossary.md#term-signature) and public-surface tests as needed. After executable Java stabilizes, run exactly one
final [model](../../../../glossary.md#term-model) suite:

```bash
./gradlew :modules:model:test
```

The separate clean-context documentation pass reuses that result and runs after its final edits:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

It must also [compile](../../../../glossary.md#compile)/run the documented newcomer example against current model classes; inspect
generated [Javadoc](../../../../glossary.md#term-javadoc); validate local Markdown links and anchors, balanced fences, final newlines,
trailing whitespace, terminology, exact 17-path [scope](../../../../glossary.md#term-scope), package/type placement, status/[dependency](../../../../glossary.md#term-dependency)
coherence, exactly one [Ready](../../../../glossary.md#term-ready) task, 0019C1/0019D/0019E Draft, and absence of a 0019C1 detailed spec.

Repository-wide validation is deferred to the sorting/top-K capability [checkpoint](../../../../glossary.md#term-checkpoint) after task
0019C1, unless this task unexpectedly changes a repository-wide contract. Architecture tests run
only if a boundary/dependency changes; that is not expected and otherwise requires stopping.
[Backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) and integration tests remain deferred until executable support exists.

## Dependencies

- Task 0018K — typed [operation](../../../../glossary.md#operation) [signatures](../../../../glossary.md#term-signature) and construction hardening: Complete.
- Task 0018L — shared [producer](../../../../glossary.md#term-producer)/output-slot [provenance](../../../../glossary.md#provenance) and factory seams: Complete.
- Tasks 0018U and 0018U1 — integral numerical order plus arg-extrema NaN/zero/tie/index policy:
  Complete.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [NaN glossary](../../../../glossary.md#term-nan))</span>
- Tasks 0001–0002, 0005–0007, and 0011–0013 — DataType, Shape, Operation, Tensor, descriptor,
  factory, and provenance foundations: Complete.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Current global signature and [Tensor](../../../../glossary.md#tensor) public-surface inventories: inspected and named in [scope](../../../../glossary.md#term-scope).

## Follow-up tasks

- 0019C1 — genuine multi-output top-K; required Draft follow-up, with no detailed spec until 0019C
  is Complete. It will add `OrderingKind.TOP_K`,
  `TopKAttrs(int axis, long k, boolean largest, boolean sorted)`, public
  `TopKResult(Tensor values, Tensor indices)`, and exactly
  `topK(long k, int axis)` plus `topK(long k, int axis, boolean largest, boolean sorted)`; the
  shorter form defaults to largest and sorted, following the conventional option names in
  [PyTorch top-k](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.topk.html). The
  output Shape replaces the selected axis with
  static `k`, and exactly one shared [producer](../../../../glossary.md#term-producer) has ordered outputs `[values, indices]` at slots zero
  and one. `TopKResult` retains those exact wrappers and rejects null `values` then `indices`; it
  does not reconstruct outputs or producers.
- 0019D — [linear](../../../../glossary.md#term-linear-meanings) convenience; remains Draft and keeps its established ID and [dependency](../../../../glossary.md#term-dependency).
- 0019E — [scaled dot-product attention](../../../../glossary.md#scaled-dot-product-attention); remains Draft and keeps its established ID/dependencies.
- Later compiler/autograd work owns capture and any valid gradient construction. Later backend/
  runtime work owns sorting algorithms, kernels, execution, and conformance.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [capture glossary](../../../../glossary.md#term-capture-meanings); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [kernels glossary](../../../../glossary.md#kernel); [conformance glossary](../../../../glossary.md#term-conformance))</span>

The 0019C1 contract will accept all six current input types and reuse this task's fixed stable
ordering: equal candidates prefer lower logical indices; non-[NaN](../../../../glossary.md#term-nan) values use requested
largest/smallest direction; NaNs [rank](../../../../glossary.md#term-rank) after all non-NaNs for either selection direction. The
selected set is the first `k` positions of that complete stable order, so NaNs enter only when
fewer than `k` non-NaNs exist. `k == 0` is valid, static `k > extent` is invalid, dynamic selected
extents defer `extent >= k` until [binding](../../../../glossary.md#term-binding), and [scalar](../../../../glossary.md#term-scalar) inputs reject every [axis](../../../../glossary.md#term-axis). A static empty axis
therefore accepts only `k == 0`. The output [Shape](../../../../glossary.md#shape) replaces the selected input [Dimension](../../../../glossary.md#dimension) with one
fresh static `k`; `sorted == true` uses requested value order, while `sorted == false` returns the
deterministically selected set in increasing original logical-index order. Values retain input
type/requiresGrad; indices are [INT64](../../../../glossary.md#data-type--datatype)/false-[gradient](../../../../glossary.md#term-gradient); both are unresolved, unlabeled, storage-free
wrappers with two IDs, output indices zero/one, and one exact producer. No execution or gradient
meaning is added. `TOP_K` has exactly one input and two outputs. Planned local [validation](../../../../glossary.md#term-validation) checks
input, axis [normalization](../../../../glossary.md#term-normalization), non-negative `k`, then a known static upper bound, all before IDs. The
selected messages are `k must be non-negative: <k>` and
`k must not exceed selected static extent: k=<k>, axis=<axis>, extent=<extent>`; an unbound [extent](../../../../glossary.md#term-extent)
defers the same inequality rather than guessing. `TopKResult` null messages are `values` then
`indices`.

## Architecture impact

Expected impact: None.

This task adds only [model](../../../../glossary.md#term-model)-owned [backend](../../../../glossary.md#backend)-independent ordering semantics and [Tensor](../../../../glossary.md#tensor) construction
through existing contracts. If implementation requires architecture, [dependency](../../../../glossary.md#term-dependency), [compiler](../../../../glossary.md#term-compiler),
[runtime](../../../../glossary.md#term-runtime), [prepare](../../../../glossary.md#prepare), backend, or another-[module](../../../../glossary.md#term-module) changes, stop and report the exact conflict.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, docs/developer-guide/documentation-rules.md,
docs/planning/planning-guide.md, docs/planning/modules/model/capabilities.md, the model master plan,
roadmap, completed tasks 0018K/0018L/0018U/0018U1/0019B1, and task 0019C in full. Inspect every
affected source/test and mandatory inventory named by 0019C.

Implement task 0019C exactly as specified, only in modules/model and the 17 authorized paths.
Stop on an architecture, cross-module, factory/provenance, semantic, validation-order, or scope
conflict. Do not implement top-K, algorithms, gradients, compiler/runtime/backend behavior, or
another task. Run focused tests while developing and exactly one final model suite after
executable Java stabilizes.

Then hand the actual diff and recorded Java-test evidence to a separate clean-context
documentation-focused agent in the same overall change. That agent must independently inspect the
final contracts, finalize Javadocs, Tensor/Compile API, glossary, planning/status/evidence and
no-change conclusions, run model Javadoc and documentation validation, and reuse the successful
Java evidence unless it changes executable behavior. Do not mark Complete until all acceptance
criteria pass. Do not create the 0019C1 spec. Do not commit or push.
```

## Separate documentation [handoff](../../../../glossary.md#term-handoff)

The implementation agent must provide the documentation agent: this task file; actual diff; exact
final [model](../../../../glossary.md#term-model)-test command/result; affected four-method [API](../../../../glossary.md#term-application-programming-interface-api) and ordering semantics; exact type,
[Shape](../../../../glossary.md#shape), [provenance](../../../../glossary.md#provenance), [validation](../../../../glossary.md#term-validation), and ID contracts; no-algorithm/no-execution/[compiler](../../../../glossary.md#term-compiler)-owned-[gradient](../../../../glossary.md#term-gradient)
boundaries; expected [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) API and glossary changes; [Runtime](../../../../glossary.md#term-runtime)/Training/architecture/
[conformance](../../../../glossary.md#term-conformance)/integration/[Gradle](../../../../glossary.md#term-gradle)/other-[module](../../../../glossary.md#term-module) no-change reviews; and all documentation validation
commands above. The docs agent must report its clean context ID and whether executable Java
changed after reused test evidence.

## Local decisions

- The original 16-path implementation [scope](../../../../glossary.md#term-scope) omitted
  `TensorBinaryArithmeticTest`, even though that test intentionally owns a global public-Tensor
  method count. The first full model run exposed its stale <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [model glossary](../../../../glossary.md#term-model); [run glossary](../../../../glossary.md#term-run))</span>`163` assertion after the four new
  methods made the count `167`. The user explicitly authorized one scope repair: add that test as
  the seventeenth path and change only its global count. No binary-arithmetic assertion changed.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- The four methods use one package-private helper because all validation and result construction
  is shared. <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`sort` and `argsort` still create independent one-output producers; no result carrier
  or shared-output registry was introduced.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [registry glossary](../../../../glossary.md#term-registry))</span>
- Documentation uses **[stable full ordering](../../../../glossary.md#stable-full-ordering)** as the reusable umbrella term and defines
  **[NaN](../../../../glossary.md#term-nan)-last** locally. Stability and NaN placement remain family semantics rather than stored
  attributes.

## Known limitations

- This task constructs model metadata only. It does not evaluate values, define an ordering
  algorithm or gradient, capture a graph, lower to a backend, or execute work.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [metadata glossary](../../../../glossary.md#term-metadata); [gradient glossary](../../../../glossary.md#term-gradient); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend))</span>
- Top-K remains Draft task 0019C1 without a detailed specification. Repository-wide validation is
  deferred to the sorting/top-K capability checkpoint or CI because this task changes one module
  without changing a dependency or architecture boundary.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/task_0019c_implementation` ran the final focused command
  `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.ordering.OrderingSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorSortExpressionTest --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest --tests io.github.pho001.synaptik.model.tensor.TensorTest --tests io.github.pho001.synaptik.model.tensor.TensorMatmulExpressionTest`:
  `BUILD SUCCESSFUL in 934ms`; three actionable tasks, two executed and one up-to-date. Earlier
  focused runs exposed and then verified corrections to test-only gradient-eligibility and
  identifier-exhaustion fixtures.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
- The first implementation-context final `./gradlew :modules:model:test` run failed in one second:
  812 tests completed with one failure,<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
  `TensorBinaryArithmeticTest.helperAndTensorMethodsHaveExactlyTheRequiredShape()` at line 72,
  solely because its unlisted global Tensor-method count remained <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>`163`. Work stopped until the
  user authorized the exact seventeenth path and count-only repair described above.
- After only that [scope](../../../../glossary.md#term-scope) repair, implementation context `/root/task_0019c_implementation` reran
  `./gradlew :modules:model:test`: `BUILD SUCCESSFUL in 1s`; three actionable tasks, two executed
  and one up-to-date. The reports contain 812 tests in 102 suites, zero failures, zero errors, and
  zero skipped tests. No executable Java changed after this successful run, so documentation
  context `/root/task_0019c_implementation/task_0019c_docs` reused the evidence and did not repeat
  the suite.
- Documentation context `/root/task_0019c_implementation/task_0019c_docs` ran
  `./gradlew :modules:model:javadoc` after the initial Javadoc pass and again after the final
  Tensor-overview refinement. Both runs reported <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`BUILD SUCCESSFUL in 1s`; both actionable tasks
  executed on each final generation.
  Generated pages expose `OrderingKind`, `SortAttrs`, all four Tensor methods, the revised Tensor
  overview, complete parameter/result/failure contracts, and no public page for the package-private
  helper.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>
- The documentation context compiled `/tmp/SortArgsortDocExample.java` against
  `modules/model/build/classes/java/main` with `javac`, then ran it. It printed [FLOAT64](../../../../glossary.md#term-float64), [INT64](../../../../glossary.md#data-type--datatype),
  `Operation[kind=SORT, attrs=SortAttrs[axis=0, descending=false]]`, and
  `Operation[kind=ARGSORT, attrs=SortAttrs[axis=0, descending=true]]`, matching the newcomer
  example.
- `javap -classpath modules/model/build/classes/java/main -p` confirmed the exact packages and
  surfaces: `OrderingKind` has SORT then ARGSORT; `SortAttrs` has only axis/descending record
  state; the field-free package-private final helper has one private constructor and one
  package-private static <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>`apply(Tensor, OrderingKind, int, boolean)`; and Tensor has exactly the
  four requested public overloads.
- Initial local-link checker invocations stopped in the temporary checker itself because its first
  version used interpolated Ruby regexp syntax and then a newer-Ruby `filter_map`; a later slug
  attempt also reported four existing glossary anchors because it collapsed adjacent heading
  spaces. After correcting those checker-only issues, the targeted Ruby checker resolved 558 links
  across the seven affected documents, including 153 heading anchors, with zero failures. An
  initial shell formatting loop also treated the zsh newline-delimited file list as one name; the
  corrected `while IFS= read -r` invocation passed. Separate checks passed for balanced fences,
  non-empty files, final newlines, trailing whitespace, terminology, generated-Javadoc content,
  package/type placement, and absence of a 0019C1 specification.
- Final inventory validation found exactly 17 changed paths: four production, six tests, and seven
  documentation/planning files. Status/dependency validation found the single 0019C Ready-to-
  Complete transition, 0019C1/0019D/0019E still Draft, no model task Ready, and no later detailed
  task specification. <span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [dependency glossary](../../../../glossary.md#term-dependency); [Ready glossary](../../../../glossary.md#term-ready); [model glossary](../../../../glossary.md#term-model))</span>`git diff --check` passed.
- Runtime API and Training API remain unchanged because no executable lifecycle or training-
  extension API was added. Compile API changed only to inventory compiler-neutral model
  expressions; it claims no capture, validation, gradient, or support.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [lifecycle glossary](../../../../glossary.md#lifecycle); [Compile glossary](../../../../glossary.md#compile); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- `ARCHITECTURE.md`, focused architecture pages, architecture decision records, architecture
  tests, backend-conformance tests, integration tests, Gradle/dependencies, and every other module
  remain unchanged because model ownership and all dependency/execution boundaries are preserved.
  TensorFactory, producer/provenance foundations, and public result carriers also remain unchanged
  because the existing one-output <span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>`createDerived` seam represents each [occurrence](../../../../glossary.md#term-occurrence) completely.

## Implementation notes

- Added exact `OrderingKind` and `SortAttrs` semantic values and the field-free construction
  helper, then exposed only the four requested Tensor methods.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Tests lock kind/signature and attribute surfaces, stable ordering examples, type/Shape/result
  metadata, valid empty/singleton/dynamic extents, scalar and validation failures, fresh
  single-output provenance, identity behavior, storage independence, and global inventories.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [singleton glossary](../../../../glossary.md#term-singleton); [scalar glossary](../../../../glossary.md#term-scalar); [validation glossary](../../../../glossary.md#term-validation); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- The separate documentation context independently finalized all four production Javadocs, the
  Tensor class overview/import inventory, Tensor and Compile API references, glossary terminology,
  and synchronized capability/task/master-plan/roadmap records. Executable Java was not changed.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Completion summary

- Completed changes: stable full-axis SORT/ARGSORT semantics, exact attributes, four public Tensor
  methods, one shared package-private construction helper, focused/global tests, and complete API,
  glossary, capability, task, master-plan, and roadmap documentation.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Files changed or created: exactly the authorized 17 paths: four production, six tests, and seven
  documentation/planning paths listed under Affected files.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: focused tests passed; the repaired final model suite passed 812 tests in
  102 suites; model Javadoc, documented example, generated pages, bytecode/surface, Markdown
  links/anchors, fences/newlines/whitespace, exact scope, status/dependencies, terminology, and<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` passed.
- Documentation-agent review: clean context
  `/root/task_0019c_implementation/task_0019c_docs` independently finalized the affected
  Javadocs/docs/glossary/planning evidence without changing executable Java or repeating the
  successful model suite.
- Documentation impact: Tensor API explains values/indices, both directions, all-type and
  Shape/metadata/provenance behavior, and current-versus-planned boundaries; Compile API inventories
  only current compiler-neutral expressions. Runtime and Training APIs require no change.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [Compile glossary](../../../../glossary.md#compile); [compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: finalized `OrderingKind`, `SortAttrs`, `TensorSortExpressions`, the Tensor class
  overview, and all four overloads; generated Javadoc passed and was inspected.
- Glossary impact: added stable full ordering and NaN-last terminology with the logical-index and
  current-model boundary.<span data-glossary-navigation="true"> ([stable full ordering glossary](../../../../glossary.md#stable-full-ordering); [NaN glossary](../../../../glossary.md#term-nan))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0019C. Draft task 0019C1 remains the planned top-K frontier.

Status: Complete
