# Task 0019C1: Top-K Values and Indices

**Editorial reading key (current context, not a status update).**

This completed record concerns model construction, not execution; its six-type inventory and
deferred-execution language belong to that historical scope. The current [Top-K
reference](../../../../reference/tensor/sort-topk.md#top-k-values-and-indices) supplies expanded
type descriptions and bounded Metal execution context without changing this task’s meaning. A caller
invokes `input.topK`; the construction helper passes an operation, the input Tensor reference, and
two output descriptors to the factory. These describe a request, not computed values. The returned
storage-free wrappers share producer provenance—an occurrence description—with values at slot zero
and INT64 indices at slot one; they neither execute nor own result buffers.

“Logical index” means an axis coordinate, not a memory offset. For the recorded row, largest-three
selection chooses indices `[0, 4, 3]`; “unsorted” returns those same pairs in index order `[0, 3,
4]`, not arbitrary order. NaNs remain last in either direction, and signed zeros remain distinct.
Model semantics fixes these rules; execution providers realize supported routes rather than redefine
them. Semantic responsibility does not imply storage ownership.

## Status

Complete

## Goal

Add deterministic [axis](../../../../glossary.md#term-axis)-wise top-K [model](../../../../glossary.md#term-model) semantics and public [Tensor](../../../../glossary.md#tensor) construction that returns
selected values and their logical input indices from one genuine two-output [producer](../../../../glossary.md#term-producer).

For planned input `[3.0, NaN, -0.0, +0.0, 3.0]`, `topK(3, 0)` selects logical indices
`[0, 4, 3]` and values `[3.0, 3.0, +0.0]`. The four-[argument](../../../../glossary.md#term-argument) form can request smallest values or
logical-index output order. This example defines model meaning only; this task does not evaluate
values.

## Rationale and selected conventions

Top-K shares task 0019C's total stable ordering but is structurally different from full sort: it
has different attributes and exactly two outputs. `OrderingKind` currently gives every enum
[constant](../../../../glossary.md#term-constant) the same exact `SortAttrs`, one-input, one-output [signature](../../../../glossary.md#term-signature). Adding a differently shaped
`TOP_K` constant would make SORT and ARGSORT accept `TopKAttrs`, or require a cross-family enum
signature redesign. This task therefore preserves the completed contracts unchanged and adds the
focused sibling `TopKKind.TOP_K` in the same `operation.ordering` package.

The selected set is always the first `k` entries of the complete stable order requested by
`largest`. `sorted == true` retains that order. `sorted == false` returns the same selected pairs
in increasing original logical-[axis index](../../../../glossary.md#axis-rank-scalar-vector-and-matrix). Logical-index order is portable, deterministic, cheap
to state, independent of physical traversal, and does not imply an implementation algorithm.

## [Scope](../../../../glossary.md#term-scope)

- Add public `TopKKind.TOP_K` with one exact `TopKAttrs`, one-input, two-output [signature](../../../../glossary.md#term-signature).
- Add public [immutable](../../../../glossary.md#term-immutable) `TopKAttrs(int axis, long k, boolean largest, boolean sorted)` in that exact
  component order.
- Add public `TopKResult(Tensor values, Tensor indices)`.
- Add one package-private, field-free `TensorTopKExpressions` construction helper.
- Add exactly these public instance methods:

  ```java
  TopKResult topK(long k, int axis)
  TopKResult topK(long k, int axis, boolean largest, boolean sorted)
  ```

- Make the two-[argument](../../../../glossary.md#term-argument) form delegate semantically to `largest == true` and `sorted == true`.
- Accept all six current input data types and reuse task 0019C's stable numerical/BOOL/NaN/
  signed-zero ordering.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Replace the selected [Shape](../../../../glossary.md#shape) [Dimension](../../../../glossary.md#dimension) with a fresh `StaticDimension(k)` while preserving every
  unselected Dimension reference.
- Construct exactly one [producer](../../../../glossary.md#term-producer) with input `[input]`, ordered outputs `[values, indices]`, two
  wrappers/IDs, and provenance output indices zero and one.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Fix exact metadata, validation order/messages, static/deferred bounds, identity effects, and
  deterministic sorted/unsorted semantics.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [validation glossary](../../../../glossary.md#term-validation); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Update all global [operation](../../../../glossary.md#operation)-signature and public-[Tensor](../../../../glossary.md#tensor) inventories up front.
- Finalize Javadocs, Tensor API, Compile API, glossary, capability/task/master/roadmap text, and
  evidence through the mandatory separate clean-context documentation pass.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- changes to `OrderingKind`, `SortAttrs`, `TensorSortExpressions`, or any existing `sort`/
  `argsort` [signature](../../../../glossary.md#term-signature), overload, [producer](../../../../glossary.md#term-producer), ordering, or result
- execution algorithms, partial sort, heap/selection networks, kernels, value evaluation, eager
  storage reads/writes, or performance promises<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [kernels glossary](../../../../glossary.md#kernel); [evaluation glossary](../../../../glossary.md#term-evaluation); [eager glossary](../../../../glossary.md#term-eager-evaluation))</span>
- gradients, autograd rules, compiler capture/inference/validation/canonicalization, planning,
  prepare, runtime, backend support, lowering, or conformance behavior<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- optional values or indices outputs, index data-type override, caller-selected stability or NaN
  placement, algorithm options, in-place mutation, output buffers, public alias/mutation APIs, or
  a public producer/sibling-output registry<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [alias glossary](../../../../glossary.md#aliasing-and-arena); [registry glossary](../../../../glossary.md#term-registry))</span>
- a default-[axis](../../../../glossary.md#term-axis) or parameterless overload, flattening, named axes, multiple axes, dynamic `k`,
  percentage K, kth-value, rank, partition, search, or lexicographic ordering<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [partition glossary](../../../../glossary.md#partition))</span>
- changing arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum) [NaN](../../../../glossary.md#term-nan)/tie policy; that reduction family remains independent
- changes to `TensorFactory`, [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), `TensorProvenance`, [`Operation`](../../../../glossary.md#operation), [`OperationKind`](../../../../glossary.md#operationkind),
  `OperationSignature`, [`DataType`](../../../../glossary.md#data-type--datatype), [`Shape`](../../../../glossary.md#shape), [`Dimension`](../../../../glossary.md#dimension), or result carriers other than the new
  `TopKResult`
- [Gradle](../../../../glossary.md#term-gradle)/dependencies, `ARCHITECTURE.md`, focused architecture documents, architecture tests,
  backend-conformance/integration tests, another module, task 0019D/0019E, or any later task<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially model ownership, public
  mutable Tensor state, backend-independent operation semantics, and execution boundaries<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [mutable glossary](../../../../glossary.md#term-mutable); [Tensor glossary](../../../../glossary.md#tensor); [backend glossary](../../../../glossary.md#backend); [operation glossary](../../../../glossary.md#operation))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
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
- [Task 0019C](0019c-sort-and-argsort.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>, [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>,
  [Runtime API](../../../../api/runtime-api.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>, [Training API](../../../../api/training-api.md), and
  [glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns only backend-neutral semantics, Tensor descriptors, public construction,
  and immutable pre-capture occurrence provenance.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Tensor glossary](../../../../glossary.md#tensor); [descriptors glossary](../../../../glossary.md#term-descriptor); [immutable glossary](../../../../glossary.md#term-immutable); [capture glossary](../../../../glossary.md#term-capture-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence); [provenance glossary](../../../../glossary.md#provenance))</span>
- Tensor remains mutable public API state, not an intermediate-representation node. The producer
  is pre-capture occurrence metadata and introduces no graph-local identity.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [node glossary](../../../../glossary.md#node); [producer glossary](../../../../glossary.md#term-producer); [metadata glossary](../../../../glossary.md#term-metadata); [graph glossary](../../../../glossary.md#graph); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `TopKKind` and `TopKAttrs` contain no comparator, algorithm, support table, cost, kernel,
  storage, compiler service, gradient implementation, backend object, or runtime state.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- One call must use the existing `TensorFactory.createDerivedOutputs(...)` seam exactly once.
  One producer retains `[input]` and both ordered descriptors; the factory creates one wrapper and
  ID for every output descriptor.
- The two returned wrappers share the exact producer. No producer retains result Tensor objects,
  and no public carrier reconstructs or discovers siblings.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Stable order is defined over logical axis coordinates and is independent of layout, strides,
  physical traversal, backend, route, and algorithm.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [layout glossary](../../../../glossary.md#layout); [strides glossary](../../../../glossary.md#term-stride))</span>
- Later compiler work owns capture, graph-wide bound validation, gradients, and backward
  construction. Backend prepare later owns algorithms/kernels; runtime later executes prepared
  work.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [prepare glossary](../../../../glossary.md#prepare); [kernels glossary](../../../../glossary.md#kernel))</span>
- If implementation requires an architecture, cross-module, factory/provenance, or existing sort
  contract change, stop and report the conflict without editing.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation.ordering` — owns stable [axis](../../../../glossary.md#term-axis)-ordering semantic values.
- `io.github.pho001.synaptik.model.tensor` — owns public Tensor/result APIs and package-private
  expression construction.
- `io.github.pho001.synaptik.model.datatype` and `.shape` — supply exact [descriptor](../../../../glossary.md#term-descriptor) facts.

Packages added or changed:

- No new package. Extend the established ordering and [tensor](../../../../glossary.md#tensor) packages.

Type placement:

- `io.github.pho001.synaptik.model.operation.ordering.TopKKind` — focused semantic identity whose
  distinct output cardinality does not weaken the exact existing OrderingKind contracts.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [cardinality glossary](../../../../glossary.md#term-cardinality))</span>
- `io.github.pho001.synaptik.model.operation.ordering.TopKAttrs` — immutable normalized axis,
  static K, direction, and output-order request.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [normalized axis glossary](../../../../glossary.md#normalized-axis))</span>
- `io.github.pho001.synaptik.model.tensor.TopKResult` — public shallowly immutable values/indices
  carrier beside the Tensor type it returns.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `io.github.pho001.synaptik.model.tensor.TensorTopKExpressions` — package-private stateless helper
  colocated with Tensor/factory access. It is final, has no fields/nested types/interfaces, one
  private zero-argument constructor, and exactly one declared package-private static<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>
  `apply(Tensor, long, int, boolean, boolean)` method returning `TopKResult`.
- `io.github.pho001.synaptik.model.tensor.Tensor` — owns exactly the two public receiver methods.

Test placement:

- `model.operation.ordering.TopKSemanticsTest` — locks kind, attributes, signature, total
  selection/output policy, and non-execution boundary.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- `model.tensor.TensorTopKExpressionTest` — locks public/helper surfaces, Shape, metadata,
  validation, producer/output slots, identity effects, result carrier, and immutability.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [validation glossary](../../../../glossary.md#term-validation); [producer glossary](../../../../glossary.md#term-producer))</span>
- Existing root signature and Tensor-surface inventory tests remain global owners and are updated
  rather than duplicated.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span>

## Required semantic contracts

### Kind, [signature](../../../../glossary.md#term-signature), and attributes

Create:

```java
public enum TopKKind implements OperationKind {
    TOP_K
}

public record TopKAttrs(
        int axis,
        long k,
        boolean largest,
        boolean sorted) implements OperationAttrs {}
```

`TOP_K` accepts exactly:

```java
OperationSignature.fixed(TopKAttrs.class, 1, 2)
```

`TopKKind` exposes only the established family-owned stable [immutable](../../../../glossary.md#term-immutable) signature list. `TopKAttrs`
retains all four exact values in component order. Its [canonical](../../../../glossary.md#term-canonical-form) constructor validates [axis](../../../../glossary.md#term-axis) then K:

```text
axis must be non-negative: <axis>
k must be non-negative: <k>
```

[Tensor](../../../../glossary.md#tensor) construction always supplies a [normalized axis](../../../../glossary.md#normalized-axis), so the public helper's observable local
order remains axis [normalization](../../../../glossary.md#term-normalization) before K [validation](../../../../glossary.md#term-validation). The record performs no [rank](../../../../glossary.md#term-rank), selected-
[extent](../../../../glossary.md#term-extent), Tensor, [descriptor](../../../../glossary.md#term-descriptor), value, [layout](../../../../glossary.md#layout), or execution validation. It explicitly documents/
overrides all four generated accessors and adds no other public behavior.

`OrderingKind` remains exactly `SORT, ARGSORT`, each with exact `SortAttrs`, one input, and one
output. `TensorSortExpressions` therefore needs no TOP_K exclusion branch.

### Public [API](../../../../glossary.md#term-application-programming-interface-api) and result [carrier](../../../../glossary.md#term-storage-carrier)

Add only:

```java
public TopKResult topK(long k, int axis)
public TopKResult topK(long k, int axis, boolean largest, boolean sorted)
```

[Argument](../../../../glossary.md#term-argument) names and order are exact. The shorter form is exactly
`topK(k, axis, true, true)`. There is no default [axis](../../../../glossary.md#term-axis).

Create:

```java
public record TopKResult(Tensor values, Tensor indices) {}
```

The compact constructor checks `values` then `indices` with
`Objects.requireNonNull(component, "component")`, retains the exact references, is shallowly
[immutable](../../../../glossary.md#term-immutable), and uses ordinary record value equality, hashing, and diagnostic text. It performs no
[descriptor](../../../../glossary.md#term-descriptor), [producer](../../../../glossary.md#term-producer), [provenance](../../../../glossary.md#provenance), [shape](../../../../glossary.md#shape), [identity](../../../../glossary.md#term-identity-meanings), storage, or sibling-consistency [validation](../../../../glossary.md#term-validation) and
exposes no mutation or [alias](../../../../glossary.md#aliasing-and-arena)-management API.

### Input eligibility and output [metadata](../../../../glossary.md#term-metadata)

All current types succeed: [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), [BFLOAT16](../../../../glossary.md#term-bfloat16), [INT32](../../../../glossary.md#data-type--datatype), [INT64](../../../../glossary.md#data-type--datatype), and [BOOL](../../../../glossary.md#data-type--datatype). Floating and signed
integral values use the ordering below; BOOL uses `false < true`.

For input [descriptor](../../../../glossary.md#term-descriptor) `(inputType, inputShape, inputLayout, inputRequiresGrad)`:

```text
output slot 0, values:
  data type: exact inputType
  Shape: derived top-K Shape
  layout: unresolved
  requiresGrad: exact inputRequiresGrad
  label/storage: absent

output slot 1, indices:
  data type: INT64
  Shape: the exact same derived Shape reference used by values
  layout: unresolved
  requiresGrad: false
  label/storage: absent
```

Values retaining `requiresGrad` is metadata only; this task defines no [gradient](../../../../glossary.md#term-gradient) rule. Indices are
non-[differentiable](../../../../glossary.md#derivative-and-chain-rule). Neither output [aliases](../../../../glossary.md#aliasing-and-arena) input storage or preserves resolved input [layout](../../../../glossary.md#layout).

### [Shape](../../../../glossary.md#shape) and bound contract

Normalize [`axis`](../../../../glossary.md#term-axis) once through the exact input Shape. Construct one fresh result Shape by copying
the input's ordered [Dimension](../../../../glossary.md#dimension) references, replacing only `normalizedAxis` with one fresh
`StaticDimension(k)`, and calling `Shape.ofDimensions(...)`. Both [descriptors](../../../../glossary.md#term-descriptor) retain the exact
same result Shape reference.

- [Rank](../../../../glossary.md#term-rank) is unchanged and every unselected `StaticDimension`, `DynamicDimension`, and
  `ExpressionDimension` reference is preserved exactly.
- The selected input Dimension may be static, named dynamic, or expression-based; its reference is
  replaced rather than retained.
- [Scalar](../../../../glossary.md#term-scalar) input rejects every axis through the existing Shape exception before K [validation](../../../../glossary.md#term-validation).
- `k == 0` is valid. The selected output axis is static zero even when the input [extent](../../../../glossary.md#term-extent) is dynamic.
- A selected `StaticDimension` requires `k <= extent` at construction. Static extent zero accepts
  only K zero. Other empty, unselected axes remain valid and preserved.
- A selected dynamic or expression extent is accepted without [binding](../../../../glossary.md#term-binding). The semantic constraint
  `bound extent >= k` is deferred to later compiler/binding validation; a conforming execution
  must reject an insufficient binding and must not clamp, pad, wrap, or return fewer outputs.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler))</span>
- `k` may be `Long.MAX_VALUE` when the selected extent is dynamic/expression or a sufficiently
  large static extent. Construction does not materialize element counts or require a Java array.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>

Known static [overflow](../../../../glossary.md#term-overflow) is impossible for the single K-vs-extent comparison. No product of Shape
extents is computed.

### Stable selection order and output order

Apply selection independently to each logical [slice](../../../../glossary.md#slice) along the [normalized axis](../../../../glossary.md#normalized-axis). Define the
complete stable selection order as follows:

- For `largest == true`, non-[NaN](../../../../glossary.md#term-nan) values use descending numerical order. For `largest == false`,
  they use ascending numerical order.
- Negative infinity, finite values, and positive infinity follow ordinary numerical order before
  direction reversal.<span data-glossary-navigation="true"> ([Negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [finite glossary](../../../../glossary.md#term-finite); [positive infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Negative zero is below positive zero in ascending/smallest order and above it after reversal for
  descending/largest order. They are not a stability tie.<span data-glossary-navigation="true"> ([Negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Every NaN is one final class after every non-NaN for both largest and smallest requests. Thus
  NaNs enter the selected set only if fewer than K non-NaNs exist.
- Equal finite, integral, and BOOL values and multiple NaNs are ordered by increasing original
  logical-axis index. Equal boundary candidates therefore prefer lower logical indices.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- The selected set is exactly the first K entries of this complete order. Duplicates are retained;
  selected values preserve their exact input representations, including NaN payload bits and
  signed zero.<span data-glossary-navigation="true"> ([NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [signed zero glossary](../../../../glossary.md#term-signed-zero))</span>

When `sorted == true`, output pairs stay in the complete stable selection order. When
`sorted == false`, output pairs are reordered by increasing original logical-[axis index](../../../../glossary.md#axis-rank-scalar-vector-and-matrix). The set
does not change, values and indices remain paired, and this order is deterministic for every type,
direction, duplicate pattern, NaN pattern, [layout](../../../../glossary.md#layout), and [backend](../../../../glossary.md#backend). `sorted` never permits a
backend-dependent [permutation](../../../../glossary.md#permutation).

Examples of planned semantics:

```text
input: [3.0, NaN, -0.0, +0.0, 3.0]

largest=true,  k=3, sorted=true  -> values [3.0, 3.0, +0.0], indices [0, 4, 3]
largest=true,  k=3, sorted=false -> values [3.0, +0.0, 3.0], indices [0, 3, 4]
largest=false, k=3, sorted=true  -> values [-0.0, +0.0, 3.0], indices [2, 3, 0]
largest=false, k=3, sorted=false -> values [3.0, -0.0, +0.0], indices [0, 2, 3]
```

These are semantic outputs, not an algorithm prescription. The policy is deliberately consistent
with task 0019C: top-K selects a prefix of the same stable ascending order for smallest or the same
stable descending order for largest.

## Construction, [provenance](../../../../glossary.md#provenance), [validation](../../../../glossary.md#term-validation), and ID contract

`TensorTopKExpressions.apply(input, k, axis, largest, sorted)` validates in this exact order:

1. non-null `input`, else `NullPointerException("input")`;
2. obtain the exact input [Shape](../../../../glossary.md#shape) and normalize [`axis`](../../../../glossary.md#term-axis) through `Shape.normalizeAxis(int)`;
3. construct `TopKAttrs(normalizedAxis, k, largest, sorted)`, thereby reject negative K with
   `k must be non-negative: <k>`;
4. inspect only the selected [Dimension](../../../../glossary.md#dimension)'s `staticSize()`; when present and `k > extent`, reject
   with `k must not exceed selected static extent: k=<k>, axis=<normalizedAxis>, extent=<extent>`;
5. construct the one result Shape, values [descriptor](../../../../glossary.md#term-descriptor), indices descriptor, and [operation](../../../../glossary.md#operation); and
6. delegate exactly once to `TensorFactory.createDerivedOutputs(...)` with operation,
   exact ordered input list `[input]`, and exact ordered descriptors `[values, indices]`.

There is no data-type rejection and booleans need no special construction branch. Every local
failure, [scalar](../../../../glossary.md#term-scalar)/axis failure, negative-K failure, and known static-bound failure occurs before
factory delegation and consumes no [Tensor](../../../../glossary.md#tensor) ID.

One successful call creates exactly one `Operation(TopKKind.TOP_K, attrs)`, one [producer](../../../../glossary.md#term-producer), two
descriptors, two provenance values, two wrappers, two fresh IDs in output order, and one result
record. Values has output index zero; indices has output index one. Both provenance objects retain
the exact same producer, whose exact ordered input [snapshot](../../../../glossary.md#term-snapshot) is `[input]` and descriptor snapshot
is `[valuesDescriptor, indicesDescriptor]`. The helper returns
`new TopKResult(outputs.get(0), outputs.get(1))` and does not recreate either output.

Repeated equal calls create distinct producers, operations, attributes, descriptors, Shapes,
wrappers, IDs, provenance, and result records. The exact input Tensor and unselected Dimension
references are retained. Input descriptor, [layout](../../../../glossary.md#layout), label, storage, provenance, and values are not
mutated or accessed.

Identifier exhaustion occurs only after all local validation and producer/[signature](../../../../glossary.md#term-signature) validation.
If the second allocation fails, the first ID remains consumed and no partial list or result is
returned, exactly as `createDerivedOutputs(...)` specifies. Allocation is not rolled back or
translated, and concurrent callers must not infer adjacent numeric IDs.

## Affected files

Expected production Java:

- new `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/ordering/TopKKind.java`
- new `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/ordering/TopKAttrs.java`
- new `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TopKResult.java`
- new `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorTopKExpressions.java`
- existing `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected [model](../../../../glossary.md#term-model) tests:

- new `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/ordering/TopKSemanticsTest.java`
- new `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTopKExpressionTest.java`
- existing `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
  for the authoritative production family/signature inventory<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- existing `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
  for the authoritative public-method count/name inventory; exact count becomes 169
- existing `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
  for its intentionally global Tensor count; exact count becomes 169 only
- existing `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
  for its intentionally global Tensor count; exact count becomes 169 only

Expected documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Mandatory inspected seams/inventories, changed only if an acceptance criterion proves this [scope](../../../../glossary.md#term-scope)
insufficient:

- `OrderingKind`, `SortAttrs`, `TensorSortExpressions`, focused sort tests, and task 0019C
- `TensorFactory.createDerivedOutputs(...)`, [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), `TensorProvenance`, and focused tests
- `DropoutResult`, [dropout](../../../../glossary.md#dropout--inverted-dropout) construction/tests, and task 0019B1 for actual [carrier](../../../../glossary.md#term-storage-carrier)/wrapper/ID behavior
- [`Operation`](../../../../glossary.md#operation), [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), `OperationSignature`, and inventory tests
- [`DataType`](../../../../glossary.md#data-type--datatype), [`Shape`](../../../../glossary.md#shape), [`Dimension`](../../../../glossary.md#dimension), `StaticDimension`, `DynamicDimension`,
  `ExpressionDimension`, [descriptor](../../../../glossary.md#term-descriptor)/Shape tests, and representative Shape-replacement helpers
- comparison and arg-extrema semantics/tests for integral, BOOL, NaN, infinity, signed-zero, tie,
  and logical-index terminology<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [BOOL glossary](../../../../glossary.md#data-type--datatype); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- [Tensor](../../../../glossary.md#tensor), [Compile](../../../../glossary.md#compile), [Runtime](../../../../glossary.md#term-runtime), and Training [API](../../../../glossary.md#term-application-programming-interface-api) references plus the glossary
- architecture/focused-architecture docs, Gradle, architecture/conformance/integration tests, and
  other modules for reasoned no-change conclusions<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify exactly the 18 expected paths above: five production, six test,
and seven documentation/[planning](../../../../glossary.md#term-planning) paths. Do not change an inspected seam merely to consume scope.

No existing sort Java file, shared factory/[provenance](../../../../glossary.md#provenance)/[operation](../../../../glossary.md#operation)/[Shape](../../../../glossary.md#shape) foundation, Java path outside
`modules/model`, [dependency](../../../../glossary.md#term-dependency)/[Gradle](../../../../glossary.md#term-gradle) file, architecture document/test, [conformance](../../../../glossary.md#term-conformance)/[integration test](../../../../glossary.md#term-integration-test),
other [module](../../../../glossary.md#term-module), or later task specification may change. If more than 18 paths or another concept is
required, stop and propose a focused follow-up.

## [Javadoc](../../../../glossary.md#term-javadoc) and explanatory documentation requirements

- Give `TopKKind`, `TopKAttrs`, `TopKResult`, `TensorTopKExpressions`, and both Tensor overloads
  meaningful Javadoc covering all inputs, results, nullability, reference retention, equality,
  normalized axes, K bounds, dynamic deferral, all types, stable selection, NaNs, signed zero,
  infinities, ties, sorted/unsorted order, Shape/metadata/provenance, ID effects, failures, and
  compiler/backend/execution boundaries.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [nullability glossary](../../../../glossary.md#term-nullability); [axes glossary](../../../../glossary.md#term-axis); [signed zero glossary](../../../../glossary.md#term-signed-zero); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- Update Tensor API with a current construction example, output table, four ordering examples,
  all-type eligibility, Shape/reference/metadata/provenance behavior, dynamic-bound contract, and
  honest current-versus-planned execution text.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Update Compile API only to inventory top-K as a current compiler-neutral model expression and
  explain the deferred dynamic bound as future compiler validation. Do not claim capture,
  inference, gradients, support, or execution.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model); [validation glossary](../../../../glossary.md#term-validation); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Update the glossary for reusable top-K/selected-set terminology and deterministic unsorted
  logical-index order without duplicating the full API contract.
- Runtime API and Training API are expected no-change because this task adds no executable or
  training-extension API. Record the reasons.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Architecture docs/ADRs/tests, conformance/integration tests, Gradle, other modules, and existing
  sort contracts are expected no-change because ownership and execution boundaries remain fixed.
  Record each conclusion.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Acceptance criteria

- Exact public surface is the two `topK` methods and `TopKResult(values, indices)` listed above;
  no default-axis, optional-output, index-type, stability, NaN, algorithm, alias, or mutation API
  is added.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [NaN glossary](../../../../glossary.md#term-nan); [alias glossary](../../../../glossary.md#aliasing-and-arena); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `TopKKind` has exactly `TOP_K` and exact fixed `TopKAttrs` one-input/two-output [signature](../../../../glossary.md#term-signature).
  `OrderingKind` and current sort/argsort contracts remain byte-for-byte unchanged.
- `TopKAttrs` has only exact ordered components `(int axis, long k, boolean largest, boolean
  sorted)`, exact [validation](../../../../glossary.md#term-validation)/messages, accessors, [immutable](../../../../glossary.md#term-immutable) value semantics, and no algorithm.
- `TopKResult` checks null values then indices, retains both exact wrappers, uses record value
  semantics, and performs no cross-component reconstruction or validation.
- Every current DataType succeeds. Values retain input type/requiresGrad; indices are INT64/false;
  both use the same exact derived Shape, unresolved layout, no label/storage.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout))</span>
- Shape derivation preserves rank and every unselected Dimension reference, replaces the selected
  Dimension with fresh static K, and handles static, dynamic, expression, scalar, zero-K, empty,
  singleton, and <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [Dimension glossary](../../../../glossary.md#dimension); [scalar glossary](../../../../glossary.md#term-scalar); [singleton glossary](../../../../glossary.md#term-singleton))</span>`Long.MAX_VALUE` cases exactly as specified.
- Static K bounds fail locally; dynamic/expression bounds are explicitly deferred without clamp,
  padding, truncation, or backend-dependent meaning.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [backend glossary](../../../../glossary.md#backend))</span>
- Stable largest/smallest selection, prefix-of-task-0019C order, equal boundary ties, NaNs-last,
  infinities, signed zero, BOOL, duplicate representation, sorted order, and deterministic
  unsorted logical-index order match this specification.<span data-glossary-navigation="true"> ([infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [signed zero glossary](../../../../glossary.md#term-signed-zero); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- One call creates one exact `[input]` [producer](../../../../glossary.md#term-producer), [descriptors](../../../../glossary.md#term-descriptor) `[values, indices]`, two wrappers/IDs,
  and output indices zero/one. Validation and exhaustion effects match the exact contract.
- The helper has the exact field-free one-method surface, does not mutate/access input storage,
  and delegates to `createDerivedOutputs(...)` exactly once.
- No value evaluation, algorithm, gradient, compiler, planning, backend, runtime, or execution
  behavior is implemented or claimed.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Global signature inventory and all three global Tensor method-count inventories are updated up
  front and lock exact count 169.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Exactly one final `./gradlew :modules:model:test` run occurs after executable Java stabilizes.
  The documentation pass reuses it unless executable behavior changes afterward.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- The separate documentation-focused pass finalizes Javadocs, Tensor/Compile API, glossary,
  planning/evidence, and no-change conclusions, then runs model Javadoc/documentation checks.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Task/master/roadmap show 0019C1 Complete after implementation and validation; 0019 through
  0019C remain Complete; 0019D, 0019E, and every later task remain Draft without detailed
  specifications. No model task is Ready because the plan defines no next detailed frontier.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) the new semantic/expression tests and directly affected [signature](../../../../glossary.md#term-signature),
public-surface, factory/[provenance](../../../../glossary.md#provenance), [Shape](../../../../glossary.md#shape), and result-[carrier](../../../../glossary.md#term-storage-carrier) tests as needed. Test semantic tables
as representation-policy fixtures without pretending [model](../../../../glossary.md#term-model) construction evaluates values.

After executable Java stabilizes, run exactly one final model suite:

```bash
./gradlew :modules:model:test
```

The separate clean-context documentation pass reuses that result and runs after final [Javadoc](../../../../glossary.md#term-javadoc)/
documentation edits:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

It must also [compile](../../../../glossary.md#compile)/run the documented construction example against current model classes;
inspect generated Javadoc; validate local Markdown links and anchors, balanced fences, final
newlines, trailing whitespace, terminology, exact 18-path [scope](../../../../glossary.md#term-scope), package/type placement,
status/[dependency](../../../../glossary.md#term-dependency) coherence, no [Ready](../../../../glossary.md#term-ready) frontier after completion, 0019D/0019E Draft, and absence of later
detailed task specifications.

Task 0019C1 is the sorting/top-K capability [checkpoint](../../../../glossary.md#term-checkpoint) named by task 0019C. After the implementation
and documentation passes are stable, run once:

```bash
./gradlew test
```

Architecture tests run only if a dependency/boundary changes; that is not expected and otherwise
requires stopping. [Backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance)/integration tests remain deferred until executable support.

## Dependencies

- Task 0019C — stable full sort/argsort ordering foundation: Complete.
- Task 0018L — shared multi-output [producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance)/factory seam: Complete.
- Task 0018K — exact [operation](../../../../glossary.md#operation) [signatures](../../../../glossary.md#term-signature)/construction [validation](../../../../glossary.md#term-validation): Complete.
- Tasks 0018U and 0018U1 — integral/[BOOL](../../../../glossary.md#data-type--datatype) order plus [NaN](../../../../glossary.md#term-nan)/zero/tie/index vocabulary: Complete.
- Task 0019B1 — production use of wrapper-per-output construction and public result carrier:
  Complete.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Tasks 0001–0002, 0005–0007, and 0011–0013 — DataType, Shape, Operation, Tensor, descriptor,
  factory, and provenance foundations: Complete.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Current global signature and [Tensor](../../../../glossary.md#tensor) public-surface inventories: inspected and named in [scope](../../../../glossary.md#term-scope).

## Follow-up tasks

- 0019D — [linear](../../../../glossary.md#term-linear-meanings) convenience; remains Draft and retains its established [dependency](../../../../glossary.md#term-dependency)/order.
- 0019E — [scaled dot-product attention](../../../../glossary.md#scaled-dot-product-attention); remains Draft and retains established dependencies/order.
- Later compiler/autograd work owns capture, dynamic-bound enforcement, and any valid values-
  gradient construction.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Later [backend](../../../../glossary.md#backend)/[runtime](../../../../glossary.md#term-runtime) work owns algorithms, [kernels](../../../../glossary.md#kernel), execution, and [conformance](../../../../glossary.md#term-conformance).

No follow-up may weaken deterministic unsorted order, make outputs optional, or retrofit TOP_K into
`OrderingKind` without a separate explicit compatibility task.

## Architecture impact

Expected impact: None.

This task uses existing [model](../../../../glossary.md#term-model)-owned [operation](../../../../glossary.md#operation), [Shape](../../../../glossary.md#shape), [Tensor](../../../../glossary.md#tensor), result, and multi-output [provenance](../../../../glossary.md#provenance)
contracts. It changes no [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings), [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [lifecycle](../../../../glossary.md#lifecycle), factory architecture,
or execution responsibility. If implementation reveals otherwise, stop and report the exact
conflict.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, docs/developer-guide/documentation-rules.md,
docs/planning/planning-guide.md, docs/planning/modules/model/capabilities.md, the model master plan,
roadmap, completed tasks 0018K/0018L/0018U/0018U1/0019B1/0019C, and task 0019C1 in full. Inspect
every affected source/test and mandatory inventory named by 0019C1.

Implement task 0019C1 exactly as specified, only in modules/model and the 18 authorized paths.
Stop on an architecture, cross-module, factory/provenance, existing-sort, semantic, validation-
order, or scope conflict. Do not implement algorithms, gradients, compiler/runtime/backend
behavior, or another task. Run focused tests while developing and exactly one final model suite
after executable Java stabilizes.

Then hand the actual diff and recorded Java-test evidence to a separate clean-context
documentation-focused agent in the same overall change. That agent must independently inspect the
final contracts, finalize Javadocs, Tensor/Compile API, glossary, planning/status/evidence and
no-change conclusions, run model Javadoc and documentation validation, and reuse the successful
Java evidence unless it changes executable behavior. Run the sorting/top-K repository checkpoint
only after both passes. Do not mark Complete until all acceptance criteria pass. Do not commit or
push.
```

## Separate documentation [handoff](../../../../glossary.md#term-handoff)

The implementation agent must provide the documentation agent: this task file; actual diff; exact
final [model](../../../../glossary.md#term-model)-test command/result; two-method/result [API](../../../../glossary.md#term-application-programming-interface-api); stable selected-set and deterministic
sorted/unsorted semantics; exact [Shape](../../../../glossary.md#shape)/type/[provenance](../../../../glossary.md#provenance)/[validation](../../../../glossary.md#term-validation)/ID contracts; no-algorithm/no-
execution/[compiler](../../../../glossary.md#term-compiler)-owned-[gradient](../../../../glossary.md#term-gradient)/dynamic-bound boundaries; expected [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) API and
glossary changes; [Runtime](../../../../glossary.md#term-runtime)/Training/architecture/[conformance](../../../../glossary.md#term-conformance)/integration/[Gradle](../../../../glossary.md#term-gradle)/other-[module](../../../../glossary.md#term-module) and
existing-sort no-change reviews; and all documentation validation commands above. The docs agent
must report its clean context ID and whether executable Java changed after reused test evidence.

## Local decisions

- Use focused `TopKKind.TOP_K`, not `OrderingKind.TOP_K`. This preserves SORT/ARGSORT's exact
  `SortAttrs`, one-input, one-output [signature](../../../../glossary.md#term-signature) without redesigning family-wide signature [dispatch](../../../../glossary.md#term-dispatch).
- `sorted == false` returns selected pairs in increasing original logical-axis index. It is a
  deterministic semantic order, not permission for backend-dependent output.<span data-glossary-navigation="true"> ([axis index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [backend glossary](../../../../glossary.md#backend))</span>
- Both output descriptors share one exact derived Shape reference. Only the selected input
  Dimension is replaced; all unselected Dimension references are preserved.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension))</span>

## Known limitations

- This Ready task specifies model construction only. No current compiler captures TOP_K, validates
  a bound dynamic extent, constructs gradients, or executes it.<span data-glossary-navigation="true"> ([captures glossary](../../../../glossary.md#term-capture-meanings))</span><span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler); [extent glossary](../../../../glossary.md#term-extent); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Backend/runtime algorithms and numerical execution remain future work. The semantic result is
  portable even though no implementation strategy is selected.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context `/root/plan_0019c1` inspected the [architecture contract](../../../../glossary.md#architecture-contract) and focused [lifecycle](../../../../glossary.md#lifecycle),
[module](../../../../glossary.md#term-module), [dependency](../../../../glossary.md#term-dependency), and training documents; documentation rules and General/[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc)/Planning/
Example profiles; planning guide, roadmap, capabilities, and [model](../../../../glossary.md#term-model) master plan; completed tasks
0018K/0018L/0018U/0018U1/0019B1/0019C; current ordering, [Tensor](../../../../glossary.md#tensor), factory/[provenance](../../../../glossary.md#provenance), [operation](../../../../glossary.md#operation),
[Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension), [descriptor](../../../../glossary.md#term-descriptor), result-[carrier](../../../../glossary.md#term-storage-carrier) source/tests; global [signature](../../../../glossary.md#term-signature)/public-method
inventories; Tensor/[Compile](../../../../glossary.md#compile)/[Runtime](../../../../glossary.md#term-runtime)/Training APIs; and glossary.

Planning validation resolved 392 local links across the four changed files with zero failures;
all four files passed final-newline, LF-ending, balanced-fence, and whitespace checks. Inventory
checks found exactly one [Ready](../../../../glossary.md#term-ready) master-plan row and exactly one Ready detailed task, both 0019C1;
0019 through 0019C remain Complete, 0019D/0019E and later rows remain Draft, and no later detailed
specification exists. The final changed-path inventory contains exactly this new task plus
`capabilities.md`, `master-plan.md`, and `roadmap.md`; no Java, test, [Gradle](../../../../glossary.md#term-gradle), architecture,
focused-architecture, other-module, or API/glossary path changed. `git diff --check` passed.

Implementation context `/root/task_0019c1_implementation` ran the final executable model suite
after Java behavior stabilized: `./gradlew :modules:model:test` was `BUILD SUCCESSFUL in 1s`.
The XML reports aggregate 104 suites and 827 tests, with zero failures, errors, or skips.

Clean documentation-focused context
`/root/task_0019c1_implementation/task_0019c1_docs` independently reviewed the architecture and
documentation contracts, General/API-Javadoc/Planning/Example profiles, planning baseline and
completed prerequisite tasks, final source/tests and actual diff, mandatory factory/provenance,
operation, Shape/Dimension, result-carrier and ordering seams, Tensor/Compile/Runtime/Training API
references, glossary, capability/master/task/roadmap state, and package/type placement. It changed
only Javadoc comments and the seven authorized documentation/planning paths after the recorded
model [run](../../../../glossary.md#term-run); executable Java did not change, so it reused the 827-test evidence.

The documentation context ran `./gradlew :modules:model:javadoc` after final Javadoc edits. An
initial successful run reported four missing-main-description warnings on `TopKAttrs` accessors;
after correcting those comments, the final run was `BUILD SUCCESSFUL in 1s` with no warnings and
two executed tasks. Generated pages for `TopKKind`, `TopKAttrs`, `TopKResult`, and both Tensor
overloads were inspected for the two-output, dynamic-bound, logical-order, and lifecycle
boundaries.

The Tensor API construction example was compiled with
`javac -cp modules/model/build/classes/java/main -d /tmp/synaptik-0019c1-doc-example
/tmp/TopKConstructionExample.java` and run with the current model classes. It printed the seven
documented lines: `Shape[2, 3]`, [`FLOAT64`](../../../../glossary.md#term-float64), [`INT64`](../../../../glossary.md#data-type--datatype), `true`, `0`, `1`, and `true`.

Targeted Markdown validation across the seven changed documentation/planning files resolved 565
local links, including 154 heading anchors, with zero failures. The same check found balanced
fences, final LF newlines, no carriage returns, and no trailing whitespace. [Scope](../../../../glossary.md#term-scope) inspection found
exactly 18 paths: five production, six test, and seven documentation/planning paths. Package/type
placement matches the established ordering and tensor packages; 0019C1 is Complete everywhere,
0019D/0019E and later work remain Draft without detailed specifications, and no model task is
Ready because no next detailed frontier exists. Final `git diff --check` passed.

The named sorting/top-K [checkpoint](../../../../glossary.md#term-checkpoint) then ran once with `./gradlew test`: `BUILD SUCCESSFUL in 1s`;
36 actionable tasks, two executed and 34 up-to-date. No dependency or architecture boundary
changed, so no focused architecture-test change or separate architecture command was required.

Runtime API and Training API remain unchanged because this task adds no prepared/runtime or
training-extension API. Existing sort contracts remain unchanged because focused `TopKKind`
preserves `OrderingKind`, `SortAttrs`, and `TensorSortExpressions`. `ARCHITECTURE.md`, focused
architecture documents, ADRs/tests, [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), integration tests, Gradle/dependencies,
and other modules remain unchanged because model [ownership](../../../../glossary.md#term-ownership-meanings) and all compile/[prepare](../../../../glossary.md#prepare)/run boundaries
are preserved. Backend conformance remains deferred until executable top-K support exists.

## Implementation notes

- Added focused `TopKKind.TOP_K` and [immutable](../../../../glossary.md#term-immutable) normalized `TopKAttrs` with an exact one-input,
  two-output signature and no algorithm or execution state.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- Added `TensorTopKExpressions`, `TopKResult`, and exactly two public Tensor overloads. One call
  derives shared result Shape/metadata and creates one producer with values/indices slots zero and
  one while preserving validation and ID-allocation effects.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [producer glossary](../../../../glossary.md#term-producer); [validation glossary](../../../../glossary.md#term-validation))</span>
- Added focused semantic and Tensor-construction tests and updated the global signature and public
  Tensor inventories to exact count 169.
- Finalized all affected Javadocs, Tensor and Compile API references, glossary terminology,
  capabilities, task evidence, model master plan, and roadmap without changing executable Java
  during the documentation pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>

## Completion summary

- Completed changes: added deterministic largest/smallest top-K model semantics, static/deferred K
  validation, exact output metadata, public values/indices construction, and shared two-output
  provenance.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- Files changed or created: exactly five production Java, six model-test, and seven
  documentation/planning paths authorized by this task.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: reused the final 104-suite/827-test model result with zero failures,
  errors, or skips; clean model Javadoc, the compiled/running documentation example, generated-page
  inspection, 565-link/154-anchor Markdown and formatting checks, exact-scope/status audits, the
  repository checkpoint, and <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>`git diff --check` passed.
- Documentation-agent review: clean context
  `/root/task_0019c1_implementation/task_0019c1_docs` completed the independent General,
  API/Javadoc, Planning, and Example-profile pass.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Documentation impact: Tensor API now explains the complete two-method/result contract and four
  deterministic orders; Compile API records current compiler-neutral metadata and deferred dynamic
  bounds without claiming support. Runtime and Training APIs require no change.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Javadoc review: all five affected production paths were reviewed and finalized; executable Java
  did not change after the recorded model suite.
- Glossary impact: added reusable top-K selected-set and deterministic unsorted logical-index-order
  terminology.
- Unresolved issues: None.
- Follow-up required: None for task 0019C1. Tasks 0019D/0019E remain Draft.

Status: Complete
