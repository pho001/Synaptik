# Task 0016H: Cumulative-Sum [Tensor](../../../../glossary.md#tensor) Expressions

**Editorial reading key (later context, not a status update).**

A cumulative sum preserves every axis position and reports partial totals, unlike a reduction that removes a domain. The application supplies an axis and optional exclusive/reverse flags. The helper preserves input shape/type and records one-input provenance without reading or summing values. The current cumulative-scan names and product sibling are later context, not new capabilities of this task. The application calls the public Tensor method; its model helper validates available descriptions and returns a fresh storage-free expression with exact input references. It neither copies input values nor calculates the result. Compiler consumes this provenance later; actual storage and execution are separate downstream concerns.

**Scan reading example.** For an exact real sequence $x=(1,2,3)$, forward inclusive cumulative sum has $y_k=\sum_{j=0}^{k}x_j$ for $k\in\{0,1,2\}$, so $y=(1,3,6)$. Forward exclusive uses $y_k=\sum_{j=0}^{k-1}x_j$, with an empty sum of zero, giving $(0,1,3)$. Both keep all three positions; these ideal sums add no floating-point evaluation or execution claim.

Focused current reading: [Current cumulative-scan modes](../../../../reference/tensor/fixed-scans.md#cumulative-scan-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Add the public [model](../../../../glossary.md#term-model)-level expression boundary for a one-[axis](../../../../glossary.md#term-axis) cumulative sum. A caller must be
able to request the inclusive forward default or explicitly select exclusive and reverse modes.
Construction validates local [metadata](../../../../glossary.md#term-metadata), normalizes the axis, preserves the input [Shape](../../../../glossary.md#shape) and [data
type](../../../../glossary.md#data-type--datatype), records exact one-input [provenance](../../../../glossary.md#provenance), and returns a fresh storage-free [Tensor](../../../../glossary.md#tensor) without reading
or accumulating values.

This task completes public expression construction for the semantic contracts introduced by task
0016G. It does not execute a [scan](../../../../glossary.md#term-cumulative-scan), define a [gradient](../../../../glossary.md#term-gradient) rule, [capture](../../../../glossary.md#term-capture-meanings) a [compiled graph](../../../../glossary.md#compiled-graph--compiledgraphmodel), or report
[backend](../../../../glossary.md#backend) support.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly `Tensor.cumSum(int axis)` and
  `Tensor.cumSum(int axis, boolean exclusive, boolean reverse)`.
- Make the short overload explicitly select inclusive forward semantics by passing
  `exclusive=false` and `reverse=false`.
- Add one package-private final `TensorCumulativeSumExpressions` construction boundary in the
  existing tensor package.
- Accept exactly the five current numeric data types: FLOAT64, FLOAT32, BFLOAT16, INT32, and
  INT64.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype))</span>
- Reject [BOOL](../../../../glossary.md#data-type--datatype) without promotion, conversion, or truthiness semantics.
- Normalize one positive or negative [axis](../../../../glossary.md#term-axis) exactly once through the exact input [Shape](../../../../glossary.md#shape).
- Retain the exact input Shape and [data type](../../../../glossary.md#data-type--datatype) in a new unresolved-[layout](../../../../glossary.md#layout) [descriptor](../../../../glossary.md#term-descriptor).
- Preserve the input descriptor's `requiresGrad` eligibility metadata. Current integral inputs
  necessarily carry false eligibility; this task adds no gradient rule.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Construct exact `CumulativeSumKind.CUM_SUM` and `CumulativeSumAttrs` [metadata](../../../../glossary.md#term-metadata).
- Record exact ordered one-input [provenance](../../../../glossary.md#provenance) `[input]` and create one fresh derived Tensor with no
  label or host storage.<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage))</span>
- Update only the exact [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) surface assertion in `TensorTest` and add one focused expression
  test.
- Finalize Javadocs, Tensor API, Compile API current-expression inventory, glossary, task evidence,
  master plan, and roadmap through the required independent documentation pass during
  implementation.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- another public Tensor method, overload, static factory, builder, alias, or task-0016I
  specification<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- cumulative product, minimum, maximum, logical scan, segmented scan, prefix count, rolling
  window, or other scan family<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- a no-axis, full-tensor, multiple-axis, axis-list, named-axis, keep-dimensions, output-type, or
  destination form<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [dimensions glossary](../../../../glossary.md#dimension))</span>
- BOOL or string truthiness, implicit cast, widening, promotion, accumulation data type, or
  caller-selectable result type<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [widening glossary](../../../../glossary.md#term-widening); [data type glossary](../../../../glossary.md#data-type--datatype))</span>
- value or host-storage inspection, allocation, copy, materialization, cumulative addition,
  exclusive-zero emission, reverse traversal, parallel prefix algorithm, mutation, or execution<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host))</span>
- numerical accumulation order, precision, associativity, reproducibility, overflow, underflow,
  NaN, infinity, signed-zero, empty-axis, or error policy<span data-glossary-navigation="true"> ([precision glossary](../../../../glossary.md#term-precision); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- [layout](../../../../glossary.md#layout) preservation, [view](../../../../glossary.md#term-view-meanings) creation, [strides](../../../../glossary.md#term-stride), offset, storage [aliasing](../../../../glossary.md#term-aliasing), or [materialization](../../../../glossary.md#materialization) policy
- a gradient rule, backward scan, autograd expansion, gradient Tensor, optimizer, or training
  behavior<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction), [`NodeId`](../../../../glossary.md#nodeid), [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), compiled graph records, compiler inference,
  canonicalization, fusion, cost, planning ownership, prepare, runtime, backend, route, kernel,
  ONNX, or execution behavior<span data-glossary-navigation="true"> ([compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [fusion glossary](../../../../glossary.md#term-fusion); [planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [kernel glossary](../../../../glossary.md#kernel); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- changing `CumulativeSumKind`, `CumulativeSumAttrs`, [`Operation`](../../../../glossary.md#operation), `TensorDescriptor`, Shape,
  DataType, TensorFactory, TensorProvenance, existing expression helpers, dependencies, Gradle,
  architecture, or another module<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [DataType glossary](../../../../glossary.md#data-type--datatype); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- implementation or detailed specification of task 0016I or any later task

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
- [Task 0006](0006-operation-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0016G](0016g-cumulative-sum-semantic-kind-and-attributes.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes exactly:

```java
Tensor cumSum(int axis)
Tensor cumSum(int axis, boolean exclusive, boolean reverse)
```

The short form delegates to inclusive [forward mode](../../../../glossary.md#term-forward-mode). The complete form supports all four
inclusion/direction combinations. Legacy evidence confirms positive and negative [axis](../../../../glossary.md#term-axis)
[normalization](../../../../glossary.md#term-normalization), [shape](../../../../glossary.md#shape) and data-type preservation, [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), [BFLOAT16](../../../../glossary.md#term-bfloat16), [INT32](../../../../glossary.md#data-type--datatype), and [INT64](../../../../glossary.md#data-type--datatype)
inputs, [BOOL](../../../../glossary.md#data-type--datatype) rejection, non-[contiguous](../../../../glossary.md#term-contiguous) input, [ONNX](../../../../glossary.md#term-onnx) round trips, [CPU](../../../../glossary.md#term-cpu) execution, and [Metal](../../../../glossary.md#term-metal) [lowering](../../../../glossary.md#lowering).

Legacy implementation details are not copied. In particular, its [operation](../../../../glossary.md#operation) class casing, generic
reduction family label, expression text, [callback](../../../../glossary.md#term-callback) builder, forced [eager](../../../../glossary.md#term-eager-evaluation) no-grad flag, storage
loops, planner traits, lowering, and [kernels](../../../../glossary.md#kernel) do not belong in this [model](../../../../glossary.md#term-model) task. In the new
architecture, `requiresGrad` is [descriptor](../../../../glossary.md#term-descriptor) eligibility [metadata](../../../../glossary.md#term-metadata) rather than a legacy eager
[gradient](../../../../glossary.md#term-gradient) callback. This task preserves that existing metadata for floating inputs without
claiming or implementing a cumulative-sum gradient rule.

For logical input `[1, 2, 3]`, the semantic contracts already selected by task 0016G are:

| Call mode | Semantic result |
|---|---|
| `cumSum(axis)` or `(false, false)` | `[1, 3, 6]` |
| `(true, false)` | `[0, 1, 3]` |
| `(false, true)` | `[6, 5, 3]` |
| `(true, true)` | `[5, 3, 0]` |

The present task constructs metadata for these meanings but calculates none of these values.

## Architecture constraints

- Public [Tensor](../../../../glossary.md#tensor) expression construction and [backend](../../../../glossary.md#backend)-independent [operation](../../../../glossary.md#operation) semantics belong to
  `modules/model`.
- `Tensor` remains public mutable API state and is not an IR node. The returned Tensor carries
  immutable descriptor/provenance metadata plus no initial host storage.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [node glossary](../../../../glossary.md#node); [immutable glossary](../../../../glossary.md#term-immutable); [descriptor glossary](../../../../glossary.md#term-descriptor); [metadata glossary](../../../../glossary.md#term-metadata); [host storage glossary](../../../../glossary.md#host-storage))</span>
- The helper performs local expression validation and composition only. It must not capture a
  compiled graph, traverse provenance, infer graph-wide facts, or select backend support.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [capture glossary](../../../../glossary.md#term-capture-meanings); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [graph glossary](../../../../glossary.md#graph))</span>
- Validation order is deterministic: non-null input at the helper boundary, numeric input
  eligibility, exact Shape access, one axis normalization, attributes construction, then common
  descriptor/operation/provenance/factory construction.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- The exact input Shape reference is retained because cumulative sum preserves logical positions,
  including dynamic and zero extents. No Shape or Dimension copy is needed.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension))</span>
- Result layout is unresolved even when the input descriptor has a resolved layout. A later
  compiler/prepare/backend path owns layout and materialization decisions.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare); [materialization glossary](../../../../glossary.md#materialization))</span>
- Result data type is exactly the input data type. BOOL is rejected; no conversion or promotion is
  inserted.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Result `requiresGrad` equals the input descriptor value. This is eligibility metadata only and
  does not define a gradient rule or executable backward scan.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- `CumulativeSumAttrs` stores the normalized non-negative [axis](../../../../glossary.md#term-axis) and the two exact caller mode flags.
- [Provenance](../../../../glossary.md#provenance) contains exactly `[input]` in order. It stores no graph IDs, output grouping,
  executable closure, or runtime state.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>
- Each valid invocation creates one fresh Tensor identity, including repeated identical requests.
  Model construction performs no common-subexpression elimination or canonicalization.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [Model glossary](../../../../glossary.md#term-model); [common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- Package direction remains `model.tensor -> model.operation.scan` and existing model
  foundations. No dependency leaves <span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>`modules/model`.
- Stop if implementation requires a semantic-contract change, another public method or type,
  value/storage access, gradient behavior, dependency, or architecture change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor methods and the package-private
  expression-construction helper.
- `io.github.pho001.synaptik.model.operation.scan` — supplies the completed cumulative-sum kind
  and attributes.
- existing datatype, shape, descriptor, operation, and provenance packages supply immutable model
  contracts without modification.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance); [immutable glossary](../../../../glossary.md#term-immutable); [model glossary](../../../../glossary.md#term-model))</span>

No package is added or renamed.

Type placement:

- `Tensor.cumSum(...)` — public fluent expression surface on the existing [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api).
- `io.github.pho001.synaptik.model.tensor.TensorCumulativeSumExpressions` — package-private
  stateless boundary that validates and composes cumulative-sum metadata.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- `TensorCumulativeSumExpressionTest` — same-package test of helper and public expression behavior.

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) surface

Add exactly:

```java
public Tensor cumSum(int axis)

public Tensor cumSum(int axis, boolean exclusive, boolean reverse)
```

Both are public, non-static, non-synchronized instance methods returning Tensor. The short overload
performs exactly one delegation equivalent to:

```java
TensorCumulativeSumExpressions.apply(this, axis, false, false)
```

The complete overload performs exactly one delegation with the exact caller booleans. Neither
method performs [validation](../../../../glossary.md#term-validation), reads state, constructs [metadata](../../../../glossary.md#term-metadata), accesses storage, or allocates an
[identity](../../../../glossary.md#term-identity-meanings) itself.

### Construction helper

Add one package-private final `TensorCumulativeSumExpressions` with:

- no field, nested type, interface, instance state, or public/protected member;
- one private zero-[argument](../../../../glossary.md#term-argument) constructor;
- exactly one package-private static
  `apply(Tensor input, int axis, boolean exclusive, boolean reverse)` entry;
- one private static `validateNumericInput(Tensor input)` method; and
- one private static
  `create(Tensor input, Shape shape, CumulativeSumAttrs attrs)` method.

Add no overload, generic [operation](../../../../glossary.md#operation)-family helper, factory, service, [registry](../../../../glossary.md#term-registry), [cache](../../../../glossary.md#term-cache), state, or test
hook.

### [Validation](../../../../glossary.md#term-validation) and construction order

`apply` performs exactly this sequence:

1. `Objects.requireNonNull(input, "input")`;
2. validate that `input.descriptor().dataType()` is floating or integral;
3. read the exact `Shape inputShape = input.descriptor().shape()`;
4. normalize [`axis`](../../../../glossary.md#term-axis) exactly once through `inputShape.normalizeAxis(axis)`;
5. create one `CumulativeSumAttrs(normalizedAxis, exclusive, reverse)`;
6. call `create(input, inputShape, attrs)` exactly once.

[BOOL](../../../../glossary.md#data-type--datatype) fails at step 2 with `IllegalArgumentException` and exact message:

```text
input must have a numeric data type, but was BOOL
```

Invalid axes use the existing [Shape](../../../../glossary.md#shape) failure type and exact message. Type validation precedes axis
validation, including for BOOL [scalar](../../../../glossary.md#term-scalar) input. Failures before factory delegation consume no [Tensor](../../../../glossary.md#tensor)
[identity](../../../../glossary.md#term-identity-meanings).

`create` performs exactly this sequence:

1. construct one `TensorDescriptor` from the exact input [data type](../../../../glossary.md#data-type--datatype), exact supplied Shape,
   `Optional.empty()` [layout](../../../../glossary.md#layout), and unchanged input `requiresGrad`;
2. construct one `Operation(CumulativeSumKind.CUM_SUM, attrs)` retaining the exact attributes;
3. construct one `TensorProvenance(operation, List.of(input))`;
4. call `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` exactly once and
   return its result.

Construction reads no input label, [provenance](../../../../glossary.md#provenance), [host storage](../../../../glossary.md#host-storage), layout geometry, [dimensions](../../../../glossary.md#dimension), [element](../../../../glossary.md#term-element)
count, or values. The result has no label or storage and a present exact one-input provenance.

### [Shape](../../../../glossary.md#shape), type, and mode behavior

- Every numeric input rank greater than zero is structurally eligible, including dynamic and zero
  extents.
- Every [scalar](../../../../glossary.md#term-scalar) [axis](../../../../glossary.md#term-axis) is invalid because [rank](../../../../glossary.md#term-rank) zero has no axis.
- Positive and negative caller axes normalize to the same non-negative stored axis when they
  address the same input position.
- The result descriptor retains the exact input Shape reference, exact data type, and exact
  gradient-eligibility value; it never retains a resolved input layout.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [data type glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient); [layout glossary](../../../../glossary.md#layout))</span>
- All four exclusive/reverse combinations retain their exact primitive flags.
- Reverse changes semantic traversal direction only. Neither Shape nor dimension order is
  reversed.<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension))</span>
- Every valid invocation is fresh and explicit. Repeated equivalent calls must not return the
  input or reuse another result.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorCumulativeSumExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorCumulativeSumExpressionTest.java`

Documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistent: Training [API](../../../../glossary.md#term-application-programming-interface-api), capabilities, cumulative-sum
semantic contracts/tests, [descriptor](../../../../glossary.md#term-descriptor)/factory/[provenance](../../../../glossary.md#provenance)/[Shape](../../../../glossary.md#shape)/[DataType](../../../../glossary.md#data-type--datatype) contracts, other expression
families, focused architecture, ADRs/tests, [conformance](../../../../glossary.md#term-conformance)/integration tests, and [Gradle](../../../../glossary.md#term-gradle).

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, two tests, and six documentation/[planning](../../../../glossary.md#term-planning) files: ten paths total.
Do not modify another existing Java contract or test. Stop beyond this scope or if implementation
requires semantic-contract changes, another public surface, value/storage behavior, [gradient](../../../../glossary.md#term-gradient)
rules, dependencies, or architecture changes. Do not create task 0016I.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document both public Tensor methods and the package-private helper, constructor, entry, and
  private methods.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Explain cumulative sum for a newcomer, including axis, prefix, inclusive, exclusive, forward,
  reverse, output order, and shape preservation.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [shape glossary](../../../../glossary.md#shape))</span>
- Include the concrete `[1, 2, 3]` four-mode example and comment on each boundary/result.
- Document every parameter, exact mode default, accepted numeric types, axis range and negative
  normalization, returned metadata, freshness, unresolved layout, absent label/storage, and
  one-input provenance.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [normalization glossary](../../../../glossary.md#term-normalization); [metadata glossary](../../../../glossary.md#term-metadata); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>
- Document every caller-visible failure type and deterministic ordering, including the exact BOOL
  failure and scalar/out-of-range axis behavior.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Explain that construction does not inspect or accumulate values and does not define numerical,
  gradient, compiler, backend, or execution behavior.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- Review Tensor, descriptor, factory, provenance, Shape, DataType, Operation, and cumulative-sum
  semantic Javadocs. Record why unchanged contracts remain accurate or stop on an out-of-scope
  discrepancy.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [DataType glossary](../../../../glossary.md#data-type--datatype); [Operation glossary](../../../../glossary.md#operation); [scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- [Tensor](../../../../glossary.md#tensor) exposes exactly the two requested `cumSum` overloads with exact signatures and one helper
  delegation each; no other public API changes.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- The short form explicitly supplies `(false, false)` and the complete form retains both caller
  flags exactly.
- The helper has exactly the planned package visibility, finality, constructor, three methods,
  imports, and no state or extra API.
- All five numeric data types succeed; [BOOL](../../../../glossary.md#data-type--datatype) fails with exact type/message before [axis](../../../../glossary.md#term-axis) [validation](../../../../glossary.md#term-validation).
- Positive and negative axes normalize exactly once; invalid/scalar axes preserve the existing
  Shape failure contract.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [Shape glossary](../../../../glossary.md#shape))</span>
- Static, dynamic, and zero-extent Shapes are accepted when the axis exists. The exact Shape
  reference, input data type, and gradient eligibility are retained; layout is unresolved.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [data type glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient); [layout glossary](../../../../glossary.md#layout))</span>
- Exact CUM_SUM/attributes semantics and exact ordered `[input]` provenance are present. Result
  label and host storage are absent.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [host storage glossary](../../../../glossary.md#host-storage))</span>
- Every valid call produces a fresh identity without returning or mutating the input, inspecting
  values/storage, canonicalizing, or executing a scan.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- `TensorTest` changes only for the deliberate two-method public API shape. One focused test owns
  all cumulative-sum behavior.
- Focused and aggregate tests, Javadoc, root tests, reflection/javap/bytecode/import/scope and
  documentation validation pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>
- A separate clean-context documentation-focused agent finalizes permitted Javadocs, Tensor API,
  Compile API, glossary, planning, examples, and no-change conclusions.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning))</span>
- 0016H becomes Complete only after both passes; 0016I remains Draft without a specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorCumulativeSumExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test covers exact public/helper [API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape); short-form delegation; all four mode flags;
all five numeric data types and [BOOL](../../../../glossary.md#data-type--datatype) rejection; positive/negative/[scalar](../../../../glossary.md#term-scalar)/out-of-range [axes](../../../../glossary.md#term-axis);
static, dynamic, zero-[extent](../../../../glossary.md#term-extent), unresolved/resolved-input-[layout](../../../../glossary.md#layout) Shapes; exact [descriptor](../../../../glossary.md#term-descriptor) facts;
same Shape/type/eligibility references or values; exact kind/attributes/[provenance](../../../../glossary.md#provenance) references;
absent label/storage; input non-mutation; freshness; failure order and [identity](../../../../glossary.md#term-identity-meanings) side effects; and
absence of value, storage, [graph](../../../../glossary.md#graph), [compiler](../../../../glossary.md#term-compiler), [runtime](../../../../glossary.md#term-runtime), or [backend](../../../../glossary.md#backend) behavior.

Manually inspect reflection, `javap -p -c -s`, source, imports, and [bytecode](../../../../glossary.md#term-bytecode) for the exact two
public delegations, one axis [normalization](../../../../glossary.md#term-normalization), validation/construction order, exact helper surface,
one descriptor/[operation](../../../../glossary.md#operation)/provenance/factory path, no hidden state, no value/storage access, and no
cross-layer imports. Validate generated [Javadoc](../../../../glossary.md#term-javadoc), [Tensor](../../../../glossary.md#tensor) API/[Compile](../../../../glossary.md#compile) API/glossary, newcomer example,
links/anchors/fences/whitespace, exact ten paths, synchronized statuses, and absence of a
task-0016I specification.

## Dependencies

- 0001 supplies numeric-versus-[BOOL](../../../../glossary.md#data-type--datatype) [DataType](../../../../glossary.md#data-type--datatype) categories and [gradient](../../../../glossary.md#term-gradient) eligibility.
- 0002 supplies exact [immutable](../../../../glossary.md#term-immutable) [Shape](../../../../glossary.md#shape) retention and positive/negative [axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization).
- 0006 supplies immutable generic [Operation](../../../../glossary.md#operation) composition.
- 0007 supplies unresolved TensorDescriptor construction.
- 0011–0013 supply public Tensor metadata, central derived identity allocation, and immutable
  operation/input provenance.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- 0016G supplies exact CUM_SUM [identity](../../../../glossary.md#term-identity-meanings) and normalized-axis/exclusive/reverse attributes.

## Follow-up tasks

- 0016I remains Draft for [softmax](../../../../glossary.md#softmax--log-softmax) and [log-softmax](../../../../glossary.md#softmax--log-softmax) semantic kinds and attributes.
- 0016J remains Draft for their public floating [Tensor](../../../../glossary.md#tensor) expression construction.
- Compiler tasks own capture, inference validation, canonicalization, autograd expansion, and
  graph optimization.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [autograd glossary](../../../../glossary.md#autograd); [graph optimization glossary](../../../../glossary.md#forward-graph-optimization))</span>
- Backend/config/conformance tasks own executable cumulative-sum algorithms, numerical policy,
  storage traversal, kernels, routes, and parity.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [kernels glossary](../../../../glossary.md#kernel))</span>

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. This task composes existing [model](../../../../glossary.md#term-model)-owned public [Tensor](../../../../glossary.md#tensor), [descriptor](../../../../glossary.md#term-descriptor),
[operation](../../../../glossary.md#operation), [provenance](../../../../glossary.md#provenance), [Shape](../../../../glossary.md#shape), and factory contracts without changing [module](../../../../glossary.md#term-module) boundaries or
[dependency direction](../../../../glossary.md#java-platform-and-build-tools). Stop if an architecture change is required.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0006/0007/0011/0012/0013/0016G/0016H, Tensor API,
Compile API, Training API, glossary, current DataType/Shape/TensorDescriptor/Tensor/TensorFactory/
TensorProvenance/Operation/CumulativeSumKind/CumulativeSumAttrs contracts and tests, and Java 26
Gradle configuration.

Implement task 0016H exactly. Modify Tensor.java and add package-private final
TensorCumulativeSumExpressions.java for production. Update TensorTest only for the exact two-method
API surface and add TensorCumulativeSumExpressionTest. Add exactly cumSum(axis) and
cumSum(axis,exclusive,reverse); each delegates once to the shared helper, and the short form uses
false/false.

The helper has exactly apply, validateNumericInput, and create. Null-check input, accept exactly
floating or integral data types, retain the exact input Shape, normalize the axis once, construct
exact CumulativeSumAttrs, preserve exact type and requiresGrad in an unresolved descriptor, create
exact CUM_SUM Operation and one-input provenance, and call createDerived once with no label/storage.
Every call is fresh.

Do not inspect or accumulate values/storage, preserve resolved layout, convert/promote types,
define numerical or gradient rules, capture graphs, add overloads, change existing contracts, or
introduce compiler/runtime/backend behavior. Stop beyond ten paths or on architecture uncertainty.

Run all task validation, then hand the actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/Compile API/glossary/planning, record related-contract/capability/Training API/
architecture no-change conclusions, and rerun validation.

Update task 0016H, model master plan, and roadmap only for planning status/evidence. Do not mark
0016H Complete until both passes succeed. Leave 0016I Draft without a specification. Do not commit
or push.
```

## Local decisions

- Provide exactly the two legacy public signatures and make the short form explicitly inclusive
  forward. A separate default, options object, or mode enum would add surface without a current
  need.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- Accept all five current numeric types and reject BOOL. This aligns with the selected capability
  evidence, including later legacy INT64 execution that supersedes one stale legacy error message
  mentioning only INT32.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [INT32 glossary](../../../../glossary.md#data-type--datatype))</span>
- Preserve exact input data type rather than promote an accumulation type. Numerical accumulation
  width and overflow are executable policy outside the model expression.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [overflow glossary](../../../../glossary.md#term-overflow); [model glossary](../../../../glossary.md#term-model))</span>
- Preserve `requiresGrad` eligibility metadata for floating input, matching current model
  expression composition. This does not copy the legacy eager builder's forced no-grad callback
  behavior and does not claim a gradient implementation.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [eager glossary](../../../../glossary.md#term-eager-evaluation); [callback glossary](../../../../glossary.md#term-callback); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Retain the exact input Shape because cumulative sum is shape-preserving. Leave layout unresolved
  because expression construction does not select output materialization.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization))</span>
- Use a dedicated tensor helper rather than extending aggregate-reduction helpers. Scans preserve
  positions and use a distinct semantic family and parameter set.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor); [parameter glossary](../../../../glossary.md#term-parameter))</span>

## Known limitations

- No values are accumulated and no output storage exists at [model](../../../../glossary.md#term-model) construction time.
- No numerical accuracy, overflow, empty-axis, gradient, compiler, ONNX, backend, or kernel
  behavior is implemented.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [ONNX glossary](../../../../glossary.md#term-onnx); [backend glossary](../../../../glossary.md#backend); [kernel glossary](../../../../glossary.md#kernel))</span>
- Only one-[axis](../../../../glossary.md#term-axis) cumulative sum exists; other scans and multi-axis forms remain unsupported.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) read the [architecture contract](../../../../glossary.md#architecture-contract) and focused architecture explanations; documentation and
planning rules; roadmap; [model](../../../../glossary.md#term-model) capabilities/master plan; tasks 0001, 0002, 0006, 0007, 0011,
0012, 0013, and 0016G; current [Tensor](../../../../glossary.md#tensor), [descriptor](../../../../glossary.md#term-descriptor), factory, [provenance](../../../../glossary.md#provenance), [Shape](../../../../glossary.md#shape), [DataType](../../../../glossary.md#data-type--datatype),
cumulative-sum semantic contracts/tests; Tensor/[Compile](../../../../glossary.md#compile)/Training APIs; glossary; and Java 26 [Gradle](../../../../glossary.md#term-gradle)
configuration.

The legacy branch was read directly. It confirms exactly two public overloads, inclusive-forward
defaults, negative [axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization), all four modes, exact shape/type preservation, all five
numeric types, [BOOL](../../../../glossary.md#data-type--datatype) rejection, non-[contiguous](../../../../glossary.md#term-contiguous) input, [ONNX](../../../../glossary.md#term-onnx), [CPU](../../../../glossary.md#term-cpu), and [Metal](../../../../glossary.md#term-metal) evidence. Legacy
[operation](../../../../glossary.md#operation) traits, stale error text, [eager](../../../../glossary.md#term-eager-evaluation) no-grad [callback](../../../../glossary.md#term-callback) policy, storage, [lowering](../../../../glossary.md#lowering), and [kernels](../../../../glossary.md#kernel)
are excluded or assigned to later [owners](../../../../glossary.md#term-owner).

Planning selected two public methods, one three-method package-private helper, one focused test,
and no new package. Existing contracts suffice; no semantic, [dependency](../../../../glossary.md#term-dependency), build, or architecture
change is required.

Pre-implementation planning validation after synchronizing this task, the model master plan, and
roadmap:

- `git diff --check` passed.
- The targeted trailing-whitespace [scan](../../../../glossary.md#term-cumulative-scan) returned no matches across the three planning files.
- All 172 relative Markdown links across the three planning files resolve locally.
- Markdown fence counts are balanced: twelve in this task, two in the master plan, and zero in the
  roadmap.
- All 20 task-template headings are present, together with the focused Capability origin,
  Required contract, and Javadoc requirements sections.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- At that pre-implementation point, task, model master plan, and roadmap consistently identified
  this task as Ready and 0016I as Draft.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- No detailed task-0016I specification exists.
- Repository scope is exactly this task, the model master plan, and the roadmap; no Java, API,
  architecture, Gradle, or other file changed during planning.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

Implementation and independent documentation validation:

- The implementation pass added the exact two public `Tensor.cumSum` overloads, the dedicated
  package-private `TensorCumulativeSumExpressions` helper, the seven-test focused expression
  suite, and only the two-method public-surface adjustment in `TensorTest`.
- Clean documentation-focused Codex context
  `019f3281-986c-70a1-abf7-3c03944a1355` applied General style, API and Javadoc style, Planning
  style, and Example format. It independently inspected the implementation diff, final source and
  tests, related foundational and cumulative-sum contracts, generated model Javadoc, bytecode,
  imports, public API references, glossary, planning state, architecture documents, and Gradle
  configuration.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- The two public method Javadocs and package-private helper Javadocs now explain the prefix mental
  model, all four <span data-glossary-navigation="true"> ([mental model glossary](../../../../glossary.md#term-mental-model))</span>`[1, 2, 3]` modes position by position, positive/negative axes, accepted numeric
  types, BOOL and axis failures, exact metadata retention, fresh identity, unresolved layout,
  absent label/storage, one-input provenance, and all deferred value/gradient/compiler/backend/
  execution behavior. Related <span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [identity glossary](../../../../glossary.md#term-identity-meanings); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>`DataType`, `Shape`, `TensorDescriptor`, `TensorFactory`,
  `TensorProvenance`, `Operation`, `CumulativeSumKind`, and `CumulativeSumAttrs` Javadocs remain
  accurate unchanged.
- `docs/api/tensor-api.md` now documents the current two-method expression surface, construction
  boundary, four-mode example with line-level numerical commentary, metadata/ownership behavior,
  deterministic failures, and semantic-versus-executable distinction. <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>`docs/api/compile-api.md`
  now includes cumulative sum in the current expression inventory while retaining compiler
  capture, inference, canonicalization, artifacts, and execution as planned. <span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [artifacts glossary](../../../../glossary.md#term-artifact))</span>`docs/glossary.md`
  synchronizes cumulative-sum, normalized-axis, provenance, Tensor, kind, and attribute status.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorCumulativeSumExpressionTest --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; 21 tests across the two
  XML suites, with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 56 XML suites contain 432 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; generated `Tensor.html` contains both
  overloads, their parameters, return facts, failures, four-mode explanation, and boundary text.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter))</span>
- `./gradlew test` — `BUILD SUCCESSFUL`; the repository lifecycle completed 36 actionable tasks,
  with one executed and 35 up-to-date and no failing task.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `javap -p -c -s` confirms each public overload delegates exactly once, the short overload passes
  `false, false`, the complete overload forwards both flags, and the helper has exactly one private
  constructor plus `apply`, `validateNumericInput`, and `create`. Bytecode confirms one axis
  normalization and one descriptor/operation/provenance/factory construction path in the required
  order, with no fields or hidden state.
- Import and source inspection found only model-owned datatype, Shape, operation, scan, tensor,
  and JDK collection/null-check dependencies. No graph, compiler, planning, prepare, runtime,
  backend, engine, trace, training, storage-access, or value-execution dependency was introduced.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [graph glossary](../../../../glossary.md#graph); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace))</span>
- Local Markdown target/anchor validation passed for all 262 relative links in the six changed
  documentation/planning files. Markdown fences are balanced, targeted trailing-whitespace and
  stale-status scans passed, the four numerical examples were recalculated, generated Javadoc was
  reviewed, and <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>`git diff --check` passed.
- Final repository scope is exactly the authorized ten paths: two production files, two tests,
  Tensor API, Compile API, glossary, this task, model master plan, and roadmap. Task 0016H is
  synchronized as Complete; task 0016I remains Draft, and no `0016i-*` specification exists.
- Training API and capabilities remain accurate unchanged: the former contains only planned
  training concepts, and the latter already inventories cumulative-sum modes while distinguishing
  model expression representation from compiler/backend execution. Foundational, cumulative-sum
  semantic, and other expression contracts remain accurate because their signatures and ownership
  did not change.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance tests, integration tests, Gradle configuration, and other modules remain
  accurate unchanged because this task changes no dependency direction, lifecycle, backend or
  end-to-end behavior, build configuration, or code outside <span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>`modules/model`.
- No sandbox-only Gradle lock failure occurred during this documentation pass. No validation was
  skipped, and no blocker remains.

## Implementation notes

- Added exactly two public delegating methods and one stateless three-method construction helper.
- The helper accepts all five numeric types, rejects BOOL before axis validation, normalizes one
  axis, preserves exact Shape/type/eligibility metadata with unresolved layout, records exact
  one-input provenance, and delegates once to central derived identity allocation.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [axis glossary](../../../../glossary.md#term-axis); [validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Added the focused seven-test contract suite and adjusted only Tensor's exact public API count and
  method set in the existing Tensor test.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- The documentation-focused pass finalized Javadocs, Tensor API, Compile API, glossary, task
  evidence, model master plan, and roadmap without adding executable behavior.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>

## Completion summary

- Completed changes: Implemented and documented public cumulative-sum expression construction for
  inclusive/exclusive and forward/reverse modes.
- Files changed or created: Exactly two production Java files, two tests, Tensor API, Compile API,
  glossary, this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused 21/21, model 432/432 across 56 suites, generated model Javadoc,
  root tests, bytecode/reflection/import/manual checks, Markdown links/anchors/fences/whitespace,
  exact scope/status checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation-agent review: Clean Codex context
  `019f3281-986c-70a1-abf7-3c03944a1355` completed the mandatory independent pass.
- Documentation impact: Tensor and Compile API references, glossary, task, master plan, and roadmap
  now describe current cumulative-sum expression construction and its non-executable boundary.
- Javadoc review: Public and package-private cumulative-sum Javadocs are complete; related
  foundational and semantic contracts remain accurate unchanged.
- Glossary impact: Current cumulative-sum construction, normalized-axis ownership, provenance, and
  Tensor/kind/attribute status are synchronized.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0016H. Task 0016I remains Draft without a detailed
  specification.

Status: Complete
