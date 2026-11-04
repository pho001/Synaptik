# Task 0016J: [Softmax](../../../../glossary.md#softmax--log-softmax) [Tensor](../../../../glossary.md#tensor) Expressions

## Status

Complete

## Goal

Add the public [model](../../../../glossary.md#term-model)-level expression boundary for one-[axis](../../../../glossary.md#term-axis) [softmax](../../../../glossary.md#softmax--log-softmax) and [log-softmax](../../../../glossary.md#softmax--log-softmax). Each method
must validate floating input [metadata](../../../../glossary.md#term-metadata), normalize the selected axis, retain the exact logical [Shape](../../../../glossary.md#shape)
and [data type](../../../../glossary.md#data-type--datatype), record exact one-input [provenance](../../../../glossary.md#provenance), and return a fresh storage-free [Tensor](../../../../glossary.md#tensor) without
evaluating exponentials, logarithms, reductions, or probabilities.

This task completes public expression construction for the semantic contracts introduced by task
0016I. It does not select a [finite](../../../../glossary.md#term-finite)-[precision](../../../../glossary.md#term-precision) algorithm, define a [gradient](../../../../glossary.md#term-gradient) rule, [capture](../../../../glossary.md#term-capture-meanings) a [compiled
graph](../../../../glossary.md#compiled-graph--compiledgraphmodel), decompose the [operation](../../../../glossary.md#operation), or report [backend](../../../../glossary.md#backend) support.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly `Tensor.softmax(int axis)` and `Tensor.logSoftmax(int axis)`.
- Make each public method delegate exactly once with the corresponding `SoftmaxKind`.
- Add one package-private final `TensorSoftmaxExpressions` construction boundary in the existing
  tensor package.
- Accept exactly [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), and [BFLOAT16](../../../../glossary.md#term-bfloat16) input.
- Reject [INT32](../../../../glossary.md#data-type--datatype), [INT64](../../../../glossary.md#data-type--datatype), and [BOOL](../../../../glossary.md#data-type--datatype) without conversion, promotion, or truthiness semantics.
- Normalize one positive or negative [axis](../../../../glossary.md#term-axis) exactly once through the exact input [Shape](../../../../glossary.md#shape).
- Retain the exact input Shape and [data type](../../../../glossary.md#data-type--datatype) in a new unresolved-[layout](../../../../glossary.md#layout) [descriptor](../../../../glossary.md#term-descriptor).
- Preserve the input descriptor's `requiresGrad` eligibility metadata without defining a gradient
  formula or backward operation.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [operation glossary](../../../../glossary.md#operation))</span>
- Construct exact `SoftmaxKind` and `SoftmaxAttrs` [metadata](../../../../glossary.md#term-metadata).
- Record exact ordered one-input [provenance](../../../../glossary.md#provenance) `[input]` and create one fresh derived Tensor with no
  label or host storage.<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage))</span>
- Update only the exact [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) surface assertion in `TensorTest` and add one focused expression
  test.
- Finalize Javadocs, Tensor API, Compile API current-expression inventory, glossary, task evidence,
  master plan, and roadmap through the required independent documentation pass during
  implementation.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- another public Tensor method, overload, static factory, builder, alias, or detailed task-0017
  specification<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- temperature, scale, mask, bias, causal mask, epsilon, stabilization toggle, output type,
  precision option, or approximation mode<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [bias glossary](../../../../glossary.md#term-bias); [causal mask glossary](../../../../glossary.md#causal-mask); [epsilon glossary](../../../../glossary.md#epsilon); [precision glossary](../../../../glossary.md#term-precision))</span>
- sparse, sampled, hierarchical, adaptive, masked, fused-attention, cross-entropy, NLL, loss, or
  optimizer API<span data-glossary-navigation="true"> ([fused glossary](../../../../glossary.md#specialization-and-fusion); [loss glossary](../../../../glossary.md#loss); [optimizer glossary](../../../../glossary.md#term-optimizer); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- integral or BOOL input, implicit cast, promotion, accumulation data type, or caller-selected
  output type<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [data type glossary](../../../../glossary.md#data-type--datatype))</span>
- value or host-storage inspection, allocation, copy, materialization, maximum, subtraction,
  exponential, logarithm, sum, division, probability calculation, mutation, or execution<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization))</span>
- numerical stability algorithm, subtract-maximum implementation, reduction order, precision,
  reproducibility, overflow, underflow, NaN, infinity, signed zero, empty-axis, or error policy<span data-glossary-navigation="true"> ([numerical stability glossary](../../../../glossary.md#term-numerical-stability); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed zero glossary](../../../../glossary.md#term-signed-zero); [axis glossary](../../../../glossary.md#term-axis))</span>
- preserving resolved layout, view construction, strides, offset, storage aliasing, or
  materialization policy<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [strides glossary](../../../../glossary.md#term-stride); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- gradient rule, Jacobian-vector product, backward operation, saved forward value, autograd
  expansion, gradient Tensor, optimizer, or training behavior<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [Jacobian-vector product glossary](../../../../glossary.md#term-jacobian-vector-product-jvp); [autograd glossary](../../../../glossary.md#autograd))</span>
- graph capture, NodeId, ValueId, compiled graph records, compiler inference, canonicalization,
  decomposition, fusion, cost, planning ownership, prepare, runtime, backend, route, kernel, ONNX,
  or execution behavior<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [NodeId glossary](../../../../glossary.md#nodeid); [ValueId glossary](../../../../glossary.md#logical-valueid-versus-physical-memory-slot); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion); [planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [kernel glossary](../../../../glossary.md#kernel); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- changing `SoftmaxKind`, `SoftmaxAttrs`, [`Operation`](../../../../glossary.md#operation), `TensorDescriptor`, Shape, DataType,
  TensorFactory, TensorProvenance, existing expression helpers, dependencies, Gradle, architecture,
  or another module<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [DataType glossary](../../../../glossary.md#data-type--datatype); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- implementation or detailed specification of task 0017 or any later task

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
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
- [Task 0016I](0016i-softmax-semantic-kinds-and-attributes.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes exactly:

```java
Tensor softmax(int axis)
Tensor logSoftmax(int axis)
```

Both require floating input, accept positive or negative [axes](../../../../glossary.md#term-axis), preserve logical [shape](../../../../glossary.md#shape) and [data
type](../../../../glossary.md#data-type--datatype), and participate in legacy [gradient](../../../../glossary.md#term-gradient) construction. Legacy evidence covers [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32),
[BFLOAT16](../../../../glossary.md#term-bfloat16), non-[contiguous](../../../../glossary.md#term-contiguous) input, attention/[loss](../../../../glossary.md#loss) composition, [ONNX](../../../../glossary.md#term-onnx), [CPU](../../../../glossary.md#term-cpu), [Metal](../../../../glossary.md#term-metal), and [CUDA](../../../../glossary.md#term-cuda) paths.

Legacy public builders decompose the meanings into maximum, subtraction, exponential, sum,
division, and logarithm expressions. The new [model](../../../../glossary.md#term-model) instead records the first-class [SOFTMAX](../../../../glossary.md#softmax--log-softmax) or
LOG_SOFTMAX semantic [identity](../../../../glossary.md#term-identity-meanings) introduced by task 0016I. A later [compiler](../../../../glossary.md#term-compiler) owns whether to preserve,
canonicalize, or decompose that meaning, while [concrete backend](../../../../glossary.md#concrete-backend) [prepare](../../../../glossary.md#prepare) owns executable routes.

For one [normalization](../../../../glossary.md#term-normalization) [slice](../../../../glossary.md#slice) `[1, 2, 3]`, the requested ideal values are approximately:

| Public method | Semantic result |
|---|---|
| `softmax(axis)` | `[0.09003057, 0.24472847, 0.66524096]` |
| `logSoftmax(axis)` | `[-2.40760596, -1.40760596, -0.40760596]` |

The first row sums to approximately one, and exponentiating the second row reconstructs the
first. This task constructs [metadata](../../../../glossary.md#term-metadata) for those meanings but computes none of the values.

## Architecture constraints

- Public [Tensor](../../../../glossary.md#tensor) expression construction and [backend](../../../../glossary.md#backend)-independent [operation](../../../../glossary.md#operation) semantics belong to
  `modules/model`.
- `Tensor` remains public mutable API state and is not an IR node. Each result carries immutable
  descriptor/provenance metadata and no initial host storage.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [node glossary](../../../../glossary.md#node); [immutable glossary](../../../../glossary.md#term-immutable); [descriptor glossary](../../../../glossary.md#term-descriptor); [metadata glossary](../../../../glossary.md#term-metadata); [host storage glossary](../../../../glossary.md#host-storage))</span>
- The helper performs local expression validation and composition only. It must not capture a
  graph, traverse provenance, infer graph-wide facts, decompose the kind, or select backend
  support.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph))</span>
- Validation order is deterministic: non-null input, non-null kind, floating input eligibility,
  exact Shape access, one axis normalization, attributes construction, then common
  descriptor/operation/provenance/factory construction.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [axis glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Both public kinds are valid at the helper boundary. No registry, compatibility map, or string
  dispatch is needed.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- The exact input Shape reference is retained because both meanings preserve every logical
  position, including dynamic and zero extents.
- Result layout is unresolved even when the input descriptor has a resolved layout. Compiler,
  prepare, and backend paths own layout and materialization decisions.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [Compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare); [materialization glossary](../../../../glossary.md#materialization))</span>
- Result [data type](../../../../glossary.md#data-type--datatype) and `requiresGrad` eligibility equal the input descriptor values. This task
  defines no gradient rule or executable backward computation.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
- `SoftmaxAttrs` stores the normalized non-negative axis. The operation stores the exact requested
  SOFTMAX or LOG_SOFTMAX kind.<span data-glossary-navigation="true"> ([SOFTMAX glossary](../../../../glossary.md#softmax--log-softmax))</span>
- [Provenance](../../../../glossary.md#provenance) contains exactly `[input]`. It stores no graph IDs, executable closure, or runtime
  state.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>
- Every valid invocation creates one fresh Tensor identity, including repeated identical requests.
  Model construction performs no common-subexpression elimination or canonicalization.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [Model glossary](../../../../glossary.md#term-model); [common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- Package direction remains `model.tensor -> model.operation.normalization` plus existing model
  foundations. No dependency leaves <span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>`modules/model`.
- Stop if implementation requires a semantic-contract change, another public method or type,
  value/storage access, numerical or gradient policy, dependency, or architecture change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor methods and the package-private
  expression-construction helper.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- `io.github.pho001.synaptik.model.operation.normalization` — supplies completed softmax kinds and
  axis attributes.<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax); [axis glossary](../../../../glossary.md#term-axis))</span>
- existing datatype, shape, descriptor, operation, and provenance packages supply immutable model
  contracts without modification.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance); [immutable glossary](../../../../glossary.md#term-immutable); [model glossary](../../../../glossary.md#term-model))</span>

No package is added or renamed.

Type placement:

- `Tensor.softmax(int)` and `Tensor.logSoftmax(int)` — public fluent expression surface on the
  existing Tensor API.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `io.github.pho001.synaptik.model.tensor.TensorSoftmaxExpressions` — package-private stateless
  boundary that validates and composes both normalization expressions.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>
- `TensorSoftmaxExpressionTest` — same-package test of helper and public expression behavior.

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) surface

Add exactly:

```java
public Tensor softmax(int axis)

public Tensor logSoftmax(int axis)
```

Both are public, non-static, non-synchronized instance methods returning Tensor. Each performs
exactly one delegation:

```java
TensorSoftmaxExpressions.apply(this, SoftmaxKind.SOFTMAX, axis)
TensorSoftmaxExpressions.apply(this, SoftmaxKind.LOG_SOFTMAX, axis)
```

Neither method validates, reads state, constructs [metadata](../../../../glossary.md#term-metadata), accesses storage, decomposes the
[operation](../../../../glossary.md#operation), or allocates an [identity](../../../../glossary.md#term-identity-meanings) itself.

### Construction helper

Add one package-private final `TensorSoftmaxExpressions` with:

- no field, nested type, interface, instance state, or public/protected member;
- one private zero-[argument](../../../../glossary.md#term-argument) constructor;
- exactly one package-private static
  `apply(Tensor input, SoftmaxKind kind, int axis)` entry;
- one private static `validateFloatingInput(Tensor input)` method; and
- one private static
  `create(Tensor input, SoftmaxKind kind, Shape shape, SoftmaxAttrs attrs)` method.

Add no overload, generic [normalization](../../../../glossary.md#term-normalization) [registry](../../../../glossary.md#term-registry), [operation](../../../../glossary.md#operation)-family helper, factory, service, [cache](../../../../glossary.md#term-cache),
state, or test hook.

### [Validation](../../../../glossary.md#term-validation) and construction order

`apply` performs exactly this sequence:

1. `Objects.requireNonNull(input, "input")`;
2. `Objects.requireNonNull(kind, "kind")`;
3. validate that `input.descriptor().dataType().isFloating()` is true;
4. read exact `Shape inputShape = input.descriptor().shape()`;
5. normalize [`axis`](../../../../glossary.md#term-axis) exactly once through `inputShape.normalizeAxis(axis)`;
6. create one `SoftmaxAttrs(normalizedAxis)`;
7. call `create(input, kind, inputShape, attrs)` exactly once.

Ineligible input fails at step 3 with `IllegalArgumentException` and exact message:

```text
input must have a floating data type, but was <dataType>
```

Invalid axes use the existing [Shape](../../../../glossary.md#shape) failure type and exact message. Type validation precedes axis
validation, including for integral, [BOOL](../../../../glossary.md#data-type--datatype), and [scalar](../../../../glossary.md#term-scalar) inputs. Failures before factory delegation
consume no [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings).

`create` performs exactly this sequence:

1. construct one `TensorDescriptor` from the exact input [data type](../../../../glossary.md#data-type--datatype), exact supplied Shape,
   `Optional.empty()` [layout](../../../../glossary.md#layout), and unchanged input `requiresGrad`;
2. construct one `Operation(kind, attrs)` retaining both exact references;
3. construct one `TensorProvenance(operation, List.of(input))`;
4. call `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` exactly once and
   return its result.

Construction reads no input label, existing [provenance](../../../../glossary.md#provenance), [host storage](../../../../glossary.md#host-storage), layout geometry,
[dimensions](../../../../glossary.md#dimension), [element](../../../../glossary.md#term-element) count, or values. The result has no label or storage and has exact one-input
provenance.

### [Shape](../../../../glossary.md#shape), type, and semantic behavior

- FLOAT64, FLOAT32, and BFLOAT16 inputs of rank greater than zero are structurally eligible,
  including dynamic and zero extents.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- [INT32](../../../../glossary.md#data-type--datatype), [INT64](../../../../glossary.md#data-type--datatype), and [BOOL](../../../../glossary.md#data-type--datatype) are rejected before [axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization).
- Every [scalar](../../../../glossary.md#term-scalar) axis is invalid because [rank](../../../../glossary.md#term-rank) zero has no axis.
- Positive and negative caller axes normalize to the same stored non-negative axis when they
  address the same input position.
- The result descriptor retains the exact input Shape reference, exact floating data type, and
  exact gradient-eligibility value; it never retains a resolved input layout.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [data type glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient); [layout glossary](../../../../glossary.md#layout))</span>
- SOFTMAX and LOG_SOFTMAX remain distinct operation identities with the same local descriptor and
  provenance rules.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([SOFTMAX glossary](../../../../glossary.md#softmax--log-softmax); [operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance))</span>
- Every valid invocation is fresh and explicit. Repeated calls must not return the input, reuse a
  result, or silently rewrite LOG_SOFTMAX into other model operations.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorSoftmaxExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSoftmaxExpressionTest.java`

Documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistent: Training [API](../../../../glossary.md#term-application-programming-interface-api), capabilities, [Softmax](../../../../glossary.md#softmax--log-softmax) semantic
contracts/tests, [descriptor](../../../../glossary.md#term-descriptor)/factory/[provenance](../../../../glossary.md#provenance)/[Shape](../../../../glossary.md#shape)/[DataType](../../../../glossary.md#data-type--datatype) contracts, other expression families,
focused architecture, ADRs/tests, [conformance](../../../../glossary.md#term-conformance)/integration tests, and [Gradle](../../../../glossary.md#term-gradle).

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, two tests, and six documentation/[planning](../../../../glossary.md#term-planning) files: ten paths total.
Do not modify another existing Java contract or test. Stop beyond this scope or if implementation
requires semantic-contract changes, another public surface, value/storage behavior, numerical or
[gradient](../../../../glossary.md#term-gradient) policy, dependencies, or architecture changes. Do not create task 0017.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document both public Tensor methods and the package-private helper, constructor, entry, and
  private methods.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Explain normalization axis and slice, probability, log-probability, shape preservation, and the
  relationship between SOFTMAX and LOG_SOFTMAX for a newcomer.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [axis glossary](../../../../glossary.md#term-axis); [slice glossary](../../../../glossary.md#slice); [log glossary](../../../../glossary.md#term-log); [shape glossary](../../../../glossary.md#shape); [SOFTMAX glossary](../../../../glossary.md#softmax--log-softmax))</span>
- Include the concrete `[1, 2, 3]` example, approximate values, sum-to-one interpretation, and
  exponentiation relationship.
- Document every parameter, accepted floating types, axis range and negative normalization,
  returned metadata, freshness, unresolved layout, absent label/storage, and one-input provenance.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [metadata glossary](../../../../glossary.md#term-metadata); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>
- Document every caller-visible failure type and deterministic ordering, including exact
  non-floating failures and scalar/out-of-range axes.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Explain that construction does not inspect values, calculate probabilities, select a stable
  algorithm, decompose operations, or define numerical, gradient, compiler, backend, or execution
  behavior.<span data-glossary-navigation="true"> ([stable algorithm glossary](../../../../glossary.md#conditioning-and-numerical-stability); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- Review Tensor, descriptor, factory, provenance, Shape, DataType, Operation, and softmax semantic
  Javadocs. Record why unchanged contracts remain accurate or stop on an out-of-scope discrepancy.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [DataType glossary](../../../../glossary.md#data-type--datatype); [Operation glossary](../../../../glossary.md#operation); [scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- Tensor exposes exactly the two requested methods with exact signatures and one helper delegation
  each; no other public API changes.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [signatures glossary](../../../../glossary.md#term-signature); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- The helper has exactly the planned package visibility, finality, constructor, three methods,
  imports, and no state or extra API.
- All three floating types succeed; INT32, INT64, and BOOL fail with exact type/messages before
  axis validation.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [axis glossary](../../../../glossary.md#term-axis); [validation glossary](../../../../glossary.md#term-validation))</span>
- Positive and negative axes normalize exactly once; invalid/scalar axes preserve the existing
  Shape failure contract.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [Shape glossary](../../../../glossary.md#shape))</span>
- Static, dynamic, and zero-extent Shapes are accepted when the axis exists. Exact Shape reference,
  input type, and gradient eligibility are retained; layout is unresolved.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [gradient glossary](../../../../glossary.md#term-gradient); [layout glossary](../../../../glossary.md#layout))</span>
- Exact [SOFTMAX](../../../../glossary.md#softmax--log-softmax) or LOG_SOFTMAX [operation](../../../../glossary.md#operation)/attributes [metadata](../../../../glossary.md#term-metadata) and ordered `[input]` provenance are
  present. Result label and host storage are absent.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [host storage glossary](../../../../glossary.md#host-storage))</span>
- Every valid call produces a fresh identity without returning or mutating input, inspecting
  values/storage, decomposing, canonicalizing, or executing normalization.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- `TensorTest` changes only for the deliberate two-method public API shape. One focused test owns
  all softmax expression behavior.
- Focused and aggregate tests, Javadoc, root tests, reflection/javap/bytecode/import/scope and
  documentation validation pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>
- A separate clean-context documentation-focused agent finalizes permitted Javadocs, Tensor API,
  Compile API, glossary, planning, examples, and no-change conclusions.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning))</span>
- 0016J becomes Complete only after both passes; task 0017 remains Draft without a specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorSoftmaxExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test covers exact public/helper [API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape); exact kind delegation; all three floating
types and three rejected types; positive/negative/[scalar](../../../../glossary.md#term-scalar)/out-of-range [axes](../../../../glossary.md#term-axis); static, dynamic,
zero-[extent](../../../../glossary.md#term-extent), unresolved/resolved-input-[layout](../../../../glossary.md#layout) Shapes; exact [descriptor](../../../../glossary.md#term-descriptor) facts; same Shape/type/
eligibility references or values; exact kind/attributes/[provenance](../../../../glossary.md#provenance) references; absent label and
storage; input non-mutation; freshness; failure order and [identity](../../../../glossary.md#term-identity-meanings) side effects; and absence of
value, storage, [graph](../../../../glossary.md#graph), [compiler](../../../../glossary.md#term-compiler), [runtime](../../../../glossary.md#term-runtime), or [backend](../../../../glossary.md#backend) behavior.

Manually inspect reflection, `javap -p -c -s`, source, imports, and [bytecode](../../../../glossary.md#term-bytecode) for exact public
delegations, one axis [normalization](../../../../glossary.md#term-normalization), validation/construction order, exact helper surface, one
descriptor/[operation](../../../../glossary.md#operation)/provenance/factory path, no hidden state, no value/storage access, and no
cross-layer imports. Validate generated [Javadoc](../../../../glossary.md#term-javadoc), [Tensor](../../../../glossary.md#tensor) API/[Compile](../../../../glossary.md#compile) API/glossary, newcomer example,
links/anchors/fences/whitespace, exact ten paths, synchronized statuses, and absence of a task-0017
specification.

## Dependencies

- 0001 supplies floating [DataType](../../../../glossary.md#data-type--datatype) categories and [gradient](../../../../glossary.md#term-gradient) eligibility.
- 0002 supplies exact [immutable](../../../../glossary.md#term-immutable) [Shape](../../../../glossary.md#shape) retention and positive/negative [axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization).
- 0006 supplies immutable generic [Operation](../../../../glossary.md#operation) composition.
- 0007 supplies unresolved TensorDescriptor construction.
- 0011–0013 supply public Tensor metadata, central derived identity allocation, and immutable
  operation/input provenance.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [metadata glossary](../../../../glossary.md#term-metadata); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
- 0016I supplies exact [SOFTMAX](../../../../glossary.md#softmax--log-softmax)/LOG_SOFTMAX identities and normalized-axis attributes.

## Follow-up tasks

- Task 0017 remains Draft for layout and view operations. Its detailed decomposition must be
  planned only after 0016J is complete.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- Compiler tasks own capture, inference validation, canonicalization, optional decomposition,
  autograd expansion, and graph optimization.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [autograd glossary](../../../../glossary.md#autograd); [graph optimization glossary](../../../../glossary.md#forward-graph-optimization))</span>
- Backend/config/conformance tasks own stable finite-precision algorithms, storage traversal,
  fused/decomposed routes, kernels, and cross-backend parity.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [finite glossary](../../../../glossary.md#term-finite); [precision glossary](../../../../glossary.md#term-precision); [fused glossary](../../../../glossary.md#specialization-and-fusion); [kernels glossary](../../../../glossary.md#kernel))</span>
- Loss and attention tasks may compose softmax semantics later without adding hidden behavior to
  this expression helper.<span data-glossary-navigation="true"> ([Loss glossary](../../../../glossary.md#loss); [softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. This task composes existing [model](../../../../glossary.md#term-model)-owned public [Tensor](../../../../glossary.md#tensor), [descriptor](../../../../glossary.md#term-descriptor),
[operation](../../../../glossary.md#operation), [provenance](../../../../glossary.md#provenance), [Shape](../../../../glossary.md#shape), and factory contracts without changing [module](../../../../glossary.md#term-module) boundaries or
[dependency direction](../../../../glossary.md#java-platform-and-build-tools). Stop if an architecture change is required.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0006/0007/0011/0012/0013/0016I/0016J, Tensor API,
Compile API, Training API, glossary, current DataType/Shape/TensorDescriptor/Tensor/TensorFactory/
TensorProvenance/Operation/SoftmaxKind/SoftmaxAttrs contracts and tests, and Java 26 Gradle
configuration.

Implement task 0016J exactly. Modify Tensor.java and add package-private final
TensorSoftmaxExpressions.java for production. Update TensorTest only for the exact two-method API
surface and add TensorSoftmaxExpressionTest. Add exactly softmax(axis) and logSoftmax(axis), each
delegating once to the shared helper and exact kind.

The helper has exactly apply, validateFloatingInput, and create. Null-check input then kind, accept
only floating data types, retain the exact input Shape, normalize the axis once, construct exact
SoftmaxAttrs, preserve exact type and requiresGrad in an unresolved descriptor, create the exact
kind Operation and one-input provenance, and call createDerived once with no label/storage. Every
call is fresh.

Do not inspect or normalize values/storage, preserve resolved layout, convert/promote types,
select a numerical algorithm, decompose operations, define gradient rules, capture graphs, add
overloads, change existing contracts, or introduce compiler/runtime/backend behavior. Stop beyond
ten paths or on architecture uncertainty.

Run all task validation, then hand the actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/Compile API/glossary/planning, record related-contract/capability/Training API/
architecture no-change conclusions, and rerun validation.

Update task 0016J, model master plan, and roadmap only for planning status/evidence. Do not mark
0016J Complete until both passes succeed. Leave task 0017 Draft without a specification. Do not
commit or push.
```

## Local decisions

- Provide exactly the two legacy public signatures and use one shared helper because both meanings
  have identical local validation, descriptor, and provenance rules.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- Accept only the three current floating types. Integral and BOOL normalization would introduce
  conversion semantics absent from the selected capability.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Preserve exact input [data type](../../../../glossary.md#data-type--datatype) and `requiresGrad` eligibility. This is model metadata and does
  not copy legacy mutable gradient callbacks or claim a backward implementation.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [metadata glossary](../../../../glossary.md#term-metadata); [mutable glossary](../../../../glossary.md#term-mutable); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Retain exact input Shape and leave layout unresolved because normalization is shape-preserving
  while materialization belongs to later lifecycle stages.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Record first-class SOFTMAX or LOG_SOFTMAX operations rather than reproducing legacy public
  decomposition. Compiler and backend owners later decide fused versus primitive representation.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([SOFTMAX glossary](../../../../glossary.md#softmax--log-softmax); [decomposition glossary](../../../../glossary.md#term-decomposition); [Compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [fused glossary](../../../../glossary.md#specialization-and-fusion))</span>
- Use a dedicated tensor helper rather than extending aggregate or scan helpers. Normalization
  preserves all positions but depends on a complete axis slice.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor); [scan glossary](../../../../glossary.md#term-cumulative-scan); [axis glossary](../../../../glossary.md#term-axis); [slice glossary](../../../../glossary.md#slice))</span>

## Known limitations

- No values or probabilities are calculated and no result storage exists at [model](../../../../glossary.md#term-model) construction.
- No finite-precision, numerical-edge, gradient, compiler, ONNX, backend, or kernel policy is
  implemented.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [precision glossary](../../../../glossary.md#term-precision); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [ONNX glossary](../../../../glossary.md#term-onnx); [backend glossary](../../../../glossary.md#backend); [kernel glossary](../../../../glossary.md#kernel))</span>
- Temperature, masking, scaling, [fused](../../../../glossary.md#specialization-and-fusion) attention, and [loss](../../../../glossary.md#loss) composition remain outside this [API](../../../../glossary.md#term-application-programming-interface-api).

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) read the [architecture contract](../../../../glossary.md#architecture-contract) and focused architecture explanations; documentation and
planning rules; roadmap; [model](../../../../glossary.md#term-model) capabilities/master plan; tasks 0001, 0002, 0006, 0007, 0011,
0012, 0013, and 0016I; current [Tensor](../../../../glossary.md#tensor), [descriptor](../../../../glossary.md#term-descriptor), factory, [provenance](../../../../glossary.md#provenance), [Shape](../../../../glossary.md#shape), [DataType](../../../../glossary.md#data-type--datatype), [softmax](../../../../glossary.md#softmax--log-softmax)
semantic contracts/tests; Tensor/[Compile](../../../../glossary.md#compile)/Training APIs; glossary; and Java 26 [Gradle](../../../../glossary.md#term-gradle) configuration.

The legacy branch was read directly. It confirms exactly one public method per kind, floating
eligibility, negative-[axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization), shape/type preservation, [mutable](../../../../glossary.md#term-mutable) [gradient](../../../../glossary.md#term-gradient) callbacks,
stable mathematical intent, [FLOAT64](../../../../glossary.md#term-float64)/[FLOAT32](../../../../glossary.md#term-float32)/[BFLOAT16](../../../../glossary.md#term-bfloat16), non-[contiguous](../../../../glossary.md#term-contiguous) input, attention/[loss](../../../../glossary.md#loss)
composition, [ONNX](../../../../glossary.md#term-onnx), [CPU](../../../../glossary.md#term-cpu), [Metal](../../../../glossary.md#term-metal), and [CUDA](../../../../glossary.md#term-cuda) evidence. Legacy [decomposition](../../../../glossary.md#term-decomposition), callbacks, [operation](../../../../glossary.md#operation)
traits, storage, [lowering](../../../../glossary.md#lowering), [fusion](../../../../glossary.md#term-fusion), and [kernels](../../../../glossary.md#kernel) are excluded or assigned to later [owners](../../../../glossary.md#term-owner).

Planning selected two public methods, one three-method package-private helper, one focused test,
and no new package. Existing contracts suffice; no semantic, [dependency](../../../../glossary.md#term-dependency), build, or architecture
change is required.

Pre-implementation planning validation after synchronizing this task, the model master plan, and
roadmap:

- `git diff --check` passed.
- The targeted trailing-whitespace [scan](../../../../glossary.md#term-cumulative-scan) returned no matches across the three planning files.
- All 179 relative Markdown links across the three planning files resolve locally.
- Markdown fence counts are balanced: twelve in this task, two in the master plan, and zero in the
  roadmap.
- All 20 task-template headings are present, together with the focused Capability origin,
  Required contract, and Javadoc requirements sections.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- The `[1, 2, 3]` SOFTMAX and LOG_SOFTMAX examples were independently recalculated; displayed
  rounding matches, the unrounded SOFTMAX values sum to one, and exponentiated LOG_SOFTMAX values
  match them.<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding))</span>
- Task, model master plan, and roadmap consistently identify 0016J as [Ready](../../../../glossary.md#term-ready) and task 0017 as Draft.
- No detailed task-0017 specification exists.
- Repository scope is exactly this task, the model master plan, and the roadmap; no Java, API,
  architecture, Gradle, or other file changed during planning.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

Implementation and independent documentation validation:

- Implementation context `/root/implement_model_0016j` added the two exact Tensor methods, the
  dedicated package-private helper, and the focused expression suite while changing `TensorTest`
  only for the deliberate public API expansion. Clean documentation context
  `/root/implement_model_0016j/review_model_0016j_docs` then independently read the required
  architecture, documentation, planning, historical task, API, glossary, Gradle, final source,
  final tests, generated Javadoc, and complete diff. It applied General, API/Javadoc, Planning,
  and Example profiles.
- Independent source and test inspection confirmed exactly one public delegation per method and
  exact SOFTMAX/LOG_SOFTMAX kind selection. The helper null-checks input then kind, validates
  floating type before axis, retains the exact Shape, normalizes the axis once, constructs one
  `SoftmaxAttrs`, then constructs exactly one unresolved descriptor, one Operation, one
  `[input]` provenance value, and one derived Tensor. Results retain exact type and
  `requiresGrad`, have no label or storage, remain fresh, and do not inspect values or preserve
  input layout.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- The submitted `Tensor.softmax`, `Tensor.logSoftmax`, and `TensorSoftmaxExpressions` Javadocs are
  complete without revision. They document every parameter, result, failure and validation-order
  condition; normalization slices; floating eligibility; exact Shape/type/eligibility retention;
  unresolved layout; absent label/storage; freshness and one-input provenance; the <span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>`[1, 2, 3]`
  approximate values; the sum-to-one and exponentiation relationships; and the numerical,
  gradient, compiler, backend, runtime, and execution boundaries.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- `docs/api/tensor-api.md` now treats both public expressions as current, adds them to the mental
  model and current expression inventory, and includes a complete Shape/axis/kind/provenance
  example whose stated output was checked against the focused tests and public contracts.<span data-glossary-navigation="true"> ([mental model glossary](../../../../glossary.md#term-mental-model))</span>
  `docs/api/compile-api.md` now accepts softmax expressions as current future-compiler inputs while
  keeping compiler entry, traversal, capture, normalization inference/canonicalization, optional
  decomposition, artifacts, and execution planned. <span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [artifacts glossary](../../../../glossary.md#term-artifact))</span>`docs/glossary.md` synchronizes softmax,
  normalized-axis, provenance, Tensor, operation-family, and current-status distinctions.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorSoftmaxExpressionTest --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; XML reports 8 focused
  softmax tests plus 14 Tensor tests, with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 58 XML suites report 449 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated `Tensor.html` includes both
  method signatures, parameter/result/failure sections, the ideal examples, floating/type/axis
  rules, exact metadata/provenance result, and deferred numerical/gradient/compiler/backend/
  runtime/execution behavior. Source review covers the package-private helper and private methods
  omitted from public generated output.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- `./gradlew test` — `BUILD SUCCESSFUL`; all 36 actionable repository test tasks completed without
  failure.
- `javap -p -c -s` confirmed the exact stateless helper shape, private constructor, one
  package-private `apply`, two private methods, deterministic validation order, one
  `Shape.normalizeAxis` call, one descriptor/operation/provenance/factory path, and each public
  method's single exact-kind helper call. Import and source scans found only permitted model/JDK
  dependencies and no value/storage access, graph/compiler/planning/runtime/prepare/backend state,
  numerical calculation, decomposition, or hidden state.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [graph glossary](../../../../glossary.md#graph); [prepare glossary](../../../../glossary.md#prepare))</span>
- A local Markdown target-and-anchor validator resolved all 271 links across the six changed
  documentation/planning files. Backtick fences are balanced, targeted trailing-whitespace checks
  found no matches, generated-Javadoc links render, status terminology is synchronized, and<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
  `git diff --check` passed.
- Exact scope review confirmed the required ten paths: two production files, two tests, Tensor
  API, Compile API, glossary, this task, model master plan, and roadmap. Task 0017 remains Draft,
  and no task-0017 specification exists. No commit or push was performed.<span data-glossary-navigation="true"> ([commit glossary](../../../../glossary.md#term-commit))</span>
- Training API remains accurate unchanged because this task adds no gradient formula, backward
  operation, autograd, optimizer, parameter, publication, or session behavior. <span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication))</span>`capabilities.md`
  remains accurate because it already inventories softmax/log-softmax and distinguishes model
  expression construction from later executable support.<span data-glossary-navigation="true"> ([log-softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>
- `SoftmaxKind`, `SoftmaxAttrs`, and `SoftmaxSemanticsTest` remain accurate unchanged because the
  new methods consume their existing first-class meanings and normalized-axis contract without
  changing semantic pairing or finite-precision policy. <span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [precision glossary](../../../../glossary.md#term-precision))</span>`DataType`, `Shape`, `TensorDescriptor`,
  `TensorFactory`, `TensorProvenance`, `Operation`, and the aggregate/scan and other expression
  contracts remain accurate because this task composes their existing validation, descriptor,
  identity, and provenance behavior without modifying those contracts.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `ARCHITECTURE.md`, focused architecture documents, ADRs, and architecture tests remain accurate
  unchanged because the work stays inside the authorized model layer and changes no module
  boundary, dependency direction, lifecycle, or architecture rule. Backend-conformance and
  integration tests remain unchanged because no backend behavior or end-to-end execution exists.
  Root/model Java 26 Gradle configuration and other modules remain unchanged because no dependency,
  source set, language level, preview/incubator feature, build task, or cross-module behavior
  changed.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [lifecycle glossary](../../../../glossary.md#lifecycle); [conformance glossary](../../../../glossary.md#term-conformance); [feature glossary](../../../../glossary.md#term-software-feature))</span>

## Implementation notes

- Added `Tensor.softmax(int)` and `Tensor.logSoftmax(int)` as exact one-call delegations to the
  dedicated package-private construction boundary.
- Added local floating validation, one Shape axis normalization, exact metadata retention,
  unresolved layout, first-class kind/axis attributes, exact one-input provenance, and fresh
  storage-free derived construction without numerical or cross-layer behavior.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [axis glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization); [metadata glossary](../../../../glossary.md#term-metadata); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>
- Added the focused eight-test suite and expanded only Tensor's public API-shape assertion for the
  two methods.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- The documentation pass left complete production Javadocs unchanged and finalized Tensor API,
  Compile API, glossary, and synchronized planning status/evidence.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: Implemented and documented public one-axis softmax and log-softmax Tensor
  expression construction with exact floating validation, axis normalization, metadata retention,
  and one-input provenance.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [Tensor glossary](../../../../glossary.md#tensor); [validation glossary](../../../../glossary.md#term-validation); [normalization glossary](../../../../glossary.md#term-normalization); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- Files changed or created: Exactly two production files, two tests, Tensor API, Compile API,
  glossary, this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused suites passed 22/22; all 449 model tests across 58 suites, model
  Javadoc, root tests, bytecode/reflection/import/source/generated-documentation/manual checks,
  271 Markdown link/anchor checks, fence/whitespace checks, exact scope/status checks, and<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0016j/review_model_0016j_docs` completed the independent pass using
  General, API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API, Compile API, and glossary now present softmax/log-softmax
  public expressions as current while preserving planned numerical, gradient, compiler, backend,
  runtime, and execution boundaries.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Javadoc review: Tensor and helper Javadocs are complete unchanged; all related foundational,
  semantic, descriptor, factory, provenance, aggregate/scan, and expression contracts remain
  accurate for the reasons recorded above.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Glossary impact: Softmax/log-softmax, normalized-axis, provenance, Tensor, and operation-family
  status now reflect current public expression construction.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0016J. Task 0017 remains Draft without a detailed
  specification.

Status: Complete
