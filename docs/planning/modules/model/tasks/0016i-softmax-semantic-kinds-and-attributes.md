# Task 0016I: [Softmax](../../../../glossary.md#softmax--log-softmax) Semantic Kinds and Attributes

## Status

Complete

## Goal

Define the typed, [backend](../../../../glossary.md#backend)-independent semantic [identities](../../../../glossary.md#term-identity-meanings) and [immutable](../../../../glossary.md#term-immutable) normalized-[axis](../../../../glossary.md#term-axis) [parameter](../../../../glossary.md#term-parameter)
for [softmax](../../../../glossary.md#softmax--log-softmax) and [log-softmax](../../../../glossary.md#softmax--log-softmax). The [model](../../../../glossary.md#term-model) must distinguish probability [normalization](../../../../glossary.md#term-normalization) from
[log](../../../../glossary.md#term-log)-probability normalization without retaining a [Tensor](../../../../glossary.md#tensor), deriving a result [descriptor](../../../../glossary.md#term-descriptor),
evaluating exponentials or logarithms, defining [gradients](../../../../glossary.md#term-gradient), or reporting backend support.

This task creates the semantic foundation consumed by task 0016J. It does not add public
`Tensor.softmax` or `Tensor.logSoftmax` expression methods.

## [Scope](../../../../glossary.md#term-scope)

- Add one public `SoftmaxKind` enum implementing [`OperationKind`](../../../../glossary.md#operationkind).
- Define exactly [`SOFTMAX`](../../../../glossary.md#softmax--log-softmax) and `LOG_SOFTMAX`, in that order.
- Add one public `SoftmaxAttrs` record implementing [`OperationAttrs`](../../../../glossary.md#operationattrs) with exactly one normalized
  non-negative `int axis` component.
- Reject every negative stored [axis](../../../../glossary.md#term-axis) with the exact normalized-axis failure contract.
- Define explicit typed composition of both kinds with `SoftmaxAttrs` without changing generic
  [`Operation`](../../../../glossary.md#operation) [validation](../../../../glossary.md#term-validation).
- Document one logical input, shape-preserving normalization slices, probability versus
  log-probability meaning, and their mathematical relationship without executing it.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [normalization glossary](../../../../glossary.md#term-normalization); [log glossary](../../../../glossary.md#term-log))</span>
- Document that the stored axis is already normalized while public negative-axis handling belongs
  to task 0016J.
- Add one focused same-package semantic-contract test.
- Add the cohesive `model.operation.normalization` package to the [model](../../../../glossary.md#term-model) package map.
- Finalize Javadocs, Tensor API semantic reference, glossary, task evidence, master plan, and
  roadmap through the required independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public `Tensor.softmax`, `Tensor.logSoftmax`, another Tensor method, overload, factory, builder,
  expression helper, or task-0016J implementation<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- temperature, scale, mask, bias, causal mask, epsilon, stabilization toggle, output type,
  precision, accumulation type, approximation mode, or algorithm option<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [bias glossary](../../../../glossary.md#term-bias); [causal mask glossary](../../../../glossary.md#causal-mask); [epsilon glossary](../../../../glossary.md#epsilon); [precision glossary](../../../../glossary.md#term-precision))</span>
- sparse, sampled, hierarchical, adaptive, masked, fused-attention, cross-entropy, NLL, loss, or
  optimizer semantics<span data-glossary-navigation="true"> ([fused glossary](../../../../glossary.md#specialization-and-fusion); [loss glossary](../../../../glossary.md#loss); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- input Tensor, Shape, caller-axis normalization, shape preservation implementation, result
  descriptor, label, identity, layout, storage, provenance, or <span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [axis glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization); [descriptor glossary](../../../../glossary.md#term-descriptor); [identity glossary](../../../../glossary.md#term-identity-meanings); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>`TensorFactory.createDerived`
- input DataType eligibility, output type, promotion, cast, local or graph-wide inference, or
  compatibility validation<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [graph glossary](../../../../glossary.md#graph); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation))</span>
- negative stored axis, an all-axes sentinel, no-axis form, multiple axes, axis collection, named
  axis, keep-dimensions flag, or optional axis<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension))</span>
- value/storage access, exponential, logarithm, maximum, sum, division, subtraction, allocation,
  materialization, mutation, aliasing, decomposition, or execution<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [aliasing glossary](../../../../glossary.md#term-aliasing); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- numerical stability algorithm, subtract-maximum implementation, reduction order, precision,
  overflow, underflow, NaN, infinity, signed zero, empty-axis, or error policy<span data-glossary-navigation="true"> ([numerical stability glossary](../../../../glossary.md#term-numerical-stability); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed zero glossary](../../../../glossary.md#term-signed-zero))</span>
- gradients, Jacobian-vector products, backward operation kinds, saved forward values, autograd,
  optimizer, or training behavior<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [Jacobian glossary](../../../../glossary.md#term-jacobian); [vector glossary](../../../../glossary.md#term-vector); [operation glossary](../../../../glossary.md#operation); [autograd glossary](../../../../glossary.md#autograd))</span>
- operation factory, registry, parser, visitor, string dispatch, arity, result-kind, family, cost,
  fusion, capability, backend route, kernel, or executable metadata<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [dispatch glossary](../../../../glossary.md#term-dispatch); [arity glossary](../../../../glossary.md#term-arity); [fusion glossary](../../../../glossary.md#term-fusion); [backend route glossary](../../../../glossary.md#backend-route); [kernel glossary](../../../../glossary.md#kernel); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- changes to Operation foundations, aggregate/scan contracts, DataType, Shape, Tensor, graph
  records, existing Java tests, dependencies, Gradle, architecture, or another module<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- compiler, planning, prepare, runtime, backend, engine, tracing, ONNX, conformance, integration,
  or a detailed task-0016J specification<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [engine glossary](../../../../glossary.md#term-engine); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance))</span>

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
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0016A](0016a-reduction-semantic-kinds-and-attributes.md)
- [Task 0016G](0016g-cumulative-sum-semantic-kind-and-attributes.md)
- [Task 0016H](0016h-cumulative-sum-tensor-expressions.md)
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

Both public methods require floating input, accept a positive or negative [axis](../../../../glossary.md#term-axis), preserve logical
[shape](../../../../glossary.md#shape) and [data type](../../../../glossary.md#data-type--datatype), and participate in legacy [gradient](../../../../glossary.md#term-gradient) construction. Legacy execution evidence
covers [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), [BFLOAT16](../../../../glossary.md#term-bfloat16), non-[contiguous](../../../../glossary.md#term-contiguous) inputs, attention and [loss](../../../../glossary.md#loss) composition, [ONNX](../../../../glossary.md#term-onnx),
[CPU](../../../../glossary.md#term-cpu), [Metal](../../../../glossary.md#term-metal), and [CUDA](../../../../glossary.md#term-cuda) paths.

Legacy public builders decompose both meanings into maximum, subtraction, exponential, sum,
division, and logarithm expressions, while separate historical [operation](../../../../glossary.md#operation) [descriptors](../../../../glossary.md#term-descriptor) also exist.
Neither representation is copied as an architectural rule. Synaptik's new [model](../../../../glossary.md#term-model) represents
[SOFTMAX](../../../../glossary.md#softmax--log-softmax) and LOG_SOFTMAX as first-class [backend](../../../../glossary.md#backend)-independent semantics. A later [compiler](../../../../glossary.md#term-compiler) may
canonicalize, decompose, or preserve them, and [concrete backend](../../../../glossary.md#concrete-backend) [prepare](../../../../glossary.md#prepare) may choose [fused](../../../../glossary.md#specialization-and-fusion) or
decomposed execution routes.

For one [normalization](../../../../glossary.md#term-normalization) [slice](../../../../glossary.md#slice) `x = [1, 2, 3]`, the ideal mathematical meanings are approximately:

| Kind | Semantic result |
|---|---|
| `SOFTMAX` | `[0.09003057, 0.24472847, 0.66524096]` |
| `LOG_SOFTMAX` | `[-2.40760596, -1.40760596, -0.40760596]` |

The SOFTMAX values sum to approximately one. Exponentiating each LOG_SOFTMAX value yields the
corresponding SOFTMAX value. These examples explain semantics only; this task calculates none of
them and selects no [finite](../../../../glossary.md#term-finite)-[precision](../../../../glossary.md#term-precision) algorithm.

## Architecture constraints

- [Operation](../../../../glossary.md#operation) kinds and attributes are [immutable](../../../../glossary.md#term-immutable) [backend](../../../../glossary.md#backend)-independent [model](../../../../glossary.md#term-model) semantics owned by
  `modules/model`.
- `SoftmaxKind.SOFTMAX` identifies normalized probabilities along one [axis](../../../../glossary.md#term-axis).
- `SoftmaxKind.LOG_SOFTMAX` identifies the logarithms of those normalized probabilities as a
  distinct first-class semantic kind, not an implicit [`SOFTMAX`](../../../../glossary.md#softmax--log-softmax) plus [`LOG`](../../../../glossary.md#term-log) [graph](../../../../glossary.md#graph) fragment.
- `SoftmaxAttrs` is the complete parameter type for both kinds. Its axis is already normalized
  and non-negative; a later Tensor expression validates and normalizes a caller axis against
  Shape.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape))</span>
- A normalization slice contains positions that differ only along the selected axis while all
  other logical coordinates remain fixed. This is semantic context, not stored Shape or arity.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [arity glossary](../../../../glossary.md#term-arity))</span>
- Both meanings have one logical input and preserve logical positions. Those facts are documented
  but not stored or implemented here.
- Ideal SOFTMAX meaning for [slice](../../../../glossary.md#slice) values `x_i` is `exp(x_i) / sum_j(exp(x_j))`.
- Ideal LOG_SOFTMAX meaning is `x_i - log(sum_j(exp(x_j)))`, mathematically equal to the logarithm
  of SOFTMAX for the same slice. Stable finite-precision evaluation remains executable policy.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [precision glossary](../../../../glossary.md#term-precision); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Generic `Operation` continues to validate only non-null kind and attributes. It must not discover
  families, enforce pairing, arity, axis bounds, DataType, Shape, gradient, or backend support.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Stable enum/record text is diagnostic, not serialization, ONNX, reflective dispatch, backend
  route, or kernel identity.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [ONNX glossary](../../../../glossary.md#term-onnx); [dispatch glossary](../../../../glossary.md#term-dispatch); [backend route glossary](../../../../glossary.md#backend-route); [kernel glossary](../../../../glossary.md#kernel); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Package direction is `model.operation.normalization -> model.operation` only. It must not depend
  on Tensor, datatype, shape, layout, storage, graph, compiler, planning, runtime, prepare,
  backend, or training packages.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare))</span>
- Stop if implementation requires another semantic component, Tensor/Shape/DataType behavior,
  numerical policy, compatibility validation, dependency, or architecture change.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing package used:

- `io.github.pho001.synaptik.model.operation` — supplies [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), and
  generic immutable <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>[`Operation`](../../../../glossary.md#operation) composition.

Package added:

```text
io.github.pho001.synaptik.model.operation.normalization
  Typed shape-preserving normalization meanings and immutable normalization parameters.
```

The package is separate from [aggregate reduction](../../../../glossary.md#aggregate-reduction) because neither operation removes or retains an
[axis](../../../../glossary.md#term-axis) with [extent](../../../../glossary.md#term-extent) one; each emits one normalized result at every input position. It is separate
from [scan](../../../../glossary.md#term-cumulative-scan) because every output depends on the complete selected [slice](../../../../glossary.md#slice) rather than an ordered
prefix. The package can later [host](../../../../glossary.md#term-host) other cohesive [normalization](../../../../glossary.md#term-normalization) semantics only when their focused
tasks define them; this task does not predict or add those contracts.

Type placement:

- `io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind` — public probability and
  log-probability semantic identities.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([log glossary](../../../../glossary.md#term-log))</span>
- `io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs` — public immutable
  normalized-axis parameter shared by those identities.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>
- `SoftmaxSemanticsTest` — same-package focused test for both cohesive contracts.

## Required contract

### Semantic kind vocabulary

Create exactly:

```java
public enum SoftmaxKind implements OperationKind {
    SOFTMAX,
    LOG_SOFTMAX
}
```

The enum declares no project field, explicit constructor, method, nested type, per-[constant](../../../../glossary.md#term-constant) class
body, [alias](../../../../glossary.md#aliasing-and-arena), [symbol](../../../../glossary.md#term-symbol), or [metadata](../../../../glossary.md#term-metadata). [Compiler](../../../../glossary.md#term-compiler)-generated enum machinery is not additional project [API](../../../../glossary.md#term-application-programming-interface-api).
Inherited `Enum.name()` satisfies `OperationKind.name()` and returns exact text [`SOFTMAX`](../../../../glossary.md#softmax--log-softmax) or
`LOG_SOFTMAX`.

Each constant means one-input [shape](../../../../glossary.md#shape)-preserving [normalization](../../../../glossary.md#term-normalization) along the [normalized axis](../../../../glossary.md#normalized-axis) carried by
`SoftmaxAttrs`. Neither defines input eligibility, result [descriptor](../../../../glossary.md#term-descriptor), numerical algorithm,
[gradient](../../../../glossary.md#term-gradient) behavior, [decomposition](../../../../glossary.md#term-decomposition), execution, or [backend](../../../../glossary.md#backend) availability.

### [Axis](../../../../glossary.md#term-axis) attributes

Create exactly:

```java
public record SoftmaxAttrs(int axis) implements OperationAttrs
```

The record has exactly one component, one public [canonical](../../../../glossary.md#term-canonical-form) constructor, an explicit documented
`axis()` accessor, and record-generated `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), and `toString`. Add no overload,
factory, builder, temperature, [epsilon](../../../../glossary.md#epsilon), stability flag, field, nested type, optional, sentinel,
[cache](../../../../glossary.md#term-cache), or helper [API](../../../../glossary.md#term-application-programming-interface-api).

The canonical constructor rejects every negative axis before ordinary record assignment with
`IllegalArgumentException` and exact message:

```text
axis must be non-negative: <axis>
```

Accept zero, positive values, and `Integer.MAX_VALUE` structurally. The record cannot prove that
the axis exists for an eventual input [Shape](../../../../glossary.md#shape).

### Typed composition and mathematical meaning

Document these valid pairings without adding generic compatibility [validation](../../../../glossary.md#term-validation):

```java
new Operation(SoftmaxKind.SOFTMAX, attrs)
new Operation(SoftmaxKind.LOG_SOFTMAX, attrs)
```

For one [normalization](../../../../glossary.md#term-normalization) [slice](../../../../glossary.md#slice) with values `x_0 ... x_n`:

- [SOFTMAX](../../../../glossary.md#softmax--log-softmax) output at position `i` is the positive normalized exponential of `x_i`; ideal outputs
  sum to one across that slice.
- LOG_SOFTMAX output at position `i` is the natural logarithm of the corresponding ideal SOFTMAX
  output.<span data-glossary-navigation="true"> ([natural logarithm glossary](../../../../glossary.md#norms-elementary-functions-and-probability-distributions))</span>
- Both preserve every input position and the axis order. They do not reduce rank, reverse
  positions, or retain a singleton reduction axis.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [singleton glossary](../../../../glossary.md#term-singleton))</span>

Do not define [finite](../../../../glossary.md#term-finite)-[precision](../../../../glossary.md#term-precision) guarantees, [evaluation](../../../../glossary.md#term-evaluation) [decomposition](../../../../glossary.md#term-decomposition), maximum subtraction,
summation order, special-value policy, empty-[axis](../../../../glossary.md#term-axis) behavior, [gradient](../../../../glossary.md#term-gradient) formulas, or executable
support. Generic [`Operation`](../../../../glossary.md#operation) remains unchanged and does not enforce the pairing or any family
rule.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/SoftmaxKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/SoftmaxAttrs.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/normalization/SoftmaxSemanticsTest.java`

Documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistent: [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api), Training API, capabilities,
[Operation](../../../../glossary.md#operation) foundations, aggregate/[scan](../../../../glossary.md#term-cumulative-scan) contracts, focused architecture, ADRs/tests,
[conformance](../../../../glossary.md#term-conformance)/integration tests, and [Gradle](../../../../glossary.md#term-gradle) configuration.

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, one test, and five documentation/[planning](../../../../glossary.md#term-planning) files: eight paths total.
Do not modify an existing Java contract or test. Stop beyond this scope or if [Tensor](../../../../glossary.md#tensor)/[Shape](../../../../glossary.md#shape)/[DataType](../../../../glossary.md#data-type--datatype)
behavior, another semantic component, numerical policy, compatibility [validation](../../../../glossary.md#term-validation), [dependency](../../../../glossary.md#term-dependency), or
architecture change is needed. Do not create task 0016J.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document the enum, both [constants](../../../../glossary.md#term-constant), the record, [canonical](../../../../glossary.md#term-canonical-form) constructor, and explicit accessor.
- Explain normalization axis and slice, shape preservation, SOFTMAX probability meaning,
  LOG_SOFTMAX log-probability meaning, and their mathematical relationship for a newcomer.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [axis glossary](../../../../glossary.md#term-axis); [slice glossary](../../../../glossary.md#slice); [shape glossary](../../../../glossary.md#shape); [SOFTMAX glossary](../../../../glossary.md#softmax--log-softmax); [log glossary](../../../../glossary.md#term-log))</span>
- Include the concrete `[1, 2, 3]` example with approximate outputs and explain the sum-to-one and
  exponentiation relationship.
- Document every parameter, structural axis validation, accepted boundaries, accessor result, and
  exact failure.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [validation glossary](../../../../glossary.md#term-validation))</span>
- Clearly distinguish ideal mathematical semantics from finite-precision algorithms and defer
  DataType eligibility, descriptor/provenance construction, numerical policy, gradients,
  compiler decomposition, storage, backend support, and execution.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [precision glossary](../../../../glossary.md#term-precision); [DataType glossary](../../../../glossary.md#data-type--datatype); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [decomposition glossary](../../../../glossary.md#term-decomposition); [backend glossary](../../../../glossary.md#backend))</span>
- Review Operation foundations and reduction/scan-family Javadocs and record reasoned no-change
  conclusions or stop.<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>

## Acceptance criteria

- Exactly one public two-constant enum, one public one-component record, and one focused test are
  added under the planned normalization package; no other Java file or API changes.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Exact enum [constants](../../../../glossary.md#term-constant)/order/names and absence of project [metadata](../../../../glossary.md#term-metadata) are verified.
- Exact record component/type, interface, constructor/accessor, and absence of extra project
  API/state are verified.
- Negative [axes](../../../../glossary.md#term-axis) fail with exact type/message; zero, positive, and maximum axes succeed.
- Both kinds compose with the exact `SoftmaxAttrs` reference through generic [`Operation`](../../../../glossary.md#operation).
- Record value semantics, diagnostic text, and typed distinction from aggregate and scan kinds or
  attributes are verified.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Javadocs explain both mathematical meanings and their relationship without selecting an
  algorithm, result contract, gradient, or backend behavior.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend))</span>
- No Tensor, DataType, Shape, storage, graph, compiler, runtime, backend, dependency, build, or
  architecture behavior is added.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [DataType glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused/aggregate tests, Javadoc, root tests, reflection/javap/import/scope and documentation
  validation pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [validation glossary](../../../../glossary.md#term-validation))</span>
- A separate clean-context documentation-focused agent finalizes permitted Javadocs, Tensor API,
  glossary, planning, examples, and no-change conclusions.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- 0016I becomes Complete only after both passes; 0016J remains Draft without a specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.normalization.SoftmaxSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test covers enum/package/[API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape); exact [constant](../../../../glossary.md#term-constant) order/names; record component and
explicit accessor; negative/boundary [axes](../../../../glossary.md#term-axis); generated value semantics and exact diagnostic text;
both typed [Operation](../../../../glossary.md#operation) compositions and exact attributes [identity](../../../../glossary.md#term-identity-meanings); distinct identities from
[aggregate reduction](../../../../glossary.md#aggregate-reduction) and cumulative-sum contracts; ideal example relationships within a declared
[tolerance](../../../../glossary.md#term-tolerance); and absence of fields, nested types, [Tensor](../../../../glossary.md#tensor)/Shape/[DataType](../../../../glossary.md#data-type--datatype)/execution state, [aliases](../../../../glossary.md#aliasing-and-arena),
registries, or [metadata](../../../../glossary.md#term-metadata).

Manually inspect reflection, `javap -p -c -s`, source, and imports for the exact enum/record shape,
constructor validation before assignment, explicit accessor, no extra project state/API, and no
cross-layer type or behavior. Validate generated [Javadoc](../../../../glossary.md#term-javadoc), Tensor API/glossary, examples,
links/anchors/fences/whitespace, exact eight paths, synchronized statuses, package map, and absence
of a task-0016J specification.

## Dependencies

- 0005 supplies [`OperationKind`](../../../../glossary.md#operationkind) and [`OperationAttrs`](../../../../glossary.md#operationattrs).
- 0006 supplies [immutable](../../../../glossary.md#term-immutable) generic [`Operation`](../../../../glossary.md#operation) composition.
- 0016A and 0016G establish normalized-axis conventions and typed separation from aggregate and
  scan semantics.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>

## Follow-up tasks

- 0016J remains Draft for public floating `Tensor.softmax` and `Tensor.logSoftmax` expression
  methods, caller-axis normalization, shape/type/eligibility retention, descriptor/provenance
  construction, and freshness.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization); [shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- Compiler tasks own capture, inference validation, canonicalization, optional decomposition,
  autograd expansion, and graph optimization.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [decomposition glossary](../../../../glossary.md#term-decomposition); [autograd glossary](../../../../glossary.md#autograd); [graph optimization glossary](../../../../glossary.md#forward-graph-optimization))</span>
- Backend/config/conformance tasks own stable finite-precision algorithms, storage traversal,
  fused/decomposed routes, kernels, and cross-backend parity.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [finite glossary](../../../../glossary.md#term-finite); [precision glossary](../../../../glossary.md#term-precision); [fused glossary](../../../../glossary.md#specialization-and-fusion); [kernels glossary](../../../../glossary.md#kernel))</span>
- Loss and attention tasks may later compose these semantics without extending this attributes
  contract silently.<span data-glossary-navigation="true"> ([Loss glossary](../../../../glossary.md#loss))</span>

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. This task adds [immutable](../../../../glossary.md#term-immutable) [model](../../../../glossary.md#term-model)-owned [normalization](../../../../glossary.md#term-normalization) semantics in a cohesive
[operation](../../../../glossary.md#operation) package without changing [module](../../../../glossary.md#term-module) boundaries, [dependency direction](../../../../glossary.md#java-platform-and-build-tools), or [lifecycle](../../../../glossary.md#lifecycle)
[ownership](../../../../glossary.md#term-ownership-meanings). Stop if an architecture change is required.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0005/0006/0016A/0016G/0016H/0016I, Tensor API, Compile API,
Training API, glossary, current OperationKind/OperationAttrs/Operation and aggregate/scan
contracts/tests, and Java 26 Gradle configuration.

Implement task 0016I exactly. Add only SoftmaxKind.java, SoftmaxAttrs.java, and
SoftmaxSemanticsTest.java under io.github.pho001.synaptik.model.operation.normalization for Java
code/tests.

The public enum implements OperationKind and contains exactly SOFTMAX and LOG_SOFTMAX in order,
with no project fields, methods, nested types, aliases, arity, or metadata. The public record
implements OperationAttrs and contains exactly int axis, an explicit documented accessor, exact
negative-axis validation/message, and no extra API/state. Document shape-preserving normalization
slices, probability versus log-probability meaning, their mathematical relationship, and explicit
kind/attributes pairings.

Do not add Tensor methods, DataType/Shape/result/provenance rules, numerical algorithms,
decomposition, gradients, graph/compiler/planning/runtime/backend behavior, factories/registries,
dependencies, build/architecture changes, existing Java edits, or later specs. Stop beyond eight
paths or on architecture uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff/evidence to a separate
clean-context documentation agent in the same change. It must inspect source/tests/generated
Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record related-contract/
capability/Compile API/Training API/architecture no-change conclusions, and rerun validation.

Update task 0016I, model master plan, and roadmap only for planning status/evidence. Do not mark
0016I Complete until both passes succeed. Leave 0016J Draft without a specification. Do not commit
or push.
```

## Local decisions

- Use one `SoftmaxKind` enum for the two closely related meanings and one shared `SoftmaxAttrs`
  axis value. Separate kind classes or duplicate attribute records would split one cohesive
  family without a current semantic difference in parameters.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- Add `operation.normalization` rather than place shape-preserving normalization under aggregate
  reduction or ordered scan. The package reflects semantic responsibility, not the reductions a
  numerical algorithm may use internally.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [normalization glossary](../../../../glossary.md#term-normalization); [aggregate reduction glossary](../../../../glossary.md#aggregate-reduction); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Represent SOFTMAX and LOG_SOFTMAX as first-class model meanings. Legacy public decomposition is
  implementation history; compiler canonicalization and backend prepare later own whether an
  executable path remains fused or becomes primitive operations.<span data-glossary-navigation="true"> ([SOFTMAX glossary](../../../../glossary.md#softmax--log-softmax); [model glossary](../../../../glossary.md#term-model); [decomposition glossary](../../../../glossary.md#term-decomposition); [compiler glossary](../../../../glossary.md#term-compiler); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare); [fused glossary](../../../../glossary.md#specialization-and-fusion))</span>
- Store only a normalized axis. One input, Shape preservation, output type, floating eligibility,
  ideal sum-to-one/log relationships, and gradients are either documented context or later-task
  behavior, not duplicated attributes.<span data-glossary-navigation="true"> ([normalized axis glossary](../../../../glossary.md#normalized-axis); [log glossary](../../../../glossary.md#term-log); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Defer temperature, masks, scaling, and numerical-stability options. Existing selected capability
  evidence requires none of them on the public softmax contracts.<span data-glossary-navigation="true"> ([masks glossary](../../../../glossary.md#term-boolean-mask))</span>

## Known limitations

- No public Tensor expressions, Shape normalization/descriptor, provenance, numerical evaluation,
  or gradients yet.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [normalization glossary](../../../../glossary.md#term-normalization); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [evaluation glossary](../../../../glossary.md#term-evaluation); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- No finite-precision, empty-axis, NaN/infinity, compiler-decomposition, backend, ONNX, or kernel
  policy.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [precision glossary](../../../../glossary.md#term-precision); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [compiler glossary](../../../../glossary.md#term-compiler); [decomposition glossary](../../../../glossary.md#term-decomposition); [backend glossary](../../../../glossary.md#backend); [ONNX glossary](../../../../glossary.md#term-onnx); [kernel glossary](../../../../glossary.md#kernel))</span>
- Only ordinary one-[axis](../../../../glossary.md#term-axis) [softmax](../../../../glossary.md#softmax--log-softmax) and [log-softmax](../../../../glossary.md#softmax--log-softmax) meanings are represented.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) read the [architecture contract](../../../../glossary.md#architecture-contract) and focused architecture explanations; documentation and
planning rules; roadmap; [model](../../../../glossary.md#term-model) capabilities/master plan; tasks 0005, 0006, 0016A, 0016G, and
0016H; current [Operation](../../../../glossary.md#operation), aggregate, and [scan](../../../../glossary.md#term-cumulative-scan) contracts/tests; [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/Training APIs;
glossary; and Java 26 [Gradle](../../../../glossary.md#term-gradle) configuration.

The legacy branch was read directly. It confirms exactly one-[axis](../../../../glossary.md#term-axis) [`softmax`](../../../../glossary.md#softmax--log-softmax) and `logSoftmax`
public methods, floating eligibility, negative-axis [normalization](../../../../glossary.md#term-normalization), [shape](../../../../glossary.md#shape)/type preservation,
[gradient](../../../../glossary.md#term-gradient) composition, stable mathematical intent, [FLOAT64](../../../../glossary.md#term-float64)/[FLOAT32](../../../../glossary.md#term-float32)/[BFLOAT16](../../../../glossary.md#term-bfloat16), non-[contiguous](../../../../glossary.md#term-contiguous) input,
attention/[loss](../../../../glossary.md#loss) composition, [ONNX](../../../../glossary.md#term-onnx), [CPU](../../../../glossary.md#term-cpu), [Metal](../../../../glossary.md#term-metal), and [CUDA](../../../../glossary.md#term-cuda) evidence. Legacy [decomposition](../../../../glossary.md#term-decomposition), [mutable](../../../../glossary.md#term-mutable)
gradient callbacks, operation traits, storage, [lowering](../../../../glossary.md#lowering), [fusion](../../../../glossary.md#term-fusion), and [kernels](../../../../glossary.md#kernel) are excluded or
assigned to later [owners](../../../../glossary.md#term-owner).

Planning selected one two-[constant](../../../../glossary.md#term-constant) kind, one one-component attributes record, one focused test,
and the new normalization package. Existing Operation foundations suffice; no [dependency](../../../../glossary.md#term-dependency),
foundational contract, or architecture change is required.

Pre-implementation planning validation after synchronizing this task, the model master plan, and
roadmap:

- `git diff --check` passed.
- The targeted trailing-whitespace scan returned no matches across the three planning files.
- All 175 relative Markdown links across the three planning files resolve locally.
- Markdown fence counts are balanced: sixteen in this task, two in the master plan, and zero in
  the roadmap.
- All 20 task-template headings are present, together with the focused Capability origin,
  Required contract, and Javadoc requirements sections.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- The `[1, 2, 3]` SOFTMAX and LOG_SOFTMAX examples were independently recalculated; the displayed
  rounded values match and the unrounded SOFTMAX values sum to one.
- Task, model master plan, and roadmap consistently identify 0016I as [Ready](../../../../glossary.md#term-ready) and 0016J as Draft.
- No detailed task-0016J specification exists.
- Repository scope is exactly this task, the model master plan, and the roadmap; no Java, API,
  architecture, Gradle, or other file changed during planning.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

Implementation and independent documentation validation:

- The implementation pass added exactly `SoftmaxKind`, `SoftmaxAttrs`, and the same-package
  `SoftmaxSemanticsTest`. It changed no existing Java source or test.
- Clean documentation-focused context `/root/review_model_0016a_docs` performed a fresh task-0016I
  review and applied General style, API and Javadoc style, Planning style, and Example format. It
  independently inspected the complete implementation diff, final normalization and related
  operation/reduction/scan source and tests, generated model Javadoc, bytecode, imports, APIs,
  glossary, planning state, architecture documents, and Java 26 build configuration.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Both new production Javadocs are complete without revision. They document one logical input,
  shape-preserving normalization slices, the normalized-axis boundary, explicit kind/attributes
  pairings, probability and log-probability meanings, the <span data-glossary-navigation="true"> ([log glossary](../../../../glossary.md#term-log))</span>`[1, 2, 3]` example, sum-to-one and
  exponentiation relationships, every constructor parameter and accessor result, the exact
  negative-axis failure, immutable value semantics, diagnostic-only text, and all deferred
  Tensor, descriptor, numerical, gradient, compiler, backend, and execution behavior.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [immutable glossary](../../../../glossary.md#term-immutable); [descriptor glossary](../../../../glossary.md#term-descriptor); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- `docs/api/tensor-api.md` now classifies `SoftmaxKind` and `SoftmaxAttrs` as current semantic
  contracts, explains a normalization slice, gives both explicit pairings and the ideal<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>
  `[1, 2, 3]` outputs, and keeps public `Tensor.softmax` and `Tensor.logSoftmax` construction
  planned for task 0016J. `docs/glossary.md` now defines softmax/log-softmax and synchronizes the
  normalized-axis, operation-kind, and operation-attributes status distinctions.<span data-glossary-navigation="true"> ([log-softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.normalization.SoftmaxSemanticsTest` —
  `BUILD SUCCESSFUL`; the XML report contains 9 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 57 XML suites contain 441 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; generated pages contain both public
  normalization types, both enum constants, the canonical constructor, explicit accessor,
  parameter/result/failure documentation, both ideal examples, and the semantic-versus-executable
  boundary.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- `./gradlew test` — `BUILD SUCCESSFUL`; the complete repository lifecycle finished 36 actionable
  tasks without a failing task.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `javap -p -c -s` confirmed exactly two ordered enum constants with no project state or behavior,
  and exactly one private final record component, one public constructor, one explicit accessor,
  generated record value methods, and negative-axis validation before field assignment. Focused
  reflection tests independently confirm the exact public/final shapes, interfaces, constructors,
  declared methods, absence of nested types and extra API/state, boundary axes, exact diagnostic
  text, typed distinction from aggregate and scan contracts, and exact-reference `Operation`
  composition.
- Production import inspection found only [`OperationKind`](../../../../glossary.md#operationkind) and [`OperationAttrs`](../../../../glossary.md#operationattrs). No Tensor,
  DataType, Shape, layout, storage, graph, compiler, planning, runtime, prepare, backend, trace, or
  training dependency or state was introduced. Package placement and direction are exactly<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [graph glossary](../../../../glossary.md#graph); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [trace glossary](../../../../glossary.md#trace))</span>
  `model.operation.normalization -> model.operation`, matching the master-plan package map.
- Local Markdown target-and-anchor validation passed for all 261 relative links in the five
  changed documentation/planning files. Markdown fences are balanced, targeted trailing-whitespace
  and stale-status scans passed, the numerical examples were recalculated by the focused test,
  generated Javadoc rendering was inspected, and <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>`git diff --check` passed.
- Final repository scope is exactly the authorized eight paths: two new production files, one new
  focused test, Tensor API, glossary, this task, model master plan, and roadmap. Task 0016I is
  synchronized as Complete. Task 0016J remains Draft, and no `0016j-*` specification exists.
- `OperationKind`, `OperationAttrs`, and `Operation` remain accurate unchanged because the new
  family implements their open typed contracts and does not change generic compatibility
  validation. Aggregate-reduction and cumulative-sum contracts and tests remain accurate
  unchanged because normalization preserves positions, depends on a complete slice, and uses a
  separate typed family without modifying either existing API.
- `capabilities.md` remains accurate unchanged because it already inventories softmax and
  log-softmax while distinguishing semantic representation from later public and executable
  support. Compile API remains accurate unchanged because this task adds no public expression,
  provenance, capture entry point, inference, canonicalization, artifact, or execution behavior.
  Training API remains accurate unchanged because no gradient eligibility, gradient rule,
  autograd, optimizer, publication, or session behavior changed.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [artifact glossary](../../../../glossary.md#term-artifact); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication))</span>
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance tests, and integration tests remain accurate unchanged because this task
  changes no module boundary, dependency direction, lifecycle, backend behavior, or end-to-end
  execution. Java 26 root/model Gradle configuration and all other modules remain accurate
  unchanged because no dependency, source set, language level, preview/incubator feature, task,
  module, or cross-module behavior changed.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [feature glossary](../../../../glossary.md#term-software-feature))</span>

## Implementation notes

- Added the exact two-[constant](../../../../glossary.md#term-constant) `SoftmaxKind` semantic enum and exact one-component `SoftmaxAttrs`
  record in the new cohesive normalization package.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>
- Added the focused nine-test contract suite covering exact API shape, structural axis validation,
  generated record semantics, both explicit pairings, typed family separation, the ideal example,
  and cross-layer exclusions.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [axis glossary](../../../../glossary.md#term-axis); [validation glossary](../../../../glossary.md#term-validation))</span>
- The documentation-focused pass finalized Tensor API, glossary, task evidence, model master plan,
  and roadmap. It changed no Java source because the submitted Javadocs were already complete.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [model glossary](../../../../glossary.md#term-model))</span>
- No public Tensor method, descriptor, provenance, numerical algorithm, gradient, compiler,
  runtime, backend, dependency, build, or architecture behavior was added.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Completion summary

- Completed changes: Implemented and documented the backend-independent softmax and log-softmax
  semantic identities and their immutable normalized-axis attributes.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [immutable glossary](../../../../glossary.md#term-immutable); [axis glossary](../../../../glossary.md#term-axis))</span>
- Files changed or created: Exactly two production Java files, one focused test, Tensor API,
  glossary, this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused normalization semantics 9/9, all 441 model tests across 57 suites,
  generated model Javadoc, root tests, bytecode/reflection/import checks, numerical-example and
  rendered-documentation review, Markdown link/anchor/fence/whitespace checks, exact scope/status
  checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [normalization glossary](../../../../glossary.md#term-normalization); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation-agent review: Clean context `/root/review_model_0016a_docs` completed the mandatory
  fresh independent pass using General, API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and glossary now distinguish current softmax/log-softmax
  semantics from planned public Tensor construction and executable behavior; task, master plan,
  and roadmap are synchronized.
- Javadoc review: Both new public contracts and all required members are complete unchanged;
  operation foundations and aggregate/scan Javadocs remain accurate.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Glossary impact: Added softmax/log-softmax terminology and synchronized normalized-axis and
  operation-family status.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0016I. Task 0016J remains Draft without a detailed
  specification.

Status: Complete
