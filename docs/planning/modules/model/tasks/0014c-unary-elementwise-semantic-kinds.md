# Task 0014C: [Unary](../../../../glossary.md#term-unary) [Elementwise](../../../../glossary.md#term-elementwise) Semantic Kinds

**Editorial reading key (later context, not a status update).**

A unary elementwise transform acts independently at each input coordinate, rather than combining a row or axis into one total. Parameterless kinds identify the requested functions; a fast variant is an explicit approximation request, not a backend capability flag. Here Model defines a typed meaning for later construction code; this semantic-family task neither receives stored operands nor evaluates them. Its completion is not evidence of Compiler capture, derivatives or backend execution.

Focused current reading: [Current unary-transform reference](../../../../reference/tensor/elementwise.md#unary-numeric-transforms-and-floating-classifications). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Introduce one typed, [backend](../../../../glossary.md#backend)-independent vocabulary for the fifteen parameterless [unary](../../../../glossary.md#term-unary)
[elementwise](../../../../glossary.md#term-elementwise) capabilities selected by the [model](../../../../glossary.md#term-model) baseline: arithmetic transforms, transcendental
functions, [activations](../../../../glossary.md#term-activation), and explicit fast approximation variants.

This task defines semantic [identities](../../../../glossary.md#term-identity-meanings) only. It gives the following public [Tensor](../../../../glossary.md#tensor)-expression task
stable kinds to compose with [`Operation`](../../../../glossary.md#operation) and `NoOperationAttrs.INSTANCE`, without adding Tensor
methods, result [descriptors](../../../../glossary.md#term-descriptor), [provenance](../../../../glossary.md#provenance), numerical execution, [gradient](../../../../glossary.md#term-gradient) behavior, or backend
[metadata](../../../../glossary.md#term-metadata).

## [Scope](../../../../glossary.md#term-scope)

- Add one public enum `UnaryElementwiseKind` implementing [`OperationKind`](../../../../glossary.md#operationkind).
- Define exactly `ABS`, `NEG`, `INV`, [`LOG`](../../../../glossary.md#term-log), [`EXP`](../../../../glossary.md#term-expx), [`ERF`](../../../../glossary.md#term-erf), `SQRT`, `FLOOR`, `CEIL`, `SIGN`,
  `RELU`, [`SIGMOID`](../../../../glossary.md#term-sigmoid), [`TANH`](../../../../glossary.md#term-tanh), `FAST_EXP`, and `FAST_TANH` in that order.
- Document the mathematical or [activation](../../../../glossary.md#term-activation) meaning of every kind.
- Establish that all fifteen kinds have one logical input but no intrinsic attributes and compose
  with `NoOperationAttrs.INSTANCE` when represented by [`Operation`](../../../../glossary.md#operation).
- Add one focused test proving exact enum vocabulary, typed OperationKind behavior, parameterless
  Operation composition, and absence of extra family state or behavior.
- Add the cohesive `model.operation.elementwise.unary` package to the [model](../../../../glossary.md#term-model) package map.
- Record the post-0014B checkpoint decision to continue the ordered model queue rather than hide a
  multi-module prerequisite chain inside a cross-module task.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [module glossary](../../../../glossary.md#term-module))</span>
- Finalize affected Javadocs, Tensor API, glossary, task evidence, model master plan, and roadmap
  through the required independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public [Tensor](../../../../glossary.md#tensor) methods for any [unary](../../../../glossary.md#term-unary) [operation](../../../../glossary.md#operation) or [activation](../../../../glossary.md#term-activation)
- result Tensor construction, descriptor creation, provenance, labels, storage, IDs, expression
  chaining, or <span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>`TensorFactory.createDerived`
- data-type eligibility, result data type, shape preservation, layout resolution, gradient
  eligibility propagation, or local/graph-wide inference<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph); [inference glossary](../../../../glossary.md#term-inference))</span>
- numeric evaluation, domain or range rules, overflow/underflow, NaN, infinity, signed-zero,
  rounding, exactness, accuracy bounds, monotonicity, or approximation algorithm selection<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- gradient rules, subgradient conventions at zero, backward graph operations, autograd, optimizer,
  or training behavior<span data-glossary-navigation="true"> ([backward graph glossary](../../../../glossary.md#backward-graph); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- a family attributes type, factory, wrapper, registry, parser, aliases, symbols, string dispatch,
  serialization token, classpath scanning, or reflective discovery<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [dispatch glossary](../../../../glossary.md#term-dispatch); [serialization glossary](../../../../glossary.md#term-serialization); [token glossary](../../../../glossary.md#term-token); [classpath glossary](../../../../glossary.md#term-classpath))</span>
- category fields, arithmetic/transcendental/activation tags, strict/fast flags, arity metadata,
  computational cost, fusion flags, result-kind tags, differentiability metadata, capability
  matrices, backend support, routes, lowering, or execution<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [metadata glossary](../../../../glossary.md#term-metadata); [fusion glossary](../../../../glossary.md#term-fusion); [matrices glossary](../../../../glossary.md#term-review-matrix); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering))</span>
- scalar multiplication, scalar power, clamp, clampMin, or clampMax; these carry scalar parameters
  and remain in tasks 0014E–0014F<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [carry glossary](../../../../glossary.md#term-carry); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- binary, comparison, logical, selection, cast, reduction, layout, indexing, linear algebra,
  convolution, normalization, loss, or compiler-generated operation kinds<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [linear glossary](../../../../glossary.md#term-linear-meanings); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [normalization glossary](../../../../glossary.md#term-normalization); [loss glossary](../../../../glossary.md#loss); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- modifying existing Operation foundation, binary arithmetic, Tensor, descriptor, provenance,
  factory, graph, Gradle, dependency, architecture, another module, or task-0014D specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>

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
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0014A](0014a-binary-arithmetic-semantic-kinds.md)
- [Task 0014B](0014b-binary-arithmetic-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Post-0014B [checkpoint](../../../../glossary.md#term-checkpoint) decision

Task 0014B completed the first public [Tensor](../../../../glossary.md#tensor) [operation](../../../../glossary.md#operation) family that a future [compiler](../../../../glossary.md#term-compiler) can [capture](../../../../glossary.md#term-capture-meanings).
The planned checkpoint therefore reconsidered whether the next task should leave `modules/model`
and begin a vertical [compile](../../../../glossary.md#compile)-to-execution [slice](../../../../glossary.md#slice).

The project will continue the ordered [model](../../../../glossary.md#term-model) queue with task 0014C. The downstream [`trace`](../../../../glossary.md#trace),
`backend-contract`, `config`, [`planning`](../../../../glossary.md#term-planning), and `compiler` [modules](../../../../glossary.md#term-module) currently contain placeholder
classes and broad master plans but no detailed foundational tasks or implemented contracts. A
so-called vertical next task would therefore cross several architecture boundaries and conceal a
large prerequisite sequence instead of validating one bounded integration seam. The global roadmap
already places those areas after the model frontier.

This is an implementation-order decision, not a new [dependency](../../../../glossary.md#term-dependency) or architecture rule. A future
explicit roadmap decision may reorder work when a bounded cross-module task and its prerequisites
are concrete. Until then, the sequential model queue remains the safer executable plan.

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes public [Tensor](../../../../glossary.md#tensor) methods `abs`, `neg`, `inv`, [`log`](../../../../glossary.md#term-log),
[`exp`](../../../../glossary.md#term-expx), [`erf`](../../../../glossary.md#term-erf), `sqrt`, `floor`, `ceil`, `sign`, `relu`, [`sigmoid`](../../../../glossary.md#term-sigmoid), [`tanh`](../../../../glossary.md#term-tanh), `fastExp`, and
`fastTanh`. Each legacy [operation](../../../../glossary.md#operation) [descriptor](../../../../glossary.md#term-descriptor) is parameterless and represents a [shape](../../../../glossary.md#shape)-preserving
[unary](../../../../glossary.md#term-unary) [elementwise](../../../../glossary.md#term-elementwise) request. Legacy tests exercise the strict and fast variants across expression
chains, multiple floating data types, non-[contiguous](../../../../glossary.md#term-contiguous) inputs, [gradients](../../../../glossary.md#term-gradient), [fusion](../../../../glossary.md#term-fusion), and [backend](../../../../glossary.md#backend) paths.

Only the fifteen semantic [identities](../../../../glossary.md#term-identity-meanings) and their stable short names are retained here. Legacy [arity](../../../../glossary.md#term-arity),
semantic-family, cost, fusion, result-kind, expression-string, gradient, capability, storage,
[lowering](../../../../glossary.md#lowering), and execution traits belong to later owning layers and are not copied. [Scalar](../../../../glossary.md#term-scalar) power,
scalar multiplication, and clamp operations are deliberately excluded because their scalar values
are intrinsic typed attributes rather than parameterless unary semantics.

## Architecture constraints

- Production remains in `modules/model`, which owns [backend](../../../../glossary.md#backend)-independent [operation](../../../../glossary.md#operation) semantics.
- `UnaryElementwiseKind` identifies a mathematical or activation meaning. It is not an input,
  graph occurrence, Tensor provenance value, result descriptor, executable operation, backend
  capability, approximation implementation, or kernel route.<span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation); [graph glossary](../../../../glossary.md#graph); [occurrence glossary](../../../../glossary.md#term-occurrence); [Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance); [descriptor glossary](../../../../glossary.md#term-descriptor); [backend capability glossary](../../../../glossary.md#backend-capability); [kernel glossary](../../../../glossary.md#kernel))</span>
- The enum implements [`OperationKind`](../../../../glossary.md#operationkind) directly through inherited `Enum.name()` and adds no
  duplicate name field or method.
- All fifteen kinds have no intrinsic parameters. When wrapped by the later expression task, their
  complete attributes value is <span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter))</span>`NoOperationAttrs.INSTANCE`, never null or an empty map.
- One-input arity is family context and documentation, not a stored arity field. Input identity
  belongs to Tensor provenance and graph nodes.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [identity glossary](../../../../glossary.md#term-identity-meanings); [nodes glossary](../../../../glossary.md#node))</span>
- `FAST_EXP` and `FAST_TANH` are explicit user-visible approximate semantic requests, distinct from
  [`EXP`](../../../../glossary.md#term-expx) and [`TANH`](../../../../glossary.md#term-tanh). This task does not choose an approximation algorithm or accuracy contract.
- Shape, data type, domain, gradient eligibility, differentiation, and numerical edge semantics are
  not enum fields. The public expression, compiler, autograd, and backend-conformance tasks own
  their applicable rules.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [data type glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Package direction is `model.operation.elementwise.unary -> model.operation`; it must not depend
  on Tensor, graph, compiler, planning, runtime, prepare, backend, storage, or training packages.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare))</span>
- Stable enum names are typed diagnostic vocabulary, not wire tokens, registry keys, reflection
  names, or string-dispatch contracts.<span data-glossary-navigation="true"> ([tokens glossary](../../../../glossary.md#term-token); [registry glossary](../../../../glossary.md#term-registry); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Stop if implementation needs another production type, attributes, Tensor behavior, inference,
  numerical policy, backend metadata, dependency, or architecture change.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [metadata glossary](../../../../glossary.md#term-metadata); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing package used:

- `io.github.pho001.synaptik.model.operation` — supplies [`OperationKind`](../../../../glossary.md#operationkind), [`Operation`](../../../../glossary.md#operation), and
  [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs) for the public contract, focused tests, and explanatory composition.

Package added:

```text
io.github.pho001.synaptik.model.operation.elementwise.unary
  Typed parameterless semantic kinds for unary arithmetic, transcendental functions,
  activations, and explicit fast-approximation requests.
```

One family is cohesive because all fifteen capabilities consume one logical [Tensor](../../../../glossary.md#tensor) input, [carry](../../../../glossary.md#term-carry) no
intrinsic attributes, preserve [elementwise](../../../../glossary.md#term-elementwise) position, and share the same future public-expression
construction boundary. Separate enums for arithmetic, transcendental, and [activation](../../../../glossary.md#term-activation) labels would
add type fragmentation without a different [arity](../../../../glossary.md#term-arity), attribute, [validation](../../../../glossary.md#term-validation), or [ownership](../../../../glossary.md#term-ownership-meanings) contract.

Type placement:

- `io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind` — public
  family enum under the owning operation hierarchy.
- `UnaryElementwiseKindTest` — same-package focused test for exact vocabulary and contract [shape](../../../../glossary.md#shape).

## Required contract

### Enum declaration

Create exactly:

```java
public enum UnaryElementwiseKind implements OperationKind {
    ABS,
    NEG,
    INV,
    LOG,
    EXP,
    ERF,
    SQRT,
    FLOOR,
    CEIL,
    SIGN,
    RELU,
    SIGMOID,
    TANH,
    FAST_EXP,
    FAST_TANH
}
```

The enum declares no fields, explicit constructor, methods, nested types, per-[constant](../../../../glossary.md#term-constant) class body,
symbols, [aliases](../../../../glossary.md#aliasing-and-arena), category values, or [metadata](../../../../glossary.md#term-metadata). [Compiler](../../../../glossary.md#term-compiler)-generated enum machinery is not an
additional project [API](../../../../glossary.md#term-application-programming-interface-api). `Enum.name()` satisfies `OperationKind.name()` and returns the exact
constant spelling.

The semantic [identities](../../../../glossary.md#term-identity-meanings) are:

| Kind | [Elementwise](../../../../glossary.md#term-elementwise) meaning |
|---|---|
| `ABS` | absolute magnitude of the input value |
| `NEG` | additive inverse of the input value |
| `INV` | multiplicative reciprocal of the input value |
| [`LOG`](../../../../glossary.md#term-log) | [natural logarithm](../../../../glossary.md#norms-elementary-functions-and-probability-distributions) of the input value |
| [`EXP`](../../../../glossary.md#term-expx) | [natural exponential](../../../../glossary.md#norms-elementary-functions-and-probability-distributions) of the input value |
| [`ERF`](../../../../glossary.md#term-erf) | [Gaussian error function](../../../../glossary.md#norms-elementary-functions-and-probability-distributions) of the input value |
| `SQRT` | principal [square root](../../../../glossary.md#norms-elementary-functions-and-probability-distributions) of the input value |
| `FLOOR` | greatest integer-valued result not greater than the input value |
| `CEIL` | least integer-valued result not less than the input value |
| `SIGN` | negative, zero, or positive sign classification represented numerically |
| `RELU` | [rectified linear unit](../../../../glossary.md#bias-and-activation) of the input value |
| [`SIGMOID`](../../../../glossary.md#term-sigmoid) | logistic sigmoid of the input value |
| [`TANH`](../../../../glossary.md#term-tanh) | [hyperbolic tangent](../../../../glossary.md#bias-and-activation) of the input value |
| `FAST_EXP` | explicitly approximate natural exponential request |
| `FAST_TANH` | explicitly approximate hyperbolic tangent request |

The table defines semantic [identity](../../../../glossary.md#term-identity-meanings) only. It does not define accepted data types, output
[descriptor](../../../../glossary.md#term-descriptor), exact special-value behavior, differentiation, fast-approximation accuracy, execution,
or [backend](../../../../glossary.md#backend) availability.

### Parameterless [Operation](../../../../glossary.md#operation) composition

Every kind is intended to compose explicitly as:

```java
Operation operation = new Operation(
        UnaryElementwiseKind.EXP,
        NoOperationAttrs.INSTANCE);
```

Do not add an enum factory, `operation()` method, attributes type, or compatibility [registry](../../../../glossary.md#term-registry).
Existing `Operation` remains an open generic pairing and does not itself prevent a caller from
supplying another [`OperationAttrs`](../../../../glossary.md#operationattrs) implementation. This task documents and tests correct [canonical](../../../../glossary.md#term-canonical-form)
composition without changing that foundation.

### Typed [identity](../../../../glossary.md#term-identity-meanings) and naming

- `values()` returns exactly the fifteen [constants](../../../../glossary.md#term-constant) in declaration order.
- `name()` returns the exact constant spelling.
- Standard Java enum identity, equality, hashing, `valueOf`, and inherited `toString` remain.
- An equal diagnostic name in another [OperationKind](../../../../glossary.md#operationkind) family does not imply equality.
- [`EXP`](../../../../glossary.md#term-expx) and `FAST_EXP` are different semantic identities, as are [`TANH`](../../../../glossary.md#term-tanh) and `FAST_TANH`; neither
  pair is an alias.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Names and text are diagnostics only, not external [serialization](../../../../glossary.md#term-serialization) or [dispatch](../../../../glossary.md#term-dispatch) contracts.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/unary/UnaryElementwiseKind.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/unary/UnaryElementwiseKindTest.java`

Documentation/[planning](../../../../glossary.md#term-planning) during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `docs/planning/modules/model/capabilities.md` — already selects all fifteen public capabilities.
- Existing OperationKind, OperationAttrs, NoOperationAttrs, Operation, BinaryArithmeticKind,
  Tensor, TensorProvenance, TensorDescriptor, DataType, and Shape Javadocs/tests.<span data-glossary-navigation="true"> ([OperationKind glossary](../../../../glossary.md#operationkind); [OperationAttrs glossary](../../../../glossary.md#operationattrs); [NoOperationAttrs glossary](../../../../glossary.md#nooperationattrs); [Operation glossary](../../../../glossary.md#operation); [Tensor glossary](../../../../glossary.md#tensor); [DataType glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape))</span>
- Compile API, Training API, focused architecture documentation, ADRs, architecture tests,
  backend-conformance material, and integration tests.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most one new production file, one new focused test, and five documentation/[planning](../../../../glossary.md#term-planning) files:
seven paths total.

Do not modify any existing Java source/test, capabilities, completed task, [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api), Training
API, [Gradle](../../../../glossary.md#term-gradle), AGENTS, architecture document/test, another [module](../../../../glossary.md#term-module), or unrelated documentation. Do
not create task 0014D. Stop beyond seven paths or if another production concept, attributes,
[Tensor](../../../../glossary.md#tensor)/public expression behavior, numerical policy, [dependency](../../../../glossary.md#term-dependency), or architecture decision is
required.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Type Javadoc must define the family as backend-independent parameterless unary elementwise
  semantic identities and distinguish it from input provenance, result inference, execution,
  gradients, approximation algorithms, and backend support.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [unary glossary](../../../../glossary.md#term-unary); [elementwise glossary](../../../../glossary.md#term-elementwise); [provenance glossary](../../../../glossary.md#provenance); [inference glossary](../../../../glossary.md#term-inference); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Explain why every kind uses `NoOperationAttrs.INSTANCE` and why one-input arity is not stored as
  metadata.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Explain typed equality and diagnostic-name limits without promising [serialization](../../../../glossary.md#term-serialization) or [dispatch](../../../../glossary.md#term-dispatch).
- Explain that `FAST_EXP` and `FAST_TANH` are explicit distinct approximate requests while accuracy
  and implementation remain deferred.
- Every enum constant must have detailed Javadoc explaining its mathematical or activation meaning
  and explicitly deferring type/domain/special-value/gradient/execution rules where applicable.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [activation glossary](../../../../glossary.md#term-activation))</span>
- Review existing Operation foundation, binary-family, and provenance Javadocs and record why they
  remain accurate or stop on an out-of-scope discrepancy.<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- Exactly one public `UnaryElementwiseKind` enum is added in the planned package.
- It implements OperationKind and declares exactly fifteen constants in the specified order and
  spelling.<span data-glossary-navigation="true"> ([OperationKind glossary](../../../../glossary.md#operationkind))</span>
- It adds no project-declared field, method, constructor, nested type, constant body, alias,
  category, arity, cost, fusion, result, differentiability, approximation, backend, inference, or
  execution metadata.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [arity glossary](../../../../glossary.md#term-arity); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [inference glossary](../../../../glossary.md#term-inference); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Inherited `name()` produces exact diagnostic text and standard enum equality/hash/toString
  behavior remains.<span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash))</span>
- Every [constant](../../../../glossary.md#term-constant) constructs a valid [Operation](../../../../glossary.md#operation) with exact kind [identity](../../../../glossary.md#term-identity-meanings) and
  `NoOperationAttrs.INSTANCE`; no family attributes or factory is added.
- Tests prove that strict and fast pairs remain distinct and that equal names from a private
  test-local OperationKind remain typed and unequal.
- Production imports only OperationKind; no Tensor/graph/compiler/planning/runtime/prepare/backend/
  storage/training dependency appears.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- No Tensor method, descriptor, provenance, storage, shape/data-type, numerical, gradient,
  compiler, backend, or execution behavior changes.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Focused and aggregate model tests, Javadoc, root tests, reflection/javap/import/scope checks,
  documentation links/formatting, and status synchronization pass.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- A separate documentation-focused agent finalizes Javadoc, Tensor API, glossary, task evidence,
  master plan, and roadmap in the same change and records reasoned no-change conclusions for
  related APIs, capabilities, and architecture.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Task 0014C becomes Complete only after both passes. Task 0014D remains Draft without a detailed
  specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKindTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test covers exact declaration/order/names, [OperationKind](../../../../glossary.md#operationkind) membership, inherited name
and toString, enum [identity](../../../../glossary.md#term-identity-meanings)/equality/[hash](../../../../glossary.md#term-hash)/valueOf, strict-versus-fast distinctness, no project
instance state or behavior, parameterless [Operation](../../../../glossary.md#operation) composition for every kind, exact reference
retention, and typed separation from a private test enum with an equal diagnostic name.

Manually inspect `javap -p -c -s` and reflection for exact enum [shape](../../../../glossary.md#shape), fifteen [constants](../../../../glossary.md#term-constant), only
[compiler](../../../../glossary.md#term-compiler)-generated static enum machinery, no extra state/[API](../../../../glossary.md#term-application-programming-interface-api), inherited names, and Operation
composition. Scan production imports/dependencies for forbidden layers. Validate generated
[Javadoc](../../../../glossary.md#term-javadoc), current-versus-planned [Tensor](../../../../glossary.md#tensor) API wording, glossary status, local links/anchors/fences/
whitespace, exact seven-path [scope](../../../../glossary.md#term-scope), synchronized status, [checkpoint](../../../../glossary.md#term-checkpoint) decision, and absence of a
task-0014D specification.

## Dependencies

- Task 0005 supplies [OperationKind](../../../../glossary.md#operationkind) and [NoOperationAttrs](../../../../glossary.md#nooperationattrs).INSTANCE.
- Task 0006 supplies the generic [immutable](../../../../glossary.md#term-immutable) [Operation](../../../../glossary.md#operation) [descriptor](../../../../glossary.md#term-descriptor).
- Tasks 0014A–0014B establish the operation-family conventions and complete the checkpoint that
  selected continued sequential model work; their Java contracts are not modified here.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [model glossary](../../../../glossary.md#term-model))</span>

## Follow-up tasks

- Task 0014D will own public Tensor unary/activation expression methods, floating validation,
  shape/data-type preservation, unresolved result descriptors, gradient-eligibility policy, exact
  Operation construction, single-input provenance, and <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [unary glossary](../../../../glossary.md#term-unary); [activation glossary](../../../../glossary.md#term-activation); [validation glossary](../../../../glossary.md#term-validation); [shape glossary](../../../../glossary.md#shape); [descriptors glossary](../../../../glossary.md#term-descriptor); [gradient glossary](../../../../glossary.md#term-gradient); [Operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance))</span>`TensorFactory.createDerived` use.
- Tasks 0014E–0014F remain Draft for [scalar](../../../../glossary.md#term-scalar) arithmetic and clamp attributes/expressions.
- Compiler capture, optimization, autograd expansion, capability analysis, numerical policies,
  backend ownership, kernels, and conformance remain in their owning module tasks.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [autograd glossary](../../../../glossary.md#autograd); [backend ownership glossary](../../../../glossary.md#backend-ownership); [kernels glossary](../../../../glossary.md#kernel); [conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module))</span>

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns [backend](../../../../glossary.md#backend)-independent [Operation](../../../../glossary.md#operation) semantics to
`modules/model` and excludes backend support, execution, and [graph](../../../../glossary.md#graph) [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names). The [checkpoint](../../../../glossary.md#term-checkpoint)
changes implementation order only and the enum implements the existing [model](../../../../glossary.md#term-model) boundary without a
[module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency) change.

If implementation requires [Tensor](../../../../glossary.md#tensor)/public-expression behavior, [inference](../../../../glossary.md#term-inference), numerical policy,
[mutable](../../../../glossary.md#term-mutable) state, backend [metadata](../../../../glossary.md#term-metadata), another module/package direction, or an architecture change, stop
and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0005/0006/0013/0014A/0014B/0014C, Tensor API, Compile API,
Training API, glossary, current OperationKind/OperationAttrs/NoOperationAttrs/Operation/
BinaryArithmeticKind/TensorProvenance contracts and tests, and Java 26 Gradle configuration.

Implement task 0014C exactly. Add only UnaryElementwiseKind.java and
UnaryElementwiseKindTest.java under io.github.pho001.synaptik.model.operation.elementwise.unary for
Java code/tests. The public enum implements OperationKind and contains exactly ABS, NEG, INV, LOG,
EXP, ERF, SQRT, FLOOR, CEIL, SIGN, RELU, SIGMOID, TANH, FAST_EXP, FAST_TANH in that order, with no
project fields/methods/nested types/metadata. Every kind is parameterless and composes explicitly
with Operation plus NoOperationAttrs.INSTANCE.

Do not add Tensor methods, operation factories/attrs, dtype/shape/layout/result rules, provenance,
numeric or approximation policies, gradients, graph/compiler/planning/runtime/backend behavior,
legacy traits, dependencies/build/architecture changes, existing Java edits, or later specs. Stop
beyond seven paths or on architecture doubt.

Run every specified focused/aggregate test, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff/evidence to a separate
clean-context documentation agent in the same change. It must inspect source/tests/generated
Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record related-contract/
capability/Compile API/Training API/architecture no-change conclusions, and rerun validation.

Update task 0014C, model master plan, and roadmap only for planning status/evidence. Do not mark
0014C Complete until both passes succeed. Leave 0014D Draft without a specification. Do not commit
or push.
```

## Local decisions

- One `UnaryElementwiseKind` family contains arithmetic, transcendental, activation, and explicit
  fast variants because all share one input, no attributes, elementwise position, and the same
  future Tensor-expression validation. Multiple enums would add type fragmentation without a
  distinct contract boundary.<span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation); [elementwise glossary](../../../../glossary.md#term-elementwise); [Tensor glossary](../../../../glossary.md#tensor); [validation glossary](../../../../glossary.md#term-validation))</span>
- The fifteen constants and their order follow the selected public capability baseline rather than
  the legacy class hierarchy.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant))</span>
- `FAST_EXP` and `FAST_TANH` remain distinct public semantic requests, not aliases or backend flags.
  Their exact numerical contracts require later backend/conformance planning.<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [planning glossary](../../../../glossary.md#term-planning))</span>
- Scalar multiplication, scalar power, and clamp stay outside this enum because their scalar values
  are intrinsic immutable attributes owned by tasks 0014E–0014F.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- No enum factory is added. Explicit Operation construction keeps the generic semantic descriptor
  visible and matches the completed binary-family convention.<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- The post-0014B checkpoint preserves the ordered model frontier because downstream prerequisite
  modules remain placeholders; it does not change architecture dependencies.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [model glossary](../../../../glossary.md#term-model); [modules glossary](../../../../glossary.md#term-module))</span>

## Known limitations

- The enum alone cannot build a Tensor expression, validate input data type, preserve a descriptor,
  create provenance, compile a graph, execute mathematics, or report backend support.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [data type glossary](../../../../glossary.md#data-type--datatype); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [compile glossary](../../../../glossary.md#compile); [graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend))</span>
- Numerical domains and special values are not defined here.
- Fast variants have no algorithm or accuracy guarantee until an owning numerical/conformance task
  defines one.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- Parameterless compatibility is documented and tested by correct construction but is not enforced
  against arbitrary callers by the intentionally open Operation record.<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation))</span>
- No [serialization](../../../../glossary.md#term-serialization) or external stable wire-name contract is provided.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed the architecture, documentation and planning rules, [model](../../../../glossary.md#term-model) master plan and
roadmap, completed [operation](../../../../glossary.md#operation) foundation and [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)-family tasks, current Operation contracts and
tests, downstream [module](../../../../glossary.md#term-module) master plans/placeholders, the model capability baseline, and the
read-only legacy [unary](../../../../glossary.md#term-unary) operation [descriptors](../../../../glossary.md#term-descriptor), [Tensor](../../../../glossary.md#tensor) methods, builders, tests, and [backend](../../../../glossary.md#backend)
coverage evidence.

The post-0014B [checkpoint](../../../../glossary.md#term-checkpoint) found that a cross-module next task would require an unplanned chain of
[trace](../../../../glossary.md#trace), backend-contract, config, planning, and [compiler](../../../../glossary.md#term-compiler) foundations. It therefore selected the
next bounded task in the existing model queue. Legacy capability review confirmed fifteen
parameterless public unary/[activation](../../../../glossary.md#term-activation) meanings and rejected legacy cost, [fusion](../../../../glossary.md#term-fusion), expression,
[gradient](../../../../glossary.md#term-gradient), storage, [lowering](../../../../glossary.md#lowering), and execution coupling. The planned package direction remains acyclic
and requires no architecture or [dependency](../../../../glossary.md#term-dependency) change.

Planning validation:

- `git diff --check` passed for the tracked master-plan and roadmap changes; a trailing-whitespace
  scan also passed for this new task file.
- The [canonical](../../../../glossary.md#term-canonical-form) task-section scan found every required planning section.
- The local Markdown-target check resolved every link in this task, the model master plan, and the
  roadmap.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Scope inspection found exactly three documentation/planning paths, no Java, Gradle, AGENTS,
  architecture, or dependency change, and no task-0014D specification.<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Task status is [`Ready`](../../../../glossary.md#term-ready) in this specification, the model master plan, and the roadmap; task 0014D
  remains `Draft`.

Implementation and documentation validation:

- Implementation context `/root/implement_model_0014c` added only
  `UnaryElementwiseKind.java` and `UnaryElementwiseKindTest.java`. Independent clean-context
  documentation context `/root/implement_model_0014b/review_model_0014b_docs` then reread the
  governing architecture, documentation and planning rules, predecessor tasks, final source and
  tests, related operation/provenance contracts, generated Javadoc, XML reports, API references,
  glossary, and complete workspace diff. It applied General, API/Javadoc, and Planning style;
  Example format was reviewed, but no executable example changed because this task adds semantic
  vocabulary rather than a callable Tensor expression.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [Javadoc glossary](../../../../glossary.md#term-javadoc); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKindTest` passed;
  its XML report contains 6 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` passed; 37 XML suites contain 279 tests with zero failures,
  errors, or skips.
- `./gradlew :modules:model:javadoc` passed. Generated `UnaryElementwiseKind.html` contains the
  type contract and all fifteen constant details, including parameterless composition, arity as
  family context, strict/fast distinctions, diagnostic-name limits, and deferred inference,
  numerical, gradient, execution, and backend behavior.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [arity glossary](../../../../glossary.md#term-arity); [inference glossary](../../../../glossary.md#term-inference))</span>
- `./gradlew test` passed for the repository with no failing task.
- `javap -p -c -s` confirmed the fifteen public constants in exact order, only compiler-generated
  enum fields, methods, constructor, and initialization, and no project state or behavior.
  Focused reflection tests independently confirm no instance fields, instance methods, nested
  types, or constant-specific bodies.
- Production import inspection found only [`OperationKind`](../../../../glossary.md#operationkind). The focused tests prove inherited
  enum naming and identity, strict/fast distinction, explicit<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
  `new Operation(kind, NoOperationAttrs.INSTANCE)` composition for every value, exact reference
  retention, and typed separation from an unrelated equally named kind.
- The Tensor API now lists both implemented production kind families, documents all fifteen unary
  meanings and explicit parameterless composition, and keeps unary Tensor expression methods,
  result inference, provenance construction, numerical policy, gradients, execution, and backend
  support planned. The glossary updates the implementation-status convention, `OperationKind`,
  and the operation-kind/attributes/operation distinction without adding a new domain term.
- The enum type and constant Javadocs already met the API profile, so the documentation pass made
  no Java edit. Existing `OperationKind`, [`OperationAttrs`](../../../../glossary.md#operationattrs), [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), `Operation`,
  `BinaryArithmeticKind`, and `TensorProvenance` Javadocs/tests remain accurate because the new
  enum implements and composes their contracts without changing validation, ownership, equality,
  parameter, provenance, or occurrence semantics. Tensor, TensorDescriptor, DataType, and Shape
  remain unchanged because this task adds no public expression, descriptor, type, or shape rule.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [parameter glossary](../../../../glossary.md#term-parameter); [occurrence glossary](../../../../glossary.md#term-occurrence); [DataType glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape))</span>
- `capabilities.md` remains unchanged because it already selects all fifteen capabilities and
  separates model vocabulary from public expression and executable support. Compile API remains
  unchanged because no Tensor expression, capture entry point, inference, graph conversion, or
  artifact changed. Training API remains unchanged because no gradient rule, autograd, optimizer,
  or training behavior changed.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [artifact glossary](../../../../glossary.md#term-artifact); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- `ARCHITECTURE.md`, focused architecture documents, ADRs, architecture tests,
  backend-conformance/integration material and tests, Gradle, and dependencies remain unchanged
  because the enum stays inside existing model ownership and changes no module boundary,
  dependency direction, lifecycle, numerical execution, backend behavior, or build contract.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Final documentation checks resolved all local Markdown file and heading links in the five
  changed documentation/planning files, found balanced fences and no trailing whitespace, and
  `git diff --check` passed. Final scope inspection found exactly the seven authorized paths: one
  production enum, one focused test, Tensor API, glossary, this task, model master plan, and
  roadmap. Task 0014C is Complete in all three planning locations; task 0014D remains Draft and no
  task-0014D specification exists.

## Implementation notes

- Added the exact public `UnaryElementwiseKind` enum with fifteen constants in the specified order
  and no project-declared fields, methods, constructor, nested types, constant bodies, aliases,
  categories, metadata, attributes, or factory.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Added one six-test focused suite for vocabulary, inherited [`OperationKind`](../../../../glossary.md#operationkind) behavior, standard
  enum identity, strict/fast distinction, absence of extra API/state, canonical parameterless
  Operation composition, and typed same-name separation.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [canonical glossary](../../../../glossary.md#term-canonical-form); [Operation glossary](../../../../glossary.md#operation))</span>
- Finalized Tensor API, glossary, and planning status after independent documentation review. No
  executable Java logic or tests changed during the documentation pass.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: Implemented and documented fifteen typed, parameterless unary elementwise
  semantic identities without adding Tensor expression or executable behavior.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [elementwise glossary](../../../../glossary.md#term-elementwise); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Files changed or created: Exactly the production enum, focused test, Tensor API, glossary, this
  task, model master plan, and roadmap authorized by the task.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused 6/6, all 279 model tests across 37 suites, model Javadoc, root
  tests, bytecode/reflection/import/API checks, generated-documentation review, Markdown links and
  anchors, fence/whitespace checks, exact scope/status checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0014b/review_model_0014b_docs` completed the independent pass using
  General, API/Javadoc, and Planning profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and glossary now describe unary semantic vocabulary as current
  while retaining public unary expressions and every numerical, gradient, compiler, backend, and
  execution layer as planned. Capabilities, Compile API, Training API, architecture/ADRs/tests,
  backend-conformance/integration material/tests, and build structure require no change for the
  reasons recorded above.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [Compile glossary](../../../../glossary.md#compile); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Javadoc review: The enum type and all fifteen constants are final and required no correction;
  adjacent operation, binary-family, provenance, Tensor, descriptor, data-type, and shape
  contracts remain accurate without edits.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [operation glossary](../../../../glossary.md#operation); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [provenance glossary](../../../../glossary.md#provenance); [descriptor glossary](../../../../glossary.md#term-descriptor); [shape glossary](../../../../glossary.md#shape))</span>
- Glossary impact: Existing implementation-status and operation-kind distinctions were updated;
  no new reusable project term was introduced.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0014C. Task 0014D remains Draft without a detailed
  specification.

Status: Complete
