# Task 0014A: [Binary](../../../../glossary.md#term-binary-descriptor-and-metadata) Arithmetic Semantic Kinds

## Status

Complete

## Goal

Introduce the first production [operation](../../../../glossary.md#operation)-family vocabulary: seven typed, [backend](../../../../glossary.md#backend)-independent,
parameterless kinds for [tensor](../../../../glossary.md#tensor)-to-tensor [elementwise](../../../../glossary.md#term-elementwise) arithmetic. This gives future public Tensor
expression construction and [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings) stable semantic [identities](../../../../glossary.md#term-identity-meanings) without adding Tensor
methods, [inference](../../../../glossary.md#term-inference), execution, backend [metadata](../../../../glossary.md#term-metadata), or legacy [planning](../../../../glossary.md#term-planning) traits.

The task is deliberately limited to the semantic-kind layer. A following task will use these kinds
with `Operation`, `NoOperationAttrs.INSTANCE`, Tensor [provenance](../../../../glossary.md#provenance), local [broadcasting](../../../../glossary.md#broadcasting), and result-
[descriptor](../../../../glossary.md#term-descriptor) rules.

## [Scope](../../../../glossary.md#term-scope)

- Add one public enum `BinaryArithmeticKind` implementing [`OperationKind`](../../../../glossary.md#operationkind).
- Define exactly `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, and `POW` in that order.
- Document the operand order and mathematical meaning of every kind.
- Establish that all seven kinds are parameterless and are paired with
  `NoOperationAttrs.INSTANCE` when an [`Operation`](../../../../glossary.md#operation) is constructed.
- Add one focused test proving exact enum vocabulary, typed OperationKind behavior, stable names,
  parameterless Operation composition, and absence of extra family state/behavior.
- Introduce the cohesive `model.operation.elementwise.binary` package in the [model](../../../../glossary.md#term-model) package map.
- Finalize affected Javadocs, Tensor API, glossary, task evidence, master plan, and roadmap through
  the required independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public `Tensor.add`, `sub`, `mul`, `div`, `min`, `max`, or [tensor](../../../../glossary.md#tensor) `pow` methods
- Tensor result creation, `TensorFactory.createDerived`, provenance, labels, host storage, IDs,
  descriptor construction, requires-grad propagation, or expression chaining<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [host storage glossary](../../../../glossary.md#host-storage); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- shape broadcasting, broadcast plans/attributes/strides, output shape inference, dynamic-symbol
  constraints, or graph-wide validation<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [broadcasting glossary](../../../../glossary.md#broadcasting); [strides glossary](../../../../glossary.md#term-stride); [inference glossary](../../../../glossary.md#term-inference); [symbol glossary](../../../../glossary.md#term-symbol); [graph glossary](../../../../glossary.md#graph); [validation glossary](../../../../glossary.md#term-validation))</span>
- data-type eligibility, numeric promotion, casting, output data type, integer division policy,
  BOOL rejection, floating NaN/signed-zero/tie behavior, or numerical execution<span data-glossary-navigation="true"> ([numeric promotion glossary](../../../../glossary.md#numeric-promotion); [data type glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- an operation-family attribute type, factory, wrapper, registry, parser, alias, symbol table,
  string dispatch, classpath/reflection discovery, or serialization token<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [registry glossary](../../../../glossary.md#term-registry); [alias glossary](../../../../glossary.md#aliasing-and-arena); [dispatch glossary](../../../../glossary.md#term-dispatch); [classpath glossary](../../../../glossary.md#term-classpath); [serialization glossary](../../../../glossary.md#term-serialization); [token glossary](../../../../glossary.md#term-token))</span>
- arity classes, semantic-family tags, computational cost, result-kind tags, fusion flags,
  differentiability metadata, gradient rules, backend support, capability matrices, kernel routes,
  lowering, or execution<span data-glossary-navigation="true"> ([matrices glossary](../../../../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [fusion glossary](../../../../glossary.md#term-fusion); [metadata glossary](../../../../glossary.md#term-metadata); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend); [kernel glossary](../../../../glossary.md#kernel); [lowering glossary](../../../../glossary.md#lowering))</span>
- unary/scalar arithmetic, activations, clamp, comparisons, logical operations, selection, cast,
  reduction, layout, indexing, linear algebra, normalization, loss, or compiler-generated kinds<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [scalar glossary](../../../../glossary.md#term-scalar); [activations glossary](../../../../glossary.md#term-activation); [layout glossary](../../../../glossary.md#layout); [linear glossary](../../../../glossary.md#term-linear-meanings); [normalization glossary](../../../../glossary.md#term-normalization); [loss glossary](../../../../glossary.md#loss); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- modifying existing Operation/OperationKind/OperationAttrs/NoOperationAttrs contracts or tests,
  another production type, dependency, Gradle, architecture, another module, or task-0014B spec<span data-glossary-navigation="true"> ([OperationKind glossary](../../../../glossary.md#operationkind); [OperationAttrs glossary](../../../../glossary.md#operationattrs); [NoOperationAttrs glossary](../../../../glossary.md#nooperationattrs); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

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
- [Task 0013A](0013a-full-value-and-identity-matrix-tensor-creation.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Glossary](../../../../glossary.md)

## Foundation [checkpoint](../../../../glossary.md#term-checkpoint) decision

The post-task-0013A checkpoint reviewed the implemented [data type](../../../../glossary.md#data-type--datatype), [shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout), [descriptor](../../../../glossary.md#term-descriptor),
[identity](../../../../glossary.md#term-identity-meanings), [immutable](../../../../glossary.md#term-immutable) [graph](../../../../glossary.md#graph), [host storage](../../../../glossary.md#host-storage), [Tensor](../../../../glossary.md#tensor), factory, [Operation](../../../../glossary.md#operation) foundation, and [provenance](../../../../glossary.md#provenance)
contracts against downstream entry needs.

The project will continue with sequential [model](../../../../glossary.md#term-model) operation-family work rather than move immediately
to a cross-[module](../../../../glossary.md#term-module) vertical [slice](../../../../glossary.md#slice). The [compiler](../../../../glossary.md#term-compiler) [lifecycle](../../../../glossary.md#lifecycle) has enough structure to consume an
expression later, but no production concrete [`OperationKind`](../../../../glossary.md#operationkind) currently exists, so [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction),
capability queries, [backend ownership](../../../../glossary.md#backend-ownership), [prepare](../../../../glossary.md#prepare), and execution would have no real semantic operation
to process. Implementing the first concrete family, followed by its public expression construction,
creates a meaningful seam for later cross-module work while preserving the ordered capability plan.

The broad former task 0014 is therefore decomposed into small semantic-vocabulary and public-
expression tasks. This specification covers only the first semantic vocabulary.

## Capability origin

The selected legacy public capability includes [tensor](../../../../glossary.md#tensor)-to-tensor `add`, `sub`, `mul`, `div`, `min`,
`max`, and `pow`, including broadcasted operands. Legacy [operation](../../../../glossary.md#operation) classes also carried [mutable](../../../../glossary.md#term-mutable) or
nullable broadcast plans, expression strings, [arity](../../../../glossary.md#term-arity) categories, [fusion](../../../../glossary.md#term-fusion) flags, semantic-family
labels, cost classes, and result-kind [metadata](../../../../glossary.md#term-metadata).

Only the seven mathematical [identities](../../../../glossary.md#term-identity-meanings) and their stable short names are retained here. Broadcast
geometry is derived from input [shapes](../../../../glossary.md#shape), not stored as an operation attribute. Cost, fusion,
capability, [lowering](../../../../glossary.md#lowering), and execution metadata belong to [planning](../../../../glossary.md#term-planning) or [concrete backend](../../../../glossary.md#concrete-backend) preparation,
not the [model](../../../../glossary.md#term-model) kind. No legacy source or class hierarchy is copied.

## Architecture constraints

- Production remains in `modules/model`, which owns [backend](../../../../glossary.md#backend)-independent [operation](../../../../glossary.md#operation) semantics.
- `BinaryArithmeticKind` is a typed semantic discriminator, not a graph node, occurrence ID,
  executable operation, backend capability, or kernel route.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [occurrence glossary](../../../../glossary.md#term-occurrence); [backend capability glossary](../../../../glossary.md#backend-capability); [kernel glossary](../../../../glossary.md#kernel))</span>
- The enum implements the existing open [`OperationKind`](../../../../glossary.md#operationkind) contract directly through inherited
  `Enum.name()`; it adds no duplicate name field or method.
- Every kind has no intrinsic [parameters](../../../../glossary.md#term-parameter). When wrapped later, its complete attributes value is
  `NoOperationAttrs.INSTANCE`, never null or an empty map.
- Input order remains semantically meaningful for `SUB`, `DIV`, and `POW`. This task stores no
  inputs and performs no validation because inputs belong to Tensor provenance and graph nodes.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance))</span>
- Broadcasting is derived state from input shapes. It is not an attribute, plan, stride set, or
  mutable field on the operation kind.<span data-glossary-navigation="true"> ([Broadcasting glossary](../../../../glossary.md#broadcasting); [shapes glossary](../../../../glossary.md#shape); [stride glossary](../../../../glossary.md#term-stride); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Package direction is `model.operation.elementwise.binary -> model.operation`; it must not depend
  on Tensor, graph, compiler, planning, runtime, prepare, backend, storage, or training packages.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare))</span>
- Stable kind names are diagnostic typed vocabulary. They are not serialization tokens, registry
  keys, reflection names, or string-dispatch contracts.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [tokens glossary](../../../../glossary.md#term-token); [registry glossary](../../../../glossary.md#term-registry); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Stop if implementation needs another type, attributes, Tensor/public expression behavior,
  inference, backend metadata, dependency, or architecture change.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [metadata glossary](../../../../glossary.md#term-metadata); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation` — supplies [`OperationKind`](../../../../glossary.md#operationkind), [`Operation`](../../../../glossary.md#operation), and
  [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs) for tests and explanatory composition.

Package added:

```text
io.github.pho001.synaptik.model.operation.elementwise.binary
  Typed parameterless semantic kinds for tensor-to-tensor elementwise arithmetic.
```

This package is cohesive because [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) arithmetic shares two ordered [Tensor](../../../../glossary.md#tensor) operands,
broadcast-aware [elementwise](../../../../glossary.md#term-elementwise) meaning, parameterless [descriptors](../../../../glossary.md#term-descriptor), and the same future public
expression boundary. It does not contain Tensor methods, attributes, [inference](../../../../glossary.md#term-inference), execution, or
[backend](../../../../glossary.md#backend) variants.

Type placement:

- `io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind` — public
  family enum under the owning operation hierarchy.
- `BinaryArithmeticKindTest` — same-package focused test for exact vocabulary and contract [shape](../../../../glossary.md#shape).

## Required contract

### Enum declaration

Create exactly:

```java
public enum BinaryArithmeticKind implements OperationKind {
    ADD,
    SUB,
    MUL,
    DIV,
    MIN,
    MAX,
    POW
}
```

The enum declares no fields, explicit constructor, methods, nested types, per-[constant](../../../../glossary.md#term-constant) class body,
symbols, [aliases](../../../../glossary.md#aliasing-and-arena), or [metadata](../../../../glossary.md#term-metadata). [Compiler](../../../../glossary.md#term-compiler)-generated enum machinery is not an additional project [API](../../../../glossary.md#term-application-programming-interface-api).
`Enum.name()` satisfies `OperationKind.name()` and returns the exact constant spelling.

Constant meanings are:

| Kind | Ordered [elementwise](../../../../glossary.md#term-elementwise) meaning |
|---|---|
| `ADD` | left value plus right value |
| `SUB` | left value minus right value |
| `MUL` | left value multiplied by right value |
| `DIV` | left value divided by right value |
| `MIN` | minimum selected from left and right values |
| `MAX` | maximum selected from left and right values |
| `POW` | left base raised to the right [exponent](../../../../glossary.md#term-exponent) |

The table defines mathematical [identity](../../../../glossary.md#term-identity-meanings) and operand order only. It does not define supported data
types, promotion, [broadcasting](../../../../glossary.md#broadcasting) success, integer/floating edge behavior, differentiation, execution,
or [backend](../../../../glossary.md#backend) availability.

### Parameterless [Operation](../../../../glossary.md#operation) composition

Every kind is intended to compose as:

```java
Operation operation = new Operation(
        BinaryArithmeticKind.ADD,
        NoOperationAttrs.INSTANCE);
```

Do not add a family factory or an `operation()` method to the enum. Explicit construction keeps the
generic Operation contract visible and avoids introducing convenience behavior before the public
expression task owns [validation](../../../../glossary.md#term-validation) and [provenance](../../../../glossary.md#provenance). Existing Operation intentionally does not prevent a
caller from pairing a kind with another [OperationAttrs](../../../../glossary.md#operationattrs) implementation; task 0014A neither changes
that generic openness nor adds reflection/[registry](../../../../glossary.md#term-registry) enforcement.

### Naming and typed [identity](../../../../glossary.md#term-identity-meanings)

- `values()` returns exactly the seven [constants](../../../../glossary.md#term-constant) in the declared order.
- `name()` returns exactly `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, or `POW`.
- Enum identity/equality/[hash](../../../../glossary.md#term-hash) semantics remain standard Java enum semantics.
- A future kind in another enum may have the same name without becoming equal. In particular,
  future scalar power may also use <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>`POW`; the concrete family type keeps the meanings distinct.
- `toString()` remains inherited enum diagnostic text and is not overridden or treated as a wire
  format.<span data-glossary-navigation="true"> ([wire format glossary](../../../../glossary.md#term-wire-format))</span>

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/binary/BinaryArithmeticKind.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/binary/BinaryArithmeticKindTest.java`

Documentation/[planning](../../../../glossary.md#term-planning) during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `docs/planning/modules/model/capabilities.md` — already selects all seven public capabilities.
- Existing OperationKind, OperationAttrs, NoOperationAttrs, Operation, TensorProvenance, Shape,
  ShapeBroadcast, DataType, and DataTypePromotion Javadocs/tests.<span data-glossary-navigation="true"> ([OperationKind glossary](../../../../glossary.md#operationkind); [OperationAttrs glossary](../../../../glossary.md#operationattrs); [NoOperationAttrs glossary](../../../../glossary.md#nooperationattrs); [Operation glossary](../../../../glossary.md#operation); [Shape glossary](../../../../glossary.md#shape); [DataType glossary](../../../../glossary.md#data-type--datatype))</span>
- [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api), focused architecture documentation, ADRs, and architecture tests.

## Maximum [scope](../../../../glossary.md#term-scope)

At most one new production file, one new focused test, and five documentation/[planning](../../../../glossary.md#term-planning) files:
seven paths total.

Do not modify any existing Java source/test, capabilities, completed task, [Gradle](../../../../glossary.md#term-gradle), AGENTS,
architecture document/test, another [module](../../../../glossary.md#term-module), or unrelated documentation. Do not create task 0014B.
Stop beyond seven paths or if another production concept, package, attribute, public expression,
[inference](../../../../glossary.md#term-inference) rule, [dependency](../../../../glossary.md#term-dependency), or architecture decision is required.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Type Javadoc must define the family as backend-independent tensor-to-tensor elementwise
  arithmetic semantic identities and distinguish it from Operation, provenance, graph occurrence,
  inference, execution, and backend support.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [tensor glossary](../../../../glossary.md#tensor); [elementwise glossary](../../../../glossary.md#term-elementwise); [Operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [occurrence glossary](../../../../glossary.md#term-occurrence); [inference glossary](../../../../glossary.md#term-inference))</span>
- Explain why broadcast [metadata](../../../../glossary.md#term-metadata) is not stored and why every kind uses [NoOperationAttrs](../../../../glossary.md#nooperationattrs).INSTANCE.
- Document typed equality and diagnostic-name limits without promising [serialization](../../../../glossary.md#term-serialization) or [dispatch](../../../../glossary.md#term-dispatch).
- Every enum constant must have detailed Javadoc explaining left/right order and mathematical
  meaning, plus explicit deferral of type/shape/numeric/execution semantics where useful.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [shape glossary](../../../../glossary.md#shape))</span>
- Review existing Operation foundation/provenance Javadocs and record why they remain accurate or
  stop on an out-of-scope discrepancy.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- Exactly one public `BinaryArithmeticKind` enum is added in the planned package.
- It implements [OperationKind](../../../../glossary.md#operationkind) and declares exactly seven [constants](../../../../glossary.md#term-constant) in exact order and spelling.
- It adds no project-declared field, method, constructor, nested type, constant body, alias, symbol,
  category, cost, fusion, backend, inference, or execution metadata.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [symbol glossary](../../../../glossary.md#term-symbol); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [inference glossary](../../../../glossary.md#term-inference); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Inherited `name()` produces exact stable diagnostic text and standard enum equality/hash/toString
  behavior remains.<span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash))</span>
- Every constant constructs a valid [Operation](../../../../glossary.md#operation) with exact kind [identity](../../../../glossary.md#term-identity-meanings) and
  `NoOperationAttrs.INSTANCE`; no family-specific attrs or factory is added.
- Tests distinguish the typed family from a private test-local enum with an equal constant name.
- Production imports only OperationKind; no Tensor/graph/compiler/planning/runtime/prepare/backend/
  storage/training dependency appears.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- No Tensor method, provenance, descriptor, broadcast, dtype, numerical, execution, gradient, or
  backend behavior changes.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [descriptor glossary](../../../../glossary.md#term-descriptor); [dtype glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Focused and aggregate model tests, Javadoc, root tests, reflection/javap/import/scope checks,
  documentation links/formatting, and status synchronization pass.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- A separate documentation-focused agent finalizes Javadoc, Tensor API, glossary, task evidence,
  master plan, and roadmap in the same change and records reasoned no-change conclusions for
  related APIs, capabilities, and architecture.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Task 0014A becomes Complete only after both passes. Task 0014B remains Draft without a detailed
  specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKindTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test covers exact declaration/order/names, [OperationKind](../../../../glossary.md#operationkind) membership, inherited name and
toString, enum [identity](../../../../glossary.md#term-identity-meanings)/equality/[hash](../../../../glossary.md#term-hash), no project instance state or extra instance behavior,
parameterless [Operation](../../../../glossary.md#operation) composition for every kind, exact reference retention, and typed
separation from a private test enum with equal `ADD` text.

Manually inspect `javap -p -c -s` and reflection for exact enum [shape](../../../../glossary.md#shape), seven [constants](../../../../glossary.md#term-constant), only
[compiler](../../../../glossary.md#term-compiler)-generated static enum machinery, no extra state/[API](../../../../glossary.md#term-application-programming-interface-api), inherited `name`, and Operation
composition. [Scan](../../../../glossary.md#term-cumulative-scan) imports/dependencies for forbidden layers. Validate generated [Javadoc](../../../../glossary.md#term-javadoc), production
example/current-versus-planned wording, glossary status, local links/anchors/fences/whitespace,
exact seven paths, synchronized status, and absence of a task-0014B specification.

## Dependencies

- Task 0005 supplies [OperationKind](../../../../glossary.md#operationkind) and [NoOperationAttrs](../../../../glossary.md#nooperationattrs).INSTANCE.
- Task 0006 supplies the generic [immutable](../../../../glossary.md#term-immutable) [Operation](../../../../glossary.md#operation) [descriptor](../../../../glossary.md#term-descriptor).
- Task 0013 supplies Tensor provenance for the later expression-construction task but is not used
  by this enum implementation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance))</span>
- The completed [model](../../../../glossary.md#term-model) foundation [checkpoint](../../../../glossary.md#term-checkpoint) selects sequential model family work.

## Follow-up tasks

- Task 0014B will own public Tensor binary arithmetic expression construction, local shape/data-
  type validation, result descriptor creation, exact parameterless Operation construction, ordered
  provenance inputs, and use of TensorFactory.createDerived.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [Operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance))</span>
- Tasks 0014C–0014F remain Draft semantic/expression groups for unary/activation and scalar/clamp
  capabilities.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [activation glossary](../../../../glossary.md#term-activation); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Compiler, planning, backend, runtime, autograd, and numerical conformance remain in their owning
  modules/tasks.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [autograd glossary](../../../../glossary.md#autograd); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module))</span>

Do not create detailed follow-up specifications in this task.

## Architecture impact

Expected impact: None. The architecture already assigns [backend](../../../../glossary.md#backend)-independent [Operation](../../../../glossary.md#operation) semantics to
[modules](../../../../glossary.md#term-module)/[model](../../../../glossary.md#term-model) and excludes backend support, execution, and [graph](../../../../glossary.md#graph) [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names). The new typed enum
implements that existing boundary without a module [dependency](../../../../glossary.md#term-dependency) change.

If implementation requires [Tensor](../../../../glossary.md#tensor)/public-expression behavior, [inference](../../../../glossary.md#term-inference), [mutable](../../../../glossary.md#term-mutable) state, backend
[metadata](../../../../glossary.md#term-metadata), another module/package direction, or a changed architecture rule, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0005/0006/0013/0013A/0014A, Tensor API, glossary, current
OperationKind/OperationAttrs/NoOperationAttrs/Operation/TensorProvenance contracts and tests, and
Java 26 Gradle configuration.

Implement task 0014A exactly. Add only BinaryArithmeticKind.java and BinaryArithmeticKindTest.java
under io.github.pho001.synaptik.model.operation.elementwise.binary for Java code/tests. The public
enum implements OperationKind and contains exactly ADD, SUB, MUL, DIV, MIN, MAX, POW in that order,
with no project fields/methods/nested types/metadata. Every kind is parameterless and composes
explicitly with Operation plus NoOperationAttrs.INSTANCE.

Do not add Tensor methods, operation factory/attrs, broadcasting/inference/dtype rules, provenance,
graph/compiler/planning/runtime/backend behavior, legacy traits, dependencies/build/architecture
changes, existing Java edits, or later specs. Stop beyond seven paths or on architecture doubt.

Run every specified focused/aggregate test, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff/evidence to a separate
clean-context documentation agent in the same change. It must inspect source/tests/generated
Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record related-contract/
capability/architecture no-change conclusions, and rerun validation.

Update task 0014A, model master plan, and roadmap only for planning status/evidence. Do not mark
0014A Complete until both passes succeed. Leave 0014B Draft without a specification. Do not commit
or push.
```

## Local decisions

- Short [constants](../../../../glossary.md#term-constant) `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, and `POW` match the public capability
  vocabulary and selected stable legacy semantic names without copying legacy classes.
- The concrete family type supplies semantic context, so a later [scalar](../../../../glossary.md#term-scalar)-family `POW` can remain a
  distinct typed kind even with equal diagnostic text.
- No BinaryArithmeticAttrs type is added because all seven operations have no intrinsic
  parameters. Broadcast geometry and input descriptors are derived from operands.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- No enum factory method is added. Explicit Operation construction belongs to the following public
  expression task and uses the existing generic descriptor visibly.<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation))</span>
- Semantic vocabulary is separated from public Tensor expression behavior so local inference and
  provenance policy can be specified and reviewed independently.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [inference glossary](../../../../glossary.md#term-inference); [provenance glossary](../../../../glossary.md#provenance))</span>

## Known limitations

- The enum alone cannot build a Tensor expression, validate operands, infer shape/data type, create
  provenance, compile a graph, execute arithmetic, or report backend support.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [shape glossary](../../../../glossary.md#shape); [data type glossary](../../../../glossary.md#data-type--datatype); [provenance glossary](../../../../glossary.md#provenance); [compile glossary](../../../../glossary.md#compile); [graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend))</span>
- Parameterless compatibility is documented and tested by correct construction but is not enforced
  against arbitrary callers by the intentionally open generic Operation record.<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation))</span>
- Numerical edge behavior, supported data types, promotion, broadcasting, gradients, and backend
  conformance remain deferred to owning tasks.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [gradients glossary](../../../../glossary.md#term-gradient); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- No [serialization](../../../../glossary.md#term-serialization) or external stable wire-name contract is provided.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) completed the post-foundation [checkpoint](../../../../glossary.md#term-checkpoint) by reviewing the implemented [model](../../../../glossary.md#term-model) value, [graph](../../../../glossary.md#graph),
storage, [Tensor](../../../../glossary.md#tensor)/factory, [Operation](../../../../glossary.md#operation), and [provenance](../../../../glossary.md#provenance) contracts against the architecture [lifecycle](../../../../glossary.md#lifecycle) and
downstream [module](../../../../glossary.md#term-module) entry conditions. It selected continued sequential model work because no
production concrete [OperationKind](../../../../glossary.md#operationkind) existed yet for a meaningful [compiler](../../../../glossary.md#term-compiler)/[backend](../../../../glossary.md#backend) vertical [slice](../../../../glossary.md#slice).

Planning also inspected the selected legacy [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) operation classes and public tests read-only.
It retained only the seven mathematical [identities](../../../../glossary.md#term-identity-meanings) and stable names, rejecting legacy broadcast-
plan state, expression strings, [arity](../../../../glossary.md#term-arity)/cost/result/[fusion](../../../../glossary.md#term-fusion) [metadata](../../../../glossary.md#term-metadata), backend coupling, and execution
behavior. The focused enum fits one production/test pair plus five documentation/planning paths
with no [dependency](../../../../glossary.md#term-dependency) or architecture change.

- Implementation context `/root/implement_model_0014a` added only
  `BinaryArithmeticKind.java` and `BinaryArithmeticKindTest.java`. Independent clean documentation
  context `/root/implement_model_0014a/review_model_0014a_docs` inspected the architecture,
  planning contracts, final source/test, generated Javadoc, XML reports, bytecode, Java 26 build
  configuration, API reference, glossary, and complete diff. It applied General plus API/Javadoc
  style to the enum, Tensor API, and glossary, and Planning style to this task, the model master
  plan, and roadmap.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKindTest` — passed;
  XML reports 5 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — passed; 35 XML suites report 264 tests with zero failures,
  errors, or skips.
- `./gradlew :modules:model:javadoc` — passed without [Javadoc](../../../../glossary.md#term-javadoc) errors. Generated
  `BinaryArithmeticKind.html` contains the type contract and all seven constant details, including
  ordered operand meaning, parameterless composition, derived broadcast geometry, typed equality,
  diagnostic-name limits, and deferred inference/execution/backend behavior.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [inference glossary](../../../../glossary.md#term-inference))</span>
- `./gradlew test` — passed for the repository with 36 actionable tasks and no failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` confirmed seven public
  constants in exact order, only compiler-generated enum fields/methods/constructor/initializer,
  and no project state or behavior. Reflection tests additionally confirmed no instance fields,
  methods, nested types, or constant-specific bodies; inherited `name`, `toString`, identity, and
  hashing remain standard enum behavior.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Production import inspection found only `OperationKind`, preserving package direction
  `model.operation.elementwise.binary -> model.operation`. Tests prove explicit
  `new Operation(kind, NoOperationAttrs.INSTANCE)` composition for every value, exact reference
  retention, and typed separation from an unrelated private `ADD` enum.
- Tensor [API](../../../../glossary.md#term-application-programming-interface-api) now presents `BinaryArithmeticKind` as the first implemented production family,
  lists the seven ordered mathematical identities, shows explicit parameterless Operation
  composition, and distinguishes semantic vocabulary from Tensor expressions, provenance, graph
  occurrences, inference, compilation, execution, and backend support. Current-versus-planned
  wording now leaves only other kind families, family attributes, and expression/execution layers
  planned.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- Glossary review updated the implementation-status convention, `OperationKind`, and the
  operation-kind/attributes/operation distinction. No new glossary entry was added because
  `BinaryArithmeticKind` is a concrete implementation of the existing OperationKind domain term,
  not a new reusable architecture, lifecycle, or ownership concept; its full vocabulary belongs
  in the Tensor API.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- The submitted `BinaryArithmeticKind` type and constant Javadocs already satisfy the API profile,
  so the documentation context made no Java edit. Existing `OperationKind`, [`OperationAttrs`](../../../../glossary.md#operationattrs),
  [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), `Operation`, and `TensorProvenance` Javadocs/tests remain accurate because
  the new enum implements and composes their existing contracts without changing validation,
  ownership, equality, provenance, or occurrence semantics. <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>[`Shape`](../../../../glossary.md#shape), `ShapeBroadcast`, [`DataType`](../../../../glossary.md#data-type--datatype),
  and `DataTypePromotion` remain accurate because the enum stores and derives no shape, broadcast,
  type, promotion, or numerical rule.
- `docs/planning/modules/model/capabilities.md` remains unchanged because it already selects all
  seven public capabilities and correctly distinguishes model semantic representation from public
  expression and executable support. The Compile API remains unchanged because no capture entry
  point, inference, graph conversion, or compile artifact changed.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [capture glossary](../../../../glossary.md#term-capture-meanings); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- `ARCHITECTURE.md`, focused architecture documents, ADRs, architecture tests,
  backend-conformance/integration documentation and tests, and build configuration remain
  unchanged because the enum stays within model-owned semantics, adds only the already-planned
  inward package dependency, and changes no module boundary, lifecycle, backend behavior,
  end-to-end execution, Java 26 toolchain/release, preview/incubator setting, or dependency.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [toolchain glossary](../../../../glossary.md#term-toolchain); [release glossary](../../../../glossary.md#term-release))</span>
- The corrected local Markdown validator resolved 171 file targets and heading anchors across the
  five changed documentation/planning files with zero errors. A preliminary invocation failed on
  Ruby interpolation in the heading regular expression, and the next correctly exposed that the
  first slug approximation collapsed a double hyphen used by an existing heading anchor; neither
  invocation changed repository files. The corrected GitHub-style check passed. Code fences are
  balanced, no changed path has trailing whitespace, terminology and status are synchronized, and<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
  `git diff --check` passed.
- Final scope review found exactly the seven authorized paths: one production enum, one focused
  test, Tensor API, glossary, this task, model master plan, and roadmap. Task 0014A is Complete in
  all three planning locations. Task 0014B remains Draft and no task-0014B specification exists.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Implementation notes

- Added the exact public `BinaryArithmeticKind` enum with `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`,
  and `POW` in order and no project-declared state, methods, constructor, nested types, constant
  bodies, metadata, attributes, or factory.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Added one five-test reflection and composition suite covering vocabulary, inherited
  [`OperationKind`](../../../../glossary.md#operationkind) behavior, standard enum identity, absence of extra API/state, explicit canonical
  no-attributes composition, and typed same-name separation.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Finalized the Tensor API and glossary for the implemented family and synchronized planning
  status after the independent documentation review and full validation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [planning glossary](../../../../glossary.md#term-planning); [validation glossary](../../../../glossary.md#term-validation))</span>

## Completion summary

- Completed changes: Implemented and documented the first production concrete OperationKind
  family as seven typed parameterless binary arithmetic semantic identities.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([OperationKind glossary](../../../../glossary.md#operationkind); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- Files changed or created: Exactly the one production enum, one focused test, Tensor API,
  glossary, this task, model master plan, and roadmap authorized by the task.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused 5/5, all 264 model tests across 35 suites, model Javadoc, root
  tests, bytecode/reflection/import/API checks, generated-documentation review, Markdown links and
  anchors, fence/terminology/whitespace checks, exact-scope/status checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`
  passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0014a/review_model_0014a_docs` completed the independent pass using
  General, API/Javadoc, and Planning profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and existing glossary/status language now describe the current
  binary arithmetic vocabulary and retain Tensor expressions, inference, compilation, and
  execution as planned work. Capabilities, Compile API, architecture/ADRs/tests,
  backend-conformance/integration documentation/tests, and build structure require no change for
  the reasons recorded above.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [Compile glossary](../../../../glossary.md#compile); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Javadoc review: The enum type and all seven constants are final and required no correction;
  adjacent operation, provenance, shape/broadcast, and data-type/promotion contracts remain
  accurate without edits.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance); [shape glossary](../../../../glossary.md#shape))</span>
- Glossary impact: Existing implementation-status and operation-kind distinctions were updated;
  no new reusable project term was introduced.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0014A. Task 0014B remains the next Draft planning frontier
  without a detailed specification.

Status: Complete
