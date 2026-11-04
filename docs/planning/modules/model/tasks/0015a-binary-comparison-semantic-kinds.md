# Task 0015A: [Binary](../../../../glossary.md#term-binary-descriptor-and-metadata) Comparison Semantic Kinds

**Editorial reading key (later context, not a status update).**

An elementwise comparison produces a truth value for each corresponding pair, not a numerical minimum or an ordering of the whole tensor. The six kinds preserve left/right roles; BOOL result descriptors and shape reasoning belong to the expression task rather than the enum. Here Model defines a typed meaning for later construction code; this semantic-family task neither receives stored operands nor evaluates them. Its completion is not evidence of Compiler capture, derivatives or backend execution.

Focused current reading: [Current comparison reference](../../../../reference/tensor/arithmetic.md#binary-comparison-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Define the typed, [backend](../../../../glossary.md#backend)-independent, parameterless semantic vocabulary for the six selected
[Tensor](../../../../glossary.md#tensor)-to-Tensor [elementwise](../../../../glossary.md#term-elementwise) comparisons. The vocabulary identifies mathematical comparison
meaning and ordered operand roles without storing operands, broadcast geometry, [BOOL](../../../../glossary.md#data-type--datatype) result facts,
execution [metadata](../../../../glossary.md#term-metadata), or backend support.

This task creates only the semantic family consumed by task 0015B. It does not add public Tensor
comparison methods or derive result [descriptors](../../../../glossary.md#term-descriptor).

## [Scope](../../../../glossary.md#term-scope)

- Add one public enum `BinaryComparisonKind` implementing [`OperationKind`](../../../../glossary.md#operationkind).
- Define exactly `GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`, and
  `NOT_EQUAL` in that order.
- Document the ordered left/right mathematical meaning of every kind.
- Establish that every kind is parameterless and composes explicitly with
  `NoOperationAttrs.INSTANCE` when represented by [`Operation`](../../../../glossary.md#operation).
- Add one focused same-package test proving exact vocabulary, typed identity, enum behavior,
  parameterless Operation composition, and absence of metadata/state.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Finalize affected Javadocs, Tensor API, glossary, task evidence, model master plan, and roadmap
  through the required independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public `Tensor.greaterThan`, `greaterOrEqual`, `lessThan`, `lessOrEqual`, `equalTo`, or
  `notEqualTo` methods
- operands, Tensor provenance, IDs, graph values/nodes, result construction, labels, storage,
  expression chaining, or <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance); [nodes glossary](../../../../glossary.md#node))</span>`TensorFactory.createDerived`
- input data-type eligibility, promotion, cross-type comparison, BOOL result type, broadcasting,
  layout, <span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [broadcasting glossary](../../../../glossary.md#broadcasting); [layout glossary](../../../../glossary.md#layout))</span>`requiresGrad`, or local/[graph](../../../../glossary.md#graph)-wide [inference](../../../../glossary.md#term-inference)
- scalar comparison, tolerance/epsilon, approximate equality, total ordering, or three-way
  comparison<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [tolerance glossary](../../../../glossary.md#term-tolerance); [epsilon glossary](../../../../glossary.md#epsilon))</span>
- [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), equality, ordering, or other numerical edge policy
- logical AND/OR/NOT, `where`, cast, reduction, selection, or another [operation](../../../../glossary.md#operation) family
- family attributes, factory, registry, parser, aliases, symbols, string dispatch, reflection
  discovery, visitor, service, map, or compatibility validator<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- short [aliases](../../../../glossary.md#aliasing-and-arena) such as `GT`, `GE`, `LT`, `LE`, `EQ`, or `NE`
- fields for arity, category, cost, fusion, result kind, differentiability, backend support,
  lowering, route, or kernel metadata<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- [mutable](../../../../glossary.md#term-mutable)/[immutable](../../../../glossary.md#term-immutable) broadcast plans, [strides](../../../../glossary.md#term-stride), output [shapes](../../../../glossary.md#shape), or [materialization](../../../../glossary.md#materialization) requirements
- [gradients](../../../../glossary.md#term-gradient), [backward graph](../../../../glossary.md#backward-graph) construction, [autograd](../../../../glossary.md#autograd), [optimizer](../../../../glossary.md#term-optimizer), or training behavior
- compiler, planning, prepare, runtime, backend, engine, tracing, dependency, Gradle, architecture,
  or another-module changes<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- changes to existing Java contracts/tests or a detailed task-0015B specification

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
- [Task 0014A](0014a-binary-arithmetic-semantic-kinds.md)
- [Task 0014C](0014c-unary-elementwise-semantic-kinds.md)
- [Task 0014E](0014e-scalar-arithmetic-and-clamp-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes public [Tensor](../../../../glossary.md#tensor) methods `greaterThan`,
`greaterOrEqual`, `lessThan`, `lessOrEqual`, `equalTo`, and `notEqualTo`. Each accepts an ordered
right Tensor operand, supports right-aligned [broadcasting](../../../../glossary.md#broadcasting) for floating inputs, and produces a [BOOL](../../../../glossary.md#data-type--datatype)
Tensor. Legacy [operation](../../../../glossary.md#operation) [descriptors](../../../../glossary.md#term-descriptor) represent six corresponding comparison [identities](../../../../glossary.md#term-identity-meanings).

Legacy descriptors also retain a precomputed `BroadcastPlan` and expose [arity](../../../../glossary.md#term-arity), [fusion](../../../../glossary.md#term-fusion),
semantic-family, cost, result-kind, and expression-string [metadata](../../../../glossary.md#term-metadata). Legacy builders mix local
broadcasting, data-type checks, [mutable](../../../../glossary.md#term-mutable) [graph](../../../../glossary.md#graph) construction, and execution-facing information.

Only the six mathematical identities and ordered operand roles are retained here. Broadcasting
and BOOL result derivation belong to task 0015B. Cost, fusion, capability, [lowering](../../../../glossary.md#lowering), and execution
metadata belong to [planning](../../../../glossary.md#term-planning) or [concrete backend](../../../../glossary.md#concrete-backend) preparation. No legacy source, hierarchy,
broadcast plan, or naming convention is copied.

## Architecture constraints

- Production remains in `modules/model`, which owns [backend](../../../../glossary.md#backend)-independent [Operation](../../../../glossary.md#operation) semantics.
- `BinaryComparisonKind` is a typed semantic discriminator, not a Tensor, graph occurrence, result
  descriptor, executable operation, backend capability, or kernel route.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [occurrence glossary](../../../../glossary.md#term-occurrence); [descriptor glossary](../../../../glossary.md#term-descriptor); [backend capability glossary](../../../../glossary.md#backend-capability); [kernel glossary](../../../../glossary.md#kernel))</span>
- The enum implements [`OperationKind`](../../../../glossary.md#operationkind) through inherited `Enum.name()` and adds no duplicate name
  field or method.
- Every kind has no intrinsic [parameters](../../../../glossary.md#term-parameter). Its complete attributes value is
  `NoOperationAttrs.INSTANCE`, never null or an empty map.
- Left/right order remains semantic for all six comparisons. Input identity belongs to future
  Tensor provenance and graph nodes.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance); [nodes glossary](../../../../glossary.md#node))</span>
- Broadcasting and BOOL output are derived from future expression inputs. They are not enum
  attributes, plans, strides, result tags, or fields.<span data-glossary-navigation="true"> ([Broadcasting glossary](../../../../glossary.md#broadcasting); [BOOL glossary](../../../../glossary.md#data-type--datatype); [strides glossary](../../../../glossary.md#term-stride))</span>
- Numeric edge behavior, accepted input types, promotion, shape compatibility, output
  differentiability, and backend availability are not kind metadata.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Stable enum names are diagnostic typed vocabulary, not wire tokens, registry keys, backend
  dispatch keys, kernel names, or reflection identifiers.<span data-glossary-navigation="true"> ([tokens glossary](../../../../glossary.md#term-token); [registry glossary](../../../../glossary.md#term-registry); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Package direction is `model.operation.elementwise.comparison -> model.operation`. It must not
  depend on Tensor, datatype, shape, graph, compiler, planning, runtime, prepare, backend, storage,
  or training packages.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare))</span>
- Stop if implementation needs another type, attributes, Tensor behavior, inference, result facts,
  backend metadata, dependency, or architecture change.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation` — supplies [`OperationKind`](../../../../glossary.md#operationkind), [`Operation`](../../../../glossary.md#operation), and
  [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs) for the semantic contract and composition tests.

Package added:

- `io.github.pho001.synaptik.model.operation.elementwise.comparison` — owns typed parameterless
  semantic kinds for ordered Tensor-to-Tensor comparisons.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>

Type placement:

- `io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind` — public
  family enum under the owning operation hierarchy.
- `BinaryComparisonKindTest` — same-package focused test for vocabulary and contract [shape](../../../../glossary.md#shape).

## Required contract

### Enum declaration

Create exactly:

```java
public enum BinaryComparisonKind implements OperationKind {
    GREATER_THAN,
    GREATER_OR_EQUAL,
    LESS_THAN,
    LESS_OR_EQUAL,
    EQUAL,
    NOT_EQUAL
}
```

The enum declares no project field, explicit constructor, method, nested type, per-[constant](../../../../glossary.md#term-constant) class
body, [symbol](../../../../glossary.md#term-symbol), [alias](../../../../glossary.md#aliasing-and-arena), or [metadata](../../../../glossary.md#term-metadata). [Compiler](../../../../glossary.md#term-compiler)-generated enum machinery is not additional project [API](../../../../glossary.md#term-application-programming-interface-api).
Inherited `Enum.name()` satisfies `OperationKind.name()` and returns exact constant text.

| Kind | Ordered [elementwise](../../../../glossary.md#term-elementwise) meaning |
|---|---|
| `GREATER_THAN` | left value is strictly greater than right value |
| `GREATER_OR_EQUAL` | left value is greater than or equal to right value |
| `LESS_THAN` | left value is strictly less than right value |
| `LESS_OR_EQUAL` | left value is less than or equal to right value |
| `EQUAL` | left and right values compare equal |
| `NOT_EQUAL` | left and right values compare unequal |

The table defines [identity](../../../../glossary.md#term-identity-meanings) and operand order only. It does not define data-type eligibility,
promotion, [broadcasting](../../../../glossary.md#broadcasting), [BOOL](../../../../glossary.md#data-type--datatype) representation, [NaN](../../../../glossary.md#term-nan)/[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) behavior, differentiation,
execution, or [backend](../../../../glossary.md#backend) availability.

### Parameterless [Operation](../../../../glossary.md#operation) composition

Every kind composes explicitly as:

```java
Operation operation = new Operation(
        BinaryComparisonKind.GREATER_THAN,
        NoOperationAttrs.INSTANCE);
```

Do not add a family factory or enum `operation()` method. Generic `Operation` remains open to
caller-supplied attributes and does not enforce family compatibility.

### Naming and typed [identity](../../../../glossary.md#term-identity-meanings)

- `values()` returns exactly six [constants](../../../../glossary.md#term-constant) in declared order.
- `name()` returns exact uppercase constant spelling.
- Equality and hashing remain standard Java enum semantics.
- A kind from another enum remains unequal even with equal diagnostic text.
- `toString()` remains diagnostic text, not [serialization](../../../../glossary.md#term-serialization) or [dispatch](../../../../glossary.md#term-dispatch) format.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/comparison/BinaryComparisonKind.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/comparison/BinaryComparisonKindTest.java`

Documentation/[planning](../../../../glossary.md#term-planning) during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `docs/api/compile-api.md`
- `docs/api/training-api.md`
- `docs/planning/modules/model/capabilities.md`
- Existing [operation](../../../../glossary.md#operation) foundations and completed kind-family Javadocs/tests.
- Tensor/expression contracts, focused architecture, ADRs, architecture tests,
  backend-conformance/integration tests, and Gradle configuration.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most one production file, one focused test, and five documentation/[planning](../../../../glossary.md#term-planning) files: seven paths
total.

Do not modify existing Java source/tests, capabilities, completed tasks, [Compile](../../../../glossary.md#compile)/Training [API](../../../../glossary.md#term-application-programming-interface-api),
[Gradle](../../../../glossary.md#term-gradle), AGENTS, architecture documents/tests, another [module](../../../../glossary.md#term-module), or unrelated documentation. Stop if
another production concept, attribute, expression, [inference](../../../../glossary.md#term-inference) rule, [dependency](../../../../glossary.md#term-dependency), or eighth path is
needed. Do not create task 0015B.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document the enum as backend-independent parameterless ordered comparison vocabulary and
  distinguish it from operands, result descriptors, graph occurrences, and executable support.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [descriptors glossary](../../../../glossary.md#term-descriptor); [graph glossary](../../../../glossary.md#graph))</span>
- Document every constant with left/right roles, strict versus inclusive relation where
  applicable, and deferral of eligibility, broadcasting, BOOL output, numerical edges, gradients,
  execution, and backend availability.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [broadcasting glossary](../../../../glossary.md#broadcasting); [BOOL glossary](../../../../glossary.md#data-type--datatype); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Document explicit [Operation](../../../../glossary.md#operation) composition with `NoOperationAttrs.INSTANCE` and absence of family
  compatibility validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Explain diagnostic names, typed equality, and non-[serialization](../../../../glossary.md#term-serialization)/non-[dispatch](../../../../glossary.md#term-dispatch) use.
- Review related Javadocs and record why they remain accurate, or stop on an inconsistency.

## Acceptance criteria

- Exactly one public enum is added in the planned comparison package.
- It implements [`OperationKind`](../../../../glossary.md#operationkind) and declares exactly six [constants](../../../../glossary.md#term-constant) in specified order/spelling.
- It adds no project field, method, constructor, nested type, constant body, alias, category,
  arity, cost, fusion, result, differentiability, backend, inference, or execution metadata.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [arity glossary](../../../../glossary.md#term-arity); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [inference glossary](../../../../glossary.md#term-inference); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Inherited names and standard enum equality/[hash](../../../../glossary.md#term-hash)/toString behavior remain.
- Every constant constructs a valid [Operation](../../../../glossary.md#operation) with exact kind and `NoOperationAttrs.INSTANCE`; no
  factory or attributes type is added.
- Operand order and meanings are documented without stored input state.
- No broadcast plan, Shape, DataType, BOOL result, Tensor, provenance, graph, compiler, storage,
  backend, dependency, or architecture behavior is added.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [DataType glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused/aggregate tests, Javadoc, root tests, reflection/javap/import/scope checks,
  documentation links/formatting, and statuses pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- A separate clean-context documentation agent finalizes Javadocs, Tensor API, glossary, evidence,
  master plan, and roadmap and records related no-change conclusions.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Task 0015A becomes Complete only after both passes. Task 0015B remains Draft without a detailed
  specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKindTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover exact package/visibility/interface/[constants](../../../../glossary.md#term-constant)/order/names; absence of
project fields/methods/nested types/constant bodies; standard enum [identity](../../../../glossary.md#term-identity-meanings); typed distinction from
a private test-local equal-name kind; explicit [Operation](../../../../glossary.md#operation) composition for every constant with exact
references; and absence of attributes, operands, broadcast/result [metadata](../../../../glossary.md#term-metadata), factories, registries,
reflection discovery, and dependencies.

Manually inspect `javap -p -c -s`, reflection, imports, and [Gradle](../../../../glossary.md#term-gradle) dependencies. Confirm no [Tensor](../../../../glossary.md#tensor),
[DataType](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape), broadcast plan, result kind, [graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) type, cost, [fusion](../../../../glossary.md#term-fusion),
route, [registry](../../../../glossary.md#term-registry), map, or service appears. Validate generated [Javadoc](../../../../glossary.md#term-javadoc), Tensor [API](../../../../glossary.md#term-application-programming-interface-api) status, glossary,
links/anchors/fences/whitespace, exact seven-path [scope](../../../../glossary.md#term-scope), synchronized statuses, and absence of a
task-0015B specification.

## Dependencies

- Task 0005 supplies [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), and `NoOperationAttrs.INSTANCE`.
- Task 0006 supplies [immutable](../../../../glossary.md#term-immutable) generic [`Operation`](../../../../glossary.md#operation) composition and reference retention.
- Completed parameterless kind families establish conventions but are not Java dependencies.

## Follow-up tasks

- 0015B remains Draft for floating validation, promotion, broadcasting, BOOL descriptor, public
  Tensor methods, and ordered provenance.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [broadcasting glossary](../../../../glossary.md#broadcasting); [descriptor glossary](../../../../glossary.md#term-descriptor); [Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance))</span>
- 0015C–0015D remain Draft for [BOOL](../../../../glossary.md#data-type--datatype) logical semantics/expressions.
- 0015E–0015F remain Draft for ternary `where` semantics/expression.
- 0015G–0015H remain Draft for typed cast semantics/expression.
- Compiler/autograd/backend/conformance tasks later own optimization, gradients, numerical edges,
  lowering, and execution.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [gradients glossary](../../../../glossary.md#term-gradient); [lowering glossary](../../../../glossary.md#lowering))</span>

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns [backend](../../../../glossary.md#backend)-independent [Operation](../../../../glossary.md#operation) semantics to
`modules/model`. The new package refines that [ownership](../../../../glossary.md#term-ownership-meanings) without dependencies, [inference](../../../../glossary.md#term-inference), [compiler](../../../../glossary.md#term-compiler)
behavior, or executable state.

If implementation requires [Tensor](../../../../glossary.md#tensor) behavior, inference, broadcast state, backend [metadata](../../../../glossary.md#term-metadata), another
[dependency](../../../../glossary.md#term-dependency), or architecture change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0005/0006/0014A/0014C/0014E/0015A, Tensor API, Compile API,
Training API, glossary, current OperationKind/OperationAttrs/NoOperationAttrs/Operation and concrete
kind-family contracts/tests, and Java 26 Gradle configuration.

Implement task 0015A exactly. Add only BinaryComparisonKind.java and BinaryComparisonKindTest.java
under io.github.pho001.synaptik.model.operation.elementwise.comparison for Java code/tests.

The public enum implements OperationKind and contains exactly GREATER_THAN, GREATER_OR_EQUAL,
LESS_THAN, LESS_OR_EQUAL, EQUAL, NOT_EQUAL in that order, with no project fields, methods, nested
types, aliases, attributes, or metadata. Every kind is parameterless and composes explicitly with
Operation plus NoOperationAttrs.INSTANCE.

Do not add Tensor methods, attributes/factories, operands, broadcasting, data-type or BOOL result
rules, provenance, graph/compiler/planning/runtime/backend behavior, legacy broadcast plans/traits,
dependencies/build/architecture changes, existing Java edits, or later specs. Stop beyond seven
paths or on architecture uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status check. Then hand the diff/evidence to a separate clean-
context documentation agent in the same change to finalize permitted Javadocs/Tensor API/glossary/
planning, record no-change conclusions, and rerun validation.

Update task 0015A, model master plan, and roadmap only for status/evidence. Do not mark Complete
until both passes succeed. Leave 0015B Draft without a specification. Do not commit or push.
```

## Local decisions

- Broad 0015 is decomposed into semantic/public-expression pairs for comparisons, [BOOL](../../../../glossary.md#data-type--datatype) logic,
  `where`, and cast so no task mixes unrelated [API](../../../../glossary.md#term-application-programming-interface-api)/result rules.
- One typed comparison family is cohesive because all six share two ordered Tensor operands,
  parameterlessness, future broadcasting, and future BOOL output.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- Descriptive [constants](../../../../glossary.md#term-constant) replace legacy abbreviations. Names are diagnostics, not wire keys.
- Broadcast geometry and BOOL result facts remain in task 0015B.
- `NoOperationAttrs.INSTANCE` is complete; no empty comparison attrs record is introduced.

## Known limitations

- No public comparison expression exists until 0015B.
- Input eligibility, promotion, [broadcasting](../../../../glossary.md#broadcasting), [BOOL](../../../../glossary.md#data-type--datatype) output, and `requiresGrad` are undefined here.
- [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), and ordering/equality edge semantics remain unspecified.
- No [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings), [gradient](../../../../glossary.md#term-gradient), [backend](../../../../glossary.md#backend) support, or execution is implied.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed architecture/documentation rules; planning guide, capabilities, master plan,
roadmap; completed [Operation](../../../../glossary.md#operation) foundations and family patterns through 0014F; current foundation and
kind source/tests; and read-only legacy comparison methods, [descriptors](../../../../glossary.md#term-descriptor), builders, [broadcasting](../../../../glossary.md#broadcasting),
[BOOL](../../../../glossary.md#data-type--datatype) results, and [kernel](../../../../glossary.md#kernel) inventory.

Planning found former 0015 too broad and decomposed it into eight ordered tasks 0015A–0015H. Only
0015A has a detailed specification. It needs one enum and one test and introduces no [dependency](../../../../glossary.md#term-dependency),
foundation change, result [inference](../../../../glossary.md#term-inference), or architecture change.

Planning validation:

- `git diff --check` passed, and the three changed planning files contain no trailing whitespace.
- The [canonical](../../../../glossary.md#term-canonical-form) section [scan](../../../../glossary.md#term-cumulative-scan) found every required task-specification section.
- The relative Markdown-[target](../../../../glossary.md#term-target) scan resolved every local `.md` link in this task, the model master
  plan, and the roadmap.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Status inspection found task 0015A [`Ready`](../../../../glossary.md#term-ready) exactly once in this specification, its model-master
  row, and its roadmap row.
- The master plan contains ordered Draft rows 0015B–0015H with explicit dependencies, and the
  roadmap sequence was expanded and renumbered consistently through task 0024.
- Scope inspection found exactly this new task plus the model master plan and roadmap changed; no
  Java, test, Gradle, AGENTS, architecture, API, glossary, or other module file changed during
  planning.<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [Gradle glossary](../../../../glossary.md#term-gradle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module))</span>
- No task-0015B specification exists; 0015B remains only a Draft queue entry.

Implementation and documentation validation:

- Implementation context `/root/implement_model_0014d` added exactly the planned enum and focused
  same-package test. Independent documentation context `/root/implement_model_0014b` then performed
  the mandatory fresh reread of the architecture contract, focused architecture pages,
  documentation and planning rules, current APIs, glossary, capability/master plans, relevant
  completed tasks, actual diff/source/tests, generated reports, bytecode, and generated Javadoc.
  It applied General plus API/Javadoc style to the Java and Tensor API review, Planning style to
  planning files, and Example format to the explicit Operation-composition example.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [bytecode glossary](../../../../glossary.md#term-bytecode); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Tensor glossary](../../../../glossary.md#tensor))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKindTest`
  passed. An explicit fresh `--rerun-tasks` execution produced 5 tests with zero failures, errors,
  or skips.
- `./gradlew :modules:model:test` passed; 41 XML suites contain 308 tests with zero failures,
  errors, or skips. The model test task executed in the recorded final aggregate run.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- `./gradlew :modules:model:javadoc` passed, and an explicit fresh `--rerun-tasks` execution
  regenerated clean public Javadoc. The generated type page contains the backend-independent,
  parameterless, ordered-family contract, explicit <span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>`Operation`/`NoOperationAttrs.INSTANCE`
  composition, family-compatibility limitation, typed diagnostic naming, and all six constant
  meanings and deferrals. The all-classes, index, and type-search pages contain the new type and
  constants.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- `./gradlew test` passed for the repository; its final run reported all 36 actionable tasks
  up-to-date and no failing task. The model suite immediately preceding it was freshly executed.
- `javap -p -c -s` confirms the exact six enum constants in specified order plus only compiler-
  generated enum machinery. Reflection tests confirm the public/final enum shape, sole<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [shape glossary](../../../../glossary.md#shape))</span>
  [`OperationKind`](../../../../glossary.md#operationkind) interface, compiler constructor, no project fields/methods/nested types or
  constant bodies, inherited enum behavior, exact names, and typed distinction from an equally
  named test-local kind.
- Composition tests construct every kind with `NoOperationAttrs.INSTANCE` and confirm exact
  kind/attributes reference retention through generic `Operation`. They do not add or imply a
  family factory or compatibility validator.
- Production has exactly one import, `OperationKind`. Source, bytecode, dependency, and diff scans
  found no operands, attributes type, Tensor, DataType, Shape, BOOL result fact, broadcast plan,
  provenance, graph/compiler/planning/prepare/runtime/backend type, gradient, cost, fusion, route,
  registry, map, service, dependency, build, or executable behavior.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [gradient glossary](../../../../glossary.md#term-gradient); [fusion glossary](../../../../glossary.md#term-fusion); [registry glossary](../../../../glossary.md#term-registry))</span>
- The independent documentation pass found `BinaryComparisonKind` and every constant Javadoc
  complete, so it changed no Java declaration, Javadoc, executable behavior, or test. It updated
  the Tensor API and existing glossary entries to describe the current semantic vocabulary while
  keeping public comparison expressions, inference, provenance, and execution planned.
- The local Markdown validator resolved every local file target and heading anchor in the five
  changed documentation/planning files. Fence counts are balanced, targeted trailing-whitespace
  scans found no matches, and `git diff --check` passed.
- Final scope contains exactly the seven authorized paths: one production file, one focused test,
  Tensor API, glossary, this task, model master plan, and roadmap. No Compile API, Training API,
  capabilities, architecture/ADR/test, Gradle/build, existing Java/test contract, another module,
  backend-conformance test, integration test, or task-0015B specification path changed.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test))</span>
- Task 0015A is synchronized as Complete in this specification, the model master plan, and the
  roadmap. Task 0015B remains the next Draft frontier without a detailed specification.
- Existing `OperationKind`, [`OperationAttrs`](../../../../glossary.md#operationattrs), [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), `Operation`, binary arithmetic,
  unary, scalar, Tensor, descriptor, provenance, and expression Javadocs remain accurate because
  the enum implements and composes existing contracts without changing validation, attributes,
  public Tensor methods, descriptors, provenance, or expression behavior.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [unary glossary](../../../../glossary.md#term-unary); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- The Compile API requires no edit because a semantic enum adds no public comparison expression,
  capture, inference, optimization, artifact, or execution contract. The Training API requires no
  edit because no <span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [artifact glossary](../../../../glossary.md#term-artifact))</span>`requiresGrad` rule, gradient rule, autograd, optimizer, or training behavior
  changed. <span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>`capabilities.md` requires no edit because it already selects all six comparisons and
  distinguishes model semantics/public expressions from compiler and execution support.
- `ARCHITECTURE.md`, focused architecture pages, ADRs, architecture tests, backend-conformance and
  integration tests, and build configuration require no edit because the change remains within
  the existing model-owned operation-semantics boundary and adds no dependency, module boundary,
  lifecycle, backend, numerical-execution, Java-toolchain, preview/incubator, or end-to-end rule.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [toolchain glossary](../../../../glossary.md#term-toolchain))</span>

## Implementation notes

- Added exactly one public `BinaryComparisonKind` enum with `GREATER_THAN`, `GREATER_OR_EQUAL`,
  `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`, and `NOT_EQUAL` in the specified order and one focused
  same-package test.
- The enum is typed semantic identity only. It stores no operands, attributes, broadcast/result
  facts, execution metadata, or backend capability, and generic <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [metadata glossary](../../../../glossary.md#term-metadata); [backend capability glossary](../../../../glossary.md#backend-capability))</span>[`Operation`](../../../../glossary.md#operation) remains open and
  family-agnostic.
- The independent documentation pass added the current comparison semantic family, exact ordered
  meanings, explicit parameterless composition, diagnostic-name distinction, and planned limits
  to the Tensor API and existing glossary concepts. No new reusable glossary term was needed.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- No Java declaration, [Javadoc](../../../../glossary.md#term-javadoc), executable logic, or test changed during the documentation pass.

## Completion summary

- Completed changes: Implemented and documented the six typed, parameterless, ordered binary
  comparison semantic kinds.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- Files changed or created: Exactly one production file, one focused test, Tensor API, glossary,
  this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused comparison semantics 5/5, all 308 model tests across 41 suites,
  regenerated model Javadoc, root tests, bytecode/reflection/import/dependency/absence checks,
  local links/anchors, fences, terminology, whitespace, exact scope/status, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`
  passed.
- Documentation-agent review: Clean documentation context `/root/implement_model_0014b` completed
  the independent pass using General, API/Javadoc, Planning, and Example-format guidance.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and glossary now describe the current comparison semantic
  vocabulary while preserving public Tensor comparison expressions and all later lifecycle stages
  as planned. Compile API, Training API, and capabilities remain accurate unchanged.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [Compile glossary](../../../../glossary.md#compile))</span>
- Javadoc review: The new enum type and all six constants are complete unchanged during the
  documentation pass; existing operation foundation, concrete family, Tensor, descriptor,
  provenance, and expression contracts remain accurate unchanged.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- Glossary impact: Existing implementation-status, [`OperationKind`](../../../../glossary.md#operationkind), and common-distinction entries
  now include the comparison family; no new reusable domain term was needed.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0015A. Task 0015B remains Draft without a specification.

Status: Complete
