# Task 0015C: [Boolean](../../../../glossary.md#data-type--datatype) Logical Semantic Kinds

## Status

Complete

## Goal

Define the typed, [backend](../../../../glossary.md#backend)-independent, parameterless semantic vocabulary for the three selected
[elementwise](../../../../glossary.md#term-elementwise) [boolean](../../../../glossary.md#data-type--datatype) logical capabilities: conjunction, disjunction, and negation.

This task creates only the semantic family consumed by task 0015D. It does not add public [Tensor](../../../../glossary.md#tensor)
logical methods, validate [BOOL](../../../../glossary.md#data-type--datatype) [descriptors](../../../../glossary.md#term-descriptor), broadcast inputs, derive results, record [provenance](../../../../glossary.md#provenance), or
execute boolean logic.

## [Scope](../../../../glossary.md#term-scope)

- Add one public enum `BooleanLogicalKind` implementing [`OperationKind`](../../../../glossary.md#operationkind).
- Define exactly `AND`, `OR`, and `NOT` in that order.
- Document the [boolean](../../../../glossary.md#data-type--datatype) truth meaning and logical input roles of every kind.
- Establish that all three kinds have no intrinsic attributes and compose explicitly with
  `NoOperationAttrs.INSTANCE` when represented by [`Operation`](../../../../glossary.md#operation).
- Document [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) family context for `AND` and `OR` and [unary](../../../../glossary.md#term-unary) family context for `NOT` without
  storing arity metadata.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Add one focused same-package test proving exact vocabulary, typed identity, enum behavior,
  parameterless Operation composition, and absence of metadata/state.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Add the cohesive `model.operation.elementwise.logical` package to the [model](../../../../glossary.md#term-model) package map.
- Finalize affected Javadocs, Tensor API, glossary, task evidence, model master plan, and roadmap
  through the required independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public `Tensor.logicalAnd`, `logicalOr`, or `logicalNot` methods
- Tensor inputs, provenance, IDs, graph values/nodes, result construction, labels, storage,
  expression chaining, or <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance); [nodes glossary](../../../../glossary.md#node))</span>`TensorFactory.createDerived`
- DataType imports, BOOL input validation, BOOL output descriptors, broadcasting, shape
  preservation, layout, <span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [validation glossary](../../../../glossary.md#term-validation); [descriptors glossary](../../../../glossary.md#term-descriptor); [broadcasting glossary](../../../../glossary.md#broadcasting); [layout glossary](../../../../glossary.md#layout))</span>`requiresGrad`, or local/[graph](../../../../glossary.md#graph)-wide [inference](../../../../glossary.md#term-inference)
- numeric truthiness, conversion from floating/integral values, raw BOOL byte normalization,
  null/unknown/three-valued logic, or short-circuit evaluation<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- XOR, NAND, NOR, XNOR, implication, equivalence, reduction `all`/`any`, or bitwise operations
- family attributes, factory, registry, parser, aliases, symbols, string dispatch, reflection
  discovery, visitor, service, map, or compatibility validator<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- [aliases](../../../../glossary.md#aliasing-and-arena) such as `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT`, `&&`, `||`, or `!`
- fields for arity, category, input/result data type, cost, fusion, differentiability, backend
  support, lowering, route, kernel, or execution metadata<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [data type glossary](../../../../glossary.md#data-type--datatype); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- [mutable](../../../../glossary.md#term-mutable) or [immutable](../../../../glossary.md#term-immutable) broadcast plans, [strides](../../../../glossary.md#term-stride), output [shapes](../../../../glossary.md#shape), or [materialization](../../../../glossary.md#materialization) requirements
- [gradients](../../../../glossary.md#term-gradient), [backward graph](../../../../glossary.md#backward-graph) construction, [autograd](../../../../glossary.md#autograd), [optimizer](../../../../glossary.md#term-optimizer), or training behavior
- comparison, `where`, cast, reduction, selection, or another [operation](../../../../glossary.md#operation) family
- compiler, planning, prepare, runtime, backend, engine, tracing, dependency, Gradle, architecture,
  or another-module changes<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- changes to existing Java contracts/tests or a detailed task-0015D specification

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
- [Task 0015A](0015a-binary-comparison-semantic-kinds.md)
- [Task 0015B](0015b-binary-comparison-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes public [Tensor](../../../../glossary.md#tensor) methods `logicalAnd(Tensor)`,
`logicalOr(Tensor)`, and `logicalNot()`. The corresponding legacy [operation](../../../../glossary.md#operation) [descriptors](../../../../glossary.md#term-descriptor) identify
[elementwise](../../../../glossary.md#term-elementwise) conjunction, disjunction, and negation. Legacy public [validation](../../../../glossary.md#term-validation) accepts [BOOL](../../../../glossary.md#data-type--datatype) inputs;
[binary](../../../../glossary.md#term-binary-descriptor-and-metadata) AND/OR use right-aligned [broadcasting](../../../../glossary.md#broadcasting), [unary](../../../../glossary.md#term-unary) NOT preserves [shape](../../../../glossary.md#shape), and all three produce
non-[gradient](../../../../glossary.md#term-gradient) BOOL results. Legacy tests cover truth results, AND broadcasting, comparison-[mask](../../../../glossary.md#term-boolean-mask)
chaining, `where` conditions, non-[contiguous](../../../../glossary.md#term-contiguous) inputs, [ONNX](../../../../glossary.md#term-onnx) mapping, and multiple [backend](../../../../glossary.md#backend) routes.

Legacy descriptors also store broadcast plans for binary operations and expose [arity](../../../../glossary.md#term-arity), semantic
family, cost, [fusion](../../../../glossary.md#term-fusion), result-kind, expression-string, [lowering](../../../../glossary.md#lowering), and execution-facing [metadata](../../../../glossary.md#term-metadata).
Those mechanisms are not copied. This task retains only the three logical [identities](../../../../glossary.md#term-identity-meanings) and their
logical input roles. BOOL descriptor validation, broadcasting, shape preservation, and [provenance](../../../../glossary.md#provenance)
belong to task 0015D. Capability, lowering, numerical storage, and execution belong to later owning
layers.

## Architecture constraints

- Production remains in `modules/model`, which owns [backend](../../../../glossary.md#backend)-independent [Operation](../../../../glossary.md#operation) semantics.
- `BooleanLogicalKind` is a typed semantic discriminator, not a Tensor, input list, graph
  occurrence, result descriptor, executable operation, backend capability, or kernel route.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [occurrence glossary](../../../../glossary.md#term-occurrence); [descriptor glossary](../../../../glossary.md#term-descriptor); [backend capability glossary](../../../../glossary.md#backend-capability); [kernel glossary](../../../../glossary.md#kernel))</span>
- The enum implements [`OperationKind`](../../../../glossary.md#operationkind) through inherited `Enum.name()` and adds no duplicate name
  field or method.
- Every kind has no intrinsic [parameters](../../../../glossary.md#term-parameter). Its complete attributes value is
  `NoOperationAttrs.INSTANCE`, never null or an empty map.
- `AND` and `OR` describe two logical input roles; `NOT` describes one. This family context is
  documentation, not stored arity metadata. Input identities belong to later Tensor provenance and
  graph nodes.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [nodes glossary](../../../../glossary.md#node))</span>
- `AND` and `OR` are mathematically commutative, but this task does not reorder inputs or define a
  canonical provenance order. Task 0015D will retain caller order.<span data-glossary-navigation="true"> ([commutative glossary](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- BOOL eligibility, BOOL output representation, broadcasting, shape preservation, layout,
  gradient eligibility, and differentiation are not enum fields.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [broadcasting glossary](../../../../glossary.md#broadcasting); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Logical truth semantics do not authorize numeric truthiness, bitwise interpretation, raw storage
  access, short-circuiting, or three-valued logic.
- Stable enum names are diagnostic typed vocabulary, not wire tokens, registry keys, backend
  dispatch keys, kernel names, operators, or reflection identifiers.<span data-glossary-navigation="true"> ([tokens glossary](../../../../glossary.md#term-token); [registry glossary](../../../../glossary.md#term-registry); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Package direction is `model.operation.elementwise.logical -> model.operation`. It must not
  depend on Tensor, datatype, shape, graph, compiler, planning, runtime, prepare, backend, storage,
  or training packages.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare))</span>
- Stop if implementation needs another type, attributes, Tensor behavior, inference, result facts,
  backend metadata, dependency, or architecture change.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing package used:

- `io.github.pho001.synaptik.model.operation` — supplies [`OperationKind`](../../../../glossary.md#operationkind), [`Operation`](../../../../glossary.md#operation), and
  [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs) for the semantic contract and composition tests.

Package added:

```text
io.github.pho001.synaptik.model.operation.elementwise.logical
  Typed parameterless semantic kinds for elementwise boolean conjunction, disjunction,
  and negation.
```

One family is cohesive because all three kinds consume logical values, [carry](../../../../glossary.md#term-carry) no intrinsic
attributes, preserve [elementwise](../../../../glossary.md#term-elementwise) position, produce logical results in the future public expression
contract, and share the same [BOOL](../../../../glossary.md#data-type--datatype)-only [validation](../../../../glossary.md#term-validation) boundary. The [unary](../../../../glossary.md#term-unary)/[binary](../../../../glossary.md#term-binary-descriptor-and-metadata) distinction does not
require separate semantic packages or stored [arity](../../../../glossary.md#term-arity) [metadata](../../../../glossary.md#term-metadata).

Type placement:

- `io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind` — public
  family enum under the owning operation hierarchy.
- `BooleanLogicalKindTest` — same-package focused test for vocabulary and contract [shape](../../../../glossary.md#shape).

## Required contract

### Enum declaration

Create exactly:

```java
public enum BooleanLogicalKind implements OperationKind {
    AND,
    OR,
    NOT
}
```

The enum declares no project field, explicit constructor, method, nested type, per-[constant](../../../../glossary.md#term-constant) class
body, [symbol](../../../../glossary.md#term-symbol), [alias](../../../../glossary.md#aliasing-and-arena), or [metadata](../../../../glossary.md#term-metadata). [Compiler](../../../../glossary.md#term-compiler)-generated enum machinery is not additional project [API](../../../../glossary.md#term-application-programming-interface-api).
Inherited `Enum.name()` satisfies `OperationKind.name()` and returns exact constant text.

| Kind | [Elementwise](../../../../glossary.md#term-elementwise) [boolean](../../../../glossary.md#data-type--datatype) meaning | Logical input roles |
|---|---|---|
| `AND` | true exactly when both input values are true | ordered left and right inputs |
| `OR` | true exactly when at least one input value is true | ordered left and right inputs |
| `NOT` | true exactly when the input value is false | one input |

The table defines semantic truth [identity](../../../../glossary.md#term-identity-meanings) and logical roles only. It does not define Java
short-circuit behavior, public [Tensor](../../../../glossary.md#tensor) [arity](../../../../glossary.md#term-arity) [validation](../../../../glossary.md#term-validation), accepted [descriptors](../../../../glossary.md#term-descriptor), [broadcasting](../../../../glossary.md#broadcasting), output
descriptor construction, storage encoding, differentiation, execution, or [backend](../../../../glossary.md#backend) availability.

### Parameterless [Operation](../../../../glossary.md#operation) composition

Every kind composes explicitly as:

```java
Operation operation = new Operation(
        BooleanLogicalKind.AND,
        NoOperationAttrs.INSTANCE);
```

Do not add a family factory, enum `operation()` method, attributes type, or compatibility [registry](../../../../glossary.md#term-registry).
Generic `Operation` remains an open typed pairing and does not enforce family-specific [arity](../../../../glossary.md#term-arity) or
kind-to-attributes compatibility.

### Naming and typed [identity](../../../../glossary.md#term-identity-meanings)

- `values()` returns exactly three [constants](../../../../glossary.md#term-constant) in declared order.
- `name()` returns exact uppercase constant spelling.
- Equality and hashing remain standard Java enum semantics.
- A kind from another enum remains unequal even with equal diagnostic text.
- `toString()` remains diagnostic text, not [serialization](../../../../glossary.md#term-serialization), parsing, or [dispatch](../../../../glossary.md#term-dispatch) format.
- The family name supplies [boolean](../../../../glossary.md#data-type--datatype)-logical context; constants are `AND`, `OR`, and `NOT`, not
  duplicated `LOGICAL_*` [aliases](../../../../glossary.md#aliasing-and-arena).

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/logical/BooleanLogicalKind.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/logical/BooleanLogicalKindTest.java`

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
- Existing Operation foundation, concrete kind families, Tensor comparison expressions, and their
  Javadocs/tests.<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Focused architecture documentation, ADRs, architecture tests, backend-conformance material,
  integration tests, and Gradle configuration.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most one production file, one focused test, and five documentation/[planning](../../../../glossary.md#term-planning) files: seven paths
total.

Do not modify existing Java source/tests, capabilities, completed tasks, [Compile](../../../../glossary.md#compile)/Training [API](../../../../glossary.md#term-application-programming-interface-api),
[Gradle](../../../../glossary.md#term-gradle), AGENTS, architecture documents/tests, another [module](../../../../glossary.md#term-module), or unrelated documentation. Stop if
another production concept, attributes, [Tensor](../../../../glossary.md#tensor) expression, [descriptor](../../../../glossary.md#term-descriptor) rule, [dependency](../../../../glossary.md#term-dependency), or eighth
path is needed. Do not create task 0015D.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document the enum as backend-independent parameterless elementwise boolean-logical vocabulary
  and distinguish it from inputs, result descriptors, graph occurrences, and executable support.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [elementwise glossary](../../../../glossary.md#term-elementwise); [boolean glossary](../../../../glossary.md#data-type--datatype); [descriptors glossary](../../../../glossary.md#term-descriptor); [graph glossary](../../../../glossary.md#graph))</span>
- Document `AND` as conjunction of left/right logical inputs, `OR` as disjunction of left/right
  logical inputs, and `NOT` as negation of one logical input.
- For every constant, explain the truth meaning and defer descriptor eligibility, broadcasting or
  shape preservation, BOOL representation, gradients, execution, and backend availability.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [broadcasting glossary](../../../../glossary.md#broadcasting); [shape glossary](../../../../glossary.md#shape); [BOOL glossary](../../../../glossary.md#data-type--datatype); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Explain that mixed unary/binary family context is not stored arity metadata and is not validated
  by generic <span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [arity glossary](../../../../glossary.md#term-arity); [metadata glossary](../../../../glossary.md#term-metadata))</span>[`Operation`](../../../../glossary.md#operation).
- Document explicit Operation composition with `NoOperationAttrs.INSTANCE` and absence of family
  compatibility validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Explain diagnostic names, typed equality, and non-[serialization](../../../../glossary.md#term-serialization)/non-[dispatch](../../../../glossary.md#term-dispatch) use.
- Review related Javadocs and record why they remain accurate, or stop on an inconsistency.

## Acceptance criteria

- Exactly one public `BooleanLogicalKind` enum is added in the planned logical package.
- It implements [`OperationKind`](../../../../glossary.md#operationkind) and declares exactly `AND`, `OR`, and `NOT` in specified order.
- It adds no project field, method, constructor, nested type, constant body, alias, category,
  arity, cost, fusion, result, differentiability, backend, inference, or execution metadata.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [inference glossary](../../../../glossary.md#term-inference); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Inherited names and standard enum equality/[hash](../../../../glossary.md#term-hash)/toString behavior remain.
- Every [constant](../../../../glossary.md#term-constant) constructs a valid [Operation](../../../../glossary.md#operation) with exact kind and `NoOperationAttrs.INSTANCE`; no
  factory or attributes type is added.
- Truth meanings and [unary](../../../../glossary.md#term-unary)/[binary](../../../../glossary.md#term-binary-descriptor-and-metadata) logical roles are documented without stored input or [arity](../../../../glossary.md#term-arity) state.
- Production imports only `OperationKind`; no DataType, Shape, Tensor, provenance, graph, compiler,
  planning, runtime, prepare, backend, storage, training, dependency, or architecture behavior is
  added.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- No broadcast plan, BOOL descriptor, numerical truthiness, storage encoding, gradient rule, or
  public Tensor method is added.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [descriptor glossary](../../../../glossary.md#term-descriptor); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Focused/aggregate tests, Javadoc, root tests, reflection/javap/import/scope checks,
  documentation links/formatting, and statuses pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- A separate clean-context documentation agent finalizes Javadocs, Tensor API, glossary, evidence,
  master plan, and roadmap and records related no-change conclusions.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Task 0015C becomes Complete only after both passes. Task 0015D remains Draft without a detailed
  specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKindTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover exact package/visibility/interface/[constants](../../../../glossary.md#term-constant)/order/names; absence of
project fields/methods/nested types/constant bodies; standard enum [identity](../../../../glossary.md#term-identity-meanings); typed distinction from
a private test-local equal-name kind; explicit [Operation](../../../../glossary.md#operation) composition for every constant with exact
references; mixed [unary](../../../../glossary.md#term-unary)/[binary](../../../../glossary.md#term-binary-descriptor-and-metadata) family vocabulary without [arity](../../../../glossary.md#term-arity) state; and absence of attributes,
inputs, [descriptors](../../../../glossary.md#term-descriptor), factories, registries, reflection discovery, and dependencies.

Manually inspect `javap -p -c -s`, reflection, imports, and [Gradle](../../../../glossary.md#term-gradle) dependencies. Confirm no
[DataType](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape), [Tensor](../../../../glossary.md#tensor), [provenance](../../../../glossary.md#provenance), broadcast plan, [BOOL](../../../../glossary.md#data-type--datatype) result tag, [graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend)
type, [gradient](../../../../glossary.md#term-gradient), cost, [fusion](../../../../glossary.md#term-fusion), route, [registry](../../../../glossary.md#term-registry), map, or service appears. Validate generated [Javadoc](../../../../glossary.md#term-javadoc),
Tensor [API](../../../../glossary.md#term-application-programming-interface-api) status, glossary, links/anchors/fences/whitespace, exact seven-path [scope](../../../../glossary.md#term-scope), synchronized
statuses, package-map placement, and absence of a task-0015D specification.

## Dependencies

- Task 0005 supplies [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), and `NoOperationAttrs.INSTANCE`.
- Task 0006 supplies [immutable](../../../../glossary.md#term-immutable) generic [`Operation`](../../../../glossary.md#operation) composition and reference retention.
- Completed parameterless kind families establish enum and test conventions but are not Java
  dependencies.
- Task 0015B confirms the comparison-to-BOOL expression seam used later for boolean chaining; it is
  implementation-order context rather than a Java dependency of this enum.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [boolean glossary](../../../../glossary.md#data-type--datatype); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Follow-up tasks

- 0015D remains Draft for BOOL-only validation, binary broadcasting, unary shape preservation,
  fixed BOOL descriptors, public Tensor methods, and ordered provenance.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [validation glossary](../../../../glossary.md#term-validation); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [broadcasting glossary](../../../../glossary.md#broadcasting); [unary glossary](../../../../glossary.md#term-unary); [shape glossary](../../../../glossary.md#shape); [descriptors glossary](../../../../glossary.md#term-descriptor); [Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance))</span>
- 0015E–0015F remain Draft for ternary `where` semantics and expression construction.
- 0015G–0015H remain Draft for typed cast semantics and expression construction.
- Compiler, backend, and conformance tasks later own capture, optimization, lowering, truth-value
  execution, and storage interpretation.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [capture glossary](../../../../glossary.md#term-capture-meanings); [lowering glossary](../../../../glossary.md#lowering))</span>

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns [backend](../../../../glossary.md#backend)-independent [Operation](../../../../glossary.md#operation) semantics to
`modules/model`. The new package refines that [ownership](../../../../glossary.md#term-ownership-meanings) without dependencies, result [inference](../../../../glossary.md#term-inference),
[compiler](../../../../glossary.md#term-compiler) behavior, storage, or executable state.

If implementation requires [Tensor](../../../../glossary.md#tensor) behavior, [descriptor](../../../../glossary.md#term-descriptor) inference, backend [metadata](../../../../glossary.md#term-metadata), another
[dependency](../../../../glossary.md#term-dependency), or architecture change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0005/0006/0014A/0014C/0014E/0015A/0015B/0015C, Tensor API,
Compile API, Training API, glossary, current OperationKind/OperationAttrs/NoOperationAttrs/
Operation and concrete kind-family contracts/tests, and Java 26 Gradle configuration.

Implement task 0015C exactly. Add only BooleanLogicalKind.java and BooleanLogicalKindTest.java
under io.github.pho001.synaptik.model.operation.elementwise.logical for Java code/tests.

The public enum implements OperationKind and contains exactly AND, OR, NOT in that order, with no
project fields, methods, nested types, aliases, attributes, arity, or metadata. Every kind is
parameterless and composes explicitly with Operation plus NoOperationAttrs.INSTANCE. Document AND
and OR as two-input conjunction/disjunction and NOT as one-input negation without storing or
validating arity.

Do not add Tensor methods, operation factories/attrs, DataType/Shape/BOOL descriptor rules,
broadcasting, provenance, numeric truthiness, storage behavior, gradients, graph/compiler/
planning/runtime/backend behavior, legacy broadcast plans/traits, dependencies/build/architecture
changes, existing Java edits, or later specs. Stop beyond seven paths or on architecture doubt.

Run every specified focused/aggregate test, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff/evidence to a separate
clean-context documentation agent in the same change. It must inspect source/tests/generated
Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record related-contract/
capability/Compile API/Training API/architecture no-change conclusions, and rerun validation.

Update task 0015C, model master plan, and roadmap only for planning status/evidence. Do not mark
0015C Complete until both passes succeed. Leave 0015D Draft without a specification. Do not commit
or push.
```

## Local decisions

- The public semantic type is `BooleanLogicalKind`: [`Boolean`](../../../../glossary.md#data-type--datatype) distinguishes truth semantics from
  numerical/bitwise operations, while `LogicalKind` alone would be unnecessarily broad.
- One family contains `AND`, `OR`, and `NOT` because all three are parameterless elementwise boolean
  semantics with the same future BOOL-only expression boundary. Mixed unary/binary arity is
  documented rather than stored as metadata.<span data-glossary-navigation="true"> ([elementwise glossary](../../../../glossary.md#term-elementwise); [BOOL glossary](../../../../glossary.md#data-type--datatype); [unary glossary](../../../../glossary.md#term-unary); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [arity glossary](../../../../glossary.md#term-arity); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Short [constants](../../../../glossary.md#term-constant) avoid redundant `LOGICAL_*` prefixes because the family type supplies context.
  Names are diagnostic typed vocabulary, not ONNX names, wire tokens, operators, or dispatch keys.<span data-glossary-navigation="true"> ([ONNX glossary](../../../../glossary.md#term-onnx); [tokens glossary](../../../../glossary.md#term-token); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- The selected baseline contains no XOR-family capability. It is not added speculatively.
- `NoOperationAttrs.INSTANCE` is complete; no empty logical attrs record is introduced.

## Known limitations

- No public logical [Tensor](../../../../glossary.md#tensor) expression exists until task 0015D.
- [Descriptor](../../../../glossary.md#term-descriptor) eligibility, [broadcasting](../../../../glossary.md#broadcasting), [shape](../../../../glossary.md#shape) preservation, [BOOL](../../../../glossary.md#data-type--datatype) output construction, and
  `requiresGrad` behavior are undefined here.
- The enum does not enforce one versus two inputs or kind-to-attributes compatibility.
- No compiler capture, gradient behavior, backend support, storage interpretation, or execution is
  implied.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed the [architecture contract](../../../../glossary.md#architecture-contract) and focused [module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency) explanations;
documentation and planning rules; roadmap; [model](../../../../glossary.md#term-model) capabilities and master plan; tasks 0005, 0006,
0014A, 0014C, 0014E, 0015A, and 0015B; current [Operation](../../../../glossary.md#operation) foundations and concrete kind-family
source/tests; [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/Training APIs and glossary; and Java 26 [Gradle](../../../../glossary.md#term-gradle) configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms public
`logicalAnd`, `logicalOr`, and `logicalNot` capabilities, their parameterless semantic [identities](../../../../glossary.md#term-identity-meanings),
[binary](../../../../glossary.md#term-binary-descriptor-and-metadata)/[unary](../../../../glossary.md#term-unary) roles, [BOOL](../../../../glossary.md#data-type--datatype)-only public validation, binary [broadcasting](../../../../glossary.md#broadcasting), unary [shape](../../../../glossary.md#shape) preservation,
BOOL no-[gradient](../../../../glossary.md#term-gradient) results, chaining, and [backend](../../../../glossary.md#backend)/[ONNX](../../../../glossary.md#term-onnx) evidence. [Mutable](../../../../glossary.md#term-mutable) legacy broadcast plans,
traits, expression strings, [runtime](../../../../glossary.md#term-runtime) coupling, [kernels](../../../../glossary.md#kernel), storage interpretation, execution, and
[compiler](../../../../glossary.md#term-compiler) behavior are excluded.

Planning selected one enum and one focused test because all three kinds share [boolean](../../../../glossary.md#data-type--datatype) [elementwise](../../../../glossary.md#term-elementwise)
semantics, parameterlessness, the same future BOOL-only public boundary, and no independent
attributes. [Arity](../../../../glossary.md#term-arity) remains family context instead of enum state. No existing package, dependency,
foundation contract, or architecture rule changes.

Planning validation:

- `git diff --check` passed, and targeted whitespace inspection found no trailing whitespace in
  the three changed planning paths.
- The required-section scan found every canonical task-specification section, including package
  impact, exact scope, validation, implementation handoff, decisions, limitations, and completion
  evidence sections.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [scope glossary](../../../../glossary.md#term-scope); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- The relative Markdown-[target](../../../../glossary.md#term-target) [scan](../../../../glossary.md#term-cumulative-scan) resolved every local `.md` link in this task, the model master
  plan, and the roadmap. Markdown fence counts are balanced and no changed link uses an unresolved
  heading anchor.
- Status inspection found 0015C [`Ready`](../../../../glossary.md#term-ready) in this specification, its linked model-master row, and
  its linked roadmap row/current-frontier text. Task 0015D remains `Draft` in both queues.
- Package-map inspection found exactly one new planned package,
  `model.operation.elementwise.logical`, with direction only to `model.operation`.
- Scope inspection found exactly this new task, the model master plan, and the roadmap changed. No
  Java, test, API, glossary, Gradle, architecture, AGENTS, or other module path changed.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [module path glossary](../../../../glossary.md#term-module-path))</span>
- No task-0015D specification exists.

Implementation and documentation validation:

- Implementation context `/root/implement_model_0015c` added exactly
  `BooleanLogicalKind.java` and `BooleanLogicalKindTest.java`. Independent clean-context
  documentation context `/root/implement_model_0015c/review_model_0015c_docs` reread the
  architecture contract, focused model/module/dependency/lifecycle pages, documentation and
  planning rules, current APIs and glossary, model capabilities/master plans, related completed
  tasks, all operation foundation and concrete kind-family source/tests, the actual diff, generated
  Javadoc, test reports, bytecode, imports, and Java 26 build configuration. It applied General
  plus API/Javadoc style to source, Tensor API, and glossary, and Planning style to this task, the
  model master plan, and roadmap.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKindTest` — `BUILD
  SUCCESSFUL`; the XML report contains 5 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 43 XML suites contain 322 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated
  `BooleanLogicalKind.html` contains the type contract and `AND`, `OR`, and `NOT` details, including
  conjunction/disjunction/negation truth meaning, mixed arity as family context, explicit
  parameterless composition, typed diagnostic identity, and deferred descriptor, shape, storage,
  gradient, execution, and backend behavior.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- `./gradlew test` — `BUILD SUCCESSFUL`; all 36 actionable tasks were up to date with no failing
  task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s
  io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind` confirmed the
  exact `AND`, `OR`, `NOT` order and only compiler-generated enum fields, methods, constructor, and
  initialization. Focused reflection tests independently confirmed the public/final enum shape,
  sole [`OperationKind`](../../../../glossary.md#operationkind) interface, absence of project instance fields/methods, nested types,
  constant bodies, arity state, and additional project API.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- Production import inspection found exactly one import, `OperationKind`. Composition tests cover
  every constant with exact `Operation` kind and `NoOperationAttrs.INSTANCE` reference retention,
  while the private equal-name `OtherKind.AND` test confirms typed separation.
- The independent documentation pass found the enum type and all three constant Javadocs complete,
  so it changed no Java declaration, Javadoc, executable behavior, or test. Tensor API now lists
  the implemented family, exact truth roles, explicit parameterless composition, diagnostic
  identity, and planned boundaries. Existing glossary implementation-status, `OperationKind`, and
  common-distinction text now includes the logical family; no new glossary term was added because
  `BooleanLogicalKind` is a concrete implementation of the existing operation-kind concept rather
  than a new reusable domain, lifecycle, or ownership term.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `OperationKind`, [`OperationAttrs`](../../../../glossary.md#operationattrs), [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), `Operation`, all existing concrete kind
  families, and Tensor expression Javadocs/tests remain accurate because the enum implements and
  composes their existing contracts without changing validation, attributes, descriptors,
  provenance, public Tensor methods, or expression behavior.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Compile API remains accurate unchanged because this semantic enum adds no public expression,
  capture entry point, traversal, inference, optimization, artifacts, or execution. Training API
  remains accurate unchanged because no gradient eligibility, rule, autograd, optimizer, or
  training behavior changed. <span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [artifacts glossary](../../../../glossary.md#term-artifact); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>`capabilities.md` remains accurate unchanged because it already
  selects `logicalAnd`, `logicalOr`, and `logicalNot` and distinguishes semantic representation
  from public expression and executable support.
- `ARCHITECTURE.md`, focused architecture pages, ADRs, architecture tests, backend-conformance and
  integration tests, and Java 26 Gradle/build configuration remain accurate unchanged because the
  change stays within model-owned operation semantics and adds no module boundary, dependency,
  lifecycle, backend behavior, end-to-end execution, toolchain/release, preview/incubator, or
  dependency rule.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [toolchain glossary](../../../../glossary.md#term-toolchain); [release glossary](../../../../glossary.md#term-release))</span>
- Local Markdown file and heading-anchor validation resolved all 201 links in the five changed
  documentation/planning files. Markdown fences are balanced, terminology distinguishes current
  semantic vocabulary from planned Tensor/executable behavior, no changed path has trailing
  whitespace, and `git diff --check` passed.
- Final scope contains exactly the seven authorized paths: the enum, focused test, Tensor API,
  glossary, this task, model master plan, and roadmap. No Compile API, Training API, capabilities,
  architecture/ADR/test, Gradle/build, existing Java/test, another module, backend-conformance,
  integration-test, or task-0015D specification path changed. Task 0015C is synchronized as
  Complete in all three planning locations; task 0015D remains Draft without a detailed
  specification.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

## Implementation notes

- Added exactly one public `BooleanLogicalKind` enum with `AND`, `OR`, and `NOT` in the specified
  order and one focused same-package five-test suite.
- The enum is typed semantic identity only. It stores no inputs, attributes, arity, descriptor or
  broadcast facts, provenance, storage interpretation, executable metadata, or backend capability;
  generic <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [arity glossary](../../../../glossary.md#term-arity); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [metadata glossary](../../../../glossary.md#term-metadata); [backend capability glossary](../../../../glossary.md#backend-capability))</span>[`Operation`](../../../../glossary.md#operation) remains open and family-agnostic.
- The independent documentation pass finalized Tensor API, glossary, and planning status/evidence
  while preserving the already-complete source Javadocs and tests unchanged.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- No public logical Tensor expression, compiler/training behavior, architecture, dependency,
  numerical execution, or backend behavior was added.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [dependency glossary](../../../../glossary.md#term-dependency); [backend glossary](../../../../glossary.md#backend))</span>

## Completion summary

- Completed changes: Implemented and documented the three typed, parameterless elementwise boolean
  logical semantic kinds with mixed unary/binary roles kept as family context.<span data-glossary-navigation="true"> ([elementwise glossary](../../../../glossary.md#term-elementwise); [boolean glossary](../../../../glossary.md#data-type--datatype); [unary glossary](../../../../glossary.md#term-unary); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- Files changed or created: Exactly one production enum, one focused test, Tensor API, glossary,
  this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused logical semantics 5/5, all 322 model tests across 43 suites, model
  Javadoc, root tests, bytecode/reflection/import/absence checks, generated-documentation review,
  local links/anchors, fences, terminology, whitespace, exact scope/status, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`
  passed.
- Documentation-agent review: Clean documentation context
  `/root/implement_model_0015c/review_model_0015c_docs` completed the independent pass using
  General, API/Javadoc, and Planning profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and glossary now describe boolean logical semantic vocabulary
  as current while retaining public logical Tensor expressions and every compiler, gradient,
  backend, storage-interpretation, and execution layer as planned. Compile API, Training API,
  capabilities, architecture/ADRs/tests, conformance/integration tests, and build configuration
  remain accurate unchanged for the reasons recorded above.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend); [Compile glossary](../../../../glossary.md#compile); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Javadoc review: The enum type and all three constants are complete unchanged during the
  documentation pass; operation foundations, existing kind families, and Tensor expression
  contracts remain accurate unchanged.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [operation glossary](../../../../glossary.md#operation))</span>
- Glossary impact: Existing implementation-status, [`OperationKind`](../../../../glossary.md#operationkind), and common-distinction entries
  now include the logical family; no new reusable project term was introduced.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0015C. Task 0015D remains Draft without a detailed
  specification.

Status: Complete
