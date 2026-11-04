# Task 0015E: Where Selection Semantic Kind

**Editorial reading key (later context, not a status update).**

WHERE makes a per-position true/false choice, unlike scalar-index select, which fixes a coordinate. Condition and the two branch roles are part of the meaning; there is no axis attribute and no promise of eager or lazy branch execution. Here Model defines a typed meaning for later construction code; this semantic-family task neither receives stored operands nor evaluates them. Its completion is not evidence of Compiler capture, derivatives or backend execution.

Focused current reading: [Current conditional-selection reference](../../../../reference/tensor/selection-casts.md#conditional-selection-expression). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Define the typed, [backend](../../../../glossary.md#backend)-independent, parameterless semantic [identity](../../../../glossary.md#term-identity-meanings) for [elementwise](../../../../glossary.md#term-elementwise) conditional
selection: `WHERE`.

The [operation](../../../../glossary.md#operation) means that one condition value chooses between corresponding true-branch and
false-branch values. This task creates only that meaning. It does not add the public three-[Tensor](../../../../glossary.md#tensor)
expression, validate a [BOOL](../../../../glossary.md#data-type--datatype) condition or floating branches, broadcast three [shapes](../../../../glossary.md#shape), derive a
result [descriptor](../../../../glossary.md#term-descriptor), record [provenance](../../../../glossary.md#provenance), calculate selected values, or define executable support.

## [Scope](../../../../glossary.md#term-scope)

- Add one public enum `WhereSelectionKind` implementing [`OperationKind`](../../../../glossary.md#operationkind).
- Define exactly one [constant](../../../../glossary.md#term-constant), `WHERE`.
- Document the three ordered logical input roles: condition, true branch, and false branch.
- Establish that `WHERE` has no intrinsic attributes and composes explicitly with
  `NoOperationAttrs.INSTANCE` when represented by [`Operation`](../../../../glossary.md#operation).
- Document ternary family context without storing or validating [arity](../../../../glossary.md#term-arity).
- Distinguish conditional `WHERE` from [scalar](../../../../glossary.md#term-scalar)-index selection and other indexing operations.
- Add one focused same-package test proving exact vocabulary, typed identity, enum behavior,
  parameterless Operation composition, and absence of metadata/state.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Add the cohesive `model.operation.elementwise.selection` package to the [model](../../../../glossary.md#term-model) package map.
- Finalize affected Javadocs, Tensor API, glossary, task evidence, model master plan, and roadmap
  through the required independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public `Tensor.where(Tensor, Tensor, Tensor)` or another Tensor method, overload, factory, or
  expression-builder type<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Tensor inputs, provenance, IDs, graph values/nodes, result construction, labels, storage,
  expression chaining, or <span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [nodes glossary](../../../../glossary.md#node))</span>`TensorFactory.createDerived`
- DataType imports, BOOL condition validation, floating branch validation or promotion, result
  data type, three-way broadcasting, output shape, layout, <span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [validation glossary](../../../../glossary.md#term-validation); [data type glossary](../../../../glossary.md#data-type--datatype); [broadcasting glossary](../../../../glossary.md#broadcasting); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout))</span>`requiresGrad`, or [inference](../../../../glossary.md#term-inference)
- eager branch evaluation, value selection, condition storage encoding, short-circuit behavior,
  lazy branch evaluation, constant folding, or numerical edge policy<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [evaluation glossary](../../../../glossary.md#term-evaluation); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [constant folding glossary](../../../../glossary.md#term-constant-folding))</span>
- gradient eligibility, branch gradient routing, backward graph construction, autograd, optimizer,
  or training behavior<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [backward graph glossary](../../../../glossary.md#backward-graph); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- family attributes, factory, registry, parser, alias, symbol, string dispatch, reflection
  discovery, visitor, service, map, or compatibility validator<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [symbol glossary](../../../../glossary.md#term-symbol); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- [aliases](../../../../glossary.md#aliasing-and-arena) such as `SELECT`, `CHOOSE`, `CONDITIONAL`, or `IF_THEN_ELSE`
- fields for arity, category, condition type, branch/result type, cost, fusion, differentiability,
  backend support, lowering, route, kernel, or execution metadata<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- a broadcast plan, effective strides, output geometry, materialization requirement, or backend
  storage facts<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [materialization glossary](../../../../glossary.md#materialization))</span>
- [scalar](../../../../glossary.md#term-scalar)-index `select`, gather, take, scatter, masking, reduction, cast, or another operation
  family<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather); [operation glossary](../../../../glossary.md#operation))</span>
- compiler, planning, prepare, runtime, backend, engine, tracing, ONNX, dependency, Gradle,
  architecture, or another-module changes<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [ONNX glossary](../../../../glossary.md#term-onnx); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- changes to existing Java contracts/tests or a detailed task-0015F specification

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
- [Task 0015A](0015a-binary-comparison-semantic-kinds.md)
- [Task 0015C](0015c-boolean-logical-semantic-kinds.md)
- [Task 0015D](0015d-boolean-logical-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes static
`Tensor.where(condition, ifTrue, ifFalse)`. Its public contract uses a [BOOL](../../../../glossary.md#data-type--datatype) condition, two floating
branches, common three-way [broadcasting](../../../../glossary.md#broadcasting), floating branch promotion, and the ordered rule “true
selects `ifTrue`, false selects `ifFalse`.” Legacy tests exercise comparison and logical [masks](../../../../glossary.md#term-boolean-mask),
broadcasting, mixed floating branch types, non-[contiguous](../../../../glossary.md#term-contiguous) inputs, [fused](../../../../glossary.md#specialization-and-fusion) chains, [autograd](../../../../glossary.md#autograd), [ONNX](../../../../glossary.md#term-onnx),
and several [backend](../../../../glossary.md#backend) routes.

The legacy semantic [descriptor](../../../../glossary.md#term-descriptor) also exposes family, cost, [fusion](../../../../glossary.md#term-fusion), result-kind, expression-string,
[lowering](../../../../glossary.md#lowering), broadcast-plan, and execution-facing facts. Those mechanisms are not copied. This task
retains only the `WHERE` [identity](../../../../glossary.md#term-identity-meanings) and the condition/true-branch/false-branch roles. Task 0015F owns
public [Tensor](../../../../glossary.md#tensor) [validation](../../../../glossary.md#term-validation), branch promotion, three-way broadcasting, result construction, and
ordered [provenance](../../../../glossary.md#provenance). [Compiler](../../../../glossary.md#term-compiler), training, [planning](../../../../glossary.md#term-planning), [prepare](../../../../glossary.md#prepare), and concrete backends retain their own
responsibilities.

## Architecture constraints

- Production remains in `modules/model`, which owns [backend](../../../../glossary.md#backend)-independent [Operation](../../../../glossary.md#operation) semantics.
- `WhereSelectionKind` is a typed semantic discriminator, not a Tensor, input list, graph
  occurrence, result descriptor, executable operation, backend capability, or kernel route.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [occurrence glossary](../../../../glossary.md#term-occurrence); [descriptor glossary](../../../../glossary.md#term-descriptor); [backend capability glossary](../../../../glossary.md#backend-capability); [kernel glossary](../../../../glossary.md#kernel))</span>
- The enum implements [`OperationKind`](../../../../glossary.md#operationkind) through inherited `Enum.name()` and adds no duplicate name
  field or method.
- `WHERE` has no intrinsic [parameters](../../../../glossary.md#term-parameter). Its complete attributes value is
  `NoOperationAttrs.INSTANCE`, never null or an empty map.
- The operation has three logical input roles in exact order: condition, true branch, false branch.
  This ternary context is documentation, not stored arity metadata. Input identities belong to
  later Tensor provenance and graph nodes.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [nodes glossary](../../../../glossary.md#node))</span>
- The meaning chooses the true-branch value when the corresponding condition is true and the
  false-branch value otherwise. It does not prescribe eager or lazy branch evaluation.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- BOOL condition eligibility, branch data-type eligibility and promotion, broadcasting, result
  descriptor facts, gradient eligibility, and differentiation are not enum fields.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [broadcasting glossary](../../../../glossary.md#broadcasting); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- `WHERE` conditional selection is separate from [scalar](../../../../glossary.md#term-scalar)-index `select`, gather, take, and scatter.
  Those operations remain under the later indexing/scatter task.<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather))</span>
- Stable enum names are diagnostic typed vocabulary, not ONNX tokens, wire values, registry keys,
  backend dispatch keys, kernel names, operators, or reflection identifiers.<span data-glossary-navigation="true"> ([ONNX glossary](../../../../glossary.md#term-onnx); [tokens glossary](../../../../glossary.md#term-token); [registry glossary](../../../../glossary.md#term-registry); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Package direction is `model.operation.elementwise.selection -> model.operation`. It must not
  depend on Tensor, datatype, shape, graph, compiler, planning, runtime, prepare, backend, storage,
  or training packages.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare))</span>
- Stop if implementation needs another type, attributes, Tensor behavior, descriptor inference,
  a broadcast plan, backend metadata, dependency, or architecture change.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing package used:

- `io.github.pho001.synaptik.model.operation` — supplies [`OperationKind`](../../../../glossary.md#operationkind), [`Operation`](../../../../glossary.md#operation), and
  [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs) for the semantic contract and composition tests.

Package added:

```text
io.github.pho001.synaptik.model.operation.elementwise.selection
  Typed parameterless semantics for elementwise conditional branch selection.
```

The package is deliberately below [`elementwise`](../../../../glossary.md#term-elementwise): `WHERE` makes one independent choice at each
broadcasted result position. The `selection` name describes semantic conditional selection, while
the parent distinguishes it from later [scalar](../../../../glossary.md#term-scalar)-index selection and indexing/scatter operations.

Type placement:

- `io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind` — public
  family enum under the owning operation hierarchy.
- `WhereSelectionKindTest` — same-package focused test for vocabulary and contract [shape](../../../../glossary.md#shape).

## Required contract

### Enum declaration

Create exactly:

```java
public enum WhereSelectionKind implements OperationKind {
    WHERE
}
```

The enum declares no project field, explicit constructor, method, nested type, per-[constant](../../../../glossary.md#term-constant) class
body, [symbol](../../../../glossary.md#term-symbol), [alias](../../../../glossary.md#aliasing-and-arena), or [metadata](../../../../glossary.md#term-metadata). [Compiler](../../../../glossary.md#term-compiler)-generated enum machinery is not additional project [API](../../../../glossary.md#term-application-programming-interface-api).
Inherited `Enum.name()` satisfies `OperationKind.name()` and returns exact constant text.

| Kind | [Elementwise](../../../../glossary.md#term-elementwise) meaning | Logical input roles |
|---|---|---|
| `WHERE` | choose the true-branch value when the corresponding condition is true; otherwise choose the false-branch value | ordered condition, true branch, false branch |

The table defines semantic [identity](../../../../glossary.md#term-identity-meanings) and roles only. It does not define public [Tensor](../../../../glossary.md#tensor) [arity](../../../../glossary.md#term-arity)
[validation](../../../../glossary.md#term-validation), accepted [descriptors](../../../../glossary.md#term-descriptor), [broadcasting](../../../../glossary.md#broadcasting), branch promotion, output descriptor construction,
storage behavior, [evaluation](../../../../glossary.md#term-evaluation) order, differentiation, execution, or [backend](../../../../glossary.md#backend) availability.

### Parameterless [Operation](../../../../glossary.md#operation) composition

The kind composes explicitly as:

```java
Operation operation = new Operation(
        WhereSelectionKind.WHERE,
        NoOperationAttrs.INSTANCE);
```

Do not add a family factory, enum `operation()` method, attributes type, or compatibility [registry](../../../../glossary.md#term-registry).
Generic `Operation` remains an open typed pairing and does not enforce family-specific [arity](../../../../glossary.md#term-arity) or
kind-to-attributes compatibility.

### Naming and typed [identity](../../../../glossary.md#term-identity-meanings)

- `values()` returns exactly one [constant](../../../../glossary.md#term-constant), `WHERE`.
- `name()` returns exact uppercase text `WHERE`.
- Equality and hashing remain standard Java enum semantics.
- A kind from another enum remains unequal even if it also has the diagnostic name `WHERE`.
- `toString()` remains diagnostic text, not serialization, parsing, ONNX mapping, or dispatch
  format.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [ONNX glossary](../../../../glossary.md#term-onnx); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- No `SELECT` [alias](../../../../glossary.md#aliasing-and-arena) is added because [scalar](../../../../glossary.md#term-scalar)-index selection is a distinct [operation](../../../../glossary.md#operation) capability.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/selection/WhereSelectionKind.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/selection/WhereSelectionKindTest.java`

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
- Existing Operation foundations, concrete kind families, logical Tensor expressions, and their
  Javadocs/tests.<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Focused architecture documentation, ADRs, architecture tests, backend-conformance material,
  integration tests, and Gradle configuration.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most one production file, one focused test, and five documentation/[planning](../../../../glossary.md#term-planning) files: seven paths
total.

Do not modify existing Java source/tests, capabilities, completed tasks, [Compile](../../../../glossary.md#compile)/Training [API](../../../../glossary.md#term-application-programming-interface-api),
[Gradle](../../../../glossary.md#term-gradle), AGENTS, architecture documents/tests, another [module](../../../../glossary.md#term-module), or unrelated documentation. Stop if
another production concept, attributes, [Tensor](../../../../glossary.md#tensor) expression, [descriptor](../../../../glossary.md#term-descriptor) rule, [dependency](../../../../glossary.md#term-dependency), or eighth
path is needed. Do not create task 0015F.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document the enum as backend-independent parameterless elementwise conditional-selection
  vocabulary and distinguish it from inputs, result descriptors, graph occurrences, indexing
  selection, and executable support.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [elementwise glossary](../../../../glossary.md#term-elementwise); [descriptors glossary](../../../../glossary.md#term-descriptor); [graph glossary](../../../../glossary.md#graph))</span>
- Document `WHERE` with the exact ordered condition, true-branch, and false-branch roles and explain
  which branch value is selected for true and false condition values.
- Explain that ternary family context is not stored [arity](../../../../glossary.md#term-arity) [metadata](../../../../glossary.md#term-metadata) and is not validated by generic
  [`Operation`](../../../../glossary.md#operation).
- Document explicit Operation composition with `NoOperationAttrs.INSTANCE` and absence of family
  compatibility validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Defer condition/branch descriptor eligibility, branch promotion, three-way broadcasting, result
  descriptor, evaluation order, gradients, execution, ONNX mapping, and backend availability.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [evaluation glossary](../../../../glossary.md#term-evaluation); [gradients glossary](../../../../glossary.md#term-gradient); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- Explain diagnostic names, typed equality, and non-[serialization](../../../../glossary.md#term-serialization)/non-[dispatch](../../../../glossary.md#term-dispatch) use.
- Review related Javadocs and record why they remain accurate, or stop on an inconsistency.

## Acceptance criteria

- Exactly one public `WhereSelectionKind` enum is added in the planned selection package.
- It implements [`OperationKind`](../../../../glossary.md#operationkind) and declares exactly `WHERE`.
- It adds no project field, method, constructor, nested type, constant body, alias, attributes,
  arity, category, cost, fusion, result, differentiability, backend, inference, or execution
  metadata.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [alias glossary](../../../../glossary.md#aliasing-and-arena); [arity glossary](../../../../glossary.md#term-arity); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [inference glossary](../../../../glossary.md#term-inference); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Inherited name and standard enum equality/[hash](../../../../glossary.md#term-hash)/toString behavior remain.
- `WHERE` constructs a valid [Operation](../../../../glossary.md#operation) with the exact kind and `NoOperationAttrs.INSTANCE`; no
  factory or attributes type is added.
- The exact three logical input roles and true/false choice meaning are documented without stored
  inputs or arity state.
- Production imports only `OperationKind`; no DataType, Shape, Tensor, provenance, graph, compiler,
  planning, runtime, prepare, backend, storage, training, dependency, or architecture behavior is
  added.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- No broadcast plan, branch/result descriptor, numerical execution, gradient rule, public Tensor
  method, scalar-index selection, or ONNX/backend mapping is added.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [gradient glossary](../../../../glossary.md#term-gradient); [scalar glossary](../../../../glossary.md#term-scalar); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- Focused/aggregate tests, Javadoc, root tests, reflection/javap/import/scope checks,
  documentation links/formatting, and statuses pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- A separate clean-context documentation agent finalizes Javadocs, Tensor API, glossary, evidence,
  master plan, and roadmap and records related no-change conclusions.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Task 0015E becomes Complete only after both passes. Task 0015F remains Draft without a detailed
  specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKindTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover exact package/visibility/interface/[constant](../../../../glossary.md#term-constant)/name; absence of project
fields/methods/nested types/constant bodies; standard enum [identity](../../../../glossary.md#term-identity-meanings); typed distinction from a
private test-local equal-name kind; explicit [Operation](../../../../glossary.md#operation) composition with exact references; ternary
family vocabulary without [arity](../../../../glossary.md#term-arity) state; and absence of attributes, inputs, [descriptors](../../../../glossary.md#term-descriptor), factories,
registries, reflection discovery, and dependencies.

Manually inspect `javap -p -c -s`, reflection, imports, and [Gradle](../../../../glossary.md#term-gradle) dependencies. Confirm no
[DataType](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape), [Tensor](../../../../glossary.md#tensor), [provenance](../../../../glossary.md#provenance), broadcast plan, result tag, [graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend)
type, [gradient](../../../../glossary.md#term-gradient), cost, [fusion](../../../../glossary.md#term-fusion), route, [registry](../../../../glossary.md#term-registry), map, or service appears. Validate generated [Javadoc](../../../../glossary.md#term-javadoc),
Tensor [API](../../../../glossary.md#term-application-programming-interface-api) status, glossary, links/anchors/fences/whitespace, exact seven-path [scope](../../../../glossary.md#term-scope), synchronized
statuses, package-map placement, and absence of a task-0015F specification.

## Dependencies

- Task 0005 supplies [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), and `NoOperationAttrs.INSTANCE`.
- Task 0006 supplies [immutable](../../../../glossary.md#term-immutable) generic [`Operation`](../../../../glossary.md#operation) composition and reference retention.
- Completed parameterless kind families establish enum and test conventions but are not Java
  dependencies.
- Task 0015D supplies the BOOL-producing public expression seam useful to future conditions; it is
  implementation-order context rather than a Java dependency of this enum.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Follow-up tasks

- 0015F remains Draft for exact public `Tensor.where` shape, BOOL condition validation, floating
  branch validation and promotion, three-way broadcasting, result descriptor construction,
  ordered provenance, and storage-free derived Tensor creation.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [BOOL glossary](../../../../glossary.md#data-type--datatype); [validation glossary](../../../../glossary.md#term-validation); [broadcasting glossary](../../../../glossary.md#broadcasting); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [Tensor glossary](../../../../glossary.md#tensor))</span>
- 0015G–0015H remain Draft for typed cast semantics and expression construction.
- Indexing/scatter tasks later own [scalar](../../../../glossary.md#term-scalar)-index `select`, [gather](../../../../glossary.md#gather), take, and functional scatter.
- Compiler, training, backend, ONNX, and conformance tasks later own capture, differentiation,
  optimization, mapping, lowering, selection execution, and storage interpretation.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance); [capture glossary](../../../../glossary.md#term-capture-meanings); [lowering glossary](../../../../glossary.md#lowering))</span>

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
model capabilities/master plan, tasks 0005/0006/0014A/0015A/0015C/0015D/0015E, Tensor API,
Compile API, Training API, glossary, current OperationKind/OperationAttrs/NoOperationAttrs/
Operation and concrete kind-family contracts/tests, and Java 26 Gradle configuration.

Implement task 0015E exactly. Add only WhereSelectionKind.java and
WhereSelectionKindTest.java under io.github.pho001.synaptik.model.operation.elementwise.selection
for Java code/tests.

The public enum implements OperationKind and contains exactly WHERE, with no project fields,
methods, nested types, aliases, attributes, arity, or metadata. The kind is parameterless and
composes explicitly with Operation plus NoOperationAttrs.INSTANCE. Document exact ordered
condition/true-branch/false-branch roles and conditional choice meaning without storing inputs or
validating arity. Keep WHERE distinct from scalar-index SELECT.

Do not add Tensor methods, operation factories/attrs, DataType/Shape/descriptor or promotion rules,
three-way broadcasting, provenance, evaluation policy, gradients, graph/compiler/planning/runtime/
backend/ONNX behavior, legacy broadcast plans/traits, dependencies/build/architecture changes,
existing Java edits, or later specs. Stop beyond seven paths or on architecture uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff/evidence to a separate
clean-context documentation agent in the same change. It must inspect source/tests/generated
Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record related-contract/
capability/Compile API/Training API/architecture no-change conclusions, and rerun validation.

Update task 0015E, model master plan, and roadmap only for planning status/evidence. Do not mark
0015E Complete until both passes succeed. Leave 0015F Draft without a specification. Do not commit
or push.
```

## Local decisions

- The type is `WhereSelectionKind`: `Where` preserves the public capability name and `Selection`
  identifies its semantic family without claiming the broader indexing meaning of a generic
  `SelectionKind`.
- The package is `operation.elementwise.selection`. The [`elementwise`](../../../../glossary.md#term-elementwise) parent makes conditional
  branch selection distinct from the future indexing/scatter hierarchy and its scalar-index<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
  `select` capability.
- The sole [constant](../../../../glossary.md#term-constant) is `WHERE`; no `SELECT` alias is introduced because those words name different
  capabilities in the selected baseline.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Ternary arity and condition/branch roles are documentation, not fields. Input identities and
  local validation belong to task 0015F.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [validation glossary](../../../../glossary.md#term-validation))</span>
- `NoOperationAttrs.INSTANCE` completely represents the absence of intrinsic parameters; an empty
  where-attributes record would carry no additional semantic fact.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [carry glossary](../../../../glossary.md#term-carry))</span>
- Evaluation order is intentionally unspecified. The model identity expresses elementwise choice,
  not eager/lazy host-language control flow.<span data-glossary-navigation="true"> ([Evaluation glossary](../../../../glossary.md#term-evaluation); [model glossary](../../../../glossary.md#term-model); [identity glossary](../../../../glossary.md#term-identity-meanings); [eager glossary](../../../../glossary.md#term-eager-evaluation); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [host glossary](../../../../glossary.md#term-host))</span>

## Known limitations

- No public `Tensor.where` expression exists until task 0015F.
- Condition and branch eligibility, branch promotion, three-way broadcasting, output descriptor,
  layout, and <span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout))</span>`requiresGrad` behavior are undefined here.
- The enum does not enforce three inputs or kind-to-attributes compatibility.
- No compiler capture, differentiation, ONNX mapping, backend support, storage interpretation, or
  execution is implied.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [ONNX glossary](../../../../glossary.md#term-onnx); [backend glossary](../../../../glossary.md#backend))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed the [architecture contract](../../../../glossary.md#architecture-contract) and focused [module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency)/[lifecycle](../../../../glossary.md#lifecycle) explanations;
documentation and planning rules; roadmap; [model](../../../../glossary.md#term-model) capabilities and master plan; tasks 0005, 0006,
0014A, 0015A, 0015C, and 0015D; current [Operation](../../../../glossary.md#operation) foundations and concrete kind-family source/tests;
[Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/Training APIs and glossary; and Java 26 [Gradle](../../../../glossary.md#term-gradle) configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms static public
`Tensor.where(condition, ifTrue, ifFalse)`, the parameterless `WHERE` semantic [identity](../../../../glossary.md#term-identity-meanings), ordered
condition/true-branch/false-branch roles, [BOOL](../../../../glossary.md#data-type--datatype) condition validation, floating branch promotion,
three-way [broadcasting](../../../../glossary.md#broadcasting), [gradient](../../../../glossary.md#term-gradient) routing, [ONNX](../../../../glossary.md#term-onnx) mapping, [fused](../../../../glossary.md#specialization-and-fusion) chains, and multiple [backend](../../../../glossary.md#backend) routes.
Legacy broadcast plans, operation traits, expression strings, [autograd](../../../../glossary.md#autograd) machinery, storage access,
[kernels](../../../../glossary.md#kernel), [lowering](../../../../glossary.md#lowering), and execution are excluded from this semantic-only task.

Planning selected one enum and one focused test because `WHERE` has one stable parameterless
semantic identity. A separate attributes record, [arity](../../../../glossary.md#term-arity) field, broadcast plan, or input container
would add no intrinsic operation [parameter](../../../../glossary.md#term-parameter). Public validation and result construction remain in
task 0015F. No existing dependency, foundation contract, or architecture rule changes.

Planning validation:

- `git diff --check` passed, and targeted whitespace inspection found no trailing whitespace in
  the three changed planning paths.
- The required-section scan found every canonical task-specification section, including package
  impact, bounded scope, validation, implementation handoff, decisions, limitations, and
  completion-evidence sections.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [canonical glossary](../../../../glossary.md#term-canonical-form); [scope glossary](../../../../glossary.md#term-scope); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- Every local Markdown file link in this task, the model master plan, and the roadmap resolves.
  Markdown fence counts are balanced.
- Status inspection found 0015E [`Ready`](../../../../glossary.md#term-ready) in this specification, its linked model-master row, and
  its linked roadmap row/current-frontier text. Task 0015F remains `Draft` in both queues.
- Package-map inspection found exactly one new planned package,
  `model.operation.elementwise.selection`, with direction only to `model.operation`.
- Scope inspection found exactly this new task, the model master plan, and the roadmap changed. No
  Java, test, API, glossary, Gradle, architecture, AGENTS, or other-module path changed.<span data-glossary-navigation="true"> ([module path glossary](../../../../glossary.md#term-module-path))</span>
- No task-0015F specification exists.

Implementation and independent documentation validation:

- Clean implementation work added exactly `WhereSelectionKind.java` and its focused same-package
  test. The enum has the sole `WHERE` [constant](../../../../glossary.md#term-constant), implements [`OperationKind`](../../../../glossary.md#operationkind), imports only that
  interface, and declares no project field, method, constructor, nested type, constant body,
  attributes, alias, arity state, or metadata.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- The focused test passed with 5 tests and zero failures, errors, or skips. It covers exact
  vocabulary and enum shape, inherited diagnostic behavior, typed family distinction, and exact<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
  `Operation` composition with `NoOperationAttrs.INSTANCE`.
- An explicit fresh aggregate model run passed with 336 tests and zero failures, errors, or skips.
  Model Javadoc and root tests also passed.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- Independent documentation context
  `/root/implement_model_0015d/review_model_0015d_docs` reread the architecture contract,
  focused architecture pages, documentation and planning rules, current API references,
  glossary, capability/master plans, related tasks, final source/tests, operation foundations and
  every concrete kind family, generated reports/Javadoc, actual diff, and Java 26 build
  configuration. It applied General plus API/Javadoc style to source and API documentation and
  Planning style to task/master/roadmap. No complete example changed, so the example profile
  required no example revision.
- The independent pass found the enum and constant Javadocs complete. They already document
  backend-independent parameterless meaning, exact condition/true-branch/false-branch roles,
  ternary context without stored arity, explicit no-attributes composition, generic compatibility
  limits, diagnostic typed identity, scalar-index-selection distinction, and every deferred
  descriptor/compiler/training/execution concern. No Java or test edit was needed.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [descriptor glossary](../../../../glossary.md#term-descriptor); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Tensor [API](../../../../glossary.md#term-application-programming-interface-api) now lists `WhereSelectionKind`, explains its sole conditional-selection identity and
  explicit composition, distinguishes it from scalar-index `select`, and keeps public
  `Tensor.where` and its validation/inference/provenance behavior planned. The glossary records the
  same implemented-versus-planned boundary without creating a redundant reusable term.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [provenance glossary](../../../../glossary.md#provenance))</span>
- Compile API remains accurate unchanged because no public expression, compiler capture,
  inference, transformation, artifact, or execution contract was added. Training API remains
  accurate unchanged because no gradient, autograd, optimizer, or training behavior changed.
  Capabilities remain accurate because they already select public <span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [artifact glossary](../../../../glossary.md#term-artifact); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>`where` while distinguishing
  model vocabulary from later expression and execution support.
- Existing operation foundations, concrete kind/expression families, Tensor and provenance
  contracts, architecture/ADRs/tests, backend-conformance/integration tests, and Java 26 Gradle
  configuration remain accurate unchanged. The task adds no dependency, module boundary,
  lifecycle, indexing, numerical, backend, preview/incubator, or executable behavior.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- `javap -p -c -s`, reflection tests, import/dependency scans, and source/diff searches confirmed
  exact enum machinery only and no DataType, Shape, Tensor, provenance, graph, compiler, planning,
  prepare, runtime, backend, storage, training, gradient, cost, fusion, route, registry, map,
  service, broadcast plan, result tag, or scalar-index-selection implementation.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [graph glossary](../../../../glossary.md#graph); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [fusion glossary](../../../../glossary.md#term-fusion); [registry glossary](../../../../glossary.md#term-registry))</span>
- Generated [Javadoc](../../../../glossary.md#term-javadoc) includes the new package, enum, and `WHERE` detail with complete prose. All 209
  local Markdown targets and anchors in the five changed documentation/planning files resolve,
  fence counts are balanced, changed files have no trailing whitespace, and <span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>`git diff --check`
  passes.
- Final scope contains exactly the seven authorized paths: one production file, one focused test,
  Tensor API, glossary, this task, model master plan, and roadmap. Status is synchronized as
  Complete. Task 0015F remains Draft, and no task-0015F specification exists.

## Implementation notes

- Added exactly one public parameterless enum in
  `model.operation.elementwise.selection` and one focused same-package test.
- Kept conditional `WHERE` separate from [scalar](../../../../glossary.md#term-scalar)-index `select` and all other indexing operations.
- Added no public Tensor method, attributes, inference, provenance, graph, gradient, compiler,
  backend, execution, dependency, build, or architecture behavior.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [inference glossary](../../../../glossary.md#term-inference); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- The independent documentation pass changed only the allowed Tensor API, glossary, and planning
  files; existing Javadocs required no correction.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: Implemented and documented the sole typed parameterless `WHERE`
  conditional-selection semantic kind.
- Files changed or created: `WhereSelectionKind.java`, `WhereSelectionKindTest.java`, Tensor API,
  glossary, this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused 5-test suite, all 336 model tests, generated model Javadoc, root
  tests, bytecode/reflection/import/dependency checks, documentation checks, scope/status checks,
  and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation review: The required independent clean-context pass completed in the same overall
  change. New Javadocs were already complete; Tensor API and glossary now expose the implemented
  semantic vocabulary while retaining public `Tensor.where` as planned.
- Documentation impact: Compile API, Training API, capabilities, and all other explanatory
  documents remain accurate without modification.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0015E. Plan task 0015F separately before implementation.

Status: Complete
