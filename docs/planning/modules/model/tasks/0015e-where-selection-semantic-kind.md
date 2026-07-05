# Task 0015E: Where Selection Semantic Kind

## Status

Complete

## Goal

Define the typed, backend-independent, parameterless semantic identity for elementwise conditional
selection: `WHERE`.

The operation means that one condition value chooses between corresponding true-branch and
false-branch values. This task creates only that meaning. It does not add the public three-Tensor
expression, validate a BOOL condition or floating branches, broadcast three shapes, derive a
result descriptor, record provenance, calculate selected values, or define executable support.

## Scope

- Add one public enum `WhereSelectionKind` implementing `OperationKind`.
- Define exactly one constant, `WHERE`.
- Document the three ordered logical input roles: condition, true branch, and false branch.
- Establish that `WHERE` has no intrinsic attributes and composes explicitly with
  `NoOperationAttrs.INSTANCE` when represented by `Operation`.
- Document ternary family context without storing or validating arity.
- Distinguish conditional `WHERE` from scalar-index selection and other indexing operations.
- Add one focused same-package test proving exact vocabulary, typed identity, enum behavior,
  parameterless Operation composition, and absence of metadata/state.
- Add the cohesive `model.operation.elementwise.selection` package to the model package map.
- Finalize affected Javadocs, Tensor API, glossary, task evidence, model master plan, and roadmap
  through the required independent documentation pass during implementation.

## Out of scope

- public `Tensor.where(Tensor, Tensor, Tensor)` or another Tensor method, overload, factory, or
  expression-builder type
- Tensor inputs, provenance, IDs, graph values/nodes, result construction, labels, storage,
  expression chaining, or `TensorFactory.createDerived`
- DataType imports, BOOL condition validation, floating branch validation or promotion, result
  data type, three-way broadcasting, output shape, layout, `requiresGrad`, or inference
- eager branch evaluation, value selection, condition storage encoding, short-circuit behavior,
  lazy branch evaluation, constant folding, or numerical edge policy
- gradient eligibility, branch gradient routing, backward graph construction, autograd, optimizer,
  or training behavior
- family attributes, factory, registry, parser, alias, symbol, string dispatch, reflection
  discovery, visitor, service, map, or compatibility validator
- aliases such as `SELECT`, `CHOOSE`, `CONDITIONAL`, or `IF_THEN_ELSE`
- fields for arity, category, condition type, branch/result type, cost, fusion, differentiability,
  backend support, lowering, route, kernel, or execution metadata
- a broadcast plan, effective strides, output geometry, materialization requirement, or backend
  storage facts
- scalar-index `select`, gather, take, scatter, masking, reduction, cast, or another operation
  family
- compiler, planning, prepare, runtime, backend, engine, tracing, ONNX, dependency, Gradle,
  architecture, or another-module changes
- changes to existing Java contracts/tests or a detailed task-0015F specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0014A](0014a-binary-arithmetic-semantic-kinds.md)
- [Task 0015A](0015a-binary-comparison-semantic-kinds.md)
- [Task 0015C](0015c-boolean-logical-semantic-kinds.md)
- [Task 0015D](0015d-boolean-logical-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes static
`Tensor.where(condition, ifTrue, ifFalse)`. Its public contract uses a BOOL condition, two floating
branches, common three-way broadcasting, floating branch promotion, and the ordered rule “true
selects `ifTrue`, false selects `ifFalse`.” Legacy tests exercise comparison and logical masks,
broadcasting, mixed floating branch types, non-contiguous inputs, fused chains, autograd, ONNX,
and several backend routes.

The legacy semantic descriptor also exposes family, cost, fusion, result-kind, expression-string,
lowering, broadcast-plan, and execution-facing facts. Those mechanisms are not copied. This task
retains only the `WHERE` identity and the condition/true-branch/false-branch roles. Task 0015F owns
public Tensor validation, branch promotion, three-way broadcasting, result construction, and
ordered provenance. Compiler, training, planning, prepare, and concrete backends retain their own
responsibilities.

## Architecture constraints

- Production remains in `modules/model`, which owns backend-independent Operation semantics.
- `WhereSelectionKind` is a typed semantic discriminator, not a Tensor, input list, graph
  occurrence, result descriptor, executable operation, backend capability, or kernel route.
- The enum implements `OperationKind` through inherited `Enum.name()` and adds no duplicate name
  field or method.
- `WHERE` has no intrinsic parameters. Its complete attributes value is
  `NoOperationAttrs.INSTANCE`, never null or an empty map.
- The operation has three logical input roles in exact order: condition, true branch, false branch.
  This ternary context is documentation, not stored arity metadata. Input identities belong to
  later Tensor provenance and graph nodes.
- The meaning chooses the true-branch value when the corresponding condition is true and the
  false-branch value otherwise. It does not prescribe eager or lazy branch evaluation.
- BOOL condition eligibility, branch data-type eligibility and promotion, broadcasting, result
  descriptor facts, gradient eligibility, and differentiation are not enum fields.
- `WHERE` conditional selection is separate from scalar-index `select`, gather, take, and scatter.
  Those operations remain under the later indexing/scatter task.
- Stable enum names are diagnostic typed vocabulary, not ONNX tokens, wire values, registry keys,
  backend dispatch keys, kernel names, operators, or reflection identifiers.
- Package direction is `model.operation.elementwise.selection -> model.operation`. It must not
  depend on Tensor, datatype, shape, graph, compiler, planning, runtime, prepare, backend, storage,
  or training packages.
- Stop if implementation needs another type, attributes, Tensor behavior, descriptor inference,
  a broadcast plan, backend metadata, dependency, or architecture change.

## Package impact

Existing package used:

- `io.github.pho001.synaptik.model.operation` — supplies `OperationKind`, `Operation`, and
  `NoOperationAttrs` for the semantic contract and composition tests.

Package added:

```text
io.github.pho001.synaptik.model.operation.elementwise.selection
  Typed parameterless semantics for elementwise conditional branch selection.
```

The package is deliberately below `elementwise`: `WHERE` makes one independent choice at each
broadcasted result position. The `selection` name describes semantic conditional selection, while
the parent distinguishes it from later scalar-index selection and indexing/scatter operations.

Type placement:

- `io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind` — public
  family enum under the owning operation hierarchy.
- `WhereSelectionKindTest` — same-package focused test for vocabulary and contract shape.

## Required contract

### Enum declaration

Create exactly:

```java
public enum WhereSelectionKind implements OperationKind {
    WHERE
}
```

The enum declares no project field, explicit constructor, method, nested type, per-constant class
body, symbol, alias, or metadata. Compiler-generated enum machinery is not additional project API.
Inherited `Enum.name()` satisfies `OperationKind.name()` and returns exact constant text.

| Kind | Elementwise meaning | Logical input roles |
|---|---|---|
| `WHERE` | choose the true-branch value when the corresponding condition is true; otherwise choose the false-branch value | ordered condition, true branch, false branch |

The table defines semantic identity and roles only. It does not define public Tensor arity
validation, accepted descriptors, broadcasting, branch promotion, output descriptor construction,
storage behavior, evaluation order, differentiation, execution, or backend availability.

### Parameterless Operation composition

The kind composes explicitly as:

```java
Operation operation = new Operation(
        WhereSelectionKind.WHERE,
        NoOperationAttrs.INSTANCE);
```

Do not add a family factory, enum `operation()` method, attributes type, or compatibility registry.
Generic `Operation` remains an open typed pairing and does not enforce family-specific arity or
kind-to-attributes compatibility.

### Naming and typed identity

- `values()` returns exactly one constant, `WHERE`.
- `name()` returns exact uppercase text `WHERE`.
- Equality and hashing remain standard Java enum semantics.
- A kind from another enum remains unequal even if it also has the diagnostic name `WHERE`.
- `toString()` remains diagnostic text, not serialization, parsing, ONNX mapping, or dispatch
  format.
- No `SELECT` alias is added because scalar-index selection is a distinct operation capability.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/selection/WhereSelectionKind.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/selection/WhereSelectionKindTest.java`

Documentation/planning during implementation:

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
  Javadocs/tests.
- Focused architecture documentation, ADRs, architecture tests, backend-conformance material,
  integration tests, and Gradle configuration.

## Maximum scope

At most one production file, one focused test, and five documentation/planning files: seven paths
total.

Do not modify existing Java source/tests, capabilities, completed tasks, Compile/Training API,
Gradle, AGENTS, architecture documents/tests, another module, or unrelated documentation. Stop if
another production concept, attributes, Tensor expression, descriptor rule, dependency, or eighth
path is needed. Do not create task 0015F.

## Javadoc requirements

- Document the enum as backend-independent parameterless elementwise conditional-selection
  vocabulary and distinguish it from inputs, result descriptors, graph occurrences, indexing
  selection, and executable support.
- Document `WHERE` with the exact ordered condition, true-branch, and false-branch roles and explain
  which branch value is selected for true and false condition values.
- Explain that ternary family context is not stored arity metadata and is not validated by generic
  `Operation`.
- Document explicit Operation composition with `NoOperationAttrs.INSTANCE` and absence of family
  compatibility validation.
- Defer condition/branch descriptor eligibility, branch promotion, three-way broadcasting, result
  descriptor, evaluation order, gradients, execution, ONNX mapping, and backend availability.
- Explain diagnostic names, typed equality, and non-serialization/non-dispatch use.
- Review related Javadocs and record why they remain accurate, or stop on an inconsistency.

## Acceptance criteria

- Exactly one public `WhereSelectionKind` enum is added in the planned selection package.
- It implements `OperationKind` and declares exactly `WHERE`.
- It adds no project field, method, constructor, nested type, constant body, alias, attributes,
  arity, category, cost, fusion, result, differentiability, backend, inference, or execution
  metadata.
- Inherited name and standard enum equality/hash/toString behavior remain.
- `WHERE` constructs a valid Operation with the exact kind and `NoOperationAttrs.INSTANCE`; no
  factory or attributes type is added.
- The exact three logical input roles and true/false choice meaning are documented without stored
  inputs or arity state.
- Production imports only `OperationKind`; no DataType, Shape, Tensor, provenance, graph, compiler,
  planning, runtime, prepare, backend, storage, training, dependency, or architecture behavior is
  added.
- No broadcast plan, branch/result descriptor, numerical execution, gradient rule, public Tensor
  method, scalar-index selection, or ONNX/backend mapping is added.
- Focused/aggregate tests, Javadoc, root tests, reflection/javap/import/scope checks,
  documentation links/formatting, and statuses pass.
- A separate clean-context documentation agent finalizes Javadocs, Tensor API, glossary, evidence,
  master plan, and roadmap and records related no-change conclusions.
- Task 0015E becomes Complete only after both passes. Task 0015F remains Draft without a detailed
  specification.

## Tests / validation

Run before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKindTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover exact package/visibility/interface/constant/name; absence of project
fields/methods/nested types/constant bodies; standard enum identity; typed distinction from a
private test-local equal-name kind; explicit Operation composition with exact references; ternary
family vocabulary without arity state; and absence of attributes, inputs, descriptors, factories,
registries, reflection discovery, and dependencies.

Manually inspect `javap -p -c -s`, reflection, imports, and Gradle dependencies. Confirm no
DataType, Shape, Tensor, provenance, broadcast plan, result tag, graph/compiler/runtime/backend
type, gradient, cost, fusion, route, registry, map, or service appears. Validate generated Javadoc,
Tensor API status, glossary, links/anchors/fences/whitespace, exact seven-path scope, synchronized
statuses, package-map placement, and absence of a task-0015F specification.

## Dependencies

- Task 0005 supplies `OperationKind`, `OperationAttrs`, and `NoOperationAttrs.INSTANCE`.
- Task 0006 supplies immutable generic `Operation` composition and reference retention.
- Completed parameterless kind families establish enum and test conventions but are not Java
  dependencies.
- Task 0015D supplies the BOOL-producing public expression seam useful to future conditions; it is
  implementation-order context rather than a Java dependency of this enum.

## Follow-up tasks

- 0015F remains Draft for exact public `Tensor.where` shape, BOOL condition validation, floating
  branch validation and promotion, three-way broadcasting, result descriptor construction,
  ordered provenance, and storage-free derived Tensor creation.
- 0015G–0015H remain Draft for typed cast semantics and expression construction.
- Indexing/scatter tasks later own scalar-index `select`, gather, take, and functional scatter.
- Compiler, training, backend, ONNX, and conformance tasks later own capture, differentiation,
  optimization, mapping, lowering, selection execution, and storage interpretation.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns backend-independent Operation semantics to
`modules/model`. The new package refines that ownership without dependencies, result inference,
compiler behavior, storage, or executable state.

If implementation requires Tensor behavior, descriptor inference, backend metadata, another
dependency, or architecture change, stop and report it.

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
- The package is `operation.elementwise.selection`. The `elementwise` parent makes conditional
  branch selection distinct from the future indexing/scatter hierarchy and its scalar-index
  `select` capability.
- The sole constant is `WHERE`; no `SELECT` alias is introduced because those words name different
  capabilities in the selected baseline.
- Ternary arity and condition/branch roles are documentation, not fields. Input identities and
  local validation belong to task 0015F.
- `NoOperationAttrs.INSTANCE` completely represents the absence of intrinsic parameters; an empty
  where-attributes record would carry no additional semantic fact.
- Evaluation order is intentionally unspecified. The model identity expresses elementwise choice,
  not eager/lazy host-language control flow.

## Known limitations

- No public `Tensor.where` expression exists until task 0015F.
- Condition and branch eligibility, branch promotion, three-way broadcasting, output descriptor,
  layout, and `requiresGrad` behavior are undefined here.
- The enum does not enforce three inputs or kind-to-attributes compatibility.
- No compiler capture, differentiation, ONNX mapping, backend support, storage interpretation, or
  execution is implied.

## Validation evidence

Planning reviewed the architecture contract and focused module/dependency/lifecycle explanations;
documentation and planning rules; roadmap; model capabilities and master plan; tasks 0005, 0006,
0014A, 0015A, 0015C, and 0015D; current Operation foundations and concrete kind-family source/tests;
Tensor/Compile/Training APIs and glossary; and Java 26 Gradle configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms static public
`Tensor.where(condition, ifTrue, ifFalse)`, the parameterless `WHERE` semantic identity, ordered
condition/true-branch/false-branch roles, BOOL condition validation, floating branch promotion,
three-way broadcasting, gradient routing, ONNX mapping, fused chains, and multiple backend routes.
Legacy broadcast plans, operation traits, expression strings, autograd machinery, storage access,
kernels, lowering, and execution are excluded from this semantic-only task.

Planning selected one enum and one focused test because `WHERE` has one stable parameterless
semantic identity. A separate attributes record, arity field, broadcast plan, or input container
would add no intrinsic operation parameter. Public validation and result construction remain in
task 0015F. No existing dependency, foundation contract, or architecture rule changes.

Planning validation:

- `git diff --check` passed, and targeted whitespace inspection found no trailing whitespace in
  the three changed planning paths.
- The required-section scan found every canonical task-specification section, including package
  impact, bounded scope, validation, implementation handoff, decisions, limitations, and
  completion-evidence sections.
- Every local Markdown file link in this task, the model master plan, and the roadmap resolves.
  Markdown fence counts are balanced.
- Status inspection found 0015E `Ready` in this specification, its linked model-master row, and
  its linked roadmap row/current-frontier text. Task 0015F remains `Draft` in both queues.
- Package-map inspection found exactly one new planned package,
  `model.operation.elementwise.selection`, with direction only to `model.operation`.
- Scope inspection found exactly this new task, the model master plan, and the roadmap changed. No
  Java, test, API, glossary, Gradle, architecture, AGENTS, or other-module path changed.
- No task-0015F specification exists.

Implementation and independent documentation validation:

- Clean implementation work added exactly `WhereSelectionKind.java` and its focused same-package
  test. The enum has the sole `WHERE` constant, implements `OperationKind`, imports only that
  interface, and declares no project field, method, constructor, nested type, constant body,
  attributes, alias, arity state, or metadata.
- The focused test passed with 5 tests and zero failures, errors, or skips. It covers exact
  vocabulary and enum shape, inherited diagnostic behavior, typed family distinction, and exact
  `Operation` composition with `NoOperationAttrs.INSTANCE`.
- An explicit fresh aggregate model run passed with 336 tests and zero failures, errors, or skips.
  Model Javadoc and root tests also passed.
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
  descriptor/compiler/training/execution concern. No Java or test edit was needed.
- Tensor API now lists `WhereSelectionKind`, explains its sole conditional-selection identity and
  explicit composition, distinguishes it from scalar-index `select`, and keeps public
  `Tensor.where` and its validation/inference/provenance behavior planned. The glossary records the
  same implemented-versus-planned boundary without creating a redundant reusable term.
- Compile API remains accurate unchanged because no public expression, compiler capture,
  inference, transformation, artifact, or execution contract was added. Training API remains
  accurate unchanged because no gradient, autograd, optimizer, or training behavior changed.
  Capabilities remain accurate because they already select public `where` while distinguishing
  model vocabulary from later expression and execution support.
- Existing operation foundations, concrete kind/expression families, Tensor and provenance
  contracts, architecture/ADRs/tests, backend-conformance/integration tests, and Java 26 Gradle
  configuration remain accurate unchanged. The task adds no dependency, module boundary,
  lifecycle, indexing, numerical, backend, preview/incubator, or executable behavior.
- `javap -p -c -s`, reflection tests, import/dependency scans, and source/diff searches confirmed
  exact enum machinery only and no DataType, Shape, Tensor, provenance, graph, compiler, planning,
  prepare, runtime, backend, storage, training, gradient, cost, fusion, route, registry, map,
  service, broadcast plan, result tag, or scalar-index-selection implementation.
- Generated Javadoc includes the new package, enum, and `WHERE` detail with complete prose. All 209
  local Markdown targets and anchors in the five changed documentation/planning files resolve,
  fence counts are balanced, changed files have no trailing whitespace, and `git diff --check`
  passes.
- Final scope contains exactly the seven authorized paths: one production file, one focused test,
  Tensor API, glossary, this task, model master plan, and roadmap. Status is synchronized as
  Complete. Task 0015F remains Draft, and no task-0015F specification exists.

## Implementation notes

- Added exactly one public parameterless enum in
  `model.operation.elementwise.selection` and one focused same-package test.
- Kept conditional `WHERE` separate from scalar-index `select` and all other indexing operations.
- Added no public Tensor method, attributes, inference, provenance, graph, gradient, compiler,
  backend, execution, dependency, build, or architecture behavior.
- The independent documentation pass changed only the allowed Tensor API, glossary, and planning
  files; existing Javadocs required no correction.

## Completion summary

- Completed changes: Implemented and documented the sole typed parameterless `WHERE`
  conditional-selection semantic kind.
- Files changed or created: `WhereSelectionKind.java`, `WhereSelectionKindTest.java`, Tensor API,
  glossary, this task, model master plan, and roadmap.
- Tests and validation: Focused 5-test suite, all 336 model tests, generated model Javadoc, root
  tests, bytecode/reflection/import/dependency checks, documentation checks, scope/status checks,
  and `git diff --check` passed.
- Documentation review: The required independent clean-context pass completed in the same overall
  change. New Javadocs were already complete; Tensor API and glossary now expose the implemented
  semantic vocabulary while retaining public `Tensor.where` as planned.
- Documentation impact: Compile API, Training API, capabilities, and all other explanatory
  documents remain accurate without modification.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0015E. Plan task 0015F separately before implementation.

Status: Complete
