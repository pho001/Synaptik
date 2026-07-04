# Task 0015A: Binary Comparison Semantic Kinds

## Status

Complete

## Goal

Define the typed, backend-independent, parameterless semantic vocabulary for the six selected
Tensor-to-Tensor elementwise comparisons. The vocabulary identifies mathematical comparison
meaning and ordered operand roles without storing operands, broadcast geometry, BOOL result facts,
execution metadata, or backend support.

This task creates only the semantic family consumed by task 0015B. It does not add public Tensor
comparison methods or derive result descriptors.

## Scope

- Add one public enum `BinaryComparisonKind` implementing `OperationKind`.
- Define exactly `GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`, and
  `NOT_EQUAL` in that order.
- Document the ordered left/right mathematical meaning of every kind.
- Establish that every kind is parameterless and composes explicitly with
  `NoOperationAttrs.INSTANCE` when represented by `Operation`.
- Add one focused same-package test proving exact vocabulary, typed identity, enum behavior,
  parameterless Operation composition, and absence of metadata/state.
- Finalize affected Javadocs, Tensor API, glossary, task evidence, model master plan, and roadmap
  through the required independent documentation pass during implementation.

## Out of scope

- public `Tensor.greaterThan`, `greaterOrEqual`, `lessThan`, `lessOrEqual`, `equalTo`, or
  `notEqualTo` methods
- operands, Tensor provenance, IDs, graph values/nodes, result construction, labels, storage,
  expression chaining, or `TensorFactory.createDerived`
- input data-type eligibility, promotion, cross-type comparison, BOOL result type, broadcasting,
  layout, `requiresGrad`, or local/graph-wide inference
- scalar comparison, tolerance/epsilon, approximate equality, total ordering, or three-way
  comparison
- NaN, infinity, signed-zero, equality, ordering, or other numerical edge policy
- logical AND/OR/NOT, `where`, cast, reduction, selection, or another operation family
- family attributes, factory, registry, parser, aliases, symbols, string dispatch, reflection
  discovery, visitor, service, map, or compatibility validator
- short aliases such as `GT`, `GE`, `LT`, `LE`, `EQ`, or `NE`
- fields for arity, category, cost, fusion, result kind, differentiability, backend support,
  lowering, route, or kernel metadata
- mutable/immutable broadcast plans, strides, output shapes, or materialization requirements
- gradients, backward graph construction, autograd, optimizer, or training behavior
- compiler, planning, prepare, runtime, backend, engine, tracing, dependency, Gradle, architecture,
  or another-module changes
- changes to existing Java contracts/tests or a detailed task-0015B specification

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
- [Task 0014C](0014c-unary-elementwise-semantic-kinds.md)
- [Task 0014E](0014e-scalar-arithmetic-and-clamp-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes public Tensor methods `greaterThan`,
`greaterOrEqual`, `lessThan`, `lessOrEqual`, `equalTo`, and `notEqualTo`. Each accepts an ordered
right Tensor operand, supports right-aligned broadcasting for floating inputs, and produces a BOOL
Tensor. Legacy operation descriptors represent six corresponding comparison identities.

Legacy descriptors also retain a precomputed `BroadcastPlan` and expose arity, fusion,
semantic-family, cost, result-kind, and expression-string metadata. Legacy builders mix local
broadcasting, data-type checks, mutable graph construction, and execution-facing information.

Only the six mathematical identities and ordered operand roles are retained here. Broadcasting
and BOOL result derivation belong to task 0015B. Cost, fusion, capability, lowering, and execution
metadata belong to planning or concrete backend preparation. No legacy source, hierarchy,
broadcast plan, or naming convention is copied.

## Architecture constraints

- Production remains in `modules/model`, which owns backend-independent Operation semantics.
- `BinaryComparisonKind` is a typed semantic discriminator, not a Tensor, graph occurrence, result
  descriptor, executable operation, backend capability, or kernel route.
- The enum implements `OperationKind` through inherited `Enum.name()` and adds no duplicate name
  field or method.
- Every kind has no intrinsic parameters. Its complete attributes value is
  `NoOperationAttrs.INSTANCE`, never null or an empty map.
- Left/right order remains semantic for all six comparisons. Input identity belongs to future
  Tensor provenance and graph nodes.
- Broadcasting and BOOL output are derived from future expression inputs. They are not enum
  attributes, plans, strides, result tags, or fields.
- Numeric edge behavior, accepted input types, promotion, shape compatibility, output
  differentiability, and backend availability are not kind metadata.
- Stable enum names are diagnostic typed vocabulary, not wire tokens, registry keys, backend
  dispatch keys, kernel names, or reflection identifiers.
- Package direction is `model.operation.elementwise.comparison -> model.operation`. It must not
  depend on Tensor, datatype, shape, graph, compiler, planning, runtime, prepare, backend, storage,
  or training packages.
- Stop if implementation needs another type, attributes, Tensor behavior, inference, result facts,
  backend metadata, dependency, or architecture change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation` — supplies `OperationKind`, `Operation`, and
  `NoOperationAttrs` for the semantic contract and composition tests.

Package added:

- `io.github.pho001.synaptik.model.operation.elementwise.comparison` — owns typed parameterless
  semantic kinds for ordered Tensor-to-Tensor comparisons.

Type placement:

- `io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind` — public
  family enum under the owning operation hierarchy.
- `BinaryComparisonKindTest` — same-package focused test for vocabulary and contract shape.

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

The enum declares no project field, explicit constructor, method, nested type, per-constant class
body, symbol, alias, or metadata. Compiler-generated enum machinery is not additional project API.
Inherited `Enum.name()` satisfies `OperationKind.name()` and returns exact constant text.

| Kind | Ordered elementwise meaning |
|---|---|
| `GREATER_THAN` | left value is strictly greater than right value |
| `GREATER_OR_EQUAL` | left value is greater than or equal to right value |
| `LESS_THAN` | left value is strictly less than right value |
| `LESS_OR_EQUAL` | left value is less than or equal to right value |
| `EQUAL` | left and right values compare equal |
| `NOT_EQUAL` | left and right values compare unequal |

The table defines identity and operand order only. It does not define data-type eligibility,
promotion, broadcasting, BOOL representation, NaN/signed-zero behavior, differentiation,
execution, or backend availability.

### Parameterless Operation composition

Every kind composes explicitly as:

```java
Operation operation = new Operation(
        BinaryComparisonKind.GREATER_THAN,
        NoOperationAttrs.INSTANCE);
```

Do not add a family factory or enum `operation()` method. Generic `Operation` remains open to
caller-supplied attributes and does not enforce family compatibility.

### Naming and typed identity

- `values()` returns exactly six constants in declared order.
- `name()` returns exact uppercase constant spelling.
- Equality and hashing remain standard Java enum semantics.
- A kind from another enum remains unequal even with equal diagnostic text.
- `toString()` remains diagnostic text, not serialization or dispatch format.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/comparison/BinaryComparisonKind.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/comparison/BinaryComparisonKindTest.java`

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
- Existing operation foundations and completed kind-family Javadocs/tests.
- Tensor/expression contracts, focused architecture, ADRs, architecture tests,
  backend-conformance/integration tests, and Gradle configuration.

## Maximum scope

At most one production file, one focused test, and five documentation/planning files: seven paths
total.

Do not modify existing Java source/tests, capabilities, completed tasks, Compile/Training API,
Gradle, AGENTS, architecture documents/tests, another module, or unrelated documentation. Stop if
another production concept, attribute, expression, inference rule, dependency, or eighth path is
needed. Do not create task 0015B.

## Javadoc requirements

- Document the enum as backend-independent parameterless ordered comparison vocabulary and
  distinguish it from operands, result descriptors, graph occurrences, and executable support.
- Document every constant with left/right roles, strict versus inclusive relation where
  applicable, and deferral of eligibility, broadcasting, BOOL output, numerical edges, gradients,
  execution, and backend availability.
- Document explicit Operation composition with `NoOperationAttrs.INSTANCE` and absence of family
  compatibility validation.
- Explain diagnostic names, typed equality, and non-serialization/non-dispatch use.
- Review related Javadocs and record why they remain accurate, or stop on an inconsistency.

## Acceptance criteria

- Exactly one public enum is added in the planned comparison package.
- It implements `OperationKind` and declares exactly six constants in specified order/spelling.
- It adds no project field, method, constructor, nested type, constant body, alias, category,
  arity, cost, fusion, result, differentiability, backend, inference, or execution metadata.
- Inherited names and standard enum equality/hash/toString behavior remain.
- Every constant constructs a valid Operation with exact kind and `NoOperationAttrs.INSTANCE`; no
  factory or attributes type is added.
- Operand order and meanings are documented without stored input state.
- No broadcast plan, Shape, DataType, BOOL result, Tensor, provenance, graph, compiler, storage,
  backend, dependency, or architecture behavior is added.
- Focused/aggregate tests, Javadoc, root tests, reflection/javap/import/scope checks,
  documentation links/formatting, and statuses pass.
- A separate clean-context documentation agent finalizes Javadocs, Tensor API, glossary, evidence,
  master plan, and roadmap and records related no-change conclusions.
- Task 0015A becomes Complete only after both passes. Task 0015B remains Draft without a detailed
  specification.

## Tests / validation

Run before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKindTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover exact package/visibility/interface/constants/order/names; absence of
project fields/methods/nested types/constant bodies; standard enum identity; typed distinction from
a private test-local equal-name kind; explicit Operation composition for every constant with exact
references; and absence of attributes, operands, broadcast/result metadata, factories, registries,
reflection discovery, and dependencies.

Manually inspect `javap -p -c -s`, reflection, imports, and Gradle dependencies. Confirm no Tensor,
DataType, Shape, broadcast plan, result kind, graph/compiler/runtime/backend type, cost, fusion,
route, registry, map, or service appears. Validate generated Javadoc, Tensor API status, glossary,
links/anchors/fences/whitespace, exact seven-path scope, synchronized statuses, and absence of a
task-0015B specification.

## Dependencies

- Task 0005 supplies `OperationKind`, `OperationAttrs`, and `NoOperationAttrs.INSTANCE`.
- Task 0006 supplies immutable generic `Operation` composition and reference retention.
- Completed parameterless kind families establish conventions but are not Java dependencies.

## Follow-up tasks

- 0015B remains Draft for floating validation, promotion, broadcasting, BOOL descriptor, public
  Tensor methods, and ordered provenance.
- 0015C–0015D remain Draft for BOOL logical semantics/expressions.
- 0015E–0015F remain Draft for ternary `where` semantics/expression.
- 0015G–0015H remain Draft for typed cast semantics/expression.
- Compiler/autograd/backend/conformance tasks later own optimization, gradients, numerical edges,
  lowering, and execution.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns backend-independent Operation semantics to
`modules/model`. The new package refines that ownership without dependencies, inference, compiler
behavior, or executable state.

If implementation requires Tensor behavior, inference, broadcast state, backend metadata, another
dependency, or architecture change, stop and report it.

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

- Broad 0015 is decomposed into semantic/public-expression pairs for comparisons, BOOL logic,
  `where`, and cast so no task mixes unrelated API/result rules.
- One typed comparison family is cohesive because all six share two ordered Tensor operands,
  parameterlessness, future broadcasting, and future BOOL output.
- Descriptive constants replace legacy abbreviations. Names are diagnostics, not wire keys.
- Broadcast geometry and BOOL result facts remain in task 0015B.
- `NoOperationAttrs.INSTANCE` is complete; no empty comparison attrs record is introduced.

## Known limitations

- No public comparison expression exists until 0015B.
- Input eligibility, promotion, broadcasting, BOOL output, and `requiresGrad` are undefined here.
- NaN, infinity, signed-zero, and ordering/equality edge semantics remain unspecified.
- No compiler capture, gradient, backend support, or execution is implied.

## Validation evidence

Planning reviewed architecture/documentation rules; planning guide, capabilities, master plan,
roadmap; completed Operation foundations and family patterns through 0014F; current foundation and
kind source/tests; and read-only legacy comparison methods, descriptors, builders, broadcasting,
BOOL results, and kernel inventory.

Planning found former 0015 too broad and decomposed it into eight ordered tasks 0015A–0015H. Only
0015A has a detailed specification. It needs one enum and one test and introduces no dependency,
foundation change, result inference, or architecture change.

Planning validation:

- `git diff --check` passed, and the three changed planning files contain no trailing whitespace.
- The canonical section scan found every required task-specification section.
- The relative Markdown-target scan resolved every local `.md` link in this task, the model master
  plan, and the roadmap.
- Status inspection found task 0015A `Ready` exactly once in this specification, its model-master
  row, and its roadmap row.
- The master plan contains ordered Draft rows 0015B–0015H with explicit dependencies, and the
  roadmap sequence was expanded and renumbered consistently through task 0024.
- Scope inspection found exactly this new task plus the model master plan and roadmap changed; no
  Java, test, Gradle, AGENTS, architecture, API, glossary, or other module file changed during
  planning.
- No task-0015B specification exists; 0015B remains only a Draft queue entry.

Implementation and documentation validation:

- Implementation context `/root/implement_model_0014d` added exactly the planned enum and focused
  same-package test. Independent documentation context `/root/implement_model_0014b` then performed
  the mandatory fresh reread of the architecture contract, focused architecture pages,
  documentation and planning rules, current APIs, glossary, capability/master plans, relevant
  completed tasks, actual diff/source/tests, generated reports, bytecode, and generated Javadoc.
  It applied General plus API/Javadoc style to the Java and Tensor API review, Planning style to
  planning files, and Example format to the explicit Operation-composition example.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKindTest`
  passed. An explicit fresh `--rerun-tasks` execution produced 5 tests with zero failures, errors,
  or skips.
- `./gradlew :modules:model:test` passed; 41 XML suites contain 308 tests with zero failures,
  errors, or skips. The model test task executed in the recorded final aggregate run.
- `./gradlew :modules:model:javadoc` passed, and an explicit fresh `--rerun-tasks` execution
  regenerated clean public Javadoc. The generated type page contains the backend-independent,
  parameterless, ordered-family contract, explicit `Operation`/`NoOperationAttrs.INSTANCE`
  composition, family-compatibility limitation, typed diagnostic naming, and all six constant
  meanings and deferrals. The all-classes, index, and type-search pages contain the new type and
  constants.
- `./gradlew test` passed for the repository; its final run reported all 36 actionable tasks
  up-to-date and no failing task. The model suite immediately preceding it was freshly executed.
- `javap -p -c -s` confirms the exact six enum constants in specified order plus only compiler-
  generated enum machinery. Reflection tests confirm the public/final enum shape, sole
  `OperationKind` interface, compiler constructor, no project fields/methods/nested types or
  constant bodies, inherited enum behavior, exact names, and typed distinction from an equally
  named test-local kind.
- Composition tests construct every kind with `NoOperationAttrs.INSTANCE` and confirm exact
  kind/attributes reference retention through generic `Operation`. They do not add or imply a
  family factory or compatibility validator.
- Production has exactly one import, `OperationKind`. Source, bytecode, dependency, and diff scans
  found no operands, attributes type, Tensor, DataType, Shape, BOOL result fact, broadcast plan,
  provenance, graph/compiler/planning/prepare/runtime/backend type, gradient, cost, fusion, route,
  registry, map, service, dependency, build, or executable behavior.
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
  backend-conformance test, integration test, or task-0015B specification path changed.
- Task 0015A is synchronized as Complete in this specification, the model master plan, and the
  roadmap. Task 0015B remains the next Draft frontier without a detailed specification.
- Existing `OperationKind`, `OperationAttrs`, `NoOperationAttrs`, `Operation`, binary arithmetic,
  unary, scalar, Tensor, descriptor, provenance, and expression Javadocs remain accurate because
  the enum implements and composes existing contracts without changing validation, attributes,
  public Tensor methods, descriptors, provenance, or expression behavior.
- The Compile API requires no edit because a semantic enum adds no public comparison expression,
  capture, inference, optimization, artifact, or execution contract. The Training API requires no
  edit because no `requiresGrad` rule, gradient rule, autograd, optimizer, or training behavior
  changed. `capabilities.md` requires no edit because it already selects all six comparisons and
  distinguishes model semantics/public expressions from compiler and execution support.
- `ARCHITECTURE.md`, focused architecture pages, ADRs, architecture tests, backend-conformance and
  integration tests, and build configuration require no edit because the change remains within
  the existing model-owned operation-semantics boundary and adds no dependency, module boundary,
  lifecycle, backend, numerical-execution, Java-toolchain, preview/incubator, or end-to-end rule.

## Implementation notes

- Added exactly one public `BinaryComparisonKind` enum with `GREATER_THAN`, `GREATER_OR_EQUAL`,
  `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`, and `NOT_EQUAL` in the specified order and one focused
  same-package test.
- The enum is typed semantic identity only. It stores no operands, attributes, broadcast/result
  facts, execution metadata, or backend capability, and generic `Operation` remains open and
  family-agnostic.
- The independent documentation pass added the current comparison semantic family, exact ordered
  meanings, explicit parameterless composition, diagnostic-name distinction, and planned limits
  to the Tensor API and existing glossary concepts. No new reusable glossary term was needed.
- No Java declaration, Javadoc, executable logic, or test changed during the documentation pass.

## Completion summary

- Completed changes: Implemented and documented the six typed, parameterless, ordered binary
  comparison semantic kinds.
- Files changed or created: Exactly one production file, one focused test, Tensor API, glossary,
  this task, model master plan, and roadmap.
- Tests and validation: Focused comparison semantics 5/5, all 308 model tests across 41 suites,
  regenerated model Javadoc, root tests, bytecode/reflection/import/dependency/absence checks,
  local links/anchors, fences, terminology, whitespace, exact scope/status, and `git diff --check`
  passed.
- Documentation-agent review: Clean documentation context `/root/implement_model_0014b` completed
  the independent pass using General, API/Javadoc, Planning, and Example-format guidance.
- Documentation impact: Tensor API and glossary now describe the current comparison semantic
  vocabulary while preserving public Tensor comparison expressions and all later lifecycle stages
  as planned. Compile API, Training API, and capabilities remain accurate unchanged.
- Javadoc review: The new enum type and all six constants are complete unchanged during the
  documentation pass; existing operation foundation, concrete family, Tensor, descriptor,
  provenance, and expression contracts remain accurate unchanged.
- Glossary impact: Existing implementation-status, `OperationKind`, and common-distinction entries
  now include the comparison family; no new reusable domain term was needed.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0015A. Task 0015B remains Draft without a specification.

Status: Complete
