# Task 0015G: Cast Semantic Kind and Attributes

## Status

Complete

## Goal

Define the typed, backend-independent semantic identity and immutable target-data-type parameter
for an explicit elementwise cast. The operation family must represent one `CAST` request paired
with one non-null target `DataType`, without retaining an input Tensor, constructing a result
descriptor, converting a value, choosing numerical conversion policy, or reporting backend
support.

This task creates the complete semantic foundation consumed by task 0015H. It does not add the
public `Tensor.cast(DataType)` expression method.

## Scope

- Add one public `CastKind` enum implementing `OperationKind`.
- Define exactly one constant, `CAST`.
- Add one public `CastAttrs` record implementing `OperationAttrs` with exactly one
  `DataType targetDataType` component.
- Require a non-null target with exact `NullPointerException("targetDataType")` behavior.
- Accept and retain every current `DataType`: `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, `INT64`,
  and `BOOL`.
- Define `CastKind.CAST` plus `CastAttrs` as the valid family composition through typed
  documentation without changing generic `Operation` validation.
- Document one logical input and elementwise target-type conversion as family context without
  storing arity, source type, input identity, result descriptor, or conversion rules.
- Add one focused same-package test covering exact vocabulary, attributes, composition, record
  value semantics, typed identity, and exclusions.
- Add the cohesive `model.operation.elementwise.cast` package to the model package map.
- Finalize affected Javadocs, Tensor API, glossary, task evidence, model master plan, and roadmap
  through the required independent documentation pass during implementation.

## Out of scope

- public `Tensor.cast(DataType)`, another Tensor method, overload, factory, builder, or expression
  helper
- input Tensor, source data type, shape, layout, result descriptor, label, identity, storage,
  provenance, expression chaining, or `TensorFactory.createDerived`
- source/target compatibility rules, same-type identity elimination, fresh-result policy, shape
  preservation implementation, result `requiresGrad`, or local/graph-wide inference
- numerical conversion behavior, rounding, truncation, saturation, overflow, underflow, precision,
  NaN, infinity, signed zero, BFLOAT16 encoding, BOOL zero/non-zero semantics, or invalid-value
  policy
- eager value conversion, storage reads or writes, allocation, copy, materialization, aliasing,
  constant folding, canonicalization, or cast-chain simplification
- gradient eligibility, floating-to-floating gradient rules, non-floating gradient barriers,
  backward graph construction, autograd, optimizer, or training behavior
- an attributes hierarchy, source-and-target pair record, conversion-mode enum, rounding-mode enum,
  optional target, boxed/string target, registry, parser, compatibility matrix, visitor, or service
- aliases such as `CONVERT`, `AS_TYPE`, `TO_DTYPE`, or one enum constant per source/target pair
- fields on `CastKind` for arity, category, source/target type, cost, fusion, differentiability,
  backend support, lowering, route, kernel, or execution metadata
- changes to `DataType`, `Operation`, `OperationKind`, `OperationAttrs`, `NoOperationAttrs`, existing
  operation families, Tensor, graph records, or existing Java tests
- compiler, planning, prepare, runtime, backend, engine, tracing, ONNX, dependency, Gradle,
  architecture, or another-module changes
- a detailed task-0015H specification

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
- [Task 0001](0001-data-type-model.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0014E](0014e-scalar-arithmetic-and-clamp-semantics.md)
- [Task 0015E](0015e-where-selection-semantic-kind.md)
- [Task 0015F](0015f-where-selection-tensor-expression.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes fluent `Tensor.cast(DataType targetType)`. Its
semantic descriptor retains one non-null target data type, while the input Tensor descriptor
provides the source type. Legacy execution evidence covers floating precision changes, BFLOAT16,
integral/floating conversion, BOOL/floating conversion, strided inputs, expression chains, ONNX,
CPU, and Metal routes. The legacy public builder also treats a same-type cast as an identity and
installs a gradient callback only for floating-to-floating conversions.

The legacy descriptor additionally exposes family, arity, cost, fusion, result-kind,
expression-string, and backend-facing facts. Those mechanisms are not copied. This task retains
only the explicit `CAST` meaning and target `DataType`. Task 0015H owns public Tensor construction,
result descriptor facts, same-type behavior, and gradient-eligibility policy. Compiler, backend,
ONNX, and conformance work later own canonicalization, differentiation, mapping, conversion
semantics, lowering, storage access, and execution.

## Architecture constraints

- Operation kinds and attributes are immutable backend-independent model semantics owned by
  `modules/model`.
- `CastKind` identifies the requested conversion family. It stores no parameter, input, source
  type, graph occurrence, result fact, executable behavior, or backend information.
- `CastAttrs` is the complete attributes value for `CastKind.CAST`; its sole semantic parameter is
  the exact non-null target `DataType`.
- The source data type belongs to the later input Tensor or graph value descriptor. Duplicating it
  in attributes would create two sources of truth and is forbidden.
- Family documentation defines the valid `CAST`/`CastAttrs` pairing. Generic `Operation` continues
  to validate only non-null components and must not gain family discovery or compatibility checks.
- All six current `DataType` values are valid targets. This representability does not promise that
  every backend implements every source/target conversion route.
- A target equal to a future input source type remains a valid semantic parameter. Whether public
  expression construction returns its input, creates an explicit node, or leaves identity removal
  to compiler optimization is not decided here.
- One logical input and elementwise conversion are family context rather than stored arity or
  shape metadata. No source value or conversion is performed.
- Conversion rounding, overflow, truth, special-value, and gradient policies are not attributes
  fields and are not implied by `DataType` retention.
- Stable enum names and record text are diagnostic, not serialization, ONNX, registry, reflection,
  backend-dispatch, or kernel identifiers.
- Package direction is `model.operation.elementwise.cast -> model.operation, model.datatype`. It
  must not depend on Tensor, shape, layout, storage, graph, compiler, planning, runtime, prepare,
  backend, or training packages.
- Stop if implementation requires another attributes component/type, a Tensor or result contract,
  conversion policy, compatibility validator, dependency, or architecture change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation` — supplies `OperationKind`, `OperationAttrs`, and
  generic `Operation` composition.
- `io.github.pho001.synaptik.model.datatype` — supplies the immutable target `DataType` vocabulary.

Package added:

```text
io.github.pho001.synaptik.model.operation.elementwise.cast
  Typed explicit data-type conversion semantics and immutable target-type attributes.
```

The package is below `elementwise` because one cast request converts corresponding logical
elements independently and has one logical input. The `cast` leaf separates conversion parameters
from arithmetic, comparison, logical, and selection families without introducing a broad dtype
utility package.

Type placement:

- `io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind` — public semantic family
  enum.
- `io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs` — public immutable target
  data-type parameter.
- `CastSemanticsTest` — same-package focused test for both cohesive contracts.

## Required contract

### Semantic kind vocabulary

Create exactly:

```java
public enum CastKind implements OperationKind {
    CAST
}
```

The enum declares no project field, explicit constructor, method, nested type, per-constant class
body, alias, symbol, or metadata. Compiler-generated enum machinery is not additional project API.
Inherited `Enum.name()` satisfies `OperationKind.name()` and returns exact text `CAST`.

`CAST` means explicit elementwise conversion of one logical input to the target data type carried
by `CastAttrs`. It does not define the source type, result descriptor, numerical conversion,
gradient behavior, execution, or backend availability.

### Target attributes

Create exactly:

```java
public record CastAttrs(DataType targetDataType) implements OperationAttrs {
    public CastAttrs {
        targetDataType = Objects.requireNonNull(targetDataType, "targetDataType");
    }

    @Override
    public DataType targetDataType() {
        return targetDataType;
    }
}
```

The record has exactly one component and no additional instance state, static state, nested type,
overload, factory, or helper. Construction validates only non-null presence and then retains the
exact enum reference through ordinary record assignment. Null fails with
`NullPointerException("targetDataType")`.

The explicit accessor exists for complete Javadoc and returns the exact stored non-null reference.
All six current targets are accepted without source knowledge, normalization, defaulting, or
capability lookup.

Record-generated equality and hashing use the target enum value. Equal targets produce equal
records and equal hashes; different targets produce unequal records. Generated diagnostic text is
not a serialization, parser, ONNX, backend, or dispatch contract.

### Operation composition

Every target composes explicitly as:

```java
CastAttrs attrs = new CastAttrs(DataType.FLOAT32);
Operation operation = new Operation(CastKind.CAST, attrs);
```

`Operation` retains the exact kind and attributes references. Do not use
`NoOperationAttrs.INSTANCE`: target data type is an intrinsic semantic parameter. Do not add a
family factory, enum `operation()` method, compatibility map, or generic-operation validation.

### Naming and typed identity

- `values()` returns exactly one constant, `CAST`.
- Equality and hashing of `CastKind` remain standard Java enum behavior.
- A private test-local or future family constant named `CAST` remains a different typed value.
- No source/target-pair constants or aliases are introduced.
- Enum and record text remain diagnostics and must not be parsed or used for dispatch.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/cast/CastKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/cast/CastAttrs.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/cast/CastSemanticsTest.java`

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
- Existing `DataType`, `OperationKind`, `OperationAttrs`, `Operation`, `NoOperationAttrs`, and
  concrete kind/attributes family Javadocs/tests.
- Tensor and Tensor expression contracts, focused architecture documents, ADRs, architecture
  tests, backend-conformance tests, integration tests, and Gradle configuration.

## Maximum scope

At most two production files, one focused test, and five documentation/planning files: eight paths
total.

No existing Java source or test may change. Do not modify DataType, Operation foundations, Tensor,
capabilities, Compile/Training API, Gradle, AGENTS, architecture documents/tests, another module,
or unrelated documentation. Stop if implementation requires another production/test concept,
source-type state, conversion policy, factory/validator, another documentation file, or a ninth
path. Do not create task 0015H.

## Javadoc requirements

- Document `CastKind` as backend-independent parameterized elementwise conversion vocabulary and
  distinguish it from input, source type, target attributes, result descriptor, graph occurrence,
  numerical conversion, and executable support.
- Document `CAST` with its one-input elementwise meaning, required `CastAttrs`, and explicit
  deferral of source/target compatibility, same-type behavior, result inference, numerical policy,
  gradients, execution, and backend availability.
- Document the valid kind/attributes pairing and state that generic `Operation` does not enforce it.
- Document `CastAttrs` with its exact target role, immutable ownership, all accepted current
  `DataType` values, null validation, record equality/hashing, and diagnostic-only text.
- Document the canonical constructor with `@param targetDataType`, exact non-null constraint,
  reference retention, and `@throws NullPointerException` message.
- Document the explicit accessor with a non-null `@return` describing exact stored-reference
  retention and absence of source knowledge, conversion, or capability validation.
- Explain why source data type is not duplicated and why target representability does not promise
  backend support or define conversion details.
- Review related foundational Javadocs and record why they remain accurate, or stop on an
  out-of-scope inconsistency.

## Acceptance criteria

- Exactly one public `CastKind` enum and one public `CastAttrs` record are added in the planned
  cast package.
- The enum implements `OperationKind`, declares exactly `CAST`, and adds no project field, method,
  nested type, alias, constant body, category, arity, cost, fusion, result, backend, or execution
  metadata.
- `CastAttrs` implements `OperationAttrs` and has exactly one `DataType targetDataType` component,
  one canonical constructor, one explicit component accessor, and no additional state or API.
- Null target fails with exact `NullPointerException` message `targetDataType`.
- Every current `DataType` is accepted, retained by exact reference, and returned unchanged.
- Equal targets produce equal records/equal hash codes; different targets produce unequal records;
  diagnostic text is not treated as serialization or dispatch.
- `CAST` constructs a valid `Operation` with every target attributes value, retaining exact kind
  and attributes references. No factory or family compatibility validator is added.
- `CAST` remains typed-distinct from an equally named kind in another enum.
- No source type, Tensor, shape, layout, descriptor, provenance, storage, numerical conversion,
  same-type policy, gradient, graph, compiler, planning, runtime, backend, dependency, or
  architecture behavior is added.
- Focused and aggregate model tests, model Javadoc, root tests, reflection/javap/import/scope
  checks, documentation links/formatting, and status synchronization pass.
- A separate clean-context documentation-focused agent finalizes new Javadocs, Tensor API,
  glossary, task evidence, model master plan, and roadmap in the same change and records reasoned
  no-change conclusions for Compile API, Training API, capabilities, architecture, and related
  contracts.
- Task 0015G becomes Complete only after both passes. Task 0015H remains Draft without a detailed
  specification.

## Tests / validation

Run before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.elementwise.cast.CastSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover:

- exact package, public/final enum and record shapes, interfaces, record component order/type,
  constructor parameters, fields, methods, and absence of nested types;
- exact sole enum constant, inherited stable name, standard enum identity, and typed distinction
  from a private test-local equal-name kind;
- null target with exact exception type/message before record construction;
- exact reference retention and accessor results for all six current `DataType` targets;
- generated equality, hashing, and diagnostic text for equal/different targets without treating
  text as serialization;
- explicit `Operation` composition for every target with exact kind/attributes references;
- target-only attributes with no source type, Tensor input, descriptor, conversion mode, factory,
  registry, compatibility map, or dependency; and
- distinction from `NoOperationAttrs.INSTANCE` and parameterless families.

Manually inspect `javap -p -c -s` and reflection for exact enum/record shape, component order,
constructor null validation, explicit accessor, and absence of extra project API/state. Scan
production imports and Gradle dependencies: `CastKind` may import only `OperationKind`; `CastAttrs`
may import only `DataType`, `OperationAttrs`, and `Objects`. Confirm no Tensor, shape, layout,
storage, provenance, graph, compiler, planning, runtime, prepare, backend, training, gradient,
cost, fusion, route, registry, map, reflection, or service type appears. Validate generated
Javadoc, Tensor API status, glossary terminology, links/anchors/fences/whitespace, exact eight-path
scope, synchronized statuses, package-map placement, and absence of a task-0015H specification.

## Dependencies

- Task 0001 supplies the complete current `DataType` vocabulary.
- Task 0005 supplies `OperationKind` and `OperationAttrs`.
- Task 0006 supplies immutable generic `Operation` composition and exact reference retention.
- Completed parameterized scalar attributes establish record/Javadoc/test conventions but are not
  Java dependencies of this family.

## Follow-up tasks

- 0015H remains Draft for public `Tensor.cast(DataType)`, source/target handling, shape/result
  descriptor construction, same-type policy, gradient eligibility, provenance, and storage-free
  derived Tensor creation.
- Compiler tasks later own cast canonicalization, redundant-chain elimination, graph capture,
  autograd expansion, and optimization legality.
- Backend, ONNX, and conformance tasks later own mapping, conversion matrices, rounding,
  saturation/truncation, BOOL behavior, special values, lowering, storage access, and execution.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns backend-independent Operation semantics,
typed attributes, and DataType vocabulary to `modules/model`. The new package refines that
ownership without Tensor behavior, dependencies, inference, storage, or executable state.

If implementation requires Tensor behavior, result inference, numerical conversion policy,
backend metadata, another dependency, or architecture change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0005/0006/0014E/0015E/0015F/0015G, Tensor API,
Compile API, Training API, glossary, current DataType/OperationKind/OperationAttrs/
NoOperationAttrs/Operation and concrete kind/attributes contracts/tests, and Java 26 Gradle
configuration.

Implement task 0015G exactly. Add only CastKind.java, CastAttrs.java, and CastSemanticsTest.java
under io.github.pho001.synaptik.model.operation.elementwise.cast for Java code/tests.

CastKind is a public enum implementing OperationKind with exactly CAST and no project fields,
methods, nested types, aliases, arity, or metadata. CastAttrs is a public record implementing
OperationAttrs with exactly non-null DataType targetDataType, exact NullPointerException message
targetDataType, explicit documented accessor, and no other state/API. Accept and retain every
current DataType. Compose CAST explicitly with CastAttrs; do not use NoOperationAttrs or add a
compatibility validator.

Do not add Tensor.cast, source type, result descriptor, shape/layout/provenance, same-type policy,
conversion mode or numerical rules, gradients, graph/compiler/planning/runtime/backend/ONNX
behavior, factories/registries, dependencies/build/architecture changes, existing Java edits, or
later specs. Stop beyond eight paths or on architecture uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff/evidence to a separate
clean-context documentation agent in the same change. It must inspect source/tests/generated
Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record related-contract/
capability/Compile API/Training API/architecture no-change conclusions, and rerun validation.

Update task 0015G, model master plan, and roadmap only for planning status/evidence. Do not mark
0015G Complete until both passes succeed. Leave 0015H Draft without a specification. Do not commit
or push.
```

## Local decisions

- The family uses `CastKind.CAST` and `CastAttrs(targetDataType)`. The kind identifies conversion;
  attributes carry the one intrinsic parameter.
- The package is `operation.elementwise.cast`: casts operate independently at corresponding
  logical positions, while the leaf name keeps conversion separate from arithmetic families.
- Only the target type is stored. The future input descriptor is the authoritative source type,
  so duplicating source type in attributes would risk inconsistency.
- All current DataType values are representable targets. Backend availability and exact
  source/target numerical behavior are separate later contracts.
- Same-type target attributes are valid. Identity elimination versus an explicit cast expression
  is deliberately deferred to task 0015H and compiler optimization policy.
- `CastAttrs` is not parameterless; `NoOperationAttrs.INSTANCE` would lose the target type and is
  therefore invalid for the documented family composition.

## Known limitations

- No public cast Tensor expression exists until task 0015H.
- Source eligibility, same-type behavior, result descriptor, shape/layout, gradient eligibility,
  and provenance are undefined here.
- The model stores no rounding, saturation, overflow, special-value, or BOOL conversion policy.
- Generic `Operation` does not enforce the documented `CAST`/`CastAttrs` pairing.
- No compiler capture, optimization, ONNX mapping, backend support, storage interpretation, or
  execution is implied.

## Validation evidence

Planning reviewed the architecture contract and focused module/dependency explanations;
documentation and planning rules; roadmap; model capabilities and master plan; tasks 0001, 0005,
0006, 0014E, 0015E, and 0015F; current DataType, Operation foundations, concrete kind/attributes
source/tests, Tensor/Compile/Training APIs and glossary; and Java 26 Gradle configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms public fluent
`Tensor.cast(DataType)`, one non-null target-type semantic parameter, all current target values,
floating/BFLOAT16/integral/BOOL conversions, same-type identity behavior, strided inputs,
expression chaining, gradients for floating conversions, ONNX mapping, and backend execution
evidence. Legacy input/result coupling, arity/fusion/cost/result traits, same-type canonicalization,
gradient callbacks, conversion loops, storage access, lowering, and backend policy are excluded
from this semantic-only task.

Planning selected one enum and one target-only attributes record. The source type remains external
to Operation attributes, all target values remain representable, and numerical/executable policy
is deferred. No existing dependency, foundational contract, or architecture rule changes.

Planning validation:

- `git diff --check` passed, and targeted whitespace inspection found no trailing whitespace in
  the three changed planning paths.
- The required-section scan found every canonical task-specification section, including package
  impact, exact enum/record shapes, bounded scope, validation, implementation handoff, decisions,
  limitations, and completion-evidence sections.
- Every local Markdown file and heading anchor linked from this task, the model master plan, and
  the roadmap resolves. Markdown fence counts are balanced.
- Initial planning status inspection found 0015G `Ready` in this specification, its linked
  model-master row, and its linked roadmap row/current-frontier text before implementation. Final
  implementation status is recorded below. Task 0015H remains `Draft` in both queues.
- Package-map inspection found exactly one new planned package,
  `model.operation.elementwise.cast`, with direction only to `model.operation` and
  `model.datatype`.
- Pre-implementation planning scope inspection found exactly this new task, the model master plan,
  and the roadmap changed. No Java, test, API, glossary, Gradle, architecture, AGENTS, or
  other-module path had changed at that stage.
- No task-0015H specification exists.

Implementation and independent documentation validation:

- Added the exact public `CastKind` enum and `CastAttrs` record in the planned cast package, plus
  one same-package focused test. No existing Java source or test changed.
- The focused `CastSemanticsTest` passed all 6 tests. The aggregate model suite passed all 353
  tests with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` and `./gradlew test` passed. Generated Javadoc contains both
  new public types, and `git diff --check` passed.
- `javap -p -c -s` and reflection-based tests confirmed the exact enum/record shapes, sole enum
  constant, one record component, constructor null check and message, explicit accessor, generated
  record methods, and absence of extra project state or API.
- Production-import inspection found only `OperationKind` for `CastKind`, and only `DataType`,
  `OperationAttrs`, and `Objects` for `CastAttrs`. No Tensor, descriptor, shape, layout, storage,
  graph, compiler, runtime, backend, training, gradient, conversion-policy, registry, or service
  dependency was added.
- The clean-context documentation-focused pass in
  `/root/implement_model_0015d/review_model_0015d_docs` applied the General, API reference,
  example, and planning profiles. It found the two new Javadocs complete without revision,
  finalized the Tensor API and glossary, and synchronized this task, the model master plan, and
  the roadmap.
- Compile API remains accurate unchanged because no public Tensor expression or compiler capture
  was added. Training API remains accurate unchanged because cast gradient eligibility and rules
  are deferred. The model capability baseline already inventories explicit cast and therefore
  needs no status-level change.
- Architecture documents, ADRs, architecture tests, backend-conformance tests, and integration
  tests remain accurate unchanged because ownership, dependencies, module boundaries, and
  executable behavior did not change. `DataType`, `OperationKind`, `OperationAttrs`, `Operation`,
  `NoOperationAttrs`, existing concrete family contracts, Tensor contracts, and Java 26 build
  configuration also remain accurate unchanged.
- Markdown review found balanced fences, resolved local links and heading anchors, consistent cast
  terminology, and no trailing whitespace. Scope inspection found exactly the authorized eight
  paths. Task 0015H remains Draft and no task-0015H specification exists.
- Final implementation-context reruns passed the 6 focused tests, all 353 model tests, model
  Javadoc, root tests, and `git diff --check`. A generated-Javadoc scan first used a nonexistent
  module-qualified output directory and failed without changing files; the corrected scan of
  `modules/model/build/docs/javadoc/io/github/pho001/synaptik/model/operation/elementwise/cast`
  found both rendered public types, the target component, constructor failure, and accessor
  documentation.

## Implementation notes

- `CastKind` contains exactly `CAST` and relies only on inherited enum identity, names, and value
  methods.
- `CastAttrs` retains one exact non-null target `DataType`; it neither stores nor derives a source
  type and performs no compatibility or backend-support lookup.
- The documented family composition is explicit `Operation(CastKind.CAST, CastAttrs)`. Generic
  `Operation` remains unchanged and does not validate family compatibility.
- No source/result inference, numerical conversion, same-type policy, gradients, public Tensor
  expression, execution, backend route, dependency, or architecture behavior was introduced.

## Completion summary

Completed changes:

- Added backend-independent cast semantic identity and immutable target-data-type attributes.
- Added focused contract tests for vocabulary, shape, null behavior, all targets, value semantics,
  typed identity, explicit operation composition, and exclusions.
- Finalized the Tensor API, glossary, task evidence, model master plan, and roadmap.

Files changed or created: exactly the two production files, one focused test, and five authorized
documentation/planning files listed under Affected files.

Validation performed: focused and aggregate model tests, model Javadoc, root tests,
`git diff --check`, generated-Javadoc inspection, `javap`, reflection, import/dependency review,
Markdown link/anchor/fence/whitespace review, exact-scope inspection, and status/spec checks.

Unresolved issues: none within task 0015G. Public expression construction and all conversion,
gradient, compiler, backend, and execution behavior remain deliberately assigned to 0015H or
later owning layers.

Required follow-up: plan task 0015H separately when it becomes the active planning frontier.

Status: Complete
