# Task 0017A: Contiguous Semantic Kind

## Status

Complete

## Goal

Define the typed, backend-independent, parameterless semantic identity for a contiguous-layout
request: `CONTIGUOUS`.

The operation preserves logical values, shape, data type, and element order while requesting a
canonical dense row-major result. This task defines only that meaning. It does not add
`Tensor.contiguous()`, derive a result descriptor, inspect an input layout, decide whether an alias
is sufficient, materialize storage, copy values, or select an executable implementation.

## Scope

- Add one public enum `ContiguousKind` implementing `OperationKind`.
- Define exactly one constant, `CONTIGUOUS`.
- Document one logical input and unchanged logical values, shape, data type, and row-major element
  order.
- Document the requested result geometry as canonical dense row-major layout with zero logical
  storage offset, without constructing a `LayoutDescriptor` in this task.
- Establish that the kind has no intrinsic attributes and composes explicitly with
  `NoOperationAttrs.INSTANCE` when represented by `Operation`.
- Distinguish a semantic contiguous request from the current geometric
  `LayoutKind.DENSE_CONTIGUOUS` classification.
- Document that later layers may preserve an already-suitable representation or materialize a
  copy, and that this enum makes neither decision.
- Add one focused same-package test proving exact vocabulary, typed identity, enum behavior,
  parameterless `Operation` composition, and absence of metadata or state.
- Add the cohesive `model.operation.layout` package to the model package map.
- Finalize affected Javadoc, Tensor API semantic reference, glossary, task evidence, model master
  plan, and roadmap through the required independent documentation pass during implementation.

## Out of scope

- public `Tensor.contiguous()`, another Tensor method, static facade, overload, factory, builder,
  expression helper, or task-0017B implementation
- input Tensor, provenance, identity, graph values/nodes, labels, host storage, result
  construction, or `TensorFactory.createDerived`
- DataType, Shape, `LayoutDescriptor`, `LayoutKind`, strides, storage offset, referenced span,
  view flag, result layout construction, or dynamic-shape layout resolution
- checking whether an input is contiguous, returning the input, aliasing storage, allocating or
  copying storage, materialization policy, mutation, publication, or execution
- reshape, expand, permute, transpose, expand-dimensions, squeeze, slice, pad, tile, concat, stack,
  unstack, unfold, fold, select, gather, or scatter semantics
- attributes, layout-requirement records, factories, registries, parsers, aliases, symbols,
  visitors, string dispatch, reflection discovery, maps, services, or compatibility validators
- arity, family, result-kind, cost, fusion, differentiability, capability, backend support,
  lowering, route, kernel, executable, or physical-buffer metadata
- compiler canonicalization, logical materialization requirements, planning ownership, prepare-time
  lowering, runtime residency, backend storage, gradient rules, autograd, or training behavior
- changes to Operation foundations, layout value contracts, Tensor, graph records, existing Java
  tests, dependencies, Gradle, architecture, or another module
- a detailed task-0017B specification or any later 0017 task specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0016J](0016j-softmax-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes:

```java
Tensor contiguous()
TensorOps.contiguous(Tensor input)
```

Its operation descriptor identifies one parameterless `CONTIGUOUS` meaning. The public operation
preserves logical shape and data type. Legacy execution can reuse an already suitable
representation or materialize logical values from offset, permuted, sliced, expanded, or other
non-dense views into canonical row-major storage. Tests cover BOOL and every numeric type,
permuted/expanded/sliced inputs, chained expressions, CPU routes, Metal, CUDA, and preparation or
runtime layout handling.

Legacy operation traits, `OpType`, result-kind labels, cost/fusion flags, storage views, physical
buffer policies, materializers, kernel classes, lowering, runtime propagation, and gradient
callbacks are evidence only and are not copied. The new model retains only the stable semantic
request. Task 0017B will own public Tensor expression construction; compiler, planning, prepare,
runtime, and concrete backends retain their architecture-defined responsibilities.

## Architecture constraints

- Production remains in `modules/model`, which owns backend-independent operation semantics.
- `ContiguousKind` is a typed semantic discriminator, not a layout descriptor, Tensor, graph
  occurrence, materialization requirement, physical copy, executable operation, backend
  capability, or kernel route.
- The enum implements `OperationKind` through inherited `Enum.name()` and adds no duplicate name
  field or method.
- `CONTIGUOUS` has no intrinsic parameters. Its complete attributes value is
  `NoOperationAttrs.INSTANCE`, never null or an empty map.
- The operation has one logical input and preserves its logical values, Shape, DataType, and
  row-major element order. These facts are documentation, not stored arity or descriptor state.
- The requested logical result geometry is canonical dense row-major layout with zero logical
  storage offset. A later Tensor expression decides how that request is represented in a
  `TensorDescriptor`, including the unresolved state required by dynamic shapes.
- `ContiguousKind.CONTIGUOUS` expresses a computation request. `LayoutKind.DENSE_CONTIGUOUS`
  classifies already-resolved geometry. Neither type replaces or depends on the other.
- Model semantics do not decide whether a request can alias an existing representation, can be
  eliminated by compiler proof, or needs materialization. Planning derives logical materialization
  requirements; backend prepare chooses lowering and a concrete route; runtime executes the
  prepared schedule.
- Stable enum text is diagnostic typed vocabulary, not serialization, ONNX mapping, a registry
  key, backend dispatch key, kernel name, or reflection identifier.
- Package direction is `model.operation.layout -> model.operation` only. This task must not import
  Tensor, datatype, shape, layout-value, storage, graph, compiler, planning, prepare, runtime,
  backend, engine, trace, or training packages.
- Stop if implementation needs another type, attributes, descriptor inference, Tensor behavior,
  materialization policy, dependency, or architecture change.

## Package impact

Existing package used:

- `io.github.pho001.synaptik.model.operation` — supplies `OperationKind`, `Operation`, and
  `NoOperationAttrs` for the semantic contract and composition test.

Package added:

```text
io.github.pho001.synaptik.model.operation.layout
  Typed backend-independent layout and view operation meanings and immutable parameters.
```

This operation package is distinct from `io.github.pho001.synaptik.model.layout`, which describes
resolved geometry. The operation package states requested computation semantics; the layout-value
package stores already-resolved shape-to-storage geometry without representing an operation.

Type placement:

- `io.github.pho001.synaptik.model.operation.layout.ContiguousKind` — public parameterless
  contiguous-request identity.
- `ContiguousKindTest` — same-package focused test for vocabulary, contract shape, composition,
  and dependency boundaries.

## Required contract

### Enum declaration

Create exactly:

```java
public enum ContiguousKind implements OperationKind {
    CONTIGUOUS
}
```

The enum declares no project field, explicit constructor, method, nested type, per-constant class
body, alias, symbol, or metadata. Compiler-generated enum machinery is not additional project API.
Inherited `Enum.name()` satisfies `OperationKind.name()` and returns exact text `CONTIGUOUS`.

The sole kind means: preserve the input's logical values, shape, data type, and element order while
requesting canonical dense row-major result geometry with logical storage offset zero. It does not
state whether the eventual implementation aliases, copies, allocates, eliminates, fuses, lowers,
or executes the request.

### Parameterless Operation composition

The kind composes explicitly as:

```java
Operation operation = new Operation(
        ContiguousKind.CONTIGUOUS,
        NoOperationAttrs.INSTANCE);
```

Do not add a family factory, enum `operation()` method, attributes record, layout requirement,
result descriptor, or compatibility registry. Generic `Operation` remains an open typed pairing
and does not enforce family-specific input count, result geometry, or kind-to-attributes pairing.

### Semantic and geometric distinction

The implementation and documentation must preserve this distinction:

| Concept | Meaning | Stored by this task |
|---|---|---|
| `ContiguousKind.CONTIGUOUS` | a request to produce logically equivalent canonical dense row-major output | yes |
| `LayoutKind.DENSE_CONTIGUOUS` | classification of resolved strides and zero offset for a static shape | no |
| materialization requirement | planning fact that a logical representation needs a copy or conversion | no |
| prepared copy or alias | backend/runtime implementation chosen before execution | no |

The table describes ownership boundaries. It does not add a Java dependency from the operation
package to the layout-value package.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/ContiguousKind.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/layout/ContiguousKindTest.java`

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
- `LayoutKind`, `LayoutDescriptor`, Operation foundations, existing operation families, Tensor
  expressions, and their Javadocs/tests
- focused architecture documents, ADRs, architecture tests, backend-conformance material,
  integration tests, and Gradle configuration

## Maximum scope

At most one production file, one focused test, and five documentation/planning files: seven paths
total.

Do not modify existing Java source/tests, capabilities, completed tasks, Compile/Training API,
Gradle, AGENTS, architecture documents/tests, another module, or unrelated documentation. Stop if
another production concept, attributes, Tensor expression, descriptor rule, dependency, or eighth
path is needed. Do not create task 0017B.

## Javadoc requirements

- Document the enum as backend-independent parameterless contiguous-request vocabulary.
- Document `CONTIGUOUS` as preserving logical values, Shape, DataType, and element order while
  requesting canonical dense row-major zero-offset result geometry.
- Explain the distinction from resolved `LayoutKind.DENSE_CONTIGUOUS` without importing or linking
  the operation type to the layout-value package.
- Explain one logical input without storing or validating arity.
- Document explicit `Operation` composition with `NoOperationAttrs.INSTANCE` and absence of generic
  family compatibility validation.
- State that alias reuse, compiler elimination, materialization, allocation, copying, lowering,
  execution, gradient behavior, and backend availability are outside this semantic type.
- Explain that enum names are diagnostic and are not serialization or dispatch contracts.
- Review related Javadocs and record why they remain accurate, or stop on an inconsistency.

## Acceptance criteria

- Exactly one public `ContiguousKind` enum is added in the planned operation-layout package.
- It implements `OperationKind` and declares exactly `CONTIGUOUS`.
- It adds no project field, method, constructor, nested type, constant body, alias, attributes,
  arity, category, result, cost, fusion, differentiability, layout state, backend, inference,
  materialization, or execution metadata.
- Inherited name and standard enum equality, hashing, and diagnostic text behavior remain.
- `CONTIGUOUS` constructs a valid `Operation` with the exact kind and
  `NoOperationAttrs.INSTANCE`; no factory or attributes type is added.
- Javadoc explains logical preservation, requested canonical dense row-major zero-offset geometry,
  and the semantic-kind versus resolved-layout distinction.
- Production imports only `OperationKind`; no Tensor, DataType, Shape, LayoutDescriptor,
  LayoutKind, provenance, graph, compiler, planning, prepare, runtime, backend, storage, training,
  dependency, or architecture behavior is added.
- No public Tensor API, result descriptor, layout inspection, alias/copy decision, storage access,
  gradient rule, compiler behavior, or execution is implemented.
- Focused and aggregate tests, Javadoc, root tests, reflection/javap/import/scope checks,
  documentation links and formatting, and synchronized statuses pass.
- A separate clean-context documentation agent finalizes Javadoc, Tensor API, glossary, evidence,
  master plan, and roadmap and records related no-change conclusions.
- Task 0017A becomes Complete only after both passes. Task 0017B remains Draft without a detailed
  specification.

## Tests / validation

Run before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.layout.ContiguousKindTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover the exact package, visibility, implemented interface, sole constant,
name, declaration order, absence of project fields/methods/nested types/constant bodies, standard
enum identity, typed distinction from a private test-local equal-name kind, and explicit
`Operation` composition retaining the exact kind and `NoOperationAttrs.INSTANCE` references.

Manually inspect `javap -p -c -s`, reflection, imports, and Gradle dependencies. Confirm that no
Tensor, DataType, Shape, LayoutDescriptor, LayoutKind, provenance, graph/compiler/planning/
prepare/runtime/backend type, storage access, materialization decision, result tag, gradient, cost,
fusion, route, registry, map, or service appears. Validate generated Javadoc, Tensor API semantic
status, glossary terminology, links, anchors, fences, whitespace, exact seven-path scope,
synchronized statuses, package-map placement, and absence of a task-0017B specification.

## Dependencies

- Task 0005 supplies `OperationKind` and `NoOperationAttrs.INSTANCE`.
- Task 0006 supplies immutable generic `Operation` composition and exact reference retention.
- Task 0003 supplies the resolved layout vocabulary used only to review and document the ownership
  distinction; this enum has no Java dependency on it.
- Task 0013 supplies the future derived-Tensor construction seam for task 0017B; it is
  implementation-order context rather than a dependency of this enum.

## Follow-up tasks

- 0017B remains Draft for the exact public `Tensor.contiguous()` expression, result descriptor,
  static/dynamic layout handling, one-input provenance, and storage-free derived construction.
- 0017C–0017L remain Draft for reshape/expand, axis transforms, slicing, pad/tile, tensor
  composition, and unfold/fold semantic and expression groups.
- Compiler later owns redundant-request canonicalization and graph-wide layout reasoning.
- Planning later derives logical materialization requirements without selecting a kernel.
- Backend prepare later chooses alias/copy lowering and executable routes; runtime executes the
  prepared schedule.
- Training and compiler-generated semantic tasks later own differentiation and backward forms.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns backend-independent operation semantics
and logical layout facts to `modules/model`, materialization requirements to planning, concrete
lowering to backend prepare, and prepared execution to runtime. The new package refines model
ownership without adding cross-module dependencies or executable state.

If implementation requires Tensor behavior, descriptor inference, materialization policy, backend
metadata, another dependency, or architecture change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0003/0005/0006/0013/0016J/0017A, Tensor API, Compile API,
Training API, glossary, current OperationKind/OperationAttrs/NoOperationAttrs/Operation and
LayoutKind/LayoutDescriptor contracts and focused tests, and Java 26 Gradle configuration.

Implement task 0017A exactly. Add only ContiguousKind.java and ContiguousKindTest.java under
io.github.pho001.synaptik.model.operation.layout for Java code and tests.

The public enum implements OperationKind and contains exactly CONTIGUOUS, with no project fields,
methods, nested types, aliases, attributes, arity, layout state, materialization state, or metadata.
It is parameterless and composes explicitly with Operation plus NoOperationAttrs.INSTANCE.
Document one logical input, unchanged values/Shape/DataType/element order, and the request for
canonical dense row-major zero-offset result geometry. Keep the semantic request distinct from
resolved LayoutKind.DENSE_CONTIGUOUS geometry and from planning/backend/runtime materialization.

Do not add Tensor methods, result descriptors, LayoutDescriptor dependencies, input-layout
inspection, alias/copy decisions, storage access, provenance, gradients, graph/compiler/planning/
prepare/runtime/backend behavior, other layout kinds/attributes, dependencies, build/architecture
changes, existing Java edits, or later specs. Stop beyond seven paths or on architecture doubt.

Run every specified focused/aggregate test, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff and evidence to a
separate clean-context documentation agent in the same change. It must inspect source/tests/
generated Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record related-
contract/capability/Compile API/Training API/architecture no-change conclusions, and rerun
validation.

Update task 0017A, model master plan, and roadmap only for planning status/evidence. Do not mark
0017A Complete until both passes succeed. Leave 0017B Draft without a specification. Do not commit
or push.
```

## Local decisions

- The type is `ContiguousKind`, not a broad frozen `LayoutOperationKind`. This task has one stable
  parameterless meaning; later parameterized layout families can receive cohesive typed contracts
  without modifying this completed enum.
- The package is `model.operation.layout`. It owns requested operation semantics and is deliberately
  distinct from `model.layout`, which owns resolved geometry values.
- The sole constant is `CONTIGUOUS`. No `COPY`, `MATERIALIZE`, `DENSE`, or `ALIAS` synonym is added
  because those names would conflate the mathematical request with an implementation decision.
- `NoOperationAttrs.INSTANCE` completely represents the absence of intrinsic parameters.
- The semantic target is canonical dense row-major zero-offset geometry. Whether that target is
  represented as resolved or unresolved descriptor metadata belongs to task 0017B because dynamic
  Shape handling is part of expression construction.
- Planning and backend prepare, not this enum, decide whether an actual representation requires a
  copy. An already suitable input does not change the identity of the requested semantic operation.

## Known limitations

- No public `Tensor.contiguous()` expression exists until task 0017B.
- No result descriptor, static/dynamic layout rule, provenance, or gradient eligibility is defined
  here.
- The enum does not inspect input geometry, prove contiguity, choose aliasing, request a physical
  buffer directly, or enforce one input.
- No compiler canonicalization, planning materialization, backend lowering, storage copy, runtime
  execution, differentiation, or backend support is implied.

## Validation evidence

Planning read the architecture contract and focused lifecycle/module/dependency/runtime-boundary
explanations; documentation and planning rules; roadmap; model capability baseline and master
plan; tasks 0003, 0005, 0006, 0013, and 0016J; current Operation and resolved-layout contracts;
Tensor/Compile/Training APIs and glossary; and Java 26 Gradle configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms one public
`Tensor.contiguous()` capability, one parameterless `CONTIGUOUS` semantic identity, unchanged
logical shape/data type, and execution evidence for numeric and BOOL tensors plus permuted,
expanded, sliced, offset, CPU, Metal, and CUDA paths. Legacy operation traits, materializers,
physical views, storage policies, kernels, lowering, runtime propagation, gradients, and backend
selection are excluded from this semantic-only task.

Planning selected one enum and one focused test because the contiguous request has one stable
parameterless meaning. Public Tensor construction, descriptor handling, provenance, and any
static-versus-dynamic layout decision remain in task 0017B. No existing dependency, layout value,
Operation foundation, or architecture contract changes.

Planning validation after synchronizing this task, the model master plan, and roadmap:

- `git diff --check` passed.
- The targeted trailing-whitespace scan returned no matches across the three changed planning
  files.
- All 180 local Markdown file links across the three planning files resolve.
- Markdown code-fence counts are balanced: twelve in this task, two in the master plan, and zero
  in the roadmap.
- All 20 canonical task-specification headings are present, together with the focused Capability
  origin, Required contract, and Javadoc requirements sections.
- Task, model master plan, and roadmap consistently identify 0017A as Ready. Tasks 0017B–0017N
  remain Draft, and no task-0017B specification exists.
- Package-map inspection finds exactly one new planned package,
  `model.operation.layout`, with direction only to `model.operation` for this task.
- Dependency review distinguishes hard Java prerequisites 0005/0006 from layout/provenance context
  0003/0013; later 0017 rows depend on the contracts they actually consume rather than forming a
  false chain.
- Repository scope is exactly this task, the model master plan, and the roadmap. No Java, API,
  glossary, architecture, Gradle, AGENTS, completed task, or other-module file changed during
  planning.

Implementation and independent documentation validation:

- Implementation context `/root/implement_model_0017a` added exactly `ContiguousKind.java` and
  `ContiguousKindTest.java`. Clean documentation context
  `/root/implement_model_0017a/review_model_0017a_docs` independently read the required
  architecture, documentation and planning profiles, plans and historical tasks, APIs, glossary,
  Java 26 build configuration, final source and tests, generated Javadoc, XML reports, and the
  complete workspace diff. It applied General and API/Javadoc style to the enum, Tensor API, and
  glossary, and Planning style to this task, the model master plan, and roadmap. No substantive
  example changed, so Example format required no new complete example.
- Independent source, reflection, and bytecode inspection confirmed one public final enum in
  `io.github.pho001.synaptik.model.operation.layout`, exactly `CONTIGUOUS`, inherited enum
  behavior, and only compiler-generated enum fields, methods, and constructor. Production source
  imports exactly `OperationKind`; it adds no Tensor, descriptor, layout-value, provenance,
  storage, graph, compiler, planning, prepare, runtime, backend, engine, trace, or training
  dependency or behavior.
- The submitted enum and constant Javadocs are complete unchanged. They document the one logical
  input; preservation of logical values, Shape, DataType, and row-major element order; requested
  canonical dense row-major zero-offset geometry; explicit `Operation` plus
  `NoOperationAttrs.INSTANCE` composition; the distinction from resolved
  `LayoutKind.DENSE_CONTIGUOUS`; diagnostic enum names; and the deferred alias, copy,
  materialization, compiler, lowering, execution, gradient, and backend boundaries.
- `docs/api/tensor-api.md` now lists `ContiguousKind` as current semantic vocabulary and adds a
  focused semantic reference. It explicitly keeps public `Tensor.contiguous()`, result-descriptor
  construction, input-layout inspection, provenance, and materialization behavior planned.
  `docs/glossary.md` adds the reusable contiguous-request term and synchronizes the operation-kind
  status while distinguishing semantic request, resolved layout classification, and later
  materialization.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.layout.ContiguousKindTest` — `BUILD SUCCESSFUL`; the
  XML report records 5 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 59 XML suites record 454 tests with zero
  failures, errors, or skips.
- One preliminary combined shell invocation printed the focused XML report and then failed before
  the aggregate test started because the sandbox denied creation of the Gradle distribution
  lock file. Rerunning each required Gradle command separately used the approved command scope and
  produced the successful focused, aggregate, Javadoc, and root results recorded here.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated
  `ContiguousKind.html` contains the enum and constant contracts, explicit composition,
  semantic/geometric distinction, and diagnostic-name and cross-layer boundaries.
- `./gradlew test` — `BUILD SUCCESSFUL`; all repository test tasks completed without failure.
- `javap -p -c -s` confirmed exactly `CONTIGUOUS`, compiler-generated `$VALUES`, standard enum
  `values`/`valueOf`, the compiler constructor and initializer, and no project field, method, or
  nested type. Focused reflection tests independently verify the same surface, standard identity,
  exact Operation composition, and typed inequality with an equally named test-local kind.
- Markdown target and heading-anchor validation, balanced-fence inspection, terminology review,
  targeted trailing-whitespace scans, package-map/status checks, and `git diff --check` passed.
  Exact scope is the seven authorized paths: one production file, one focused test, Tensor API,
  glossary, this task, model master plan, and roadmap. Task 0017B remains Draft and no task-0017B
  specification exists. No commit or push was performed.
- Operation/attribute foundations remain accurate unchanged because this task only supplies one
  conforming parameterless kind and composes it through their existing contracts.
  `LayoutKind` and `LayoutDescriptor` remain accurate because resolved geometry is neither changed
  nor imported. Current operation families and Tensor expressions remain accurate because no
  public expression, result descriptor, provenance, input inspection, or storage behavior was
  added. `capabilities.md` already inventories contiguous support and its layer separation.
- Compile API remains accurate unchanged because no public Tensor expression, capture input,
  compiler pass, canonicalization, logical materialization requirement, or artifact was added.
  Training API remains accurate because no gradient, autograd, parameter, optimizer, publication,
  or session behavior changed. `ARCHITECTURE.md`, focused architecture explanations, ADRs, and
  architecture tests remain accurate because module ownership, dependency direction, and
  lifecycle boundaries did not change. Backend conformance and integration tests remain unchanged
  because no backend or end-to-end behavior exists. Root/model Java 26 Gradle configuration and
  other modules remain unchanged because no dependency, language level, source-set, build-task,
  or cross-module behavior changed.

## Implementation notes

- Added the exact public `ContiguousKind` enum with sole `CONTIGUOUS` identity and no project
  state or behavior beyond inherited `OperationKind` naming.
- Added the focused five-test suite for exact vocabulary and enum shape, typed identity, canonical
  parameterless Operation composition, and absence of project metadata.
- Finalized Tensor API and glossary semantics plus synchronized planning status. The production
  Javadocs required no correction after independent review.

## Completion summary

- Completed changes: Implemented and documented the parameterless contiguous-layout semantic
  request without public Tensor construction or materialization behavior.
- Files changed or created: Exactly one production enum, one focused test, Tensor API, glossary,
  this task, model master plan, and roadmap.
- Tests and validation: Focused tests passed 5/5; all 454 model tests across 59 suites, generated
  model Javadoc, root tests, javap/reflection/import/source/generated-documentation/manual checks,
  Markdown link/anchor/fence/terminology/whitespace checks, exact scope/status checks, and
  `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0017a/review_model_0017a_docs` completed the independent pass using
  General, API/Javadoc, and Planning profiles.
- Documentation impact: Tensor API and glossary now present the contiguous semantic kind as
  current while keeping public `Tensor.contiguous()`, descriptor construction, materialization,
  and cross-layer behavior planned.
- Javadoc review: The enum and constant Javadocs are complete unchanged; related operation,
  resolved-layout, Tensor-expression, and cross-layer contracts remain accurate for the reasons
  recorded above.
- Glossary impact: The contiguous-request term and operation-kind status now distinguish semantic
  intent from resolved `DENSE_CONTIGUOUS` geometry and later materialization.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0017A. Task 0017B remains Draft without a detailed
  specification.

Status: Complete
