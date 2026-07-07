# Task 0018A: Scalar Select Semantics

## Status

Complete

## Goal

Define the typed, backend-independent semantic identity and immutable normalized parameters for
selecting one scalar coordinate from one tensor axis.

Scalar select fixes one coordinate on one existing source axis and removes that axis from the
logical result. For example, selecting axis `1`, index `2` from shape `[2, 3, 4]` later produces
shape `[2, 4]`. This operation is distinct from elementwise conditional `WHERE` and from gather
operations whose indices are themselves tensors.

This task defines meaning and intrinsic normalized parameters only. Public request syntax, input
Shape validation, negative axis/index normalization, result Shape and layout derivation, Tensor
provenance, gradients, compiler behavior, lowering, and execution remain later responsibilities.

## Scope

- Add one public `SelectKind` enum implementing `OperationKind` with exactly `SELECT`.
- Add one public `SelectAttrs` record implementing `OperationAttrs` with exactly `int axis` and
  `long index`, in that order.
- Require both stored values to be normalized and non-negative.
- Document that one logical input axis is fixed to one coordinate and removed from the result.
- Document the exact `SELECT` plus `SelectAttrs` composition without adding a compatibility
  validator.
- Add one focused same-package structural, validation, value-semantics, and composition test.
- Introduce the cohesive package `io.github.pho001.synaptik.model.operation.index`.
- Finalize Javadocs, Tensor API terminology, glossary, task evidence, master plan, and roadmap
  through the mandatory independent documentation pass during implementation.

## Out of scope

- public `Tensor.select`, another Tensor method, overload, factory, expression helper, or task
  0018B implementation
- input Tensor, input Shape, rank lookup, negative request normalization, bounds checking, static
  versus dynamic selected-dimension policy, result Shape, descriptor, or provenance
- result layout, offset/stride calculation, view or alias state, storage access, allocation,
  copying, materialization, or value selection
- tensor-valued indices, gather, gather-axis, take, take-along-axis, gather-ND, scatter, or their
  parameter contracts
- using `WHERE`, `UNSTACK`, or `SLICE` as a replacement for the first-class `SELECT` meaning
- data-type validation, output type, gradient rules, graph capture, canonicalization,
  decomposition, compiler, planning, prepare, runtime, backend, engine, trace, ONNX, training, or
  execution behavior
- factories, registries, parsers, maps, string dispatch, reflection discovery, arity, cost,
  fusion, backend-support, route, or kernel metadata
- changing existing Java/tests, dependencies, Gradle, architecture, another module, or creating a
  task-0018B specification

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
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0015E](0015e-where-selection-semantic-kind.md)
- [Task 0017G](0017g-slice-semantics.md)
- [Task 0017K](0017k-tensor-composition-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The selected capability baseline requires scalar-index `select`. The read-only legacy code
represents it as an axis plus one scalar coordinate, removes the selected axis from the logical
shape, and uses it as one building block for unstack.

The new contract retains that capability but stores normalized coordinates using current model
conventions: an `int` axis and a `long` index compatible with long-valued Shape dimensions. The
semantic record does not store an input or output Shape because bounds and rank are properties of
one operation occurrence, not intrinsic operation parameters.

Legacy mutable graph nodes, immediate view construction, storage offsets, gradients, traits,
lowering, kernels, and runtime/backend coupling are capability evidence only and are not copied.
Task 0018B will own the public Tensor request, input-dependent normalization, Shape/layout rules,
descriptor construction, and provenance.

## Architecture constraints

- Production remains in `modules/model`, which owns backend-independent operation meaning.
- `SelectKind` identifies mathematical indexing meaning only, not an occurrence, Tensor, graph
  node, descriptor, executable, view, materialization plan, kernel, or backend route.
- `SelectAttrs.axis` is already normalized and non-negative; it stores no rank and cannot prove
  that the axis exists.
- `SelectAttrs.index` is already normalized and non-negative; it stores no selected extent and
  cannot prove that the coordinate exists.
- Scalar select has one logical input and removes exactly one selected axis, but this task neither
  validates nor constructs the result Shape.
- `SELECT` pairs with `SelectAttrs`. Generic `Operation` remains an open kind/attributes pair and
  does not enforce family pairing, arity, rank, bounds, result Shape, gradients, or backend support.
- `SELECT` is distinct from conditional `WHERE`, individually indexed `UNSTACK`, general `SLICE`,
  and every tensor-index gather form.
- Package direction is `model.operation.index -> model.operation + java.base` only.
- Stop if implementation requires Tensor, Shape, layout, provenance, another production type,
  another test, dependency, architecture change, or cross-layer behavior.

## Package impact

Add one cohesive operation-family package:

```text
io.github.pho001.synaptik.model.operation.index
  SelectKind
  SelectAttrs
```

The focused test uses the same package. This package will later own typed gather and functional
scatter semantic contracts; it must not depend on public Tensor or compiled graph packages.

## Required contract

### Semantic kind

Create exactly:

```java
public enum SelectKind implements OperationKind {
    SELECT
}
```

The enum declares no project field, explicit constructor, method, nested type, per-constant body,
alias, arity, axis, index, Shape, result, layout, cost, or backend metadata. Inherited
`Enum.name()` satisfies `OperationKind.name()`.

Javadoc must explain that `SELECT` fixes one scalar coordinate on one source axis and removes that
axis from the logical result. It must distinguish scalar index selection from conditional `WHERE`
and tensor-index gather without claiming Shape validation or execution.

### Normalized attributes

Create exactly:

```java
public record SelectAttrs(int axis, long index) implements OperationAttrs
```

The record has exactly two components in that order, one canonical constructor, two explicit
documented accessors, and record-generated `equals`, `hashCode`, and `toString`. Add no factory,
overload, builder, input/output Shape, rank, dimension extent, normalized flag, result descriptor,
layout, nested type, helper, or extra state/API.

Constructor validation order is exact:

1. reject a negative `axis` with `IllegalArgumentException` and exact message
   `axis must be non-negative: <axis>`;
2. reject a negative `index` with `IllegalArgumentException` and exact message
   `index must be non-negative: <index>`;
3. assign both primitive values unchanged.

Zero, `Integer.MAX_VALUE`, and `Long.MAX_VALUE` are structurally valid. Rank and bounds checks are
deferred because this value has no input Shape.

The explicit accessors must document normalized non-negative meaning, inputs, output values, and
the absence of input-dependent validation.

### Typed composition

The valid composition is explicit:

```java
SelectAttrs attrs = new SelectAttrs(1, 2L);
Operation operation = new Operation(SelectKind.SELECT, attrs);
```

For a later input with shape `[2, 3, 4]`, this means fixing coordinate `2` on axis `1`; the later
Tensor-expression task derives result shape `[2, 4]`. This task adds no compatibility validator
to `Operation`, no factory, and no result inference.

## Affected files

Expected implementation change:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/SelectKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/SelectAttrs.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/index/SelectSemanticsTest.java`
- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may create or modify at most the eight paths listed above.

If implementation requires another production type, another test, a Tensor or Shape change,
dependency, build change, focused architecture change, or more than eight paths, stop and propose
a follow-up task. Do not trade away validation or documentation to fit the limit.

## Javadoc requirements

- Document every public type, enum constant, record component/accessor, and canonical constructor.
- Explain scalar select using both words and the concrete `[2, 3, 4]`, axis `1`, index `2` example.
- Define `axis` and `index` as normalized and zero-based.
- Explain why rank/bounds/result Shape are not validated by the attributes.
- Distinguish `SELECT` from `WHERE`, `UNSTACK`, `SLICE`, and tensor-index gather.
- Do not promise view construction, aliasing, gradients, compiler capture, backend support, or
  execution.

## Acceptance criteria

- `SelectKind` is a public enum implementing `OperationKind` with exactly `SELECT`.
- `SelectKind` adds no project-declared fields, methods, constructors, or nested types.
- `SelectAttrs` is a public record implementing `OperationAttrs` with exactly `axis` then `index`.
- Negative axis/index failures use the exact order, exception type, and messages specified above.
- Valid primitive values are retained unchanged and record value semantics remain generated.
- Exact `SELECT` plus `SelectAttrs` composition works through the unchanged `Operation` record.
- No existing Java contract or test changes.
- No Tensor, Shape/result, layout, provenance, graph, gradient, compiler, runtime, backend, or
  execution behavior is introduced.
- Javadoc, Tensor API, glossary, planning evidence, master plan, and roadmap are independently
  reviewed and synchronized.
- All validation passes and the final diff contains exactly the permitted paths.

## Tests / validation

Run:

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.operation.index.SelectSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must verify:

- exact enum constant count, name, order, and `OperationKind` implementation;
- absence of project-declared enum state/API/nested types;
- exact record status, component names/order/types, interface, and absence of extra instance state;
- zero and primitive maximum values;
- exact negative-axis then negative-index validation and messages;
- generated equality, hashing, and diagnostic value semantics;
- exact `Operation(SelectKind.SELECT, SelectAttrs)` reference/value composition;
- the operation contracts contain no Tensor, Shape, layout, graph, backend, or runtime imports.

Manually inspect `javap`/reflection output, imports, generated Javadoc pages, Markdown links,
anchors, code fences, trailing whitespace, task/master/roadmap status, exact eight-path scope, and
absence of a task-0018B specification.

## Dependencies

- Task 0002 defines current axis/Shape terminology used by the documentation.
- Task 0005 defines `OperationKind` and `OperationAttrs`.
- Task 0006 defines the open typed `Operation` pair.
- Task 0018B depends on this task for public Tensor expression construction.

## Follow-up tasks

- 0018B: public scalar select Tensor expression, normalization, Shape/layout derivation, and
  provenance.
- 0018C–0018J: focused tensor-index gather and functional-scatter semantic/expression pairs.

Do not create any follow-up specification during this task.

## Architecture impact

Expected impact: None.

The task fills the existing model-owned operation vocabulary. If implementation requires a new
architecture rule or conflicts with `ARCHITECTURE.md`, stop and report the issue.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0005/0006/0015E/0017G/0017K/0018A, Tensor API,
Compile API, Training API, glossary, current OperationKind/OperationAttrs/Operation and related
selection/layout semantic contracts/tests, Shape terminology, and Java 26 Gradle configuration.

Implement task 0018A exactly. Add only SelectKind.java, SelectAttrs.java, and
SelectSemanticsTest.java under io.github.pho001.synaptik.model.operation.index.

SelectKind contains exactly SELECT and no project state/methods/nested types/metadata. SelectAttrs
contains exactly normalized non-negative int axis then long index, validates in the exact order
with exact messages, retains valid primitives unchanged, and exposes explicit documented
accessors. Document scalar axis removal, the [2,3,4]/axis-1/index-2 -> [2,4] example, exact typed
pairing, and distinctions from WHERE, UNSTACK, SLICE, and tensor-index gather.

Do not add Tensor methods, Shape/rank/bounds/result/layout/provenance logic, gather/scatter types,
gradients, graph/compiler/planning/runtime/backend behavior, factories, dependencies, build or
architecture changes, existing Java edits, or later specs. Stop beyond eight paths or on
architecture uncertainty.

Run all specified validation, then hand the actual diff/evidence to a separate clean-context docs
agent in the same change. It must inspect source/tests/generated Javadoc, finalize permitted
Javadocs/Tensor API/glossary/planning, record Compile API/Training API/capability/architecture and
related-contract no-change conclusions, and rerun validation.

Update task 0018A, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0018B Draft without a specification. Do not commit/push.
```

## Local decisions

- Scalar select is represented by the first-class `SelectKind.SELECT` identity rather than by
  reusing conditional `WHERE`, individually indexed `UNSTACK`, general `SLICE`, or a tensor-index
  gather meaning.
- `SelectAttrs` stores a normalized `int axis` followed by a normalized `long index`. The widths
  match current rank/axis positions and long-valued Shape dimensions without storing an input
  Shape or duplicating input-dependent rank and bounds facts.
- Both non-negative checks occur in the canonical record constructor, axis first, and the explicit
  accessors return the stored primitive fields directly. Generated record object methods remain
  the value-semantics contract.
- The submitted production Javadocs were retained unchanged after independent review because they
  already document every public type, constant, component, constructor, accessor, failure,
  example, family distinction, and deferred boundary required by this task.

## Known limitations

- This task provides semantic identity and normalized intrinsic parameters only. It intentionally
  provides no public `Tensor.select`, raw negative normalization, rank or bounds check, result
  Shape/layout, view geometry, provenance, value access, gradient, graph/compiler behavior,
  materialization, backend lowering, or execution. Task 0018B remains Draft for the public
  expression boundary.

## Validation evidence

- Implementation context `/root/implement_model_0018a` created exactly `SelectKind`,
  `SelectAttrs`, and `SelectSemanticsTest`. Independent documentation context
  `/root/implement_model_0018a/review_model_0018a_docs` inspected the actual uncommitted diff,
  implementation evidence, source, tests, generated Javadoc, related contracts, APIs, glossary,
  planning state, and build configuration before finalizing documentation in the same change.
- The documentation pass applied General plus API/Javadoc style and Example format to the two
  production Javadocs, Tensor API, and glossary, and Planning style to this task, the model master
  plan, and roadmap.
- Reviewed in full: `AGENTS.md`, `ARCHITECTURE.md`, the current architecture plan, overview,
  lifecycle, module-boundary, dependency-rule, and runtime/prepare/backend-boundary explanations;
  documentation rules and General/API-Javadoc/Planning/Example profiles; planning guide and
  roadmap; model capabilities and master plan; tasks 0002, 0005, 0006, 0015E, 0017G, 0017K, and
  0018A; Tensor, Compile, and Training API references; glossary; both new production sources and
  the focused test; OperationKind, OperationAttrs, Operation, WHERE, SLICE, UNSTACK, Shape, and
  related semantic contracts/tests; generated model Javadoc; and Java 26 root/model Gradle
  configuration.
- `SelectKind` review confirmed exactly `SELECT`, `OperationKind` implementation, no project
  field, method, explicit constructor, per-constant body, or nested type, and no metadata beyond
  enum compiler machinery. `SelectAttrs` review confirmed exactly private final `int axis` then
  `long index`, one public `(int, long)` canonical constructor, explicit direct accessors,
  axis-first validation, exact failure messages, unchanged assignments, generated equality,
  hashing, and diagnostic text, and no nested or extra state.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.index.SelectSemanticsTest` — `BUILD SUCCESSFUL`; XML
  reports 9 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; XML aggregation reports 638 tests across
  75 suites with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL` without Javadoc errors. Generated pages
  contain the type, enum constant, record components, canonical constructor, explicit accessors,
  exact failures, normalized semantics, `[2, 3, 4]`/axis-`1`/index-`2` to `[2, 4]` example,
  SELECT/attributes pairing, family distinctions, and deferred boundaries.
- `./gradlew test` — `BUILD SUCCESSFUL`; the repository lifecycle completed with 36 actionable
  tasks and no failing task.
- The implementation handoff also recorded one intentionally parallel model/root Gradle attempt
  that collided in shared test-result files (`EOFException` and a missing
  `in-progress-results-generic.bin`). Required sequential reruns of both lifecycles passed; this
  superseded result is shared-build-output concurrency evidence, not a product-test failure.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` confirmed the exact enum and
  record structures, `(int, long)` constructor, axis-first/index-second checks, direct field
  assignments/accessors, and generated object methods. Focused reflection tests independently
  confirm visibility, interfaces, component/member order, absence of extra project API/state, and
  exact typed `Operation` composition.
- Production import inspection found only `OperationKind` in `SelectKind` and `OperationAttrs` in
  `SelectAttrs`. Source, bytecode, and focused-test review found no Tensor, Shape, data-type,
  layout, provenance, graph, compiler, planning, prepare, runtime, backend, training, or execution
  dependency or behavior.
- Tensor API and glossary now document the current semantic values, normalized axis/index,
  scalar-axis removal, conceptual example, exact composition, validation boundary, and
  distinctions from WHERE, UNSTACK, SLICE, and tensor-index gather while keeping public
  `Tensor.select` and cross-layer work planned.
- Local Markdown validation passed for the five changed documentation/planning files: 346 local
  links and 87 anchors resolve, including the new scalar-select heading and glossary link. All 210
  code-fence markers are balanced. Trailing-whitespace scans found no matches, all eight files end
  with a newline, and `git diff --check` passed.
- Final scope contains exactly the authorized eight paths: two new production sources, one focused
  test, Tensor API, glossary, this task, model master plan, and roadmap. Task, master plan, and
  roadmap consistently mark 0018A Complete. Task 0018B remains Draft, and no task-0018B
  specification exists.
- `docs/api/compile-api.md` remains accurate unchanged because 0018A adds no public Tensor
  expression, compiler entry point, capture, inference, canonicalization, graph conversion, or
  compile artifact. `docs/api/training-api.md` remains accurate unchanged because no gradient,
  autograd, optimizer, parameter, session, or training behavior changed. The capability baseline
  already lists scalar-index select and correctly separates model semantics, public Tensor API,
  compiler, planning, backend prepare, runtime, and tests, so it required no status change.
- OperationKind, OperationAttrs, and Operation remain accurate because the new values implement
  their open typed roles without changing generic compatibility enforcement. WHERE remains ternary
  elementwise choice, UNSTACK remains an individually indexed logical multi-result output, SLICE
  remains same-rank half-open interval selection, and Shape/layout/Tensor/provenance/graph
  contracts remain unchanged because this task derives no result or occurrence state.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, and architecture tests remain
  unchanged because the work stays inside model-owned backend-independent semantics and changes no
  module boundary, dependency direction, or lifecycle. Backend-conformance and integration tests
  remain unchanged because there is no backend or end-to-end behavior. Java 26 Gradle files,
  dependencies, other modules, and later task specifications remain unchanged because the task
  needs no build, dependency, cross-module, or follow-up implementation change.

## Implementation notes

- Added the exact one-value `SelectKind` vocabulary and exact two-component `SelectAttrs` record in
  `model.operation.index`, plus one focused structural, validation, value-semantics, distinction,
  composition, and dependency-boundary test.
- Finalized the Tensor API, glossary, and synchronized planning evidence through the independent
  documentation pass. No production Javadoc correction was necessary.

## Completion summary

- Completed changes: Implemented and documented first-class scalar-axis SELECT meaning and
  immutable normalized axis/index attributes without public Tensor or cross-layer behavior.
- Files changed or created: Two production Java files, one focused test, Tensor API, glossary,
  this task specification, model master plan, and implementation roadmap.
- Tests and validation: Focused 9-test suite, all 638 model tests across 75 suites, generated model
  Javadoc, full repository tests, javap/reflection/import/source/generated-page review, Markdown
  link/anchor/fence/whitespace checks, exact-scope/status checks, and `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0018a/review_model_0018a_docs` completed the required independent pass in
  the same overall change.
- Documentation impact: Tensor API and glossary now explain scalar select, normalized parameters,
  axis removal, exact pairing, family distinctions, and current-versus-planned boundaries. Compile
  API, Training API, capabilities, architecture/ADRs/tests, conformance/integration material,
  Gradle, dependencies, other modules, and later tasks remain accurate unchanged for the recorded
  reasons.
- Javadoc review: Complete for both production types, the enum constant, record components,
  canonical constructor, accessors, failures, example, distinctions, and exclusions; no correction
  was required.
- Glossary impact: Added scalar-select terminology and aligned operation-kind and attribute
  inventories with the implemented semantic family.
- Unresolved issues: None.
- Follow-up required: None for task 0018A. Task 0018B remains Draft without a detailed
  specification.

Status: Complete
