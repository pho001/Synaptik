# Task 0018E: Gather-ND Semantics

## Status

Complete

## Goal

Define the typed, backend-independent semantic identity and immutable normalized batch-dimension
parameter for tuple-index `GATHER_ND`.

Gather-ND consumes ordered logical inputs `[data, indices]`. The final indices Dimension is the
tuple depth `K`; each tuple selects `K` data axes after the shared leading batch dimensions. The
conceptual result Shape is:

```text
indices.shape[:-1] + data.shape[batchDimensions + K:]
```

This task defines meaning and intrinsic `batchDimensions` only. Public Tensor methods, input ranks,
batch compatibility, tuple-depth extraction, result Shape construction, index data type,
provenance, gradients, compiler behavior, lowering, and execution remain later responsibilities.

## Scope

- Add one public `GatherNdKind` enum implementing `OperationKind` with exactly `GATHER_ND`.
- Add one public `GatherNdAttrs` record implementing `OperationAttrs` with exactly one
  non-negative normalized `int batchDimensions` component.
- Define ordered logical inputs as `[data, indices]`.
- Define the final indices Dimension as tuple depth rather than storing tuple depth in attributes.
- Document shared leading batch dimensions, indexed axes, untouched data suffix, and exact
  conceptual result formula.
- Document exact `GATHER_ND` plus `GatherNdAttrs` composition without adding a generic validator.
- Add one focused same-package structural, validation, value-semantics, and composition test.
- Keep production in the existing `io.github.pho001.synaptik.model.operation.index` package.
- Finalize Javadocs, Tensor API semantic reference, glossary, task evidence, master plan, and
  roadmap through the mandatory independent documentation pass during implementation.

## Out of scope

- public `Tensor.gatherNd`, default/no-batch overload, expression helper, factory, or task-0018F
  implementation
- axis gather, scalar select, scatter-ND, axis scatter, masks, slices, or another operation kind
- input Tensor, input count, data/index rank, Shape, Dimension, tuple depth, index data type,
  result data type, descriptor, layout, requiresGrad, provenance, label, storage, or backend facts
- storing tuple depth or output Shape in attributes; tuple depth belongs to the final indices
  Dimension of each operation occurrence
- validating `batchDimensions` against data/indices rank, matching leading batch Dimensions,
  requiring static positive tuple depth, or checking tuple depth against remaining data rank
- index-value normalization, bounds checks, negative-index policy, value access, or execution
- a default constructor, factory, singleton, separate zero-batch kind, or sentinel batch count
- arity/result/cost/fusion/backend-support metadata, registries, parsers, visitors, maps, string
  dispatch, or reflective discovery
- gradients, repeated-index accumulation, scatter-ND backward, graph capture, compiler, planning,
  prepare, runtime, backend, engine, trace, ONNX implementation, training, or execution behavior
- changing existing Java/tests, dependencies, Gradle, architecture, another module, or creating a
  task-0018F specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Documentation profile index](../../../../developer-guide/documentation/README.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0018A](0018a-scalar-select-semantics.md)
- [Task 0018C](0018c-axis-gather-semantics.md)
- [Task 0018D](0018d-axis-gather-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only legacy branch represents ONNX-style GatherND with one `GATHER_ND` identity and a
non-negative `batchDims` value. It exposes both zero-batch and explicit-batch public requests.

Its Shape rule treats the final indices extent as tuple depth `K`, requires shared leading batch
Dimensions, and forms the result from the indices prefix plus the unindexed data suffix. The legacy
implementation represents a scalar result as `[1]`; the current model instead uses canonical
rank-zero scalar Shape, which task 0018F will construct when both formula parts are empty.

The new semantic value uses the descriptive component name `batchDimensions`. It does not copy
legacy mutable Shapes, graph builders, gradient callbacks, scatter implementations, traits,
lowering, kernels, or runtime/backend behavior.

## Architecture constraints

- Production remains in `modules/model`, which owns backend-neutral operation meanings.
- `GatherNdKind` identifies tuple-index mathematical meaning only, not an occurrence, Tensor,
  graph node, descriptor, executable, kernel, or backend route.
- `GatherNdAttrs.batchDimensions` is already normalized and non-negative. It stores no rank and
  cannot prove compatibility with a particular data or indices input.
- Ordered input roles are semantically `[data, indices]`, but attributes store neither input.
- Tuple depth is occurrence-specific Shape data from the final indices Dimension and must not be
  duplicated in immutable semantic attributes.
- The Shape formula is explanatory; these contracts perform no Shape inspection or calculation.
- Generic `Operation` remains an open kind/attributes pair and does not validate family pairing,
  arity, ranks, Shapes, data types, bounds, gradients, or backend support.
- Package direction is `model.operation.index -> model.operation + java.base` only.
- Stop if implementation requires Tensor, Shape, DataType, provenance, another production type,
  another test, dependency, architecture change, or cross-layer behavior.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.operation.index.GatherNdKind` — exact tuple-index semantic
  identity.
- `io.github.pho001.synaptik.model.operation.index.GatherNdAttrs` — normalized leading batch count.
- `GatherNdSemanticsTest` — same-package focused structural, validation, and composition test.

The existing index-operation package remains cohesive and independent of public Tensor and graph
packages.

## Required contract

### Semantic kind

Create exactly:

```java
public enum GatherNdKind implements OperationKind {
    GATHER_ND
}
```

The enum declares no project field, explicit constructor, method, nested type, per-constant body,
alias, arity, batch count, tuple depth, Shape, data type, result, cost, fusion, route, or backend
metadata. Inherited `Enum.name()` satisfies `OperationKind.name()`.

Javadoc must explain ordered `[data, indices]` inputs, final-dimension tuple depth, shared batch
prefix, indexed data axes, remaining suffix, and the conceptual result formula. It must distinguish
axis gather and scalar select without claiming validation or execution.

### Normalized batch attributes

Create exactly:

```java
public record GatherNdAttrs(int batchDimensions) implements OperationAttrs
```

The record has exactly one component, one canonical constructor, one explicit documented accessor,
and record-generated `equals`, `hashCode`, and `toString`. Add no tuple depth, rank, Shape, inputs,
default constructor, factory, builder, nested type, or extra state/API.

Constructor behavior is exact:

- if `batchDimensions < 0`, throw `IllegalArgumentException` with exact message
  `batchDimensions must be non-negative: <batchDimensions>`;
- otherwise retain the primitive unchanged.

Zero and `Integer.MAX_VALUE` are structurally valid because no input ranks are present. Task 0018F
will decide whether a stored count fits one concrete data/indices pair.

### Typed composition

Valid composition is explicit:

```java
GatherNdAttrs attrs = new GatherNdAttrs(1);
Operation operation = new Operation(GatherNdKind.GATHER_ND, attrs);
```

The exact attributes reference is retained. Generic Operation does not enforce the pairing. Add no
operation factory, default attrs singleton, or compatibility matrix.

### Shape terminology and examples

For data rank `R`, indices rank `Q`, batch count `B`, and final indices extent `K`:

- `Q >= 1`;
- `0 <= B < Q`;
- leading data and indices Dimensions `[0, B)` match;
- `1 <= K <= R - B`;
- every tuple indexes data axes `[B, B + K)`;
- result Shape is `indices.shape[0:Q-1] + data.shape[B+K:R]`.

These are later input-aware validation rules, not constructor checks in this task.

Use these non-executable examples:

- data `[2, 3, 4]`, indices `[5, 2]`, `B=0`, `K=2` gives result `[5, 4]`;
- data `[2, 3, 4]`, indices `[2, 5, 1]`, `B=1`, `K=1` gives result `[2, 5, 4]`;
- data `[2, 3]`, indices `[2]`, `B=0`, `K=2` gives canonical scalar result `[]`.

No production code in this task stores or computes these Shapes.

## Affected files

Expected implementation change:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/GatherNdKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/GatherNdAttrs.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/index/GatherNdSemanticsTest.java`
- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may create or modify at most the eight paths listed above.

If implementation requires another production type, another test, existing Java edits, public
Tensor behavior, Compile API change, capability-baseline edit, dependency, build change,
architecture document, another module, or more than eight paths, stop and propose a follow-up task.

## Javadoc requirements

- Document every public type, enum constant, record component/accessor, and canonical constructor.
- Define data, indices, batch Dimensions, tuple depth, indexed axes, and untouched suffix before
  using the formula.
- Include all three concrete Shape examples and canonical scalar distinction.
- Explain why tuple depth is not duplicated in attributes.
- Explain why input rank/batch/tuple validation is deferred to task 0018F.
- State that future zero-batch convenience uses `GatherNdAttrs(0)` rather than another kind.
- Distinguish Gather-ND from scalar select, axis gather, and scatter-ND.
- Do not promise gradients, compiler capture, backend support, bounds checks, or execution.

## Acceptance criteria

- `GatherNdKind` is a public enum implementing `OperationKind` with exactly `GATHER_ND`.
- The enum adds no project-declared state, methods, constructors, nested types, or metadata.
- `GatherNdAttrs` is a public record implementing `OperationAttrs` with exactly one
  `int batchDimensions` component.
- Negative-count failure uses the exact exception type and message; valid values are unchanged.
- Record-generated value semantics remain the object contract and the accessor is documented.
- Exact `GATHER_ND` plus `GatherNdAttrs` composition works through unchanged `Operation`.
- Javadocs explain ordered inputs, formula, tuple depth, batch meaning, examples, and deferred
  validation without storing occurrence-specific facts.
- No Tensor, DataType, Shape, descriptor, provenance, graph, gradient, compiler, runtime, backend,
  or execution behavior is introduced.
- Tensor API, glossary, task evidence, master plan, and roadmap are independently reviewed and
  synchronized. Compile API, Training API, capabilities, architecture, and related contracts
  receive reasoned no-change conclusions.
- All validation passes and the final diff contains exactly the eight permitted paths.

## Tests / validation

Run:

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.operation.index.GatherNdSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests must verify exact enum/API shape, exact record component/type/accessor, zero/ordinary/
maximum retention, negative validation/message, generated value semantics, exact typed Operation
composition, distinction from axis-gather/select kinds, absence of extra kinds/state/APIs, and
absence of Tensor/DataType/Shape/layout/graph/compiler/runtime/backend dependencies.

Manual validation must inspect `javap -p -c -s`, reflection, imports, source, generated Javadoc,
Markdown links/anchors/fences/whitespace, exact eight-path scope, synchronized status, and absence
of a task-0018F specification.

## Dependencies

- Task 0005 defines minimal operation-kind and typed-attributes contracts.
- Task 0006 defines the open immutable Operation pair.
- Tasks 0018C–0018D establish the distinction from axis gather and the index package vocabulary.
- Task 0018F depends on this task for public Gather-ND construction.

## Follow-up tasks

- 0018F: public Gather-ND Tensor expressions with exact type/rank/batch/tuple/Shape validation.
- 0018I: Scatter-ND semantics and reduction policy.

Do not create either follow-up specification during this task.

## Architecture impact

Expected impact: None.

The task fills the existing model-owned operation vocabulary. If implementation requires a new
architecture rule or cross-module dependency, stop and report the issue.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0005/0006/0018A/0018C/0018D/0018E, Tensor API,
Compile API, Training API, glossary, current OperationKind/OperationAttrs/Operation and index-family
contracts/tests, and Java 26 Gradle configuration.

Implement task 0018E exactly. Add only GatherNdKind.java, GatherNdAttrs.java, and
GatherNdSemanticsTest.java under io.github.pho001.synaptik.model.operation.index.

GatherNdKind contains exactly GATHER_ND with no project state/methods/nested types/metadata.
GatherNdAttrs contains exactly one normalized non-negative int batchDimensions, exact validation/
message, and an explicit documented accessor. Document ordered [data, indices] roles, final-index-
dimension tuple depth, shared batch prefix, indexed data axes, untouched suffix, exact result-Shape
formula/examples, and exact typed pairing without performing input-aware validation.

Do not add Tensor methods, Shape/DataType/result/provenance validation, scatter/gradient types,
factories, graph/compiler/planning/runtime/backend behavior, dependencies, build/architecture
changes, existing Java edits, or later specs. Stop beyond eight paths or on uncertainty.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs
agent in the same change. It must inspect source/tests/generated Javadoc, finalize permitted
Javadocs/Tensor API/glossary/planning, record Compile API/Training API/capability/architecture and
related-contract no-change conclusions, and rerun validation.

Update task 0018E, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0018F Draft without a specification. Do not commit/push.
```

## Local decisions

- Gather-ND is represented by the sole `GatherNdKind.GATHER_ND` identity and one
  `GatherNdAttrs(batchDimensions)` value. No zero-batch alias kind, default constructor, factory,
  registry, or compatibility layer was added.
- The attributes retain only an already normalized non-negative batch count. Tuple depth remains
  the final indices Dimension because it varies with each `[data, indices]` occurrence and is not
  intrinsic operation-attribute state.
- The canonical record constructor performs one exact negative check and retains every valid
  primitive unchanged. The explicit accessor returns the stored field directly, while generated
  record equality, hashing, and diagnostic text remain the value contract.
- The submitted production Javadocs were retained unchanged after independent review because they
  already document every public type, constant, component, constructor, accessor, failure,
  formula, example, distinction, and deferred boundary required by this task.

## Known limitations

- These types provide semantic identity and one intrinsic normalized parameter only. They do not
  inspect input ranks or Shapes, validate shared batch Dimensions, obtain or validate tuple depth,
  validate index type or values, construct a result Shape/Tensor/provenance, define gradients,
  capture a graph, select a backend route, or execute indexing.
- Task 0018F remains Draft without a detailed specification and owns future public Tensor
  construction plus rank, batch-prefix, tuple-depth, index-type, and result-Shape validation.

## Validation evidence

- Clean implementation context `/root/implement_model_0018e` created exactly `GatherNdKind`,
  `GatherNdAttrs`, and `GatherNdSemanticsTest`. Independent documentation context
  `/root/implement_model_0018e/review_model_0018e_docs` inspected the shared-tree diff, final
  source/test, generated Javadoc, related contracts, APIs, glossary, planning state, and build
  configuration before finalizing documentation in the same overall change.
- The documentation pass applied General plus API/Javadoc style and Example format to the two
  production Javadocs, Tensor API, and glossary, and Planning style to this task, the model master
  plan, and roadmap. It retained both production Javadocs unchanged after verifying ordered
  `[data, indices]` roles, final-Dimension tuple depth, shared batch prefix, indexed axes,
  untouched suffix, both formula notations, all three exact examples, exact composition,
  constructor/accessor/failure behavior, zero-batch convention, family distinctions, and deferred
  boundaries.
- Reviewed architecture and process material included `AGENTS.md`, `ARCHITECTURE.md`, the focused
  current architecture, overview, lifecycle, module-boundary, dependency, and
  runtime/prepare/backend explanations; documentation rules and General/API-Javadoc/Planning/
  Example profiles; planning guide and roadmap; model capabilities/master plan; tasks 0002, 0005,
  0006, 0018A, 0018C, 0018D, and 0018E; Tensor, Compile, and Training API references; glossary;
  Java 26 root/model Gradle configuration; final implementation/tests; and generated model
  Javadoc.
- Related-contract review covered `OperationKind`, `OperationAttrs`, `Operation`, Shape/Dimension
  terminology, `SelectKind`/`SelectAttrs`, `AxisGatherKind`/`IndexAxisAttrs`, their focused tests,
  and current Tensor/provenance boundaries. The new contracts implement the open typed roles
  without changing generic kind/attributes compatibility or any existing family behavior.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.index.GatherNdSemanticsTest` — `BUILD SUCCESSFUL`; XML
  reports 9 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; XML aggregation reports 684 tests across
  80 suites with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; generated public pages contain the enum,
  constant, record component, canonical constructor, explicit accessor, exact failure, tuple/batch
  model, formula, examples, composition, distinctions, and exclusions.
- `./gradlew test` — `BUILD SUCCESSFUL`; all 36 root lifecycle tasks completed or were up-to-date
  with no failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` confirmed exactly one enum
  constant plus compiler-generated enum machinery. The record has exactly one private final
  `int batchDimensions`, one public canonical constructor with one negative check and direct
  assignment, one direct accessor, and generated `equals`, `hashCode`, and `toString`.
- Focused reflection, source, and import inspection confirmed exact public/package/interface
  structure, absence of project-declared enum API/state/nested types, exact record component and
  method set, exact validation message, generated value semantics, exact typed `Operation`
  composition, and only the permitted local `OperationKind`/`OperationAttrs` production imports.
  No Tensor, DataType, Shape, layout, provenance, graph, compiler, planning, prepare, runtime,
  backend, training, or execution dependency or behavior was introduced.
- Targeted Markdown validation resolved 403 local links, including 119 heading anchors, across the
  five changed documentation/planning files. All 230 code-fence markers are balanced, targeted
  trailing-whitespace scans found no matches, all eight paths have final newlines, and
  `git diff --check` passes.
- Final changed-path inventory contains exactly the eight authorized paths: two production
  contracts, one focused test, Tensor API, glossary, this task, model master plan, and roadmap.
  Task/master-plan/roadmap status is synchronized as Complete. Task 0018F remains Draft, and no
  task-0018F specification exists.
- Compile API remains accurate unchanged because this task adds no Tensor expression, graph
  capture, inference, validation, optimization, compile artifact, or engine behavior. Training API
  remains accurate unchanged because no gradient, autograd, parameter, optimizer, publication,
  session, or training execution behavior changed. The capability baseline already lists
  `gatherNd` with batch dimensions and separates model, public API, compiler, planning, backend,
  runtime, and test responsibilities, so it required no edit.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance tests, integration tests, Gradle/dependencies, and other modules remain
  accurate unchanged because the task stays inside model-owned semantics and changes no module
  boundary, dependency rule, backend behavior, end-to-end behavior, or build requirement.
- Existing Operation, Shape, DataType, Tensor, provenance, select, axis-gather, and other
  operation-family contracts remain accurate unchanged because this task adds only independent
  immutable semantic values. Scatter-ND, gradients, compiler/backends, and every later task remain
  deferred; no later specification was created.

## Implementation notes

- Added `GatherNdKind` with exactly `GATHER_ND` and no project-declared behavior or metadata.
- Added `GatherNdAttrs(int batchDimensions)` with exact non-negative validation, unchanged
  retention, a documented direct accessor, and record-generated object methods.
- Added nine focused tests covering exact enum/record structure, retained values, exact failure,
  generated value semantics, typed composition, family distinctions, and dependency boundaries.
- Finalized the Tensor API, glossary terminology and inventories, and synchronized planning
  evidence/status through the independent documentation pass. No production Javadoc correction
  was required.

## Completion summary

- Completed changes: Implemented and documented first-class tuple-index `GATHER_ND` semantics and
  immutable normalized shared-batch attributes without public Tensor or cross-layer behavior.
- Files changed or created: Exactly the eight authorized production, test, API, glossary, task,
  master-plan, and roadmap paths.
- Tests and validation: Focused 9-test and all 684-model-test/80-suite runs, model Javadoc, root
  tests, javap/reflection/import/source/generated-page review, 403-link/119-anchor checks,
  fence/whitespace/newline checks, exact scope/status and no-0018F-spec checks, and
  `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0018e/review_model_0018e_docs` completed the required independent pass
  with General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API and glossary now explain Gather-ND ordered roles, tuple depth,
  batch prefix, indexed axes, untouched suffix, formula, examples, exact composition, family
  distinctions, and current-versus-planned boundaries. Compile API, Training API, capabilities,
  architecture/ADRs/tests, conformance/integration material, Gradle, dependencies, other modules,
  and later tasks remain accurate unchanged for the recorded reasons.
- Javadoc review: Both new public types, the enum constant, record component, canonical
  constructor, accessor, failure, formula, examples, and exclusions are complete; no correction
  was required.
- Glossary impact: Added reusable Gather-ND terminology and synchronized `OperationKind` and
  `OperationAttrs` current-family inventories.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0018E. Task 0018F remains Draft without a detailed
  specification.

Status: Complete
