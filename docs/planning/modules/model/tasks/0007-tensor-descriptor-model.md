# Task 0007: Tensor Descriptor Model

## Status

Complete

## Goal

Define the minimal immutable, backend-independent `TensorDescriptor` value that combines one
non-null `DataType`, one non-null `Shape`, an explicit resolved-or-unresolved layout state, and a
`requiresGrad` flag. The descriptor validates only relationships that these completed model
contracts can prove, without adding public tensor state, graph identity, storage, inference,
execution, or backend facts.

## Scope

- Add one public `TensorDescriptor` record in the existing `model.tensor` package.
- Store exactly `DataType dataType`, `Shape shape`, `Optional<LayoutDescriptor> layout`, and
  `boolean requiresGrad`, in that order.
- Use a non-null `Optional`: present means the numeric layout is resolved; empty means layout
  geometry is unresolved. Never use `null` as the unresolved-layout sentinel.
- Permit unresolved layout for both dynamic and fully static shapes.
- Require every present layout to be compatible with the paired fully static shape.
- Reject `requiresGrad == true` when the data type is not differentiable.
- Use record-generated structural equality, hashing, and diagnostic text over all four components.
- Add one focused unit-test class for construction, validation, value semantics, edge shapes,
  layout geometry, gradient eligibility, ownership, and diagnostics.
- During implementation, update the public Tensor API, glossary, task evidence, model master plan,
  and implementation roadmap through the required clean-context documentation pass.

## Out of scope

- mutable public `Tensor`, `TensorFactory`, host or device storage, buffers, allocation, residency,
  or physical memory ownership
- `TensorId`, labels, trainable-parameter state, gradient values or publication state, and
  provenance
- `Operation`, operation families, graph IDs, graph values, graph nodes, compiled graph state, or
  publication bindings
- shape, data type, or layout inference; symbolic dimension binding; layout resolution; reshape or
  view creation; compiler transformations; and graph validation
- runtime, prepare, backend, device, kernel, route, capability, cost, transfer, or execution facts
- materialization policy or a `requiresMaterialization`-style decision
- convenience factories, overloaded constructors, layout-state wrapper types, additional
  production helpers, or changes to completed data type, shape, dimension, or layout contracts
- Gradle, dependency, capability-baseline, architecture-contract, focused-architecture, or other
  module changes

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially `modules/model` ownership,
  the public Tensor versus immutable graph distinction, and forbidden model dependencies
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Model capability baseline](../capabilities.md), especially the data type, shape, layout, and
  public Tensor baselines
- [Model master plan](../master-plan.md), especially the `model.tensor` package and package
  dependency direction
- [Task 0002](0002-shape-and-dimension-model.md), which defines `Shape` and dynamic dimensions
- [Task 0003](0003-layout-descriptor-model.md), which defines resolved static layout geometry
- [Task 0003B](0003b-shape-package-migration.md) and
  [task 0003C](0003c-layout-package-migration.md), which establish the current packages
- [Tensor API](../../../../api/tensor-api.md) and [glossary](../../../../glossary.md), which explain
  the completed component contracts and current terminology

## Legacy evidence

The read-only `legacy/pre-rewrite` branch provides capability evidence through
`tensor.TensorMetadata`, `planning.descriptor.CompiledTensorDescriptor`, its builder and index, and
`CompiledTensorDescriptorIndexTest`.

Legacy `TensorMetadata` carried data type, shape, strides, storage offset, and `requiresGrad`, but it
was mutable and also owned label, trainable-parameter state, flat-index helpers, and storage-oriented
geometry. It defaulted a null data type and allowed later mutation. The new descriptor preserves the
useful logical facts as a strict immutable value; it does not copy those mutable, labeling,
trainable, indexing, defaulting, or storage-oriented responsibilities.

Legacy `CompiledTensorDescriptor` was an immutable snapshot, but it mixed tensor facts with a node
ID, operation type, input node IDs, leaf/backward flags, byte lengths, planning layout classes, and
facts consumed by prepare/runtime code. Its tests verify graph lookup and runtime snapshot behavior
in addition to defensive array handling. The new descriptor does not copy graph, planning,
publication, runtime, storage-size, or execution responsibilities. It composes the new immutable
`DataType`, `Shape`, and `LayoutDescriptor` contracts instead of importing the legacy arrays,
packages, builder/index structure, or architecture.

Legacy training compilation could accept `requiresGrad` on an integral tensor until a later compile
failure. That behavior is not retained: the current capability baseline states that only floating
data types are differentiable, so this model value rejects an impossible gradient requirement at
construction.

## Architecture constraints

- All production packages remain below `io.github.pho001.synaptik.*`.
- `TensorDescriptor` lives in `io.github.pho001.synaptik.model.tensor` and composes only public
  contracts from `model.datatype`, `model.shape`, and `model.layout` plus the JDK.
- Package direction is `model.tensor -> model.datatype`, `model.shape`, and `model.layout`.
  Foundational packages must not depend on `model.tensor`, storage, operation, graph, or any higher
  module.
- The descriptor is immutable and backend-independent. Its complete instance state is the four
  record components specified by this task.
- A resolved layout is a present numeric `LayoutDescriptor`; an unresolved layout is
  `Optional.empty()` because numeric strides and span are not yet known. Unresolved does not mean
  invalid, absent tensor geometry, or a runtime fallback.
- Dynamic shapes require unresolved layout because `LayoutDescriptor` supports only fully static
  shapes. Resolving symbolic dimensions belongs to later compiler/runtime contracts.
- Fully static shapes may also have unresolved layout. A static shape can exist as a compile-time
  descriptor before layout inference or resolution, and this task must not force a default layout,
  perform compiler inference, or imply runtime materialization behavior.
- Every present layout must be reconstructed against the paired shape through the public
  `LayoutDescriptor.of(shape, layout.strides(), layout.storageOffset(), layout.isView())` contract
  and must compare equal to the supplied layout. This validates rank, strides, offset, view flag,
  derived kind, and referenced span rather than relying on rank alone.
- Reconstruction validates geometric compatibility, not source provenance. Because
  `LayoutDescriptor` deliberately does not retain its source shape, an equal layout value that is
  valid for more than one shape may be paired with any of those shapes. The descriptor must not
  invent source-shape identity or change `LayoutDescriptor` to obtain it.
- `requiresGrad == true` is valid exactly when `dataType.isDifferentiable()` is true. This records
  model eligibility only; it does not promise an operation gradient rule or backend support.
- No component may contain or derive storage, operation, graph, compiler, planning, backend,
  runtime, device, or materialization state.
- If implementation requires a new layout-state type, another production helper, a component
  contract change, or an architecture/dependency change, stop and report the issue.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.datatype` — owns the immutable `DataType` enum.
- `io.github.pho001.synaptik.model.shape` — owns immutable static and dynamic `Shape` values.
- `io.github.pho001.synaptik.model.layout` — owns immutable resolved `LayoutDescriptor` values.
- `io.github.pho001.synaptik.model.tensor` — already owns `TensorId` and is the planned owner of
  public tensor contracts.

Packages added or changed:

- No package is added. The existing `model.tensor` package gains one public descriptor.

Type placement:

- `io.github.pho001.synaptik.model.tensor.TensorDescriptor` — immutable logical tensor facts shared
  by later public Tensor and graph-value contracts without storage or graph identity.

Test placement:

- `io.github.pho001.synaptik.model.tensor.TensorDescriptorTest` — mirrors the production package
  and exercises the public value contract without package-private access.

## Required contract

Implement exactly this public record shape:

```java
public record TensorDescriptor(
        DataType dataType,
        Shape shape,
        Optional<LayoutDescriptor> layout,
        boolean requiresGrad) { ... }
```

A record is the smallest Java form that makes the complete state final and supplies structural
value semantics without a parallel builder or hierarchy. The component name `layout` is retained
because its `Optional` type makes the resolved-or-unresolved state explicit at the API boundary.
Callers construct a resolved descriptor with `Optional.of(layout)` and an unresolved descriptor
with `Optional.empty()`.

The compact canonical constructor must use `Objects.requireNonNull` for component null checks and:

1. reject null `dataType`, `shape`, or `layout` references with `NullPointerException` and
   component-specific messages;
2. when `layout` is present, reject a dynamic shape with `IllegalArgumentException`;
3. reconstruct the present layout from the paired shape and the layout's public strides, offset,
   and view flag, then reject it with `IllegalArgumentException` when the reconstructed value is
   not equal to the supplied value; and
4. reject `requiresGrad == true` with `IllegalArgumentException` when
   `dataType.isDifferentiable()` is false.

`LayoutDescriptor.of(...)` exceptions remain observable during reconstruction. In particular,
checked layout arithmetic overflow propagates as `ArithmeticException`; null cannot arise from a
present `Optional`; and rank or other argument incompatibility is an `IllegalArgumentException`.
Do not catch these failures and replace them with a sentinel or silently discard the layout.

The record stores the validated components through ordinary record assignment and does not copy or
wrap them: `DataType` is an enum, and `Shape` and `LayoutDescriptor` are immutable values whose own
contracts already isolate mutable arrays. The supplied non-null `Optional` may therefore be stored
unchanged, but `Optional` is a value-based class and its container identity is not part of this
public contract. Callers and tests compare its value and inspect presence; they must not use `==`,
`assertSame`, identity hash codes, or synchronization on the `Optional`. When layout is present,
`layout().orElseThrow()` returns the exact immutable `LayoutDescriptor` reference contained in the
supplied optional.

Do not override `equals`, `hashCode`, or `toString`. Record-generated equality and hashing cover
all four components structurally. Diagnostic text must therefore expose the component names and
representative resolved or unresolved layout state, but it is not a serialization, parser, or
runtime dispatch contract.

## Affected files

Expected production file:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorDescriptor.java`

Expected test file:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorDescriptorTest.java`

Expected documentation and planning updates during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may create or modify at most:

- one production Java file;
- one focused test Java file; and
- the five documentation and planning files listed above.

Do not modify existing Java source or tests, Gradle files, `ARCHITECTURE.md`, focused architecture
documentation, the capability baseline, another module, or unrelated documentation. If another
file or type is required, stop and propose a separately reviewed follow-up instead of expanding
this task.

## Javadoc requirements

- The record Javadoc must define a tensor descriptor, explain resolved versus unresolved layout,
  distinguish a descriptor from public mutable `Tensor`, graph values/nodes, and storage, and state
  the compiler/runtime/backend/materialization boundaries.
- Record-component `@param` tags must document non-nullness, semantic ownership, the meaning of
  present/empty layout, `requiresGrad` eligibility, and unchanged retention of the contained layout
  value without promising identity for the value-based `Optional` container.
- The canonical constructor Javadoc must document all four inputs and separately describe
  `NullPointerException`, `IllegalArgumentException`, and reconstruction `ArithmeticException`
  conditions.
- Explicit `dataType()` and `shape()` Javadocs must each include `@return`, non-nullness, the stored
  immutable value, ownership, and the value's meaning.
- Explicit `layout()` Javadoc must include `@return`, non-nullness, present/empty semantics, and
  value equality. It must prohibit reliance on `Optional` container identity and state that a
  present result contains the exact immutable `LayoutDescriptor` reference supplied at
  construction.
- Explicit `requiresGrad()` Javadoc must state that true means requested model-level gradient
  eligibility for a differentiable data type, not that an operation/backend can differentiate it.
- The type Javadoc must describe record-generated equality and hashing over all components and
  diagnostic, non-serialization `toString()` text.
- Javadoc must explain that a fully static shape may remain unresolved and that present-layout
  reconstruction proves public geometric compatibility but not original source-shape identity.
- Javadoc must be meaningful contract documentation. No public constructor, component, or
  accessor may be left with only implicit generated documentation.

## Acceptance criteria

- `TensorDescriptor` is a public record with exactly the four required components, order, and
  types and has no additional instance state.
- Construction retains the validated immutable data type and shape values and, when present, the
  exact underlying `LayoutDescriptor` reference. The non-null `Optional` is tested through value
  equality and presence, never container identity, and no mutable array can be obtained through the
  descriptor.
- Null data type, shape, or optional layout references independently fail with
  `NullPointerException`; unresolved layout is represented only by non-null `Optional.empty()`.
- Dynamic shapes construct successfully with unresolved layout and fail with
  `IllegalArgumentException` when a resolved layout is supplied.
- Fully static ordinary, scalar, and zero-sized shapes construct successfully with unresolved
  layout, proving layout resolution is not mandatory for static compile-time descriptors.
- Present canonical contiguous layouts for ordinary, scalar, and empty shapes are accepted.
- Present zero-stride broadcast-view and non-zero-offset layouts are accepted when reconstruction
  against the paired shape produces the same value.
- Present layouts with a different rank are rejected.
- A same-rank layout derived for an incompatible shape is rejected when reconstruction changes its
  derived kind or span; tests prove validation is stronger than rank equality.
- An equal layout geometry that reconstructs identically for another compatible shape is accepted;
  no source-shape provenance requirement is invented.
- Every `DataType` is covered: `requiresGrad == true` is accepted for `FLOAT64`, `FLOAT32`, and
  `BFLOAT16`, and rejected for `INT32`, `INT64`, and `BOOL`; false is accepted for all six types.
- Equal components produce equal descriptors and equal hash codes. Changing data type, shape,
  layout state/value, or `requiresGrad` produces an unequal descriptor.
- Diagnostic text identifies `TensorDescriptor`, `dataType`, `shape`, `layout`, and
  `requiresGrad`, distinguishes resolved from unresolved examples, and is not asserted as an exact
  wire format.
- Tests cover all constructor failures, resolved/unresolved states, scalar, empty, dynamic,
  contiguous, offset, zero-stride, defensive/value semantics, gradient eligibility, equality,
  hashing, and diagnostics required above.
- No factory, builder, layout-state wrapper, storage, identity, label, trainable/gradient value,
  provenance, operation, graph, publication, inference, runtime, backend, device, or
  materialization contract is introduced.
- Production imports are limited to the three owning model packages, `java.util.Objects`, and
  `java.util.Optional`; no project-module dependency is added.
- All Javadoc requirements are satisfied and generated model Javadoc includes the record,
  canonical constructor, components, and explicit accessors.
- A separate documentation-focused agent or thread with clean context independently reviews the
  final implementation and tests, finalizes Javadoc, updates the Tensor API and glossary, and
  records terminology, link, example, status, and formatting evidence in the same overall change.
- Task, master-plan row/current status/notes, and roadmap frontier/table have matching final status.
- No existing Java/test file, Gradle file, architecture document, capability baseline, other
  module, or unrelated documentation is changed.

## Tests / validation

Run after implementation and again after the documentation-focused pass:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorDescriptorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Manually verify:

- the final diff contains only one new production file, one new test file, and the five allowed
  documentation/planning files;
- reflection reports a record with exactly the four required components in the specified order and
  types, and no additional declared instance field;
- constructor inspection shows only component null checks, present-layout compatibility
  reconstruction, and differentiability validation, with no inference or hidden defaulting;
- tests include rank-equal incompatible geometry, not only rank mismatch, and include compatible
  shape reuse where reconstructed public geometry remains equal;
- dynamic, static-unresolved, scalar, empty, zero-stride view, and offset cases have the specified
  results;
- tests compare the `Optional` layout component by value and presence, never with `==` or
  `assertSame`; a resolved case verifies exact identity only for
  `descriptor.layout().orElseThrow()` and its originally supplied `LayoutDescriptor`;
- all six data types have explicit `requiresGrad` coverage;
- generated Javadoc documents the four components/accessors, resolved/unresolved state,
  reconstruction semantics, failures, ownership, value semantics, diagnostics, and cross-layer
  exclusions;
- `docs/api/tensor-api.md` presents `TensorDescriptor` as implemented, explains the explicit
  resolved/unresolved layout state and static-unresolved case, and does not claim inference,
  storage, graph, materialization, runtime, or backend behavior;
- `docs/glossary.md` marks the descriptor implemented and aligns tensor descriptor, layout,
  gradient, and current-versus-planned terminology without changing architecture authority;
- local Markdown links and anchors in all five changed documentation/planning files resolve, code
  fences are balanced, and no changed file has trailing whitespace;
- the separate documentation context follows
  `docs/developer-guide/documentation-rules.md`, applies General style plus API and Javadoc style to
  API/Javadoc work and Planning style to planning updates, and records its identity, inspected
  diff/source/tests, commands, results, limitations, Javadoc review, and glossary impact; and
- package direction remains `model.tensor -> datatype/shape/layout` with no storage, operation,
  graph, backend, compiler, planning, prepare, runtime, or engine dependency.

## Dependencies

- Tasks 0001–0003 and package migrations 0003A–0003C are complete and provide the component
  contracts in their current packages.
- Task 0004 established the `model.tensor` package through `TensorId`; it is complete by ordered
  frontier but is not a component dependency of this descriptor.
- Tasks 0005–0006 are complete by ordered frontier but are not dependencies of this descriptor.

## Follow-up tasks

- Task 0008 will compose `TensorDescriptor` into immutable graph values. It owns graph identity,
  node/value relationships, and graph occurrence semantics.
- Task 0011 will reuse `TensorDescriptor` from public mutable `Tensor` state rather than duplicating
  descriptor validation. It owns labels, tensor identity, gradient value/state, trainable metadata,
  host-storage association, and Tensor lifecycle.
- Later compiler contracts own shape/data type/layout inference and symbolic binding. Planning,
  prepare, runtime, and backends own their architecture-assigned materialization and execution
  responsibilities.

Do not create a detailed task specification for task 0008 or any later task as part of this work.

## Architecture impact

Expected impact: None.

The architecture already assigns `TensorDescriptor` and public tensor model facts to
`modules/model`. This task adds one immutable value in the planned package and uses the existing
package direction. It changes no module boundary, dependency rule, lifecycle contract, or backend
contract, so architecture documentation and architecture tests require no update. If implementation
reveals otherwise, stop and report the conflicting rule and required decision before editing
architecture files.

## Implementation prompt

Use this prompt in a separate agentic task/thread with a clean context:

```text
You are a clean-context implementation agent working in the Synaptik repository.

Read first and in full:
- AGENTS.md
- ARCHITECTURE.md
- docs/developer-guide/documentation-rules.md
- docs/planning/planning-guide.md
- docs/planning/modules/model/capabilities.md
- docs/planning/modules/model/master-plan.md
- docs/planning/modules/model/tasks/0002-shape-and-dimension-model.md
- docs/planning/modules/model/tasks/0003-layout-descriptor-model.md
- docs/planning/modules/model/tasks/0003b-shape-package-migration.md
- docs/planning/modules/model/tasks/0003c-layout-package-migration.md
- docs/planning/modules/model/tasks/0007-tensor-descriptor-model.md
- docs/api/tensor-api.md
- docs/glossary.md
- the relevant completed datatype, shape, and layout production/test files

Implement task 0007 exactly as specified. Create only TensorDescriptor.java and
TensorDescriptorTest.java for code and tests. Use a public record with exactly DataType dataType,
Shape shape, Optional<LayoutDescriptor> layout, and boolean requiresGrad. Use Optional.empty() for
unresolved layout; never use null as a sentinel. Permit fully static unresolved descriptors. Reject
resolved layouts for dynamic shapes. Validate every present layout by reconstructing it against the
paired shape through LayoutDescriptor.of(...) and comparing the complete value, not rank alone.
Reject requiresGrad for every non-differentiable data type. Use Objects for component null checks.
Store the supplied non-null Optional through ordinary record assignment, but treat it as a
value-based container: tests and Javadoc must use equality/presence rather than Optional identity.
For a resolved descriptor, verify exact reference retention only for the underlying
LayoutDescriptor obtained through layout().orElseThrow().

Do not add factories, overloads, helpers, layout-state types, storage, tensor identity/labels,
trainable or gradient values/state, provenance, operations, graph contracts, publication,
inference, symbolic binding, materialization policy, compiler/planning/runtime/backend/device facts,
dependencies, or unrelated refactors. Do not modify existing Java contracts/tests, Gradle,
ARCHITECTURE.md, focused architecture docs, capabilities.md, another module, or unrelated docs.
Stop and report if a required change exceeds the affected-file or maximum-scope list, if public
component contracts cannot perform the specified compatibility check, or if architecture
uncertainty appears.

Add every Javadoc contract required by the task. Run every validation command and manual check.

After implementation and initial validation, hand the resulting diff to a separate
documentation-focused agent or thread with a clean context in the same overall change. The handoff
must include this task, the implementation diff, affected descriptor API/behavior, architecture
constraints, required Tensor API and glossary updates, Javadoc requirements, and all validation
commands. That agent must read AGENTS.md, ARCHITECTURE.md,
docs/developer-guide/documentation-rules.md, General style, API and Javadoc style, Planning style,
this task, final source/tests, docs/api/tensor-api.md, and docs/glossary.md. It must independently
finalize TensorDescriptor Javadoc, the API explanation, glossary/status terminology, planning
evidence/status, links, anchors, examples, and formatting. It must inspect the source and tests,
not rely only on the handoff summary, and record reasoned reviews of existing DataType, Shape, and
LayoutDescriptor Javadocs, including why they remain accurate without changes or why an
out-of-scope discrepancy requires stopping.

At the end, update only this task file, the model master plan, and the roadmap for planning status.
Record local decisions, known limitations, exact validation evidence including the documentation
agent identity and results, implementation notes, and the canonical completion summary. Do not mark
task 0007 Complete until all acceptance criteria, both validation passes, documentation changes,
and status synchronization are complete. Do not create a task-0008 specification.
```

## Local decisions

- `Optional<LayoutDescriptor>` is the smallest explicit layout-state representation. A sealed
  wrapper would add a production type without another state or behavior, while a nullable layout
  would hide unresolved state behind a forbidden sentinel.
- The supplied `Optional` needs no defensive wrapper, but its value-based JDK contract means
  container identity is deliberately not observable or tested. Value equality and presence define
  the layout-state contract; a present optional retains its immutable `LayoutDescriptor` value.
- Fully static unresolved descriptors are valid because shape resolution and layout resolution are
  distinct compile-time facts. The record does not synthesize a contiguous layout or imply later
  materialization.
- Present-layout reconstruction plus full value equality is the strongest compatibility check
  available through current public contracts. It validates every exposed derived layout fact but
  deliberately does not invent source-shape provenance.
- Gradient eligibility is enforced in the record because `DataType.isDifferentiable()` is an
  existing model contract and the capability baseline restricts differentiation to floating types.
- Record-generated equality, hashing, and text cover the complete state; additional aliases,
  factories, and custom rendering would broaden the API without a current need.

## Known limitations

- An unresolved layout carries no symbolic strides, offset, kind, or span. Later owning contracts
  must resolve layout rather than interpreting empty as contiguous.
- Reconstruction cannot identify the historical source shape of a layout because
  `LayoutDescriptor` intentionally does not store one. Geometrically identical values may be valid
  for multiple paired shapes.
- `requiresGrad` records eligibility/request only. It does not contain a gradient, identify a
  parameter, create backward operations, or prove operation/backend differentiability.
- The descriptor does not validate storage capacity because it owns no storage.

## Validation evidence

- Clean planning/documentation context `/root/plan_model_0007` read the complete architecture,
  documentation-workflow, General style, Planning style, planning, capability, task-0002,
  task-0003, task-0003B, task-0003C, task-0006, API, glossary, and relevant current production/test
  contracts before defining this task.
- Read-only legacy inspection used `git ls-tree`, `git grep`, and `git show
  legacy/pre-rewrite:<path>` for `TensorMetadata`, `CompiledTensorDescriptor`, its builder/index,
  and descriptor tests. The branch was not checked out or modified, and no legacy source or
  package structure was copied.
- `git status --short`, `git diff --name-only`, and manual scope review confirmed exactly three
  planning-document changes: this new task, the model master plan, and the roadmap. No Java, test,
  Gradle, `ARCHITECTURE.md`, focused architecture, capability-baseline, other-module, API, glossary,
  or unrelated documentation file changed during planning.
- A targeted Ruby path-and-heading check validated all 50 local Markdown links and anchors in the
  three changed documents.
- Fence inspection reported even counts in every changed document (six backtick-fence markers in
  this task and two each in the master plan and roadmap). `rg -n '[[:blank:]]+$'` returned no
  trailing-whitespace matches.
- `git diff --check` passed for tracked changes. `git diff --no-index --check /dev/null
  docs/planning/modules/model/tasks/0007-tensor-descriptor-model.md` produced no whitespace
  diagnostics; its exit status was the expected `1` because the complete new file differs from
  `/dev/null`.
- Status review confirmed task 0006 remains `Complete` and task 0007 is `Ready` in this task, the
  master-plan row/current status/notes, and the roadmap frontier/table. Task order and dependencies
  are unchanged, task 0008 remains `Draft`, and no task-0008 specification exists.
- Follow-up planning review added `java.util.Objects` to the exact production import allowance and
  removed `Optional` container-identity requirements from the contract, Javadoc, tests, manual
  checks, implementation prompt, and local decisions. Repeated validation still resolved all 50
  local links/anchors, found balanced fences and no trailing whitespace, passed
  `git diff --check`, produced no new-file whitespace diagnostics, and confirmed the same
  three-file scope and synchronized `Ready` status.
- Gradle tests and Javadoc were not run during this planning-only change because no Java, test,
  dependency, or build file changed. The implementation task requires the focused model test,
  complete model tests, model Javadoc, and full repository tests both before and after the separate
  documentation pass.
- The implementation context added only
  `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorDescriptor.java` and
  `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorDescriptorTest.java`.
  Independent source, test, reflection, and bytecode inspection confirmed the required four record
  components, four instance fields, one canonical constructor, component-specific null checks,
  present-layout reconstruction through `LayoutDescriptor.of(...)`, complete layout equality,
  differentiability validation, ordinary component assignment, and record-generated equality,
  hashing, and diagnostic text. No inference, hidden default, or additional instance state exists.
- Clean documentation-focused context `/root/review_model_0007_docs` applied General style, API and
  Javadoc style, Planning style, and Example format after reading the complete architecture,
  documentation workflow, planning guide, roadmap, capability baseline, model master plan, tasks
  0002/0003/0003B/0003C/0007, Tensor API, glossary, and relevant current production/test sources.
  It independently inspected the final implementation, tests, generated Javadoc, test results, and
  diff rather than relying on the implementation handoff.
- The documentation pass changed only `TensorDescriptor` Javadoc among Java sources. It finalized
  purpose, resolved/unresolved layout semantics, fully static unresolved values, `Optional` value
  semantics, exact underlying-layout retention, complete geometric reconstruction versus source
  provenance, gradient eligibility, constructor failures including reconstruction overflow,
  explicit accessor contracts, record value/diagnostic semantics, and Tensor, graph, storage,
  compiler, planning/materialization, prepare, runtime, and backend boundaries. Constructor logic,
  imports, record declaration, accessors, and tests remained unchanged during this pass.
- Existing `DataType`, `Shape`, and `LayoutDescriptor` Javadocs were reviewed without modification.
  `DataType.isDifferentiable()` already defines floating-only semantic eligibility and disclaims
  operation/backend support; `Shape` already defines immutable static/dynamic extents and excludes
  layout/storage/execution state; and `LayoutDescriptor` already defines resolved fully static
  geometry, no retained source shape, defensive stride ownership, and storage/materialization/device
  boundaries. Those contracts remain accurate for composition by `TensorDescriptor`.
- The Tensor API now presents `TensorDescriptor` as current, explains its four components and
  compatibility/failure boundaries, removes it from planned contracts, and includes a complete
  beginner-oriented example covering static unresolved, static resolved, and dynamic unresolved
  values. `javac -cp modules/model/build/classes/java/main -d
  /tmp/synaptik-doc-example /tmp/TensorDescriptorExample.java` passed, and `java -cp
  modules/model/build/classes/java/main:/tmp/synaptik-doc-example TensorDescriptorExample` printed
  `true`, `DENSE_CONTIGUOUS`, `6`, and `Shape[batch, 3]` on separate lines.
- The glossary implementation-status convention and Tensor descriptor entry now mark the contract
  implemented and define resolved/unresolved `Optional` state, full geometric compatibility,
  source-provenance limits, gradient eligibility, value semantics, and architecture boundaries.
  No other glossary term changed meaning or implementation status.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorDescriptorTest` — passed in the documentation context;
  12 tests, zero failures, errors, or skips.
- `./gradlew :modules:model:test` — passed in the documentation context; 90 tests, zero failures,
  errors, or skips across the model module.
- `./gradlew :modules:model:javadoc` — passed after the Javadoc revision. Generated output contains
  the record, canonical constructor, all explicit accessors, component contracts, three documented
  exception categories, resolved/unresolved semantics, ownership, equality/diagnostic semantics,
  and cross-layer boundaries.
- `./gradlew test` — passed for the complete repository with 36 actionable tasks in the final run.
- `javap -classpath modules/model/build/classes/java/main -p
  io.github.pho001.synaptik.model.tensor.TensorDescriptor` and the corresponding `-p -c` inspection
  passed the record-shape, field, constructor, assignment, reconstruction, and generated-method
  checks. Production imports are exactly the three owning model packages plus `Objects` and
  `Optional`; package direction remains `tensor -> datatype/shape/layout`.
- A targeted Ruby path-and-heading check resolved all 97 local Markdown links and anchors in the
  five changed documentation/planning files. Fence counts are balanced, trailing-whitespace checks
  found no matches, terminology agrees across Javadoc/API/glossary/planning, and generated Javadoc
  contains `TensorDescriptor` in the tensor package summary and all-classes index.
- Final scope review confirmed exactly seven changed or new repository files: the new production
  record and focused test plus the Tensor API, glossary, this task, model master plan, and roadmap.
  No existing Java/test contract, Gradle file, architecture document, capability baseline, other
  module, or unrelated documentation changed. No task-0008 specification exists.
- `git diff --check` passed after documentation and planning synchronization.
- The final coordinating context independently reran the focused descriptor test, all model tests,
  model Javadoc, and the complete repository test lifecycle — every command reported
  `BUILD SUCCESSFUL`. Final XML inspection recorded 90 tests with zero failures, errors, or skips;
  `javap -p` reconfirmed exactly four component fields and accessors; `git diff --check` passed;
  and the final workspace still contained exactly the seven allowed files and no task-0008
  specification.

## Implementation notes

- Added the exact four-component public `TensorDescriptor` record in `model.tensor` with explicit
  unresolved layout, full public-geometry compatibility validation, and gradient eligibility.
- Added one focused 12-test class covering record shape, null validation, dynamic/static unresolved
  states, scalar/empty/contiguous/offset/broadcast layouts, rank and same-rank incompatibility,
  compatible geometry reuse, overflow propagation, all data types, ownership, value semantics,
  equality, hashing, and diagnostics.
- Finalized the new record's Javadoc and moved Tensor descriptor documentation from planned to
  current in the Tensor API and glossary, including a compiled concrete example.
- Synchronized this task, the model master plan, and the roadmap after all required validation
  passed. Task 0008 remains a `Draft` row without a detailed specification.

## Completion summary

- Completed changes: Implemented and documented the immutable backend-independent tensor
  descriptor with explicit resolved/unresolved layout state, geometric compatibility validation,
  and differentiable-data-type eligibility.
- Files changed or created: One production record, one focused unit-test class, the Tensor API,
  glossary, this task specification, model master plan, and implementation roadmap.
- Tests and validation: Focused 12-test suite, all 90 model tests, model Javadoc, full repository
  tests, compiled documentation example, reflection/bytecode inspection, link/anchor checks, fence
  and whitespace checks, scope review, and `git diff --check` passed.
- Documentation-agent review: Clean context `/root/review_model_0007_docs` independently reviewed
  implementation, tests, generated output, component Javadocs, API explanation, glossary impact,
  architecture boundaries, and synchronized planning evidence.
- Documentation impact: `TensorDescriptor` is now a current public model contract with a complete
  concrete example; planned Tensor, graph, storage, compiler, materialization, runtime, and backend
  responsibilities remain separate.
- Javadoc review: New `TensorDescriptor` Javadoc is complete. Existing `DataType`, `Shape`, and
  `LayoutDescriptor` Javadocs remain accurate for the reasons recorded in validation evidence and
  required no out-of-scope edits.
- Glossary impact: The implementation-status convention and Tensor descriptor definition now
  reflect the implemented contract; no unrelated terminology changed.
- Unresolved issues: None.
- Follow-up required: None. Task 0008 is the next planning frontier and remains `Draft` without a
  detailed task specification.

Status: Complete
