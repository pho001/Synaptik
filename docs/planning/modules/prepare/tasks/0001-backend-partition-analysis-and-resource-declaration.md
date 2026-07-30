# Task 0001: Backend Partition Analysis and Resource Declaration

## Status

Complete

## Goal

Replace the Prepare placeholder with the smallest typed analysis-side collaboration for the first
stage of [ADR 0010](../../../../design/decisions/0010-staged-backend-preparation.md). A concrete
backend must be able to analyze one fully resolved planned partition, retain its selected
lowering/route opaquely, and declare exact shared buffer/workspace requirements before Runtime
assigns slots.

The exact public surface is:

```java
package io.github.pho001.synaptik.prepare.analysis;

public interface BackendAnalysisInputs {}

public interface BackendPreparationPlan {}

public record PrepareContext<I extends BackendAnalysisInputs>(
        PlannedPartition partition,
        List<CompiledNode> nodes,
        List<GraphValue> values,
        List<LogicalMemoryRequirement> memoryRequirements,
        Map<ValueId, ScalarValue> constants,
        I backendInputs) {}

public sealed interface PreparationResourceRequirement {
    record Buffer(ValueId valueId, long byteSize, long byteAlignment)
            implements PreparationResourceRequirement {}

    record Workspace(long requirementId, long byteSize, long byteAlignment)
            implements PreparationResourceRequirement {}
}

public record BackendPartitionAnalysis<P extends BackendPreparationPlan>(
        PlannedPartition partition,
        P plan,
        List<PreparationResourceRequirement> requirements) {}

public interface BackendPartitionPreparer<
        I extends BackendAnalysisInputs,
        P extends BackendPreparationPlan> {
    BackendPartitionAnalysis<P> analyze(PrepareContext<I> context);
}
```

`BackendAnalysisInputs` is an opaque typed role implemented by one concrete backend's immutable
input record. That record carries the resolved target/backend capabilities, applicable
configuration, and compatible cached decision needed by that backend. `BackendPreparationPlan`
is the corresponding opaque immutable result retaining selected lowering, route, and private
configuration for later finalization. Shared Prepare interprets neither role.

## Scope

- Delete `PrepareModule`.
- Add the exact six public declarations above and `prepare.analysis` package documentation.
- Validate and snapshot partition-scoped nodes, values, and logical-memory requirements in
  deterministic supplied order.
- Require the node list to match the planned partition's node IDs exactly and in order.
- Require every node input/output value to be present exactly once in the projected values and
  every projected value to have one matching logical-memory requirement.
- Project compile-time logical-splat constants as an immutable `ValueId`-to-`ScalarValue` map;
  never expose the Compiler-owned constant plan. Every constant key must be one projected graph
  input and its exact scalar type must match that value's descriptor.
- Accept only fully static projected `TensorDescriptor` shapes. Unresolved/dynamic descriptors
  fail before backend analysis because no public resolved-binding contract exists.
- Define exact non-negative byte size and positive power-of-two byte alignment for both resource
  variants.
- Scope `Workspace.requirementId` to one `BackendPartitionAnalysis`; it is non-negative and is not
  a Runtime slot.
- Require unique buffer `ValueId` and workspace IDs within one analysis result.
- Require analysis to return the exact `PlannedPartition` reference supplied in its context.
- Document that `analyze` is deterministic from the complete context, performs no measurement or
  cache mutation, and returns no executable or slot.
- Add focused contract tests with immutable fake backend input/plan records.
- Finalize Javadoc, the backend partition-preparer guide, focused architecture status, glossary,
  master plan, roadmap, and this task in a separate clean documentation context.

## Out of scope

- slot assignment, `WorkspaceSlot`, `PreparedMemoryPlan`, or workspace reuse/aliasing
- `PreparedExecutable`, `PreparedUnit`, `PreparedPartition`, finalization, schedules, or execution
- physical allocation, storage, resource handles, native compilation, or cleanup
- dynamic-dimension binding or a public binding/substitution language
- backend-specific production implementations, routes, candidates, or configuration types
- `PrepareConfig`, Config module changes, tuning search/measurement, or cache loading/mutation
- Compiler changes or exposure of `CompileArtifacts`, `CompileDiagnostics`, `PublicationPlan`,
  `CompileConstantPlan`, `DeferredGraphConstraint`, or another Compiler-owned type
- Runtime, Backend Contract, Planning, Model, Trace, Engine, concrete backend, Gradle dependency,
  architecture-test, conformance-test, or integration-test changes
- Prepare 0002, Runtime 0002, or any later detailed task specification

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core invariants
  - `modules/prepare`
  - Concrete backend modules
  - Prepare lifecycle
  - Dependency rules
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0010](../../../../design/decisions/0010-staged-backend-preparation.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)

## Architecture constraints

- Prepare owns orchestration, the backend analysis boundary, analysis result, and shared resource
  declarations.
- Concrete backends own lowering, route/configuration selection, opaque analysis plans, and later
  executable construction.
- Backend Contract remains a closed DTO/capability leaf and gains no prepare service.
- A concrete backend receives only Prepare-owned projection contracts plus Model/Planning facts it
  is already allowed to consume; it never receives Compiler-owned types.
- Runtime owns stable slot identities and later runtime binding. This task creates neither.
- Runtime hot-path code remains free of `Operation` and `CompiledNode`; those model facts appear
  only in the prepare-time projection.
- Shared Prepare must not inspect or downcast backend-private inputs/plans, interpret route
  vocabulary, or use a generic string/object parameter map.
- Analysis is deterministic and fail-closed. No autotuning workflow is added.
- The dependency direction in `ARCHITECTURE.md` is unchanged. Stop if another module edge or
  architecture rule is required.

## Package impact

Existing package used:

- `io.github.pho001.synaptik.prepare` — delete only the placeholder.

Package added:

- `io.github.pho001.synaptik.prepare.analysis` — public analysis-stage handoff contracts.

Type placement:

- `BackendAnalysisInputs` — opaque typed role for one backend's explicit capabilities/config/cache
  input projection.
- `BackendPreparationPlan` — opaque typed role for the backend-selected lowering/route result.
- `PrepareContext` — Prepare-owned partition-scoped semantic/planning projection.
- `PreparationResourceRequirement` — shared exact buffer/workspace declarations needed for slot
  assignment.
- `BackendPartitionAnalysis` — immutable association of the exact partition, opaque plan, and
  declarations.
- `BackendPartitionPreparer` — concrete-backend-implemented typed analysis collaboration.

Tests mirror `io.github.pho001.synaptik.prepare.analysis`. No other Java package changes.

## Affected files

Expected production/test paths:

- delete `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/PrepareModule.java`;
- add the six declarations and `package-info.java` under
  `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/`;
- add up to four focused test files under the mirrored test package.

Expected documentation/planning paths:

- `docs/backend-guide/partition-preparer.md`
- `docs/architecture/runtime-prepare-backend-boundary.md` — implementation-status update only
- `docs/glossary.md`
- this task
- `docs/planning/modules/prepare/master-plan.md`
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless a concrete contradiction is found: architecture contract and ADR 0010,
Runtime/Compile APIs, lifecycle/module/dependency docs, Config/Compiler/Planning/Backend Contract
plans, current source/build inventories, and architecture tests.

## Maximum scope

At most 20 paths:

- 12 Prepare production/test paths, including placeholder deletion;
- 3 explanatory documentation paths; and
- 5 task/master/roadmap paths.

No Java, test, or Gradle path outside `modules/prepare` may change. If another type, package, module
edge, or path is required, stop and propose a later task.

## Acceptance criteria

- The placeholder is absent and the exact six-declaration surface exists in `prepare.analysis`.
- Public generic bounds preserve the association between one backend's input, opaque plan, and
  analyzer without casts or `Object`.
- Context collections and analysis requirements are immutable snapshots with deterministic order,
  no null elements, and precise failure messages.
- Context validation proves exact planned-node order, closed referenced values, one logical
  requirement per projected value, and fully static descriptors before any backend call.
- Resource declarations accept zero byte size, require positive power-of-two alignment, and reject
  negative/duplicate IDs or values as specified.
- `Buffer.valueId` remains a compile/prepare association only; `Workspace.requirementId` is
  analysis-local. Neither is a Runtime slot, address, allocation, or handle.
- `BackendPartitionAnalysis` retains the exact partition and opaque plan references, snapshots
  requirements, rejects duplicate buffer/workspace keys, and contains no executable.
- Focused tests lock public shape, generic relationships, validation order, immutability,
  deterministic ordering, duplicate rejection, and a fake deterministic analyzer.
- No Compiler import or type name occurs in the public surface or implementation.
- No backend route, config, cache, physical resource, slot, execution, scheduling, or runtime
  implementation is added.
- Javadocs document every parameter, return, failure, ownership, nullability, immutability,
  lifecycle, deterministic behavior, and opaque boundary.
- A separate clean documentation pass finalizes affected explanations, glossary, status, and
  evidence without repeating successful Java tests unless executable behavior changes.
- Exactly this detailed task exists for the Prepare frontier; Prepare 0002 and Runtime 0002 have
  no detailed specifications.

## Tests / validation

Implementation:

```bash
./gradlew :modules:prepare:test
```

Documentation pass:

```bash
./gradlew :modules:prepare:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

Also verify exact public source shape, forbidden Compiler imports/types, only fully static
descriptors accepted, the 20-path ceiling, no Java outside Prepare, no Gradle change, synchronized
status, and absence of later detailed Prepare/Runtime specifications.

Repository-wide tests and architecture tests are deferred to the Prepare contract checkpoint or
continuous integration. This task uses only already-authorized dependency directions and changes
no Gradle edge or dependency rule. Backend conformance and integration tests are not applicable
because no concrete backend or end-to-end execution exists.

## Dependencies

- Compiler 0001–0006 — Complete.
- Planning 0001–0006 — Complete.
- Backend Contract 0001–0004 — Complete and closed.
- Runtime 0001 `BufferSlot` identity — Complete but not consumed by this analysis-only task.
- ADR 0010 staged backend preparation — Accepted.
- Current Prepare module dependencies already expose the Model/Planning facts used here through
  its existing inward dependency graph; no build change is authorized.

## Follow-up tasks

- Runtime 0002 consumes these declarations to define `WorkspaceSlot`, assignment, and prepared
  memory.
- Runtime 0003–0004 establish typed per-run slot access and executable contracts.
- Prepare 0002 later defines finalization against assigned slots.

All remain Draft or Blocked without detailed task specifications.

## Architecture impact

Expected impact: None. This task implements the already-adopted ADR 0010 analysis boundary without
changing module ownership or dependency direction. Stop if implementation requires a Compiler
type, a new module edge, shared interpretation of backend-private state, or slot/finalization
behavior.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, ADR 0010, the focused
runtime/prepare/backend and dependency docs, documentation rules and General/API-Javadoc/
Architecture/Planning profiles, the Prepare and Runtime master plans, and
docs/planning/modules/prepare/tasks/0001-backend-partition-analysis-and-resource-declaration.md.

Implement Prepare 0001 exactly within its surface, package, and 20-path limits. Do not add slot
assignment, finalization, executable/runtime/backend implementation, dynamic binding, tuning,
Compiler types, Gradle/dependency changes, or later task specifications. Stop on an architecture,
dependency, package, or scope conflict.

Run the focused Prepare module validation, then hand the actual diff and test evidence to a
separate clean documentation-focused context. That pass must finalize Javadocs, affected
explanations, glossary, planning evidence/status, Markdown, scope, and whitespace without
repeating Java tests unless executable behavior changes.
```

## Local decisions

- Fully static descriptors are the initial definition of resolved preparation facts. Dynamic
  binding is deferred rather than represented by an untyped map.
- Backend-specific capability/config/cache inputs and the selected plan use separate generic
  marker roles. Their concrete records remain backend-owned and typed; shared Prepare does not
  interpret them.
- Resource byte sizes may be zero. Alignment is a positive power of two in bytes.
- Buffer keys are graph `ValueId` values scoped by the exact partition analysis. Workspace keys
  are non-negative analysis-local IDs. Runtime later assigns both kinds of stable slots.
- One workspace declaration will map to one distinct Runtime slot initially; no lifetime field or
  aliasing is introduced.
- `PrepareContext` preserves supplied deterministic order with immutable list snapshots and a
  linked insertion-order immutable constant-map copy while retaining exact immutable element and
  backend-input references.
- Projection validation is intentionally asymmetric. Every node input/output must resolve to a
  unique projected value and every projected value must have exactly one descriptor-matching
  logical requirement; an otherwise valid projected value is not rejected merely because no
  partition node references it.
- `BackendPartitionAnalysis` snapshots requirements in encounter order and checks buffer
  `ValueId` values and workspace requirement IDs in separate uniqueness domains. The record
  remains constructible independently; the preparer collaboration is responsible for returning
  the exact context partition reference.
- The two backend marker roles remain method-free. Their Javadocs place the immutability
  obligation on concrete implementations without adding shared inspection or another runtime
  abstraction.

## Known limitations

- No unresolved-dimension preparation, physical representation class, device placement, slot
  assignment, resource lifetime/reuse, executable finalization, or concrete backend exists.
- The analysis contract does not load or validate cache persistence. It receives only a concrete
  backend's already-compatible immutable decision input.
- No current Runtime workspace identity or requirement-assignment aggregate consumes the
  declarations. Runtime 0002 is the next Draft planning frontier and has no detailed
  specification yet.
- `BackendPartitionAnalysis` has no `PrepareContext` component by design. Exact
  context-to-analysis partition association is an `analyze` collaboration contract rather than a
  record-constructor cross-object check.

## Validation evidence

- Implementation-focused command, run by the separate implementation context before the
  documentation handoff:

  ```bash
  ./gradlew :modules:prepare:test \
    --tests io.github.pho001.synaptik.prepare.analysis.AnalysisPublicShapeTest \
    --tests io.github.pho001.synaptik.prepare.analysis.BackendPartitionPreparerTest \
    --tests io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirementTest \
    --tests io.github.pho001.synaptik.prepare.analysis.PrepareContextTest
  ```

  Passed 4 suites and 11 tests with no failures, errors, or skips.
- Final implementation validation, run once by the separate implementation context:

  ```bash
  ./gradlew :modules:prepare:test
  ```

  Passed 4 suites and 11 tests with no failures, errors, or skips. Executable Java did not change
  after this evidence. The documentation context changed only Javadoc/package documentation in
  Java source and therefore did not repeat either successful Java test command.
- Documentation-focused context:
  `/root/implement_prepare_0001/prepare_0001_docs`. It applied the General,
  API/Javadoc, Backend Guide, Developer Guide, Architecture, Planning, and Example profiles. It
  independently reviewed `AGENTS.md`, `ARCHITECTURE.md`, documentation rules and profiles, ADR
  0010, the focused architecture/backend/glossary/planning contracts, the complete Prepare
  production and test source, the Prepare build contract, relevant Model/Planning API types, and
  the actual diff.
- Final Prepare Javadoc generation:

  ```bash
  ./gradlew :modules:prepare:javadoc
  ```

  Passed. Gradle reported `BUILD SUCCESSFUL`; 9 tasks were actionable, 2 executed and 7 were
  up-to-date.
- The required no-argument Markdown command ran against the supplied temporary script:

  ```bash
  python3 /tmp/validate_synaptik_markdown.py
  ```

  Passed and reported `validated 0 Markdown files`; inspection confirmed that this script expects
  paths as arguments. The same script was therefore run against all seven authorized
  documentation/planning paths and passed with `validated 7 Markdown files`. This checked local
  targets and anchors, final newlines, trailing whitespace, and balanced backtick/tilde fences.
  The validator reports no separate link or anchor counts.
- Final manual/source gates passed: the exact six top-level declarations and generic
  relationships remain under `prepare.analysis`; the placeholder is absent; projected Shapes are
  required to be fully static; no Compiler import or type occurs in Prepare source; no Java
  outside Prepare and no Gradle path belongs to this task; the isolated Prepare task surface is
  exactly 19 authorized paths, within the 20-path ceiling; and no route, slot, physical resource,
  finalization, executable, schedule, dynamic binding, tuning measurement/cache mutation, or
  Runtime behavior was added.
- Planning/status gates passed: this task and Prepare master row are Complete; Prepare is In
  progress after its first task; Runtime 0002 is Draft rather than Blocked because the producer
  now exists, but has no detailed task specification; Prepare 0002 and Runtime 0002 detailed task
  specifications are absent.
- Final whitespace validation:

  ```bash
  git diff --check
  ```

  Passed with no output.
- Repository-wide tests and architecture tests remain deferred to the Prepare contract checkpoint
  or continuous integration. No dependency, Gradle, architecture-rule, concrete-backend, or
  end-to-end behavior changed, so backend conformance and integration tests remain not
  applicable.

## Implementation notes

- Deleted the root `PrepareModule` placeholder and added the exact public six-declaration
  `io.github.pho001.synaptik.prepare.analysis` API plus package documentation.
- Implemented immutable ordered projections, exact partition-node matching, unique projected
  values, referenced-value resolution, one descriptor-matching logical requirement per projected
  value, fully static Shape validation, and exact-typed logical splats limited to projected graph
  inputs.
- Implemented the sealed `Buffer`/`Workspace` declaration family, separate duplicate domains, the
  opaque typed analysis result, and deterministic backend analysis collaboration.
- Added four focused test suites covering public shape, validation order, immutable snapshots,
  duplicate rejection, exact reference retention, and a deterministic fake preparer.
- The clean documentation pass clarified the asymmetric projection rule in Javadoc, finalized
  package documentation, replaced the planned-only backend guide with a current-contract guide and
  focused example, distinguished implemented analysis from later lifecycle stages in the
  architecture page, added reusable glossary terms, and synchronized Prepare/Runtime/roadmap
  status.
- Reviewed adjacent contracts with no changes: `ARCHITECTURE.md` and ADR 0010 already authorize
  this exact staged boundary; Compile and Runtime APIs have no current consumer surface to update;
  lifecycle, module-boundary, and dependency explanations remain accurate; architecture tests do
  not change because dependencies do not; backend-conformance/integration tests are not
  applicable without a concrete backend or execution; and the existing Prepare Gradle dependency
  contract already supplies the used Model/Planning types.

## Completion summary

- Completed changes: implemented and documented the exact Prepare analysis projection, opaque
  backend input/plan roles, exact resource declarations, immutable analysis result, and typed
  backend preparer collaboration.
- Files changed or created: 12 Prepare production/test paths including placeholder deletion, plus
  the seven authorized explanatory/planning paths; 19 task-isolated paths total.
- Tests and validation: reused the implementation context's two successful 4-suite/11-test
  commands; documentation Javadoc, targeted seven-file Markdown, exact source/import/static-shape
  and 19-path scope, status/spec-absence, final-newline/fence/terminology, and whitespace checks
  passed.
- Documentation-agent review: completed in
  `/root/implement_prepare_0001/prepare_0001_docs` with no executable Java change and no repeated
  Java test suite.
- Documentation impact: current Prepare analysis and later slot/finalization stages are now
  distinguished in the backend guide and focused architecture explanation.
- Javadoc review: all seven Prepare production Java paths were reviewed; `PrepareContext` and
  package documentation were finalized to state the exact asymmetric projection and lifecycle
  limits. Other production Javadocs remained accurate and complete.
- Glossary impact: added current implementation status and reusable definitions for
  `PrepareContext`, opaque backend analysis roles, preparation resource requirements,
  `BackendPartitionAnalysis`, and `BackendPartitionPreparer`.
- Unresolved issues: None.
- Follow-up required: None for this task. Plan Runtime 0002 in a separate planning step before
  implementation; do not create Prepare 0002 until its Runtime 0002–0004 dependencies are stable.

Status: Complete
