# Task 0001: [Backend](../../../../glossary.md#backend) [Partition](../../../../glossary.md#partition) Analysis and Resource Declaration

## Status

Complete

## Goal

Replace the [Prepare](../../../../glossary.md#prepare) placeholder with the smallest typed analysis-side collaboration for the first
stage of [ADR 0010](../../../../design/decisions/0010-staged-backend-preparation.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>. A [concrete
backend](../../../../glossary.md#concrete-backend) must be able to analyze one fully resolved planned [partition](../../../../glossary.md#partition), retain its selected
[lowering](../../../../glossary.md#lowering)/route opaquely, and declare exact shared [buffer](../../../../glossary.md#term-buffer)/workspace requirements before [Runtime](../../../../glossary.md#term-runtime)
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

`BackendAnalysisInputs` is an [opaque](../../../../glossary.md#term-opaque) typed role implemented by one concrete backend's [immutable](../../../../glossary.md#term-immutable)
input record. That record carries the resolved [target](../../../../glossary.md#term-target)/[backend](../../../../glossary.md#backend) capabilities, applicable
configuration, and compatible cached decision needed by that backend. `BackendPreparationPlan`
is the corresponding opaque immutable result retaining selected lowering, route, and private
configuration for later finalization. Shared Prepare interprets neither role.

## [Scope](../../../../glossary.md#term-scope)

- Delete `PrepareModule`.
- Add the exact six public declarations above and `prepare.analysis` package documentation.
- Validate and snapshot partition-scoped nodes, values, and logical-memory requirements in
  deterministic supplied order.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Require the [node](../../../../glossary.md#node) list to match the planned [partition](../../../../glossary.md#partition)'s node IDs exactly and in order.
- Require every node input/output value to be present exactly once in the projected values and
  every projected value to have one matching logical-memory requirement.
- Project [compile](../../../../glossary.md#compile)-time logical-splat [constants](../../../../glossary.md#term-constant) as an [immutable](../../../../glossary.md#term-immutable) [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot)-to-[`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) map;
  never expose the Compiler-owned constant plan. Every constant key must be one projected graph
  input and its exact scalar type must match that value's descriptor.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [scalar glossary](../../../../glossary.md#term-scalar); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Accept only fully static projected `TensorDescriptor` shapes. Unresolved/dynamic descriptors
  fail before backend analysis because no public resolved-binding contract exists.<span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape); [binding glossary](../../../../glossary.md#term-binding))</span>
- Define exact non-negative byte size and positive power-of-two byte alignment for both resource
  variants.<span data-glossary-navigation="true"> ([byte alignment glossary](../../../../glossary.md#term-byte-alignment))</span>
- Scope `Workspace.requirementId` to one [`BackendPartitionAnalysis`](../../../../glossary.md#backend-partition-analysis-backendpartitionanalysis); it is non-negative and is not
  a Runtime slot.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Require unique [buffer](../../../../glossary.md#term-buffer) `ValueId` and workspace IDs within one analysis result.
- Require analysis to return the exact `PlannedPartition` reference supplied in its context.
- Document that `analyze` is deterministic from the complete context, performs no measurement or
  cache mutation, and returns no executable or slot.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- Add focused contract tests with immutable fake [backend](../../../../glossary.md#backend) input/plan records.
- Finalize Javadoc, the backend partition-preparer guide, focused architecture status, glossary,
  master plan, roadmap, and this task in a separate clean documentation context.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- slot assignment, [`WorkspaceSlot`](../../../../glossary.md#workspace-slot--workspaceslot), [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan), or workspace reuse/[aliasing](../../../../glossary.md#term-aliasing)
- [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable), `PreparedUnit`, [`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition), finalization, schedules, or execution
- physical allocation, storage, resource handles, native [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), or cleanup
- dynamic-[dimension](../../../../glossary.md#dimension) [binding](../../../../glossary.md#term-binding) or a public binding/substitution language
- [backend](../../../../glossary.md#backend)-specific production implementations, routes, candidates, or configuration types
- `PrepareConfig`, Config [module](../../../../glossary.md#term-module) changes, tuning search/measurement, or [cache](../../../../glossary.md#term-cache) loading/mutation
- [Compiler](../../../../glossary.md#term-compiler) changes or exposure of `CompileArtifacts`, `CompileDiagnostics`, `PublicationPlan`,
  `CompileConstantPlan`, `DeferredGraphConstraint`, or another Compiler-owned type
- Runtime, Backend Contract, Planning, Model, Trace, Engine, concrete backend, Gradle dependency,
  architecture-test, conformance-test, or integration-test changes<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Model glossary](../../../../glossary.md#term-model); [Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- [Prepare](../../../../glossary.md#prepare) 0002, [Runtime](../../../../glossary.md#term-runtime) 0002, or any later detailed task specification

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core invariants
  - `modules/prepare`
  - [Concrete backend](../../../../glossary.md#concrete-backend) [modules](../../../../glossary.md#term-module)
  - [Prepare](../../../../glossary.md#prepare) [lifecycle](../../../../glossary.md#lifecycle)
  - [Dependency](../../../../glossary.md#term-dependency) rules
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Backend glossary](../../../../glossary.md#backend))</span>
- [ADR 0010](../../../../design/decisions/0010-staged-backend-preparation.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)

## Architecture constraints

- Prepare owns orchestration, the backend analysis boundary, analysis result, and shared resource
  declarations.
- Concrete backends own lowering, route/configuration selection, opaque analysis plans, and later
  executable construction.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [opaque glossary](../../../../glossary.md#term-opaque))</span>
- [Backend](../../../../glossary.md#backend) Contract remains a closed [DTO](../../../../glossary.md#data-transfer-object--dto)/capability [leaf](../../../../glossary.md#term-leaf) and gains no [prepare](../../../../glossary.md#prepare) service.
- A concrete backend receives only Prepare-owned projection contracts plus Model/Planning facts it
  is already allowed to consume; it never receives Compiler-owned types.<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- [Runtime](../../../../glossary.md#term-runtime) owns stable slot [identities](../../../../glossary.md#term-identity-meanings) and later runtime [binding](../../../../glossary.md#term-binding). This task creates neither.
- Runtime hot-path code remains free of [`Operation`](../../../../glossary.md#operation) and `CompiledNode`; those model facts appear
  only in the prepare-time projection.
- Shared Prepare must not inspect or downcast backend-private inputs/plans, interpret route
  vocabulary, or use a generic string/object parameter map.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>
- Analysis is deterministic and [fail-closed](../../../../glossary.md#term-fail-closed). No [autotuning](../../../../glossary.md#autotuning--model-autotuning) workflow is added.
- The [dependency direction](../../../../glossary.md#java-platform-and-build-tools) in `ARCHITECTURE.md` is unchanged. Stop if another module edge or
  architecture rule is required.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing package used:

- `io.github.pho001.synaptik.prepare` — delete only the placeholder.

Package added:

- `io.github.pho001.synaptik.prepare.analysis` — public analysis-stage [handoff](../../../../glossary.md#term-handoff) contracts.

Type placement:

- `BackendAnalysisInputs` — opaque typed role for one backend's explicit capabilities/config/cache
  input projection.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- `BackendPreparationPlan` — [opaque](../../../../glossary.md#term-opaque) typed role for the [backend](../../../../glossary.md#backend)-selected [lowering](../../../../glossary.md#lowering)/route result.
- [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) — [Prepare](../../../../glossary.md#prepare)-owned [partition](../../../../glossary.md#partition)-scoped semantic/[planning](../../../../glossary.md#term-planning) projection.
- `PreparationResourceRequirement` — shared exact buffer/workspace declarations needed for slot
  assignment.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer))</span>
- [`BackendPartitionAnalysis`](../../../../glossary.md#backend-partition-analysis-backendpartitionanalysis) — immutable association of the exact partition, opaque plan, and
  declarations.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- [`BackendPartitionPreparer`](../../../../glossary.md#backend-partition-preparer--backendpartitionpreparer) — concrete-backend-implemented typed analysis collaboration.

Tests mirror `io.github.pho001.synaptik.prepare.analysis`. No other Java package changes.

## Affected files

Expected production/test paths:

- delete `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/PrepareModule.java`;
- add the six declarations and `package-info.java` under
  `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/`;
- add up to four focused test files under the mirrored test package.

Expected documentation/[planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/partition-preparer.md`
- `docs/architecture/runtime-prepare-backend-boundary.md` — implementation-status update only
- `docs/glossary.md`
- this task
- `docs/planning/modules/prepare/master-plan.md`
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless a concrete contradiction is found: [architecture contract](../../../../glossary.md#architecture-contract) and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0010,
[Runtime](../../../../glossary.md#term-runtime)/[Compile](../../../../glossary.md#compile) APIs, [lifecycle](../../../../glossary.md#lifecycle)/[module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency) docs, Config/[Compiler](../../../../glossary.md#term-compiler)/Planning/[Backend](../../../../glossary.md#backend) Contract
plans, current source/build inventories, and architecture tests.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 20 paths:

- 12 [Prepare](../../../../glossary.md#prepare) production/test paths, including placeholder deletion;
- 3 explanatory documentation paths; and
- 5 task/master/roadmap paths.

No Java, test, or [Gradle](../../../../glossary.md#term-gradle) path outside `modules/prepare` may change. If another type, package, [module](../../../../glossary.md#term-module)
edge, or path is required, stop and propose a later task.

## Acceptance criteria

- The placeholder is absent and the exact six-declaration surface exists in `prepare.analysis`.
- Public generic bounds preserve the association between one backend's input, opaque plan, and
  analyzer without casts or <span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [opaque glossary](../../../../glossary.md#term-opaque))</span>`Object`.
- Context collections and analysis requirements are immutable snapshots with deterministic order,
  no null elements, and precise failure messages.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [elements glossary](../../../../glossary.md#term-element))</span>
- Context validation proves exact planned-node order, closed referenced values, one logical
  requirement per projected value, and fully static descriptors before any backend call.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [node glossary](../../../../glossary.md#node); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- Resource declarations accept zero byte size, require positive power-of-two alignment, and reject
  negative/duplicate IDs or values as specified.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span>
- `Buffer.valueId` remains a [compile](../../../../glossary.md#compile)/[prepare](../../../../glossary.md#prepare) association only; `Workspace.requirementId` is
  analysis-local. Neither is a Runtime slot, address, allocation, or handle.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [`BackendPartitionAnalysis`](../../../../glossary.md#backend-partition-analysis-backendpartitionanalysis) retains the exact partition and opaque plan references, snapshots
  requirements, rejects duplicate buffer/workspace keys, and contains no executable.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Focused tests lock public shape, generic relationships, validation order, immutability,
  deterministic ordering, duplicate rejection, and a fake deterministic analyzer.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- No [Compiler](../../../../glossary.md#term-compiler) import or type name occurs in the public surface or implementation.
- No backend route, config, cache, physical resource, slot, execution, scheduling, or runtime
  implementation is added.<span data-glossary-navigation="true"> ([backend route glossary](../../../../glossary.md#backend-route); [cache glossary](../../../../glossary.md#term-cache))</span>
- Javadocs document every parameter, return, failure, ownership, nullability, immutability,
  lifecycle, deterministic behavior, and opaque boundary.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [nullability glossary](../../../../glossary.md#term-nullability); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- A separate clean documentation pass finalizes affected explanations, glossary, status, and
  evidence without repeating successful Java tests unless executable behavior changes.
- Exactly this detailed task exists for the Prepare frontier; Prepare 0002 and Runtime 0002 have
  no detailed specifications.

## Tests / [validation](../../../../glossary.md#term-validation)

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

Also verify exact public source [shape](../../../../glossary.md#shape), forbidden [Compiler](../../../../glossary.md#term-compiler) imports/types, only fully static
[descriptors](../../../../glossary.md#term-descriptor) accepted, the 20-path ceiling, no Java outside [Prepare](../../../../glossary.md#prepare), no [Gradle](../../../../glossary.md#term-gradle) change, synchronized
status, and absence of later detailed Prepare/[Runtime](../../../../glossary.md#term-runtime) specifications.

Repository-wide tests and architecture tests are deferred to the Prepare contract [checkpoint](../../../../glossary.md#term-checkpoint) or
[continuous integration](../../../../glossary.md#term-continuous-integration-ci). This task uses only already-authorized [dependency](../../../../glossary.md#term-dependency) directions and changes
no Gradle edge or dependency rule. [Backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) and integration tests are not applicable
because no [concrete backend](../../../../glossary.md#concrete-backend) or end-to-end execution exists.

## Dependencies

- [Compiler](../../../../glossary.md#term-compiler) 0001–0006 — Complete.
- [Planning](../../../../glossary.md#term-planning) 0001–0006 — Complete.
- [Backend](../../../../glossary.md#backend) Contract 0001–0004 — Complete and closed.
- [Runtime](../../../../glossary.md#term-runtime) 0001 [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) [identity](../../../../glossary.md#term-identity-meanings) — Complete but not consumed by this analysis-only task.
- [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0010 staged backend preparation — Accepted.
- Current Prepare module dependencies already expose the Model/Planning facts used here through
  its existing inward dependency graph; no build change is authorized.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [module glossary](../../../../glossary.md#term-module); [Model glossary](../../../../glossary.md#term-model); [dependency glossary](../../../../glossary.md#term-dependency); [graph glossary](../../../../glossary.md#graph))</span>

## Follow-up tasks

- [Runtime](../../../../glossary.md#term-runtime) 0002 consumes these declarations to define [`WorkspaceSlot`](../../../../glossary.md#workspace-slot--workspaceslot), assignment, and prepared
  memory.
- Runtime 0003–0004 establish typed per-[run](../../../../glossary.md#term-run) slot access and executable contracts.
- [Prepare](../../../../glossary.md#prepare) 0002 later defines finalization against assigned slots.

All remain Draft or Blocked without detailed task specifications.

## Architecture impact

Expected impact: None. This task implements the already-adopted [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0010 analysis boundary without
changing [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings) or [dependency direction](../../../../glossary.md#java-platform-and-build-tools). Stop if implementation requires a [Compiler](../../../../glossary.md#term-compiler)
type, a new module edge, shared interpretation of [backend](../../../../glossary.md#backend)-private state, or slot/finalization
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
  binding is deferred rather than represented by an untyped map.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [binding glossary](../../../../glossary.md#term-binding))</span>
- Backend-specific capability/config/cache inputs and the selected plan use separate generic
  marker roles. Their concrete records remain backend-owned and typed; shared Prepare does not
  interpret them.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [cache glossary](../../../../glossary.md#term-cache); [Prepare glossary](../../../../glossary.md#prepare))</span>
- Resource byte sizes may be zero. [Alignment](../../../../glossary.md#term-alignment) is a positive power of two in bytes.
- [Buffer](../../../../glossary.md#term-buffer) keys are [graph](../../../../glossary.md#graph) [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) values scoped by the exact partition analysis. Workspace keys
  are non-negative analysis-local IDs. Runtime later assigns both kinds of stable slots.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- One workspace declaration will map to one distinct Runtime slot initially; no lifetime field or
  aliasing is introduced.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) preserves supplied deterministic order with immutable list snapshots and a
  linked insertion-order immutable constant-map copy while retaining exact immutable element and
  backend-input references.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [constant glossary](../../../../glossary.md#term-constant); [element glossary](../../../../glossary.md#term-element))</span>
- Projection validation is intentionally asymmetric. Every node input/output must resolve to a
  unique projected value and every projected value must have exactly one descriptor-matching
  logical requirement; an otherwise valid projected value is not rejected merely because no
  partition node references it.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [node glossary](../../../../glossary.md#node))</span>
- [`BackendPartitionAnalysis`](../../../../glossary.md#backend-partition-analysis-backendpartitionanalysis) snapshots requirements in encounter order and checks buffer
  `ValueId` values and workspace requirement IDs in separate uniqueness domains. The record
  remains constructible independently; the preparer collaboration is responsible for returning
  the exact context partition reference.
- The two backend marker roles remain method-free. Their Javadocs place the immutability
  obligation on concrete implementations without adding shared inspection or another runtime
  abstraction.

## Known limitations

- No unresolved-dimension preparation, physical representation class, device placement, slot
  assignment, resource lifetime/reuse, executable finalization, or concrete backend exists.<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension); [device glossary](../../../../glossary.md#term-device); [resource lifetime glossary](../../../../glossary.md#ownership-borrowing-and-lease); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- The analysis contract does not load or validate cache persistence. It receives only a concrete
  backend's already-compatible immutable decision input.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [persistence glossary](../../../../glossary.md#term-persistence); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- No current Runtime workspace identity or requirement-assignment aggregate consumes the
  declarations. Runtime 0002 is the next Draft planning frontier and has no detailed
  specification yet.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [identity glossary](../../../../glossary.md#term-identity-meanings); [planning glossary](../../../../glossary.md#term-planning))</span>
- [`BackendPartitionAnalysis`](../../../../glossary.md#backend-partition-analysis-backendpartitionanalysis) has no [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) component by design. Exact
  context-to-analysis partition association is an <span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition))</span>`analyze` collaboration contract rather than a
  record-constructor cross-object check.

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation-focused command, run by the separate implementation context before the
  documentation handoff:<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>

  ```bash
  ./gradlew :modules:prepare:test \
    --tests io.github.pho001.synaptik.prepare.analysis.AnalysisPublicShapeTest \
    --tests io.github.pho001.synaptik.prepare.analysis.BackendPartitionPreparerTest \
    --tests io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirementTest \
    --tests io.github.pho001.synaptik.prepare.analysis.PrepareContextTest
  ```

  Passed 4 suites and 11 tests with no failures, errors, or skips.
- Final implementation validation, [run](../../../../glossary.md#term-run) once by the separate implementation context:

  ```bash
  ./gradlew :modules:prepare:test
  ```

  Passed 4 suites and 11 tests with no failures, errors, or skips. Executable Java did not change
  after this evidence. The documentation context changed only Javadoc/package documentation in
  Java source and therefore did not repeat either successful Java test command.
- Documentation-focused context:
  `/root/implement_prepare_0001/prepare_0001_docs`. It applied the General,
  API/Javadoc, Backend Guide, Developer Guide, Architecture, Planning, and Example profiles. It
  independently reviewed <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Backend glossary](../../../../glossary.md#backend); [Planning glossary](../../../../glossary.md#term-planning))</span>`AGENTS.md`, `ARCHITECTURE.md`, documentation rules and profiles, ADR
  0010, the focused architecture/backend/glossary/planning contracts, the complete Prepare
  production and test source, the Prepare build contract, relevant Model/Planning API types, and
  the actual diff.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Model glossary](../../../../glossary.md#term-model))</span>
- Final [Prepare](../../../../glossary.md#prepare) [Javadoc](../../../../glossary.md#term-javadoc) generation:

  ```bash
  ./gradlew :modules:prepare:javadoc
  ```

  Passed. [Gradle](../../../../glossary.md#term-gradle) reported `BUILD SUCCESSFUL`; 9 tasks were actionable, 2 executed and 7 were
  up-to-date.
- The required no-[argument](../../../../glossary.md#term-argument) Markdown command ran against the supplied temporary script:

  ```bash
  python3 /tmp/validate_synaptik_markdown.py
  ```

  Passed and reported `validated 0 Markdown files`; inspection confirmed that this script expects
  paths as arguments. The same script was therefore run against all seven authorized
  documentation/planning paths and passed with `validated 7 Markdown files`. This checked local
  targets and anchors, final newlines, trailing whitespace, and balanced backtick/tilde fences.
  The validator reports no separate link or anchor counts.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- Final manual/source gates passed: the exact six top-level declarations and generic
  relationships remain under `prepare.analysis`; the placeholder is absent; projected Shapes are
  required to be fully static; no Compiler import or type occurs in Prepare source; no Java
  outside Prepare and no Gradle path belongs to this task; the isolated Prepare task surface is
  exactly 19 authorized paths, within the 20-path ceiling; and no route, slot, physical resource,
  finalization, executable, schedule, dynamic binding, tuning measurement/cache mutation, or
  Runtime behavior was added.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [Compiler glossary](../../../../glossary.md#term-compiler); [binding glossary](../../../../glossary.md#term-binding); [cache glossary](../../../../glossary.md#term-cache); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Planning/status gates passed: this task and Prepare master row are Complete; Prepare is In
  progress after its first task; Runtime 0002 is Draft rather than Blocked because the producer
  now exists, but has no detailed task specification; Prepare 0002 and Runtime 0002 detailed task
  specifications are absent.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- Final whitespace validation:

  ```bash
  git diff --check
  ```

  Passed with no output.
- Repository-wide tests and architecture tests remain deferred to the Prepare contract checkpoint
  or continuous integration. No dependency, Gradle, architecture-rule, concrete-backend, or
  end-to-end behavior changed, so backend conformance and integration tests remain not
  applicable.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [continuous integration glossary](../../../../glossary.md#term-continuous-integration-ci); [dependency glossary](../../../../glossary.md#term-dependency); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Implementation notes

- Deleted the root `PrepareModule` placeholder and added the exact public six-declaration
  `io.github.pho001.synaptik.prepare.analysis` [API](../../../../glossary.md#term-application-programming-interface-api) plus package documentation.
- Implemented immutable ordered projections, exact partition-node matching, unique projected
  values, referenced-value resolution, one descriptor-matching logical requirement per projected
  value, fully static Shape validation, and exact-typed logical splats limited to projected graph
  inputs.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [partition glossary](../../../../glossary.md#partition); [node glossary](../../../../glossary.md#node); [descriptor glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation); [graph glossary](../../../../glossary.md#graph))</span>
- Implemented the [sealed](../../../../glossary.md#term-sealed) [`Buffer`](../../../../glossary.md#term-buffer)/`Workspace` declaration family, separate duplicate domains, the
  opaque typed analysis result, and deterministic backend analysis collaboration.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [backend glossary](../../../../glossary.md#backend))</span>
- Added four focused test suites covering public shape, validation order, immutable snapshots,
  duplicate rejection, exact reference retention, and a deterministic fake preparer.
- The clean documentation pass clarified the asymmetric projection rule in Javadoc, finalized
  package documentation, replaced the planned-only backend guide with a current-contract guide and
  focused example, distinguished implemented analysis from later lifecycle stages in the
  architecture page, added reusable glossary terms, and synchronized Prepare/Runtime/roadmap
  status.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [lifecycle glossary](../../../../glossary.md#lifecycle); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Reviewed adjacent contracts with no changes: `ARCHITECTURE.md` and ADR 0010 already authorize
  this exact staged boundary; Compile and Runtime APIs have no current consumer surface to update;
  lifecycle, module-boundary, and dependency explanations remain accurate; architecture tests do
  not change because dependencies do not; backend-conformance/integration tests are not
  applicable without a concrete backend or execution; and the existing Prepare Gradle dependency
  contract already supplies the used Model/Planning types.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Compile glossary](../../../../glossary.md#compile); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [conformance glossary](../../../../glossary.md#term-conformance); [concrete backend glossary](../../../../glossary.md#concrete-backend); [Gradle glossary](../../../../glossary.md#term-gradle); [Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: implemented and documented the exact Prepare analysis projection, opaque
  backend input/plan roles, exact resource declarations, immutable analysis result, and typed
  backend preparer collaboration.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [backend glossary](../../../../glossary.md#backend); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Files changed or created: 12 Prepare production/test paths including placeholder deletion, plus
  the seven authorized explanatory/planning paths; 19 task-isolated paths total.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: reused the implementation context's two successful 4-suite/11-test
  commands; documentation Javadoc, targeted seven-file Markdown, exact source/import/static-shape
  and 19-path scope, status/spec-absence, final-newline/fence/terminology, and whitespace checks
  passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [shape glossary](../../../../glossary.md#shape); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: completed in
  `/root/implement_prepare_0001/prepare_0001_docs` with no executable Java change and no repeated
  Java test suite.
- Documentation impact: current Prepare analysis and later slot/finalization stages are now
  distinguished in the backend guide and focused architecture explanation.
- [Javadoc](../../../../glossary.md#term-javadoc) review: all seven [Prepare](../../../../glossary.md#prepare) production Java paths were reviewed; [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) and
  package documentation were finalized to state the exact asymmetric projection and lifecycle
  limits. Other production Javadocs remained accurate and complete.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Glossary impact: added current implementation status and reusable definitions for
  `PrepareContext`, [opaque backend analysis roles](../../../../glossary.md#opaque-backend-analysis-roles), preparation resource requirements,
  [`BackendPartitionAnalysis`](../../../../glossary.md#backend-partition-analysis-backendpartitionanalysis), and [`BackendPartitionPreparer`](../../../../glossary.md#backend-partition-preparer--backendpartitionpreparer).
- Unresolved issues: None.
- Follow-up required: None for this task. Plan Runtime 0002 in a separate planning step before
  implementation; do not create Prepare 0002 until its Runtime 0002–0004 dependencies are stable.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>

Status: Complete
