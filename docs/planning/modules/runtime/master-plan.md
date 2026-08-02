# Runtime Master Plan

## Goal

Execute prepared schedules and own dynamic per-run state, residency, transfers, resources, and publication.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Lifecycle](../../../architecture/lifecycle.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend Boundary](../../../architecture/runtime-prepare-backend-boundary.md)

## Scope

- prepared execution contracts
- prepared schedules, slots, and memory plans
- run state and runtime resources
- residency, transfer, materialization, publication, and execution runner
- passive runtime profiling translated through typed trace contracts

## Out of scope

- graph optimization
- autograd construction
- backend discovery
- kernel selection and backend lowering
- model-autotuning search, tuning-cache lookup or mutation, hot-path graph inspection, and
  production-setting selection

## Module invariants

- Runtime executes already-prepared work.
- The hot path does not use `Operation` or `CompiledNode`.
- Runtime does not depend on concrete backend implementations.
- Runtime profiling observes execution but never mutates settings.

## Allowed dependencies

- modules/config
- modules/backend-contract
- modules/trace
- architecture-approved model contracts required by runtime

## Forbidden dependencies

- modules/engine
- concrete backend modules

## Package structure

```text
io.github.pho001.synaptik.runtime/
  memory/     public prepared-memory identity and later runtime slot-access contracts
  resource/   nominal physical representation roles plus immutable prepared creation descriptions
  execution/  public prepared-execution aggregate, prepared executable/transfer recipes,
              explicit executable buffer-access declarations, and direct-reference bound actions
  schedule/   public prepared schedule and closed semantic step contracts
  run/        per-run ownership, binding, cold creation, validity, publication, result leases,
              and complete dynamic runner orchestration
```

The module root is not a catch-all facade. Runtime 0001 opened `memory` with the immutable
`BufferSlot` identity. Complete Runtime 0002 extends only that package with `WorkspaceSlot` and
immutable final per-slot byte-size/alignment geometry in `PreparedMemoryPlan`. Runtime remains
independent of Prepare and Model: Complete Prepare 0002 translates analysis requirements and
retains source-to-slot associations. Runtime 0003 opens `resource` and `run` with the minimal
representation/lifecycle foundation. Complete Runtime 0007 extends those packages with an immutable
prepared representation plan, package-private cold per-run creation, and explicit buffer-copy
validity; its creation prefix stays in the existing `schedule` package. Runtime 0004 opens
`execution` with the reusable recipe and per-run direct-reference invocation boundary; at that
frontier residency, configuration, result, and runner contracts remained planned until their
consumers justified exact surfaces. Runtime 0006 uses
that existing package for the smallest complete prepared-execution root over the current memory
plan and schedule. Complete Runtime 0009 keeps that root unchanged and adds Runtime-owned prepared
publication coordinates, per-run bound publication state, and a whole-`RunState` result lease in
`run`; its ordered occurrence extends the existing `schedule` package.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Prepared buffer-slot identity](tasks/0001-prepared-buffer-slot-identity.md) | Complete | Compiler 0006; Planning 0006; Backend Contract 0004; Trace 0002 | Replaced the placeholder with one immutable prepared-plan-local `BufferSlot` identity required to bind later prepared-unit inputs and outputs, without physical storage, allocation, graph values, workspace, execution, or run state. |
| 0002 | [Prepared memory and workspace contracts](tasks/0002-prepared-memory-and-workspace-contracts.md) | Complete | 0001; Prepare 0001 | Added `WorkspaceSlot` and immutable final per-buffer-slot/per-workspace-slot byte-size/alignment geometry without importing Prepare/Model facts; Complete Prepare 0002 retains exact requirement associations and conservatively assigns distinct slots. |
| 0003 | [Run-state and runtime resource foundation](tasks/0003-run-state-and-runtime-resource-foundation.md) | Complete | 0002; ADR 0011 | Added nominal backend-owned buffer/workspace representation roles, borrowed/run-owned buffer bindings, and one array-backed closeable `RunState` per complete logical run without executable binding, residency, scheduling, transfer, publication, or allocation. |
| 0004 | [Prepared executable and bound invocation](tasks/0004-prepared-executable-and-bound-invocation.md) | Complete | 0001–0003; ADR 0011 | Added immutable dense resource selections, final checked cold binding, and one per-run backend-owned typed bound invocation without a redundant prepared-unit wrapper. |
| 0005 | [Prepared schedule contract](tasks/0005-prepared-schedule-contract.md) | Complete | 0002–0004; Prepare 0002 finalization | Added one immutable exact-plan schedule and its execution-step variant; no `PreparedUnit`, transfer, materialization, or publication payload is invented before its Runtime-owned facts exist. |
| 0006 | [Prepared execution aggregate](tasks/0006-prepared-execution-aggregate.md) | Complete | 0002–0005 | Added the smallest exact-plan/exact-schedule immutable Runtime root while keeping every invocation mutation and resource lifecycle in `RunState`. |
| 0007 | [Representation creation and residency foundation](tasks/0007-representation-creation-and-residency-foundation.md) | Complete | 0003; 0005–0006; Prepare 0002 | Added immutable caller-input/backend-creation descriptions, deterministic per-run creation and rollback, schedule reachability, and explicit per-copy validity without transfer, execution, or Config. |
| 0008 | [Prepared buffer transfer and materialization schedule](tasks/0008-prepared-buffer-transfer-and-materialization-schedule.md) | Complete | 0004–0005; 0007 | Added one backend-supplied prepared/bound buffer-transfer pair and schedule occurrence; materialization is the same action between distinct already-created representations, with destination-valid no-op and success-only validity transition. |
| 0009 | [Publication and result schedule steps](tasks/0009-publication-and-result-schedule-steps.md) | Complete | 0005; 0007–0008; stable publication/result ownership | Added exact prepared/run coordinates and a dense publication suffix, cold-bound direct selected representations, and leased the complete `RunState` to an ordered `RunResult` without importing Compiler publication identities or exposing output values. |
| 0010 | [Prepared runner and dynamic execution](tasks/0010-prepared-runner-and-dynamic-execution.md) | Complete | 0003; 0005–0009; preserve Trace 0001–0002 boundary | Cold-creates one isolated state, binds every direct occurrence before ordered traversal, applies explicit executable read/write validity, and transfers the whole-state result lease without hot graph/backend lookup. |
| 0011 | [Runtime contract closure audit](tasks/0011-runtime-contract-closure-audit.md) | Complete | 0001–0010 | Recorded `BLOCKING_GAP`: the audit is complete, but shared-throwable cleanup, stale general architecture status, and absent Runtime architecture enforcement keep the milestone open. |
| 0012 | [Run-state shared-throwable cleanup](tasks/0012-run-state-shared-throwable-cleanup.md) | Complete | 0003; 0007; 0009–0011 | Repaired `RUNTIME-CLEANUP-001`: closed-first reverse cleanup now skips impossible self-suppression when distinct owned resources throw the same exact primary `Throwable`, preserves that primary, and attempts all remaining owned resources. |
| 0013 | [General architecture status correction](tasks/0013-general-architecture-status-correction.md) | Complete | 0011; 0012 | Corrected `DOCUMENTATION-STATUS-001` in four implicated explanatory documents, preserving authoritative architecture and leaving enforcement to task 0014. |
| 0014 | [Runtime architecture enforcement](tasks/0014-runtime-architecture-enforcement.md) | Complete | 0011; 0012–0013 | Added one dependency-free focused architecture suite that locks Runtime's exact project edges, exhaustively classifies production sources, and rejects `Operation`/`CompiledNode` in the explicit hot path. |

## Milestones

- Prepared execution contracts
- Run state and resources
- Schedule runner and publication
- Runtime contract closure audit

## Current status

Complete after [Runtime 0014](tasks/0014-runtime-architecture-enforcement.md) resolved
`ARCHITECTURE-ENFORCEMENT-001`. The current public Runtime
surface now includes immutable `runtime.memory` geometry, nominal `runtime.resource`
buffer/workspace cleanup roles, the `runtime.run` ownership and one-run lifecycle foundation, the
`runtime.execution` prepared-recipe/cold-bound-invocation boundary, and the `runtime.schedule`
creation-plus-execution ordered recipe.

Runtime 0003's focused command passed three suites and 20 tests; the single final module command
passed six suites and 45 tests with no failures, errors, or skips. The separate clean
documentation pass finalized all seven production/package Javadocs, Runtime API, focused
architecture implementation status, glossary, and planning records without changing executable
Java or repeating the successful suites. Runtime Javadoc, targeted Markdown, exact public
surface/import/mechanism/build/scope/status/later-specification, generated-page, and whitespace
gates passed.

The former task-0002 producer blocker is resolved by Complete
[Prepare 0001](../prepare/tasks/0001-backend-partition-analysis-and-resource-declaration.md).
Prepare now supplies immutable `BackendPartitionAnalysis` results with exact buffer/workspace
byte-size and alignment declarations, unique projected buffer IDs, unique analysis-local
workspace IDs, and an opaque selected backend plan. The producer adds no Runtime slot,
allocation, lifetime, or binding behavior.

Runtime 0002 deliberately does not import or retain `PreparationResourceRequirement`,
`BackendPartitionAnalysis`, `ValueId`, `LogicalMemoryRequirement`, or `PlannedPartition`. Runtime
keeps its existing dependency boundary and contains only final slot geometry. Complete Prepare
0002 now traverses ordered analyses and requirements, retains exact source-to-slot associations,
and constructs the Runtime plan. It assigns one distinct buffer slot per distinct declared buffer
value, combining repeated declarations with maximum geometry, and one distinct workspace slot per
workspace declaration; no reuse or lifetime interval is invented.

ADR 0011 resolves the Runtime 0003 blocker. Prepared recipes are immutable/reusable; each active
complete logical run has one isolated `RunState`; Runtime orchestrates logical state and cleanup;
and concrete backends implement physical representations and mechanics. Heterogeneous
compatibility is checked once during current cold binding, which creates backend-owned typed direct-
reference invocation objects before the hot path.

Complete Runtime 0003 provides the smallest foundation: two nominal closeable physical-
representation roles, one borrowed/run-owned buffer-binding value, and array-indexed `RunState`
access ordered exactly like `PreparedMemoryPlan`. It carries more than one explicitly supplied
buffer representation without defining validity/coherence semantics. Workspace positions bind one
run-owned backend-local representation. Construction retains exact references and transfers
cleanup only after all validation succeeds. Closed-first deterministic reverse cleanup skips
borrowed buffers, preserves unchecked failures, and is idempotent.

At the Runtime 0003 frontier, allocation, full residency, transfers, publication/results,
scheduling, and runner behavior remained later tasks. Complete
[Runtime 0004](tasks/0004-prepared-executable-and-bound-invocation.md) defines only the immutable
executable recipe's dense buffer/representation and workspace selections, final common cold-
binding validation, concrete-backend compatibility hooks, and one per-run `BoundInvocation` with
a minimal execute-time closed-state guard. Its focused command passed two suites and 18 tests;
the single final Runtime command passed eight suites and 63 tests with no failures, errors, or
skips. The clean documentation pass finalized all five permitted production/package Javadocs,
Runtime/Public APIs, focused boundary status, backend guide, glossary, and planning records
without changing executable Java or repeating the successful Java tests. Runtime Javadoc, two
Java 26 example compilations and one execution, Markdown, exact surface/messages/order/hooks,
direct-hot-path bytecode, imports/mechanisms/build, exact 15-path scope, synchronized status and
later-specification absence, and whitespace gates passed.

Runtime 0004 deliberately omits `PreparedUnit`. Complete Prepare 0002 adds only the exact
`PreparedPartition(partition, executable)` association, so no current consumer establishes a
distinct unit invariant beyond the partition association and executable selections. Complete
[Runtime 0005](tasks/0005-prepared-schedule-contract.md) resolves that question from
the actual schedule consumer: Prepare's partition association cannot cross into Runtime, while
one schedule occurrence needs only the exact `PreparedExecutable`. Runtime 0005 therefore defines
one exact-plan immutable schedule, a sealed plan-associated step contract, and only its current
executable variant. Its focused implementation command passed one suite and 11 tests with no
failures, errors, or skips. The clean documentation pass finalized the two production/package
Javadocs, five explanatory documents, and synchronized planning records without changing
executable Java or repeating the successful Java test. Runtime Javadoc, the Java 26 schedule
example, Markdown, exact surface and sealed family, imports/mechanisms, exact 11-path scope,
status, and whitespace gates passed.

Complete [Runtime 0006](tasks/0006-prepared-execution-aggregate.md) adds the exact immutable
two-component `PreparedExecution` root. It retains one exact `PreparedMemoryPlan` and one exact
`PreparedSchedule`, rejects a schedule associated with any other plan reference, and adds no
resource ownership, close/run lifecycle, configuration, or per-run state. Its focused command
passed one suite and 8 tests; the final Runtime command passed 10 suites and 82 tests, with no
skips, failures, or errors. The separate clean documentation pass finalized both affected
production/package Javadocs, Runtime/Public APIs, focused architecture status, glossary, and
planning records without changing executable Java or repeating those tests. Runtime Javadoc,
eight-file Markdown, exact public surface/mechanism/import/build/scope/status, and whitespace
gates passed.

Run configuration is not an aggregate dependency and remains later runner/publication input.
Complete Runtime 0007 supplies the stable Runtime-owned representation creation and per-copy
validity/residency facts required before transfer/materialization can be planned. Its focused
command passed three suites and 37 tests, and the single final Runtime module command passed 11
suites and 94 tests, with no skips, failures, or errors. The separate clean documentation pass
finalized seven production/package Javadocs, Runtime/Public APIs, focused architecture status,
backend guide, glossary, and planning records without changing executable Java or repeating the
successful tests. Runtime Javadoc, generated-page inspection, the focused Runtime API example,
eight-file Markdown validation, exact surface/mechanism/boundary/build/scope/status checks, and
whitespace validation passed.

Runtime 0008 selects one explicit buffer transfer between distinct already-created representation
positions of one logical buffer. Materialization is that same operation when it produces an
equivalent destination representation; no second kind is planned. Cold binding resolves exact
physical references once, while the bound action makes a valid destination a no-op, otherwise
requires a valid source, invokes backend work once, and marks only the destination valid after
success. The task adds no allocation, route search, runner, executable-output invalidation,
publication, Prepare orchestration, concrete backend, or coherence policy.

Its focused implementation command passed three suites and 31 tests, and the single final Runtime
module command passed 13 suites and 113 tests, with no failures, errors, or skips. The separate
clean documentation pass finalized the five affected production/package Javadocs, Runtime/Public
APIs, focused architecture status, backend guide, glossary, and planning records without changing
executable Java or repeating the successful tests. Runtime Javadoc/generated-page inspection,
the current transfer examples, eight-file Markdown validation, exact surface/hot-path/direct-field/
mechanism/build/scope/status checks, and whitespace validation passed.

Complete [Runtime 0009](tasks/0009-publication-and-result-schedule-steps.md) adds one immutable
`PreparedPublication`, one per-run `BoundPublication`, a
dense publication-only schedule suffix, and one `RunResult` that leases the complete `RunState`.
The contract requires the named already-created representation to be valid at publication time,
permits distinct ordered result positions to alias it, performs no fallback or physical work, and
keeps result values private. `PreparedExecution` remains exactly memory plan plus schedule.
Its focused four-suite command passed 32 tests, and the single final Runtime module command passed
16 suites and 130 tests, with no failures, errors, or skips. Clean documentation context
`019fbe69-07e8-7a20-b132-c3b70c663d4d` finalized affected Javadocs, Runtime/Public APIs, focused
architecture status, backend guidance, glossary, and planning evidence without changing
executable behavior or repeating the successful Java tests. Runtime Javadoc, generated-page
inspection, eight-file Markdown validation, exact 18-path scope, status, and whitespace checks
passed.
Complete [Runtime 0010](tasks/0010-prepared-runner-and-dynamic-execution.md) preserves
`PreparedExecution` exactly as memory plan
plus schedule, adds explicit executable read/write declarations, and places one narrow runner in
`runtime.run`. One call creates an isolated state, cold-binds all non-creation occurrences before
traversal, performs conservative output-validity transitions, and either leases the complete open
state to `RunResult` or closes it after failure. Trace 0001–0002 are preserved but not consumed
because the Trace run-payload family remains Draft; the task invents no payload or emission.
Its focused command passed 26 tests, and the final Runtime command passed 17 suites and 143 tests
without failures, errors, or skips. Clean documentation context `/root/runtime0010_docs`
finalized the affected Javadocs and documentation and passed Javadoc, generated-page,
eight-file Markdown, exact 14-path, status, and whitespace gates.

Complete [Runtime 0011](tasks/0011-runtime-contract-closure-audit.md) produced the durable
[`BLOCKING_GAP` audit](runtime-contract-closure-audit.md). Runtime 0001-0010 otherwise form a
cohesive immutable-prepared/per-run-state boundary, but the selected Runtime milestone remains in
progress. `RunState.close()` can abort attempt-all cleanup when distinct resources throw the same
exception object; three general architecture pages and the architecture-test guide retain stale
implementation-status prose; and the architecture-test project lacks Runtime dependency/hot-path
enforcement. These findings need separate bounded planning because 0011 is documentation-only and
creates no repair task.

The one combined checkpoint passed 205 suites and 1,530 tests with zero failures, errors, or
skips, including Runtime's 17 suites/143 tests and architecture's 3 suites/3 tests. Runtime
Javadoc and final documentation/scope/status checks passed. That audit selected bounded
[Runtime 0012](tasks/0012-run-state-shared-throwable-cleanup.md) as the sole executable repair for
`RUNTIME-CLEANUP-001`; Runtime 0013 and 0014 retained the other two findings as explicit Draft
follow-ups without detailed specifications. Complete
[Prepare 0003](../prepare/tasks/0003-prepare-orchestration-and-validation.md) is the downstream
consumer-driven integration task. It narrowly extends the current representation-origin family
with one backend-created initially-valid buffer for compile-time logical splats while preserving
all other Runtime contracts and the closed Runtime milestone. Backend Contract remains closed,
and module dependency directions are unchanged.

Complete [Runtime 0012](tasks/0012-run-state-shared-throwable-cleanup.md) adds the constant-space
primary-identity guard to both `RunState.close()` cleanup loops and one exact same-`Throwable`
regression. Cleanup now preserves the original primary object, omits only its impossible
self-suppressed recurrence, continues reverse traversal, and suppresses later distinct failures
in encounter order. The implementation context's focused `RunStateTest` run passed 16 tests, and
its one final Runtime run passed 17 suites and 144 tests with zero failures, errors, or skips.
Clean documentation context `019fbefd-f12e-7450-b554-81a816c3e6b8` finalized the Javadoc review,
Runtime API, glossary, and planning evidence without executable Java changes or repeated Java
tests; Runtime Javadoc,
five-file Markdown, exact seven-owned-plus-two-preserved-path scope, status/history/later-spec,
and whitespace gates passed. Complete
[Runtime 0013](tasks/0013-general-architecture-status-correction.md) corrects the five stale
implementation-status statements in three architecture pages and the architecture-test guide
without changing architecture, APIs, Java, tests, or build behavior. Clean documentation context
`019fc161-1298-72e1-a2bb-82ac8cbfb672` passed seven-file Markdown, preserved-history, exact
replacement, fourteen-path scope, status, later-file-absence, and whitespace gates. Detailed
[Runtime 0014](tasks/0014-runtime-architecture-enforcement.md) is Complete. Its dependency-free
architecture suite locks the exact Runtime Gradle edges, exhaustively classifies the 25 current
production sources, and rejects exact `Operation`/`CompiledNode` identities in the five-file
direct hot subset. The focused suite and final combined checkpoint passed; therefore
`ARCHITECTURE-ENFORCEMENT-001` is resolved and the selected Runtime milestone is Complete.

## Open questions

- Runtime 0012 resolved repeated primary throwable identity, Runtime 0013 corrected the stale
  general architecture status, and Runtime 0014 resolved the final audited enforcement finding.
  Runtime 0011 remains an unchanged historical `BLOCKING_GAP` audit rather than a current blocker.
- Output value access remains a later Engine/result decision rather than an unresolved Runtime
  semantic gap. Prepare orchestration, concrete backend execution, Trace run payloads, Config
  policy, and tuning likewise retain their existing downstream owners.
- Prepare 0003 owns the only selected post-closure Runtime surface extension: a generic
  run-owned initialized-buffer origin needed to make non-bindable compiler constant sources
  runnable. It carries no graph or scalar fact and does not reopen Runtime orchestration.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- `BufferSlot` is a prepared-plan-local identity, not a graph `ValueId`, storage object, address,
  allocation, device, or residency fact.
- Compile-time `ValueId` and `LogicalMemoryRequirement`, prepared `BufferSlot` identity, physical
  backend storage, and per-run binding remain four distinct concepts. No numeric conversion or
  implicit ownership relationship connects them.
- Concrete backend preparation decides backend-specific scratch/workspace needs. Runtime owns
  only the later shared prepared identity, plan, and access contracts; it does not invent backend
  size, alignment, route, storage, or handle vocabulary.
- Runtime 0002 must consume Prepare 0001's implemented declaration shape rather than inventing
  backend size, alignment, or route facts. Runtime retains only the resulting exact slot geometry;
  Prepare retains declaration associations.
- `PreparedMemoryPlan` must not import Prepare or Model types. It stores ordered unique
  `BufferSlot`/`WorkspaceSlot` entries with exact byte size and alignment, while current Prepare
  translation owns declaration coverage and deterministic slot assignment.
- `BackendPartitionPreparer`, `PrepareContext`, and `PreparedPartition` are Prepare-owned.
  Backend Contract remains closed, and Runtime owns no backend collaboration interface.
- Prepare may consume Compiler internally, but a backend-facing input must not expose
  `CompileArtifacts` or create a concrete-backend-to-Compiler dependency.
- The authoritative contract selects staged backend analysis, shared slot assignment, and backend
  finalization. Prepare owns analysis/declaration; Runtime owns stable slots and binding.
- Runtime 0001 does not need a runtime-facing config contract. Later tasks must wait for the
  specific config surface they consume rather than inventing defaults or aggregates early.
- A prepared executable invocation contract is deferred until its per-run access context is
  concrete; runtime must not compensate with a zero-argument executable, untyped object map, or
  compile-time graph parameter.
- ADR 0011 assigns physical representation implementations/mechanics to concrete backends and
  per-run logical state/lifecycle orchestration to Runtime. Backend Contract remains closed.
- `PreparedMemoryPlan` and later prepared recipes are immutable and reusable. Every active
  complete logical run has exactly one distinct `RunState`; run-owned mutable resources are
  isolated, while caller inputs may be borrowed under an explicit lifetime obligation.
- Prepare 0003 adds `InitializedBuffer(BufferCreator)` beside unchanged `CallerInput` and
  `CreatedBuffer`. Its concrete backend creator materializes the logical value; Runtime records
  only run ownership and initial validity and remains unaware of `ScalarValue` and `ValueId`.
- Runtime 0003 uses nominal closeable representation roles and array-indexed state only. Runtime
  0004 owns checked cold binding to backend-owned typed direct-reference invocation objects; full
  validity/residency and transfer associations remain later.
- Runtime 0004 uses a Runtime-owned abstract template for common plan identity, dense selection,
  invocation-association, and closed-state validation. Concrete backend subclasses use explicit
  checked compatibility and retain direct typed references. Binding owns no auxiliary closeable
  resource. Runtime 0005 later established that current scheduling needs no distinct
  `PreparedUnit`.
- Runtime 0005 does not introduce `PreparedUnit`: list position is the execution occurrence, and
  `PreparedExecutable` supplies the only stable Runtime-owned work and exact-plan invariant.
- Runtime 0005 uses one immutable exact-plan schedule with a sealed plan-associated step family
  and only an executable step. Empty schedules and repeated executable occurrences are valid.
- Runtime 0006 uses only `PreparedMemoryPlan` and `PreparedSchedule` as the complete current
  aggregate components, validates their association by exact plan reference identity, and adds no
  close/run/configuration/persistent-resource contract.
- Runtime 0007 keeps that aggregate exactly unchanged. One immutable prepared representation plan
  becomes reachable through a first-only schedule step; a package-private cold operation binds
  caller inputs and invokes backend-owned creators with reverse partial-failure cleanup.
- Runtime 0007 treats a bound representation as resident until `RunState` closure and adds one
  explicit boolean validity fact per buffer representation. Borrowed inputs begin valid, created
  run-owned buffers begin invalid, multiple or zero valid copies are permitted, and workspaces
  never carry logical validity.
- Config 0007 is not a dependency of representation creation or validity. Run/publication options
  remain inputs to their later consumers.
- Runtime 0008 uses Runtime 0007's stable representation positions and validity facts for one
  explicit prepared/bound transfer. Materialization is the same transfer to an equivalent
  already-created destination; no allocation, second kind, route search, or hidden coherence is
  added.
- Runtime 0009 uses a dense `resultIndex` plus exact buffer/representation positions as the full
  Runtime publication identity. Compiler/Planning/Prepare identities do not cross into Runtime.
- Publication is a dense suffix and requires the named already-created representation valid at
  that moment. Another representation requires Runtime 0008's explicit transfer beforehand; no
  fallback, discovery, materialization, allocation, conversion, or copy occurs in publication.
- `RunResult` leases the complete `RunState` instead of transferring individual representations.
  This preserves immutable ownership bindings, borrowed-input rules, duplicate-identity
  protection, and deterministic cleanup while keeping `PreparedExecution` unchanged.
- Buffer/workspace identity domains distinguish shared buffer positions from scratch positions,
  but they do not distinguish caller input, internal value, or published-output roles. Those
  roles must come from later Prepare/publication associations rather than a speculative Runtime
  role enum in task 0003.
- Runtime 0010 places the complete-run orchestrator in `runtime.run`, keeps the two-component
  prepared root unchanged, and requires aligned `READ_ONLY`, `WRITE_ONLY`, or `READ_WRITE`
  declarations for executable buffer selections.
- Reads validate before every copy of each output buffer is invalidated. Backend success validates
  exact declared writes; backend failure leaves every output copy invalid. All occurrences bind
  before traversal, which uses direct bound references and primitive coordinates only.
- Runtime 0010 emits no Trace event because the stable Trace foundation has no current run-payload
  DTO.
- Runtime 0012 preserves the first cleanup failure by identity and attaches only later distinct
  failure objects. A repeated occurrence of the primary object is skipped solely to avoid Java
  self-suppression and does not stop attempt-all reverse cleanup.

## Risks

- Moving backend discovery or implementation selection into the hot path.
- Letting runtime profiling become hidden online tuning.
- Giving a slot identity physical storage, graph-value, device, or residency semantics before the
  owning prepared-memory and run-state contracts exist.
- Letting a later concrete backend bypass the checked binding hooks, retain nominal arrays in the
  hot path, or acquire auxiliary binding resources without an explicit cleanup lifecycle.
- Treating a prepared schedule as a hot-path interpreter instead of cold-binding its typed steps
  to direct per-run work.
- Turning Runtime 0008's exact prepared source/destination action into route search, allocation,
  hidden coherence, or hot representation lookup.
- Letting a later implementation replace Runtime 0009's exact coordinates and whole-state lease
  with graph identities, individual-resource detachment, hidden fallback, or public output access.
- Inferring executable read/write roles from selection order, validity, graph facts, or physical
  types instead of explicit immutable declarations.
- Interleaving binding with execution, preserving stale output validity after a possible write, or
  adding map/registry/backend/graph work to bound traversal.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
