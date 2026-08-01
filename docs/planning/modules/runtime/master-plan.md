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
  resource/   nominal physical representation roles implemented by concrete backends
  execution/  public prepared-executable recipe and per-run bound-invocation contracts, plus
              later execution and runner contracts
  schedule/   public prepared schedule and closed semantic step contracts
  run/        per-run ownership/binding state and later residency/result contracts
```

The module root is not a catch-all facade. Runtime 0001 opened `memory` with the immutable
`BufferSlot` identity. Complete Runtime 0002 extends only that package with `WorkspaceSlot` and
immutable final per-slot byte-size/alignment geometry in `PreparedMemoryPlan`. Runtime remains
independent of Prepare and Model: Complete Prepare 0002 translates analysis requirements and
retains source-to-slot associations. Runtime 0003 opens `resource` and `run` with the minimal
representation/lifecycle foundation. Runtime 0004 opens `execution` with the reusable recipe and
per-run direct-reference invocation boundary; residency, schedule, configuration, result, and
runner contracts remain planned until their consumers justify exact surfaces.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Prepared buffer-slot identity](tasks/0001-prepared-buffer-slot-identity.md) | Complete | Compiler 0006; Planning 0006; Backend Contract 0004; Trace 0002 | Replaced the placeholder with one immutable prepared-plan-local `BufferSlot` identity required to bind later prepared-unit inputs and outputs, without physical storage, allocation, graph values, workspace, execution, or run state. |
| 0002 | [Prepared memory and workspace contracts](tasks/0002-prepared-memory-and-workspace-contracts.md) | Complete | 0001; Prepare 0001 | Added `WorkspaceSlot` and immutable final per-buffer-slot/per-workspace-slot byte-size/alignment geometry without importing Prepare/Model facts; Complete Prepare 0002 retains exact requirement associations and conservatively assigns distinct slots. |
| 0003 | [Run-state and runtime resource foundation](tasks/0003-run-state-and-runtime-resource-foundation.md) | Complete | 0002; ADR 0011 | Added nominal backend-owned buffer/workspace representation roles, borrowed/run-owned buffer bindings, and one array-backed closeable `RunState` per complete logical run without executable binding, residency, scheduling, transfer, publication, or allocation. |
| 0004 | [Prepared executable and bound invocation](tasks/0004-prepared-executable-and-bound-invocation.md) | Complete | 0001–0003; ADR 0011 | Added immutable dense resource selections, final checked cold binding, and one per-run backend-owned typed bound invocation without a redundant prepared-unit wrapper. |
| 0005 | [Prepared schedule contract](tasks/0005-prepared-schedule-contract.md) | Complete | 0002–0004; Prepare 0002 finalization | Added one immutable exact-plan schedule and its execution-step variant; no `PreparedUnit`, transfer, materialization, or publication payload is invented before its Runtime-owned facts exist. |
| 0006 | Prepared execution aggregate and lifecycle | Draft | 0002–0005; stable run configuration | Compose reusable prepared state while keeping all invocation mutation in `RunState`. |
| 0007 | Representation creation and residency foundation | Draft | 0003; 0006; stable run resource/configuration contracts | Define per-run representation creation plus validity/residency facts before transfer or materialization work can be scheduled. |
| 0008 | Transfer and materialization schedule steps | Draft | 0005; 0007 | Add stable Runtime-owned transfer/materialization recipes only after their representation and residency inputs exist. |
| 0009 | Publication and result schedule steps | Draft | 0005; 0007–0008; stable publication/result ownership | Translate prepared resources into Runtime-owned delivery/result contracts without importing Compiler publication identities. |
| 0010 | Prepared runner and dynamic execution | Draft | 0003; 0005–0009; stable run trace contracts | Cold-bind and execute prepared schedules without graph inspection, backend rediscovery, allocation, or lookup in the hot path. |
| 0011 | Runtime contract closure | Draft | 0001–0010 | Audit Runtime API cohesion, lifecycle, dependencies, performance boundaries, documentation, and validation. |

## Milestones

- Prepared execution contracts
- Run state and resources
- Schedule runner and publication

## Current status

In progress after completion of
[Runtime 0005](tasks/0005-prepared-schedule-contract.md). The current public Runtime
surface now includes immutable `runtime.memory` geometry, nominal `runtime.resource`
buffer/workspace cleanup roles, the `runtime.run` ownership and one-run lifecycle foundation, the
`runtime.execution` prepared-recipe/cold-bound-invocation boundary, and the `runtime.schedule`
executable-only ordered recipe.

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

Allocation, full residency, transfers, publication/results, scheduling, and runner behavior remain
later tasks. Complete
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

Transfer, materialization, and publication lack stable Runtime-owned
representation/residency or delivery/result facts and are split into Draft tasks 0007–0009 before
the Draft runner. Runtime 0006–0011 and Prepare 0003 remain Draft without detailed specifications.
Backend Contract remains closed, and module dependency directions are unchanged.

## Open questions

- Runtime 0007 must establish the smallest representation-creation and residency/validity facts
  required before transfer or materialization steps can be specified.
- Runtime 0009 must establish a Runtime-owned prepared-resource-to-result association and result
  ownership policy without importing Compiler publication identities.

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
- Runtime 0003 uses nominal closeable representation roles and array-indexed state only. Runtime
  0004 owns checked cold binding to backend-owned typed direct-reference invocation objects; full
  validity/residency and transfer associations remain later.
- Runtime 0004 uses a Runtime-owned abstract template for common plan identity, dense selection,
  invocation-association, and closed-state validation. Concrete backend subclasses use explicit
  checked compatibility and retain direct typed references. Binding owns no auxiliary closeable
  resource, and `PreparedUnit` remains deferred until a real schedule/finalization consumer
  establishes its distinct role.
- Runtime 0005 does not introduce `PreparedUnit`: list position is the execution occurrence, and
  `PreparedExecutable` supplies the only stable Runtime-owned work and exact-plan invariant.
- Runtime 0005 uses one immutable exact-plan schedule with a sealed plan-associated step family
  and only an executable step. Empty schedules and repeated executable occurrences are valid.
- Transfer, materialization, and publication steps remain Draft until Runtime owns stable
  representation/residency and delivery/result facts; Compiler/Planning identities do not cross
  into Runtime to fill that gap.
- Buffer/workspace identity domains distinguish shared buffer positions from scratch positions,
  but they do not distinguish caller input, internal value, or published-output roles. Those
  roles must come from later Prepare/publication associations rather than a speculative Runtime
  role enum in task 0003.

## Risks

- Moving backend discovery or implementation selection into the hot path.
- Letting runtime profiling become hidden online tuning.
- Giving a slot identity physical storage, graph-value, device, or residency semantics before the
  owning prepared-memory and run-state contracts exist.
- Letting a later concrete backend bypass the checked binding hooks, retain nominal arrays in the
  hot path, or acquire auxiliary binding resources without an explicit cleanup lifecycle.
- Treating a prepared schedule as a hot-path interpreter instead of cold-binding its typed steps
  to direct per-run work.
- Inventing transfer, materialization, or publication payloads before their Runtime-owned inputs
  and consumers are stable.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
