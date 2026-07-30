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
  execution/  later prepared executable, unit, execution, and runner contracts
  schedule/   later prepared schedule and step contracts
  run/        later per-run state, resource, residency, and result contracts
```

The module root is not a catch-all facade. Runtime 0001 opens only `memory` with the immutable
`BufferSlot` identity. Later packages remain planned until their invocation, storage, schedule,
configuration, and lifecycle consumers justify exact contracts.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Prepared buffer-slot identity](tasks/0001-prepared-buffer-slot-identity.md) | Complete | Compiler 0006; Planning 0006; Backend Contract 0004; Trace 0002 | Replaced the placeholder with one immutable prepared-plan-local `BufferSlot` identity required to bind later prepared-unit inputs and outputs, without physical storage, allocation, graph values, workspace, execution, or run state. |
| 0002 | Prepared memory and workspace contracts | Draft | 0001; Prepare 0001 | Define the workspace-slot, resource-assignment, and prepared-memory contracts that consume the completed exact backend analysis declarations; initially assign one distinct slot per workspace requirement. |
| 0003 | Run-state and runtime slot-access foundation | Draft | 0002; stable input-binding, storage-access, and resource contracts | Define per-run mutable state and typed slot access without graph inspection, backend discovery, or implementation selection. |
| 0004 | Prepared executable and unit contracts | Draft | 0001–0003 | Define the backend-neutral invocation boundary and unit-to-slot bindings only after its concrete run-state access contract is stable. |
| 0005 | Prepared schedule contract | Draft | 0002–0004; Prepare 0002 finalization | Define ordered executable, transfer, materialization, and publication work without performing prepare-time selection. |
| 0006 | Prepared execution aggregate and lifecycle | Draft | 0002–0005; stable run and publication configuration | Compose reusable prepared state while keeping all invocation mutation in `RunState`. |
| 0007 | Prepared runner and dynamic execution | Draft | 0003–0006; stable run trace and result contracts | Execute prepared schedules, residency, transfers, materialization, and publication without graph or backend rediscovery. |
| 0008 | Runtime contract closure | Draft | 0001–0007 | Audit runtime API cohesion, lifecycle, dependencies, performance boundaries, documentation, and validation. |

## Milestones

- Prepared execution contracts
- Run state and resources
- Schedule runner and publication

## Current status

In progress. [Runtime 0001](tasks/0001-prepared-buffer-slot-identity.md) is Complete with the
public immutable plan-local `BufferSlot` identity, focused tests, Javadoc, package documentation,
and synchronized explanatory documentation. It adds no physical storage, allocation, graph-value
relationship, workspace, slot access, executable, schedule, or run-state contract.

The first task deliberately does not define `PreparedExecutable.execute(...)`. A stable invocation
signature requires the later runtime slot-access and `RunState` contracts; choosing that signature
now would freeze an unproven storage or per-run context. Runtime 0001 establishes the smallest
architecture-named prerequisite that later `PreparedUnit` input/output bindings,
`PreparedMemoryPlan`, `RuntimeSlotTable`, shared prepare, and concrete backend preparation all
need.

The former task-0002 producer blocker is resolved by Complete
[Prepare 0001](../prepare/tasks/0001-backend-partition-analysis-and-resource-declaration.md).
Prepare now supplies immutable `BackendPartitionAnalysis` results with exact buffer/workspace
byte-size and alignment declarations, unique projected buffer IDs, unique analysis-local
workspace IDs, and an opaque selected backend plan. The producer adds no Runtime slot,
allocation, lifetime, or binding behavior.

Runtime 0002 is therefore Draft rather than Blocked and is the next cross-area planning frontier.
Its detailed task specification must select the smallest `WorkspaceSlot`, requirement-to-slot
assignment, and prepared-memory carrier that consume the implemented producer. The first
workspace model remains deliberately conservative: one distinct `WorkspaceSlot` per declaration,
with no aliasing or invented lifetime interval. No Runtime 0002 detailed specification is created
by Prepare 0001.

Tasks 0003–0008 remain Draft without detailed specifications. Runtime 0003–0004 still need the
prepared-memory result before they can define typed per-run slot access and the backend-neutral
executable invocation contract. Prepare finalization follows those contracts, and physical
allocation and per-run binding remain later Runtime/backend work.

## Open questions

- The exact physical buffer, workspace, resource-lifetime, and runtime slot-access contracts remain
  open until their concrete prepare, backend, and runner consumers are stable.
- Runtime 0002 must finalize the exact slot-assignment aggregate consumed by later backend
  finalization from Prepare 0001's implemented declaration contracts.
- The exact `PreparedExecutable.execute(...)` parameter and failure contract remain open until
  `RunState` and runtime slot access are defined.
- Concurrency and reusable prepared-resource ownership remain open for the later prepared
  execution lifecycle task.

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
  backend size, alignment, or route facts.
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

## Risks

- Moving backend discovery or implementation selection into the hot path.
- Letting runtime profiling become hidden online tuning.
- Giving a slot identity physical storage, graph-value, device, or residency semantics before the
  owning prepared-memory and run-state contracts exist.
- Freezing a prepared-executable call shape before typed per-run access and resource ownership are
  known.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
