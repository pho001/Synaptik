# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0010: Stage [backend](../../glossary.md#backend) preparation around shared slot assignment

## Status

Accepted — 2026-07-30

## Context

[Planning](../../glossary.md#term-planning) assigns each `PlannedPartition` to a [backend](../../glossary.md#backend), but only the [concrete backend](../../glossary.md#concrete-backend) can lower the
region, select a route, and discover the route's exact [buffer](../../glossary.md#term-buffer) and workspace needs. Shared [Prepare](../../glossary.md#prepare)
must use those needs to construct one [prepared memory plan](../../glossary.md#prepared-memory-plan--preparedmemoryplan) and assign stable [Runtime](../../glossary.md#term-runtime)-owned slots.
A backend cannot construct its final executable before those slots exist, while shared Prepare
cannot assign correct slots before the backend has selected a route.

The previous [lifecycle](../../glossary.md#lifecycle) named one undifferentiated backend-preparation call. It did not define
whether resource discovery occurred before or after prepared-memory construction, who carried the
declaration, or how final executable construction received assigned slots.

## Decision drivers

- keep [lowering](../../glossary.md#lowering), [fusion](../../glossary.md#term-fusion), route selection, and private configuration inside concrete [backends](../../glossary.md#backend);
- let shared [Prepare](../../glossary.md#prepare) own orchestration and the backend-neutral resource [handoff](../../glossary.md#term-handoff);
- assign stable slots only after exact backend requirements are known;
- keep [Compiler](../../glossary.md#term-compiler) aggregates and internals out of concrete backends;
- complete route selection and executable construction before [Runtime](../../glossary.md#term-runtime); and
- avoid inventing workspace [aliasing](../../glossary.md#term-aliasing), physical allocation, or [autotuning](../../glossary.md#autotuning--model-autotuning) behavior.

## Options considered

### One-pass [backend](../../glossary.md#backend) preparation before shared memory [planning](../../glossary.md#term-planning)

The backend returns a final executable and its resource needs together. This leaves the executable
without stable shared slots or requires [mutable](../../glossary.md#term-mutable) patching after construction.

### Shared slot assignment before [backend route](../../glossary.md#backend-route) selection

[Prepare](../../glossary.md#prepare) assigns slots from [compile](../../glossary.md#compile)-time logical memory alone. This cannot account for
route-specific representation, [alignment](../../glossary.md#term-alignment), scratch, or workspace requirements.

### Staged [backend](../../glossary.md#backend) analysis and finalization

The backend selects and lowers first, shared [Prepare](../../glossary.md#prepare) assigns slots from the resulting exact
declarations, and the backend then constructs the executable against those assignments.

## Decision

Synaptik adopts staged [backend](../../glossary.md#backend) preparation.

[Prepare](../../glossary.md#prepare) owns the analysis request, [`BackendPartitionPreparer`](../../glossary.md#backend-partition-preparer--backendpartitionpreparer) collaboration,
[`BackendPartitionAnalysis`](../../glossary.md#backend-partition-analysis-backendpartitionanalysis), and backend-neutral shared-resource declarations. A [concrete backend](../../glossary.md#concrete-backend)
analyzes one planned [partition](../../glossary.md#partition) from an explicit Prepare projection of stable semantic and [Planning](../../glossary.md#term-planning)
facts, resolved prepare-time [bindings](../../glossary.md#term-binding), [target](../../glossary.md#term-target)/backend capabilities, configuration, and compatible
cached tuning decisions. The request never exposes `CompileArtifacts` or [Compiler](../../glossary.md#term-compiler) internals.

The backend analysis deterministically selects a supported [lowering](../../glossary.md#lowering), route, and configuration. It
returns an [opaque](../../glossary.md#term-opaque) backend-owned plan together with exact [buffer](../../glossary.md#term-buffer) and workspace requirements needed
by shared preparation. Shared Prepare assigns stable [Runtime](../../glossary.md#term-runtime)-owned buffer and workspace slots and
builds the [prepared memory plan](../../glossary.md#prepared-memory-plan--preparedmemoryplan). The same backend then finalizes the opaque plan against those
assignments to construct the [`PreparedExecutable`](../../glossary.md#prepared-executable--preparedexecutable) and [`PreparedPartition`](../../glossary.md#prepared-partition--preparedpartition).

Finalization cannot change route choice or add undeclared shared requirements. Physical allocation
and per-[run](../../glossary.md#term-run) slot binding remain outside this [handoff](../../glossary.md#term-handoff). The initial shared assignment gives each
workspace declaration its own slot; reuse waits for a later proved [lifetime](../../glossary.md#term-lifetime)/interference [model](../../glossary.md#term-model).
Any unresolved fact needed for route selection or exact resource declaration fails closed unless
a later explicit prepared contract supports it as run-dynamic without changing the route or slot
plan.

This decision does not add [model autotuning](../../glossary.md#autotuning--model-autotuning). Compatible cached decisions may be explicit inputs,
and an explicitly enabled later tuning workflow may produce such a decision before analysis.
Ordinary analysis remains deterministic and uses safe backend heuristics when no compatible
decision exists.

## Rationale

The staged flow places each decision with the component that has the required knowledge. Concrete
[backends](../../glossary.md#backend) keep private [lowering](../../glossary.md#lowering) and route vocabulary. Shared [Prepare](../../glossary.md#prepare) receives only the exact
declarations it must interpret to assign slots. [Runtime](../../glossary.md#term-runtime) receives only finalized executable and
slot contracts, so no [graph](../../glossary.md#graph), [planning](../../glossary.md#term-planning), routing, or [cache](../../glossary.md#term-cache) work enters the [hot path](../../glossary.md#term-hot-path).

## Consequences

### Positive

- Route-specific workspace needs participate in memory [planning](../../glossary.md#term-planning) before executable construction.
- [Backend](../../glossary.md#backend)-private plans remain [opaque](../../glossary.md#term-opaque) to shared code.
- [Runtime](../../glossary.md#term-runtime) executes final prepared work without [`Operation`](../../glossary.md#operation), `CompiledNode`, or backend discovery.
- Backend Contract remains a closed [identity](../../glossary.md#term-identity-meanings)/requirement [leaf](../../glossary.md#term-leaf) with no [prepare](../../glossary.md#prepare) service.

### Negative and risks

- [Backends](../../glossary.md#backend) must retain an [immutable](../../glossary.md#term-immutable) analysis [artifact](../../glossary.md#term-artifact) until finalization.
- Analysis and finalization need exact association and failure [validation](../../glossary.md#term-validation).
- Conservative one-slot-per-workspace assignment may use more memory until reuse is proved.
- Fully dynamic preparation remains unavailable until a typed resolved-binding or run-dynamic
  resource contract exists.<span data-glossary-navigation="true"> ([binding glossary](../../glossary.md#term-binding); [run glossary](../../glossary.md#term-run))</span>

### Migration, testing, and follow-up

[Prepare](../../glossary.md#prepare) 0001 introduces only the analysis-side collaboration, explicit projection, [opaque](../../glossary.md#term-opaque) plan,
and exact resource declarations. [Runtime](../../glossary.md#term-runtime) 0002 follows with the workspace-slot and prepared-memory
consumer contracts. Later Runtime executable/access contracts then allow Prepare finalization.

No [module](../../glossary.md#term-module) [dependency](../../glossary.md#term-dependency) rule changes: concrete [backends](../../glossary.md#backend) already may depend inward on [Model](../../glossary.md#term-model), Config,
[Planning](../../glossary.md#term-planning), Runtime, Prepare, Backend Contract, and [Trace](../../glossary.md#trace), while Prepare remains independent of
concrete backends. Architecture tests therefore need no update for this decision. Implementation
tasks must add focused contract tests and [run](../../glossary.md#term-run) existing dependency [validation](../../glossary.md#term-validation) when their concrete
[Gradle](../../glossary.md#term-gradle)/[API](../../glossary.md#term-application-programming-interface-api) changes require it.

## Related documentation

- [Architecture contract](../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../glossary.md#architecture-contract))</span>
- [Runtime, Prepare, and Backend Boundary](../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime); [Prepare glossary](../../glossary.md#prepare); [Backend glossary](../../glossary.md#backend))</span>
- [Prepare master plan](../../planning/modules/prepare/master-plan.md)
- [Runtime master plan](../../planning/modules/runtime/master-plan.md)
