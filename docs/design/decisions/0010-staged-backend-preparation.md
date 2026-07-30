# ADR 0010: Stage backend preparation around shared slot assignment

## Status

Accepted — 2026-07-30

## Context

Planning assigns each `PlannedPartition` to a backend, but only the concrete backend can lower the
region, select a route, and discover the route's exact buffer and workspace needs. Shared Prepare
must use those needs to construct one prepared memory plan and assign stable Runtime-owned slots.
A backend cannot construct its final executable before those slots exist, while shared Prepare
cannot assign correct slots before the backend has selected a route.

The previous lifecycle named one undifferentiated backend-preparation call. It did not define
whether resource discovery occurred before or after prepared-memory construction, who carried the
declaration, or how final executable construction received assigned slots.

## Decision drivers

- keep lowering, fusion, route selection, and private configuration inside concrete backends;
- let shared Prepare own orchestration and the backend-neutral resource handoff;
- assign stable slots only after exact backend requirements are known;
- keep Compiler aggregates and internals out of concrete backends;
- complete route selection and executable construction before Runtime; and
- avoid inventing workspace aliasing, physical allocation, or autotuning behavior.

## Options considered

### One-pass backend preparation before shared memory planning

The backend returns a final executable and its resource needs together. This leaves the executable
without stable shared slots or requires mutable patching after construction.

### Shared slot assignment before backend route selection

Prepare assigns slots from compile-time logical memory alone. This cannot account for
route-specific representation, alignment, scratch, or workspace requirements.

### Staged backend analysis and finalization

The backend selects and lowers first, shared Prepare assigns slots from the resulting exact
declarations, and the backend then constructs the executable against those assignments.

## Decision

Synaptik adopts staged backend preparation.

Prepare owns the analysis request, `BackendPartitionPreparer` collaboration,
`BackendPartitionAnalysis`, and backend-neutral shared-resource declarations. A concrete backend
analyzes one planned partition from an explicit Prepare projection of stable semantic and Planning
facts, resolved prepare-time bindings, target/backend capabilities, configuration, and compatible
cached tuning decisions. The request never exposes `CompileArtifacts` or Compiler internals.

The backend analysis deterministically selects a supported lowering, route, and configuration. It
returns an opaque backend-owned plan together with exact buffer and workspace requirements needed
by shared preparation. Shared Prepare assigns stable Runtime-owned buffer and workspace slots and
builds the prepared memory plan. The same backend then finalizes the opaque plan against those
assignments to construct the `PreparedExecutable` and `PreparedPartition`.

Finalization cannot change route choice or add undeclared shared requirements. Physical allocation
and per-run slot binding remain outside this handoff. The initial shared assignment gives each
workspace declaration its own slot; reuse waits for a later proved lifetime/interference model.
Any unresolved fact needed for route selection or exact resource declaration fails closed unless
a later explicit prepared contract supports it as run-dynamic without changing the route or slot
plan.

This decision does not add model autotuning. Compatible cached decisions may be explicit inputs,
and an explicitly enabled later tuning workflow may produce such a decision before analysis.
Ordinary analysis remains deterministic and uses safe backend heuristics when no compatible
decision exists.

## Rationale

The staged flow places each decision with the component that has the required knowledge. Concrete
backends keep private lowering and route vocabulary. Shared Prepare receives only the exact
declarations it must interpret to assign slots. Runtime receives only finalized executable and
slot contracts, so no graph, planning, routing, or cache work enters the hot path.

## Consequences

### Positive

- Route-specific workspace needs participate in memory planning before executable construction.
- Backend-private plans remain opaque to shared code.
- Runtime executes final prepared work without `Operation`, `CompiledNode`, or backend discovery.
- Backend Contract remains a closed identity/requirement leaf with no prepare service.

### Negative and risks

- Backends must retain an immutable analysis artifact until finalization.
- Analysis and finalization need exact association and failure validation.
- Conservative one-slot-per-workspace assignment may use more memory until reuse is proved.
- Fully dynamic preparation remains unavailable until a typed resolved-binding or run-dynamic
  resource contract exists.

### Migration, testing, and follow-up

Prepare 0001 introduces only the analysis-side collaboration, explicit projection, opaque plan,
and exact resource declarations. Runtime 0002 follows with the workspace-slot and prepared-memory
consumer contracts. Later Runtime executable/access contracts then allow Prepare finalization.

No module dependency rule changes: concrete backends already may depend inward on Model, Config,
Planning, Runtime, Prepare, Backend Contract, and Trace, while Prepare remains independent of
concrete backends. Architecture tests therefore need no update for this decision. Implementation
tasks must add focused contract tests and run existing dependency validation when their concrete
Gradle/API changes require it.

## Related documentation

- [Architecture contract](../../../ARCHITECTURE.md)
- [Runtime, Prepare, and Backend Boundary](../../architecture/runtime-prepare-backend-boundary.md)
- [Prepare master plan](../../planning/modules/prepare/master-plan.md)
- [Runtime master plan](../../planning/modules/runtime/master-plan.md)
