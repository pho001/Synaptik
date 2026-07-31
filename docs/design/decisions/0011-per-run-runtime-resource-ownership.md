# ADR 0011: Per-run Runtime resource ownership and cold binding

## Status

Accepted — 2026-07-31

## Context

Runtime already owns immutable buffer/workspace slot identities and final byte geometry, but no
contract says what a slot binds to during a run. A heterogeneous prepared schedule can require
more than one backend or device representation of one logical buffer, while backend executable
implementations need their own concrete representation types. The design must also permit one
prepared execution to serve concurrent runs without sharing mutable invocation state or leaking
native-resource cleanup into shared Runtime code.

A method-free resource marker alone would move unchecked casts or reflective inspection into
execution. One undifferentiated physical object per slot would not represent explicit cross-
backend transfers. Conversely, a generic Runtime resource manager or coherence layer would make
Runtime understand backend storage and introduce policy not required by the current lifecycle.

## Decision drivers

- immutable reusable prepared recipes and isolated concurrent runs;
- explicit ownership and failure cleanup for caller, run, result, and prepared resources;
- backend-owned physical representations and mechanics without concrete-backend dependencies;
- heterogeneous Java types resolved once before the execution hot path;
- direct-reference hot-path access without maps, reflection, string dispatch, or repeated casts;
- explicit prepared transfer work rather than discovery or fallback during execution; and
- no speculative pooling, aliasing, coherence, or multi-device scheduler.

## Options considered

### One physical object per slot

This is compact for a single backend, but it cannot represent one logical buffer on both sides of
an explicitly prepared transfer without replacing the binding or adding hidden coherence.

### Runtime-owned generic multi-representation manager

Runtime could store arbitrary backend objects and select among them dynamically. This centralizes
backend type knowledge, encourages raw `Object` or unchecked generic access, and risks turning
residency into an implicit coherence protocol.

### Backend-owned representations with per-run Runtime orchestration

Runtime keeps logical per-run slot state and lifecycle roles. Concrete backends implement nominal
buffer/workspace representations and physical operations. A cold binding phase performs explicit
checked compatibility validation and creates typed backend-owned invocation objects holding direct
references.

## Decision

Synaptik adopts backend-owned physical representations with per-run Runtime orchestration.

`PreparedExecution` and its prepared memory, schedule, executable recipes, and immutable
persistent prepared resources are immutable and reusable. Every active logical invocation has
exactly one mutable `RunState` covering its complete heterogeneous schedule. Concurrent runs have
distinct `RunState` instances, mutable slot state, and run-owned resources.

Runtime owns logical buffer/workspace slot state, resource-lifecycle orchestration, the validity
and residency state needed to follow prepared work, cleanup after construction or execution
failure, and run isolation. Concrete backends own physical buffer/workspace representation
classes and the actual allocation, release, transfer, and access mechanics. Runtime does not know
`MemorySegment`, Metal, CUDA, or other native representation classes and does not choose a
backend.

A buffer slot may carry multiple representations only when an explicit prepared schedule requires
them; a pure single-backend run normally carries one. Representation creation and transfer are
prepared work. A workspace slot is per-run backend-local scratch and normally has one physical
representation for its declared use. Host staging and device scratch are separate workspace
requirements and slots.

Caller inputs are borrowed for the run. Internal buffers and workspaces are run-owned. Publication
transfers or leases selected output ownership to a later `RunResult`. Immutable persistent
resources needed by prepared executables are `PreparedExecution`-owned and are not ordinary
workspace. Runtime requests cleanup, while each concrete representation performs its physical
release. Cleanup releases only resources still owned by the run.

Before execution, cold binding resolves the representations required by each prepared executable.
Any dynamic compatibility check is explicit, checked, and confined to that boundary. The backend
then creates a typed bound invocation or binding object with direct concrete references. The hot
path contains no map lookup, reflection, string dispatch, graph inspection, backend discovery,
kernel selection, global registry/service lookup, or repeated unsafe cast. Public Runtime
contracts expose no raw `Object`, unchecked generic API, or switch over concrete backend types.

## Rationale

This split gives Runtime the cross-backend lifecycle facts it must coordinate without making it a
physical storage abstraction. Backend code retains the concrete Java types needed for safe direct
access. A single whole-run `RunState` can represent transfers between prepared partitions while
still isolating concurrent invocations. Cold binding pays heterogeneous compatibility costs once
and keeps the execution path predictable.

## Consequences

### Positive

- One immutable prepared execution can serve any number of isolated concurrent runs.
- Physical cleanup stays with the code that understands the physical representation.
- Explicit prepared representations and transfers support heterogeneous schedules without hidden
  backend discovery or write-back.
- Backend-owned bound invocations can use direct typed references in the hot path.

### Negative and risks

- Java cannot express the complete heterogeneous type relationship in one shared static generic
  signature, so a checked dynamic boundary remains necessary.
- Runtime must maintain careful ownership transitions so failure cleanup does not close borrowed
  inputs or outputs already transferred to a result.
- Multiple explicit representations require later validity/residency state and prepared transfer
  associations; a nominal carrier alone is not a coherence protocol.
- Concrete backend implementations must test idempotent lifecycle behavior and partial-failure
  cleanup around their physical resources.

### Migration, testing, and follow-up

Runtime 0003 introduces only the minimal per-run representation/lifecycle carrier and array-
indexed `RunState` foundation. Runtime 0004 defines typed cold-bound executable invocation. Later
Runtime/Prepare tasks define full representation validity/residency, transfer, publication,
`RunResult`, and runner behavior. No automatic resource pooling/reuse/aliasing, distributed
sharding, hidden mutation/coherence, or multi-device scheduling is authorized.

The decision changes no module dependency direction. Runtime still depends only on existing
inward modules; Prepare remains independent of concrete backends; Backend Contract remains a
closed declarative leaf; and concrete backends implement inward Runtime/Prepare contracts.
Architecture tests therefore need no update for this ADR.

## Related documentation

- [Architecture contract](../../../ARCHITECTURE.md)
- [Lifecycle](../../architecture/lifecycle.md)
- [Runtime, Prepare, and Backend Boundary](../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0006: No runtime service locator](0006-no-runtime-service-locator.md)
- [ADR 0010: Staged backend preparation](0010-staged-backend-preparation.md)
- [Runtime master plan](../../planning/modules/runtime/master-plan.md)
