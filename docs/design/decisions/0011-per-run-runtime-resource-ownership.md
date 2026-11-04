# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0011: Per-[run](../../glossary.md#term-run) [Runtime](../../glossary.md#term-runtime) resource [ownership](../../glossary.md#term-ownership) and cold [binding](../../glossary.md#term-binding)


## Editorial reading context

Runtime is the caller of backend representation and binding contracts. It borrows application input resources for one invocation and coordinates cleanup of resources acquired by that run. Backend implementations perform physical allocation and release. Publishing an output can transfer or lease its lifetime to a result; run cleanup must not release inputs it only borrowed.

Use the [run/result resource transitions](../../architecture/runtime-prepare-backend-boundary.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted — 2026-07-31

## Context

[Runtime](../../glossary.md#term-runtime) already owns [immutable](../../glossary.md#term-immutable) [buffer](../../glossary.md#term-buffer)/[workspace slot](../../glossary.md#workspace-slot--workspaceslot) [identities](../../glossary.md#term-identity-meanings) and final byte geometry, but no
contract says what a slot binds to during a [run](../../glossary.md#term-run). A [heterogeneous](../../glossary.md#term-heterogeneous) [prepared schedule](../../glossary.md#prepared-schedule--preparedschedule) can require
more than one [backend](../../glossary.md#backend) or [device](../../glossary.md#term-device) representation of one logical buffer, while backend executable
implementations need their own concrete representation types. The design must also permit one
[prepared execution](../../glossary.md#prepared-execution--preparedexecution) to serve concurrent runs without sharing [mutable](../../glossary.md#term-mutable) invocation state or leaking
native-resource cleanup into shared Runtime code.

A method-free resource marker alone would move unchecked casts or reflective inspection into
execution. One undifferentiated physical object per slot would not represent explicit cross-
backend transfers. Conversely, a generic Runtime resource manager or coherence layer would make
Runtime understand backend storage and introduce policy not required by the current [lifecycle](../../glossary.md#lifecycle).

## Decision drivers

- [immutable](../../glossary.md#term-immutable) reusable prepared recipes and isolated concurrent runs;
- explicit [ownership](../../glossary.md#term-ownership-meanings) and failure cleanup for caller, [run](../../glossary.md#term-run), result, and prepared resources;
- [backend](../../glossary.md#backend)-owned physical representations and mechanics without concrete-backend dependencies;
- [heterogeneous](../../glossary.md#term-heterogeneous) Java types resolved once before the execution [hot path](../../glossary.md#term-hot-path);
- direct-reference hot-path access without maps, reflection, string [dispatch](../../glossary.md#term-dispatch), or repeated casts;
- explicit prepared transfer work rather than discovery or fallback during execution; and
- no speculative [pooling](../../glossary.md#term-pooling), [aliasing](../../glossary.md#term-aliasing), coherence, or multi-[device](../../glossary.md#term-device) scheduler.

## Options considered

### One physical object per slot

This is compact for a single [backend](../../glossary.md#backend), but it cannot represent one logical [buffer](../../glossary.md#term-buffer) on both sides of
an explicitly prepared transfer without replacing the [binding](../../glossary.md#term-binding) or adding hidden coherence.

### [Runtime](../../glossary.md#term-runtime)-owned generic multi-representation manager

Runtime could store arbitrary [backend](../../glossary.md#backend) objects and select among them dynamically. This centralizes
backend type knowledge, encourages raw `Object` or unchecked generic access, and risks turning
[residency](../../glossary.md#residency) into an implicit coherence protocol.

### [Backend](../../glossary.md#backend)-owned representations with per-[run](../../glossary.md#term-run) [Runtime](../../glossary.md#term-runtime) orchestration

Runtime keeps logical per-run slot state and [lifecycle](../../glossary.md#lifecycle) roles. Concrete backends implement [nominal](../../glossary.md#term-nominal)
[buffer](../../glossary.md#term-buffer)/workspace representations and physical operations. A cold [binding](../../glossary.md#term-binding) phase performs explicit
checked compatibility [validation](../../glossary.md#term-validation) and creates typed backend-owned invocation objects holding direct
references.

## Decision

Synaptik adopts [backend](../../glossary.md#backend)-owned physical representations with per-[run](../../glossary.md#term-run) [Runtime](../../glossary.md#term-runtime) orchestration.

[`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution) and its prepared memory, schedule, executable recipes, and [immutable](../../glossary.md#term-immutable)
persistent prepared resources are immutable and reusable. Every active logical invocation has
exactly one [mutable](../../glossary.md#term-mutable) [`RunState`](../../glossary.md#run-state--runstate) covering its complete [heterogeneous](../../glossary.md#term-heterogeneous) schedule. Concurrent runs have
distinct `RunState` instances, mutable slot state, and run-owned resources.

Runtime owns logical [buffer](../../glossary.md#term-buffer)/[workspace slot](../../glossary.md#workspace-slot--workspaceslot) state, resource-[lifecycle](../../glossary.md#lifecycle) orchestration, the validity
and [residency](../../glossary.md#residency) state needed to follow prepared work, cleanup after construction or execution
failure, and run isolation. Concrete backends own physical buffer/workspace representation
classes and the actual allocation, [release](../../glossary.md#term-release), transfer, and access mechanics. Runtime does not know
[`MemorySegment`](../../glossary.md#term-memorysegment), [Metal](../../glossary.md#term-metal), [CUDA](../../glossary.md#term-cuda), or other native representation classes and does not choose a
backend.

A [buffer slot](../../glossary.md#buffer-slot--bufferslot) may [carry](../../glossary.md#term-carry) multiple representations only when an explicit [prepared schedule](../../glossary.md#prepared-schedule--preparedschedule) requires
them; a pure single-backend run normally carries one. Representation creation and transfer are
prepared work. A workspace slot is per-run backend-local scratch and normally has one physical
representation for its declared use. [Host](../../glossary.md#term-host) staging and [device](../../glossary.md#term-device) scratch are separate workspace
requirements and slots.

Caller inputs are [borrowed](../../glossary.md#ownership-borrowing-and-lease) for the run. Internal buffers and workspaces are run-owned. [Publication](../../glossary.md#term-publication)
transfers or [leases](../../glossary.md#term-lease) selected output [ownership](../../glossary.md#term-ownership-meanings) to a later [`RunResult`](../../glossary.md#run-result--runresult). Immutable persistent
resources needed by prepared executables are `PreparedExecution`-owned and are not ordinary
workspace. Runtime requests cleanup, while each concrete representation performs its physical
release. Cleanup [releases](../../glossary.md#commit-release-distribution-log-and-seed-meanings) only resources still owned by the run.

Before execution, cold [binding](../../glossary.md#term-binding) resolves the representations required by each [prepared executable](../../glossary.md#prepared-executable--preparedexecutable).
Any dynamic compatibility check is explicit, checked, and confined to that boundary. The backend
then creates a typed [bound invocation](../../glossary.md#bound-invocation--boundinvocation) or binding object with direct concrete references. The [hot
path](../../glossary.md#term-hot-path) contains no map [lookup](../../glossary.md#term-lookup), reflection, string [dispatch](../../glossary.md#term-dispatch), [graph](../../glossary.md#graph) inspection, backend discovery,
[kernel](../../glossary.md#kernel) selection, global [registry](../../glossary.md#term-registry)/service lookup, or repeated unsafe cast. Public Runtime
contracts expose no raw `Object`, unchecked generic [API](../../glossary.md#term-application-programming-interface-api), or switch over [concrete backend](../../glossary.md#concrete-backend) types.

## Rationale

This split gives [Runtime](../../glossary.md#term-runtime) the cross-[backend](../../glossary.md#backend) [lifecycle](../../glossary.md#lifecycle) facts it must coordinate without making it a
physical storage abstraction. Backend code retains the concrete Java types needed for safe direct
access. A single whole-[run](../../glossary.md#term-run) [`RunState`](../../glossary.md#run-state--runstate) can represent transfers between prepared partitions while
still isolating concurrent invocations. Cold [binding](../../glossary.md#term-binding) pays [heterogeneous](../../glossary.md#term-heterogeneous) compatibility costs once
and keeps the execution path predictable.

## Consequences

### Positive

- One [immutable](../../glossary.md#term-immutable) [prepared execution](../../glossary.md#prepared-execution--preparedexecution) can serve any number of isolated concurrent runs.
- Physical cleanup stays with the code that understands the physical representation.
- Explicit prepared representations and transfers support heterogeneous schedules without hidden
  backend discovery or write-back.<span data-glossary-navigation="true"> ([heterogeneous glossary](../../glossary.md#term-heterogeneous))</span>
- [Backend](../../glossary.md#backend)-owned bound invocations can use direct typed references in the [hot path](../../glossary.md#term-hot-path).

### Negative and risks

- Java cannot express the complete heterogeneous type relationship in one shared static generic
  signature, so a checked dynamic boundary remains necessary.<span data-glossary-navigation="true"> ([heterogeneous glossary](../../glossary.md#term-heterogeneous); [signature glossary](../../glossary.md#term-signature))</span>
- Runtime must maintain careful ownership transitions so failure cleanup does not close borrowed
  inputs or outputs already transferred to a result.<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime); [ownership glossary](../../glossary.md#term-ownership-meanings); [borrowed glossary](../../glossary.md#ownership-borrowing-and-lease))</span>
- Multiple explicit representations require later validity/residency state and prepared transfer
  associations; a nominal carrier alone is not a coherence protocol.<span data-glossary-navigation="true"> ([residency glossary](../../glossary.md#residency); [nominal glossary](../../glossary.md#term-nominal); [carrier glossary](../../glossary.md#term-storage-carrier))</span>
- Concrete backend implementations must test idempotent lifecycle behavior and partial-failure
  cleanup around their physical resources.<span data-glossary-navigation="true"> ([Concrete backend glossary](../../glossary.md#concrete-backend); [idempotent glossary](../../glossary.md#term-idempotent); [lifecycle glossary](../../glossary.md#lifecycle))</span>

### Migration, testing, and follow-up

[Runtime](../../glossary.md#term-runtime) 0003 introduces only the minimal per-[run](../../glossary.md#term-run) representation/[lifecycle](../../glossary.md#lifecycle) [carrier](../../glossary.md#term-storage-carrier) and array-
indexed [`RunState`](../../glossary.md#run-state--runstate) foundation. Runtime 0004 defines typed cold-bound executable invocation. Later
Runtime/[Prepare](../../glossary.md#prepare) tasks define full representation validity/[residency](../../glossary.md#residency), transfer, [publication](../../glossary.md#term-publication),
[`RunResult`](../../glossary.md#run-result--runresult), and runner behavior. No automatic resource [pooling](../../glossary.md#term-pooling)/reuse/[aliasing](../../glossary.md#term-aliasing), distributed
sharding, hidden mutation/coherence, or multi-[device](../../glossary.md#term-device) scheduling is authorized.

The decision changes no [module](../../glossary.md#term-module) [dependency direction](../../glossary.md#java-platform-and-build-tools). Runtime still depends only on existing
inward modules; Prepare remains independent of concrete [backends](../../glossary.md#backend); Backend Contract remains a
closed declarative [leaf](../../glossary.md#term-leaf); and concrete backends implement inward Runtime/Prepare contracts.
Architecture tests therefore need no update for this [ADR](../../glossary.md#term-architecture-decision-record-adr).

## Related documentation

- [Architecture contract](../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../glossary.md#architecture-contract))</span>
- [Lifecycle](../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../glossary.md#lifecycle))</span>
- [Runtime, Prepare, and Backend Boundary](../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime); [Prepare glossary](../../glossary.md#prepare); [Backend glossary](../../glossary.md#backend))</span>
- [ADR 0006: No runtime service locator](0006-no-runtime-service-locator.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr); [service locator glossary](../../glossary.md#term-service-locator))</span>
- [ADR 0010: Staged backend preparation](0010-staged-backend-preparation.md)
- [Runtime master plan](../../planning/modules/runtime/master-plan.md)
