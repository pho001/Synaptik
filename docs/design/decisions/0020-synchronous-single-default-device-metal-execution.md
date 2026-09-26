# ADR 0020: Synchronous Single-Default-Device Metal Execution

## Status

Accepted — 2026-09-25

## Context

Synaptik currently exposes synchronous public and Runtime run methods. Runtime creates one isolated
mutable state, cold-binds all actions, traverses the prepared schedule, and returns a `RunResult`
only after execution and prepared-lease release succeed. Metal native routes commit their work and
wait for device completion before returning. Outputs may nevertheless remain live after the method
returns because the result owns the completed run-state publication lease.

A Metal integration opens one context through the system-default-device path. Configuration names
the caller-supplied native library, not a physical device. Planning proves abstract eligibility but
retains backend ownership rather than a selected Metal device. Trace and tuning also carry tokens
that could be misread as hardware identities even though they are not.

Without an explicit decision, synchronous return can be mistaken for output-lifetime completion,
concurrent caller support can be mistaken for an overlap or scheduling guarantee, and the abstract
`"default"`/zero/session tokens can be mistaken for one stable device identity. Those mistakes
would make later async, pooling, explicit-device, multi-device, trace, or cache work start from a
false contract.

## Decision drivers

- State exactly when current execution and native device work complete.
- Keep result publication/resource lifetime distinct from work completion.
- Preserve safe concurrent calls without promising overlap, order, fairness, or throughput.
- State the exact one-integration/one-default-context topology and context-bound resource rule.
- Prevent unrelated backend, trace, and tuning tokens from becoming an accidental physical-device
  identity.
- Retain the reviewed persistent-immutable-splat optimization without authorizing a general pool.
- Give future async and explicit-device/multi-device work concrete contract prerequisites.
- Preserve every current API, ABI, capability, route, trace, cache, and tuning identity.

## Options considered

### Infer completion and device topology from implementation

Implementation details are easy to misread and do not establish a durable cross-module boundary.
They also leave public result lifetime, planner identity, trace identity, and tuning identity
ambiguous. Rejected.

### Add asynchronous wrappers or device selection now

There is no concrete end-to-end consumer requirement or evidence that the synchronous barrier or
system-default choice is the limiting constraint. Adding futures, events, cancellation, selectors,
enumeration, or scheduling without defining ownership, failure, close races, identity, and
cross-device behavior would be an incomplete contract. Rejected.

### Treat concurrent calls as a device-overlap guarantee

Isolated run state makes concurrent callers safe; it does not specify native queue behavior,
submission or completion order, overlap, fairness, or throughput. Rejected.

### Treat default, zero, and session tokens as one device identity

`BackendDeviceId(metal, "default")`, `TraceDeviceId(0)`, and the tuning `SessionNonce` serve
different scopes. None is a stable physical-device selector or fingerprint. Equating them would
make trace and cache semantics false. Rejected.

### Record the current synchronous, single-default-device boundary

This states observable behavior without expanding implementation or identity. Selected.

## Decision

### Completion and result lifetime

A successful public or Runtime run return is a synchronous completion barrier for that invocation.
Every prepared schedule action and native operation has completed, and published destinations are
usable. Materialization and transfers are synchronous at their existing call boundaries. A failed
execution returns no `RunResult`.

A `RunResult` is a completed-state resource/publication lease over the finished run state. It may
remain open after the run call; it is not a future, event, fence, command-buffer handle, or
representation of in-flight work. Runtime `RunResult` and `RunState` remain non-thread-safe. The
outward Engine result retains its existing synchronized materialization-versus-close arbitration.

`PreparedExecution` admits exactly one synchronous run per acquired execution lease. Close rejects
new acquisition without waiting. An admitted run may finish, and deferred persistent-resource
cleanup occurs on its final lease release. Successful execution releases the prepared execution
lease before returning the completed-state result; result closure separately releases run-owned
resources.

Concurrent callers receive isolated mutable run state. This permits concurrent calls but promises
no GPU overlap, submission order, completion order, fairness, queue multiplicity, throughput, or
scheduler behavior. Schedule order is defined only within one invocation. There is no asynchronous
run, cancellation, timeout, polling, callback, completion-stage, or explicit event API.

### Metal device topology

Each `MetalBackendIntegration` owns exactly one native context created through the system-default-
device path and one associated command queue. `MetalBackendConfiguration` selects only the native
library; no public physical-device selector or enumeration exists. The availability snapshot's
`BackendDeviceId(metal, "default")` is the one abstract Metal availability slot, not a stable
hardware identity or selection handle.

An Engine registry is keyed by `BackendId` and rejects duplicates, so one `Engine` owns at most one
Metal integration. Planning may use an exact-device requirement to prove eligibility against the
snapshot, but the selected plan retains the owning `BackendId`, not a physical Metal device.
Prepared buffers, workspaces, route resources, and native handles remain authenticated and bound to
the exact `MetalDeviceContext`; foreign-context or foreign-`MTLDevice` use fails closed.

Separate Engines or integrations may create separate contexts, but they are not a coordinated
multi-device system and may resolve to the same default device. There is no device enumeration,
explicit selection, load balancing, topology, failover, hot-plug refresh, cross-device transfer,
migration, replication, coherence, sharding, or scheduling.

### Identity separation

`BackendDeviceId(metal, "default")` is an abstract availability identity. `TraceDeviceId(0)` is a
fixed stream-local Metal trace correlation token. The tuning `SessionNonce` identifies one
session-compatibility context and explicitly carries no stable-device claim. They remain distinct;
no mapping among them or to a physical device is introduced.

Metal uses ABI 5 with the same thirteen export names, bounded route-bearing node schema 15, and
version-16 workload, exact-policy, candidate, compatibility, route, and codec identities. Task
0059's clean schema/identity cutover changes none of this decision's device-identity semantics.
Tuning remains session-scoped and non-persistent. No cache key, trace payload, or compatibility
identity gains a physical-device fingerprint.

### Persistent immutable resources and no general pool

Immutable source-owned splat buffers may persist only within their exact `PreparedExecution` and
issue fresh authenticated read-only run bindings. Mutable outputs and MPSGraph address workspaces
remain fresh per run. Synchronous device completion does not make those resources generally
poolable: an open `RunResult` still owns output resources, and address workspaces contain mutable
per-run pointer arrays. No general output/workspace pool, backend-global constant cache, or cross-
preparation reuse is authorized.

## Successor triggers

### Asynchronous execution

A successor requires a concrete end-to-end consumer need and evidence that synchronous completion
is the limiting constraint. A separately reviewed cross-module Class C decision must first define:
completion and error delivery; cancellation/timeout behavior; input borrow duration; output and
workspace ownership before and after completion; prepared-lease lifetime through terminal device
completion; close races; cleanup-failure ownership; dependencies among Metal, CPU, transfer,
materialization, and publication work; trace submission versus terminal outcomes; tuning timing and
identity; and only then pool return/reset/capacity/eviction/context-close behavior. It must review
all affected public API, Runtime/Backend SPI, native ABI, version/capability, route, trace, cache,
tuning, and ownership contracts.

### Explicit-device or multi-device execution

A successor requires a real supported multi-device environment and concrete workflow. The smallest
successor should first consider selecting one device per integration or Engine, not a scheduler. It
must define a stable selector, truthful enumeration and failure behavior, native enumeration/context-
creation ABI evolution, and trace/tuning identity for the selected device. True multi-device work
additionally requires registry identity beyond one `BackendId`, planner retention of selected
`BackendDeviceId`, device-qualified partitions/resources, cross-device transfer and coherence,
topology, scheduling, fallback, hot-plug, and lifetime rules. Persistent reuse additionally requires
a truthful stable physical-device fingerprint. Real-device evidence and ABI/version/export review
are mandatory.

## Consequences

### Positive

- Callers can distinguish completed work from the lifetime of published resources.
- Concurrent-run safety no longer implies undocumented scheduling or performance behavior.
- Planning, trace, and tuning identities cannot be mistaken for a stable physical-device selector.
- Context-bound resources and fail-closed foreign-device checks remain explicit.
- Future async, pool, and multi-device proposals have concrete prerequisites.

### Negative and limits

- The caller blocks until the current run and its native work finish.
- A caller cannot select or enumerate Metal devices through Synaptik.
- Multiple independent integrations do not provide coordinated multi-device execution.
- Output and workspace allocation remain per run; only immutable source-owned splats persist per
  prepared execution.

## Compatibility

This decision documents current behavior only. It changes no public API, Runtime/Backend SPI,
native ABI or export, schema, capability, occurrence support, route, schedule, transfer,
materialization, trace event/token, cache, tuning identity, session scope, error category, cleanup
order, or resource lifetime. It adds no code, tests, build, benchmark, or device probe.

## Related documentation

- [Backend execution contract](../../architecture/contracts/backend-execution.md#metal-backend)
- [Runtime/Prepare/Engine contract](../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)
- [Public API](../../api/public-api.md)
- [Runtime API](../../api/runtime-api.md#current-prepared-runner)
- [Metal backend guide](../../backend-guide/metal-backend.md)
- [Task 0049](../../planning/backends/metal/tasks/0049-synchronous-single-default-device-contract.md)
