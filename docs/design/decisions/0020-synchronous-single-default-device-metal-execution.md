# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0020: [Synchronous](../../glossary.md#term-synchronous) Single-Default-[Device](../../glossary.md#term-device) [Metal](../../glossary.md#term-metal) Execution


## Editorial reading context

The application calls run and receives a result only after synchronous work completes. That does not release the completed output buffers: the result keeps their lifetime until close. A Metal context identifies a concrete integration; the abstract availability token default is not a stable physical-device ID. Concurrent calls do not themselves promise overlapping GPU work.

Use the [current Metal execution and resource boundaries](../../backend-guide/metal-backend.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted — 2026-09-25

## Context

Synaptik currently exposes [synchronous](../../glossary.md#term-synchronous) public and [Runtime](../../glossary.md#term-runtime) [run](../../glossary.md#term-run) methods. Runtime creates one isolated
[mutable](../../glossary.md#term-mutable) state, cold-binds all actions, traverses the [prepared schedule](../../glossary.md#prepared-schedule--preparedschedule), and returns a [`RunResult`](../../glossary.md#run-result--runresult)
only after execution and prepared-[lease](../../glossary.md#term-lease) [release](../../glossary.md#term-release) succeed. [Metal](../../glossary.md#term-metal) native routes [commit](../../glossary.md#term-commit) their work and
wait for [device](../../glossary.md#term-device) completion before returning. Outputs may nevertheless remain live after the method
returns because the result owns the completed run-state [publication](../../glossary.md#term-publication) lease.

A Metal integration opens one context through the system-default-device path. Configuration names
the caller-supplied [native library](../../glossary.md#term-native-library), not a physical device. [Planning](../../glossary.md#term-planning) proves abstract eligibility but
retains [backend ownership](../../glossary.md#backend-ownership) rather than a selected Metal device. [Trace](../../glossary.md#trace) and tuning also carry [tokens](../../glossary.md#term-token)
that could be misread as hardware [identities](../../glossary.md#term-identity-meanings) even though they are not.

Without an explicit decision, synchronous return can be mistaken for output-[lifetime](../../glossary.md#term-lifetime) completion,
concurrent caller support can be mistaken for an overlap or scheduling guarantee, and the abstract
`"default"`/zero/session tokens can be mistaken for one stable device [identity](../../glossary.md#term-identity-meanings). Those mistakes
would make later async, pooling, explicit-device, multi-device, trace, or [cache](../../glossary.md#term-cache) work start from a
false contract.

## Decision drivers

- State exactly when current execution and native [device](../../glossary.md#term-device) work complete.
- Keep result [publication](../../glossary.md#term-publication)/[resource lifetime](../../glossary.md#ownership-borrowing-and-lease) distinct from work completion.
- Preserve safe concurrent calls without promising overlap, order, fairness, or [throughput](../../glossary.md#term-throughput).
- State the exact one-integration/one-default-context [topology](../../glossary.md#term-topology) and context-bound resource rule.
- Prevent unrelated backend, trace, and tuning tokens from becoming an accidental physical-device
  identity.<span data-glossary-navigation="true"> ([backend glossary](../../glossary.md#backend); [tokens glossary](../../glossary.md#term-token))</span>
- Retain the reviewed persistent-[immutable](../../glossary.md#term-immutable)-splat optimization without authorizing a general pool.
- Give future async and explicit-device/multi-device work concrete contract prerequisites.
- Preserve every current [API](../../glossary.md#term-application-programming-interface-api), [ABI](../../glossary.md#term-application-binary-interface-abi), capability, route, [trace](../../glossary.md#trace), [cache](../../glossary.md#term-cache), and tuning [identity](../../glossary.md#term-identity-meanings).

## Options considered

### Infer completion and [device](../../glossary.md#term-device) [topology](../../glossary.md#term-topology) from implementation

Implementation details are easy to misread and do not establish a durable cross-[module](../../glossary.md#term-module) boundary.
They also leave public result [lifetime](../../glossary.md#term-lifetime), planner [identity](../../glossary.md#term-identity-meanings), [trace](../../glossary.md#trace) identity, and tuning identity
ambiguous. Rejected.

### Add [asynchronous](../../glossary.md#term-asynchronous) wrappers or [device](../../glossary.md#term-device) selection now

There is no concrete end-to-end consumer requirement or evidence that the [synchronous](../../glossary.md#term-synchronous) barrier or
system-default choice is the limiting constraint. Adding futures, events, [cancellation](../../glossary.md#term-cancellation), selectors,
enumeration, or scheduling without defining [ownership](../../glossary.md#term-ownership-meanings), failure, close races, [identity](../../glossary.md#term-identity-meanings), and
cross-device behavior would be an incomplete contract. Rejected.

### Treat concurrent calls as a [device](../../glossary.md#term-device)-overlap guarantee

Isolated [run state](../../glossary.md#run-state--runstate) makes concurrent callers safe; it does not specify native queue behavior,
submission or completion order, overlap, fairness, or [throughput](../../glossary.md#term-throughput). Rejected.

### Treat default, zero, and session [tokens](../../glossary.md#term-token) as one [device](../../glossary.md#term-device) [identity](../../glossary.md#term-identity-meanings)

`BackendDeviceId(metal, "default")`, `TraceDeviceId(0)`, and the tuning `SessionNonce` serve
different scopes. None is a stable physical-device selector or [fingerprint](../../glossary.md#term-fingerprint). Equating them would
make [trace](../../glossary.md#trace) and [cache](../../glossary.md#term-cache) semantics false. Rejected.

### Record the current [synchronous](../../glossary.md#term-synchronous), single-default-[device](../../glossary.md#term-device) boundary

This states observable behavior without expanding implementation or [identity](../../glossary.md#term-identity-meanings). Selected.

## Decision

### Completion and result [lifetime](../../glossary.md#term-lifetime)

A successful public or [Runtime](../../glossary.md#term-runtime) [run](../../glossary.md#term-run) return is a [synchronous](../../glossary.md#term-synchronous) completion barrier for that invocation.
Every [prepared schedule](../../glossary.md#prepared-schedule--preparedschedule) action and native [operation](../../glossary.md#operation) has completed, and published destinations are
usable. [Materialization](../../glossary.md#materialization) and transfers are synchronous at their existing call boundaries. A failed
execution returns no [`RunResult`](../../glossary.md#run-result--runresult).

A `RunResult` is a completed-state resource/[publication](../../glossary.md#term-publication) [lease](../../glossary.md#term-lease) over the finished [run state](../../glossary.md#run-state--runstate). It may
remain open after the run call; it is not a future, event, fence, command-[buffer](../../glossary.md#term-buffer) handle, or
representation of in-flight work. Runtime `RunResult` and [`RunState`](../../glossary.md#run-state--runstate) remain non-[thread-safe](../../glossary.md#term-thread-safe). The
outward [Engine](../../glossary.md#term-engine) result retains its existing synchronized materialization-versus-close arbitration.

[`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution) admits exactly one synchronous run per acquired execution lease. Close rejects
new acquisition without waiting. An admitted run may finish, and deferred persistent-resource
cleanup occurs on its final lease [release](../../glossary.md#term-release). Successful execution [releases](../../glossary.md#commit-release-distribution-log-and-seed-meanings) the [prepared execution](../../glossary.md#prepared-execution--preparedexecution)
lease before returning the completed-state result; result closure separately releases run-owned
resources.

Concurrent callers receive isolated [mutable](../../glossary.md#term-mutable) run state. This permits concurrent calls but promises
no [GPU](../../glossary.md#term-gpu) overlap, submission order, completion order, fairness, queue multiplicity, [throughput](../../glossary.md#term-throughput), or
scheduler behavior. Schedule order is defined only within one invocation. There is no [asynchronous](../../glossary.md#term-asynchronous)
run, [cancellation](../../glossary.md#term-cancellation), timeout, polling, [callback](../../glossary.md#term-callback), completion-stage, or explicit event [API](../../glossary.md#term-application-programming-interface-api).

### [Metal](../../glossary.md#term-metal) [device](../../glossary.md#term-device) [topology](../../glossary.md#term-topology)

Each `MetalBackendIntegration` owns exactly one native context created through the system-default-
device path and one associated [command queue](../../glossary.md#term-command-queue). `MetalBackendConfiguration` selects only the [native
library](../../glossary.md#term-native-library); no public physical-device selector or enumeration exists. The availability [snapshot](../../glossary.md#term-snapshot)'s
`BackendDeviceId(metal, "default")` is the one abstract Metal availability slot, not a stable
hardware [identity](../../glossary.md#term-identity-meanings) or selection handle.

An [Engine](../../glossary.md#term-engine) [registry](../../glossary.md#term-registry) is keyed by [`BackendId`](../../glossary.md#backend-identity--backendid) and rejects duplicates, so one `Engine` owns at most one
Metal integration. [Planning](../../glossary.md#term-planning) may use an exact-device requirement to prove eligibility against the
snapshot, but the selected plan retains the owning `BackendId`, not a physical Metal device.
Prepared [buffers](../../glossary.md#term-buffer), workspaces, route resources, and native handles remain authenticated and bound to
the exact `MetalDeviceContext`; foreign-context or foreign-`MTLDevice` use fails closed.

Separate Engines or integrations may create separate contexts, but they are not a coordinated
multi-device system and may resolve to the same default device. There is no device enumeration,
explicit selection, load balancing, topology, failover, hot-plug refresh, cross-device transfer,
migration, replication, coherence, sharding, or scheduling.

### [Identity](../../glossary.md#term-identity-meanings) separation

`BackendDeviceId(metal, "default")` is an abstract availability identity. `TraceDeviceId(0)` is a
fixed stream-local [Metal](../../glossary.md#term-metal) [trace](../../glossary.md#trace) correlation [token](../../glossary.md#term-token). The tuning `SessionNonce` identifies one
session-compatibility context and explicitly carries no stable-[device](../../glossary.md#term-device) claim. They remain distinct;
no mapping among them or to a physical device is introduced.

Metal uses [ABI](../../glossary.md#term-application-binary-interface-abi) 5 with the same thirteen export names, bounded route-bearing [node](../../glossary.md#node) [schema](../../glossary.md#term-schema) 18, and
version-27 workload, exact-policy, candidate, compatibility, route, and [codec](../../glossary.md#term-codec) identities. Identity
26 and every older value [fail closed](../../glossary.md#admission-preflight-and-fail-closed-validation) without changing this decision's device-identity semantics.
Tuning remains session-scoped and non-persistent. No [cache key](../../glossary.md#term-cache-key), [trace payload](../../glossary.md#trace-payload--tracepayload), or [compatibility
identity](../../glossary.md#identity-meanings-and-algebraic-identities) gains a physical-device [fingerprint](../../glossary.md#term-fingerprint).

### Persistent [immutable](../../glossary.md#term-immutable) resources and no general pool

Immutable source-owned splat [buffers](../../glossary.md#term-buffer) may persist only within their exact [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution) and
issue fresh authenticated read-only [run](../../glossary.md#term-run) [bindings](../../glossary.md#term-binding). [Mutable](../../glossary.md#term-mutable) outputs and [MPSGraph](../../glossary.md#term-mpsgraph) address workspaces
remain fresh per run. [Synchronous](../../glossary.md#term-synchronous) [device](../../glossary.md#term-device) completion does not make those resources generally
poolable: an open [`RunResult`](../../glossary.md#run-result--runresult) still owns output resources, and address workspaces contain mutable
per-run pointer arrays. No general output/workspace pool, [backend](../../glossary.md#backend)-global [constant](../../glossary.md#term-constant) [cache](../../glossary.md#term-cache), or cross-
preparation reuse is authorized.

## Successor triggers

### [Asynchronous](../../glossary.md#term-asynchronous) execution

A successor requires a concrete end-to-end consumer need and evidence that [synchronous](../../glossary.md#term-synchronous) completion
is the limiting constraint. A separately reviewed cross-[module](../../glossary.md#term-module) Class C decision must first define:
completion and error delivery; [cancellation](../../glossary.md#term-cancellation)/timeout behavior; input borrow duration; output and
workspace [ownership](../../glossary.md#term-ownership-meanings) before and after completion; prepared-[lease](../../glossary.md#term-lease) [lifetime](../../glossary.md#term-lifetime) through terminal [device](../../glossary.md#term-device)
completion; close races; cleanup-failure ownership; dependencies among [Metal](../../glossary.md#term-metal), [CPU](../../glossary.md#term-cpu), transfer,
[materialization](../../glossary.md#materialization), and [publication](../../glossary.md#term-publication) work; [trace](../../glossary.md#trace) submission versus terminal outcomes; tuning timing and
[identity](../../glossary.md#term-identity-meanings); and only then pool return/reset/capacity/eviction/context-close behavior. It must review
all affected public [API](../../glossary.md#term-application-programming-interface-api), [Runtime](../../glossary.md#term-runtime)/[Backend](../../glossary.md#backend) [SPI](../../glossary.md#term-service-provider-interface-spi), native [ABI](../../glossary.md#term-application-binary-interface-abi), version/capability, route, trace, [cache](../../glossary.md#term-cache),
tuning, and ownership contracts.

### Explicit-[device](../../glossary.md#term-device) or multi-device execution

A successor requires a real supported multi-device environment and concrete workflow. The smallest
successor should first consider selecting one device per integration or [Engine](../../glossary.md#term-engine), not a scheduler. It
must define a stable selector, truthful enumeration and failure behavior, native enumeration/context-
creation [ABI](../../glossary.md#term-application-binary-interface-abi) evolution, and [trace](../../glossary.md#trace)/tuning [identity](../../glossary.md#term-identity-meanings) for the selected device. True multi-device work
additionally requires [registry](../../glossary.md#term-registry) identity beyond one [`BackendId`](../../glossary.md#backend-identity--backendid), planner retention of selected
[`BackendDeviceId`](../../glossary.md#backend-device-identity--backenddeviceid), device-qualified partitions/resources, cross-device transfer and coherence,
[topology](../../glossary.md#term-topology), scheduling, fallback, hot-plug, and [lifetime](../../glossary.md#term-lifetime) rules. Persistent reuse additionally requires
a truthful stable physical-device [fingerprint](../../glossary.md#term-fingerprint). Real-device evidence and ABI/version/export review
are mandatory.

## Consequences

### Positive

- Callers can distinguish completed work from the [lifetime](../../glossary.md#term-lifetime) of published resources.
- Concurrent-[run](../../glossary.md#term-run) safety no longer implies undocumented scheduling or performance behavior.
- [Planning](../../glossary.md#term-planning), [trace](../../glossary.md#trace), and tuning [identities](../../glossary.md#term-identity-meanings) cannot be mistaken for a stable physical-[device](../../glossary.md#term-device) selector.
- Context-bound resources and [fail-closed](../../glossary.md#term-fail-closed) foreign-device checks remain explicit.
- Future async, pool, and multi-device proposals have concrete prerequisites.

### Negative and limits

- The caller blocks until the current [run](../../glossary.md#term-run) and its native work finish.
- A caller cannot select or enumerate [Metal](../../glossary.md#term-metal) devices through Synaptik.
- Multiple independent integrations do not provide coordinated multi-[device](../../glossary.md#term-device) execution.
- Output and workspace allocation remain per run; only immutable source-owned splats persist per
  prepared execution.<span data-glossary-navigation="true"> ([immutable glossary](../../glossary.md#term-immutable); [prepared execution glossary](../../glossary.md#prepared-execution--preparedexecution))</span>

## Compatibility

This decision documents current behavior only. It changes no public [API](../../glossary.md#term-application-programming-interface-api), [Runtime](../../glossary.md#term-runtime)/[Backend](../../glossary.md#backend) [SPI](../../glossary.md#term-service-provider-interface-spi),
native [ABI](../../glossary.md#term-application-binary-interface-abi) or export, [schema](../../glossary.md#term-schema), capability, [occurrence](../../glossary.md#term-occurrence) support, route, schedule, transfer,
[materialization](../../glossary.md#materialization), [trace](../../glossary.md#trace) event/[token](../../glossary.md#term-token), [cache](../../glossary.md#term-cache), tuning [identity](../../glossary.md#term-identity-meanings), session [scope](../../glossary.md#term-scope), error category, cleanup
order, or [resource lifetime](../../glossary.md#ownership-borrowing-and-lease). It adds no code, tests, build, benchmark, or [device](../../glossary.md#term-device) probe.

## Related documentation

- [Backend execution contract](../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Backend glossary](../../glossary.md#backend))</span>
- [Runtime/Prepare/Engine contract](../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime); [Prepare glossary](../../glossary.md#prepare); [Engine glossary](../../glossary.md#term-engine))</span>
- [Public API](../../api/public-api.md)<span data-glossary-navigation="true"> ([API glossary](../../glossary.md#term-application-programming-interface-api))</span>
- [Runtime API](../../api/runtime-api.md#current-prepared-runner)
- [Metal backend guide](../../backend-guide/metal-backend.md)<span data-glossary-navigation="true"> ([Metal glossary](../../glossary.md#term-metal))</span>
- [Task 0049](../../planning/backends/metal/tasks/0049-synchronous-single-default-device-contract.md)
