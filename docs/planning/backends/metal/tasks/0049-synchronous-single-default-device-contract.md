# Task 0049: Synchronous single-default-device Metal contract

## Status

Review needed

Executed from planning commit `e32ad68dbb24b3c9bec9a2bdb09159f0db56ee6a` on clean Task 0048
finalization revision `a4ff40c2b4fc328926c15722241f7359744ff4ba`. The exact nine researched
Markdown paths now record the current contract. Independent Class C documentation review is
pending; Task 0049 must not be marked Complete before approval.

## Change class

Class C. Documentation-only cross-module execution/lifetime/device contract decision. Independent
Class C review is mandatory before Complete.

## Intent

Codify the behavior the current implementation already provides:

- public and Runtime execution is a synchronous completion barrier;
- a successful `RunResult` is a completed-state resource/publication lease, not a future or device
  completion primitive;
- each `MetalBackendIntegration` owns one Metal context created from the system-default device and
  offers no public device selection or multi-device scheduling surface;
- concurrent callers are supported through isolated mutable run state, but this does not promise
  device overlap, submission or completion order, fairness, queue topology, or throughput; and
- immutable source-owned splats may persist per `PreparedExecution`, while no general mutable
  output/workspace pool is authorized.

This task records existing truth only. It does not add an asynchronous API, futures, events,
cancellation, timeout, device enumeration, explicit device selection, multi-device scheduling,
cross-device migration/coherence, or pooling.

## Decision boundary

### Synchronous completion barrier

A successful public or Runtime `run(...)` return means every scheduled action and native operation
for that invocation completed and its published destinations are usable. Native Metal execution
commits and waits for completion before returning. Materialization and transfers on the execution
path are also synchronous. A failed execution publishes no `RunResult`.

The returned `RunResult` owns completed run-state resources and the publication lifetime of outputs.
It may outlive the call. It is not a promise, future, event, fence, command buffer, or handle to
in-flight work. Runtime `RunResult` and `RunState` remain non-thread-safe; the outward Engine result
retains its existing synchronized materialization-versus-close behavior.

Prepared-execution acquisition leases exactly one synchronous run. Close rejects new acquisitions,
does not wait, and defers owned-resource cleanup only through already-active execution/result
leases. The prepared owner must remain alive through native completion; the current synchronous
barrier establishes that point before the result is returned.

Concurrent calls may execute from distinct caller threads and use isolated mutable run state. This
is not a promise of GPU overlap, command ordering between runs, fairness, queue multiplicity,
submission policy, completion ordering, throughput, or scheduler behavior. Schedule order remains
the order within one invocation.

There is no asynchronous run, cancellation, timeout, polling, callback, completion-stage, explicit
event, or deferred-error API.

### One system-default device context per integration

One `MetalBackendIntegration` creates and owns one native Metal device context by calling the system
default-device path and creating its command queue. Configuration selects only the native library;
it exposes no physical-device selector. The runtime availability snapshot presents the abstract
`BackendDeviceId(metal, "default")` slot.

One `Engine` may register at most one integration for the Metal `BackendId`; duplicate backend
registration is rejected. Planning uses device requirements only to establish eligibility and
retains the owning `BackendId`, not a selected physical Metal device. Resources remain authenticated
and bound to their exact `MetalDeviceContext`; native code rejects resources from another
`MTLDevice`.

Separate Engines or integrations may create separate contexts, but this is not coordinated
multi-device execution and they may resolve to the same system-default physical device. The current
contract provides no device enumeration, stable hardware selector, explicit selection, load
balancing, topology, failover, hot-plug refresh, cross-device transfer, migration, replication,
coherence, partitioning, or scheduling.

### Identity separation

The following identities are deliberately distinct and none is a stable physical-device identity:

- `BackendDeviceId(metal, "default")` is the sole abstract availability slot;
- `TraceDeviceId(0)` is the fixed Metal trace device token used with stream-local correlation; and
- the tuning `SessionNonce` distinguishes a process/session compatibility context and explicitly
  makes no stable device claim.

The task changes none of these identities and introduces no mapping among them. Tuning remains
version 12, ABI v4, session-scoped, and non-persistent. There is no persistent route or tuning cache
whose reuse could imply a physical-device fingerprint.

### Resource ownership and no-general-pool closure

Immutable source-owned splat buffers continue to persist exactly within their owning
`PreparedExecution` and supply fresh read-only run bindings. Mutable outputs and address workspaces
remain per-run resources. Synchronous device completion alone does not end their ownership: a
`RunResult` can retain output resources after `run(...)` returns, and address workspaces carry
per-run mutable pointer arrays. This decision therefore retains Task 0048's rejection of a general
output/workspace pool.

## Compatibility invariants

This task preserves every existing:

- public Java API and module boundary;
- Runtime/Backend SPI and behavior;
- native ABI v4 version, thirteen-export set, status space, node schema 11, and ownership rule;
- capability matrix, occurrence support, route selection, fallback, and eligibility rule;
- schedule topology and action order;
- trace event kind, phase, backend/device token, correlation, and failure behavior;
- workload, exact-policy, candidate, compatibility, route, and codec identity at version 12;
- tuning session scope, `SessionNonce`, cache behavior, and absence of persistence;
- resource, prepared-execution, result, transfer, materialization, and close lifetime; and
- Task 0048 immutable-splat optimization and no-general-pool decision.

No version bump, capability increase, route change, serialization, cache, benchmark, retry,
fallback, packaging, discovery, or performance claim is authorized.

## Concrete successor triggers

### Asynchronous execution

A successor may be planned only after a concrete end-to-end consumer need and evidence that the
synchronous barrier is the limiting constraint. It must be a separately reviewed cross-module Class
C decision that defines, before implementation:

1. completion and failure delivery, including when native asynchronous errors become observable;
2. cancellation and timeout semantics and their relationship to already-submitted device work;
3. input borrowing duration and exact output/workspace ownership before and after completion;
4. prepared-execution acquisition lifetime through terminal device completion, including close
   races and cleanup-failure ownership;
5. dependency representation across Metal work, CPU work, transfers, materialization, and public
   publication;
6. trace truth for submission versus terminal outcome and representative tuning timing/identity;
   and
7. only after those rules, any pool return/reset, capacity, eviction, and context-close protocol.

An asynchronous successor must review all affected public API, Runtime/Backend SPI, native ABI,
version/capability, route, trace, cache, tuning, and ownership contracts rather than attaching a
future to the current synchronous method.

### Explicit-device or multi-device execution

A successor may be planned only for a real supported multi-device environment and concrete user
workflow. The smallest successor should first consider selecting exactly one device per integration
or Engine rather than introducing a scheduler. It must define and prove:

1. a stable selector and truthful enumeration/availability/failure behavior;
2. any native enumeration/context-create ABI evolution and export/version consequences;
3. a registry identity that can represent more than one integration for a backend if true
   multi-device ownership is required;
4. planner retention of the selected `BackendDeviceId`, device-qualified partitions/resources,
   and resource-context authentication;
5. cross-device transfer, migration, replication, coherence, topology, fallback, hot-plug, and
   scheduling/lifetime rules required by the chosen scope;
6. trace mapping to the selected device and tuning compatibility keyed by truthful selected-device
   and session information; and
7. a stable physical-device fingerprint before any persistent tuning/cache reuse is claimed.

The successor must preserve fail-closed behavior until those contracts and real-device evidence are
available. Multiple independent default-device contexts are not evidence of multi-device support.

## Dependencies and integration

- Depends on: Task 0048 Complete at independently approved implementation `89f9fbb9` plus
  documentation finalization `a4ff40c2`; Runtime 0016; Prepare 0006; Engine 0010; Planning's current
  device eligibility model; Trace 0003; Metal 0041–0042
- Conflicts with: every concurrent edit to the nine files below, or any asynchronous execution,
  result/resource lifetime, Metal device/context, planner device-selection, native ABI, trace,
  cache, tuning, route, capability, or pooling decision
- Parallel group: None
- Common base revision: `a4ff40c2b4fc328926c15722241f7359744ff4ba`
- Integration order: planning commit `e32ad68d`; documentation execution; independent Class C
  review before Complete
- Integration validation: exact nine-path scope, local links, balanced fences, final newlines, and
  `git diff --check` passed; clean committed status is required for handoff; no code, tests, Javadocs,
  builds, benchmarks, or probes ran
- Shared-document integration owner: Task 0049 implementation owner

## Exact file scope

Task 0049 may change exactly these nine researched Markdown paths:

1. `docs/planning/backends/metal/tasks/0049-synchronous-single-default-device-contract.md`
2. `docs/planning/backends/metal/master-plan.md`
3. `docs/planning/roadmap.md`
4. `docs/design/decisions/0020-synchronous-single-default-device-metal-execution.md`
5. `docs/architecture/contracts/backend-execution.md`
6. `docs/architecture/contracts/runtime-prepare-engine.md`
7. `docs/api/public-api.md`
8. `docs/api/runtime-api.md`
9. `docs/backend-guide/metal-backend.md`

No Java, Objective-C, Gradle, workflow, test, benchmark, generated, native, package, cache, or probe
path is in scope.

## Documentation execution plan

1. Add ADR 0020 as Accepted, recording the synchronous completion barrier, completed-state
   `RunResult` lease, one system-default context per integration, identity separation,
   context-bound resources, no multi-device behavior, successor triggers, and rejected pool.
2. Align Backend execution architecture with synchronous completion and context-bound resource
   ownership without inventing a public completion primitive or physical-device identity.
3. Align Runtime/Prepare/Engine architecture with one synchronous lease, publication lifetime,
   close races, concurrent-caller non-guarantees, and one Metal integration per Engine.
4. Clarify public and Runtime API references: successful return is completed state; the result is a
   resource/publication lease; there is no async/cancel/timeout/device-selection API.
5. Clarify the Metal backend guide's default-context topology, native waits, identity separation,
   no cross-device semantics, successor boundary, and retained no-general-pool decision.
6. Synchronize this brief, the Metal master, and the roadmap to `Review needed` after execution.

## Validation

Planning commit `e32ad68dbb24b3c9bec9a2bdb09159f0db56ee6a` passed exact three-path scope,
local-link, balanced-fence, final-newline, diff, and clean-commit checks. Execution changed exactly
the nine authorized Markdown paths relative to Task 0048 finalization revision `a4ff40c2`, and
passed local-link, anchor, fence, final-newline, trailing-whitespace, and `git diff --check`
validation. No code, tests, Javadocs, builds, benchmarks, or device probes ran.

Independent Class C review must verify that the final documentation describes only current
behavior, preserves every identity and compatibility boundary listed above, retains Task 0048's
no-general-pool decision, and does not imply asynchronous or multi-device capability.

## Acceptance criteria

1. Public/Runtime successful return is documented as a synchronous completion barrier; failure
   publishes no `RunResult`.
2. `RunResult` is documented as a completed-state resource/publication lease, not a future/event;
   its lifetime and materialization-versus-close behavior remain exact.
3. Concurrent callers retain isolated mutable state without any overlap, ordering, fairness, queue,
   or throughput promise.
4. One integration owns one system-default Metal context/queue; one Engine owns at most one Metal
   integration; resources are context-bound.
5. `BackendDeviceId(metal, "default")`, `TraceDeviceId(0)`, and `SessionNonce` are documented as
   distinct non-physical identities with no stable selector or mapping claim.
6. No enumeration, explicit selection, multi-device scheduling, cross-device migration/coherence,
   fallback, or hot-plug behavior is implied.
7. Concrete successor gates cover asynchronous lifetime/completion and explicit-device/multi-device
   identity/planning/ABI/trace/tuning/coherence requirements.
8. Every API/ABI/version/capability/route/trace/cache/tuning identity and Task 0048's persistent-
   splat/no-general-pool decision remains unchanged.
9. Exactly the nine researched Markdown paths change for Task 0049; Markdown/link/fence/newline/diff
   checks pass; no code, tests, builds, benchmarks, or probes run.
10. Status is `Review needed` after execution and remains so until independent Class C review.

## Result

Executed from planning commit `e32ad68dbb24b3c9bec9a2bdb09159f0db56ee6a`. ADR 0020 and the
Backend, Runtime/Prepare/Engine, public API, Runtime API, and Metal guide contracts now state that
successful execution is a synchronous completion barrier while `RunResult` remains a completed-
state resource/publication lease. They record one system-default-device context/queue per Metal
integration, at most one Metal integration per Engine, exact context-bound resources, and no
overlap/order/fairness, async, selection, multi-device, or cross-device promise.

`BackendDeviceId(metal, "default")`, `TraceDeviceId(0)`, and `SessionNonce` remain distinct non-
physical identities. Every API/ABI/version/capability/route/trace/cache/tuning identity is
unchanged. Immutable source-owned splats still persist per `PreparedExecution`, and no general
mutable output/workspace pool is authorized. Concrete async and explicit-device/multi-device
successor gates are recorded. Status is Review needed pending independent Class C approval.
