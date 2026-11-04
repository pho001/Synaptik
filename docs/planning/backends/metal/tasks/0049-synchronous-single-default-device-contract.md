# Task 0049: [Synchronous](../../../../glossary.md#term-synchronous) single-default-[device](../../../../glossary.md#term-device) [Metal](../../../../glossary.md#term-metal) contract

## Reader context

The application calls Engine or Runtime `run(...)`; the Metal executable submits native device work and waits before a successful return. `RunResult` then keeps completed output storage available for publication—it is not a future for unfinished GPU work. A result can therefore still need closing after computation finishes.

One `MetalBackendIntegration` owns its creating context and default-device queue. Two contexts on the same GPU remain different resource owners. The recorded ABI, schema and tuning versions are the task's preserved snapshot, not today's versions; the completion/lifetime distinction is the reason for this decision.

- [Current completion and context topology](../../../../backend-guide/metal-backend.md#completion-and-device-topology) — Connects synchronous return to result lifetime and exact-context authentication.

## Status

Complete

Executed from [planning](../../../../glossary.md#term-planning) [commit](../../../../glossary.md#term-commit) `e32ad68dbb24b3c9bec9a2bdb09159f0db56ee6a` on clean Task 0048
finalization revision `a4ff40c2b4fc328926c15722241f7359744ff4ba`; original execution
`6e95d523d5bfdd01638a9e6c159e35afc438c9eb` and P1 review remediation
`6d4246f723d5a1991b5d4af7ef1b8ebb78bad9f3` passed final cumulative independent Class C review.
The reviewer returned `APPROVE` with zero findings after remediation.

## Change class

Class C. Documentation-only cross-[module](../../../../glossary.md#term-module) execution/[lifetime](../../../../glossary.md#term-lifetime)/[device](../../../../glossary.md#term-device) contract decision. Independent
Class C review is mandatory before Complete.

## Intent

Codify the behavior the current implementation already provides:

- public and [Runtime](../../../../glossary.md#term-runtime) execution is a [synchronous](../../../../glossary.md#term-synchronous) completion barrier;
- a successful [`RunResult`](../../../../glossary.md#run-result--runresult) is a completed-state resource/publication lease, not a future or device
  completion primitive;<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [lease glossary](../../../../glossary.md#term-lease))</span>
- each `MetalBackendIntegration` owns one Metal context created from the system-default device and
  offers no public device selection or multi-device scheduling surface;<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>
- concurrent callers are supported through isolated mutable run state, but this does not promise
  device overlap, submission or completion order, fairness, queue topology, or throughput; and<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [run state glossary](../../../../glossary.md#run-state--runstate); [topology glossary](../../../../glossary.md#term-topology); [throughput glossary](../../../../glossary.md#term-throughput))</span>
- [immutable](../../../../glossary.md#term-immutable) source-owned splats may persist per [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), while no general mutable
  output/workspace pool is authorized.

This task records existing truth only. It does not add an [asynchronous](../../../../glossary.md#term-asynchronous) [API](../../../../glossary.md#term-application-programming-interface-api), futures, events,
[cancellation](../../../../glossary.md#term-cancellation), timeout, [device](../../../../glossary.md#term-device) enumeration, explicit device selection, multi-device scheduling,
cross-device migration/coherence, or pooling.

## Decision boundary

### [Synchronous](../../../../glossary.md#term-synchronous) completion barrier

A successful public or [Runtime](../../../../glossary.md#term-runtime) `run(...)` return means every scheduled action and native [operation](../../../../glossary.md#operation)
for that invocation completed and its published destinations are usable. Native [Metal](../../../../glossary.md#term-metal) execution
[commits](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) and waits for completion before returning. [Materialization](../../../../glossary.md#materialization) and transfers on the execution
path are also synchronous. A failed execution publishes no [`RunResult`](../../../../glossary.md#run-result--runresult).

The returned `RunResult` owns completed [run](../../../../glossary.md#term-run)-state resources and the [publication](../../../../glossary.md#term-publication) [lifetime](../../../../glossary.md#term-lifetime) of outputs.
It may outlive the call. It is not a promise, future, event, fence, [command buffer](../../../../glossary.md#term-command-buffer), or handle to
in-flight work. Runtime `RunResult` and [`RunState`](../../../../glossary.md#run-state--runstate) remain non-[thread-safe](../../../../glossary.md#term-thread-safe); the outward [Engine](../../../../glossary.md#term-engine) result
retains its existing synchronized materialization-versus-close behavior.

Prepared-execution acquisition [leases](../../../../glossary.md#term-lease) exactly one synchronous run. Close rejects new acquisitions,
does not wait, and defers owned-resource cleanup only through already-active execution/result
leases. The prepared [owner](../../../../glossary.md#term-owner) must remain alive through native completion; the current synchronous
barrier establishes that point before the result is returned.

Concurrent calls may execute from distinct caller threads and use isolated [mutable](../../../../glossary.md#term-mutable) [run state](../../../../glossary.md#run-state--runstate). This
is not a promise of [GPU](../../../../glossary.md#term-gpu) overlap, command ordering between runs, fairness, queue multiplicity,
submission policy, completion ordering, [throughput](../../../../glossary.md#term-throughput), or scheduler behavior. Schedule order remains
the order within one invocation.

There is no [asynchronous](../../../../glossary.md#term-asynchronous) run, [cancellation](../../../../glossary.md#term-cancellation), timeout, polling, [callback](../../../../glossary.md#term-callback), completion-stage, explicit
event, or deferred-error [API](../../../../glossary.md#term-application-programming-interface-api).

### One system-default [device context](../../../../glossary.md#device-submission-synchronization-and-execution-context) per integration

One `MetalBackendIntegration` creates and owns one native [Metal](../../../../glossary.md#term-metal) device context by calling the system
default-[device](../../../../glossary.md#term-device) path and creating its [command queue](../../../../glossary.md#term-command-queue). Configuration selects only the [native library](../../../../glossary.md#term-native-library);
it exposes no physical-device selector. The [runtime](../../../../glossary.md#term-runtime) availability [snapshot](../../../../glossary.md#term-snapshot) presents the abstract
`BackendDeviceId(metal, "default")` slot.

One [`Engine`](../../../../glossary.md#term-engine) may register at most one integration for the Metal [`BackendId`](../../../../glossary.md#backend-identity--backendid); duplicate [backend](../../../../glossary.md#backend)
registration is rejected. [Planning](../../../../glossary.md#term-planning) uses device requirements only to establish eligibility and
retains the owning `BackendId`, not a selected physical Metal device. Resources remain authenticated
and bound to their exact `MetalDeviceContext`; [native code](../../../../glossary.md#native-numerical-library-and-instruction-architecture) rejects resources from another
`MTLDevice`.

Separate Engines or integrations may create separate contexts, but this is not coordinated
multi-device execution and they may resolve to the same system-default physical device. The current
contract provides no device enumeration, stable hardware selector, explicit selection, load
balancing, [topology](../../../../glossary.md#term-topology), failover, hot-plug refresh, cross-device transfer, migration, replication,
coherence, partitioning, or scheduling.

### [Identity](../../../../glossary.md#term-identity-meanings) separation

The following identities are deliberately distinct and none is a stable physical-[device](../../../../glossary.md#term-device) identity:

- `BackendDeviceId(metal, "default")` is the sole abstract availability slot;
- `TraceDeviceId(0)` is the fixed [Metal](../../../../glossary.md#term-metal) [trace](../../../../glossary.md#trace) device [token](../../../../glossary.md#term-token) used with stream-local correlation; and
- the tuning `SessionNonce` distinguishes a process/session compatibility context and explicitly
  makes no stable device claim.

The task changes none of these identities and introduces no mapping among them. Tuning remains
version 12, [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4, session-scoped, and non-persistent. There is no persistent route or tuning [cache](../../../../glossary.md#term-cache)
whose reuse could imply a physical-device [fingerprint](../../../../glossary.md#term-fingerprint).

### Resource [ownership](../../../../glossary.md#term-ownership) and no-general-pool closure

[Immutable](../../../../glossary.md#term-immutable) source-owned splat [buffers](../../../../glossary.md#term-buffer) continue to persist exactly within their owning
[`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) and supply fresh read-only [run](../../../../glossary.md#term-run) [bindings](../../../../glossary.md#term-binding). [Mutable](../../../../glossary.md#term-mutable) outputs and address workspaces
remain per-run resources. [Synchronous](../../../../glossary.md#term-synchronous) [device](../../../../glossary.md#term-device) completion alone does not end their [ownership](../../../../glossary.md#term-ownership-meanings): a
[`RunResult`](../../../../glossary.md#run-result--runresult) can retain output resources after `run(...)` returns, and address workspaces carry
per-run mutable pointer arrays. This decision therefore retains Task 0048's rejection of a general
output/workspace pool.

## Compatibility invariants

This task preserves every existing:

- public Java [API](../../../../glossary.md#term-application-programming-interface-api) and [module](../../../../glossary.md#term-module) boundary;
- [Runtime](../../../../glossary.md#term-runtime)/[Backend](../../../../glossary.md#backend) [SPI](../../../../glossary.md#term-service-provider-interface-spi) and behavior;
- native [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4 version, thirteen-export set, status space, [node](../../../../glossary.md#node) [schema](../../../../glossary.md#term-schema) 11, and [ownership](../../../../glossary.md#term-ownership-meanings) rule;
- capability [matrix](../../../../glossary.md#term-review-matrix), [occurrence](../../../../glossary.md#term-occurrence) support, route selection, fallback, and eligibility rule;
- schedule [topology](../../../../glossary.md#term-topology) and action order;
- [trace](../../../../glossary.md#trace) event kind, phase, backend/[device](../../../../glossary.md#term-device) [token](../../../../glossary.md#term-token), correlation, and failure behavior;
- workload, exact-policy, candidate, compatibility, route, and [codec](../../../../glossary.md#term-codec) [identity](../../../../glossary.md#term-identity-meanings) at version 12;
- tuning session [scope](../../../../glossary.md#term-scope), `SessionNonce`, [cache](../../../../glossary.md#term-cache) behavior, and absence of [persistence](../../../../glossary.md#term-persistence);
- resource, prepared-execution, result, transfer, [materialization](../../../../glossary.md#materialization), and close [lifetime](../../../../glossary.md#term-lifetime); and
- Task 0048 [immutable](../../../../glossary.md#term-immutable)-splat optimization and no-general-pool decision.

No version bump, capability increase, route change, [serialization](../../../../glossary.md#term-serialization), cache, benchmark, retry,
fallback, packaging, discovery, or performance claim is authorized.

## Concrete successor triggers

### [Asynchronous](../../../../glossary.md#term-asynchronous) execution

A successor may be planned only after a concrete end-to-end consumer need and evidence that the
[synchronous](../../../../glossary.md#term-synchronous) barrier is the limiting constraint. It must be a separately reviewed cross-[module](../../../../glossary.md#term-module) Class
C decision that defines, before implementation:

1. completion and failure delivery, including when native asynchronous errors become observable;
2. [cancellation](../../../../glossary.md#term-cancellation) and timeout semantics and their relationship to already-submitted [device](../../../../glossary.md#term-device) work;
3. input [borrowing](../../../../glossary.md#term-borrowing) duration and exact output/workspace [ownership](../../../../glossary.md#term-ownership-meanings) before and after completion;
4. prepared-execution acquisition lifetime through terminal device completion, including close
   races and cleanup-failure ownership;<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime))</span>
5. dependency representation across Metal work, CPU work, transfers, materialization, and public
   publication;<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Metal glossary](../../../../glossary.md#term-metal); [CPU glossary](../../../../glossary.md#term-cpu); [materialization glossary](../../../../glossary.md#materialization); [publication glossary](../../../../glossary.md#term-publication))</span>
6. trace truth for submission versus terminal outcome and representative tuning timing/identity;
   and<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
7. only after those rules, any pool return/reset, capacity, eviction, and context-close protocol.

An asynchronous successor must review all affected public [API](../../../../glossary.md#term-application-programming-interface-api), [Runtime](../../../../glossary.md#term-runtime)/[Backend](../../../../glossary.md#backend) [SPI](../../../../glossary.md#term-service-provider-interface-spi), native [ABI](../../../../glossary.md#term-application-binary-interface-abi),
version/capability, route, [trace](../../../../glossary.md#trace), [cache](../../../../glossary.md#term-cache), tuning, and ownership contracts rather than attaching a
future to the current synchronous method.

### Explicit-[device](../../../../glossary.md#term-device) or multi-device execution

A successor may be planned only for a real supported multi-device environment and concrete user
workflow. The smallest successor should first consider selecting exactly one device per integration
or [Engine](../../../../glossary.md#term-engine) rather than introducing a scheduler. It must define and prove:

1. a stable selector and truthful enumeration/availability/failure behavior;
2. any native enumeration/context-create [ABI](../../../../glossary.md#term-application-binary-interface-abi) evolution and export/version consequences;
3. a registry identity that can represent more than one integration for a backend if true
   multi-device ownership is required;<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [identity glossary](../../../../glossary.md#term-identity-meanings); [backend glossary](../../../../glossary.md#backend); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
4. planner retention of the selected [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid), device-qualified partitions/resources,
   and resource-context authentication;<span data-glossary-navigation="true"> ([authentication glossary](../../../../glossary.md#term-authentication))</span>
5. cross-device transfer, migration, replication, coherence, topology, fallback, hot-plug, and
   scheduling/lifetime rules required by the chosen scope;<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [lifetime glossary](../../../../glossary.md#term-lifetime); [scope glossary](../../../../glossary.md#term-scope))</span>
6. trace mapping to the selected device and tuning compatibility keyed by truthful selected-device
   and session information; and<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace))</span>
7. a stable physical-device [fingerprint](../../../../glossary.md#term-fingerprint) before any persistent tuning/[cache](../../../../glossary.md#term-cache) reuse is claimed.

The successor must preserve [fail-closed behavior](../../../../glossary.md#admission-preflight-and-fail-closed-validation) until those contracts and real-device evidence are
available. Multiple independent default-device contexts are not evidence of multi-device support.

## Dependencies and integration

- Depends on: Task 0048 Complete at independently approved implementation `89f9fbb9` plus
  documentation finalization `a4ff40c2`; Runtime 0016; Prepare 0006; Engine 0010; Planning's current
  device eligibility model; Trace 0003; Metal 0041–0042<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [device glossary](../../../../glossary.md#term-device); [Trace glossary](../../../../glossary.md#trace); [Metal glossary](../../../../glossary.md#term-metal))</span>
- Conflicts with: every concurrent edit to the original nine execution files below or the
  review-remediation Trace master path, and any asynchronous execution, result/resource lifetime,
  Metal device/context, planner device-selection, native ABI, trace, cache, tuning, route,
  capability, or pooling decision<span data-glossary-navigation="true"> ([asynchronous glossary](../../../../glossary.md#term-asynchronous); [resource lifetime glossary](../../../../glossary.md#ownership-borrowing-and-lease); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [cache glossary](../../../../glossary.md#term-cache))</span>
- Parallel group: None
- Common base revision: `a4ff40c2b4fc328926c15722241f7359744ff4ba`
- Integration order: [planning](../../../../glossary.md#term-planning) `e32ad68d`; execution `6e95d523`; review-remediation `6d4246f7`;
  final cumulative independent Class C `APPROVE` with zero findings
- Integration validation: the original exact nine-path execution passed local-link/anchor, fence,
  final-newline, trailing-whitespace, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`git diff --check`; the review remediation passed its exact
  two-document commit, cumulative ten-path scope, planning-wide status consistency, and same
  Markdown/diff checks; final review approved; no code, tests, Javadocs, builds, benchmarks, or
  probes ran<span data-glossary-navigation="true"> ([commit glossary](../../../../glossary.md#term-commit); [scope glossary](../../../../glossary.md#term-scope))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): Task 0049 implementation owner

## Original execution file [scope](../../../../glossary.md#term-scope) and review remediation

The original Task 0049 execution changed exactly these nine researched Markdown paths:

1. `docs/planning/backends/metal/tasks/0049-synchronous-single-default-device-contract.md`
2. `docs/planning/backends/metal/master-plan.md`
3. `docs/planning/roadmap.md`
4. `docs/design/decisions/0020-synchronous-single-default-device-metal-execution.md`
5. `docs/architecture/contracts/backend-execution.md`
6. `docs/architecture/contracts/runtime-prepare-engine.md`
7. `docs/api/public-api.md`
8. `docs/api/runtime-api.md`
9. `docs/backend-guide/metal-backend.md`

Final cumulative Class C review authorized one P1 documentation-only remediation outside that
original execution scope:

10. `docs/planning/modules/trace/master-plan.md`

The cumulative Task 0049 change set is therefore ten Markdown paths relative to Task 0048
finalization revision `a4ff40c2`; this remediation [commit](../../../../glossary.md#term-commit) changes only the Task 0049 brief and the
[Trace](../../../../glossary.md#trace) master plan.

No Java, [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary), [Gradle](../../../../glossary.md#term-gradle), workflow, test, benchmark, generated, native, package, [cache](../../../../glossary.md#term-cache), or probe
path is in scope.

## Documentation execution plan

1. Add [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0020 as Accepted, recording the [synchronous](../../../../glossary.md#term-synchronous) completion barrier, completed-state
   [`RunResult`](../../../../glossary.md#run-result--runresult) lease, one system-default context per integration, identity separation,
   context-bound resources, no multi-device behavior, successor triggers, and rejected pool.<span data-glossary-navigation="true"> ([lease glossary](../../../../glossary.md#term-lease); [identity glossary](../../../../glossary.md#term-identity-meanings); [device glossary](../../../../glossary.md#term-device))</span>
2. Align Backend execution architecture with synchronous completion and context-bound resource
   ownership without inventing a public completion primitive or physical-device identity.<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [ownership glossary](../../../../glossary.md#term-ownership))</span>
3. Align Runtime/Prepare/Engine architecture with one synchronous lease, publication lifetime,
   close races, concurrent-caller non-guarantees, and one Metal integration per Engine.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [publication glossary](../../../../glossary.md#term-publication); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
4. Clarify public and Runtime API references: successful return is completed state; the result is a
   resource/publication lease; there is no async/cancel/timeout/device-selection API.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
5. Clarify the Metal backend guide's default-context topology, native waits, identity separation,
   no cross-device semantics, successor boundary, and retained no-general-pool decision.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology))</span>
6. Synchronize this brief, the [Metal](../../../../glossary.md#term-metal) master, and the roadmap to `Review needed` after execution.

## [Validation](../../../../glossary.md#term-validation)

[Planning](../../../../glossary.md#term-planning) [commit](../../../../glossary.md#term-commit) `e32ad68dbb24b3c9bec9a2bdb09159f0db56ee6a` passed exact three-path [scope](../../../../glossary.md#term-scope),
local-link, balanced-fence, final-newline, diff, and clean-commit checks. Original execution commit
`6e95d523d5bfdd01638a9e6c159e35afc438c9eb` changed exactly the nine researched Markdown paths and
passed local-link, anchor, fence, final-newline, trailing-whitespace, and `git diff --check`.

Final cumulative Class C review found one P1 status drift in
`docs/planning/modules/trace/master-plan.md`: it still named completed [Metal](../../../../glossary.md#term-metal) 0041 as the sole [Ready](../../../../glossary.md#term-ready)
repository frontier. Remediation `6d4246f723d5a1991b5d4af7ef1b8ebb78bad9f3` changed only that
[Trace](../../../../glossary.md#trace) master and this brief, recorded Metal 0041 Complete with no Ready repository frontier, and
made the cumulative Task 0049 scope ten Markdown paths relative to Task 0048 finalization
`a4ff40c2`.

Both remediation documents, planning-wide Ready/status consistency, cumulative scope, links,
anchors, fences, final newlines, trailing whitespace, `git diff --check`, and clean commit passed.
No code, tests, Javadocs, builds, benchmarks, or [device](../../../../glossary.md#term-device) probes ran. The same final independent
Class C reviewer then returned `APPROVE` with zero findings.

## Acceptance criteria

1. Public/Runtime successful return is documented as a synchronous completion barrier; failure
   publishes no <span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [synchronous glossary](../../../../glossary.md#term-synchronous))</span>[`RunResult`](../../../../glossary.md#run-result--runresult).
2. `RunResult` is documented as a completed-state resource/publication lease, not a future/event;
   its lifetime and materialization-versus-close behavior remain exact.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [lease glossary](../../../../glossary.md#term-lease); [lifetime glossary](../../../../glossary.md#term-lifetime); [materialization glossary](../../../../glossary.md#materialization))</span>
3. Concurrent callers retain isolated mutable state without any overlap, ordering, fairness, queue,
   or throughput promise.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [throughput glossary](../../../../glossary.md#term-throughput))</span>
4. One integration owns one system-default Metal context/queue; one Engine owns at most one Metal
   integration; resources are context-bound.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Engine glossary](../../../../glossary.md#term-engine))</span>
5. `BackendDeviceId(metal, "default")`, `TraceDeviceId(0)`, and `SessionNonce` are documented as
   distinct non-physical identities with no stable selector or mapping claim.
6. No enumeration, explicit selection, multi-device scheduling, cross-device migration/coherence,
   fallback, or hot-plug behavior is implied.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>
7. Concrete successor gates cover asynchronous lifetime/completion and explicit-device/multi-device
   identity/planning/ABI/trace/tuning/coherence requirements.<span data-glossary-navigation="true"> ([asynchronous glossary](../../../../glossary.md#term-asynchronous); [identity glossary](../../../../glossary.md#term-identity-meanings); [planning glossary](../../../../glossary.md#term-planning); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [trace glossary](../../../../glossary.md#trace))</span>
8. Every API/ABI/version/capability/route/trace/cache/tuning identity and Task 0048's persistent-
   splat/no-general-pool decision remains unchanged.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [cache glossary](../../../../glossary.md#term-cache))</span>
9. The original execution changes exactly nine researched Markdown paths; the sole review
   remediation adds the Trace master as the cumulative tenth path and changes only it plus this
   brief. Markdown/link/anchor/fence/newline/whitespace/diff checks pass; no code, tests, builds,
   benchmarks, or probes run.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
10. Status becomes `Complete` only after final cumulative independent Class C approval.

## Result

Executed from [planning](../../../../glossary.md#term-planning) [commit](../../../../glossary.md#term-commit) `e32ad68dbb24b3c9bec9a2bdb09159f0db56ee6a`. [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0020 and the
[Backend](../../../../glossary.md#backend), [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine), public [API](../../../../glossary.md#term-application-programming-interface-api), Runtime API, and [Metal](../../../../glossary.md#term-metal) guide contracts now state that
successful execution is a [synchronous](../../../../glossary.md#term-synchronous) completion barrier while [`RunResult`](../../../../glossary.md#run-result--runresult) remains a completed-
state resource/[publication](../../../../glossary.md#term-publication) [lease](../../../../glossary.md#term-lease). They record one system-default-[device context](../../../../glossary.md#device-submission-synchronization-and-execution-context)/queue per Metal
integration, at most one Metal integration per Engine, exact context-bound resources, and no
overlap/order/fairness, async, selection, multi-[device](../../../../glossary.md#term-device), or cross-device promise.

`BackendDeviceId(metal, "default")`, `TraceDeviceId(0)`, and `SessionNonce` remain distinct non-
physical [identities](../../../../glossary.md#term-identity-meanings). Every API/[ABI](../../../../glossary.md#term-application-binary-interface-abi)/version/capability/route/[trace](../../../../glossary.md#trace)/[cache](../../../../glossary.md#term-cache)/tuning [identity](../../../../glossary.md#term-identity-meanings) is
unchanged. [Immutable](../../../../glossary.md#term-immutable) source-owned splats still persist per [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), and no general
[mutable](../../../../glossary.md#term-mutable) output/workspace pool is authorized. Concrete async and explicit-device/multi-device
successor gates are recorded.

Final cumulative Class C review found one P1 documentation drift: the Trace master still called
completed Metal 0041 the sole [Ready](../../../../glossary.md#term-ready) repository frontier. Remediation `6d4246f7` records Metal 0041
as Complete and no repository task as Ready. The original execution remains the exact nine-path
commit `6e95d523`; cumulative Task 0049 [scope](../../../../glossary.md#term-scope) is ten Markdown paths, with only the Trace master and
this brief changed for remediation. The same reviewer returned `APPROVE` with zero findings after
remediation; Task 0049 is Complete.

## Post-completion program verification

[Task 0050](0050-final-packaged-metal-repository-verification.md) performed the separately reserved
program verification from exact Task 0049 completion revision
`2c6326a9663bea5aa99e407351d685919630bbcb`. It built, fixed-identifier ad-hoc-signed, packaged,
archived, extracted, and independently reverified the current [Metal](../../../../glossary.md#term-metal) [dylib](../../../../glossary.md#term-dylib), then ran the sole final
full `./gradlew build --rerun-tasks` against the extracted absolute dylib. All 87 actionable tasks
executed successfully. [JUnit](../../../../glossary.md#term-junit) XML recorded 130 Metal-[backend](../../../../glossary.md#backend) tests across 19 suites and 15 explicit
Metal integration tests, including two real-Metal public `prepareTuned` cases, with zero failures,
errors, or skips. This evidence changes none of Task 0049's documentation-only contract or [scope](../../../../glossary.md#term-scope).
