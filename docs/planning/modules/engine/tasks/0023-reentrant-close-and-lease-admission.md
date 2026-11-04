# Task 0023: [Reentrant](../../../../glossary.md#memory-management-atomic-counters-and-complexity) close and failure-[atomic](../../../../glossary.md#resource-transaction-and-rollback) [lease](../../../../glossary.md#term-lease) [admission](../../../../glossary.md#term-admission)

## Status

Complete — exact reviewed nine-file correction is integrated. Actual component/[Javadoc](../../../../glossary.md#term-javadoc) and repaired
public [Metal](../../../../glossary.md#term-metal) integration gates, unchanged real-[GPU](../../../../glossary.md#term-gpu) self-close and bounded-[heap](../../../../glossary.md#term-heap) [OOM](../../../../glossary.md#term-out-of-memory-oom) consumers passed.
Independent final source/evidence/docs review APPROVE0; whole-[release](../../../../glossary.md#term-release) acceptance is separate.

## Class and authority

Class C. Read the root [scope](../../../../glossary.md#term-scope) index, Core [lifecycle](../../../../glossary.md#lifecycle), and
[reentrant Engine close and lease admission](../../../../architecture/contracts/runtime-prepare-engine.md#reentrant-engine-close-and-lease-admission)<span data-glossary-navigation="true"> ([reentrant glossary](../../../../glossary.md#memory-management-atomic-counters-and-complexity))</span>.
[Runtime](../../../../glossary.md#term-runtime) remains the sole prepared-resource [run](../../../../glossary.md#term-run)-[lease](../../../../glossary.md#term-lease) [owner](../../../../glossary.md#term-owner); [Engine](../../../../glossary.md#term-engine) only owns its existing outward
[operation](../../../../glossary.md#operation) [admission](../../../../glossary.md#term-admission). This corrects shutdown/[rollback](../../../../glossary.md#term-rollback), not [backend](../../../../glossary.md#backend) numerics or a new execution protocol.

## Evidence and exact behavior

- ReentrantEngineCloseSmoke.java uses the public Metal observer and its owning Engine. The first
  synchronous callback enters Engine.close and hangs. The one before run timed out after15seconds;
  SIGQUIT shows AdvancedEngine.close waiting at1884 underneath the same admitted compute/prepare
  callback. reentrant-engine-close-before.json/log retain command, source SHA and actual thread dump.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [observer glossary](../../../../glossary.md#term-observer); [synchronous glossary](../../../../glossary.md#term-synchronous); [run glossary](../../../../glossary.md#term-run); [prepare glossary](../../../../glossary.md#prepare); [log glossary](../../../../glossary.md#term-log))</span>
- LeaseAllocationSmoke.java uses public Runtime PreparedExecution, an existing nominal persistent
  resource and -Xmx64m. Allocation failed at acquireRunLease:134 after1,816,127 acquired tokens.
  Closing every returned token and the preparation produced physical cleanup count0. No injected
  allocator/fault seam. lease-allocation-before.json/log retain this FAILED actual witness.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [nominal glossary](../../../../glossary.md#term-nominal))</span>
- MetalDeviceContext.acquireChildLease has the equivalent increment-before-token allocation gap;
  this one is source-confirmed, not a native-OOM measurement. Do not claim otherwise.<span data-glossary-navigation="true"> ([OOM glossary](../../../../glossary.md#term-out-of-memory-oom))</span>

A thread holding an [admission](../../../../glossary.md#term-admission) for this exact [Engine](../../../../glossary.md#term-engine) must receive IllegalStateException before
Engine.close changes [lifecycle](../../../../glossary.md#lifecycle) or waits, including when another closer is already waiting. Closing
an unrelated Engine stays legal. [Reentrant](../../../../glossary.md#memory-management-atomic-counters-and-complexity) close by the thread already performing Engine cleanup
is an [idempotent](../../../../glossary.md#term-idempotent) no-op; other close callers still wait for completed cleanup and replay its retained
failure. Keep interruption restoration, reverse attempt-all cleanup and exact throwable precedence.

Track admission origin in the existing AdvancedEngine lifecycle [owner](../../../../glossary.md#term-owner), replacing the bare aggregate
count rather than adding a second [lease](../../../../glossary.md#term-lease) protocol or duplicate count. Prefer one reusable unordered
ArrayList of origin Thread references, [identity](../../../../glossary.md#term-identity-meanings) matching and swap-remove; no per-[operation](../../../../glossary.md#operation) [token](../../../../glossary.md#term-token),
boxing, ThreadLocal, [registry](../../../../glossary.md#term-registry), [callback](../../../../glossary.md#term-callback) or new lock. Capacity-allocation failure must publish no
admission. Nested operations and all success/[publication](../../../../glossary.md#term-publication)/failure exits must balance exactly once.
RepresentativeExecutionSession preserves its existing single-threaded contract and exact admission
[ownership](../../../../glossary.md#term-ownership-meanings) through every terminal/preparation [handoff](../../../../glossary.md#term-handoff) and cleanup; no new concurrent-session promise.

Allocate [PreparedExecution](../../../../glossary.md#prepared-execution--preparedexecution).RunLease and MetalDeviceContext.ChildLease before incrementing their
existing counts. Keep existing cleanup arbitration; add no factory or production fault-injection seam.
Do not expand into the separately documented closed-borrow retention caveat or native algorithms.

## Source and tests

Existing production [owners](../../../../glossary.md#term-owner): AdvancedEngine, RepresentativeExecutionSession, [Engine](../../../../glossary.md#term-engine) close [Javadoc](../../../../glossary.md#term-javadoc),
[Runtime](../../../../glossary.md#term-runtime) [PreparedExecution](../../../../glossary.md#prepared-execution--preparedexecution) and MetalDeviceContext. Existing focused [lifecycle](../../../../glossary.md#lifecycle)/representative/native
[ownership](../../../../glossary.md#term-ownership-meanings) tests and the public EngineExplicitCompositionMetalIntegrationTest are the test owners.
Discover exact files and reuse their seams. No new production type/package/[dependency](../../../../glossary.md#term-dependency)/public [API](../../../../glossary.md#term-application-programming-interface-api),
executor, [resource lease](../../../../glossary.md#ownership-borrowing-and-lease), numerical profile, native source/[schema](../../../../glossary.md#term-schema)/[identity](../../../../glossary.md#term-identity-meanings) or [cache](../../../../glossary.md#term-cache)-key change.

The first actual public [observer](../../../../glossary.md#term-observer) integration [compile](../../../../glossary.md#compile) exposed an existing visibility defect:
[MetalTraceObserver](../../../../glossary.md#metal-trace-observer--metaltraceobserver) exposes TraceEvent, but [Metal](../../../../glossary.md#term-metal)'s existing [Trace](../../../../glossary.md#trace) dependency was implementation-only.
The independently reviewed ninth-file amendment exports that existing edge as api in
[modules](../../../../glossary.md#term-module)/[backend](../../../../glossary.md#backend)/metal/build.[gradle](../../../../glossary.md#term-gradle).kts. No new dependency or test-only/proxy workaround is added.

Permanent regressions cover consumer-visible [reentrant](../../../../glossary.md#memory-management-atomic-counters-and-complexity) close, nested [admission](../../../../glossary.md#term-admission), lifecycle state
preservation, cleanup recursion and existing cross-thread wait/representative completion. They must
be bounded and full-suite-safe, not hang on a regressed implementation. No source/reflection/member/
message/default/count pins or mocked forwarding tests. The actual [OOM](../../../../glossary.md#term-out-of-memory-oom) driver stays throwaway, not a
[heap](../../../../glossary.md#term-heap)-exhausting permanent test. Preserve all already-qualified [publication](../../../../glossary.md#term-publication) and caller-storage behavior.

## Dependencies and integration

- Depends on: Engine0022 Complete; existing Runtime0016; exact actual before evidence above.
- Conflicts with: the five production ownership sources and their focused lifecycle tests; shared
  root/scoped/API/planning documentation. No concurrent writer owns these implementation files.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Parallel group: PRODUCTION-[LIFETIME](../../../../glossary.md#term-lifetime); independent of frozen Model0046 source/[guard](../../../../glossary.md#term-guard) review.
- Common base revision: HEAD07a281edd3086efde2853211b71b48288db027e1 plus the complete dirty
  engine0023-base.json captured by MAIN after accepted root/owner/Ready updates, not HEAD alone.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Integration order: accepted authority/Ready and exact freeze -> isolated complete source/tests ->
  independent source/Javadoc review -> exact guarded bytes -> unchanged actual drivers and gates ->
  independent evidence/docs acceptance. Model0046 integration may occur before or after this slice.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [slice glossary](../../../../glossary.md#slice))</span>
- Integration validation: MAIN runs Engine/Runtime/Metal tests and Javadocs, the affected public
  Metal integration class, unchanged actual real-GPU close and bounded-heap OOM consumers, and short
  relevant admission/lifecycle checks. Reuse source-matched installed final38 native bytes because
  native source is unchanged. One final production repository checkpoint joins all work.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Engine glossary](../../../../glossary.md#term-engine); [Runtime glossary](../../../../glossary.md#term-runtime); [Metal glossary](../../../../glossary.md#term-metal); [GPU glossary](../../../../glossary.md#term-gpu); [heap glossary](../../../../glossary.md#term-heap); [OOM glossary](../../../../glossary.md#term-out-of-memory-oom); [admission glossary](../../../../glossary.md#term-admission); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN. Worker edits source/Javadocs/behavior tests only.

No child builds/tests/lint/formatters/executable checks, [commits](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) or pushes. Freeze all changed-file
preimages/final SHA for independent review. MAIN integrates only exact approved bytes.

## Result

MAIN integrated the exact eight reviewed Java files and visibility-only existing [Trace](../../../../glossary.md#trace) [API](../../../../glossary.md#term-application-programming-interface-api) export.
Actual Engine159, Runtime159 and Metal376 passed, with2existing [Engine](../../../../glossary.md#term-engine) timing and10opt-in [Metal](../../../../glossary.md#term-metal)
[observer](../../../../glossary.md#term-observer) skips; Javadocs/jars passed. The first gate remains FAILED because the public observer
lambda could not [compile](../../../../glossary.md#compile) without Trace on its API [classpath](../../../../glossary.md#term-classpath). After the real export repair, all34
public Metal integration cases passed with no skips:728passed/12skipped across affected [modules](../../../../glossary.md#term-module).

The unchanged real-Metal observer driver now receives IllegalStateException, computes correctly
again and closes externally instead of self-waiting. The unchanged -Xmx64m driver reached actual
[token](../../../../glossary.md#term-token)-allocation [OOM](../../../../glossary.md#term-out-of-memory-oom) after1,813,798returned [leases](../../../../glossary.md#term-lease); closing them and the preparation produced
physical cleanup count1, rather than the recorded before count0. Native ChildLease OOM safety is
source-reviewed; no native OOM was induced. Native source/library/[identity](../../../../glossary.md#term-identity-meanings) remain unchanged.

engine0023-integrated-files, source-[authentication](../../../../glossary.md#term-authentication)-final, first-failed-gate,
public-integration-final and public-after-observed receipts retain exact sources and actual proof.
Both before failures stay FAILED. Independent final source/evidence/docs acceptance returned APPROVE0.
This does not establish native-memory plateau, performance or whole-[release](../../../../glossary.md#term-release) readiness.
