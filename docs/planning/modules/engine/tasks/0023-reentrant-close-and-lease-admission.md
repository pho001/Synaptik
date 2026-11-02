# Task 0023: Reentrant close and failure-atomic lease admission

## Status

Complete — exact reviewed nine-file correction is integrated. Actual component/Javadoc and repaired
public Metal integration gates, unchanged real-GPU self-close and bounded-heap OOM consumers passed.
Independent final source/evidence/docs review APPROVE0; whole-release acceptance is separate.

## Class and authority

Class C. Read the root scope index, Core lifecycle, and
[reentrant Engine close and lease admission](../../../../architecture/contracts/runtime-prepare-engine.md#reentrant-engine-close-and-lease-admission).
Runtime remains the sole prepared-resource run-lease owner; Engine only owns its existing outward
operation admission. This corrects shutdown/rollback, not backend numerics or a new execution protocol.

## Evidence and exact behavior

- ReentrantEngineCloseSmoke.java uses the public Metal observer and its owning Engine. The first
  synchronous callback enters Engine.close and hangs. The one before run timed out after15seconds;
  SIGQUIT shows AdvancedEngine.close waiting at1884 underneath the same admitted compute/prepare
  callback. reentrant-engine-close-before.json/log retain command, source SHA and actual thread dump.
- LeaseAllocationSmoke.java uses public Runtime PreparedExecution, an existing nominal persistent
  resource and -Xmx64m. Allocation failed at acquireRunLease:134 after1,816,127 acquired tokens.
  Closing every returned token and the preparation produced physical cleanup count0. No injected
  allocator/fault seam. lease-allocation-before.json/log retain this FAILED actual witness.
- MetalDeviceContext.acquireChildLease has the equivalent increment-before-token allocation gap;
  this one is source-confirmed, not a native-OOM measurement. Do not claim otherwise.

A thread holding an admission for this exact Engine must receive IllegalStateException before
Engine.close changes lifecycle or waits, including when another closer is already waiting. Closing
an unrelated Engine stays legal. Reentrant close by the thread already performing Engine cleanup
is an idempotent no-op; other close callers still wait for completed cleanup and replay its retained
failure. Keep interruption restoration, reverse attempt-all cleanup and exact throwable precedence.

Track admission origin in the existing AdvancedEngine lifecycle owner, replacing the bare aggregate
count rather than adding a second lease protocol or duplicate count. Prefer one reusable unordered
ArrayList of origin Thread references, identity matching and swap-remove; no per-operation token,
boxing, ThreadLocal, registry, callback or new lock. Capacity-allocation failure must publish no
admission. Nested operations and all success/publication/failure exits must balance exactly once.
RepresentativeExecutionSession preserves its existing single-threaded contract and exact admission
ownership through every terminal/preparation handoff and cleanup; no new concurrent-session promise.

Allocate PreparedExecution.RunLease and MetalDeviceContext.ChildLease before incrementing their
existing counts. Keep existing cleanup arbitration; add no factory or production fault-injection seam.
Do not expand into the separately documented closed-borrow retention caveat or native algorithms.

## Source and tests

Existing production owners: AdvancedEngine, RepresentativeExecutionSession, Engine close Javadoc,
Runtime PreparedExecution and MetalDeviceContext. Existing focused lifecycle/representative/native
ownership tests and the public EngineExplicitCompositionMetalIntegrationTest are the test owners.
Discover exact files and reuse their seams. No new production type/package/dependency/public API,
executor, resource lease, numerical profile, native source/schema/identity or cache-key change.

The first actual public observer integration compile exposed an existing visibility defect:
MetalTraceObserver exposes TraceEvent, but Metal's existing Trace dependency was implementation-only.
The independently reviewed ninth-file amendment exports that existing edge as api in
modules/backend/metal/build.gradle.kts. No new dependency or test-only/proxy workaround is added.

Permanent regressions cover consumer-visible reentrant close, nested admission, lifecycle state
preservation, cleanup recursion and existing cross-thread wait/representative completion. They must
be bounded and full-suite-safe, not hang on a regressed implementation. No source/reflection/member/
message/default/count pins or mocked forwarding tests. The actual OOM driver stays throwaway, not a
heap-exhausting permanent test. Preserve all already-qualified publication and caller-storage behavior.

## Dependencies and integration

- Depends on: Engine0022 Complete; existing Runtime0016; exact actual before evidence above.
- Conflicts with: the five production ownership sources and their focused lifecycle tests; shared
  root/scoped/API/planning documentation. No concurrent writer owns these implementation files.
- Parallel group: PRODUCTION-LIFETIME; independent of frozen Model0046 source/guard review.
- Common base revision: HEAD07a281edd3086efde2853211b71b48288db027e1 plus the complete dirty
  engine0023-base.json captured by MAIN after accepted root/owner/Ready updates, not HEAD alone.
- Integration order: accepted authority/Ready and exact freeze -> isolated complete source/tests ->
  independent source/Javadoc review -> exact guarded bytes -> unchanged actual drivers and gates ->
  independent evidence/docs acceptance. Model0046 integration may occur before or after this slice.
- Integration validation: MAIN runs Engine/Runtime/Metal tests and Javadocs, the affected public
  Metal integration class, unchanged actual real-GPU close and bounded-heap OOM consumers, and short
  relevant admission/lifecycle checks. Reuse source-matched installed final38 native bytes because
  native source is unchanged. One final production repository checkpoint joins all work.
- Shared-document integration owner: MAIN. Worker edits source/Javadocs/behavior tests only.

No child builds/tests/lint/formatters/executable checks, commits or pushes. Freeze all changed-file
preimages/final SHA for independent review. MAIN integrates only exact approved bytes.

## Result

MAIN integrated the exact eight reviewed Java files and visibility-only existing Trace API export.
Actual Engine159, Runtime159 and Metal376 passed, with2existing Engine timing and10opt-in Metal
observer skips; Javadocs/jars passed. The first gate remains FAILED because the public observer
lambda could not compile without Trace on its API classpath. After the real export repair, all34
public Metal integration cases passed with no skips:728passed/12skipped across affected modules.

The unchanged real-Metal observer driver now receives IllegalStateException, computes correctly
again and closes externally instead of self-waiting. The unchanged -Xmx64m driver reached actual
token-allocation OOM after1,813,798returned leases; closing them and the preparation produced
physical cleanup count1, rather than the recorded before count0. Native ChildLease OOM safety is
source-reviewed; no native OOM was induced. Native source/library/identity remain unchanged.

engine0023-integrated-files, source-authentication-final, first-failed-gate,
public-integration-final and public-after-observed receipts retain exact sources and actual proof.
Both before failures stay FAILED. Independent final source/evidence/docs acceptance returned APPROVE0.
This does not establish native-memory plateau, performance or whole-release readiness.
