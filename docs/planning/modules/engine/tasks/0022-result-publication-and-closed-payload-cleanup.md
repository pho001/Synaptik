# Task 0022: Result-publication rollback and closed-payload cleanup

## Status

Complete — independent design and final source/Javadoc/test review APPROVE, zero P1/P2.
MAIN accepted the exact frozen-before/passing-after regression, bounded public F32/F64 heap
comparison, affected gate and actual public lifetime/precedence/race scenarios below.

## Change class

Class C — cross Engine/Runtime cleanup, Error rollback and retained caller-payload lifetimes.
Clean implementation and independent source/Javadoc review; meaningful before/after regression and
actual public CPU smokes. This repairs existing ownership contracts, not a new architecture/API.

## Observations before repair

1. AdvancedEngine.run protects runner execution but allocates/registers AdvancedRunResult outside
   that rollback region. Constructor/ArrayList.add failure leaves the inward result unclosed and
   activeOperations unreleased; Engine.close can wait forever. Ordinary finishOrdinaryRun already
   protects the equivalent publication. Allocation failure is a real possibility, but no safe
   deterministic public allocation-failure switch exists; do not exhaust the coordinator heap.
2. Retained closed RunResult.Publication keeps AdvancedRunResult.ownedBorrowedInputs, inward result
   representations and RunState buffer/workspace references reachable. Metadata-after-close is
   supported; fresh per-run heap payload retention is not required for it. Native close calls do
   happen: this is not a demonstrated native-arena release failure.

Actual source-matched public diagnostic: storage-free neg prototype, fresh heap input each call,
materialize/check, close result, retain only Publication. Separate -Xmx96m JVMs: FLOAT32/FLOAT64
release-metadata controls each complete 128 numeric runs; closed metadata exhausts heap after 42
FLOAT32 runs and 27 FLOAT64 runs. Source chains establish the cause; no GC timing assertion or
production instrumentation. The unchanged before/after source hash and logs are retained in
coordinator evidence; the temporary diagnostic source is removed after acceptance.

## Bounded implementation

Extend the existing advanced publication transaction through wrapper creation and registration.
On every unchecked failure, close whichever inward/wrapper owner actually exists, preserve primary
and distinct reverse-cleanup suppression, and release Engine admission exactly once even when
rollback itself fails. Reuse existing closeWithSuppression/finishFailure conventions; no generic
publication framework, callback factory, new owner/lease/lock or normal-path allocation.
Close the constructed wrapper or the inward result, never both; rollback is outside lifecycleLock.
Release admission in a finally protecting rollback, preserving current argument-validation order.

After close begins, clear direct per-run payload references while retaining immutable metadata,
counts and plans. AdvancedRunResult transfers its borrow-list to the first cleanup's local scope
and drops the field reference, while still closing wrappers reverse-order exactly once. Runtime
RunResult releases publication representation references even if state cleanup fails. RunState
releases each borrowed/owned buffer binding and workspace reference after its required cleanup;
borrowed caller storage is never closed. Keep reverse encounter order, attempt-all cleanup,
unique identity rules, preserved count/plan inspection and current failure replay semantics.
Clear reference cells, not inner-array dimensions: bufferRepresentationCount remains inspectable.
Publication aliases do not add cleanup occurrences. Runtime repeated close does not replay a prior
failure; outward AdvancedRunResult retains exact replay. Keep that distinction.

Runtime RunState/RunResult remain non-thread-safe by contract. Outward serialized materialize/
close and close replay/wait/interruption behavior remain unchanged. Do not weaken current access-
after-close guards, add second leases, close caller arenas, collect metadata early, or retain
payload to make inspection tests pass. Do not change CpuBorrowedBuffer's caller-owned no-op close.
Shared prepared recipes may remain; the fix releases fresh per-run ownership references, not
immutable metadata or required retained cleanup throwables.

## Contracts

- [Root lifecycle/invariants](../../../../../ARCHITECTURE.md#core-lifecycle).
- [Runtime ownership](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle),
  [Engine](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine): sole Runtime
  leases, completed synchronous results, caller borrowing, metadata and attempt-all rollback.
- MAIN accepted the exact existing-contract clarification in root/owning Run/Engine lifecycle.
  No dependency, application API, native/generated ABI, algorithm or numerical policy changes.
- [Production resource gate](../../../production-release-goal.md#release-acceptance-checklist).

## Files and ownership

Production: existing modules/engine AdvancedEngine.run/ordinary publication conventions,
AdvancedRunResult.close and RunResult Javadoc; existing modules/runtime run.RunState.close and
run.RunResult.close. No new/moved production type/package or changed public signature.

Tests: existing AdvancedEngineLifecycleTest/EngineTypedLifecycleTest publication-failure helpers,
Runtime RunStateTest/RunResultTest cleanup/access/count cases, and a focused real-CPU integration
regression for retaining closed metadata under a bounded child heap. Reuse PlainJvmCpuIntegrationTest's
fresh-JVM classpath/environment pattern rather than inventing another launcher framework.
MAIN owns shared architecture/plans/API guide/changelog/evidence. CPU0078 source is separate.

## Dependencies and integration

- Depends on: source audit and actual retention before evidence; reviewed exact repair and authority.
- Conflicts with: advanced Engine result/publication/close, Runtime result/state cleanup and their
  focused tests/docs. No CPU provider/generator/finalizer/ledger changes.
- Parallel group: RELEASE-CPU78-LIFECYCLE; disjoint files and semantic ownership from CPU0078.
- Common base revision: HEAD 07a281edd3086efde2853211b71b48288db027e1 plus cpu0078-base.json
  production source and authority-only Engine0022 addendum; exact engine0022-base.json/worktree
  recorded by MAIN before launch. CPU0078 receives the same additive unchanged-meaning authority.
- Integration order: test-first publication witness -> MAIN before gate -> existing-owner fixes ->
  independent review -> ordered MAIN integration/gates and unchanged actual bounded-heap after.
- Integration validation: Engine/Runtime/Javadocs, affected Prepare and actual Engine CPU integration,
  actual F32/F64 lifetime/precedence/race and bounded metadata heap scenarios; final release join later.
- Shared-document integration owner: MAIN.

## Acceptance and validation

- Deterministic injected publication failure proves result cleanup, original/suppressed errors and
  admitted-operation release/terminating Engine.close; it tests behavior, not private field counts.
  Reuse the existing rejecting-publication-list test convention. First add/freeze the regression,
  notify MAIN, and wait for MAIN's failing-before evidence before production edits. Do not add a
  production fault/backdoor flag or use a success-only actual run as rollback proof.
  Inject Error; the failing-before test must not unconditionally close its stranded baseline Engine
  in teardown and hang. Wrapper-allocation failure remains source-inspected, not a fake factory.
- Unchanged actual bounded child program completes 128 retained-metadata runs in F32 and F64 after
  the fix, with correct numeric outputs and readable metadata. Add an isolated full-suite-safe
  behavior regression; no flaky System.gc/WeakReference deadline or bare not-throw assertion.
  Cover reference release when cleanup throws as well as normal close, without asserting private
  field layout or using an admission/GC timing backdoor.
- Open results remain independently readable after prepared close; detached host values survive
  caller/Engine closure; caller storage remains owned/live until the caller closes it.
- Preserve invalid-input/closed/foreign/null precedence, parallel run/copy-close isolation, failure/
  retry, reverse/duplicate cleanup and exact failure replay. Independent actual CPU stress plus
  deterministic focused tests are complementary, not a claim that public barriers prove lease order.
- No new instrumentation, native-byte counter or throughput/zero-growth claim from isClosed alone.
  Record the heap-pressure result separately from native release. No historical failure relabeling.

No builds/tests/lint/formatters in the worker. MAIN runs the new publication test before source
fixes, then final gates once reviewed sources stabilize. Source and Javadoc review before acceptance.

MAIN Ready verification: complete Engine0021/Runtime0016 baseline, audited source, both-type
actual retention before and no-metadata controls, independent exact design and coordinated authority
verified. Existing-contract repair needs no new ADR/type/dependency/protocol. Parallel CPU0078
scope is disjoint; MAIN integrates test-first witness then reviewed lifecycle source and final gates.

## Result

Implemented in the existing Engine/Runtime owners; ten approved source/test files integrated
against the frozen base, with the test-first publication witness unchanged.

- Advanced wrapper construction and registration now share execution's rollback boundary.
  Cleanup runs outside lifecycleLock, closes the wrapper or inward result once, and releases
  admission in finally even on cleanup failure. The injected-Error regression genuinely FAILED
  before source repair (missing suppressed cleanup), then passed unchanged. Wrapper-allocation
  failure coverage remains source inspection, not a fake allocation hook.
- Closed AdvancedRunResult, Runtime RunResult and RunState release their direct fresh-payload
  references even on cleanup failure. Dimensions, counts, plan/geometry, borrowed caller ownership,
  exact outward failure replay and Runtime non-replay remain unchanged.
- Unchanged public `ClosedMetadataHeapSmoke` now completes 128 retained-publication runs per
  dtype under -Xmx96m, with numeric checks. The earlier FLOAT32/FLOAT64 OOM at 42/27 runs remains
  recorded as FAILED-before; both release-metadata controls completed 128. Permanent isolated
  real-CPU child regressions and cleanup-failure pressure coverage passed.
- Actual public F32/F64 scenarios passed independent results, invalid-shape/dead-input retry,
  bounded-copy and invalid-softmax retry, null/foreign/closed precedence, prepared close with
  live results, caller-arena ownership and detached bytes after caller/Engine close. Each type
  also passed 16 prepared-close/run, 16 copy/close and 16 paired concurrent-run stress groups.
  These public barriers are not deterministic internal lease-order proof.

MAIN's source-matched gate completed successfully in 9s:

```text
:modules:engine:test :modules:runtime:test :modules:prepare:test
:testing:integration-tests:test
  --tests '*ClosedPublicationHeapIntegrationTest'
  --tests '*EngineCpuSoftmaxBackwardIntegrationTest'
:modules:engine:javadoc :modules:runtime:javadoc
:modules:engine:jar :modules:runtime:jar
```

| Project | Passed | Skipped |
|---|---:|---:|
| Engine | 154 | 2 |
| Runtime | 159 | 0 |
| Prepare | 98 | 0 |
| Selected real-CPU integration | 6 | 0 |
| Total | 417 | 2 |

No failures/errors. The two skipped Engine timing methods require
`synaptik.cpu.dynamicBatch.performance=true`; no timing campaign was run for this repair.
The initial separate public-lifetime fixture failed by closing its Arena twice; its corrected
single-owner fixture passed without a product change, and that first failure is retained honestly.

Evidence: frozen integration SHA manifest, before/after JUnit XML, gate counts, heap logs and
public-lifecycle source hash/log are in the coordinator's source-snapshot evidence. Native-memory
plateau, arbitrary retained cleanup-throwable graphs, a full-root PASS and complete production
release readiness are not claimed.
