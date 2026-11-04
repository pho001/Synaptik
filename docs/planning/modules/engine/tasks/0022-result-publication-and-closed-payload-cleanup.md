# Task 0022: Result-[publication](../../../../glossary.md#term-publication) [rollback](../../../../glossary.md#term-rollback) and closed-[payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) cleanup

> **Editorial historical reading key.**
>
> Read this as a historical repair of two distinct boundaries: AdvancedEngine publishing a completed
> inward Runtime result as an outward wrapper, and closed result/state owners releasing fresh
> invocation references while preserving metadata. Engine admission must be released even when
> publication rollback fails; cleanup closes the wrapper or inward result, not both. Runtime remains
> the lease authority, while outward Engine wrappers retain their documented synchronization and
> failure-replay behavior.
>
> The [current Engine
> boundary](../../../../reference/runtime/engine-boundary.md#current-ordinary-engine-boundary)
> describes publication rollback and closed-payload release. The [Runtime cleanup
> contract](../../../../reference/runtime/creation.md#run-state-lifecycle-and-failures) distinguishes
> run-owned cleanup from borrowed inputs: releasing references does not close caller storage. The
> recorded bounded-heap comparisons concern retained fresh payloads, not demonstrated native-arena
> leakage or a native-memory plateau; the archived measurements and their limits remain historical
> evidence.

## Status

Complete — independent design and final source/[Javadoc](../../../../glossary.md#term-javadoc)/test review APPROVE, zero P1/P2.
MAIN accepted the exact frozen-before/passing-after [regression](../../../../glossary.md#term-regression), bounded public [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) [heap](../../../../glossary.md#term-heap)
comparison, affected gate and actual public [lifetime](../../../../glossary.md#term-lifetime)/precedence/[race](../../../../glossary.md#term-race) scenarios below.

## Change class

Class C — cross [Engine](../../../../glossary.md#term-engine)/[Runtime](../../../../glossary.md#term-runtime) cleanup, Error [rollback](../../../../glossary.md#term-rollback) and retained caller-[payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) lifetimes.
Clean implementation and independent source/[Javadoc](../../../../glossary.md#term-javadoc) review; meaningful before/after [regression](../../../../glossary.md#term-regression) and
actual public [CPU](../../../../glossary.md#term-cpu) smokes. This repairs existing [ownership](../../../../glossary.md#term-ownership-meanings) contracts, not a new architecture/[API](../../../../glossary.md#term-application-programming-interface-api).

## Observations before repair

1. AdvancedEngine.run protects runner execution but allocates/registers AdvancedRunResult outside
   that rollback region. Constructor/ArrayList.add failure leaves the inward result unclosed and
   activeOperations unreleased; Engine.close can wait forever. Ordinary finishOrdinaryRun already
   protects the equivalent publication. Allocation failure is a real possibility, but no safe
   deterministic public allocation-failure switch exists; do not exhaust the coordinator heap.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [rollback glossary](../../../../glossary.md#term-rollback); [Engine glossary](../../../../glossary.md#term-engine))</span>
2. Retained closed RunResult.Publication keeps AdvancedRunResult.ownedBorrowedInputs, inward result
   representations and RunState buffer/workspace references reachable. Metadata-after-close is
   supported; fresh per-run heap payload retention is not required for it. Native close calls do
   happen: this is not a demonstrated native-arena release failure.<span data-glossary-navigation="true"> ([RunResult glossary](../../../../glossary.md#run-result--runresult); [RunState glossary](../../../../glossary.md#run-state--runstate); [buffer glossary](../../../../glossary.md#term-buffer); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [arena glossary](../../../../glossary.md#term-arena))</span>

Actual source-matched public diagnostic: storage-free neg [prototype](../../../../glossary.md#term-prototype), fresh [heap](../../../../glossary.md#term-heap) input each call,
materialize/check, close result, retain only [Publication](../../../../glossary.md#term-publication). Separate -Xmx96m JVMs: [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64)
[release](../../../../glossary.md#term-release)-[metadata](../../../../glossary.md#term-metadata) controls each complete 128 numeric runs; closed metadata exhausts heap after 42
FLOAT32 runs and 27 FLOAT64 runs. Source chains establish the cause; no [GC](../../../../glossary.md#term-garbage-collection-gc) timing assertion or
production instrumentation. The unchanged before/after source [hash](../../../../glossary.md#term-hash) and [logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) are retained in
coordinator evidence; the temporary diagnostic source is removed after acceptance.

## Bounded implementation

Extend the existing advanced [publication](../../../../glossary.md#term-publication) [transaction](../../../../glossary.md#term-transaction) through wrapper creation and registration.
On every unchecked failure, close whichever inward/wrapper [owner](../../../../glossary.md#term-owner) actually exists, preserve primary
and distinct reverse-cleanup suppression, and [release](../../../../glossary.md#term-release) [Engine](../../../../glossary.md#term-engine) [admission](../../../../glossary.md#term-admission) exactly once even when
[rollback](../../../../glossary.md#term-rollback) itself fails. Reuse existing closeWithSuppression/finishFailure conventions; no generic
publication framework, [callback](../../../../glossary.md#term-callback) factory, new owner/[lease](../../../../glossary.md#term-lease)/lock or normal-path allocation.
Close the constructed wrapper or the inward result, never both; rollback is outside lifecycleLock.
Release admission in a finally protecting rollback, preserving current [argument](../../../../glossary.md#term-argument)-[validation](../../../../glossary.md#term-validation) order.

After close begins, clear direct per-[run](../../../../glossary.md#term-run) [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) references while retaining [immutable](../../../../glossary.md#term-immutable) [metadata](../../../../glossary.md#term-metadata),
counts and plans. AdvancedRunResult transfers its borrow-list to the first cleanup's local [scope](../../../../glossary.md#term-scope)
and drops the field reference, while still closing wrappers reverse-order exactly once. [Runtime](../../../../glossary.md#term-runtime)
[RunResult](../../../../glossary.md#run-result--runresult) [releases](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) publication representation references even if state cleanup fails. [RunState](../../../../glossary.md#run-state--runstate)
releases each [borrowed](../../../../glossary.md#ownership-borrowing-and-lease)/owned [buffer](../../../../glossary.md#term-buffer) [binding](../../../../glossary.md#term-binding) and workspace reference after its required cleanup;
borrowed caller storage is never closed. Keep reverse encounter order, attempt-all cleanup,
unique [identity](../../../../glossary.md#term-identity-meanings) rules, preserved count/plan inspection and current failure replay semantics.
Clear reference cells, not inner-array [dimensions](../../../../glossary.md#dimension): bufferRepresentationCount remains inspectable.
Publication [aliases](../../../../glossary.md#aliasing-and-arena) do not add cleanup occurrences. Runtime repeated close does not replay a prior
failure; outward AdvancedRunResult retains exact replay. Keep that distinction.

Runtime RunState/RunResult remain non-[thread-safe](../../../../glossary.md#term-thread-safe) by contract. Outward serialized materialize/
close and close replay/wait/interruption behavior remain unchanged. Do not weaken current access-
after-close guards, add second leases, close caller [arenas](../../../../glossary.md#term-arena), collect metadata early, or retain
payload to make inspection tests pass. Do not change CpuBorrowedBuffer's caller-owned no-op close.
Shared prepared recipes may remain; the fix releases fresh per-run [ownership](../../../../glossary.md#term-ownership-meanings) references, not
immutable metadata or required retained cleanup throwables.

## Contracts

- [Root lifecycle/invariants](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>.
- [Runtime ownership](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>,
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>,
  [Engine](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>: sole Runtime
  leases, completed synchronous results, caller borrowing, metadata and attempt-all rollback.<span data-glossary-navigation="true"> ([leases glossary](../../../../glossary.md#term-lease); [synchronous glossary](../../../../glossary.md#term-synchronous); [borrowing glossary](../../../../glossary.md#term-borrowing); [metadata glossary](../../../../glossary.md#term-metadata); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- MAIN accepted the exact existing-contract clarification in root/owning Run/Engine lifecycle.
  No dependency, application API, native/generated ABI, algorithm or numerical policy changes.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- [Production resource gate](../../../production-release-goal.md#release-acceptance-checklist).

## Files and [ownership](../../../../glossary.md#term-ownership-meanings)

Production: existing [modules](../../../../glossary.md#term-module)/[engine](../../../../glossary.md#term-engine) AdvancedEngine.[run](../../../../glossary.md#term-run)/ordinary [publication](../../../../glossary.md#term-publication) conventions,
AdvancedRunResult.close and [RunResult](../../../../glossary.md#run-result--runresult) [Javadoc](../../../../glossary.md#term-javadoc); existing modules/[runtime](../../../../glossary.md#term-runtime) run.[RunState](../../../../glossary.md#run-state--runstate).close and
run.RunResult.close. No new/moved production type/package or changed public [signature](../../../../glossary.md#term-signature).

Tests: existing AdvancedEngineLifecycleTest/EngineTypedLifecycleTest publication-failure helpers,
Runtime RunStateTest/RunResultTest cleanup/access/count cases, and a focused real-[CPU](../../../../glossary.md#term-cpu) integration
[regression](../../../../glossary.md#term-regression) for retaining closed [metadata](../../../../glossary.md#term-metadata) under a bounded child [heap](../../../../glossary.md#term-heap). Reuse PlainJvmCpuIntegrationTest's
fresh-[JVM](../../../../glossary.md#term-java-virtual-machine-jvm) [classpath](../../../../glossary.md#term-classpath)/environment pattern rather than inventing another launcher framework.
MAIN owns shared architecture/plans/[API](../../../../glossary.md#term-application-programming-interface-api) guide/changelog/evidence. CPU0078 source is separate.

## Dependencies and integration

- Depends on: source audit and actual retention before evidence; reviewed exact repair and authority.
- Conflicts with: advanced Engine result/publication/close, Runtime result/state cleanup and their
  focused tests/docs. No CPU provider/generator/finalizer/ledger changes.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [publication glossary](../../../../glossary.md#term-publication); [Runtime glossary](../../../../glossary.md#term-runtime); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Parallel group: [RELEASE](../../../../glossary.md#term-release)-CPU78-[LIFECYCLE](../../../../glossary.md#lifecycle); disjoint files and semantic [ownership](../../../../glossary.md#term-ownership-meanings) from CPU0078.
- Common base revision: HEAD 07a281edd3086efde2853211b71b48288db027e1 plus cpu0078-base.json
  production source and authority-only Engine0022 addendum; exact engine0022-base.json/worktree
  recorded by MAIN before launch. CPU0078 receives the same additive unchanged-meaning authority.
- Integration order: test-first publication witness -> MAIN before gate -> existing-owner fixes ->
  independent review -> ordered MAIN integration/gates and unchanged actual bounded-heap after.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap))</span>
- Integration validation: Engine/Runtime/Javadocs, affected Prepare and actual Engine CPU integration,
  actual F32/F64 lifetime/precedence/race and bounded metadata heap scenarios; final release join later.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Prepare glossary](../../../../glossary.md#prepare); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [lifetime glossary](../../../../glossary.md#term-lifetime); [race glossary](../../../../glossary.md#term-race); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN.

## Acceptance and [validation](../../../../glossary.md#term-validation)

- Deterministic injected publication failure proves result cleanup, original/suppressed errors and
  admitted-operation release/terminating Engine.close; it tests behavior, not private field counts.
  Reuse the existing rejecting-publication-list test convention. First add/freeze the regression,
  notify MAIN, and wait for MAIN's failing-before evidence before production edits. Do not add a
  production fault/backdoor flag or use a success-only actual run as rollback proof.
  Inject Error; the failing-before test must not unconditionally close its stranded baseline Engine
  in teardown and hang. Wrapper-allocation failure remains source-inspected, not a fake factory.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [release glossary](../../../../glossary.md#term-release); [Engine glossary](../../../../glossary.md#term-engine); [regression glossary](../../../../glossary.md#term-regression); [run glossary](../../../../glossary.md#term-run); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Unchanged actual bounded child program completes 128 retained-metadata runs in F32 and F64 after
  the fix, with correct numeric outputs and readable metadata. Add an isolated full-suite-safe
  behavior regression; no flaky System.gc/WeakReference deadline or bare not-throw assertion.
  Cover reference release when cleanup throws as well as normal close, without asserting private
  field layout or using an admission/GC timing backdoor.<span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [layout glossary](../../../../glossary.md#layout); [admission glossary](../../../../glossary.md#term-admission); [GC glossary](../../../../glossary.md#term-garbage-collection-gc))</span>
- Open results remain independently readable after prepared close; detached host values survive
  caller/Engine closure; caller storage remains owned/live until the caller closes it.<span data-glossary-navigation="true"> ([detached glossary](../../../../glossary.md#detached-model-results); [host glossary](../../../../glossary.md#term-host))</span>
- Preserve invalid-input/closed/foreign/null precedence, parallel run/copy-close isolation, failure/
  retry, reverse/duplicate cleanup and exact failure replay. Independent actual CPU stress plus
  deterministic focused tests are complementary, not a claim that public barriers prove lease order.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [lease glossary](../../../../glossary.md#term-lease))</span>
- No new instrumentation, native-byte counter or throughput/zero-growth claim from isClosed alone.
  Record the heap-pressure result separately from native release. No historical failure relabeling.<span data-glossary-navigation="true"> ([throughput glossary](../../../../glossary.md#term-throughput); [heap glossary](../../../../glossary.md#term-heap))</span>

No builds/tests/lint/formatters in the worker. MAIN runs the new [publication](../../../../glossary.md#term-publication) test before source
fixes, then final gates once reviewed sources stabilize. Source and [Javadoc](../../../../glossary.md#term-javadoc) review before acceptance.

MAIN [Ready](../../../../glossary.md#term-ready) verification: complete Engine0021/Runtime0016 baseline, audited source, both-type
actual retention before and no-[metadata](../../../../glossary.md#term-metadata) controls, independent exact design and coordinated authority
verified. Existing-contract repair needs no new [ADR](../../../../glossary.md#term-architecture-decision-record-adr)/type/[dependency](../../../../glossary.md#term-dependency)/protocol. Parallel CPU0078
[scope](../../../../glossary.md#term-scope) is disjoint; MAIN integrates test-first witness then reviewed [lifecycle](../../../../glossary.md#lifecycle) source and final gates.

## Result

Implemented in the existing [Engine](../../../../glossary.md#term-engine)/[Runtime](../../../../glossary.md#term-runtime) [owners](../../../../glossary.md#term-owner); ten approved source/test files integrated
against the frozen base, with the test-first [publication](../../../../glossary.md#term-publication) witness unchanged.

- Advanced wrapper construction and registration now share execution's rollback boundary.
  Cleanup runs outside lifecycleLock, closes the wrapper or inward result once, and releases
  admission in finally even on cleanup failure. The injected-Error regression genuinely FAILED
  before source repair (missing suppressed cleanup), then passed unchanged. Wrapper-allocation
  failure coverage remains source inspection, not a fake allocation hook.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback); [releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [admission glossary](../../../../glossary.md#term-admission); [regression glossary](../../../../glossary.md#term-regression))</span>
- Closed AdvancedRunResult, Runtime RunResult and RunState release their direct fresh-payload
  references even on cleanup failure. Dimensions, counts, plan/geometry, borrowed caller ownership,
  exact outward failure replay and Runtime non-replay remain unchanged.<span data-glossary-navigation="true"> ([RunResult glossary](../../../../glossary.md#run-result--runresult); [RunState glossary](../../../../glossary.md#run-state--runstate); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [Dimensions glossary](../../../../glossary.md#dimension); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Unchanged public `ClosedMetadataHeapSmoke` now completes 128 retained-publication runs per
  dtype under -Xmx96m, with numeric checks. The earlier FLOAT32/FLOAT64 OOM at 42/27 runs remains
  recorded as FAILED-before; both release-metadata controls completed 128. Permanent isolated
  real-CPU child regressions and cleanup-failure pressure coverage passed.<span data-glossary-navigation="true"> ([dtype glossary](../../../../glossary.md#data-type--datatype); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [OOM glossary](../../../../glossary.md#term-out-of-memory-oom); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Actual public F32/F64 scenarios passed independent results, invalid-shape/dead-input retry,
  bounded-copy and invalid-softmax retry, null/foreign/closed precedence, prepared close with
  live results, caller-arena ownership and detached bytes after caller/Engine close. Each type
  also passed 16 prepared-close/run, 16 copy/close and 16 paired concurrent-run stress groups.
  These public barriers are not deterministic internal lease-order proof.<span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [shape glossary](../../../../glossary.md#shape); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [detached glossary](../../../../glossary.md#detached-model-results); [lease glossary](../../../../glossary.md#term-lease))</span>

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
| [Prepare](../../../../glossary.md#prepare) | 98 | 0 |
| Selected real-[CPU](../../../../glossary.md#term-cpu) integration | 6 | 0 |
| Total | 417 | 2 |

No failures/errors. The two skipped Engine timing methods require
`synaptik.cpu.dynamicBatch.performance=true`; no timing campaign was [run](../../../../glossary.md#term-run) for this repair.
The initial separate public-[lifetime](../../../../glossary.md#term-lifetime) [fixture](../../../../glossary.md#term-fixture) failed by closing its [Arena](../../../../glossary.md#term-arena) twice; its corrected
single-[owner](../../../../glossary.md#term-owner) fixture passed without a product change, and that first failure is retained honestly.

Evidence: frozen integration SHA [manifest](../../../../glossary.md#term-manifest), before/after [JUnit](../../../../glossary.md#term-junit) XML, gate counts, [heap](../../../../glossary.md#term-heap) [logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) and
public-[lifecycle](../../../../glossary.md#lifecycle) source [hash](../../../../glossary.md#term-hash)/[log](../../../../glossary.md#term-log) are in the coordinator's source-[snapshot](../../../../glossary.md#term-snapshot) evidence. Native-memory
plateau, arbitrary retained cleanup-throwable [graphs](../../../../glossary.md#graph), a full-root PASS and complete production
[release](../../../../glossary.md#term-release) readiness are not claimed.
