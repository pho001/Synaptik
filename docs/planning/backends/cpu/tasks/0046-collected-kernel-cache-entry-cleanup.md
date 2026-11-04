# Task CPU0046: Reclaim collected generated-[kernel](../../../../glossary.md#kernel) [cache](../../../../glossary.md#term-cache) entries

## Status

Complete — MAIN accepted the bounded [cache](../../../../glossary.md#term-cache) repair on 2026-10-28; the full-suite legacy
performance-gate limitation is recorded below, not reported as a green full [run](../../../../glossary.md#term-run).

## Change class

Class C: process-local weak-[cache](../../../../glossary.md#term-cache) [lifetime](../../../../glossary.md#term-lifetime) and synchronized [publication](../../../../glossary.md#term-publication). MAIN implements and
reviews under the user's explicit context-isolation exception after separate workers stalled.

## Goal and [scope](../../../../glossary.md#term-scope)

Remove keys whose weakly interned generated [artifact](../../../../glossary.md#term-artifact) has been collected, on the next cold artifact
realization. Preserve strongly held prepared artifacts and same-key reuse. Queue notifications for
an older reference must not remove a newer artifact registered under the same key.

## Non-goals

No new public [API](../../../../glossary.md#term-application-programming-interface-api), [cache](../../../../glossary.md#term-cache) quota/[LRU](../../../../glossary.md#single-flight-and-least-recently-used-caching), disk expiry, background thread, forced [GC](../../../../glossary.md#term-garbage-collection-gc), geometry-baked
[specialization](../../../../glossary.md#term-specialization), generated [bytecode](../../../../glossary.md#term-bytecode)/[identity](../../../../glossary.md#term-identity-meanings) change, [runtime](../../../../glossary.md#term-runtime) hot-path work or numerical change.

## Contracts

Start at [the authority index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
Read root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), and
`backend-execution.md` [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span>,
[CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>.
[CPU](../../../../glossary.md#term-cpu) finalization owns [artifact](../../../../glossary.md#term-artifact) reuse; [Runtime](../../../../glossary.md#term-runtime) still receives fixed executables only.

## Dependencies and integration

- Depends on: CPU0045 and Compiler0016 Complete, accepted by MAIN.
- Conflicts with: [artifact](../../../../glossary.md#term-artifact)-store writers, [cache](../../../../glossary.md#term-cache)/global-state tests and shared [planning](../../../../glossary.md#term-planning)/docs.
- Parallel group: None; no concurrent writer or benchmark.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted
  dirty tree `/private/tmp/synaptik-cpu-repairs.8TWk5O/compiler0016-accepted.json`.
- Integration order: verified packet → source/tests → owning [module](../../../../glossary.md#term-module) [validation](../../../../glossary.md#term-validation) → targeted review/docs.
- Integration validation: focused cache/specialization tests, then CPU tests/Javadoc once with
  performance opt-ins false; inspect unchanged generated-artifact identities and preserved evidence.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [CPU glossary](../../../../glossary.md#term-cpu); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN.

[Ready](../../../../glossary.md#term-ready) verification: completed predecessor results and validation, unchanged HEAD/origin/index,
accepted dirty tree and current owning contract checked. [Compiler](../../../../glossary.md#term-compiler) frontier is closed; CPU0046 is
the sole implementation frontier. No worker/test/measurement is competing for its state.

## Files and symbols

- `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStore.java`:
  weak map, `loadOrGenerateObserved`, existing isolated-test reset and focused package-private test seam.
- Matching `CpuGeneratedKernelArtifactStoreTest.java`: deterministic queue cleanup, late stale
  notification, live reuse and concurrent single-publication cases, without waiting for GC.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [GC glossary](../../../../glossary.md#term-garbage-collection-gc))</span>
- `docs/backend-guide/cpu-backend.md#generated-class-byte-persistence-and-evidence`:
  distinguish weak-object reuse from persisted-byte reuse and explain cold queue cleanup.
- Existing glossary [artifact](../../../../glossary.md#term-artifact)-store entry: the same weak-versus-persisted distinction, no new term.
- [CPU](../../../../glossary.md#term-cpu) master and roadmap: current status only.

## Acceptance criteria

1. After deterministic enqueue of a dead reference, the next different-key realization removes
   its key while leaving live cache entries reusable by identity.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
2. An old enqueued reference cannot erase its same-key replacement, including after test reset.
3. Concurrent same-key realizations still publish one [artifact](../../../../glossary.md#term-artifact) and reuse that exact live object.
4. Class bytes, specialization identities, persisted validation/fallback and runtime execution
   semantics remain unchanged; cleanup occurs only under the existing cold lock.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [validation glossary](../../../../glossary.md#term-validation); [runtime glossary](../../../../glossary.md#term-runtime))</span>

## [Validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) `:backends:cpu:test --tests '*CpuGeneratedKernelArtifactStoreTest' --tests '*CpuKernelSpecializationTest'`,
then `:backends:cpu:test :backends:cpu:javadoc`, each with generalScan.performance,
generalScan.pointwisePerformance and dynamicBatch.performance false, `--max-workers=1 --console=plain`.
No repository-wide rerun or benchmark; reviewers reuse successful results.

## Documentation and review impact

[Planning](../../../../glossary.md#term-planning) profile for this brief; [Backend](../../../../glossary.md#backend)-guide/Developer-guide profiles for the [cache](../../../../glossary.md#term-cache) explanation.
Update implementation [Javadoc](../../../../glossary.md#term-javadoc) for cold cleanup and reference [ownership](../../../../glossary.md#term-ownership-meanings). Review glossary terms
without introducing a new abstraction or terminology. MAIN's review is not independent-context
review; record the explicit user exception and any remaining limitations.

## Result

- Added key-bearing weak references and a reference queue. The existing cold lookup/publication
  lock drains queued entries with <span data-glossary-navigation="true"> ([reference queue glossary](../../../../glossary.md#weak-reference-and-weak-interning); [lookup glossary](../../../../glossary.md#term-lookup); [publication glossary](../../../../glossary.md#term-publication))</span>`remove(key, exactReference)`, preserving newer replacements
  and live interned artifacts. Reset also drains pending notifications; no runtime hot-path work.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Added deterministic collection/late-notification/reset tests and four-caller atomic publication
  coverage, with a package-private weak-token snapshot seam; no reliance on GC timing.<span data-glossary-navigation="true"> ([atomic publication glossary](../../../../glossary.md#term-atomic-publication); [token glossary](../../../../glossary.md#term-token); [snapshot glossary](../../../../glossary.md#term-snapshot); [GC glossary](../../../../glossary.md#term-garbage-collection-gc))</span>
- Focused cache/specialization run PASS. Full CPU run: 1346 tests, 2 failures, 52 skipped.
  Compiler0016's changed reverse graph required a current-fixture hash update; actual normalized
  disassembly is unchanged and the repaired methodology/cache/specialization run passed <span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [specialization glossary](../../../../glossary.md#term-specialization); [run glossary](../../../../glossary.md#term-run); [graph glossary](../../../../glossary.md#graph); [fixture glossary](../../../../glossary.md#term-fixture); [hash glossary](../../../../glossary.md#term-hash))</span>**35 tests**.
  Successful unaffected results are reused, not a second full-suite run.
- Remaining full-suite failure: unchanged legacy multi-store NEG five-fork timing gate, no accepted
  sample for fork 3 at <=1.15. It directly generates its kernel, bypassing both changed production
  paths. It runs unconditionally despite the three false performance opt-ins; no retry, weakened
  assertion, generator edit or new performance claim. Future ordinary-test isolation from such
  historical benchmarks needs separate scope. A fully green CPU suite is not claimed.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [kernel glossary](../../../../glossary.md#kernel); [scope glossary](../../../../glossary.md#term-scope))</span>
- [CPU](../../../../glossary.md#term-cpu) [Javadoc](../../../../glossary.md#term-javadoc) generated successfully, 94 existing warnings outside this repair. [Logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings):
  `/private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0046-{focused,validation,repaired-validation}.log`;
  `compiler0016-identity-review.txt` records the precise integration-failure disposition.
- MAIN reviewed source, exact-token concurrency/lifetime behavior, tests and documentation under
  the user-approved context exception; this was not an independent-context review. No production
  Compiler dependency, capability, generated format/identity, ABI, kernel arithmetic or ADR change.<span data-glossary-navigation="true"> ([concurrency glossary](../../../../glossary.md#term-concurrency); [lifetime glossary](../../../../glossary.md#term-lifetime); [Compiler glossary](../../../../glossary.md#term-compiler); [dependency glossary](../../../../glossary.md#term-dependency); [identity glossary](../../../../glossary.md#term-identity-meanings); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Documentation/Javadoc updated in the store, CPU guide, existing glossary and planning. Per-entry
  envelope bounds are explicitly not a global live-class or disk quota. Pending cleanup waits for
  the next cold realization and does not retain the collected artifact.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

Status: Complete
