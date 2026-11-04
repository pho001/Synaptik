# Task 0001A: Bound Training [Parameter](../../../../glossary.md#term-parameter) [Snapshot](../../../../glossary.md#term-snapshot) Byte Sizes

> **Editorial reading context — not a status update.** Native parameter storage can be
> larger than a Java array, but a detached session snapshot still needs array-sized payloads.
> Training checks both represented parameter bytes and optimizer working-width bytes before
> allocating that parameter's staging state. Low-precision logical values can therefore fit
> while their wider master/optimizer payload does not. This is a representation/admission
> bound, not a general memory budget or a native-allocation ownership change.
> Today's [parameter domain](../../../../reference/training/parameters.md#initial-parameter-domain)
> explains those widths and [snapshot state](../../../../reference/training/optimizers.md#atomicity-state-concurrency-and-close)
> explains what the detached payloads contain.

## Status

Complete

## Change class

Class B — changes public session-open rejection for [parameters](../../../../glossary.md#term-parameter) that cannot be represented by the current in-memory [snapshot](../../../../glossary.md#term-snapshot) format.

## Goal

Reject any training [parameter](../../../../glossary.md#term-parameter) whose logical or [optimizer](../../../../glossary.md#term-optimizer)-width byte [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) exceeds a Java byte-array length before allocating that parameter's update [buffers](../../../../glossary.md#term-buffer), a [seed](../../../../glossary.md#term-seed), or compiling the [training graph](../../../../glossary.md#training-graph).

## [Scope](../../../../glossary.md#term-scope)

- Validate both checked byte counts during `TrainingParameter.capture` using the existing supported [parameter](../../../../glossary.md#term-parameter) types and storage contract.
- Keep [`snapshot`](../../../../glossary.md#term-snapshot) and `validateState` byte-length conversions safe by construction.
- Add focused boundary tests that do not allocate giant [tensors](../../../../glossary.md#tensor) or arrays; preserve existing accepted-domain behavior.
- Review affected public [Javadoc](../../../../glossary.md#term-javadoc) and training [API](../../../../glossary.md#term-application-programming-interface-api)/user documentation.

## Non-goals

- Chunked snapshots or support for [parameters](../../../../glossary.md#term-parameter) larger than a Java array.
- Changes to native-storage [ownership](../../../../glossary.md#term-ownership-meanings), [optimizer](../../../../glossary.md#term-optimizer) mathematics, [buffer](../../../../glossary.md#term-buffer) count, or per-[run](../../../../glossary.md#term-run) overlap checks.
- Changes to the authoritative [architecture contract](../../../../glossary.md#architecture-contract).

## Contracts

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md#core-invariants) — exact state and failure behavior.
- [Extensions/training — optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [lifecycle glossary](../../../../glossary.md#lifecycle))</span> — precompile [capture](../../../../glossary.md#term-capture-meanings), supported [parameter](../../../../glossary.md#term-parameter) domain, [detached](../../../../glossary.md#detached-model-results) snapshots, and [validation](../../../../glossary.md#term-validation)-before-mutation.

Stop if these contracts are insufficient or contradictory.

## Dependencies and integration

- Depends on: Training 0001 (Complete).
- Conflicts with: subsequent Training [capture](../../../../glossary.md#term-capture-meanings)/[run](../../../../glossary.md#term-run)-[validation](../../../../glossary.md#term-validation) work and any concurrent [Training session](../../../../glossary.md#training-session--trainingsession)/state edits.
- Parallel group: None.
- Common base revision: `fa6e2738` on `main`.
- Integration order: 0001A, then any further Training review fix after review.
- Integration validation: `./gradlew :extensions:training:test :extensions:training:javadoc` and focused Training [integration test](../../../../glossary.md#term-integration-test) if acceptance behavior changes there.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main coordination context.

Training 0001 is Complete, this brief is the sole authorized Training write frontier, dependencies are satisfied at the clean common base, and no conflicting Training write task is active. The user authorized the Training-review follow-up.

## Files and symbols

- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingParameter.java`: [`capture`](../../../../glossary.md#term-capture-meanings), size helpers, [`snapshot`](../../../../glossary.md#term-snapshot), `validateState`.
- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java`: `open` [Javadoc](../../../../glossary.md#term-javadoc) if needed.
- Focused tests under `extensions/training/src/test/java/io/github/pho001/synaptik/training/`.
- `docs/api/training-api.md` and `docs/user-guide/training.md` only if their stated accepted domain or failure timing needs clarification.

## Acceptance criteria

- A positive [element](../../../../glossary.md#term-element) count with a logical byte count or [optimizer](../../../../glossary.md#term-optimizer)-width byte count above `Integer.MAX_VALUE` produces a clear `IllegalArgumentException` at [capture](../../../../glossary.md#term-capture-meanings) before that [parameter](../../../../glossary.md#term-parameter)'s update-[buffer](../../../../glossary.md#term-buffer) allocation or [compile](../../../../glossary.md#compile).
- Boundary tests cover [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), and low-[precision](../../../../glossary.md#term-precision) logical-versus-master width without allocating huge payloads.
- Existing small-parameter [snapshot](../../../../glossary.md#term-snapshot)/restore and failure-[atomicity](../../../../glossary.md#term-atomicity) tests still pass.
- [Javadoc](../../../../glossary.md#term-javadoc) and explanatory documentation accurately describe the current bound.

## [Validation](../../../../glossary.md#term-validation)

- Worker: targeted new test, then `./gradlew :extensions:training:test :extensions:training:javadoc`; record counts and skips.
- Integration: focused [CPU](../../../../glossary.md#term-cpu) Training [integration test](../../../../glossary.md#term-integration-test) if affected, `git diff --check`, documentation link/terminology checks. Do not rerun successful Java suites in the independent documentation pass without a concrete reason.

## Documentation and review impact

Independent targeted Class B documentation/review context required because the accepted public input domain changes. Review public `open` [Javadoc](../../../../glossary.md#term-javadoc), training [API](../../../../glossary.md#term-application-programming-interface-api)/user guide and glossary terminology; update only what the actual diff requires.

## Result

- `TrainingParameter.capture` now bounds both declared-width and [optimizer](../../../../glossary.md#term-optimizer)-width [snapshot](../../../../glossary.md#term-snapshot) bytes before inspecting that [parameter](../../../../glossary.md#term-parameter)'s storage or allocating its [buffers](../../../../glossary.md#term-buffer); later snapshot/restore length conversions use `Math.toIntExact`.
- Two new focused tests cover [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), [BFLOAT16](../../../../glossary.md#term-bfloat16), and [FLOAT16](../../../../glossary.md#term-float16) boundaries without giant allocations, including the public session-open path.
- Updated `TrainingSession.open` [Javadoc](../../../../glossary.md#term-javadoc), [API](../../../../glossary.md#term-application-programming-interface-api) and user guides, and the Training glossary entry. The independent Class B review found and resolved an overbroad allocation-timing claim; its targeted recheck found no remaining issue.
- Worker [validation](../../../../glossary.md#term-validation): targeted boundary tests passed; `./gradlew :extensions:training:test :extensions:training:javadoc` passed with 32 tests and no skips. Focused [CPU](../../../../glossary.md#term-cpu) Training integration passed four tests; two configured-[Metal](../../../../glossary.md#term-metal) cases skipped because `SYNAPTIK_METAL_TEST_LIBRARY` was absent. Final Javadoc-only clarification was followed by successful `./gradlew :extensions:training:javadoc`; `git diff --check` passed.
