# Task 0001A: Bound Training Parameter Snapshot Byte Sizes

## Status

Complete

## Change class

Class B — changes public session-open rejection for parameters that cannot be represented by the current in-memory snapshot format.

## Goal

Reject any training parameter whose logical or optimizer-width byte payload exceeds a Java byte-array length before allocating that parameter's update buffers, a seed, or compiling the training graph.

## Scope

- Validate both checked byte counts during `TrainingParameter.capture` using the existing supported parameter types and storage contract.
- Keep `snapshot` and `validateState` byte-length conversions safe by construction.
- Add focused boundary tests that do not allocate giant tensors or arrays; preserve existing accepted-domain behavior.
- Review affected public Javadoc and training API/user documentation.

## Non-goals

- Chunked snapshots or support for parameters larger than a Java array.
- Changes to native-storage ownership, optimizer mathematics, buffer count, or per-run overlap checks.
- Changes to the authoritative architecture contract.

## Contracts

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md#core-invariants) — exact state and failure behavior.
- [Extensions/training — optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle) — precompile capture, supported parameter domain, detached snapshots, and validation-before-mutation.

Stop if these contracts are insufficient or contradictory.

## Dependencies and integration

- Depends on: Training 0001 (Complete).
- Conflicts with: subsequent Training capture/run-validation work and any concurrent Training session/state edits.
- Parallel group: None.
- Common base revision: `fa6e2738` on `main`.
- Integration order: 0001A, then any further Training review fix after review.
- Integration validation: `./gradlew :extensions:training:test :extensions:training:javadoc` and focused Training integration test if acceptance behavior changes there.
- Shared-document integration owner: main coordination context.

Training 0001 is Complete, this brief is the sole authorized Training write frontier, dependencies are satisfied at the clean common base, and no conflicting Training write task is active. The user authorized the Training-review follow-up.

## Files and symbols

- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingParameter.java`: `capture`, size helpers, `snapshot`, `validateState`.
- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java`: `open` Javadoc if needed.
- Focused tests under `extensions/training/src/test/java/io/github/pho001/synaptik/training/`.
- `docs/api/training-api.md` and `docs/user-guide/training.md` only if their stated accepted domain or failure timing needs clarification.

## Acceptance criteria

- A positive element count with a logical byte count or optimizer-width byte count above `Integer.MAX_VALUE` produces a clear `IllegalArgumentException` at capture before that parameter's update-buffer allocation or compile.
- Boundary tests cover FLOAT64, FLOAT32, and low-precision logical-versus-master width without allocating huge payloads.
- Existing small-parameter snapshot/restore and failure-atomicity tests still pass.
- Javadoc and explanatory documentation accurately describe the current bound.

## Validation

- Worker: targeted new test, then `./gradlew :extensions:training:test :extensions:training:javadoc`; record counts and skips.
- Integration: focused CPU Training integration test if affected, `git diff --check`, documentation link/terminology checks. Do not rerun successful Java suites in the independent documentation pass without a concrete reason.

## Documentation and review impact

Independent targeted Class B documentation/review context required because the accepted public input domain changes. Review public `open` Javadoc, training API/user guide and glossary terminology; update only what the actual diff requires.

## Result

- `TrainingParameter.capture` now bounds both declared-width and optimizer-width snapshot bytes before inspecting that parameter's storage or allocating its buffers; later snapshot/restore length conversions use `Math.toIntExact`.
- Two new focused tests cover FLOAT64, FLOAT32, BFLOAT16, and FLOAT16 boundaries without giant allocations, including the public session-open path.
- Updated `TrainingSession.open` Javadoc, API and user guides, and the Training glossary entry. The independent Class B review found and resolved an overbroad allocation-timing claim; its targeted recheck found no remaining issue.
- Worker validation: targeted boundary tests passed; `./gradlew :extensions:training:test :extensions:training:javadoc` passed with 32 tests and no skips. Focused CPU Training integration passed four tests; two configured-Metal cases skipped because `SYNAPTIK_METAL_TEST_LIBRARY` was absent. Final Javadoc-only clarification was followed by successful `./gradlew :extensions:training:javadoc`; `git diff --check` passed.
