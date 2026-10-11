# Task 0001B: Remove Repeated Parameter Overlap Scan

## Status

Complete

## Change class

Class C — changes the checks on a training hot path while preserving the fixed-storage, no-overlap lifecycle invariant.

## Goal

Keep complete per-run binding and storage validation without repeating pairwise overlap checks for the immutable parameter-storage set captured at session open.

## Scope

- Prove from Model's sealed `HostTensorStorage`/final `MemorySegmentStorage.segment` and Training capture/binding contracts that a parameter pair accepted at open cannot start overlapping while wrappers and storage associations remain fixed.
- Remove only the redundant parameter-to-parameter pair scan from `validateModuleAndParameters`; retain all per-parameter liveness, binding, and finite-state checks.
- Preserve the open-time overlap check and per-run overlap checks involving dynamic compiled inputs.
- Optionally precompute fixed compiled-input diagnostic roles during binding-plan construction to avoid a repeated parameter-path scan, without changing errors or input mapping.
- Add targeted regression tests for overlapping parameter rejection at open, changed binding rejection before mutation, and dynamic input overlap rejection before mutation.

## Non-goals

- Weakening exclusive caller ownership, native-storage lifetime, input aliasing, or failure atomicity.
- Changing the dynamic-input overlap algorithm or its complexity; it needs a separate proof for heap/native/sliced storage.
- Optimizer state-buffer redesign or NN initializer/storage integration.

## Contracts

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md#core-invariants) — failure and performance invariants.
- [Extensions/training — optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle) — stable parameter storage occurrence, complete precommit view checks, and overlap rejection.
- [Foundational modules — `modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel) — Model owns the borrowed host-storage abstraction and Tensor's mutable association.

## Dependencies and integration

- Depends on: Training 0001A (Complete and independently reviewed).
- Conflicts with: concurrent Training session/binding changes.
- Parallel group: None.
- Common base revision: `fa6e2738` on `main` plus the reviewed, uncommitted 0001A worktree change.
- Integration order: after 0001A independent review.
- Integration validation: focused Training module tests, CPU integration smoke, `git diff --check`.
- Shared-document integration owner: main coordination context.

Training 0001A is Complete with its independent Class B review and successful validation. This is the sole authorized Training write frontier; no conflicting Training execution context is active. The user authorized continuation of the Training-review fixes.

## Files and symbols

- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java`: `openInternal`, `bindingPlan`, `bindInputs`, `validateModuleAndParameters`.
- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingParameter.java`: storage identity/segment capture, read-only proof; edit only if required.
- Focused Training unit/integration tests and affected Javadoc.

## Acceptance criteria

- Open still rejects any two overlapping parameter storages before compile.
- Run/restore/state still reject a replaced parameter or invalid/dead storage before mutation.
- Any nonempty dynamic-input overlap, including parameter alias, still rejects before mutation.
- The per-run validation no longer makes pairwise calls over the fixed parameter set; checks scale linearly with parameter count apart from necessary element-wise state validation and dynamic input checks.
- Error precedence and failure atomicity remain unchanged.

## Validation

- Worker: focused Training tests and `./gradlew :extensions:training:test :extensions:training:javadoc`.
- Integration: CPU Training integration smoke and targeted independent Class C review; reuse successful Java evidence in a documentation-only review.

## Documentation and review impact

Independent targeted Class C review is mandatory. Inspect the lifecycle Javadoc and training user/API docs; update only if external behavior or the described validation sequence changes. Record a reasoned no-change conclusion otherwise.

## Result

- Removed per-operation fixed parameter-pair overlap comparisons from both complete parameter-view validation and compiled-input binding. Open still checks every parameter pair; each operation still checks fixed bindings, storage, finiteness, and all compiled-input pairs involving an external input or the cotangent seed, in original compiled-input order.
- Added tests for overlapping parameter slices at open, changed/dead parameter storage, dynamic input aliases, and exact first-pair diagnostic ordering under multiple overlaps. Updated lifecycle Javadocs; the API/user guides and glossary need no 0001B change because public behavior and terminology are unchanged.
- Worker validation: targeted ordering test passed; `./gradlew :extensions:training:test` passed 35 tests with no failures; `./gradlew :extensions:training:javadoc` passed. Focused CPU Training integration passed four tests; two configured-Metal cases skipped without `SYNAPTIK_METAL_TEST_LIBRARY`. `git diff --check` passed.
- Independent Class C review confirmed the fixed-segment proof, all dynamic-pair coverage and diagnostic order, and signed off after a Javadoc correction and exact-order regression assertion. No unresolved issue remains in this task.
