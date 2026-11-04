# Task 0001B: Remove Repeated [Parameter](../../../../glossary.md#term-parameter) Overlap Scan

> **Editorial reading context — not a status update.** “Scan” here means pairwise memory
> overlap checking, not a recurrent Tensor operation. Parameters and their storage identities
> are fixed for the session, so their mutual overlap is established at open; liveness and
> unchanged-binding checks still run later. External batch storage can change each run and
> still needs overlap validation. The application must exclude conflicting mutation; removing
> one repeated check does not transfer that duty to Training.
> Today's [live parameter admission](../../../../reference/training/parameters.md#generic-live-parameter-admission)
> and [input mapping/overlap rules](../../../../reference/training/sessions.md#session-construction-and-input-mapping)
> distinguish the fixed set from the fresh borrowed inputs.

## Status

Complete

## Change class

Class C — changes the checks on a training [hot path](../../../../glossary.md#term-hot-path) while preserving the fixed-storage, no-overlap [lifecycle](../../../../glossary.md#lifecycle) [invariant](../../../../glossary.md#term-invariant).

## Goal

Keep complete per-[run](../../../../glossary.md#term-run) [binding](../../../../glossary.md#term-binding) and storage [validation](../../../../glossary.md#term-validation) without repeating pairwise overlap checks for the [immutable](../../../../glossary.md#term-immutable) [parameter](../../../../glossary.md#term-parameter)-storage set captured at session open.

## [Scope](../../../../glossary.md#term-scope)

- Prove from [Model](../../../../glossary.md#term-model)'s [sealed](../../../../glossary.md#term-sealed) `HostTensorStorage`/final `MemorySegmentStorage.segment` and Training [capture](../../../../glossary.md#term-capture-meanings)/[binding](../../../../glossary.md#term-binding) contracts that a [parameter](../../../../glossary.md#term-parameter) pair accepted at open cannot start overlapping while wrappers and storage associations remain fixed.
- Remove only the redundant parameter-to-parameter pair scan from `validateModuleAndParameters`; retain all per-parameter [liveness](../../../../glossary.md#term-liveness), [binding](../../../../glossary.md#term-binding), and [finite](../../../../glossary.md#term-finite)-state checks.
- Preserve the open-time overlap check and per-[run](../../../../glossary.md#term-run) overlap checks involving dynamic compiled inputs.
- Optionally precompute fixed compiled-input diagnostic roles during binding-plan construction to avoid a repeated parameter-path scan, without changing errors or input mapping.
- Add targeted [regression](../../../../glossary.md#term-regression) tests for overlapping parameter rejection at open, changed binding rejection before mutation, and dynamic input overlap rejection before mutation.

## Non-goals

- Weakening exclusive caller [ownership](../../../../glossary.md#term-ownership-meanings), native-storage [lifetime](../../../../glossary.md#term-lifetime), input [aliasing](../../../../glossary.md#term-aliasing), or [failure atomicity](../../../../glossary.md#term-failure-atomicity).
- Changing the dynamic-input overlap algorithm or its complexity; it needs a separate proof for [heap](../../../../glossary.md#term-heap)/native/sliced storage.
- [Optimizer state](../../../../glossary.md#term-optimizer-state)-[buffer](../../../../glossary.md#term-buffer) redesign or [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) initializer/storage integration.

## Contracts

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md#core-invariants) — failure and performance invariants.
- [Extensions/training — optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [lifecycle glossary](../../../../glossary.md#lifecycle))</span> — stable [parameter](../../../../glossary.md#term-parameter) storage [occurrence](../../../../glossary.md#term-occurrence), complete [precommit](../../../../glossary.md#term-precommit) [view](../../../../glossary.md#term-view-meanings) checks, and overlap rejection.
- [Foundational modules — `modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span> — [Model](../../../../glossary.md#term-model) owns the [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) [host](../../../../glossary.md#term-host)-storage abstraction and [Tensor](../../../../glossary.md#tensor)'s [mutable](../../../../glossary.md#term-mutable) association.

## Dependencies and integration

- Depends on: Training 0001A (Complete and independently reviewed).
- Conflicts with: concurrent [Training session](../../../../glossary.md#training-session--trainingsession)/[binding](../../../../glossary.md#term-binding) changes.
- Parallel group: None.
- Common base revision: `fa6e2738` on `main` plus the reviewed, uncommitted 0001A worktree change.
- Integration order: after 0001A independent review.
- Integration [validation](../../../../glossary.md#term-validation): focused Training [module](../../../../glossary.md#term-module) tests, [CPU](../../../../glossary.md#term-cpu) integration [smoke](../../../../glossary.md#term-smoke-test), `git diff --check`.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main coordination context.

Training 0001A is Complete with its independent Class B review and successful validation. This is the sole authorized Training write frontier; no conflicting Training [execution context](../../../../glossary.md#term-execution-context) is active. The user authorized continuation of the Training-review fixes.

## Files and symbols

- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java`: `openInternal`, `bindingPlan`, `bindInputs`, `validateModuleAndParameters`.
- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingParameter.java`: storage [identity](../../../../glossary.md#term-identity-meanings)/segment [capture](../../../../glossary.md#term-capture-meanings), read-only proof; edit only if required.
- Focused Training unit/integration tests and affected [Javadoc](../../../../glossary.md#term-javadoc).

## Acceptance criteria

- Open still rejects any two overlapping [parameter](../../../../glossary.md#term-parameter) storages before [compile](../../../../glossary.md#compile).
- [Run](../../../../glossary.md#term-run)/restore/state still reject a replaced parameter or invalid/dead storage before mutation.
- Any nonempty dynamic-input overlap, including parameter [alias](../../../../glossary.md#aliasing-and-arena), still rejects before mutation.
- The per-run [validation](../../../../glossary.md#term-validation) no longer makes pairwise calls over the fixed parameter set; checks scale linearly with parameter count apart from necessary [element](../../../../glossary.md#term-element)-wise state validation and dynamic input checks.
- Error precedence and [failure atomicity](../../../../glossary.md#term-failure-atomicity) remain unchanged.

## [Validation](../../../../glossary.md#term-validation)

- Worker: focused Training tests and `./gradlew :extensions:training:test :extensions:training:javadoc`.
- Integration: [CPU](../../../../glossary.md#term-cpu) Training integration [smoke](../../../../glossary.md#term-smoke-test) and targeted independent Class C review; reuse successful Java evidence in a documentation-only review.

## Documentation and review impact

Independent targeted Class C review is mandatory. Inspect the [lifecycle](../../../../glossary.md#lifecycle) [Javadoc](../../../../glossary.md#term-javadoc) and training user/[API](../../../../glossary.md#term-application-programming-interface-api) docs; update only if external behavior or the described [validation](../../../../glossary.md#term-validation) sequence changes. Record a reasoned no-change conclusion otherwise.

## Result

- Removed per-[operation](../../../../glossary.md#operation) fixed [parameter](../../../../glossary.md#term-parameter)-pair overlap comparisons from both complete parameter-[view](../../../../glossary.md#term-view-meanings) [validation](../../../../glossary.md#term-validation) and compiled-input [binding](../../../../glossary.md#term-binding). Open still checks every parameter pair; each operation still checks fixed bindings, storage, finiteness, and all compiled-input pairs involving an external input or the [cotangent seed](../../../../glossary.md#primal-adjoint-and-differentiation-seed), in original compiled-input order.
- Added tests for overlapping parameter slices at open, changed/dead parameter storage, dynamic input [aliases](../../../../glossary.md#aliasing-and-arena), and exact first-pair diagnostic ordering under multiple overlaps. Updated [lifecycle](../../../../glossary.md#lifecycle) Javadocs; the [API](../../../../glossary.md#term-application-programming-interface-api)/user guides and glossary need no 0001B change because public behavior and terminology are unchanged.
- Worker validation: targeted ordering test passed; `./gradlew :extensions:training:test` passed 35 tests with no failures; `./gradlew :extensions:training:javadoc` passed. Focused [CPU](../../../../glossary.md#term-cpu) Training integration passed four tests; two configured-[Metal](../../../../glossary.md#term-metal) cases skipped without `SYNAPTIK_METAL_TEST_LIBRARY`. `git diff --check` passed.
- Independent Class C review confirmed the fixed-segment proof, all dynamic-pair coverage and diagnostic order, and signed off after a [Javadoc](../../../../glossary.md#term-javadoc) correction and exact-order [regression](../../../../glossary.md#term-regression) assertion. No unresolved issue remains in this task.
