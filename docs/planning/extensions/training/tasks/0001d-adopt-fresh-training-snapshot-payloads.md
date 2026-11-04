# Task 0001D: Adopt Fresh Training [Snapshot](../../../../glossary.md#term-snapshot) Payloads

> **Editorial reading context — not a status update.** Trusted adoption is internal:
> TrainingParameter has just allocated and filled arrays that no external caller can retain,
> so ParameterState can take those arrays without a redundant clone. Public/caller-supplied
> array construction remains defensive. This removes a copy inside snapshot creation, not
> the snapshot's detached storage or its immutability. Returned read-only views still have
> independent movable cursors and survive session closure.
> Today's [snapshot ownership](../../../../reference/training/optimizers.md#atomicity-state-concurrency-and-close)
> and [external reconstruction example](../../../../reference/training/optimizers.md#example-reconstruct-a-detached-training-value)
> explain why an internal transfer and an external defensive copy have different obligations.

## Status

Complete

## Change class

Class B — changes the [module](../../../../glossary.md#term-module)-local [ownership](../../../../glossary.md#term-ownership-meanings) [handoff](../../../../glossary.md#term-handoff) for [immutable](../../../../glossary.md#term-immutable) [snapshot](../../../../glossary.md#term-snapshot) payloads without changing the public snapshot format or [lifecycle](../../../../glossary.md#lifecycle).

## Goal

Remove the four redundant full-size array clones made while `TrainingSession.state()` constructs each [detached](../../../../glossary.md#detached-model-results) `TrainingState.ParameterState`, while preserving its [immutable](../../../../glossary.md#term-immutable) public behavior and defensive construction from caller-supplied arrays.

## [Scope](../../../../glossary.md#term-scope)

- Let `TrainingParameter.snapshot()` transfer its four newly allocated, privately filled [canonical](../../../../glossary.md#term-canonical-form) [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) arrays directly to `TrainingState.ParameterState` through a narrowly named package-private owned-array construction path.
- Keep the existing package-private `ParameterState` constructor defensive for arrays that another caller may retain. Validate exact payload widths and non-null inputs on both paths.
- Do not expose [mutable](../../../../glossary.md#term-mutable) arrays publicly; public `ByteBuffer` [views](../../../../glossary.md#term-view-meanings) remain read-only, independent-position views. The [snapshot](../../../../glossary.md#term-snapshot) remains [detached](../../../../glossary.md#detached-model-results) from [Module](../../../../glossary.md#term-module), [Tensor](../../../../glossary.md#tensor), [Engine](../../../../glossary.md#term-engine), and [parameter](../../../../glossary.md#term-parameter) storage and valid after session close.
- Add focused tests for array [identity](../../../../glossary.md#term-identity-meanings) on the trusted internal path, defensive copies on the ordinary constructor, and existing state/snapshot behavior through a real session.

## Non-goals

- Changing the four-field [snapshot](../../../../glossary.md#term-snapshot) representation, serialized format, numerical bits, restore rules, native [parameter](../../../../glossary.md#term-parameter) storage, public [API](../../../../glossary.md#term-application-programming-interface-api), or persistent [optimizer](../../../../glossary.md#term-optimizer) staging/[rollback](../../../../glossary.md#term-rollback) [buffers](../../../../glossary.md#term-buffer).
- Removing the retained snapshot payloads themselves; this task eliminates only their redundant temporary clones.

## Contracts

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts) routes Training [lifecycle](../../../../glossary.md#lifecycle) [ownership](../../../../glossary.md#term-ownership-meanings) to the incorporated scoped contract; [module ownership](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span> remains unchanged.
- [Extensions/training — optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer))</span> requires [detached](../../../../glossary.md#detached-model-results) [canonical](../../../../glossary.md#term-canonical-form) [immutable](../../../../glossary.md#term-immutable) state and exact logical/master/[momentum](../../../../glossary.md#term-momentum)/accumulation widths. The owned-array [handoff](../../../../glossary.md#term-handoff) must preserve that public contract.

No architecture change is needed. If a selected contract requires copying even privately created arrays, stop and report the conflict before editing.

## Dependencies and integration

- Depends on: Training 0001C (Complete and pushed at `19dab821`); Training 0001 [snapshot](../../../../glossary.md#term-snapshot)/state [lifecycle](../../../../glossary.md#lifecycle) (Complete).
- Conflicts with: concurrent changes to `TrainingParameter.snapshot()`, `TrainingState.ParameterState`, or Training state/restore semantics.
- Parallel group: None.
- Common base revision: `19dab821` on `main`.
- Integration order: implementation, focused [validation](../../../../glossary.md#term-validation), independent targeted [ownership](../../../../glossary.md#term-ownership-meanings) review.
- Integration validation: `./gradlew :extensions:training:test :extensions:training:javadoc` and `git diff --check`; no repository-wide rerun because no [module](../../../../glossary.md#term-module) edge or cross-module behavior changes.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main coordination context.

The predecessors have final results, validation, and review; the worktree is clean at the pushed common base. This is the sole authorized Training write frontier and no conflicting writer is active.

## Files and symbols

- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingParameter.java` — `snapshot()` owned [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) [handoff](../../../../glossary.md#term-handoff).
- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingState.java` — `ParameterState` exact-size [validation](../../../../glossary.md#term-validation) and private [ownership](../../../../glossary.md#term-ownership-meanings) path.
- `extensions/training/src/test/java/io/github/pho001/synaptik/training/` — targeted ownership and session-state tests.

## Acceptance criteria

- The internal [snapshot](../../../../glossary.md#term-snapshot) path retains the exact four freshly filled arrays, rather than cloning any of them; no caller retains those arrays or can mutate them through public accessors.
- The ordinary package-private constructor still clones all four supplied arrays, so later mutation of its [arguments](../../../../glossary.md#term-argument) does not change the `ParameterState`.
- Public [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) bytes, widths, read-only [views](../../../../glossary.md#term-view-meanings), state detachment, restore [validation](../../../../glossary.md#term-validation), and post-close readability are unchanged for [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), [BFLOAT16](../../../../glossary.md#term-bfloat16), and [FLOAT16](../../../../glossary.md#term-float16).
- Array adoption introduces no extra per-[element](../../../../glossary.md#term-element) [dispatch](../../../../glossary.md#term-dispatch), boxing, or allocation in the snapshot fill loops.

## [Validation](../../../../glossary.md#term-validation)

Worker: focused new tests, then one `./gradlew :extensions:training:test :extensions:training:javadoc` after executable code stabilizes; `git diff --check`.

Integration [owner](../../../../glossary.md#term-owner): reuse the successful [module](../../../../glossary.md#term-module) evidence if review is read-only. No full-repository rerun is required for this module-internal optimization.

## Documentation and review impact

Independent targeted Class B review is required because the [immutable](../../../../glossary.md#term-immutable)-state array [ownership](../../../../glossary.md#term-ownership-meanings) [handoff](../../../../glossary.md#term-handoff) changes. Update [Javadoc](../../../../glossary.md#term-javadoc) on affected implementation contracts. Review the [API](../../../../glossary.md#term-application-programming-interface-api)/user guide and glossary for drift; if no public behavior or term changes, record `Documentation impact: none; public snapshot contract and workflow are unchanged`.

## Result

`TrainingParameter.snapshot()` now transfers its four freshly filled [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) arrays to
`TrainingState.ParameterState.adoptFreshSnapshotArrays(...)` without cloning. The ordinary
package-private constructor retains exact-width [validation](../../../../glossary.md#term-validation) and defensive copies. Public
read-only [views](../../../../glossary.md#term-view-meanings) and [detached](../../../../glossary.md#detached-model-results) state behavior are unchanged. Changed files:
`TrainingParameter.java`, `TrainingState.java`, `TrainingStateOwnershipTest.java`, and
`TrainingSessionTest.java` under `extensions/training`.

The worker passed focused tests and `./gradlew :extensions:training:test :extensions:training:javadoc`
(50 tests, 0 skipped/failures). The integration [owner](../../../../glossary.md#term-owner) verified the
four files were byte-identical after worktree integration and `git diff --check` passed.
Independent targeted Class B review returned APPROVE with no findings; it did not repeat tests.
Javadocs were updated. Documentation impact: none; [API](../../../../glossary.md#term-application-programming-interface-api)/user guides and glossary still describe
the same public [snapshot](../../../../glossary.md#term-snapshot) contract and workflow.

This removes temporary clone allocation of 32 bytes per [FLOAT64](../../../../glossary.md#term-float64) [element](../../../../glossary.md#term-element), 16 per [FLOAT32](../../../../glossary.md#term-float32) element,
or 14 per [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16) element during `state()`. The four retained snapshot payloads and
persistent [optimizer](../../../../glossary.md#term-optimizer) [buffers](../../../../glossary.md#term-buffer) remain unchanged. No unresolved issues; persistent-buffer policy
is a separate follow-up.

Status: Complete
