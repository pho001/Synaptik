# Task 0001D: Adopt Fresh Training Snapshot Payloads

## Status

Complete

## Change class

Class B — changes the module-local ownership handoff for immutable snapshot payloads without changing the public snapshot format or lifecycle.

## Goal

Remove the four redundant full-size array clones made while `TrainingSession.state()` constructs each detached `TrainingState.ParameterState`, while preserving its immutable public behavior and defensive construction from caller-supplied arrays.

## Scope

- Let `TrainingParameter.snapshot()` transfer its four newly allocated, privately filled canonical payload arrays directly to `TrainingState.ParameterState` through a narrowly named package-private owned-array construction path.
- Keep the existing package-private `ParameterState` constructor defensive for arrays that another caller may retain. Validate exact payload widths and non-null inputs on both paths.
- Do not expose mutable arrays publicly; public `ByteBuffer` views remain read-only, independent-position views. The snapshot remains detached from Module, Tensor, Engine, and parameter storage and valid after session close.
- Add focused tests for array identity on the trusted internal path, defensive copies on the ordinary constructor, and existing state/snapshot behavior through a real session.

## Non-goals

- Changing the four-field snapshot representation, serialized format, numerical bits, restore rules, native parameter storage, public API, or persistent optimizer staging/rollback buffers.
- Removing the retained snapshot payloads themselves; this task eliminates only their redundant temporary clones.

## Contracts

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts) routes Training lifecycle ownership to the incorporated scoped contract; [module ownership](../../../../../ARCHITECTURE.md#module-ownership-routing) remains unchanged.
- [Extensions/training — optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle) requires detached canonical immutable state and exact logical/master/momentum/accumulation widths. The owned-array handoff must preserve that public contract.

No architecture change is needed. If a selected contract requires copying even privately created arrays, stop and report the conflict before editing.

## Dependencies and integration

- Depends on: Training 0001C (Complete and pushed at `19dab821`); Training 0001 snapshot/state lifecycle (Complete).
- Conflicts with: concurrent changes to `TrainingParameter.snapshot()`, `TrainingState.ParameterState`, or Training state/restore semantics.
- Parallel group: None.
- Common base revision: `19dab821` on `main`.
- Integration order: implementation, focused validation, independent targeted ownership review.
- Integration validation: `./gradlew :extensions:training:test :extensions:training:javadoc` and `git diff --check`; no repository-wide rerun because no module edge or cross-module behavior changes.
- Shared-document integration owner: main coordination context.

The predecessors have final results, validation, and review; the worktree is clean at the pushed common base. This is the sole authorized Training write frontier and no conflicting writer is active.

## Files and symbols

- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingParameter.java` — `snapshot()` owned payload handoff.
- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingState.java` — `ParameterState` exact-size validation and private ownership path.
- `extensions/training/src/test/java/io/github/pho001/synaptik/training/` — targeted ownership and session-state tests.

## Acceptance criteria

- The internal snapshot path retains the exact four freshly filled arrays, rather than cloning any of them; no caller retains those arrays or can mutate them through public accessors.
- The ordinary package-private constructor still clones all four supplied arrays, so later mutation of its arguments does not change the `ParameterState`.
- Public payload bytes, widths, read-only views, state detachment, restore validation, and post-close readability are unchanged for FLOAT64, FLOAT32, BFLOAT16, and FLOAT16.
- Array adoption introduces no extra per-element dispatch, boxing, or allocation in the snapshot fill loops.

## Validation

Worker: focused new tests, then one `./gradlew :extensions:training:test :extensions:training:javadoc` after executable code stabilizes; `git diff --check`.

Integration owner: reuse the successful module evidence if review is read-only. No full-repository rerun is required for this module-internal optimization.

## Documentation and review impact

Independent targeted Class B review is required because the immutable-state array ownership handoff changes. Update Javadoc on affected implementation contracts. Review the API/user guide and glossary for drift; if no public behavior or term changes, record `Documentation impact: none; public snapshot contract and workflow are unchanged`.

## Result

`TrainingParameter.snapshot()` now transfers its four freshly filled payload arrays to
`TrainingState.ParameterState.adoptFreshSnapshotArrays(...)` without cloning. The ordinary
package-private constructor retains exact-width validation and defensive copies. Public
read-only views and detached state behavior are unchanged. Changed files:
`TrainingParameter.java`, `TrainingState.java`, `TrainingStateOwnershipTest.java`, and
`TrainingSessionTest.java` under `extensions/training`.

The worker passed focused tests and `./gradlew :extensions:training:test :extensions:training:javadoc`
(50 tests, 0 skipped/failures). The integration owner verified the
four files were byte-identical after worktree integration and `git diff --check` passed.
Independent targeted Class B review returned APPROVE with no findings; it did not repeat tests.
Javadocs were updated. Documentation impact: none; API/user guides and glossary still describe
the same public snapshot contract and workflow.

This removes temporary clone allocation of 32 bytes per FLOAT64 element, 16 per FLOAT32 element,
or 14 per BFLOAT16/FLOAT16 element during `state()`. The four retained snapshot payloads and
persistent optimizer buffers remain unchanged. No unresolved issues; persistent-buffer policy
is a separate follow-up.

Status: Complete
