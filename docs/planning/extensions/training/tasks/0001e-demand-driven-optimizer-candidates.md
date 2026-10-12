# Task 0001E: Demand-driven optimizer candidate buffers

## Status

Complete

## Change class

Class B — changes the lasting module-local optimizer staging lifecycle and resident memory use,
while preserving public training, snapshot, restore, and ordinary-failure atomicity semantics.

## Goal

Avoid retaining full-size inactive accumulation and momentum candidate arrays when a session does
not need them. Preserve the existing active optimizer slots and exact rollback guarantee.

## Scope

- Allocate accumulation candidates only for an accumulation operation or restore, and release the
  inactive candidate after a successful accumulation clear.
- Keep one active momentum slot per parameter. Allocate a momentum candidate when a nonzero
  configured momentum update or restore needs staging; do not retain an inactive candidate for
  exactly zero configured momentum.
- Preserve signed-zero state bytes during restore; a later successful momentum-disabled step must
  have the same momentum-slot bytes as before this change. Positive configured momentum that
  narrows to zero in FLOAT32 arithmetic must also keep the existing behavior.
- Keep parameter candidate and rollback arrays, low-precision master/rollback arrays, and the
  existing commit/rollback ordering unchanged.
- Update focused tests and documentation describing buffer allocation timing.

## Non-goals

- No public rollback toggle, optimizer-state format change, backend-resident optimizer, chunked
  snapshot representation, or relaxation of failure atomicity.

## Contracts

- [`ARCHITECTURE.md` scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
  [module ownership](../../../../../ARCHITECTURE.md#module-ownership-routing), and
  [dependency rules](../../../../../ARCHITECTURE.md#dependency-rules) route this module-local
  optimization without adding an Engine or concrete-backend edge.
- [Extensions/training — `extensions/training`](../../../../architecture/contracts/extensions-training.md#extensionstraining)
  owns optimizer algorithms and session state.
- [Extensions/training — optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)
  requires one active zero-initialized momentum slot per parameter, exact detached state, finite
  validation, and unchanged state on ordinary failure.

If a selected contract is ambiguous, stop and request clarification before editing.

## Dependencies and integration

- Depends on: Training 0001D (Complete, pushed at `58dd0e30`).
- Conflicts with: concurrent Training optimizer staging, snapshot, restore, or lifecycle changes.
- Parallel group: None.
- Common base revision: `58dd0e30` on `main`.
- Integration order: implementation, affected-module validation, independent targeted review.
- Integration validation: `./gradlew :extensions:training:test :extensions:training:javadoc`
  and `git diff --check`; no repository-wide rerun for a module-local staging optimization.
- Shared-document integration owner: main coordination context.

The predecessor has its final result, validation, documentation review, and independent approval.
The implementation worktree began clean at the pushed common base; 0001E was the sole authorized
Training write frontier, and no conflicting writer was active.

## Files and symbols

- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingParameter.java` —
  candidate allocation, staging, commit, clear, restore, and snapshot behavior.
- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java` —
  optimizer commit handoff and allocation Javadoc.
- `extensions/training/src/test/java/io/github/pho001/synaptik/training/` — focused staging,
  state, rollback, and type coverage.
- `docs/api/training-api.md` — accurate allocation-lifetime description.

## Acceptance criteria

- A fresh zero-momentum session keeps active gradient, accumulation, momentum, parameter candidate,
  and rollback storage, but has neither inactive candidate array. Reset-and-step does not add them.
- Accumulate, zeroGrad, accumulate-and-step, and restore produce the same bytes and counters as
  before; inactive accumulation staging is released after a successful clear.
- Positive configured momentum, including a value that narrows to FLOAT32 zero, behaves as before.
- Restore preserves signed zeros and failed updates/restores preserve parameter, optimizer, and
  pending-gradient state; rollback buffers remain provisioned.
- FLOAT64, FLOAT32, BFLOAT16, and FLOAT16 are covered by automated tests. No per-element object
  allocation or dynamic dispatch is introduced in update loops.

## Validation

Worker validation: focused new and existing Training tests, then one final
`./gradlew :extensions:training:test :extensions:training:javadoc`; `git diff --check`.

Integration owner: reuse the successful module evidence after read-only review. No full-repository
run unless executable code or a cross-module contract changes during integration.

## Documentation and review impact

Independent targeted Class B review is required because staging allocation timing changes a
lasting module-local lifecycle. It finalizes relevant Javadocs and the API allocation explanation,
checks the glossary for terminology drift, and does not rerun successful executable tests without
a specific reason.

## Result

`TrainingParameter` now creates inactive accumulation and momentum candidates only before an
operation needs them. A successful clear or restore releases inactive accumulation staging;
exactly zero configured momentum retains no inactive candidate. Active slots, parameter
candidates, rollback storage, snapshot bytes, and commit order are unchanged. Restored negative
zero remains exact until a successful zero-momentum step; ordinary zero-momentum steps add no
extra full-array pass. Positive configured momentum narrowed to FLOAT32 zero still swaps a staged
candidate. Failed restore staging is discarded without changing active state.

Changed files: `TrainingParameter.java`, `TrainingSession.java`, and `TrainingSessionTest.java`
under `extensions/training`, plus `docs/api/training-api.md`. Focused Training session/failure
tests passed. Final `./gradlew :extensions:training:test :extensions:training:javadoc` passed:
54 tests, zero failures/errors/skips. `git diff --check` passed. The integration owner
cherry-picked the reviewed worker change without a main-worktree commit, verified all four files
byte-identical to the tested isolated worktree, and did not duplicate the module test run.

Javadocs and the API allocation-lifetime explanation were updated. Independent targeted Class B
review returned APPROVE with no P1/P2 findings, verified local links/anchors and relevant glossary
terms, and found no architecture or glossary change necessary. No unresolved issues; optional
rollback policy and chunked snapshots remain separate decisions. At idle after an ordinary
zero-momentum step, the two eliminated candidates save 16 bytes per FLOAT64 element, 8 bytes per
FLOAT32 element, or 8 bytes per BFLOAT16/FLOAT16 element of retained optimizer arrays.

Status: Complete
