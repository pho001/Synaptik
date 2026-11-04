# Task 0001E: Demand-driven [optimizer](../../../../glossary.md#term-optimizer) candidate [buffers](../../../../glossary.md#term-buffer)

## Status

Complete

## Change class

Class B — changes the lasting [module](../../../../glossary.md#term-module)-local [optimizer](../../../../glossary.md#term-optimizer) staging [lifecycle](../../../../glossary.md#lifecycle) and resident memory use,
while preserving public training, [snapshot](../../../../glossary.md#term-snapshot), restore, and ordinary-[failure atomicity](../../../../glossary.md#term-failure-atomicity) semantics.

## Goal

Avoid retaining full-size inactive accumulation and [momentum](../../../../glossary.md#term-momentum) candidate arrays when a session does
not need them. Preserve the existing active [optimizer](../../../../glossary.md#term-optimizer) slots and exact [rollback](../../../../glossary.md#term-rollback) guarantee.

## [Scope](../../../../glossary.md#term-scope)

- Allocate accumulation candidates only for an accumulation operation or restore, and release the
  inactive candidate after a successful accumulation clear.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [release glossary](../../../../glossary.md#term-release))</span>
- Keep one active momentum slot per parameter. Allocate a momentum candidate when a nonzero
  configured momentum update or restore needs staging; do not retain an inactive candidate for
  exactly zero configured momentum.<span data-glossary-navigation="true"> ([momentum glossary](../../../../glossary.md#term-momentum); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- Preserve signed-zero state bytes during restore; a later successful momentum-disabled step must
  have the same momentum-slot bytes as before this change. Positive configured momentum that
  narrows to zero in FLOAT32 arithmetic must also keep the existing behavior.<span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>
- Keep parameter candidate and rollback arrays, low-precision master/rollback arrays, and the
  existing commit/rollback ordering unchanged.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback); [precision glossary](../../../../glossary.md#term-precision); [commit glossary](../../../../glossary.md#term-commit))</span>
- Update focused tests and documentation describing [buffer](../../../../glossary.md#term-buffer) allocation timing.

## Non-goals

- No public rollback toggle, optimizer-state format change, backend-resident optimizer, chunked
  snapshot representation, or relaxation of failure atomicity.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback); [optimizer glossary](../../../../glossary.md#term-optimizer); [backend glossary](../../../../glossary.md#backend); [snapshot glossary](../../../../glossary.md#term-snapshot); [failure atomicity glossary](../../../../glossary.md#term-failure-atomicity))</span>

## Contracts

- [`ARCHITECTURE.md` scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>,
  [module ownership](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>, and
  [dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span> route this module-local
  optimization without adding an Engine or concrete-backend edge.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend))</span>
- [Extensions/training — `extensions/training`](../../../../architecture/contracts/extensions-training.md#extensionstraining)
  owns optimizer algorithms and session state.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- [Extensions/training — optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  requires one active zero-initialized momentum slot per parameter, exact detached state, finite
  validation, and unchanged state on ordinary failure.<span data-glossary-navigation="true"> ([momentum glossary](../../../../glossary.md#term-momentum); [parameter glossary](../../../../glossary.md#term-parameter); [detached glossary](../../../../glossary.md#detached-model-results); [finite glossary](../../../../glossary.md#term-finite); [validation glossary](../../../../glossary.md#term-validation))</span>

If a selected contract is ambiguous, stop and request clarification before editing.

## Dependencies and integration

- Depends on: Training 0001D (Complete, pushed at `58dd0e30`).
- Conflicts with: concurrent Training [optimizer](../../../../glossary.md#term-optimizer) staging, [snapshot](../../../../glossary.md#term-snapshot), restore, or [lifecycle](../../../../glossary.md#lifecycle) changes.
- Parallel group: None.
- Common base revision: `58dd0e30` on `main`.
- Integration order: implementation, affected-[module](../../../../glossary.md#term-module) [validation](../../../../glossary.md#term-validation), independent targeted review.
- Integration validation: `./gradlew :extensions:training:test :extensions:training:javadoc`
  and `git diff --check`; no repository-wide rerun for a module-local staging optimization.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main coordination context.

The predecessor has its final result, validation, documentation review, and independent approval.
The implementation worktree began clean at the pushed common base; 0001E was the sole authorized
Training write frontier, and no conflicting writer was active.

## Files and symbols

- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingParameter.java` —
  candidate allocation, staging, commit, clear, restore, and snapshot behavior.<span data-glossary-navigation="true"> ([commit glossary](../../../../glossary.md#term-commit); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java` —
  optimizer commit handoff and allocation Javadoc.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [handoff glossary](../../../../glossary.md#term-handoff); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `extensions/training/src/test/java/io/github/pho001/synaptik/training/` — focused staging,
  state, rollback, and type coverage.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>
- `docs/api/training-api.md` — accurate allocation-[lifetime](../../../../glossary.md#term-lifetime) description.

## Acceptance criteria

- A fresh zero-momentum session keeps active gradient, accumulation, momentum, parameter candidate,
  and rollback storage, but has neither inactive candidate array. Reset-and-step does not add them.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [parameter glossary](../../../../glossary.md#term-parameter); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Accumulate, zeroGrad, accumulate-and-step, and restore produce the same bytes and counters as
  before; inactive accumulation staging is released after a successful clear.
- Positive configured [momentum](../../../../glossary.md#term-momentum), including a value that narrows to [FLOAT32](../../../../glossary.md#term-float32) zero, behaves as before.
- Restore preserves signed zeros and failed updates/restores preserve parameter, optimizer, and
  pending-gradient state; rollback buffers remain provisioned.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- FLOAT64, FLOAT32, BFLOAT16, and FLOAT16 are covered by automated tests. No per-element object
  allocation or dynamic dispatch is introduced in update loops.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [element glossary](../../../../glossary.md#term-element); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker validation: focused new and existing Training tests, then one final
`./gradlew :extensions:training:test :extensions:training:javadoc`; `git diff --check`.

Integration [owner](../../../../glossary.md#term-owner): reuse the successful [module](../../../../glossary.md#term-module) evidence after read-only review. No full-repository
[run](../../../../glossary.md#term-run) unless executable code or a cross-module contract changes during integration.

## Documentation and review impact

Independent targeted Class B review is required because staging allocation timing changes a
lasting [module](../../../../glossary.md#term-module)-local [lifecycle](../../../../glossary.md#lifecycle). It finalizes relevant Javadocs and the [API](../../../../glossary.md#term-application-programming-interface-api) allocation explanation,
checks the glossary for terminology drift, and does not rerun successful executable tests without
a specific reason.

## Result

`TrainingParameter` now creates inactive accumulation and [momentum](../../../../glossary.md#term-momentum) candidates only before an
[operation](../../../../glossary.md#operation) needs them. A successful clear or restore [releases](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) inactive accumulation staging;
exactly zero configured momentum retains no inactive candidate. Active slots, [parameter](../../../../glossary.md#term-parameter)
candidates, [rollback](../../../../glossary.md#term-rollback) storage, [snapshot](../../../../glossary.md#term-snapshot) bytes, and [commit](../../../../glossary.md#term-commit) order are unchanged. Restored [negative
zero](../../../../glossary.md#nan-infinity-and-signed-zero) remains exact until a successful zero-momentum step; ordinary zero-momentum steps add no
extra full-array pass. Positive configured momentum narrowed to [FLOAT32](../../../../glossary.md#term-float32) zero still swaps a staged
candidate. Failed restore staging is discarded without changing active state.

Changed files: `TrainingParameter.java`, `TrainingSession.java`, and `TrainingSessionTest.java`
under `extensions/training`, plus `docs/api/training-api.md`. Focused [Training session](../../../../glossary.md#training-session--trainingsession)/failure
tests passed. Final `./gradlew :extensions:training:test :extensions:training:javadoc` passed:
54 tests, zero failures/errors/skips. `git diff --check` passed. The integration [owner](../../../../glossary.md#term-owner)
cherry-picked the reviewed worker change without a main-worktree commit, verified all four files
byte-identical to the tested isolated worktree, and did not duplicate the [module](../../../../glossary.md#term-module) test [run](../../../../glossary.md#term-run).

Javadocs and the [API](../../../../glossary.md#term-application-programming-interface-api) allocation-[lifetime](../../../../glossary.md#term-lifetime) explanation were updated. Independent targeted Class B
review returned APPROVE with no P1/P2 findings, verified local links/anchors and relevant glossary
terms, and found no architecture or glossary change necessary. No unresolved issues; optional
rollback policy and chunked snapshots remain separate decisions. At idle after an ordinary
zero-momentum step, the two eliminated candidates save 16 bytes per [FLOAT64](../../../../glossary.md#term-float64) [element](../../../../glossary.md#term-element), 8 bytes per
FLOAT32 element, or 8 bytes per [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16) element of retained [optimizer](../../../../glossary.md#term-optimizer) arrays.

Status: Complete
