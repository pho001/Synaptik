# Task 0005: Training-owned detached state reconstruction

## Status

Complete

## Change class

Class B — minimal supported public construction of existing immutable Training values. No
ownership, dependency, persistence, concurrency, or restore-commit contract changes are planned.
Use a clean implementation context and an independent targeted API/documentation review context.

## Goal

Allow Java callers outside Training's package to reconstruct the existing detached `TrainingState`
from its exposed configuration, ordered schema, counters, and canonical payload bytes. This is
the Training-owned prerequisite at Checkpoint's existing 0005 gate, not a durable codec.

## Scope

- In `extensions/training`, package `io.github.pho001.synaptik.training`, make the existing full
  grouped `TrainingState` constructor and both defensive `TrainingState.ParameterState`
  payload constructors public. An empty group list covers ungrouped state; do not publish the
  legacy ungrouped overload. Existing optimizer and `ParameterGroup` constructors suffice.
- Preserve constructor structural checks and defensive list/array capture, exact raw bytes,
  signed zero, pending accumulation bits (including nonzero payload with zero pending count),
  ordered groups and within-group paths, and ordered parameter schema. Do not normalize values.
- Keep trusted `adoptFreshSnapshotArrays`, ownership dispatch, internal array accessors, and
  other package-private helpers internal. Preserve `state()`'s trusted adoption path.
- Preserve the distinction between structural construction and full session-specific validation
  in `TrainingSession.restore`; no new validator or strengthened constructor acceptance policy.

## Non-goals

No codec, file format, optional module/build edge, generic registry, new state type, owner
preflight API, transaction, caller-state product choice, or exact-durable continuation claim.
No optimizer mathematics, session lifecycle, hot-path, or cross-owner restore changes.
The user selected Model + TrainingState for Checkpoint 0006; external state remains application-
owned. Its guarantee is restoration of stored state, not complete automatic exact continuation.
0006 remains Draft pending concrete composition/codec/architecture design; none is implemented
here. Training's broader capability checkpoint 0005 is distinct.

## Contracts

Read [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts), then
these exact root headings and scoped destinations:

- Root [Java namespace](../../../../../ARCHITECTURE.md#java-namespace),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules), and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Scoped [`extensions/training`](../../../../architecture/contracts/extensions-training.md#extensionstraining)
  and [`extensions/checkpoint`](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint)
  — Training owns snapshot meaning; model-only Checkpoint has no Training dependency; the
  downstream adapter remains separately gated.
- Scoped [Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)
  — detached byte ownership, ordered compatibility, complete validation before mutation, and
  caller-owned private-candidate recovery remain unchanged.

No additive normative clarification is needed: the lifecycle's “No new public Training value or
API is required to capture” statement concerns existing capture, not external reconstruction.
If implementation exposes an applicable ambiguity or conflict, stop and report the exact heading;
do not edit normative contracts within this task.

## Dependencies and integration

- Depends on: Training 0004 and Checkpoint 0004A, both Complete; completed Training 0002A/0002B
  supply ordered groups and SGD/Adam/AdamW state transitively.
- Conflicts with: any edits to TrainingState construction/ownership, optimizer/group state
  compatibility, Training restore validation, or the same Training API/planning documents;
  Checkpoint 0006 and Training broader checkpoint 0005 must not overlap this scope.
- Parallel group: None (serial implementation, then independent review).
- Common base revision: `7a2a7337381e2ee8c7c09d8ff4829f9e5045dd01`;
  planning branch `work/training-state-reconstruction`.
- Integration order: completed Checkpoint 0004A → this Training-owned 0005 → independent
  review/documentation finalization → coordinator acceptance; do not launch 0006.
- Integration validation: reuse successful affected-Training test evidence, generate final Training
  Javadoc after review, validate changed links/anchors, and run `git diff --check`.
- Shared-document integration owner: Training/Checkpoint coordinator; serialize updates to this
  brief, both masters, and roadmap. Implementation owns Training API text/Javadoc until handoff;
  independent review finalizes them before coordinator closure.

Frontier verification (2026-10-14): the user authorized continuing 0005 after 0004A. The planning
worktree was clean on the recorded branch/base, with `main` and `origin/main` at that same revision.
Current predecessor Result sections record Complete status, focused validation, independent
Class C review, and configured repository test/Javadoc gates. Checkpoint 0004A resolves Training
0004's direct-save limitation; no unresolved prerequisite remains. No conflicting writer is
authorized in this serial packet. Recheck base/conflicts before execution if integration moves.

## Files and symbols

- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingState.java` —
  grouped constructor, two defensive nested payload constructors, affected type/constructor Javadoc.
- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java`
  (`restore`) and `TrainingParameter.java` (`validateState`) — read-only validation oracle.
- Existing focused tests in the Training package: `TrainingStateOwnershipTest`,
  `TrainingSessionTest` (grouped restore and zero-momentum bits), `AdamOptimizerTest`
  (adaptive slot rejection and detachment), `TrainingSessionFailureContractTest`.
- Add `extensions/training/src/test/java/io/github/pho001/synaptik/training/external/TrainingStateReconstructionTest.java`
  in `io.github.pho001.synaptik.training.external`; use only public APIs, not reflection or
  same-package helper bridges. No production type is added or moved.
- [Training API](../../../../api/training-api.md#atomicity-state-concurrency-and-close) —
  construction example and structural-versus-restore validation distinction; review the
  [continuation explanation](../../../../architecture/training-graph.md#continuing-at-the-next-training-input)
  for consistency without widening its claim.

## Acceptance criteria

- External-package Java compilation directly exercises the three public constructors and existing
  optimizer/group constructors. No legacy overload, trusted adoption, or internal mutable accessor
  becomes public. No new dependencies or Checkpoint production changes.
- Reconstruct snapshots for all four parameter types (FLOAT64, FLOAT32, BFLOAT16, FLOAT16) and
  SGD, Adam, AdamW defaults plus mixed ordered groups. Compare every configuration, counter,
  path/type/Shape, slot-presence flag, and payload byte; restore compatible reconstructions and
  compare fresh snapshots exactly. Include nonzero pending accumulation and signed-zero payloads.
- Mutate all supplied payload arrays (including adaptive second moment), group/path input lists,
  and parameter input lists after construction. Values remain unchanged; lists are immutable;
  every public payload view is independent, read-only, big-endian, and position-zero initially.
  Reconstructed values remain readable after session/Engine close.
- External tests exercise representative structural rejection (one required null, one wrong
  payload width, and one invalid counter). Preserve current size/overflow checks and acceptance
  boundaries; do not add finiteness or group membership preflight at construction.
- Add one external multi-parameter restore rejection: valid changed earlier entries and a
  structurally constructible invalid last entry. Assert every live parameter byte, slot, counter,
  and pending sum is unchanged. Reuse existing Training ownership/session/Adam/failure suites
  for exhaustive null, width, finiteness, order, and algorithm-specific slot rejection; do not
  duplicate their failure oracle in the new external test. Cover zero pending count with retained
  pending bits in the roundtrip, not another rejection matrix.
- Public Javadoc documents canonical bytes, each input/constraint/copy policy, failure conditions,
  and structural-only validation; do not imply construction proves session compatibility or
  durable exact resume. Independent review approves the complete code/test/documentation diff.

## Validation

Worker focused validation:

```bash
./gradlew :extensions:training:test --tests '*TrainingStateReconstructionTest' --tests '*TrainingStateOwnershipTest' --tests '*TrainingSessionTest' --tests '*AdamOptimizerTest' --tests '*TrainingSessionFailureContractTest' --console=plain
```

After executable code stabilizes, run the affected-module gate once:

```bash
./gradlew :extensions:training:check :extensions:training:javadoc --console=plain
git diff --check
```

Independent review reuses successful Java test evidence unless executable behavior changes or a
concrete risk warrants a rerun; generate `:extensions:training:javadoc` after final Javadoc edits.
Validate changed Markdown links/anchors and new-file whitespace. Repository-wide, backend-
conformance, and cross-module integration reruns are deferred to CI: this task changes one
Training API's accessibility, not dependencies, backend behavior, or cross-owner workflow.

## Documentation and review impact

Independent targeted review is mandatory for the public construction API. Follow
[documentation rules](../../../../developer-guide/documentation-rules.md), API/Javadoc and general
profiles for Training API text, and the planning profile for closure records. Add a short public
reconstruction example using the grouped constructor (empty groups if ungrouped); distinguish
structural construction from restore's full validation. Review affected Javadocs and search
glossary entries for Training session, exact continuation, and optimizer slots; update only if
meaning changes, otherwise record the reasoned no-change conclusion. No normative/ADR change.

## Result

- `TrainingState.java`: only the grouped constructor and two defensive `ParameterState`
  constructors became public; structural checks, exact bytes, and trusted adoption remain unchanged.
- `TrainingSession.java`: restore Javadoc only; both captured and reconstructed values receive
  complete session validation. Affected Javadocs were finalized by independent review.
- New `external/TrainingStateReconstructionTest.java`: four methods, 342 lines, covering all four
  floating types, SGD/Adam/AdamW and mixed groups, defensive ownership, retained pending bits,
  signed zero, structural rejection, and invalid-last-entry restore atomicity.
- `docs/api/training-api.md`: 69-line public reconstruction example and validation distinction.
- `./gradlew :extensions:training:test --tests '*TrainingStateReconstructionTest' --tests '*TrainingStateOwnershipTest' --tests '*TrainingSessionTest' --tests '*AdamOptimizerTest' --tests '*TrainingSessionFailureContractTest' --console=plain`: BUILD SUCCESSFUL (3s), 61 tests, zero failures/skips.
- `./gradlew :extensions:training:check :extensions:training:javadoc --console=plain`:
  BUILD SUCCESSFUL (2s), 72 tests, zero failures/skips; XML totals independently confirmed by
  coordinator and reviewer. Final `./gradlew :extensions:training:javadoc --console=plain`
  after reviewer edits: BUILD SUCCESSFUL (1s). Successful Java tests were reused, not repeated.
- Exact guide example compiled with Java 26 and executed for 12 dtype/algorithm inputs: PASS;
  13 local API links/anchors, rendered Javadoc crosslinks, new-file whitespace, and diff checks: PASS.
- Independent API/documentation review: APPROVE, no unresolved findings.
- Closure synchronizes this brief, both masters, and roadmap; 123 local planning links/anchors,
  whitespace checks, and `git diff --check` pass. Continuation explanation/glossary unchanged: ownership, meanings,
  and exact-continuation limitations remain accurate. No normative/ADR changes or unresolved issues.
- No successor authorized: 0006–0007 remain Draft pending design approval; user-selected first
  0006 scope is Model + TrainingState, external state application-owned, not complete exact resume.
- Status: Complete
