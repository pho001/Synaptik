# Task 0005: Training-owned [detached](../../../../glossary.md#detached-model-results) state reconstruction

## Status

Complete

## Change class

Class B — minimal supported public construction of existing [immutable](../../../../glossary.md#term-immutable) Training values. No
[ownership](../../../../glossary.md#term-ownership-meanings), [dependency](../../../../glossary.md#term-dependency), [persistence](../../../../glossary.md#term-persistence), [concurrency](../../../../glossary.md#term-concurrency), or restore-[commit](../../../../glossary.md#term-commit) contract changes are planned.
Use a clean implementation context and an independent targeted [API](../../../../glossary.md#term-application-programming-interface-api)/documentation review context.

## Goal

Allow Java callers outside Training's package to reconstruct the existing [detached](../../../../glossary.md#detached-model-results) `TrainingState`
from its exposed configuration, ordered [schema](../../../../glossary.md#term-schema), counters, and [canonical](../../../../glossary.md#term-canonical-form) [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) bytes. This is
the Training-owned prerequisite at [Checkpoint](../../../../glossary.md#term-checkpoint)'s existing 0005 gate, not a durable [codec](../../../../glossary.md#term-codec).

## [Scope](../../../../glossary.md#term-scope)

- In `extensions/training`, package `io.github.pho001.synaptik.training`, make the existing full
  grouped `TrainingState` constructor and both defensive `TrainingState.ParameterState`
  payload constructors public. An empty group list covers ungrouped state; do not publish the
  legacy ungrouped overload. Existing optimizer and <span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>[`ParameterGroup`](../../../../glossary.md#parameter-group--parametergroup) constructors suffice.
- Preserve constructor structural checks and defensive list/array capture, exact raw bytes,
  signed zero, pending accumulation bits (including nonzero payload with zero pending count),
  ordered groups and within-group paths, and ordered parameter schema. Do not normalize values.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [signed zero glossary](../../../../glossary.md#term-signed-zero); [parameter glossary](../../../../glossary.md#term-parameter); [schema glossary](../../../../glossary.md#term-schema))</span>
- Keep trusted `adoptFreshSnapshotArrays`, ownership dispatch, internal array accessors, and
  other package-private helpers internal. Preserve <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>`state()`'s trusted adoption path.
- Preserve the distinction between structural construction and full session-specific validation
  in <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`TrainingSession.restore`; no new validator or strengthened constructor acceptance policy.

## Non-goals

No [codec](../../../../glossary.md#term-codec), file format, optional [module](../../../../glossary.md#term-module)/build edge, generic [registry](../../../../glossary.md#term-registry), new state type, [owner](../../../../glossary.md#term-owner)
[preflight](../../../../glossary.md#term-preflight) [API](../../../../glossary.md#term-application-programming-interface-api), [transaction](../../../../glossary.md#term-transaction), caller-state product choice, or exact-durable continuation claim.
No [optimizer](../../../../glossary.md#term-optimizer) mathematics, session [lifecycle](../../../../glossary.md#lifecycle), hot-path, or cross-owner restore changes.
The user selected [Model](../../../../glossary.md#term-model) + TrainingState for [Checkpoint](../../../../glossary.md#term-checkpoint) 0006; external state remains application-
owned. Its guarantee is restoration of stored state, not complete automatic exact continuation.
0006 remains Draft pending concrete composition/codec/architecture design; none is implemented
here. Training's broader capability checkpoint 0005 is distinct.

## Contracts

Read [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>, then
these exact root headings and scoped destinations:

- Root [Java namespace](../../../../../ARCHITECTURE.md#java-namespace),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>,
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>, and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Scoped [`extensions/training`](../../../../architecture/contracts/extensions-training.md#extensionstraining)
  and [`extensions/checkpoint`](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint)
  — Training owns snapshot meaning; model-only Checkpoint has no Training dependency; the
  downstream adapter remains separately gated.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [model glossary](../../../../glossary.md#term-model); [Checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Scoped [Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([Optimizer glossary](../../../../glossary.md#term-optimizer))</span>
  — detached byte ownership, ordered compatibility, complete validation before mutation, and
  caller-owned private-candidate recovery remain unchanged.<span data-glossary-navigation="true"> ([detached glossary](../../../../glossary.md#detached-model-results); [validation glossary](../../../../glossary.md#term-validation))</span>

No additive [normative](../../../../glossary.md#term-normative) clarification is needed: the [lifecycle](../../../../glossary.md#lifecycle)'s “No new public Training value or
[API](../../../../glossary.md#term-application-programming-interface-api) is required to [capture](../../../../glossary.md#term-capture-meanings)” statement concerns existing capture, not external reconstruction.
If implementation exposes an applicable ambiguity or conflict, stop and report the exact heading;
do not edit normative contracts within this task.

## Dependencies and integration

- Depends on: Training 0004 and Checkpoint 0004A, both Complete; completed Training 0002A/0002B
  supply ordered groups and SGD/Adam/AdamW state transitively.<span data-glossary-navigation="true"> ([SGD glossary](../../../../glossary.md#term-stochastic-gradient-descent-sgd); [Adam glossary](../../../../glossary.md#adaptive-optimizer--adam-adamw); [AdamW glossary](../../../../glossary.md#adaptive-optimizer--adam-adamw))</span>
- Conflicts with: any edits to TrainingState construction/ownership, optimizer/group state
  compatibility, Training restore validation, or the same Training API/planning documents;
  Checkpoint 0006 and Training broader checkpoint 0005 must not overlap this scope.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [optimizer glossary](../../../../glossary.md#term-optimizer); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [scope glossary](../../../../glossary.md#term-scope))</span>
- Parallel group: None (serial implementation, then independent review).
- Common base revision: `7a2a7337381e2ee8c7c09d8ff4829f9e5045dd01`;
  planning branch `work/training-state-reconstruction`.
- Integration order: completed Checkpoint 0004A → this Training-owned 0005 → independent
  review/documentation finalization → coordinator acceptance; do not launch 0006.
- Integration validation: reuse successful affected-Training test evidence, generate final Training
  Javadoc after review, validate changed links/anchors, and run <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>`git diff --check`.
- Shared-document integration owner: Training/Checkpoint coordinator; serialize updates to this
  brief, both masters, and roadmap. Implementation owns Training API text/Javadoc until handoff;
  independent review finalizes them before coordinator closure.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [handoff glossary](../../../../glossary.md#term-handoff))</span>

Frontier verification (2026-10-14): the user authorized continuing 0005 after 0004A. The [planning](../../../../glossary.md#term-planning)
worktree was clean on the recorded branch/base, with `main` and `origin/main` at that same revision.
Current predecessor Result sections record Complete status, focused [validation](../../../../glossary.md#term-validation), independent
Class C review, and configured repository test/[Javadoc](../../../../glossary.md#term-javadoc) gates. [Checkpoint](../../../../glossary.md#term-checkpoint) 0004A resolves Training
0004's direct-save limitation; no unresolved prerequisite remains. No conflicting writer is
authorized in this serial packet. Recheck base/conflicts before execution if integration moves.

## Files and symbols

- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingState.java` —
  grouped constructor, two defensive nested payload constructors, affected type/constructor Javadoc.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java`
  (`restore`) and `TrainingParameter.java` (`validateState`) — read-only [validation](../../../../glossary.md#term-validation) [oracle](../../../../glossary.md#term-test-oracle).
- Existing focused tests in the Training package: `TrainingStateOwnershipTest`,
  `TrainingSessionTest` (grouped restore and zero-[momentum](../../../../glossary.md#term-momentum) bits), `AdamOptimizerTest`
  (adaptive slot rejection and detachment), `TrainingSessionFailureContractTest`.
- Add `extensions/training/src/test/java/io/github/pho001/synaptik/training/external/TrainingStateReconstructionTest.java`
  in `io.github.pho001.synaptik.training.external`; use only public APIs, not reflection or
  same-package helper bridges. No production type is added or moved.
- [Training API](../../../../api/training-api.md#atomicity-state-concurrency-and-close)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span> —
  construction example and structural-versus-restore validation distinction; review the
  [continuation explanation](../../../../architecture/training-graph.md#continuing-at-the-next-training-input)
  for consistency without widening its claim.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening))</span>

## Acceptance criteria

- External-package Java compilation directly exercises the three public constructors and existing
  optimizer/group constructors. No legacy overload, trusted adoption, or internal mutable accessor
  becomes public. No new dependencies or Checkpoint production changes.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [optimizer glossary](../../../../glossary.md#term-optimizer); [mutable glossary](../../../../glossary.md#term-mutable); [Checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Reconstruct snapshots for all four parameter types (FLOAT64, FLOAT32, BFLOAT16, FLOAT16) and
  SGD, Adam, AdamW defaults plus mixed ordered groups. Compare every configuration, counter,
  path/type/Shape, slot-presence flag, and payload byte; restore compatible reconstructions and
  compare fresh snapshots exactly. Include nonzero pending accumulation and signed-zero payloads.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [SGD glossary](../../../../glossary.md#term-stochastic-gradient-descent-sgd); [Adam, AdamW glossary](../../../../glossary.md#adaptive-optimizer--adam-adamw); [Shape glossary](../../../../glossary.md#shape); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Mutate all supplied payload arrays (including adaptive second moment), group/path input lists,
  and parameter input lists after construction. Values remain unchanged; lists are immutable;
  every public payload view is independent, read-only, big-endian, and position-zero initially.
  Reconstructed values remain readable after session/Engine close.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [view glossary](../../../../glossary.md#term-view-meanings); [big-endian glossary](../../../../glossary.md#text-encoding-and-file-boundaries); [Engine glossary](../../../../glossary.md#term-engine))</span>
- External tests exercise representative structural rejection (one required null, one wrong
  payload width, and one invalid counter). Preserve current size/overflow checks and acceptance
  boundaries; do not add finiteness or group membership preflight at construction.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [preflight glossary](../../../../glossary.md#term-preflight))</span>
- Add one external multi-parameter restore rejection: valid changed earlier entries and a
  structurally constructible invalid last entry. Assert every live parameter byte, slot, counter,
  and pending sum is unchanged. Reuse existing Training ownership/session/Adam/failure suites
  for exhaustive null, width, finiteness, order, and algorithm-specific slot rejection; do not
  duplicate their failure oracle in the new external test. Cover zero pending count with retained
  pending bits in the roundtrip, not another rejection matrix.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [Adam glossary](../../../../glossary.md#adaptive-optimizer--adam-adamw); [oracle glossary](../../../../glossary.md#term-test-oracle); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Public Javadoc documents canonical bytes, each input/constraint/copy policy, failure conditions,
  and structural-only validation; do not imply construction proves session compatibility or
  durable exact resume. Independent review approves the complete code/test/documentation diff.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [canonical glossary](../../../../glossary.md#term-canonical-form); [validation glossary](../../../../glossary.md#term-validation); [resume glossary](../../../../glossary.md#schema-serialization-and-checkpoint))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker focused validation:

```bash
./gradlew :extensions:training:test --tests '*TrainingStateReconstructionTest' --tests '*TrainingStateOwnershipTest' --tests '*TrainingSessionTest' --tests '*AdamOptimizerTest' --tests '*TrainingSessionFailureContractTest' --console=plain
```

After executable code stabilizes, [run](../../../../glossary.md#term-run) the affected-[module](../../../../glossary.md#term-module) gate once:

```bash
./gradlew :extensions:training:check :extensions:training:javadoc --console=plain
git diff --check
```

Independent review reuses successful Java test evidence unless executable behavior changes or a
concrete risk warrants a rerun; generate `:extensions:training:javadoc` after final [Javadoc](../../../../glossary.md#term-javadoc) edits.
Validate changed Markdown links/anchors and new-file whitespace. Repository-wide, [backend](../../../../glossary.md#backend)-
[conformance](../../../../glossary.md#term-conformance), and cross-module integration reruns are deferred to [CI](../../../../glossary.md#term-continuous-integration-ci): this task changes one
Training [API](../../../../glossary.md#term-application-programming-interface-api)'s accessibility, not dependencies, backend behavior, or cross-[owner](../../../../glossary.md#term-owner) workflow.

## Documentation and review impact

Independent targeted review is mandatory for the public construction [API](../../../../glossary.md#term-application-programming-interface-api). Follow
[documentation rules](../../../../developer-guide/documentation-rules.md), API/[Javadoc](../../../../glossary.md#term-javadoc) and general
profiles for Training API text, and the [planning](../../../../glossary.md#term-planning) profile for closure records. Add a short public
reconstruction example using the grouped constructor (empty groups if ungrouped); distinguish
structural construction from restore's full [validation](../../../../glossary.md#term-validation). Review affected Javadocs and search
glossary entries for [Training session](../../../../glossary.md#training-session--trainingsession), exact continuation, and [optimizer](../../../../glossary.md#term-optimizer) slots; update only if
meaning changes, otherwise record the reasoned no-change conclusion. No [normative](../../../../glossary.md#term-normative)/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) change.

## Result

- `TrainingState.java`: only the grouped constructor and two defensive `ParameterState`
  constructors became public; structural checks, exact bytes, and trusted adoption remain unchanged.
- `TrainingSession.java`: restore Javadoc only; both captured and reconstructed values receive
  complete session validation. Affected Javadocs were finalized by independent review.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- New `external/TrainingStateReconstructionTest.java`: four methods, 342 lines, covering all four
  floating types, SGD/Adam/AdamW and mixed groups, defensive ownership, retained pending bits,
  signed zero, structural rejection, and invalid-last-entry restore atomicity.<span data-glossary-navigation="true"> ([SGD glossary](../../../../glossary.md#term-stochastic-gradient-descent-sgd); [Adam glossary](../../../../glossary.md#adaptive-optimizer--adam-adamw); [AdamW glossary](../../../../glossary.md#adaptive-optimizer--adam-adamw); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [signed zero glossary](../../../../glossary.md#term-signed-zero); [atomicity glossary](../../../../glossary.md#term-atomicity))</span>
- `docs/api/training-api.md`: 69-line public reconstruction example and [validation](../../../../glossary.md#term-validation) distinction.
- `./gradlew :extensions:training:test --tests '*TrainingStateReconstructionTest' --tests '*TrainingStateOwnershipTest' --tests '*TrainingSessionTest' --tests '*AdamOptimizerTest' --tests '*TrainingSessionFailureContractTest' --console=plain`: BUILD SUCCESSFUL (3s), 61 tests, zero failures/skips.
- `./gradlew :extensions:training:check :extensions:training:javadoc --console=plain`:
  BUILD SUCCESSFUL (2s), 72 tests, zero failures/skips; XML totals independently confirmed by
  coordinator and reviewer. Final `./gradlew :extensions:training:javadoc --console=plain`
  after reviewer edits: BUILD SUCCESSFUL (1s). Successful Java tests were reused, not repeated.
- Exact guide example compiled with Java 26 and executed for 12 dtype/algorithm inputs: PASS;
  13 local API links/anchors, rendered Javadoc crosslinks, new-file whitespace, and diff checks: PASS.<span data-glossary-navigation="true"> ([dtype glossary](../../../../glossary.md#data-type--datatype))</span>
- Independent [API](../../../../glossary.md#term-application-programming-interface-api)/documentation review: APPROVE, no unresolved findings.
- Closure synchronizes this brief, both masters, and roadmap; 123 local planning links/anchors,
  whitespace checks, and <span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>`git diff --check` pass. Continuation explanation/glossary unchanged: ownership, meanings,
  and exact-continuation limitations remain accurate. No normative/ADR changes or unresolved issues.<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- No successor authorized: 0006–0007 remain Draft pending design approval; user-selected first
  0006 scope is Model + TrainingState, external state application-owned, not complete exact resume.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [Model glossary](../../../../glossary.md#term-model); [resume glossary](../../../../glossary.md#schema-serialization-and-checkpoint))</span>
- Status: Complete
