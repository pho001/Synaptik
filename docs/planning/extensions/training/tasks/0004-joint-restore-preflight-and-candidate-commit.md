# Task 0004: Joint restore on an unpublished candidate

## Status

Complete — the bounded caller-coordinated, unpublished-candidate protocol in
[ADR 0035](../../../../design/decisions/0035-private-candidate-training-restore.md) has executable
integration evidence and independent Class C review. This is not durable Training persistence.

## Change class

Class C — restore crosses NN binding/mode ownership, Training's fixed-session lifecycle,
caller-owned continuation state, and the future Checkpoint adapter boundary. Require clean
implementation and independent targeted documentation/review contexts.

## Goal

Prove a bounded, executable caller-coordinated in-memory recovery protocol using the existing
public NN, Checkpoint, Training, and Engine APIs. A failed private replacement must not publish
partly restored NN, Training, or typed caller state. Checkpoint 0005–0006 may consume the proven
protocol later; no durable training artifact or new production API is required by this task.

## Scope

- Add an integration-test caller fixture that captures one complete, quiescent successful-run
  boundary and owns exclusion, a distinct private fully bound NN tree, a private Training session,
  a shareable native parameter Arena, and a typed staged next-input/progress/external-state value.
  The model artifact supplies detached NN values; `TrainingState` and caller values stay in memory.
- Exercise `ModelCheckpoint.load` on the private tree (independent of any earlier `inspect`),
  restore per-module modes (parent first, then child overrides for mixed modes), call
  `TrainingParameterStorage.copyToSharedNative`, reconstruct the
  objective from final Tensors and recorded `ForwardContext` values, then use
  `TrainingSession.open`, `state()`, and `restore`. Compare canonical candidate logical parameter
  bytes with `TrainingState` at every stable path before restore and publication.
- Publish one caller-owned typed run-holder only after candidate and external-state validation
  succeeds under exclusion. Retire the quiescent old session at handoff; on ordinary failure,
  close/discard the private session and Arena without publishing the candidate. The old session
  may already be closed if retirement fails. Keep all production dependencies unchanged.

## Non-goals

Production restore/coordinator API or mandatory pure pre-open Training validator; in-place
rollback or a generic transaction/registry; durable Training encoding or file publication; new
Training-to-Checkpoint or NN-to-Training dependency; DataLoader, scheduler, scaler, or per-run
graph-random API; backend-specific restore; cross-device bitwise replay; training-mode BatchNorm
exact continuation without step-aligned running-buffer proof; large-model memory claims.

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules), and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements): preserve ownership,
  stored-state exactness, public Engine composition, and one-way dependencies.
- [`extensions-training.md` — `extensions/nn`](../../../../architecture/contracts/extensions-training.md#extensionsnn),
  [`extensions/training`](../../../../architecture/contracts/extensions-training.md#extensionstraining),
  [`extensions/checkpoint`](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint),
  and [Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle):
  NN installs tree state/modes, Training owns session-state meaning, Checkpoint stays model-only,
  and the caller owns cross-owner exclusion and external state.
- [`runtime-prepare-engine.md` — `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine):
  create the replacement session through only the public Engine facade.
- [`foundational-modules.md` — `modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel):
  Tensor/storage and fixed graph-random construction remain Model concerns.

ADR 0035 selects the private-candidate order and refines ADR 0030; these are decision evidence,
not substitutes for the linked contracts. If the executable fixture exposes a missing applicable
contract or needs a new ownership/dependency rule, stop for a coordinated architecture decision.

## Dependencies and integration

- Depends on: Training 0003A, Checkpoint 0001 and 0004, and CPU 0024 — each has a final
  `Status: Complete` result. Training 0003 is Superseded; Checkpoint 0005–0006 are Draft successors.
- Conflicts with: concurrent Training session/state API or restore-lifecycle edits, NN whole-tree
  installation/mode edits, Checkpoint model-load handoff edits, and Checkpoint 0005–0006 work on
  the same cross-owner restore semantics.
- Parallel group: None; serialize this shared semantic boundary.
- Common base revision: `c1c8fbcaffa44bff2f9a662be5658458bf2b52fc` (`main` HEAD). The
  accepted uncommitted contract snapshot is `ARCHITECTURE.md` SHA-256
  `071cc93e3d60f2d131c7ef3eb3539f9ffb9854e58d1eb9d3ea258ecb90806053` and
  `extensions-training.md` SHA-256
  `66dc3d833d6c4c368c9d4c3e5192a440d31d3e7a33c22b05d6823c4d23f67152`;
  ADR 0035 SHA-256 is
  `b529d77e650a39fb3e8951d15015bf0a5dad1f615b432d11815c34a9507a5411`.
  Reverify this decision and both contracts before launch. No parallel group uses this base.
- Integration order: accepted ADR 0035/contract snapshot → this Ready test/documentation task
  → independent Class C review → one integration gate → Checkpoint 0005, then 0006.
- Integration validation: `./gradlew :testing:integration-tests:check :testing:integration-tests:javadoc`
  after final test code; then one configured
  `./gradlew test javadoc --max-workers=2 --console=plain` checkpoint because the integration-test
  build gains a Checkpoint test edge. Architecture tests are needed only if a dependency rule
  or supported boundary changes; neither is planned.
- Shared-document integration owner: Training/Checkpoint coordinator; serial scope, no concurrent
  writer to the restore semantics or shared documents.

Frontier verification (2026-10-14): the four predecessors above are Complete with final results
and validation/review evidence; the Training master plan identifies 0004 as the next frontier,
and Checkpoint 0005–0006 are inactive Draft dependents. No active conflicting Training, NN,
Checkpoint, or CPU task appears in the area plans or current worktree changes. ADR 0035 and the
two authoritative contract files named above form the stable current-worktree decision snapshot;
recheck SHA-256 and status before a clean executor starts. Existing historical review worktrees
do not authorize concurrent writes to this shared semantic scope.

## Files and symbols

- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/`:
  new focused private-candidate recovery integration test and typed caller fixture; use real CPU
  Engine composition, bounded temporary model artifact, and fault stages under test ownership.
- `testing/integration-tests/build.gradle.kts`: add only the required
  `testImplementation(project(":extensions:checkpoint"))` edge; no production edge.
- `ModelCheckpoint.load`/`inspect`, NN `Module`/`ForwardContext`,
  `TrainingParameterStorage.copyToSharedNative`, `TrainingSession.open/state/restore`, and
  `TrainingState.ParameterState.parameterBytes`: existing public boundaries to exercise, not edit.
- `docs/api/training-api.md` and `docs/api/model-checkpoint-api.md`: targeted workflow/status
  review; affected Javadocs and `docs/architecture/training-graph.md`: accuracy review.

## Acceptance criteria

- At a successful-run boundary, the fixture uses a known next input and a concrete typed caller
  cursor/random state that determines it, detached `TrainingState`, a complete model artifact
  (parameters and buffers), every relevant local mode, and actual objective-construction
  contexts. It rejects an absent required fixture participant. A shallow
  `StateDictionary` or `ModelCheckpoint.inspect` is not value evidence.
- The fixture validates available detached facts before private mutation. `ModelCheckpoint.load`
  revalidates complete paths/kinds/types/Shapes, expected digests, and target-defined aliases on
  the fully bound private tree; it never loads into the published tree. A restored heap Tensor is
  directly Engine-executable after CPU 0024, but `TrainingSession.open` still requires writable,
  shareable native parameter storage, so the explicit copy precedes objective construction/open.
- Compare the candidate session's canonical big-endian `state().parameters()` logical bytes with
  supplied `TrainingState` logical bytes by exact stable-path set and byte content, including a
  negative equal-schema/different-value case. Do not compare raw native-order NN carrier bytes.
  Equality is checked before `restore` and before publication. `restore` then owns exact
  optimizer, ordered group/path, schema, counters, logical/master/slot/accumulation width,
  presence, and finite-value validation; no duplicate pure validator is introduced.
- Final NN bindings, native associations, and per-module modes precede objective construction
  from final Tensors and recorded effective contexts. All candidate checks and typed caller-state
  staging precede a single publication under caller exclusion. Every ordinary failure stage
  leaves the published holder and caller state unchanged and closes/discards the private session
  and Arena. A close failure while retiring the old session aborts publication; the old session
  is not guaranteed to remain runnable afterward. Readers outside exclusion have no atomic-view
  guarantee. Test failure before load, after private load/copy/open, at byte comparison, at a
  publicly constructible Training restore mismatch, and at typed caller-state validation with
  the concrete fixture. Owner unit tests retain exhaustive validation-order coverage.
- A positive same-route CPU test continues from the recorded next input and matches an
  uninterrupted control at each captured Training field, NN parameter/buffer value, local mode,
  effective context, and typed caller cursor/random state. Include pending gradients and
  optimizer slots, plus a local-mode/effective-context difference. Document that a missing
  cursor/affecting external state, changing per-run graph-random stream, or training-mode
  BatchNorm without step-aligned buffer proof is outside this exact claim. A fixed
  `GraphRngState.initial` expression may be rebuilt but is not advanced by the session.
- The deliverable is executable integration evidence plus accurate API-guide wording. No
  production facade, transaction, pure validator, dependency-direction change, or durable
  Training codec is required unless execution reveals a specific missing contract and stops.

## Validation

Worker: add a focused integration test with a typed caller-owned holder and concrete fixture
failures: absent required cursor, incompatible model load, canonical parameter-byte mismatch,
invalid objective or Engine open rejection after private installation, one late Training restore
rejection through an optimizer/group/schema mismatch available via public APIs, and typed
caller-state validation rejection. For each, assert no holder swap or caller-state publication
and candidate cleanup. Reuse existing owner unit tests for exhaustive alias/schema,
counter/slot/non-finite, and native-copy validation; do not construct malformed package-private
`TrainingState` payloads outside Training or overconstrain test-helper shape. Run the new focused
test, then `./gradlew :testing:integration-tests:check :testing:integration-tests:javadoc` once after
test code stabilizes. No Training/NN/Checkpoint module rerun is needed absent production edits.

Integration owner: run one configured `./gradlew test javadoc --max-workers=2 --console=plain`
after the new test-only Checkpoint project edge is integrated. Review reuses successful Java
evidence unless executable code changes or a concrete risk warrants a rerun. Check final
documentation links/anchors, targeted glossary terms, and `git diff --check`.

## Documentation and review impact

- The execution deliverable updates `docs/api/training-api.md` from a future gate to the
  tested caller-owned recovery sequence, including heap-to-native copy, exact-claim limits, and
  an example; review `docs/api/model-checkpoint-api.md` for the model-only distinction and
  `docs/architecture/training-graph.md` for consistency. Apply the API/user-guide and
  architecture profiles as appropriate, with targeted glossary search.
- Independently review Javadocs for `ModelCheckpoint.load/inspect`, `Module` state/mode methods,
  `TrainingParameterStorage.copyToSharedNative`, and `TrainingSession.open/state/restore` against
  the executable path. Existing Javadocs may remain unchanged if accurate: no public Java
  semantics or signature is planned. Any new Java test helper follows repository Javadoc rules.
  Class C cross-owner workflow requires a clean independent targeted review even if production
  Java remains unchanged; no normative doc edit is authorized by this execution brief.

## Result

- Added `PrivateTrainingRestoreIntegrationTest.java` and the test-only Checkpoint project edge in
  `testing/integration-tests/build.gradle.kts`. The fixture proves same-route CPU next-input
  continuation and ordinary-failure discard with a distinct private NN tree/session, canonical
  parameter-byte comparison, and typed caller-state handoff; it adds no production API.
- Focused XML: 2 tests, 0 skipped, 0 failures, 0 errors. Implementation reported
  `./gradlew :testing:integration-tests:check :testing:integration-tests:javadoc` passed.
  Independent Class C review found no blocking P1/P2 and updated `docs/api/training-api.md`,
  `docs/api/model-checkpoint-api.md`, `docs/architecture/training-graph.md`, and `docs/glossary.md`.
  Affected public Javadocs were reviewed and remained accurate; no production Java changed.
- Integration owner ran `SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc --max-workers=2 --console=plain`
  successfully (106 tasks up-to-date; successful prior worker results reused). `git diff --check`
  passed.
- Limitation: direct `ModelCheckpoint.save` on the trained leaf-only tree fails Engine
  empty-partition preparation. The test saves a private raw-preserving expression view over exact
  live Tensor references; it does not prove direct leaf-only save. Follow up separately in the
  current Checkpoint plan. No durable `TrainingCheckpoint` exists; Checkpoint 0005 remains Draft.
- Status: Complete.
