# Task 0004: Joint restore on an unpublished candidate

> **Editorial reading context — not a status update.** The integration fixture acts as
> the application coordinator. It restores a distinct unpublished NN tree, copies final
> parameter values into its own native Arena, constructs an objective from those final Tensor
> identities, opens/restores a private session and validates external progress before publishing
> one replacement holder. NN and Training each validate their own state; the caller provides
> exclusion and publication. Discarding a failed private candidate is not rollback of an
> already retired old session. Today's [recovery protocol](../../../../reference/training/recovery.md#caller-coordinated-in-memory-recovery)
> and [native-copy lifetime](../../../../reference/training/parameters.md#explicit-native-parameter-copy)
> explain both the integration proof and its bounded resource/atomicity claim.

## Status

Complete — the bounded caller-coordinated, unpublished-candidate protocol in
[ADR 0035](../../../../design/decisions/0035-private-candidate-training-restore.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span> has executable
integration evidence and independent Class C review. This is not durable Training [persistence](../../../../glossary.md#term-persistence).

## Change class

Class C — restore crosses [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [binding](../../../../glossary.md#term-binding)/mode [ownership](../../../../glossary.md#term-ownership-meanings), Training's fixed-session [lifecycle](../../../../glossary.md#lifecycle),
caller-owned continuation state, and the future [Checkpoint](../../../../glossary.md#term-checkpoint) adapter boundary. Require clean
implementation and independent targeted documentation/review contexts.

## Goal

Prove a bounded, executable caller-coordinated in-memory recovery protocol using the existing
public [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), [Checkpoint](../../../../glossary.md#term-checkpoint), Training, and [Engine](../../../../glossary.md#term-engine) APIs. A failed private replacement must not publish
partly restored NN, Training, or typed caller state. Checkpoint 0005–0006 may consume the proven
protocol later; no durable training [artifact](../../../../glossary.md#term-artifact) or new production [API](../../../../glossary.md#term-application-programming-interface-api) is required by this task.

## [Scope](../../../../glossary.md#term-scope)

- Add an integration-test caller fixture that captures one complete, quiescent successful-run
  boundary and owns exclusion, a distinct private fully bound NN tree, a private Training session,
  a shareable native parameter Arena, and a typed staged next-input/progress/external-state value.
  The model artifact supplies detached NN values; <span data-glossary-navigation="true"> ([captures glossary](../../../../glossary.md#term-capture-meanings))</span><span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [quiescent glossary](../../../../glossary.md#term-quiescent); [run glossary](../../../../glossary.md#term-run); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Training session glossary](../../../../glossary.md#training-session--trainingsession); [parameter glossary](../../../../glossary.md#term-parameter); [Arena glossary](../../../../glossary.md#term-arena); [model glossary](../../../../glossary.md#term-model); [artifact glossary](../../../../glossary.md#term-artifact); [detached glossary](../../../../glossary.md#detached-model-results))</span>`TrainingState` and caller values stay in memory.
- Exercise `ModelCheckpoint.load` on the private tree (independent of any earlier `inspect`),
  restore per-module modes (parent first, then child overrides for mixed modes), call<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
  `TrainingParameterStorage.copyToSharedNative`, reconstruct the
  objective from final Tensors and recorded <span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective); [Tensors glossary](../../../../glossary.md#tensor))</span>`ForwardContext` values, then use
  `TrainingSession.open`, `state()`, and `restore`. Compare canonical candidate logical parameter
  bytes with <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>`TrainingState` at every stable path before restore and [publication](../../../../glossary.md#term-publication).
- Publish one caller-owned typed run-holder only after candidate and external-state validation
  succeeds under exclusion. Retire the quiescent old session at handoff; on ordinary failure,
  close/discard the private session and Arena without publishing the candidate. The old session
  may already be closed if retirement fails. Keep all production dependencies unchanged.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [handoff glossary](../../../../glossary.md#term-handoff))</span>

## Non-goals

Production restore/coordinator [API](../../../../glossary.md#term-application-programming-interface-api) or mandatory pure pre-open Training validator; in-place
[rollback](../../../../glossary.md#term-rollback) or a generic [transaction](../../../../glossary.md#term-transaction)/[registry](../../../../glossary.md#term-registry); durable Training encoding or file [publication](../../../../glossary.md#term-publication); new
Training-to-[Checkpoint](../../../../glossary.md#term-checkpoint) or [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-to-Training [dependency](../../../../glossary.md#term-dependency); DataLoader, scheduler, scaler, or per-[run](../../../../glossary.md#term-run)
[graph](../../../../glossary.md#graph)-random API; [backend](../../../../glossary.md#backend)-specific restore; cross-[device](../../../../glossary.md#term-device) bitwise replay; training-mode BatchNorm
exact continuation without step-aligned running-[buffer](../../../../glossary.md#term-buffer) proof; large-[model](../../../../glossary.md#term-model) memory claims.

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>,
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>,
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>,
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules), and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements): preserve ownership,
  stored-state exactness, public Engine composition, and one-way dependencies.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- [`extensions-training.md` — `extensions/nn`](../../../../architecture/contracts/extensions-training.md#extensionsnn),
  [`extensions/training`](../../../../architecture/contracts/extensions-training.md#extensionstraining),
  [`extensions/checkpoint`](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint),
  and [Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([Optimizer glossary](../../../../glossary.md#term-optimizer))</span>:
  NN installs tree state/modes, Training owns session-state meaning, Checkpoint stays model-only,
  and the caller owns cross-owner exclusion and external state.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner))</span>
- [`runtime-prepare-engine.md` — `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine):
  create the replacement session through only the public Engine facade.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>
- [`foundational-modules.md` — `modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel):
  Tensor/storage and fixed graph-random construction remain Model concerns.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph))</span>

[ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0035 selects the private-candidate order and refines ADR 0030; these are decision evidence,
not substitutes for the linked contracts. If the executable [fixture](../../../../glossary.md#term-fixture) exposes a missing applicable
contract or needs a new [ownership](../../../../glossary.md#term-ownership-meanings)/[dependency](../../../../glossary.md#term-dependency) rule, stop for a coordinated architecture decision.

## Dependencies and integration

- Depends on: Training 0003A, [Checkpoint](../../../../glossary.md#term-checkpoint) 0001 and 0004, and [CPU](../../../../glossary.md#term-cpu) 0024 — each has a final
  `Status: Complete` result. Training 0003 is Superseded; Checkpoint 0005–0006 are Draft successors.
- Conflicts with: concurrent Training session/state API or restore-lifecycle edits, NN whole-tree
  installation/mode edits, Checkpoint model-load handoff edits, and Checkpoint 0005–0006 work on
  the same cross-owner restore semantics.<span data-glossary-navigation="true"> ([Training session glossary](../../../../glossary.md#training-session--trainingsession); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [lifecycle glossary](../../../../glossary.md#lifecycle); [model glossary](../../../../glossary.md#term-model); [handoff glossary](../../../../glossary.md#term-handoff); [owner glossary](../../../../glossary.md#term-owner))</span>
- Parallel group: None; serialize this shared semantic boundary.
- Common base revision: `c1c8fbcaffa44bff2f9a662be5658458bf2b52fc` (`main` HEAD). The
  accepted uncommitted contract snapshot is `ARCHITECTURE.md` [SHA-256](../../../../glossary.md#term-sha-256)
  `071cc93e3d60f2d131c7ef3eb3539f9ffb9854e58d1eb9d3ea258ecb90806053` and
  `extensions-training.md` SHA-256
  `66dc3d833d6c4c368c9d4c3e5192a440d31d3e7a33c22b05d6823c4d23f67152`;
  ADR 0035 SHA-256 is
  `b529d77e650a39fb3e8951d15015bf0a5dad1f615b432d11815c34a9507a5411`.
  Reverify this decision and both contracts before launch. No parallel group uses this base.
- Integration order: accepted ADR 0035/contract snapshot → this Ready test/documentation task
  → independent Class C review → one integration gate → Checkpoint 0005, then 0006.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Integration [validation](../../../../glossary.md#term-validation): `./gradlew :testing:integration-tests:check :testing:integration-tests:javadoc`
  after final test code; then one configured
  `./gradlew test javadoc --max-workers=2 --console=plain` checkpoint because the integration-test
  build gains a Checkpoint test edge. Architecture tests are needed only if a dependency rule
  or supported boundary changes; neither is planned.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Shared-document integration owner: Training/Checkpoint coordinator; serial scope, no concurrent
  writer to the restore semantics or shared documents.

Frontier verification (2026-10-14): the four predecessors above are Complete with final results
and validation/review evidence; the Training master plan identifies 0004 as the next frontier,
and Checkpoint 0005–0006 are inactive Draft dependents. No active conflicting Training, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context),
Checkpoint, or CPU task appears in the area plans or current worktree changes. [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0035 and the
two authoritative contract files named above form the stable current-worktree decision [snapshot](../../../../glossary.md#term-snapshot);
recheck SHA-256 and status before a clean executor starts. Existing historical review worktrees
do not authorize concurrent writes to this shared semantic [scope](../../../../glossary.md#term-scope).

## Files and symbols

- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/`:
  new focused private-candidate recovery integration test and typed caller fixture; use real CPU
  Engine composition, bounded temporary model artifact, and fault stages under test ownership.<span data-glossary-navigation="true"> ([integration test glossary](../../../../glossary.md#term-integration-test); [fixture glossary](../../../../glossary.md#term-fixture); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [model glossary](../../../../glossary.md#term-model); [artifact glossary](../../../../glossary.md#term-artifact); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `testing/integration-tests/build.gradle.kts`: add only the required
  `testImplementation(project(":extensions:checkpoint"))` edge; no production edge.
- `ModelCheckpoint.load`/`inspect`, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [`Module`](../../../../glossary.md#term-module)/`ForwardContext`,
  `TrainingParameterStorage.copyToSharedNative`, `TrainingSession.open/state/restore`, and
  `TrainingState.ParameterState.parameterBytes`: existing public boundaries to exercise, not edit.
- `docs/api/training-api.md` and `docs/api/model-checkpoint-api.md`: targeted workflow/status
  review; affected Javadocs and `docs/architecture/training-graph.md`: accuracy review.

## Acceptance criteria

- At a successful-run boundary, the fixture uses a known next input and a concrete typed caller
  cursor/random state that determines it, detached <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>`TrainingState`, a complete model artifact
  (parameters and buffers), every relevant local mode, and actual objective-construction
  contexts. It rejects an absent required fixture participant. A shallow<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [artifact glossary](../../../../glossary.md#term-artifact); [parameters glossary](../../../../glossary.md#term-parameter); [buffers glossary](../../../../glossary.md#term-buffer); [objective glossary](../../../../glossary.md#term-objective))</span>
  `StateDictionary` or `ModelCheckpoint.inspect` is not value evidence.
- The [fixture](../../../../glossary.md#term-fixture) validates available [detached](../../../../glossary.md#detached-model-results) facts before private mutation. `ModelCheckpoint.load`
  revalidates complete paths/kinds/types/Shapes, expected digests, and target-defined aliases on
  the fully bound private tree; it never loads into the published tree. A restored heap Tensor is
  directly Engine-executable after CPU 0024, but <span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [target glossary](../../../../glossary.md#term-target); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [heap glossary](../../../../glossary.md#term-heap); [Tensor glossary](../../../../glossary.md#tensor); [Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu))</span>`TrainingSession.open` still requires writable,
  shareable native parameter storage, so the explicit copy precedes objective construction/open.
- Compare the candidate session's [canonical](../../../../glossary.md#term-canonical-form) [big-endian](../../../../glossary.md#text-encoding-and-file-boundaries) `state().parameters()` logical bytes with
  supplied `TrainingState` logical bytes by exact stable-path set and byte content, including a
  negative equal-schema/different-value case. Do not compare raw native-order NN carrier bytes.
  Equality is checked before <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>`restore` and before [publication](../../../../glossary.md#term-publication). `restore` then owns exact
  optimizer, ordered group/path, schema, counters, logical/master/slot/accumulation width,
  presence, and finite-value validation; no duplicate pure validator is introduced.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [finite glossary](../../../../glossary.md#term-finite); [validation glossary](../../../../glossary.md#term-validation))</span>
- Final NN bindings, native associations, and per-module modes precede objective construction
  from final Tensors and recorded effective contexts. All candidate checks and typed caller-state
  staging precede a single publication under caller exclusion. Every ordinary failure stage
  leaves the published holder and caller state unchanged and closes/discards the private session
  and Arena. A close failure while retiring the old session aborts publication; the old session
  is not guaranteed to remain runnable afterward. Readers outside exclusion have no atomic-view
  guarantee. Test failure before load, after private load/copy/open, at byte comparison, at a
  publicly constructible Training restore mismatch, and at typed caller-state validation with
  the concrete fixture. Owner unit tests retain exhaustive validation-order coverage.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding); [module glossary](../../../../glossary.md#term-module); [Arena glossary](../../../../glossary.md#term-arena); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [view glossary](../../../../glossary.md#term-view-meanings); [Owner glossary](../../../../glossary.md#term-owner))</span>
- A positive same-route CPU test continues from the recorded next input and matches an
  uninterrupted control at each captured Training field, NN parameter/buffer value, local mode,
  effective context, and typed caller cursor/random state. Include pending gradients and
  optimizer slots, plus a local-mode/effective-context difference. Document that a missing
  cursor/affecting external state, changing per-run graph-random stream, or training-mode
  BatchNorm without step-aligned buffer proof is outside this exact claim. A fixed<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph); [random stream glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
  `GraphRngState.initial` expression may be rebuilt but is not advanced by the session.
- The deliverable is executable integration evidence plus accurate API-guide wording. No
  production facade, transaction, pure validator, dependency-direction change, or durable
  Training codec is required unless execution reveals a specific missing contract and stops.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [facade glossary](../../../../glossary.md#term-facade); [transaction glossary](../../../../glossary.md#term-transaction); [dependency glossary](../../../../glossary.md#term-dependency); [codec glossary](../../../../glossary.md#term-codec))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker: add a focused [integration test](../../../../glossary.md#term-integration-test) with a typed caller-owned holder and concrete [fixture](../../../../glossary.md#term-fixture)
failures: absent required cursor, incompatible [model](../../../../glossary.md#term-model) load, [canonical](../../../../glossary.md#term-canonical-form) [parameter](../../../../glossary.md#term-parameter)-byte mismatch,
invalid [objective](../../../../glossary.md#term-objective) or [Engine](../../../../glossary.md#term-engine) open rejection after private installation, one late Training restore
rejection through an [optimizer](../../../../glossary.md#term-optimizer)/group/[schema](../../../../glossary.md#term-schema) mismatch available via public APIs, and typed
caller-state validation rejection. For each, assert no holder swap or caller-state [publication](../../../../glossary.md#term-publication)
and candidate cleanup. Reuse existing [owner](../../../../glossary.md#term-owner) unit tests for exhaustive [alias](../../../../glossary.md#aliasing-and-arena)/schema,
counter/slot/non-[finite](../../../../glossary.md#term-finite), and native-copy validation; do not construct malformed package-private
`TrainingState` payloads outside Training or overconstrain test-helper [shape](../../../../glossary.md#shape). [Run](../../../../glossary.md#term-run) the new focused
test, then `./gradlew :testing:integration-tests:check :testing:integration-tests:javadoc` once after
test code stabilizes. No Training/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[Checkpoint](../../../../glossary.md#term-checkpoint) [module](../../../../glossary.md#term-module) rerun is needed absent production edits.

Integration owner: run one configured `./gradlew test javadoc --max-workers=2 --console=plain`
after the new test-only Checkpoint project edge is integrated. Review reuses successful Java
evidence unless executable code changes or a concrete risk warrants a rerun. Check final
documentation links/anchors, targeted glossary terms, and `git diff --check`.

## Documentation and review impact

- The execution deliverable updates `docs/api/training-api.md` from a future gate to the
  tested caller-owned recovery sequence, including heap-to-native copy, exact-claim limits, and
  an example; review <span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap))</span>`docs/api/model-checkpoint-api.md` for the [model](../../../../glossary.md#term-model)-only distinction and
  `docs/architecture/training-graph.md` for consistency. Apply the API/user-guide and
  architecture profiles as appropriate, with targeted glossary search.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Independently review Javadocs for `ModelCheckpoint.load/inspect`, [`Module`](../../../../glossary.md#term-module) state/mode methods,
  `TrainingParameterStorage.copyToSharedNative`, and `TrainingSession.open/state/restore` against
  the executable path. Existing Javadocs may remain unchanged if accurate: no public Java
  semantics or signature is planned. Any new Java test helper follows repository Javadoc rules.
  Class C cross-owner workflow requires a clean independent targeted review even if production
  Java remains unchanged; no normative doc edit is authorized by this execution brief.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [Javadoc glossary](../../../../glossary.md#term-javadoc); [owner glossary](../../../../glossary.md#term-owner); [normative glossary](../../../../glossary.md#term-normative))</span>

## Result

- Added `PrivateTrainingRestoreIntegrationTest.java` and the test-only [Checkpoint](../../../../glossary.md#term-checkpoint) project edge in
  `testing/integration-tests/build.gradle.kts`. The fixture proves same-route CPU next-input
  continuation and ordinary-failure discard with a distinct private NN tree/session, canonical
  parameter-byte comparison, and typed caller-state handoff; it adds no production API.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [CPU glossary](../../../../glossary.md#term-cpu); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [canonical glossary](../../../../glossary.md#term-canonical-form); [parameter glossary](../../../../glossary.md#term-parameter); [handoff glossary](../../../../glossary.md#term-handoff); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Focused XML: 2 tests, 0 skipped, 0 failures, 0 errors. Implementation reported
  `./gradlew :testing:integration-tests:check :testing:integration-tests:javadoc` passed.
  Independent Class C review found no blocking P1/P2 and updated `docs/api/training-api.md`,
  `docs/api/model-checkpoint-api.md`, `docs/architecture/training-graph.md`, and `docs/glossary.md`.
  Affected public Javadocs were reviewed and remained accurate; no production Java changed.
- Integration [owner](../../../../glossary.md#term-owner) ran `SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc --max-workers=2 --console=plain`
  successfully (106 tasks up-to-date; successful prior worker results reused). `git diff --check`
  passed.
- Limitation: direct `ModelCheckpoint.save` on the trained leaf-only tree fails Engine
  empty-partition preparation. The test saves a private raw-preserving expression view over exact
  live Tensor references; it does not prove direct leaf-only save. Follow up separately in the
  current Checkpoint plan. No durable <span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf); [Engine glossary](../../../../glossary.md#term-engine); [partition glossary](../../../../glossary.md#partition); [view glossary](../../../../glossary.md#term-view-meanings); [Tensor glossary](../../../../glossary.md#tensor))</span>`TrainingCheckpoint` exists; Checkpoint 0005 remains Draft.
- Status: Complete.
