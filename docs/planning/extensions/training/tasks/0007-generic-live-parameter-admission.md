# Task 0007: Generic live parameter admission and native-copy cutover

## Status

Complete — Pascal's definitive APPROVE and coordinator acceptance received. Authorized Training
frontier: None; Compiler0008 is the next planning candidate, still Draft with no brief or launch.

## Change class

Class C — reverses a module edge, relocates Training, and changes public admission across live
NN identity and native-storage ownership boundaries. Independent targeted code/docs review required.

## Goal and scope

Make Training usable with an NN-independent live ordered parameter source, preserving the existing
session/native-copy/state algorithms. Move `extensions/training` to `modules/training`, retain
`:extensions:training` and Java packages, and migrate NN callers atomically through its live adapter.
An NN-taking overload prevents edge removal; adding NN's adapter first creates a cycle.
Relocation, API, consumers, tests and authority land together.

Size exception: required scope/result plus the atomic dependency inversion/live native boundary
exceed the compact target; splitting would create an invalid transitional edge. Total stays under 25 KB.

## Non-goals

No TrainingSession.runValues (Training0008), high-level Model/withEngine implementation (NN0028),
optimizer redesign, new storage domain, automatic transfer, codec/Checkpoint API or package move,
new dependency on Compiler, backend/ABI change, native rebuild, performance campaign or history rewrite.
Engine0021 remains Complete; exact-ID Engine.run and the existing prepared owner remain unchanged.

## Contracts and accepted target cutover

The executor read the root scope index and these exact authoritative headings:

- [Repository layout](../../../../../ARCHITECTURE.md#repository-layout),
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules),
  [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Owning extensions-training: [`extensions/nn`](../../../../architecture/contracts/extensions-training.md#extensionsnn),
  [`extensions/training`](../../../../architecture/contracts/extensions-training.md#extensionstraining),
  [Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle).
- Preserved Engine0021 boundary only:
  [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution).

Launch-baseline authority required Training→NN and forbade NN→Training. The implementation now
updates the linked root/scoped authority for only the approved cutover; this brief is not authority.
[Accepted ADR0039](../../../../design/decisions/0039-portable-graph-model-persistence.md#necessary-boundary-types-not-a-new-model-hierarchy)
selects the coordinated TARGET: Training has exactly `api(Model, Engine)`; NN has exactly
`api(Model, Training)` at this cutover, not its later direct Engine/Checkpoint edges.
During implementation, update root layout/scope routing/ownership/dependencies/testing and Core
lifecycle's Training owner wording plus the owning scoped headings before code relies on the target.
Retain the old `extensionstraining` anchor when renaming its physical-path heading; keep the root
compatibility anchors. Reconcile the lifecycle's old blanket NN→Training prohibition narrowly;
do not widen recovery, optimizer, native, Engine, Checkpoint or lower-layer permissions.
If any additional contract change is needed, stop and report it rather than infer permission.

## Dependencies and integration

- Depends on: approved independently reviewed P2/ADR0039; Complete Training0001–0001E,
  0002A–0002C, 0003A/0004 and Training-owned Checkpoint0005 state reconstruction.
- Conflicts with: shared root/scoped NN/Training lifecycle, settings, dependency tests and consumers;
  Compiler0008/Checkpoint0008/Training0008/NN0028 shared-document cutovers; no active other writer.
- Parallel group: None.
- Common base revision: `9e7e8aa15e0833a1bc91263a7add99a80d88cd92` plus integrated dirty
  P1/P2/Engine0021 completed snapshot, 2246 effective files; coordinator's SHA-256 manifest.
- Integration order: Engine0021 (Complete) → Training0007 → Compiler0008 → Checkpoint0008 →
  Training0008 → NN0028. Engine0021 is the serial base, not an algorithmic fresh-binding dependency.
- Integration validation: coordinator's one native-enabled repository `test javadoc` boundary gate below.
- Shared-document integration owner: model-execution coordinator; executor is the exclusive delegated
  writer for this packet's coordinated documents, with independent targeted review before acceptance.

2026-10-16 Ready verification: Rawls/Astra xhigh independent plan review APPROVE; zero findings.
Coordinator verified listed predecessors/P2 Complete and reviewed, no active conflicts, and HEAD
`9e7e8aa15e0833a1bc91263a7add99a80d88cd92` plus completed 2246-file Engine0021 source baseline
independently unchanged outside these six planning documents. All seven metadata fields and the
shared serial integration order remain valid; coordinator owns integration/shared documents.
User authorized implementation after readiness review. Later four tasks remain Draft. Preserve
dirty P1/P2/0021 work; no reset, staging, commit or push. Coordinator owns executor launch.

## Files and symbols

- Move `extensions/training/{build.gradle.kts,src/}` to `modules/training/`; exclude generated build
  output. All existing public types stay in `io.github.pho001.synaptik.training`: `Adam`, `AdamW`,
  `GradientMode`, `Optimizer`, `ParameterGroup`, `Sgd`, `TrainingParameterStorage`, `TrainingSession`,
  `TrainingState`, `TrainingStep`. Package-private `AdaptiveCoefficients`, `TrainingParameter` and
  `package-info.java` move too; no renamed Java type or compatibility project.
- Add public `TrainingParameters.java` in that package: `List<Entry> entries()`,
  `void validateUnchanged()`, nested public `Entry(String path, Tensor tensor)`.
  Change `TrainingSession` open/openForTesting/constructor/capture/validateModuleAndParameters,
  `TrainingParameter` capture/validateBinding (remove NN wrapper), and `TrainingParameterStorage` copy.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Module.java`: add final public
  `TrainingParameters trainingParameters()` and private nested `LiveTrainingParameters`; retain
  current traversal, state replacement and caller-exclusion behavior. Review `package-info.java`.
- `settings.gradle.kts`: explicit Training projectDir; moved Training build and
  `extensions/nn/build.gradle.kts`: exact target API edges, no test-only NN back-edge in Training.
- Moved Training tests in `src/test/java/io/github/pho001/synaptik/training/`:
  `TrainingSessionTest`, `TrainingSessionFailureContractTest`, `TrainingParameterStorageTest`,
  `AdamOptimizerTest`, `external/TrainingStateReconstructionTest` migrate NN fixtures to generic
  live sources; retain `TrainingStateOwnershipTest`. Add `TrainingParametersTest.java` for generic
  admission. Keep test seams private/package-private, including deterministic copy-install failure.
- Add `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModuleTrainingParametersTest.java`:
  real NN traversal/wrapper/replacement/deferred-state and copy-adapter tests; move NN-specific
  assertions from Training tests here without losing coverage.
- `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/`:
  `NnTrainingDependencyContractTest.java` exact edges/mapping/import/public-shape rules;
  `EngineCompositionContractTest.java` role-aware explicit Training→public Engine exception;
  `CheckpointDependencyContractTest.java` Training physical path only, preserving Checkpoint edges.
- `testing/architecture-tests/src/{trainingApiConsumer,nnApiConsumer}/java/io/github/pho001/synaptik/testing/architecture/`
  `TrainingApiConsumer.java`/`NnApiConsumer.java`: Training-only generic consumer and NN-only adapter
  consumer. `testing/architecture-tests/build.gradle.kts`: declare both new Training and NN source
  trees as inputs for source-reading tests; retain isolated classpaths, no broad modules exemption.
- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/`
  `TrainingSessionCpuIntegrationTest.java` and `PrivateTrainingRestoreIntegrationTest.java`: use
  `module.trainingParameters()` for open/copy, preserve private candidate/native/next-input tests.
  Existing integration dependencies suffice. Repository search found no Compiler Training consumer;
  no Compiler edit/validation is planned unless a fresh scoped search establishes one.

## Acceptance criteria

1. Generic Training builds/tests with no NN imports/dependency or NN-taking compatibility overload.
   Both public open overloads accept TrainingParameters; copyToSharedNative does too. Null source,
   list/entry/path/Tensor reject; blank/duplicate paths reject before seed/compile/preparation or
   copy mutation. Empty sources reject; order is caller encounter order, never sorted.
2. entries exposes an immutable ordered membership snapshot; Training snapshots/validates admission.
   validateUnchanged reads the live owner, not that frozen list alone. The NN adapter retains the
   root and full ordered paths/wrappers/exact Tensors, re-traverses and rejects count/order/path,
   wrapper or Tensor changes, including late changes. Unbound NN reservations fail at adapter creation.
   Adapter membership validation deliberately does not freeze storage associations or parameter bytes.
3. Training invokes live validation at capture and every existing complete-view validation site:
   before run, state/restore and precommit for accumulation, parameter update and restore. It owns
   descriptor, exact storage identity/schema/liveness/accessibility/capacity and finiteness checks.
   A generic owner's late validation failure and a real NN replacement must reject without publishing
   optimizer slots, logical/master bytes, counters or pending gradients. No stale frozen-map bypass.
4. Preserve scalar FLOAT32/FLOAT64 objective and connected positive-static dense offset-zero non-view
   FLOAT64/FLOAT32/BFLOAT16/FLOAT16 parameters, bounded snapshot sizes and native writable shareable
   exact-capacity session storage. Preserve duplicate Tensor and overlapping-region rejection,
   including aliased paths; do not split aliases. Keep exact-ID external inputs and input/state overlap
   checks, compiler occurrence/target order, low FLOAT32 masters and all finite-value checks.
5. Copy remains explicit, pre-session and caller-excluded into caller-owned shared Arena.
   Validate complete source before destination parameter allocation or association installation;
   preserve existing zero-byte Arena shareability probe and failure precedence.
   Copy represented bits including signed zero, install
   in path order preserving Tensors/wrappers and old source bytes/lifetimes. Same live source remains
   valid after successful copy and can open a session; intentional storage replacement must not
   self-invalidate membership. Check live membership before installation. Allocation/copy failures
   install nothing; injected later installation failure restores exact prior associations and retains
   suppressed rollback failures. Neither copy nor session closes caller arenas; no copy while open.
6. Keep optimizer arithmetic/order/groups, demand-driven staging, snapshot/restore payloads,
   rollback, concurrency and lifecycle/error precedence. Preserve Engine.prepare/run reuse, one
   prepared owner, result-before-prepared-before-seed cleanup, failed-open rollback, close waiting,
   attempt-all cleanup and repeated/concurrent close failure replay. No new per-element dispatch.
7. Preserve all existing tests' assertions through migration; add generic-owner negative admission,
   NN live traversal/replacement, same-source copy→open and deterministic install rollback coverage.
   Architecture tests enforce physical mapping and role-specific edges; isolated consumers compile
   without accidental NN transitivity. Current runnable docs/Javadocs use new signatures/paths;
   old readers, declarative behavior and completed historical tasks remain unchanged.

## Documentation and review impact

Coordinated write scope in addition to code/build/tests above:

- `ARCHITECTURE.md` and `docs/architecture/contracts/extensions-training.md` headings listed above.
- `docs/architecture/{module-boundaries,dependency-rules,lifecycle,training-graph,overview}.md`;
  `docs/developer-guide/{repository-layout,architecture-tests}.md` (include Training path migration);
  `docs/api/{training-api,public-api}.md`, `docs/user-guide/training.md`, targeted `docs/glossary.md`.
- `docs/design/decisions/0039-portable-graph-model-persistence.md` records only this enacted cutover;
  narrow status/link reconciliation in ADRs `0007-neural-network-module-and-training-boundary.md`,
  `0018-public-training-session-and-sgd-lifecycle.md`, `0028-explicit-native-parameter-copy.md`,
  `0030-exact-training-resume-state-ownership.md`, `0035-private-candidate-training-restore.md` under
  the same directory; preserve their historical decisions, algorithms and recovery guarantees.
- This brief, Training/NN master plans, `docs/planning/{model-execution-refactor,roadmap}.md` statuses.

Apply documentation rules plus each edited document's profile; final Javadocs cover every changed
input/result/failure/lifetime. Review only relevant glossary terms. Independent Class C review is
mandatory and reuses successful executable evidence; it does not rerun suites for documentation alone.
No Model lifecycle is enacted here, including approved future withEngine or configureTraining.

## Validation

Worker: focused checks, then affected modules once; final Javadoc after documentation edits.
Run from the repository root, not during planning:

```bash
./gradlew :extensions:training:test :extensions:nn:test :extensions:training:javadoc :extensions:nn:javadoc
./gradlew :testing:architecture-tests:test
./gradlew :testing:integration-tests:test --tests '*TrainingSessionCpuIntegrationTest' --tests '*PrivateTrainingRestoreIntegrationTest'
git diff --check
```

Architecture runs its full suite and both isolated consumers. Check local links/anchors, new-file
whitespace and no stale live-source paths/NN-taking signatures. If new
Compiler consumers are found, migrate within a coordinator-approved scope and run its test/Javadoc once.
Coordinator runs one `./gradlew test javadoc` boundary gate after ordered integration with
`SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-checkpoint-leaf.oCW897/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib`
(verified SHA-256 `03cba3ebfa0bbb78c1bff369ff3451c591846fd975c0034fa7f67fdd2209985a`). Reuse
this existing dylib; do not rebuild it. Record source-matched outcomes/skips, not an unconfigured pass.

## Result

Complete — 2026-10-16; Pascal's definitive independent APPROVE, zero remaining code/docs findings, coordinator acceptance; both P3 findings fixed.
Coverage is the exact [Files and symbols](#files-and-symbols) and [documentation scope](#documentation-and-review-impact) above, plus:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Parameter.java`: Javadoc only.
- `modules/training/src/test/java/io/github/pho001/synaptik/training/TestParameterSource.java`: package-private generic test fixture.
- `docs/planning/modules/engine/master-plan.md`: narrow status reconciliation; Engine stays Complete.

Delivered physical relocation with preserved identities, generic LIVE admission/native copy and NN adapter;
optimizer/native/exact-ID/rollback/lifecycle algorithms preserved. Docs/Javadocs/glossary cover the cutover.
Successful evidence (reused, not rerun for closeout):

- `./gradlew :extensions:training:test :extensions:nn:test :extensions:training:javadoc :extensions:nn:javadoc --console=plain`: passed Training 83/83, NN 333/333; zero failures/errors/skips; both Javadocs generated.
- `./gradlew :testing:architecture-tests:test --console=plain`: passed 12/12, zero skips; both isolated API consumers compiled; scanned Training/NN sources are Gradle inputs.
- `SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-checkpoint-leaf.oCW897/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew :testing:integration-tests:test --tests '*TrainingSessionCpuIntegrationTest' --tests '*PrivateTrainingRestoreIntegrationTest' --console=plain`: passed 11/11, zero skips; both real-Metal cases passed.
- Coordinator `SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-checkpoint-leaf.oCW897/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc :testing:integration-tests:test --rerun --no-build-cache --no-configuration-cache --max-workers=2 --console=plain`: PASSED, 5s, 106 tasks (4 executed, 102 up-to-date).
- Final repository reports: 4151 total, zero failures/errors, 41 unchanged historical/opt-in skips.
  Full integration: 131 total, 130 passed, 1 skip (host CPU autotuning has no alternative candidate);
  both Metal training cases passed. Existing dylib SHA-256 matches Validation; no native rebuild.
- After P3 Javadoc fixes, `./gradlew :extensions:training:javadoc --console=plain`: PASSED, 1s,
  16 tasks (2 executed, 14 up-to-date). TrainingState executable text is identical after removing Javadocs.
- `git diff --check`: passed. `node /private/tmp/synaptik-engine0021-check.kNoazJ/check-links.mjs /Users/phujka/IdeaProjects/Synaptik`: same seven historical errors, zero new; changed Markdown links/anchors/whitespace checked.
Protected `THIRD_PARTY_NOTICES.md` and `CpuOrdinaryPointwiseCastCleanJavaOracle.java` remain unchanged.
This closeout changes only seven approved Markdown files; no source/Javadoc edits, tests/gate reruns,
staging, commit or push. No unresolved issue or required Training0007 follow-up.
Authorized Training frontier: None. Compiler0008 is only the next planning candidate (Draft, no brief/launch);
Training0008/NN0028 stay Draft; withEngine is strictly TARGET, not implemented. No successor planned here.

Status: Complete
