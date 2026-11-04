# Task 0007: Generic live [parameter](../../../../glossary.md#term-parameter) [admission](../../../../glossary.md#term-admission) and native-copy cutover

> **Editorial reading context — not a status update.** The application or NN adapter
> supplies ordered stable paths and exact Tensor references through TrainingParameters.
> validateUnchanged must consult the live owner again: comparing a frozen list to itself
> cannot detect replacement. Training separately checks descriptors, current storage and
> values, then borrows that fixed state for the session. NN's adapter owns module traversal;
> generic Training does not import NN to perform it. Today's [live-source admission](../../../../reference/training/parameters.md#generic-live-parameter-admission)
> and [explicit native-copy boundary](../../../../reference/training/parameters.md#explicit-native-parameter-copy)
> explain the dependency inversion, live validation and continuing application-owned Arena lifetime.

## Status

Complete — Pascal's definitive APPROVE and coordinator acceptance received. Authorized Training
frontier: None; Compiler0008 is the next [planning](../../../../glossary.md#term-planning) candidate, still Draft with no brief or launch.

## Change class

Class C — reverses a [module](../../../../glossary.md#term-module) edge, relocates Training, and changes public [admission](../../../../glossary.md#term-admission) across live
[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [identity](../../../../glossary.md#term-identity-meanings) and native-storage [ownership](../../../../glossary.md#term-ownership-meanings) boundaries. Independent targeted code/docs review required.

## Goal and [scope](../../../../glossary.md#term-scope)

Make Training usable with an [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-independent live ordered [parameter](../../../../glossary.md#term-parameter) source, preserving the existing
session/native-copy/state algorithms. Move `extensions/training` to `modules/training`, retain
`:extensions:training` and Java packages, and migrate NN callers atomically through its live adapter.
An NN-taking overload prevents edge removal; adding NN's adapter first creates a cycle.
Relocation, [API](../../../../glossary.md#term-application-programming-interface-api), consumers, tests and authority land together.

Size exception: required scope/result plus the [atomic](../../../../glossary.md#resource-transaction-and-rollback) [dependency](../../../../glossary.md#term-dependency) inversion/live native boundary
exceed the compact [target](../../../../glossary.md#term-target); splitting would create an invalid transitional edge. Total stays under 25 KB.

## Non-goals

No [TrainingSession](../../../../glossary.md#training-session--trainingsession).runValues (Training0008), high-level [Model](../../../../glossary.md#term-model)/withEngine implementation (NN0028),
[optimizer](../../../../glossary.md#term-optimizer) redesign, new storage domain, automatic transfer, [codec](../../../../glossary.md#term-codec)/[Checkpoint](../../../../glossary.md#term-checkpoint) [API](../../../../glossary.md#term-application-programming-interface-api) or package move,
new [dependency](../../../../glossary.md#term-dependency) on [Compiler](../../../../glossary.md#term-compiler), [backend](../../../../glossary.md#backend)/[ABI](../../../../glossary.md#term-application-binary-interface-abi) change, native rebuild, performance campaign or history rewrite.
Engine0021 remains Complete; exact-ID [Engine](../../../../glossary.md#term-engine).[run](../../../../glossary.md#term-run) and the existing prepared [owner](../../../../glossary.md#term-owner) remain unchanged.

## Contracts and accepted [target](../../../../glossary.md#term-target) cutover

The executor read the root [scope](../../../../glossary.md#term-scope) index and these exact authoritative headings:

- [Repository layout](../../../../../ARCHITECTURE.md#repository-layout),
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>,
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>,
  [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Owning extensions-training: [`extensions/nn`](../../../../architecture/contracts/extensions-training.md#extensionsnn),
  [`extensions/training`](../../../../architecture/contracts/extensions-training.md#extensionstraining),
  [Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle).
- Preserved Engine0021 boundary only:
  [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution)<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>.

Launch-baseline authority required Training→[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) and forbade NN→Training. The implementation now
updates the linked root/scoped authority for only the approved cutover; this brief is not authority.
[Accepted ADR0039](../../../../design/decisions/0039-portable-graph-model-persistence.md#necessary-boundary-types-not-a-new-model-hierarchy)
selects the coordinated TARGET: Training has exactly `api(Model, Engine)`; NN has exactly
`api(Model, Training)` at this cutover, not its later direct [Engine](../../../../glossary.md#term-engine)/[Checkpoint](../../../../glossary.md#term-checkpoint) edges.
During implementation, update root [layout](../../../../glossary.md#layout)/scope routing/[ownership](../../../../glossary.md#term-ownership-meanings)/dependencies/testing and Core
[lifecycle](../../../../glossary.md#lifecycle)'s Training [owner](../../../../glossary.md#term-owner) wording plus the owning scoped headings before code relies on the target.
Retain the old `extensionstraining` anchor when renaming its physical-path heading; keep the root
compatibility anchors. Reconcile the lifecycle's old blanket NN→Training prohibition narrowly;
do not widen recovery, [optimizer](../../../../glossary.md#term-optimizer), native, Engine, Checkpoint or lower-layer permissions.
If any additional contract change is needed, stop and report it rather than infer permission.

## Dependencies and integration

- Depends on: approved independently reviewed P2/ADR0039; Complete Training0001–0001E,
  0002A–0002C, 0003A/0004 and Training-owned Checkpoint0005 state reconstruction.
- Conflicts with: shared root/scoped NN/Training lifecycle, settings, dependency tests and consumers;
  Compiler0008/Checkpoint0008/Training0008/NN0028 shared-document cutovers; no active other writer.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [lifecycle glossary](../../../../glossary.md#lifecycle); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Parallel group: None.
- Common base revision: `9e7e8aa15e0833a1bc91263a7add99a80d88cd92` plus integrated dirty
  P1/P2/Engine0021 completed snapshot, 2246 effective files; coordinator's SHA-256 manifest.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [SHA-256 glossary](../../../../glossary.md#term-sha-256); [manifest glossary](../../../../glossary.md#term-manifest))</span>
- Integration order: Engine0021 (Complete) → Training0007 → Compiler0008 → Checkpoint0008 →
  Training0008 → NN0028. Engine0021 is the serial base, not an algorithmic fresh-binding dependency.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Integration [validation](../../../../glossary.md#term-validation): coordinator's one native-enabled repository `test javadoc` boundary gate below.
- Shared-document integration owner: model-execution coordinator; executor is the exclusive delegated
  writer for this packet's coordinated documents, with independent targeted review before acceptance.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [model glossary](../../../../glossary.md#term-model))</span>

2026-10-16 [Ready](../../../../glossary.md#term-ready) verification: Rawls/Astra xhigh independent plan review APPROVE; zero findings.
Coordinator verified listed predecessors/P2 Complete and reviewed, no active conflicts, and HEAD
`9e7e8aa15e0833a1bc91263a7add99a80d88cd92` plus completed 2246-file Engine0021 source baseline
independently unchanged outside these six [planning](../../../../glossary.md#term-planning) documents. All seven [metadata](../../../../glossary.md#term-metadata) fields and the
shared serial integration order remain valid; coordinator owns integration/shared documents.
User authorized implementation after readiness review. Later four tasks remain Draft. Preserve
dirty P1/P2/0021 work; no reset, staging, [commit](../../../../glossary.md#term-commit) or push. Coordinator owns executor launch.

## Files and symbols

- Move `extensions/training/{build.gradle.kts,src/}` to `modules/training/`; exclude generated build
  output. All existing public types stay in `io.github.pho001.synaptik.training`: [`Adam`](../../../../glossary.md#adaptive-optimizer--adam-adamw), [`AdamW`](../../../../glossary.md#adaptive-optimizer--adam-adamw),
  `GradientMode`, [`Optimizer`](../../../../glossary.md#term-optimizer), [`ParameterGroup`](../../../../glossary.md#parameter-group--parametergroup), `Sgd`, [`TrainingParameterStorage`](../../../../glossary.md#native-parameter-copy--trainingparameterstorage), [`TrainingSession`](../../../../glossary.md#training-session--trainingsession),
  `TrainingState`, `TrainingStep`. Package-private `AdaptiveCoefficients`, `TrainingParameter` and
  `package-info.java` move too; no renamed Java type or compatibility project.
- Add public `TrainingParameters.java` in that package: `List<Entry> entries()`,
  `void validateUnchanged()`, nested public `Entry(String path, Tensor tensor)`.
  Change `TrainingSession` open/openForTesting/constructor/[capture](../../../../glossary.md#term-capture-meanings)/validateModuleAndParameters,
  `TrainingParameter` capture/validateBinding (remove [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) wrapper), and `TrainingParameterStorage` copy.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Module.java`: add final public
  `TrainingParameters trainingParameters()` and private nested `LiveTrainingParameters`; retain
  current traversal, state replacement and caller-exclusion behavior. Review `package-info.java`.
- `settings.gradle.kts`: explicit Training projectDir; moved Training build and
  `extensions/nn/build.gradle.kts`: exact [target](../../../../glossary.md#term-target) [API](../../../../glossary.md#term-application-programming-interface-api) edges, no test-only NN back-edge in Training.
- Moved Training tests in `src/test/java/io/github/pho001/synaptik/training/`:
  `TrainingSessionTest`, `TrainingSessionFailureContractTest`, `TrainingParameterStorageTest`,
  `AdamOptimizerTest`, `external/TrainingStateReconstructionTest` migrate NN fixtures to generic
  live sources; retain `TrainingStateOwnershipTest`. Add `TrainingParametersTest.java` for generic
  admission. Keep test seams private/package-private, including deterministic copy-install failure.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission))</span>
- Add `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModuleTrainingParametersTest.java`:
  real NN traversal/wrapper/replacement/deferred-state and copy-adapter tests; move NN-specific
  assertions from Training tests here without losing coverage.
- `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/`:
  `NnTrainingDependencyContractTest.java` exact edges/mapping/import/public-[shape](../../../../glossary.md#shape) rules;
  `EngineCompositionContractTest.java` role-aware explicit Training→public [Engine](../../../../glossary.md#term-engine) exception;
  `CheckpointDependencyContractTest.java` Training physical path only, preserving [Checkpoint](../../../../glossary.md#term-checkpoint) edges.
- `testing/architecture-tests/src/{trainingApiConsumer,nnApiConsumer}/java/io/github/pho001/synaptik/testing/architecture/`
  `TrainingApiConsumer.java`/`NnApiConsumer.java`: Training-only generic consumer and NN-only adapter
  consumer. `testing/architecture-tests/build.gradle.kts`: declare both new Training and NN source
  trees as inputs for source-reading tests; retain isolated classpaths, no broad modules exemption.<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/`
  `TrainingSessionCpuIntegrationTest.java` and `PrivateTrainingRestoreIntegrationTest.java`: use
  `module.trainingParameters()` for open/copy, preserve private candidate/native/next-input tests.
  Existing integration dependencies suffice. Repository search found no Compiler Training consumer;
  no Compiler edit/validation is planned unless a fresh scoped search establishes one.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [validation glossary](../../../../glossary.md#term-validation))</span>

## Acceptance criteria

1. Generic Training builds/tests with no NN imports/dependency or NN-taking compatibility overload.
   Both public open overloads accept TrainingParameters; copyToSharedNative does too. Null source,
   list/entry/path/Tensor reject; blank/duplicate paths reject before seed/compile/preparation or
   copy mutation. Empty sources reject; order is caller encounter order, never sorted.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [dependency glossary](../../../../glossary.md#term-dependency); [TrainingParameters glossary](../../../../glossary.md#live-parameter-source--trainingparameters); [Tensor glossary](../../../../glossary.md#tensor); [seed glossary](../../../../glossary.md#term-seed); [compile glossary](../../../../glossary.md#compile))</span>
2. entries exposes an immutable ordered membership snapshot; Training snapshots/validates admission.
   validateUnchanged reads the live owner, not that frozen list alone. The NN adapter retains the
   root and full ordered paths/wrappers/exact Tensors, re-traverses and rejects count/order/path,
   wrapper or Tensor changes, including late changes. Unbound NN reservations fail at adapter creation.
   Adapter membership validation deliberately does not freeze storage associations or parameter bytes.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [snapshot glossary](../../../../glossary.md#term-snapshot); [admission glossary](../../../../glossary.md#term-admission); [owner glossary](../../../../glossary.md#term-owner); [validation glossary](../../../../glossary.md#term-validation); [parameter glossary](../../../../glossary.md#term-parameter))</span>
3. Training invokes live validation at capture and every existing complete-view validation site:
   before run, state/restore and precommit for accumulation, parameter update and restore. It owns
   descriptor, exact storage identity/schema/liveness/accessibility/capacity and finiteness checks.
   A generic owner's late validation failure and a real NN replacement must reject without publishing
   optimizer slots, logical/master bytes, counters or pending gradients. No stale frozen-map bypass.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [view glossary](../../../../glossary.md#term-view-meanings); [run glossary](../../../../glossary.md#term-run); [precommit glossary](../../../../glossary.md#term-precommit); [descriptor glossary](../../../../glossary.md#term-descriptor); [identity glossary](../../../../glossary.md#term-identity-meanings); [schema glossary](../../../../glossary.md#term-schema); [liveness glossary](../../../../glossary.md#term-liveness); [optimizer glossary](../../../../glossary.md#term-optimizer); [gradients glossary](../../../../glossary.md#term-gradient))</span>
4. Preserve scalar FLOAT32/FLOAT64 objective and connected positive-static dense offset-zero non-view
   FLOAT64/FLOAT32/BFLOAT16/FLOAT16 parameters, bounded snapshot sizes and native writable shareable
   exact-capacity session storage. Preserve duplicate Tensor and overlapping-region rejection,
   including aliased paths; do not split aliases. Keep exact-ID external inputs and input/state overlap
   checks, compiler occurrence/target order, low FLOAT32 masters and all finite-value checks.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [objective glossary](../../../../glossary.md#term-objective); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [compiler glossary](../../../../glossary.md#term-compiler); [occurrence glossary](../../../../glossary.md#term-occurrence); [target glossary](../../../../glossary.md#term-target); [finite glossary](../../../../glossary.md#term-finite))</span>
5. Copy remains explicit, pre-session and caller-excluded into caller-owned shared Arena.
   Validate complete source before destination parameter allocation or association installation;
   preserve existing zero-byte Arena shareability probe and failure precedence.
   Copy represented bits including signed zero, install
   in path order preserving Tensors/wrappers and old source bytes/lifetimes. Same live source remains
   valid after successful copy and can open a session; intentional storage replacement must not
   self-invalidate membership. Check live membership before installation. Allocation/copy failures
   install nothing; injected later installation failure restores exact prior associations and retains
   suppressed rollback failures. Neither copy nor session closes caller arenas; no copy while open.<span data-glossary-navigation="true"> ([Arena glossary](../../../../glossary.md#term-arena); [represented bits glossary](../../../../glossary.md#term-represented-bits); [signed zero glossary](../../../../glossary.md#term-signed-zero); [rollback glossary](../../../../glossary.md#term-rollback))</span>
6. Keep optimizer arithmetic/order/groups, demand-driven staging, snapshot/restore payloads,
   rollback, concurrency and lifecycle/error precedence. Preserve Engine.prepare/run reuse, one
   prepared owner, result-before-prepared-before-seed cleanup, failed-open rollback, close waiting,
   attempt-all cleanup and repeated/concurrent close failure replay. No new per-element dispatch.<span data-glossary-navigation="true"> ([concurrency glossary](../../../../glossary.md#term-concurrency); [lifecycle glossary](../../../../glossary.md#lifecycle); [Engine glossary](../../../../glossary.md#term-engine); [prepare glossary](../../../../glossary.md#prepare); [element glossary](../../../../glossary.md#term-element); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
7. Preserve all existing tests' assertions through migration; add generic-owner negative admission,
   NN live traversal/replacement, same-source copy→open and deterministic install rollback coverage.
   Architecture tests enforce physical mapping and role-specific edges; isolated consumers compile
   without accidental NN transitivity. Current runnable docs/Javadocs use new signatures/paths;
   old readers, declarative behavior and completed historical tasks remain unchanged.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>

## Documentation and review impact

Coordinated write [scope](../../../../glossary.md#term-scope) in addition to code/build/tests above:

- `ARCHITECTURE.md` and `docs/architecture/contracts/extensions-training.md` headings listed above.
- `docs/architecture/{module-boundaries,dependency-rules,lifecycle,training-graph,overview}.md`;
  `docs/developer-guide/{repository-layout,architecture-tests}.md` (include Training path migration);
  `docs/api/{training-api,public-api}.md`, `docs/user-guide/training.md`, targeted `docs/glossary.md`.
- `docs/design/decisions/0039-portable-graph-model-persistence.md` records only this enacted cutover;
  narrow status/link reconciliation in ADRs `0007-neural-network-module-and-training-boundary.md`,
  `0018-public-training-session-and-sgd-lifecycle.md`, `0028-explicit-native-parameter-copy.md`,
  `0030-exact-training-resume-state-ownership.md`, `0035-private-candidate-training-restore.md` under
  the same directory; preserve their historical decisions, algorithms and recovery guarantees.
- This brief, Training/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) master plans, `docs/planning/{model-execution-refactor,roadmap}.md` statuses.

Apply documentation rules plus each edited document's profile; final Javadocs cover every changed
input/result/failure/[lifetime](../../../../glossary.md#term-lifetime). Review only relevant glossary terms. Independent Class C review is
mandatory and reuses successful executable evidence; it does not rerun suites for documentation alone.
No [Model](../../../../glossary.md#term-model) [lifecycle](../../../../glossary.md#lifecycle) is enacted here, including approved future withEngine or configureTraining.

## [Validation](../../../../glossary.md#term-validation)

Worker: focused checks, then affected [modules](../../../../glossary.md#term-module) once; final [Javadoc](../../../../glossary.md#term-javadoc) after documentation edits.
[Run](../../../../glossary.md#term-run) from the repository root, not during [planning](../../../../glossary.md#term-planning):

```bash
./gradlew :extensions:training:test :extensions:nn:test :extensions:training:javadoc :extensions:nn:javadoc
./gradlew :testing:architecture-tests:test
./gradlew :testing:integration-tests:test --tests '*TrainingSessionCpuIntegrationTest' --tests '*PrivateTrainingRestoreIntegrationTest'
git diff --check
```

Architecture runs its full suite and both isolated consumers. Check local links/anchors, new-file
whitespace and no stale live-source paths/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-taking [signatures](../../../../glossary.md#term-signature). If new
[Compiler](../../../../glossary.md#term-compiler) consumers are found, migrate within a coordinator-approved [scope](../../../../glossary.md#term-scope) and run its test/Javadoc once.
Coordinator runs one `./gradlew test javadoc` boundary gate after ordered integration with
`SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-checkpoint-leaf.oCW897/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib`
(verified [SHA-256](../../../../glossary.md#term-sha-256) `03cba3ebfa0bbb78c1bff369ff3451c591846fd975c0034fa7f67fdd2209985a`). Reuse
this existing [dylib](../../../../glossary.md#term-dylib); do not rebuild it. Record source-matched outcomes/skips, not an unconfigured pass.

## Result

Complete — 2026-10-16; Pascal's definitive independent APPROVE, zero remaining code/docs findings, coordinator acceptance; both P3 findings fixed.
Coverage is the exact [Files and symbols](#files-and-symbols) and [documentation scope](#documentation-and-review-impact)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span> above, plus:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Parameter.java`: [Javadoc](../../../../glossary.md#term-javadoc) only.
- `modules/training/src/test/java/io/github/pho001/synaptik/training/TestParameterSource.java`: package-private generic [test fixture](../../../../glossary.md#adr-ci-and-qualification).
- `docs/planning/modules/engine/master-plan.md`: narrow status reconciliation; [Engine](../../../../glossary.md#term-engine) stays Complete.

Delivered physical relocation with preserved [identities](../../../../glossary.md#term-identity-meanings), generic LIVE [admission](../../../../glossary.md#term-admission)/native copy and [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) adapter;
[optimizer](../../../../glossary.md#term-optimizer)/native/exact-ID/[rollback](../../../../glossary.md#term-rollback)/[lifecycle](../../../../glossary.md#lifecycle) algorithms preserved. Docs/Javadocs/glossary cover the cutover.
Successful evidence (reused, not rerun for closeout):

- `./gradlew :extensions:training:test :extensions:nn:test :extensions:training:javadoc :extensions:nn:javadoc --console=plain`: passed Training 83/83, NN 333/333; zero failures/errors/skips; both Javadocs generated.
- `./gradlew :testing:architecture-tests:test --console=plain`: passed 12/12, zero skips; both isolated [API](../../../../glossary.md#term-application-programming-interface-api) consumers compiled; scanned Training/NN sources are [Gradle](../../../../glossary.md#term-gradle) inputs.
- `SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-checkpoint-leaf.oCW897/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew :testing:integration-tests:test --tests '*TrainingSessionCpuIntegrationTest' --tests '*PrivateTrainingRestoreIntegrationTest' --console=plain`: passed 11/11, zero skips; both real-[Metal](../../../../glossary.md#term-metal) cases passed.
- Coordinator `SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-checkpoint-leaf.oCW897/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc :testing:integration-tests:test --rerun --no-build-cache --no-configuration-cache --max-workers=2 --console=plain`: PASSED, 5s, 106 tasks (4 executed, 102 up-to-date).
- Final repository reports: 4151 total, zero failures/errors, 41 unchanged historical/opt-in skips.
  Full integration: 131 total, 130 passed, 1 skip (host CPU autotuning has no alternative candidate);
  both Metal training cases passed. Existing dylib SHA-256 matches Validation; no native rebuild.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [CPU glossary](../../../../glossary.md#term-cpu); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [dylib glossary](../../../../glossary.md#term-dylib); [SHA-256 glossary](../../../../glossary.md#term-sha-256); [Validation glossary](../../../../glossary.md#term-validation))</span>
- After P3 Javadoc fixes, `./gradlew :extensions:training:javadoc --console=plain`: PASSED, 1s,
  16 tasks (2 executed, 14 up-to-date). TrainingState executable text is identical after removing Javadocs.
- `git diff --check`: passed. `node /private/tmp/synaptik-engine0021-check.kNoazJ/check-links.mjs /Users/phujka/IdeaProjects/Synaptik`: same seven historical errors, zero new; changed Markdown links/anchors/whitespace checked.
Protected `THIRD_PARTY_NOTICES.md` and `CpuOrdinaryPointwiseCastCleanJavaOracle.java` remain unchanged.
This closeout changes only seven approved Markdown files; no source/Javadoc edits, tests/gate reruns,
staging, [commit](../../../../glossary.md#term-commit) or push. No unresolved issue or required Training0007 follow-up.
Authorized Training frontier: None. Compiler0008 is only the next [planning](../../../../glossary.md#term-planning) candidate (Draft, no brief/launch);
Training0008/NN0028 stay Draft; withEngine is strictly [TARGET](../../../../glossary.md#term-target), not implemented. No successor planned here.

Status: Complete
