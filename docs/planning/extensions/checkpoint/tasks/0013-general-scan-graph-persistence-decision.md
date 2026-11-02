# Task CP0013: Bounded general-scan graph persistence

## Status

Complete — main accepted final independent review and all recorded gates on 2026-10-23.
User approved selective v4 and the existing bounded cold inventory (“ok, pust se do toho”).
Cicero pre-code and final independent APPROVE: zero outstanding P1/P2; worker/reviewer closed.
All frontiers None; no successor, commit or push. Historical launch was main-verified on 2026-10-23:
Complete prerequisites/reviews/native join, inactive conflicts, unchanged HEAD/index and 2420 files
(16 accepted authority/planning deltas only). Main retains shared integration ownership.
Accepted pre-task tree: `/private/tmp/synaptik-cp0013-baseline.8kiib4/accepted`;
focused evidence: `/private/tmp/synaptik-cp0013-evidence.xLBA70`. Preserve unrelated dirty work.

## Change class and goal

Class C: save ordinary general `Scan` source/values through Checkpoint, load cold through Compiler.
Clean implementation/independent reviews; NN0032's backend/layout gate stays separate.

## Scope and non-goals

One capability: authority, wire/save-load, tests/Javadoc/docs and consumer proof.
New/moved production types/packages/public signatures/concepts: **None**; ArtifactLimits unchanged.
No `.rnn` migration, rewrite/unrolled export, fallback/executor, all-PURE or execution/AD/padding grant.
Persist no compiled/prepared/device/runtime state, batches, tape, optimizer/RNG/cursor or callbacks.

## Contracts

Read [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts) first, then:
Executors read this brief, affected source/tests and these headings, not full global plans/history.

- [Root](../../../../../ARCHITECTURE.md): `Authority, incorporation, and precedence`, `Core lifecycle`,
  `Core invariants`, `Scope-indexed normative contracts`, `Dependency rules`,
  `Documentation authority and architecture updates`, `Testing requirements`.
- Extensions/training: [extensions/checkpoint](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint),
  [Generic graph source and values](../../../../architecture/contracts/extensions-training.md#generic-graph-source-and-values),
  [Canonical graph-model version 1](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-1),
  [version 2](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-2),
  [version 3](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-3),
  [version 4](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-4),
  [extensions/nn](../../../../architecture/contracts/extensions-training.md#extensionsnn) / [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models)
  (consumer save/load boundary only).
- Compiler: [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle),
  [Bounded general-scan cold reconstruction](../../../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction).
- Recurrent scan: [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan),
  including [Source body, attachments and scoped identities](../../../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities),
  [Canonical residual ports and demand](../../../../architecture/contracts/recurrent-scan.md#canonical-residual-ports-and-demand),
  [Reverse scan and exact primal bindings](../../../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings),
  [Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification),
  [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates).

Current v1/v2/v3 remain body-free. Stop on ambiguous authority or any new production type.

## Dependencies and integration

- Depends on: Compiler0012, CP0010/0011/0012 and NN0031/CPU0030–0036/low-level and owned-workflow
  joins Complete; required predecessor reviews/validation accepted. Schema/product proposal approved.
- Conflicts with: Checkpoint wire/source/limits, Compiler cold inventory, NN persistence, scan and
  root/owning/shared-document writes; serial, no other writer active.
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus
  292 accepted dirty paths (291-path baseline + ADR0047), empty index, verified 2026-10-23; preserve all.
- Integration order: accepted authority → verified clean codec launch → worker yield → independent
  code/docs APPROVE → final module/Javadoc → named integration/architecture → one configured root;
  all complete, then factual closure. No successor launch.
- Integration validation: recorded commands below PASS; final documentation checks/preservation,
  with successful executable evidence reused and no closure reruns.
- Shared-document integration owner: main; exclusive sequential delegation.

Launch verification retained: all predecessors/reviews/NN0031 configured join Complete, matching
HEAD/origin, empty index, accepted authority including Cicero's P2 fix, unchanged source and inactive
conflicts. CP0012's earlier environment exception remains historical. Final preservation is in Result.

## Files and source evidence

Java prefix: `src/main/java/io/github/pho001/synaptik/`; all implementation writers are closed.

- `modules/checkpoint`, `checkpoint.graph`: `GraphCheckpoint.save/load`, `GraphArtifactCodec`
  encode/decode/write/opcode/canonicalTopology/Reader/Sink (selective v4); `GraphSource.capture`
  (outer-only normalization/attachments/roles/aliases); `GraphValueStorage` (seven raw dtypes/host reads).
- `modules/compiler`, `compiler`: `GraphCompilationPort.SourceCapture/reconstructSource`,
  `SourceGraphReconstruction.preflight/bodyAdditional/reconstructScan`, `ScanBodyInference.capture/infer/validateResiduals`,
  `GraphCapture.captureBody/captureSource` (existing cold seams).
- `modules/model`, `model.graph.{CompiledGraphModel,CompiledNode,CompiledScanBody}`, `model.operation.scan.ScanAttrs`,
  `model.tensor.{Scan,ScanBody,ScanResult}` metadata. `extensions/nn`,
  `nn.module.ModelExecutionState.save` / `Model.load/loadGraph`: consumers, no new wire.
- Focused tests: `GraphCheckpointGeneralScanTest`, `GraphCheckpointTest`, `SourceGraphReconstructionTest`,
  `GraphArtifactCodecTest`/`GraphArtifactV3Test`; integration `GeneralScanOwnedModelIntegrationTest`
  (loaded compute/train/save), `FluentModelBuilderIntegrationTest` (static persistence).

Authorized implementation scope (same existing Java/test packages, now closed):

- `modules/checkpoint`, `checkpoint.graph.GraphArtifactCodec.java`: executable codec and Javadoc.
  `GraphCheckpoint.java`: Javadoc only; existing `GraphSource.java`/`GraphValueStorage.java`: optional
  relevant Javadoc only, no behavior changes.
- `extensions/nn`, `nn.module.{Model,ModelExecutionState,OwnedModelSource}.java`: Javadoc only.
- Checkpoint `src/test/java/.../checkpoint/graph`: focused GraphCheckpointGeneralScanTest,
  GraphCheckpointTest, GraphArtifactCodecTest, GraphArtifactV3Test and new **test-only** GraphArtifactV4Test.
- NN `src/test/java/.../nn/module/GeneralScanOwnedModelTest.java`; integration existing
  `src/test/java/.../testing/integration/{GeneralScanOwnedModelIntegrationTest,GraphCheckpointEngineIntegrationTest}.java`
  as needed. FluentModelBuilderIntegrationTest is a read-only static-persistence regression oracle.

No Compiler/Model/Engine/Runtime/Prepare/Backend Java, native/build/dependency changes.
All other Java/tests read-only. New production types/public API: None; no new abstraction.

## Approved source and wire

The enacted [Canonical graph-model version 4](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-4)
is the sole wire/bounds/failure-order reference; do not duplicate or extend its table here.
It selects v4 for general SCAN with exactly the existing bounded cold inventory and keeps old
v1/v2/v3 bytes/readers unchanged. No RNN-only restriction, all-PURE expansion or CPU dtype filter.
Authority, codec, documentation, independent review and main validation are complete.

## Bounds, reconstruction and observable acceptance

- Validate depth-one scopes, complete role/formal/port metadata, checked JVM budgets/static geometry
  and every dry-sized field under the owning v4 contract. All six ArtifactLimits precede host
  admission/copy; no backend rank-16 bound, peak-heap promise or time-unrolling allowance.
- Decode all wire/keys/aliases/BOOL/descriptors/topology before host import. Reject dead producers,
  duplicate definitions/roots, forward/dangling/cyclic/cross-scope refs, nested scans, free leaves,
  missing/extra attachments, malformed roles/residuals, nonempty demand and trailing bytes.
  Compiler SourceCapture preflights the complete outer graph/every body using descriptor-only
  candidate leaves before import, and before any body formal/producer identifiers are allocated.
- Invoke existing Compiler reconstruction once for the outer source: it constructs each scoped
  body/node once with fresh canonical formals and Scan.apply per occurrence, including T=0; no
  Checkpoint Tensor reconstruction table. Normalized full recapture equality precedes publication.
  Save/load create no Engine; copy/rebind unchanged. Validation failure changes no source/state/target
  and returns no partial result. Retain ArtifactFile uncertain-move/failure/cleanup, not new durability.
- Test valid last records as well as late-invalid bodies after valid outer work, zero time/empty
  dimensions, multiple occurrences/reused templates, unused formals, repeated roles/actuals, all
  ports/aliases/raw bits and applicable cold dtypes. Exact/one-below bounds/malformed fixtures and
  storage/ID probes prove rejection before copy/import and formal/producer construction.
- Save before compute/after training; close source, load, cold-compose, compute/train fresh batches
  and re-save F32/F64 against independent unrolled oracles. Retain static-RNN and old-format tests;
  low/other dtype cold roundtrip is not execution qualification or exact Training continuation.

## Documentation, review and validation

Root + one extensions-training owner and [ADR 0047](../../../../design/decisions/0047-bounded-general-scan-graph-persistence.md)
are accepted; Compiler/recurrent edits were status-only. Scan/reconstruction/ABI/backend/generator/
numerical/AD/padding/NN-migration semantics stay unchanged. Stop on missing seams or ambiguity.
Worker yielded frozen implementation; Cicero finalized 11 authority/explanatory documents, confirmed
accurate Javadocs and gave final independent APPROVE without Java review edits. Main's gates below
followed the source/Javadoc freeze. Closure changes status only, not rules, examples or Java.

Final affected-module commands, once after code/docs/Javadocs stabilized:

```bash
./gradlew :extensions:checkpoint:test :extensions:checkpoint:javadoc :extensions:nn:test :extensions:nn:javadoc --max-workers=2 --console=plain
```

Main consumer checkpoint:

```bash
./gradlew :testing:integration-tests:test --tests '*GeneralScanOwnedModelIntegrationTest' --tests '*GraphCheckpointEngineIntegrationTest' --tests '*FluentModelBuilderIntegrationTest' --max-workers=2 --console=plain
```

Loaded single/list compute/train and old static persistence are covered. Main also ran
`./gradlew :testing:architecture-tests:test --max-workers=2 --console=plain`, then ONE configured
root, reusing unchanged successful module/Javadoc results:

```bash
SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc --max-workers=2 --console=plain -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false
```

No native build, timing campaign or backend/conformance widening. Actual counts/skips are in Result;
closure reuses evidence, with no Gradle rerun.

<a id="frozen-implementation-handoff--2026-10-23"></a>

## Result

Status: Complete — main accepted final Cicero APPROVE (zero P1/P2) and all gates, 2026-10-23.

- Delivered selective v4/scoped body persistence, checked bounds/preflight/recapture and loaded
  F32/F64 single/list compute/train/save against unrolled oracles; raw bits, aliases and old formats retained.
- Executable Java: GraphArtifactCodec only. Javadoc-only: GraphCheckpoint, NN Model/ModelExecutionState.
  Tests: GraphCheckpointGeneralScanTest, GraphCheckpointTest, new test-only GraphArtifactV4Test,
  NN GeneralScanOwnedModelTest and GeneralScanOwnedModelIntegrationTest. Other Java unchanged.
- 16 authority/planning paths plus those nine Java/test paths comprise the exact 25 task deltas.
  Independent review finalized 11 docs; Javadocs accurate, no Java review edits/new terminology.
- Focused: 63 distinct cases PASS, including old-wire 17; strengthened reruns overlap, not additive.
  Development fixture/compile/substitution failures were resolved; complete history remains in logs.
- Final module test/Javadoc PASS 26s: checkpoint 61 + NN 542 = **603 fresh cases**, zero failures/errors/skips;
  both Javadocs passed, 7 tasks executed/33 up-to-date. Named consumer PASS 10s: 7+5+4=16 cases.
- Architecture PASS 2s: 25 cases. ONE configured root PASS 18s/exit 0: 119 tasks, 1 executed/118 reused;
  full integration reran, unchanged module/Javadoc evidence reused. No subsequent Java edits.
- Current-project root XML: 684 suites/4791 reported = 4742 passed + 49 skipped, zero failures/errors;
  includes cached results, excludes 13 obsolete pre-relocation XML. Integration 210/209 passed/1 skipped.
- Six required native cases present/pass/unskipped; two Metal training cases fresh, four Exp/Sigmoid reused.
  Existing source-matched library unchanged (ABI7/schema20). Existing FFM/vector/final-field warnings only.
- Evidence: `final-validation.txt`, `final-*-gate*.log`, `implementation-evidence.txt` and XML snapshots
  under the evidence directory above. Codec SHA-256 `b7191579d03236e900f4fb4836710c42b23a7e1c221b1d934963b513645342bb`.
- Main verified 2421 files/exact 25 deltas, all baseline outside preserved, HEAD/origin unchanged/index empty.
  No unresolved issue/follow-up. NN0032 stays Draft at its separate backend/layout gate; all frontiers None.
  No padding/AD/backend/migration or optimizer-resume grant; no native build/campaign/commit/push/successor.
