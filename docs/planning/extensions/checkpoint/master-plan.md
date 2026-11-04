# [Checkpoint](../../../glossary.md#term-checkpoint) Extension Master Plan

## Goal and authority

Accepted whole-[model](../../../glossary.md#term-model) product: store a supported declarative model and load it automatically for
[inference](../../../glossary.md#term-inference). Training-owned [resume](../../../glossary.md#schema-serialization-and-checkpoint) from model + recipe + TrainingState remains Proposed.
Normal usage needs no restore [callback](../../../glossary.md#term-callback), manual model construction, config [digest](../../../glossary.md#term-digest), ArtifactEnvelope
or [Arena](../../../glossary.md#term-arena). Existing explicit APIs remain the advanced foundation.
This is non-authoritative coordination; [root architecture](../../../../ARCHITECTURE.md) wins.
[ADR 0038](../../../design/decisions/0038-declarative-model-and-training-resume.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../glossary.md#term-architecture-decision-record-adr))</span>
replaces the withdrawn Proposed [snapshot](../../../glossary.md#term-snapshot)-only direction of
[ADR 0037](../../../design/decisions/0037-stored-training-checkpoint-adapter.md).
The user approved CP0005B direction; the earlier recurrent-first successor ordering is now on hold.
CP0005B's coordinated contract is accepted; final independent approval and coordinator
verification authorized implementation; CP0005B is Complete after final reviews/gates (2026-10-15).
Recurrent/Training/adapter decisions remain Proposed.

Completed direction: [ADR 0039 model execution + persistence](../../../design/decisions/0039-portable-graph-model-persistence.md)<span data-glossary-navigation="true"> ([persistence glossary](../../../glossary.md#term-persistence))</span>
and [Complete CP0005D planning/review](tasks/0005d-portable-graph-persistence-decision.md) record the user-approved
direction: the same implemented model.compute/train/save/load for manual and file-restored [tensors](../../../glossary.md#tensor).
The [cross-area program](../../model-execution-refactor.md) owns phases, [target](../../../glossary.md#term-target) paths/edges and the
independently approved P2 [planning](../../../glossary.md#term-planning) packet. Complete Checkpoint0008 implements generic [modules](../../../glossary.md#term-module)/[checkpoint](../../../glossary.md#term-checkpoint)
and the [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) glue relocation with the coordinated reverse-edge cutover; current contracts enact those bounded edges.
Hold NN0027/CP0005C/Training/adapter expansion; completed closed-NN v1 stays supported.

## Contracts and common base

- Root: authority/[scope](../../../glossary.md#term-scope) index, [module](../../../glossary.md#term-module) [ownership](../../../glossary.md#term-ownership-meanings), [dependency](../../../glossary.md#term-dependency) rules and testing requirements.
- Owning [extensions contract](../../../architecture/contracts/extensions-training.md):
  extensions/nn, extensions/training, extensions/checkpoint and Optimizer/training lifecycle.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../glossary.md#term-checkpoint); [Optimizer glossary](../../../glossary.md#term-optimizer); [lifecycle glossary](../../../glossary.md#lifecycle))</span>
- Historical CP0005B execution base/accepted packet remains in its completed brief.
- CP0012 common base: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus
  274 accepted dirty paths; historical Ready verification, baseline and scoped acceptance are in its brief.<span data-glossary-navigation="true"> ([Ready glossary](../../../glossary.md#term-ready))</span>
- [CP0013](tasks/0013-general-scan-graph-persistence-decision.md) base: same HEAD/origin plus
  292 accepted dirty paths (291-path baseline + ADR0047), empty index (2026-10-23).
  Historical verified launch base; implementation, final Cicero APPROVE and main gates now accepted.
- [Whole-model contract](../../../architecture/contracts/extensions-training.md#whole-model-definition-and-composition)<span data-glossary-navigation="true"> ([model glossary](../../../glossary.md#term-model))</span>
  accepts CP0005B API/wire/defaults/failures/preflight; no dependency changes.<span data-glossary-navigation="true"> ([API glossary](../../../glossary.md#term-application-programming-interface-api); [preflight glossary](../../../glossary.md#term-preflight))</span>
- Current published values-only `model.v1` and ArtifactFile v1 remain unchanged.
  [ADR 0036](../../../design/decisions/0036-model-envelope-composition.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../glossary.md#term-architecture-decision-record-adr))</span> remains implemented.

## [Scope](../../../glossary.md#term-scope) and non-goals

COMPLETE (Checkpoint0008, 2026-10-17): generic [Checkpoint](../../../glossary.md#term-checkpoint) in `modules/checkpoint` owns [opaque](../../../glossary.md#term-opaque)
files and bounded ordinary [Tensor](../../../glossary.md#tensor) [graph](../../../glossary.md#graph)/value [persistence](../../../glossary.md#term-persistence) with exactly `api(Model)` and cold
`implementation(Compiler)` edges; no [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training/[Engine](../../../glossary.md#term-engine) [dependency](../../../glossary.md#term-dependency). NN owns declarative
configuration/reconstruction and the relocated `nn.checkpoint` [facade](../../../glossary.md#term-facade)/codecs, with exactly public
[Model](../../../glossary.md#term-model)/Training/Checkpoint/Engine edges. The old Checkpoint→NN edge is removed; v1 readers and
behavior are retained with explicit import/dependency migration and no compatibility forwarders.
Halley's final independent APPROVE and the coordinator's source-matched test/[Javadoc](../../../glossary.md#term-javadoc) gate passed.

Training recipe/owned-[run](../../../glossary.md#term-run) and optional adapter below are held ADR0038 proposals, not current APIs
or [target](../../../glossary.md#term-target) authority. No arbitrary Java [serialization](../../../glossary.md#term-serialization)/reflection/[registry](../../../glossary.md#term-registry), [backend](../../../glossary.md#backend) discovery,
compiled/[device](../../../glossary.md#term-device)-state persistence, general transactions or mandatory ModelExecutor.

Initial support is implemented NN0026 nested Sequential, [Linear](../../../glossary.md#term-linear-meanings)/LayerNorm/[Embedding](../../../glossary.md#embedding)/Conv1d/2d/3d and
closed RELU/[TANH](../../../glossary.md#term-tanh)/[SIGMOID](../../../glossary.md#term-sigmoid) steps. CP0005B retains that exact [matrix](../../../glossary.md#term-matrix-meanings)/attributes/guards/[aliases](../../../glossary.md#aliasing-and-arena).
Custom Model.define callbacks/record I/O stay explicit. Declarative branches/contextual layers are
named successors, not a speculative detailed cascade. Fixed RecurrentScan [BPTT](../../../glossary.md#term-backpropagation-through-time-bptt)/[Conv3d](../../../glossary.md#conv3d--ncdhw) [gradients](../../../glossary.md#term-gradient) and training-mode
BatchNorm [buffer](../../../glossary.md#term-buffer) [alignment](../../../glossary.md#term-alignment) remain [fail-closed](../../../glossary.md#term-fail-closed); no changing per-step [dropout](../../../glossary.md#dropout--inverted-dropout) or exact cursor/[RNG](../../../glossary.md#graph-rng-state--graphrngstate) claim.
Ordinary static recurrent [unrolling](../../../glossary.md#term-unrolling) depends on normal constituent-[operation](../../../glossary.md#operation) adjoints; no blanket ban.
The closed Sequence tree explicitly determines execution order; arbitrary callbacks are not captured.
[Inference](../../../glossary.md#term-inference) aliases must reconstruct [identity](../../../glossary.md#term-identity-meanings) or reject; training rejects repeated [parameter](../../../glossary.md#term-parameter) paths/
overlapping stores, not every repeated forward use of one uniquely owned layer.

## Package map

| Package | Delivered or proposed [owner](../../../glossary.md#term-owner) |
|---|---|
| `checkpoint.artifact` | Existing ArtifactEnvelope, ArtifactFile, ArtifactLimits and [publication](../../../glossary.md#term-publication). |
| `checkpoint.model` → `nn.checkpoint` | Checkpoint0008 moves nine existing [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [persistence](../../../glossary.md#term-persistence) types; behavior/readers remain unchanged. |
| `checkpoint.graph` | Checkpoint0008: GraphCheckpoint with nested Source/[Leaf](../../../glossary.md#term-leaf)/Role; private GraphSource, GraphValueStorage, GraphArtifactCodec (optional private Reader/Sink). No runner or second [graph](../../../glossary.md#graph) representation. |
| `nn.module` | Implemented ModelSpecification/DeclarativeModel (NN 0026); existing [callback](../../../glossary.md#term-callback) ModelDefinition unchanged. |
| `training` | Existing snapshots/session; proposed TrainingRecipe/TrainingRun and narrow [capture](../../../glossary.md#term-capture-meanings) borrow (Training 0006). |
| `training.checkpoint` | Proposed TrainingCheckpoint/[Snapshot](../../../glossary.md#term-snapshot) and private TrainingState/recipe composition codecs in optional project (0006). |

Package names are relative to `io.github.pho001.synaptik`. CP0005B names/[API](../../../glossary.md#term-application-programming-interface-api) are accepted;
Training/adapter names remain Proposed.

## [Dependency](../../../glossary.md#term-dependency) [DAG](../../../glossary.md#term-directed-acyclic-graph-dag) and [authorized frontier](../../../glossary.md#term-authorized-frontier)

Completed chain: [Checkpoint](../../../glossary.md#term-checkpoint) 0001 → 0002 → 0003 → 0004; [CPU](../../../glossary.md#term-cpu) 0024 → 0004.
Training 0004 + Checkpoint 0004/CPU 0024 → 0004A → Training-owned 0005 → 0005A.

Earlier proposed capability program (held after its completed prefix):

```text
NN0026 Complete → CP0005B Complete | HOLD: NN0027 → CP0005C → Training0006 → optional CP0006
completed CP0005A ────────────────────────────────────────────────────────────────────┘
```

The [program DAG](../../model-execution-refactor.md#phases-and-owner-dag) replaces the old
cold-reconnection-first proposal. CP0005D is Complete for bounded [planning](../../../glossary.md#term-planning)/review after independent
Astra/xhigh review, the parent-project mapping correction and coordinator acceptance. Checkpoint0008
is Complete after final independent approval and the coordinator gate; successor 0009 is also Complete below. The [backend](../../../glossary.md#backend)/native-only
[Engine0020 cutover](../../modules/engine/tasks/0020-backend-native-layout-relocation.md) is Complete
after integration/review/combined [validation](../../../glossary.md#term-validation). The user's subsequent five product choices are now
accepted. Engine0021 delivered thin-session removal, coordinated authority and mechanical consumers;
independent Class C approval and the native-enabled test/[Javadoc](../../../glossary.md#term-javadoc) gate passed. Its brief owns evidence.
Checkpoint0008 is Complete after Halley's definitive APPROVE and coordinator acceptance on 2026-10-17;
its brief owns scoped and repository test/Javadoc evidence. Training0008 and NN0028 are Complete
after independent reviews and coordinator gates; their bounded downstream contracts are enacted.
The selected program remains Complete. Checkpoint0009 is a separately accepted Complete successor.

Checkpoint0008 separated the [opaque](../../../glossary.md#term-opaque) [artifact](../../../glossary.md#term-artifact) infrastructure and generic [graph](../../../glossary.md#graph)/value [codec](../../../glossary.md#term-codec) from
[NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-specific glue, retained old v1 readers, and obtained coordinated cold [Compiler](../../../glossary.md#term-compiler) access.
Compiler0008 already enacted bounded reconstruction authority. Load reconstructs ordinary [Tensor](../../../glossary.md#tensor) expressions/[weights](../../../glossary.md#term-weight) without
[Engine](../../../glossary.md#term-engine) or future [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) before [compile](../../../glossary.md#compile). Normal repeated [model](../../../glossary.md#term-model).compute/train uses stable prepared
[signatures](../../../glossary.md#term-signature) through existing Engine [PreparedExecution](../../../glossary.md#prepared-execution--preparedexecution)/[Engine.runValues](../../../glossary.md#fresh-value-binding--enginerunvalues) and fresh per-[run](../../../glossary.md#term-run) storage
snapshots, not outputs(actualInputs) reconstruction or a replacement session [facade](../../../glossary.md#term-facade).
Engine0021 removed [InferenceSession](../../../glossary.md#former-inference-session--inferencesession)/Engine.session; old file readers are unaffected.
Completed Checkpoint0008 depended on Compiler0008 and Training0007; it moved the six artifact files
to [modules](../../../glossary.md#term-module)/checkpoint and nine NN glue classes to extensions/nn under nn.checkpoint. Retain
:extensions:checkpoint via projectDir; generic Checkpoint has api(Model), implementation(Compiler),
no NN/Training/Engine edge. NN owns old Engine-materializing ModelCheckpoint behavior and readers.
The new GraphCheckpoint Source/[Leaf](../../../glossary.md#term-leaf)/Role boundary carries ordinary Tensor sources, not a runner.
[Host](../../../glossary.md#term-host)-only values allow cold copy/save/load without Engine; owned loaded values are adopted once.
The original selective v1/v2 codecs covered ADD/MUL/NEG/[MATMUL](../../../glossary.md#matrix-multiplication--matmul)/PERMUTE[1,0]/RELU/[TANH](../../../glossary.md#term-tanh)/[SIGMOID](../../../glossary.md#term-sigmoid).
Accepted v3-v5 successors below add selected flat [CNN](../../../glossary.md#term-convolutional-neural-network-cnn)/recurrent and bounded general-[scan](../../../glossary.md#term-cumulative-scan) sources,
not an all-family codec campaign.
Exact APIs, [ownership](../../../glossary.md#term-ownership-meanings) and source guards live in current root/scoped authority; program/[ADR](../../../glossary.md#term-architecture-decision-record-adr) explain them.

Current Checkpoint source frontier: **None**. [CP0013 Result](tasks/0013-general-scan-graph-persistence-decision.md#result)
is Complete, accepted by main 2026-10-23 after final independent Cicero APPROVE and module/Javadoc,
named consumer, architecture and one configured root PASS. User-approved selective v4/existing cold
inventory is delivered; no new [API](../../../glossary.md#term-application-programming-interface-api)/type/dependency. The brief owns exact evidence, historical launch
facts and all seven fields. Serial, Parallel group None; main owns integration, no successor launch.
NN0032 remains Draft at its independent backend/[layout](../../../glossary.md#layout) gate. CP0012's environment exception stays historical.

[Checkpoint0014](tasks/0014-length-aware-general-scan-v5.md#result) is Complete under continuous
production delivery after accepted Compiler0018. Independent exact-wire/final code review,
619 affected tests/Javadocs and public loaded [F32](../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../glossary.md#floating-point-precision-and-rounding) Model numerical proof passed. Selective
v5/opcode33 preserves old records; direct encoder [preflight](../../../glossary.md#term-preflight) reuses Compiler before sizing/storage.
MAIN retains the frozen base/integration evidence; NN0033 owns the next recurrent-composition [slice](../../../glossary.md#slice).
[NN0031 Result](../nn/tasks/0031-general-scan-owned-model-workflow.md#result) records later Complete
owned acquisition/compute/train [qualification](../../../glossary.md#term-qualification) and the successful configured native program join;
body [persistence](../../../glossary.md#term-persistence) is now separately delivered by CP0013. CP0005B is Complete after final independent code/docs review,
57 module tests, final Javadoc and coordinator architecture/3-test integration gates. Contract approval and
2026-10-14 coordinator base/dependency/contract/frontier/conflict verification are recorded in its
brief. NN0027/CP0005C remain Draft rows only (no speculative briefs).
No task waits on optimized scan/backend [BPTT](../../../glossary.md#term-backpropagation-through-time-bptt)/Data/Text or broader Training checkpoint.
Each implementation/review uses separate clean contexts; parallel group None for now because
shared contract cutovers need [serialization](../../../glossary.md#term-serialization). The new [activation](../../../glossary.md#term-activation) successor is [atomic](../../../glossary.md#resource-transaction-and-rollback), not another
phase of the completed program or part of the held recurrent chain.
Any later launch requires a separately accepted contract packet and verified [Ready](../../../glossary.md#term-ready) brief.
CP0005B's root/scoped contract/explanations/ADR are accepted; future recurrent policies,
Training bridges/recipe and adapter wire need their own decisions. Build inclusion happens at implementation,
not in this plan.

## Task list

Completed joint [DAG](../../../glossary.md#term-directed-acyclic-graph-dag): Compiler0008/0009 + Checkpoint0008/0009 + Training0008 +
NN0028/0029 plus Model0025E/0025F/Compiler0006A → [CP0010](tasks/0010-fluent-cnn-and-static-recurrent-graph-persistence.md)
with [CP0011 scan supplement](tasks/0011-flat-recurrent-scan-persistence.md), on published CPU0027 base.
CP0010/CP0011 Complete together on 2026-10-20: bounded [CNN](../../../glossary.md#term-convolutional-neural-network-cnn)/static recurrence and cold [scan](../../../glossary.md#term-cumulative-scan) v3
[persistence](../../../glossary.md#term-persistence) accepted after independent APPROVE and main's focused/native root gates;
[CP0010 Result](tasks/0010-fluent-cnn-and-static-recurrent-graph-persistence.md#result) owns evidence.
NN0030 approved decision + Compiler0012 + CP0008–0011/low-level join Complete →
[CP0012 Complete](tasks/0012-general-scan-in-memory-copy-and-rebind.md#result), accepted 2026-10-23:
in-memory copy/[rebind](../../../glossary.md#capture-graph-ports-and-reconstruction), independent disk rejection and then-temporary compatibility-only [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) early guards.
Clean implementation/independent code/docs review and [module](../../../glossary.md#term-module)/required consumer gates passed;
main accepted scoped completion with unrelated missing-native-environment root failures, not a green
unrestricted root at that [checkpoint](../../../glossary.md#term-checkpoint). Its brief preserves that evidence. Later
[NN0031](../nn/tasks/0031-general-scan-owned-model-workflow.md#result) completed the separate owned
workflow gate, removed those temporary guards and passed the single configured native join using
the existing library; no native/gating code or build changed. The explicit library environment is
still required. At that join all frontiers were None; CP0013 has now separately completed body
persistence and its final configured root gate. Current frontiers None; no successor.

Completed rows preserve published [API](../../../glossary.md#term-application-programming-interface-api) history; unfinished rows describe future capability only.

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| [0001](tasks/0001-checkpoint-module-boundary.md) | [Model](../../../glossary.md#term-model) Checkpoint module boundary | Complete | None; architecture decision is the deliverable | Empty project, contract, [ADR](../../../glossary.md#term-architecture-decision-record-adr), explanations, and [dependency](../../../glossary.md#term-dependency) test reviewed; focused and repository gates passed. No persistence API or optional adapter. |
| [0002](tasks/0002-bounded-artifact-envelope.md) | Bounded [artifact envelope](../../../glossary.md#term-artifact-envelope) | Complete | 0001 Complete; corrected v1 policy in brief | [Opaque](../../../glossary.md#term-opaque) file API, bounds, [SHA-256](../../../glossary.md#term-sha-256), file flush and [atomic live publication](../../../glossary.md#artifact-envelope-and-atomic-publication) implemented; 15 checkpoint tests, [Javadoc](../../../glossary.md#term-javadoc), [architecture test](../../../glossary.md#term-architecture-test), and independent Class C review passed. |
| [0003](tasks/0003-model-schema-and-materialization-handoff.md) | Model [schema](../../../glossary.md#term-schema) and [Engine](../../../glossary.md#term-engine) [materialization](../../../glossary.md#materialization) [handoff](../../../glossary.md#term-handoff) | Complete | 0002 Complete; existing NN export and Engine compute | Internal fixed-ID schema, [detached](../../../glossary.md#detached-model-results) candidate, and bounded public Engine handoff implemented; 22 module tests, Javadoc, and independent Class C review passed. No public save/load. |
| [0004](tasks/0004-public-model-checkpoint-save-inspect-load.md) | Model checkpoint save, inspect, and strict load | Complete | 0003; [CPU 0024](../../backends/cpu/tasks/0024-default-segment-route-heap-carrier-binding.md) for direct restored-[heap](../../../glossary.md#term-heap) execution | Public save/inspect/load and direct restored-heap Engine [regression](../../../glossary.md#term-regression) passed, with independent review, documentation, and combined [validation](../../../glossary.md#term-validation). |
| [0004A](tasks/0004a-direct-leaf-model-save.md) | Direct [leaf](../../../glossary.md#term-leaf)-model save | Complete | Checkpoint 0004; [CPU](../../../glossary.md#term-cpu) 0024; Training 0004, all Complete | Exact native/heap leaf [capture](../../../glossary.md#term-capture-meanings), direct trained save/load and loaded re-save passed; workaround removed, independent review and configured coordinator gate passed. |
| [0005](tasks/0005-training-state-reconstruction.md) | Training-owned detached state reconstruction prerequisite | Complete | Training 0004; Checkpoint 0004A, both Complete | Three defensive constructors exposed; external-package tests, final Javadoc/example validation, and independent API review passed. No [codec](../../../glossary.md#term-codec). |
| [0005A](tasks/0005a-model-envelope-composition.md) | Model-envelope composition | Complete | 0004A and 0005 Complete | Four existing-ArtifactEnvelope operations implemented; independent Class C API/documentation review and configured coordinator gates passed; file semantics and module edges unchanged. |
| [0005B](tasks/0005b-whole-model-save-load.md) | Whole-model save and automatic [inference](../../../glossary.md#term-inference) load | Complete | NN0026; CP0002/0004/0004A/0005A Complete; accepted CP contract | Ten overloads/typed [identity](../../../glossary.md#term-identity-meanings) and closed codec/composition; independent reviews, 57 module tests, final Javadoc and coordinator gates passed. |
| 0005C | Versioned recurrent definition and whole-model persistence adoption | Draft | CP0005B; NN0027; separately selected recurrent [scope](../../../glossary.md#term-scope) | Optional held proposal, not a new [graph](../../../glossary.md#graph)-path prerequisite; no detailed speculative brief. |
| [0005D](tasks/0005d-portable-graph-persistence-decision.md) | Model execution/persistence program decision | Complete | NN0026/CP0005B Complete; approved direction/[layout](../../../glossary.md#layout) | Independent review, sole P2 correction and bounded coordinator acceptance; Engine0020 [Ready](../../../glossary.md#term-ready), semantic defaults still Proposed. |
| [0006](tasks/0006-training-checkpoint-adapter.md) | Optional whole-model TrainingCheckpoint | Draft | NN0026/0027; CP0005B/0005C; Training0006; completed 0005A/Training0004 | Proposed high-level save/loadModel/[resume](../../../glossary.md#schema-serialization-and-checkpoint); separate contract gate. |
| 0007 | Broader recovery/text checkpoint | Draft | 0005B/0006; Text 0003 only for text integration | Later broader corruption/recovery/text validation; not an extra implementation prerequisite or duplicate gate. |
| [0008](tasks/0008-generic-graph-value-persistence-and-nn-glue-cutover.md) | Generic graph/value persistence and NN glue cutover | Complete | Engine0021; Training0007; Compiler0008; CP0005B; accepted P2 Complete | Bounded graph-model.v1 and six/nine-file split delivered; old readers retained, Halley final APPROVE and coordinator test/Javadoc gate passed. |
| [0009](tasks/0009-owned-model-activation-source-and-persistence.md) | [Owned Model](../../../glossary.md#model-module-buffer-and-scan-distinctions) [activation](../../../glossary.md#term-activation) source and persistence | Complete | Compiler0008; Checkpoint0008; Training0008; NN0028 Complete | Eight-kind/selective-v2 capability accepted 2026-10-19 after Archimedes final APPROVE and main's integration/native root gates; evidence in brief. |
| [0010](tasks/0010-fluent-cnn-and-static-recurrent-graph-persistence.md) | Fluent CNN/static recurrence and joint v3 foundation | Complete | Completed [Compiler](../../../glossary.md#term-compiler)/Checkpoint/Training/NN predecessors; joint approval accepted | Joint CP0011 acceptance after independent APPROVE and main's focused/native root gates; evidence in brief. |
| [0011](tasks/0011-flat-recurrent-scan-persistence.md) | Flat scan persistence supplement to CP0010 | Complete | Same packet/predecessors plus completed Model0025E/0025F/Compiler0006A | Cold-only scans accepted jointly with CP0010; shared evidence in CP0010 Result. |
| [0012](tasks/0012-general-scan-in-memory-copy-and-rebind.md) | General-scan in-memory copy and rebind | Complete | NN0030 approved decision; Compiler0012; CP0008–0011 and accepted low-level join Complete | Scoped acceptance with historical native-environment root exception; later [NN0031](../nn/tasks/0031-general-scan-owned-model-workflow.md#result) qualified owned acquisition/training and passed configured native join. Body wire stays closed. |
| [0013](tasks/0013-general-scan-graph-persistence-decision.md#result) | Bounded general-scan graph persistence | Complete | Compiler0012; CP0010/0011/0012; NN0031/CPU0030–0036/joins Complete; user proposal/authority accepted | Selective v4, scoped cold bodies and loaded consumer proof; final Cicero APPROVE/main module/Javadoc, consumer, architecture and configured root PASS. No successor. |
| [0014](tasks/0014-length-aware-general-scan-v5.md#result) | Length-aware [general Scan](../../../glossary.md#general-differentiable-scan) v5 | Complete | Compiler0018 and CP0013 Complete; independently approved exact wire/implementation | Selective opcode33, bounded pre-import validation, 619 affected tests/Javadocs and public loaded [F32](../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../glossary.md#floating-point-precision-and-rounding) Model training proof. |

Exact conflict/base/order/validation [metadata](../../../glossary.md#term-metadata) live in briefs. Checkpoint0008/0009 are Complete;
CP0010–CP0014 are Complete; current Checkpoint source frontier is None.
0008 had Parallel group None; completed serial order Engine0021 → Training0007 → Compiler0008
→ Checkpoint0008 → Training0008 → NN0028. Shared root/owning-contract/NN-state/Checkpoint-wire writes
conflict; main model-execution coordinator owns shared documents with exclusive-writer delegation.
The brief records all seven metadata fields and focused plus one repository integration validation.

CP0005C ordering metadata (held, revalidate if recurrent scope is separately selected): Depends on CP0005B and NN0027; Conflicts with shared Checkpoint wire/API
and NN semantic contract cutovers/Training0006/adapter0006; Parallel group None; Common base revision
`f892cd3fbde4f8b7722b7eb5b14f4b0a5f3b2669` (reverify after predecessors); Integration order after
NN0027 and before Training0006; Integration validation future affected Checkpoint round trips plus
NN coherence/architecture at its join; Shared-document integration [owner](../../../glossary.md#term-owner) NN/Training/Checkpoint
coordinator. NN owns semantic reconstruction only; this split follows [ownership](../../../glossary.md#term-ownership-meanings), not mechanical
API/test staging. Initial-state/output-selection, format version and Training [gradient](../../../glossary.md#term-gradient) [admission](../../../glossary.md#term-admission)
remain future decisions. CP0005B v1 supports no recurrent files.

## [Atomic](../../../glossary.md#resource-transaction-and-rollback) [activation](../../../glossary.md#term-activation)-source successor

Compiler0008 + Checkpoint0008 + Training0008 + NN0028 (all Complete) →
[Checkpoint0009](tasks/0009-owned-model-activation-source-and-persistence.md). This completed Class C
capability adds RELU/[TANH](../../../glossary.md#term-tanh)/[SIGMOID](../../../glossary.md#term-sigmoid) to the five-kind source [slice](../../../glossary.md#slice), with owned-[Model](../../../glossary.md#term-model) construction and
direct-[oracle](../../../glossary.md#term-test-oracle) [MLP](../../../glossary.md#term-multilayer-perceptron-mlp) compute/train/save/load coverage. [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) remains a consumer; no new production types,
public [signatures](../../../glossary.md#term-signature), [dependency](../../../glossary.md#term-dependency) edges, [runtime](../../../glossary.md#term-runtime) behavior or numerical rules changed.

[Compiler](../../../glossary.md#term-compiler) owns cold [admission](../../../glossary.md#term-admission)/reconstruction; [Checkpoint](../../../glossary.md#term-checkpoint) owns explicit [fail-closed](../../../glossary.md#term-fail-closed) wires. The former
encoder fallback to PERMUTE required the atomic cutover. Current [graph](../../../glossary.md#graph)-model.v2 appends wires
6/7/8 for RELU/TANH/SIGMOID under SYNGRF02/version 2; five-kind saves retain exact v1.
All old readers remain; old graph readers reject
v2. The brief owns the complete compatibility policy and required authority cutover before Java.

Checkpoint0009 is Complete, accepted 2026-10-19 after Archimedes's final Astra/xhigh APPROVE,
zero findings, and main's source-matched selected integration/native root gates. The brief owns evidence.
Authorized Checkpoint/Compiler/NN frontiers: None. No follow-up, separate Compiler0009 or reopened
selected program. Common base and serial integration [metadata](../../../glossary.md#term-metadata) remain in the brief.

## Selected whole-[model](../../../glossary.md#term-model) and proposed Training flow

The v1 description below remains current; the future Training flow is the held ADR0038 proposal,
not a commitment to extend layer-family [persistence](../../../glossary.md#term-persistence) before a separately selected recurrent/Training-[resume](../../../glossary.md#schema-serialization-and-checkpoint) capability.

The new `whole-model.v1` contains actual declarative configuration under `definition/schema`
and prefixed unchanged `model.v1` entries. Public ModelCheckpoint whole-model envelope operations
use existing ArtifactEnvelope; private value [codec](../../../glossary.md#term-codec) parsing is not exposed.
Short save accepts [Module](../../../glossary.md#term-module) recognized as Sequential/DeclarativeModel; short load returns DeclarativeModel.
Advanced limits/[fingerprint](../../../glossary.md#term-fingerprint) [signatures](../../../glossary.md#term-signature), field order/IDs and failure types are frozen in the owning
contract. Defaults: 256 MiB file/aggregate, 4 MiB [manifest](../../../glossary.md#term-manifest), 16384 entries, 64-byte keys, 64 MiB
[payload](../../../glossary.md#artifact-envelope-and-atomic-publication); fixed 1 MiB definition and [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 4096/depth64 under advanced limits too, not peak [heap](../../../glossary.md#term-heap).
Dry-size/check residuals → bounded definition encode/[hash](../../../glossary.md#term-hash) → public values [capture](../../../glossary.md#term-capture-meanings)'s exact projected
[preflight](../../../glossary.md#term-preflight) before [Engine](../../../glossary.md#term-engine)/bulk values → outer check/coherence → prefixed copies. For n inner entries,
manifest delta 67+6n, file delta D+67+6n, aggregate delta D, entry delta 1, key ceiling minus 6.
Capture and NN description may each read state under [quiescence](../../../glossary.md#term-quiescence); no combined-[snapshot](../../../glossary.md#term-snapshot) guarantee.
[Canonical](../../../glossary.md#term-canonical-form) definition [SHA-256](../../../glossary.md#term-sha-256) is computed inside [Checkpoint](../../../glossary.md#term-checkpoint), not user plumbing or [authentication](../../../glossary.md#term-authentication).
High-level load rejects old values-only files with an existing [target](../../../glossary.md#term-target)-load hint, never guesses [topology](../../../glossary.md#term-topology).
The held proposed separate `training-model.v1` would add supported recipe and raw TrainingState. Unpublished 0037
`training-checkpoint.v1` is replaced, not declared same-file-v1 compatible.

Conceptual future flow:

```text
ModelCheckpoint.load(path) → independent heap-backed eval model → engine.compute(model.forward(input))
TrainingCheckpoint.loadModel(path) → independent heap-backed eval model
TrainingCheckpoint.resume(path, engine) → exclusively owned TrainingRun → trainBatch(input, target)
```

[Inference](../../../glossary.md#term-inference) has no hidden [autograd](../../../glossary.md#autograd) or native [Arena](../../../glossary.md#term-arena) [lifetime](../../../glossary.md#term-lifetime); [parameters](../../../glossary.md#term-parameter) remain [gradient](../../../glossary.md#term-gradient)-eligible.
Engine.compute returns [detached](../../../glossary.md#detached-model-results) [HostTensorValue](../../../glossary.md#host-tensor-value--hosttensorvalue). NN inference constraints can allow compatible
leading [dimensions](../../../glossary.md#dimension); Training recipe separately fixes static dense input/target signatures.
Short loads require absent input [digest](../../../glossary.md#term-digest); stored fingerprints require explicit [owner](../../../glossary.md#term-owner)-verified expected
[identity](../../../glossary.md#term-identity-meanings) overload. Text is Draft, so no automatic [tokenizer](../../../glossary.md#frozen-whitespace-tokenizer) load is promised.

The earlier ADR0038 resume design remains historical Draft input, not the selected [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[binding](../../../glossary.md#term-binding)
contract. Target [admission](../../../glossary.md#term-admission) reuses [TrainingParameterStorage](../../../glossary.md#native-parameter-copy--trainingparameterstorage) on a private/exclusively transferred
candidate before [objective](../../../glossary.md#term-objective)/session construction, preserving exact bytes and session-before-Arena
cleanup. Do not mutate [borrowed](../../../glossary.md#ownership-borrowing-and-lease) shared [weights](../../../glossary.md#term-weight) or bind batches by changing shared placeholder
storage. Engine's future stable-slot binding owns per-[run](../../../glossary.md#term-run) snapshots; Training retains live
parameter stability/[precommit](../../../glossary.md#term-precommit) [validation](../../../glossary.md#term-validation) and failure-[atomic](../../../glossary.md#resource-transaction-and-rollback) [optimizer state](../../../glossary.md#term-optimizer-state). The program requires review of the selected
[ownership](../../../glossary.md#term-ownership-meanings)/[API](../../../glossary.md#term-application-programming-interface-api) packet before implementation and makes no automatic exact-continuation claim.

## Integration [ownership](../../../glossary.md#term-ownership-meanings) and [validation](../../../glossary.md#term-validation)

[Model](../../../glossary.md#term-model)-execution coordinator ([NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training/[Checkpoint](../../../glossary.md#term-checkpoint) coordinator with [Engine](../../../glossary.md#term-engine)/[backend](../../../glossary.md#backend) [owners](../../../glossary.md#term-owner)) owns
shared plan/roadmap/explanatory integration.
Workers own focused code/tests/[Javadoc](../../../glossary.md#term-javadoc) and [module](../../../glossary.md#term-module) validation once; independent targeted Class C
review/docs reuse successful executable evidence. CP0005B's coordinator separately runs architecture
and named WholeModelCheckpointEngineIntegrationTest after review (exact commands in its brief),
using existing NN/[CPU](../../../glossary.md#term-cpu)/Checkpoint test edges without build/whitelist edits. Coordinator adds join checks
and runs one combined source-matched `test javadoc --max-workers=2 --console=plain` checkpoint,
with bounded actual [device](../../../glossary.md#term-device) [smoke](../../../glossary.md#term-smoke-test)/skips. No performance campaign.
[Planning](../../../glossary.md#term-planning) now checks links/anchors, brief limits and diff only, with no Java reruns.

## Open decisions and risks

- The five product choices and final P2 review are approved with no remaining findings.
  Engine0021 and its bounded prepared-owner authority cutover are Complete. Checkpoint0008 now
  completed generic modules/checkpoint and the bounded cold Compiler edge in coordinated authority,
  with final independent approval/coordinator gate passed; this is no longer NN-transitive packaging.
  Compiler0008's bounded map exception and Engine fresh-input authority remain unchanged.
  Load opens no Engine/backend resources; current contracts win until cutover.<span data-glossary-navigation="true"> ([owner glossary](../../../glossary.md#term-owner); [modules glossary](../../../glossary.md#term-module); [checkpoint glossary](../../../glossary.md#term-checkpoint); [Compiler glossary](../../../glossary.md#term-compiler); [NN glossary](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Engine glossary](../../../glossary.md#term-engine); [backend glossary](../../../glossary.md#backend))</span>
- CP0005B is Complete with independent implementation/docs approval and recorded validation;
  successors require their own contract/frontier verification before Ready.<span data-glossary-navigation="true"> ([validation glossary](../../../glossary.md#term-validation))</span>
- NN0027 owns semantic recurrent input/output/default-state policy only. CP0005C owns explicit
  versioned wire/save/load adoption; no detailed briefs before those future decisions. Unknown
  recurrent wire types reject now; no registry or recurrent files in CP0005B v1.<span data-glossary-navigation="true"> ([registry glossary](../../../glossary.md#term-registry))</span>
- Durable Training-recipe loss/signatures remain a held proposal; P2's configured scalar-objective
  and fixed-signature API is selected separately from variable NN inference constraints.<span data-glossary-navigation="true"> ([loss glossary](../../../glossary.md#loss); [signatures glossary](../../../glossary.md#term-signature); [scalar glossary](../../../glossary.md#term-scalar); [objective glossary](../../../glossary.md#term-objective); [API glossary](../../../glossary.md#term-application-programming-interface-api); [inference glossary](../../../glossary.md#term-inference))</span>
- Freeze Training-[owned model](../../../glossary.md#model-module-buffer-and-scan-distinctions)-transfer/restore/[capture](../../../glossary.md#term-capture-meanings) bridge, stable [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [admission](../../../glossary.md#term-admission) and cleanup.
- CP definition/defaults/typed identity are accepted; recipe/Training/adapter schema and
  outer defaults remain Proposed. Encoding limits are not peak-heap/large-model evidence.<span data-glossary-navigation="true"> ([identity glossary](../../../glossary.md#term-identity-meanings); [schema glossary](../../../glossary.md#term-schema); [heap glossary](../../../glossary.md#term-heap); [model glossary](../../../glossary.md#term-model))</span>
- Future recurrent/Training/adapter decisions require their own review and coordinated cutovers before [Ready](../../../glossary.md#term-ready).
- Preserve atomic live publication (force/revalidate/one atomic move), not portable power-loss
  durability or authentication. Version retention and broader external-state recovery remain later
  decisions; no caller cursor/RNG/scheduler/scaler data is implicitly recovered.<span data-glossary-navigation="true"> ([atomic live publication glossary](../../../glossary.md#artifact-envelope-and-atomic-publication); [atomic move glossary](../../../glossary.md#term-atomic-move); [durability glossary](../../../glossary.md#term-durability); [authentication glossary](../../../glossary.md#term-authentication); [RNG glossary](../../../glossary.md#graph-rng-state--graphrngstate))</span>

## Related plans and published APIs

- [NN](../nn/master-plan.md)<span data-glossary-navigation="true"> ([NN glossary](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>, [Training](../training/master-plan.md), [roadmap](../../roadmap.md)
- [Published model checkpoint API](../../../api/model-checkpoint-api.md)<span data-glossary-navigation="true"> ([model glossary](../../../glossary.md#term-model); [checkpoint glossary](../../../glossary.md#term-checkpoint); [API glossary](../../../glossary.md#term-application-programming-interface-api))</span>
- [Artifact API](../../../api/checkpoint-artifact-api.md)<span data-glossary-navigation="true"> ([Artifact glossary](../../../glossary.md#term-artifact))</span>
- [Planning guide](../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../glossary.md#term-planning))</span>
