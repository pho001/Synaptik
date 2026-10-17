# Checkpoint Extension Master Plan

## Goal and authority

Accepted whole-model product: store a supported declarative model and load it automatically for
inference. Training-owned resume from model + recipe + TrainingState remains Proposed.
Normal usage needs no restore callback, manual model construction, config digest, ArtifactEnvelope
or Arena. Existing explicit APIs remain the advanced foundation.
This is non-authoritative coordination; [root architecture](../../../../ARCHITECTURE.md) wins.
[ADR 0038](../../../design/decisions/0038-declarative-model-and-training-resume.md)
replaces the withdrawn Proposed snapshot-only direction of
[ADR 0037](../../../design/decisions/0037-stored-training-checkpoint-adapter.md).
The user approved CP0005B direction; the earlier recurrent-first successor ordering is now on hold.
Only CP0005B's coordinated contract is accepted; final independent approval and coordinator
verification authorized implementation; CP0005B is Complete after final reviews/gates (2026-10-15).
Recurrent/Training/adapter decisions remain Proposed.

Current proposal: [ADR 0039 model execution + persistence](../../../design/decisions/0039-portable-graph-model-persistence.md)
and [Complete CP0005D planning/review](tasks/0005d-portable-graph-persistence-decision.md) record the user-approved
direction: the same proposed model.compute/train/save/load for manual and file-restored tensors.
The [cross-area program](../../model-execution-refactor.md) owns phases, target paths/edges and the
independently approved P2 planning packet. Generic Checkpoint targets modules/checkpoint; NN-specific glue moves
to NN after removing the reverse edge. Current contracts remain in force until coordinated cutover.
Hold NN0027/CP0005C/Training/adapter expansion; completed closed-NN v1 stays supported.

## Contracts and common base

- Root: authority/scope index, module ownership, dependency rules and testing requirements.
- Owning [extensions contract](../../../architecture/contracts/extensions-training.md):
  extensions/nn, extensions/training, extensions/checkpoint and Optimizer/training lifecycle.
- Historical CP0005B execution base/accepted packet remains in its completed brief.
- Current planning base: `9e7e8aa15e0833a1bc91263a7add99a80d88cd92`; future cutovers
  require independent review and coordinator verification, not prior-v1 approval.
- [Whole-model contract](../../../architecture/contracts/extensions-training.md#whole-model-definition-and-composition)
  accepts CP0005B API/wire/defaults/failures/preflight; no dependency changes.
- Current published values-only `model.v1` and ArtifactFile v1 remain unchanged.
  [ADR 0036](../../../design/decisions/0036-model-envelope-composition.md) remains implemented.

## Scope and non-goals

CURRENT: NN owns declarative configuration/reconstruction with a Model-only edge; Checkpoint's
published whole-model API has exact NN/Engine/Model edges. TARGET: generic Checkpoint in
modules/checkpoint owns opaque files and ordinary tensor graph/value persistence with Model
API and a proposed cold Compiler implementation edge; no NN/Training dependency.
NN consumes generic Checkpoint only after its old reverse edge is removed. NN-specific v1 glue
moves to nn.checkpoint with explicit import/dependency migration; no cycle or automatic deletion.

Training recipe/owned-run and optional adapter below are held ADR0038 proposals, not current APIs
or target authority. No arbitrary Java serialization/reflection/registry, backend discovery,
compiled/device-state persistence, general transactions or mandatory ModelExecutor.

Initial support is implemented NN0026 nested Sequential, Linear/LayerNorm/Embedding/Conv1d/2d/3d and
closed RELU/TANH/SIGMOID steps. CP0005B retains that exact matrix/attributes/guards/aliases.
Custom Model.define callbacks/record I/O stay explicit. Declarative branches/contextual layers are
named successors, not a speculative detailed cascade. Fixed RecurrentScan BPTT/Conv3d gradients and training-mode
BatchNorm buffer alignment remain fail-closed; no changing per-step dropout or exact cursor/RNG claim.
Ordinary static recurrent unrolling depends on normal constituent-operation adjoints; no blanket ban.
The closed Sequence tree explicitly determines execution order; arbitrary callbacks are not captured.
Inference aliases must reconstruct identity or reject; training rejects repeated parameter paths/
overlapping stores, not every repeated forward use of one uniquely owned layer.

## Package map

| Package | Delivered or proposed owner |
|---|---|
| `checkpoint.artifact` | Existing ArtifactEnvelope, ArtifactFile, ArtifactLimits and publication. |
| `checkpoint.model` | ModelCheckpoint whole-model overloads and InputArtifactFingerprint implemented (0005B); WholeModelDefinitionCodec/WholeModelComposition remain package-private. |
| `nn.module` | Implemented ModelSpecification/DeclarativeModel (NN 0026); existing callback ModelDefinition unchanged. |
| `training` | Existing snapshots/session; proposed TrainingRecipe/TrainingRun and narrow capture borrow (Training 0006). |
| `training.checkpoint` | Proposed TrainingCheckpoint/Snapshot and private TrainingState/recipe composition codecs in optional project (0006). |

Package names are relative to `io.github.pho001.synaptik`. CP0005B names/API are accepted;
Training/adapter names remain Proposed.

## Dependency DAG and authorized frontier

Completed chain: Checkpoint 0001 → 0002 → 0003 → 0004; CPU 0024 → 0004.
Training 0004 + Checkpoint 0004/CPU 0024 → 0004A → Training-owned 0005 → 0005A.

Earlier proposed capability program (held after its completed prefix):

```text
NN0026 Complete → CP0005B Complete | HOLD: NN0027 → CP0005C → Training0006 → optional CP0006
completed CP0005A ────────────────────────────────────────────────────────────────────┘
```

The [program DAG](../../model-execution-refactor.md#phases-and-owner-dag) replaces the old
cold-reconnection-first proposal. CP0005D is Complete for bounded planning/review after independent
Astra/xhigh review, the parent-project mapping correction and coordinator acceptance. No Checkpoint
implementation is Ready. The backend/native-only
[Engine0020 cutover](../../modules/engine/tasks/0020-backend-native-layout-relocation.md) is Complete
after integration/review/combined validation. The user's subsequent five product choices are now
accepted. Engine0021 delivered thin-session removal, coordinated authority and mechanical consumers;
independent Class C approval and the native-enabled test/Javadoc gate passed. Its brief owns evidence.
Checkpoint and all downstream tasks remain Draft; their separate contracts are not enacted here.

Checkpoint's later owner phase separates the opaque artifact infrastructure and generic graph/value
codec from NN-specific glue, retains old v1 readers, and obtains coordinated cold Compiler access
and reconstruction-map authority. Load reconstructs ordinary Tensor expressions/weights without
Engine or future batch before compile. Normal repeated model.compute/train uses stable prepared
signatures through existing Engine PreparedExecution/Engine.runValues and fresh per-run storage
snapshots, not outputs(actualInputs) reconstruction or a replacement session facade.
Engine0021 removed InferenceSession/Engine.session; old file readers are unaffected.
Selected Checkpoint0008 depends on Compiler0008 and Training0007; it moves the six artifact files
to modules/checkpoint and nine NN glue classes to extensions/nn under nn.checkpoint. Retain
:extensions:checkpoint via projectDir; generic Checkpoint has api(Model), implementation(Compiler),
no NN/Training/Engine edge. NN owns old Engine-materializing ModelCheckpoint behavior and readers.
The new GraphCheckpoint Source/Leaf/Role boundary carries ordinary Tensor sources, not a runner.
Host-only values allow cold copy/save/load without Engine; owned loaded values are adopted once.
First graph codecs cover ADD/MUL/NEG/MATMUL/PERMUTE[1,0], not an all-family codec campaign.
Exact APIs, ownership, migration and source guards live in the program/ADR, not another framework.

Authorized Checkpoint frontier: None. CP0005B is Complete after final independent code/docs review,
57 module tests, final Javadoc and coordinator architecture/3-test integration gates. Contract approval and
2026-10-14 coordinator base/dependency/contract/frontier/conflict verification are recorded in its
brief. NN0027/CP0005C remain Draft rows only (no speculative briefs).
No task waits on optimized scan/backend BPTT/Data/Text or broader Training checkpoint.
Each implementation/review uses separate clean contexts; parallel group None for now because
shared contract cutovers need serialization. Integration order for new graph work follows the program DAG, not the held recurrent chain.
Any later launch requires a separately accepted contract packet and verified Ready brief.
CP0005B's root/scoped contract/explanations/ADR are accepted; future recurrent policies,
Training bridges/recipe and adapter wire need their own decisions. Build inclusion happens at implementation,
not in this plan.

## Task list

Completed rows preserve published API history; unfinished rows describe future capability only.

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| [0001](tasks/0001-checkpoint-module-boundary.md) | Model Checkpoint module boundary | Complete | None; architecture decision is the deliverable | Empty project, contract, ADR, explanations, and dependency test reviewed; focused and repository gates passed. No persistence API or optional adapter. |
| [0002](tasks/0002-bounded-artifact-envelope.md) | Bounded artifact envelope | Complete | 0001 Complete; corrected v1 policy in brief | Opaque file API, bounds, SHA-256, file flush and atomic live publication implemented; 15 checkpoint tests, Javadoc, architecture test, and independent Class C review passed. |
| [0003](tasks/0003-model-schema-and-materialization-handoff.md) | Model schema and Engine materialization handoff | Complete | 0002 Complete; existing NN export and Engine compute | Internal fixed-ID schema, detached candidate, and bounded public Engine handoff implemented; 22 module tests, Javadoc, and independent Class C review passed. No public save/load. |
| [0004](tasks/0004-public-model-checkpoint-save-inspect-load.md) | Model checkpoint save, inspect, and strict load | Complete | 0003; [CPU 0024](../../backends/cpu/tasks/0024-default-segment-route-heap-carrier-binding.md) for direct restored-heap execution | Public save/inspect/load and direct restored-heap Engine regression passed, with independent review, documentation, and combined validation. |
| [0004A](tasks/0004a-direct-leaf-model-save.md) | Direct leaf-model save | Complete | Checkpoint 0004; CPU 0024; Training 0004, all Complete | Exact native/heap leaf capture, direct trained save/load and loaded re-save passed; workaround removed, independent review and configured coordinator gate passed. |
| [0005](tasks/0005-training-state-reconstruction.md) | Training-owned detached state reconstruction prerequisite | Complete | Training 0004; Checkpoint 0004A, both Complete | Three defensive constructors exposed; external-package tests, final Javadoc/example validation, and independent API review passed. No codec. |
| [0005A](tasks/0005a-model-envelope-composition.md) | Model-envelope composition | Complete | 0004A and 0005 Complete | Four existing-ArtifactEnvelope operations implemented; independent Class C API/documentation review and configured coordinator gates passed; file semantics and module edges unchanged. |
| [0005B](tasks/0005b-whole-model-save-load.md) | Whole-model save and automatic inference load | Complete | NN0026; CP0002/0004/0004A/0005A Complete; accepted CP contract | Ten overloads/typed identity and closed codec/composition; independent reviews, 57 module tests, final Javadoc and coordinator gates passed. |
| 0005C | Versioned recurrent definition and whole-model persistence adoption | Draft | CP0005B; NN0027; separately selected recurrent scope | Optional held proposal, not a new graph-path prerequisite; no detailed speculative brief. |
| [0005D](tasks/0005d-portable-graph-persistence-decision.md) | Model execution/persistence program decision | Complete | NN0026/CP0005B Complete; approved direction/layout | Independent review, sole P2 correction and bounded coordinator acceptance; Engine0020 Ready, semantic defaults still Proposed. |
| [0006](tasks/0006-training-checkpoint-adapter.md) | Optional whole-model TrainingCheckpoint | Draft | NN0026/0027; CP0005B/0005C; Training0006; completed 0005A/Training0004 | Proposed high-level save/loadModel/resume; separate contract gate. |
| 0007 | Broader recovery/text checkpoint | Draft | 0005B/0006; Text 0003 only for text integration | Later broader corruption/recovery/text validation; not an extra implementation prerequisite or duplicate gate. |
| 0008 | Generic graph/value persistence and NN glue cutover | Draft | Compiler0008; Training0007; reviewed P2 packet | Cold bounded graph-model.v1 and six/nine-file owner split, retain old-v1 semantics/readers; no Ready brief. |

Exact conflict/base/order/validation metadata live in briefs. No Checkpoint task is currently Ready.

CP0005C ordering metadata (held, revalidate if recurrent scope is separately selected): Depends on CP0005B and NN0027; Conflicts with shared Checkpoint wire/API
and NN semantic contract cutovers/Training0006/adapter0006; Parallel group None; Common base revision
`f892cd3fbde4f8b7722b7eb5b14f4b0a5f3b2669` (reverify after predecessors); Integration order after
NN0027 and before Training0006; Integration validation future affected Checkpoint round trips plus
NN coherence/architecture at its join; Shared-document integration owner NN/Training/Checkpoint
coordinator. NN owns semantic reconstruction only; this split follows ownership, not mechanical
API/test staging. Initial-state/output-selection, format version and Training gradient admission
remain future decisions. CP0005B v1 supports no recurrent files.

## Selected whole-model and proposed Training flow

The v1 description below remains current; the future Training flow is the held ADR0038 proposal,
not a commitment to extend layer-family persistence before a separately selected recurrent/Training-resume capability.

The new `whole-model.v1` contains actual declarative configuration under `definition/schema`
and prefixed unchanged `model.v1` entries. Public ModelCheckpoint whole-model envelope operations
use existing ArtifactEnvelope; private value codec parsing is not exposed.
Short save accepts Module recognized as Sequential/DeclarativeModel; short load returns DeclarativeModel.
Advanced limits/fingerprint signatures, field order/IDs and failure types are frozen in the owning
contract. Defaults: 256 MiB file/aggregate, 4 MiB manifest, 16384 entries, 64-byte keys, 64 MiB
payload; fixed 1 MiB definition and NN 4096/depth64 under advanced limits too, not peak heap.
Dry-size/check residuals → bounded definition encode/hash → public values capture's exact projected
preflight before Engine/bulk values → outer check/coherence → prefixed copies. For n inner entries,
manifest delta 67+6n, file delta D+67+6n, aggregate delta D, entry delta 1, key ceiling minus 6.
Capture and NN description may each read state under quiescence; no combined-snapshot guarantee.
Canonical definition SHA-256 is computed inside Checkpoint, not user plumbing or authentication.
High-level load rejects old values-only files with an existing target-load hint, never guesses topology.
The held proposed separate `training-model.v1` would add supported recipe and raw TrainingState. Unpublished 0037
`training-checkpoint.v1` is replaced, not declared same-file-v1 compatible.

Conceptual future flow:

```text
ModelCheckpoint.load(path) → independent heap-backed eval model → engine.compute(model.forward(input))
TrainingCheckpoint.loadModel(path) → independent heap-backed eval model
TrainingCheckpoint.resume(path, engine) → exclusively owned TrainingRun → trainBatch(input, target)
```

Inference has no hidden autograd or native Arena lifetime; parameters remain gradient-eligible.
Engine.compute returns detached HostTensorValue. NN inference constraints can allow compatible
leading dimensions; Training recipe separately fixes static dense input/target signatures.
Short loads require absent input digest; stored fingerprints require explicit owner-verified expected
identity overload. Text is Draft, so no automatic tokenizer load is promised.

The earlier ADR0038 resume design remains historical Draft input, not the selected batch-binding
contract. Target admission reuses TrainingParameterStorage on a private/exclusively transferred
candidate before objective/session construction, preserving exact bytes and session-before-Arena
cleanup. Do not mutate borrowed shared weights or bind batches by changing shared placeholder
storage. Engine's future stable-slot binding owns per-run snapshots; Training retains live
parameter stability/precommit validation and failure-atomic optimizer state. The program requires review of the selected
ownership/API packet before implementation and makes no automatic exact-continuation claim.

## Integration ownership and validation

Model-execution coordinator (NN/Training/Checkpoint coordinator with Engine/backend owners) owns
shared plan/roadmap/explanatory integration.
Workers own focused code/tests/Javadoc and module validation once; independent targeted Class C
review/docs reuse successful executable evidence. CP0005B's coordinator separately runs architecture
and named WholeModelCheckpointEngineIntegrationTest after review (exact commands in its brief),
using existing NN/CPU/Checkpoint test edges without build/whitelist edits. Coordinator adds join checks
and runs one combined source-matched `test javadoc --max-workers=2 --console=plain` checkpoint,
with bounded actual device smoke/skips. No performance campaign.
Planning now checks links/anchors, brief limits and diff only, with no Java reruns.

## Open decisions and risks

- The five product choices and final P2 review are approved with no remaining findings.
  Engine0021 and its bounded prepared-owner authority cutover are Complete. Separate Checkpoint
  authority updates remain future implementation work, not current normative permission.
  Generic modules/checkpoint is the approved target direction, not NN-transitive packaging.
  Cold Compiler access/map exception need coordinated contracts; Engine fresh-input authority is current.
  Load opens no Engine/backend resources; current contracts win until cutover.
- CP0005B is Complete with independent implementation/docs approval and recorded validation;
  successors require their own contract/frontier verification before Ready.
- NN0027 owns semantic recurrent input/output/default-state policy only. CP0005C owns explicit
  versioned wire/save/load adoption; no detailed briefs before those future decisions. Unknown
  recurrent wire types reject now; no registry or recurrent files in CP0005B v1.
- Durable Training-recipe loss/signatures remain a held proposal; P2's configured scalar-objective
  and fixed-signature API is selected separately from variable NN inference constraints.
- Freeze Training-owned model-transfer/restore/capture bridge, stable batch admission and cleanup.
- CP definition/defaults/typed identity are accepted; recipe/Training/adapter schema and
  outer defaults remain Proposed. Encoding limits are not peak-heap/large-model evidence.
- Future recurrent/Training/adapter decisions require their own review and coordinated cutovers before Ready.
- Preserve atomic live publication (force/revalidate/one atomic move), not portable power-loss
  durability or authentication. Version retention and broader external-state recovery remain later
  decisions; no caller cursor/RNG/scheduler/scaler data is implicitly recovered.

## Related plans and published APIs

- [NN](../nn/master-plan.md), [Training](../training/master-plan.md), [roadmap](../../roadmap.md)
- [Published model checkpoint API](../../../api/model-checkpoint-api.md)
- [Artifact API](../../../api/checkpoint-artifact-api.md)
- [Planning guide](../../planning-guide.md)
