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
The user approved CP0005B direction, then recurrent reconstruction before simple Training resume.
Only CP0005B's coordinated contract is accepted; final independent approval and coordinator
verification authorized implementation; CP0005B is Complete after final reviews/gates (2026-10-15).
Recurrent/Training/adapter decisions remain Proposed.

## Contracts and common base

- Root: authority/scope index, module ownership, dependency rules and testing requirements.
- Owning [extensions contract](../../../architecture/contracts/extensions-training.md):
  extensions/nn, extensions/training, extensions/checkpoint and Optimizer/training lifecycle.
- Execution base: source `f892cd3fbde4f8b7722b7eb5b14f4b0a5f3b2669` plus accepted contract packet;
  coordinator records final contract commit in launcher after integration.
- [Whole-model contract](../../../architecture/contracts/extensions-training.md#whole-model-definition-and-composition)
  accepts CP0005B API/wire/defaults/failures/preflight; no dependency changes.
- Current published values-only `model.v1` and ArtifactFile v1 remain unchanged.
  [ADR 0036](../../../design/decisions/0036-model-envelope-composition.md) remains implemented.

## Scope and non-goals

NN owns declarative configuration/forward reconstruction and depends only on Model.
Checkpoint owns bounded byte codecs, persistent versions, file publication and whole-model API.
Training owns recipe meaning, typed reusable batch orchestration and owned-run lifecycle.
The optional adapter composes these public owners with exactly Training + Checkpoint api edges;
NN/Model/Engine are transitive. Model-only Checkpoint keeps its exact NN/Engine/Model edges.
No reverse dependencies, arbitrary Java serialization/reflection/registry, automatic backend
discovery, saved compiled/device state, general participants/transactions or ModelExecutor.

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

Selected capability program (arrows mean prerequisites; only the Ready frontier may launch):

```text
NN0026 Complete → CP0005B → NN0027 semantics → CP0005C wire → Training0006 → optional CP0006
completed CP0005A ────────────────────────────────────────────────────────────────────┘
```

Authorized Checkpoint frontier: None. CP0005B is Complete after final independent code/docs review,
57 module tests, final Javadoc and coordinator architecture/3-test integration gates. Contract approval and
2026-10-14 coordinator base/dependency/contract/frontier/conflict verification are recorded in its
brief. NN0027/CP0005C remain Draft rows only (no speculative briefs).
No task waits on optimized scan/backend BPTT/Data/Text or broader Training checkpoint.
Each implementation/review uses separate clean contexts; parallel group None for now because
shared contract cutovers need serialization. Integration order follows the serial chain above.
The coordinator integrates/commits the accepted packet before launching a new clean implementation.
CP's root/scoped contract/explanations/ADR are accepted; future recurrent policies,
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
| 0005C | Versioned recurrent definition and whole-model persistence adoption | Draft | CP0005B; NN0027 | Checkpoint owns bytes/versions/save/load; exact version/policy future, no detailed speculative brief. |
| [0006](tasks/0006-training-checkpoint-adapter.md) | Optional whole-model TrainingCheckpoint | Draft | NN0026/0027; CP0005B/0005C; Training0006; completed 0005A/Training0004 | Proposed high-level save/loadModel/resume; separate contract gate. |
| 0007 | Broader recovery/text checkpoint | Draft | 0005B/0006; Text 0003 only for text integration | Later broader corruption/recovery/text validation; not an extra implementation prerequisite or duplicate gate. |

Exact conflict/base/order/validation metadata live in briefs. No Checkpoint task is currently Ready.

CP0005C ordering metadata: Depends on CP0005B and NN0027; Conflicts with shared Checkpoint wire/API
and NN semantic contract cutovers/Training0006/adapter0006; Parallel group None; Common base revision
`f892cd3fbde4f8b7722b7eb5b14f4b0a5f3b2669` (reverify after predecessors); Integration order after
NN0027 and before Training0006; Integration validation future affected Checkpoint round trips plus
NN coherence/architecture at its join; Shared-document integration owner NN/Training/Checkpoint
coordinator. NN owns semantic reconstruction only; this split follows ownership, not mechanical
API/test staging. Initial-state/output-selection, format version and Training gradient admission
remain future decisions. CP0005B v1 supports no recurrent files.

## Selected whole-model and proposed Training flow

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
The separate `training-model.v1` adds supported recipe and raw TrainingState. Unpublished 0037
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

Resume validates complete definition/state/recipe/raw parameter coherence, NN-loads a private
descriptor-bound tree, then Training copies native parameters BEFORE constructing objective/opening
session. Ordinary TrainingSession.state/restore validate session state. TrainingRun owns its private
model, shared parameter Arena, placeholders and session; Engine/batch storage remain borrowed.
Close session before arenas, attempt all cleanup and retain failures. Native model access is
run-bound; detached inference models alone may outlive the run.
TrainingRun's narrow exclusive capture borrow spans session.state + downstream model capture.
Stable placeholders bind fresh batch storage with exact descriptor/layout checks and clear it in
finally; compiled metadata/Tensor identity resolve roles once. One prepared session, no per-batch
compile, storage copies by default or hot-path role-name lookup.

## Integration ownership and validation

NN/Training/Checkpoint coordinator owns shared plan/roadmap/explanatory integration.
Workers own focused code/tests/Javadoc and module validation once; independent targeted Class C
review/docs reuse successful executable evidence. CP0005B's coordinator separately runs architecture
and named WholeModelCheckpointEngineIntegrationTest after review (exact commands in its brief),
using existing NN/CPU/Checkpoint test edges without build/whitelist edits. Coordinator adds join checks
and runs one combined source-matched `test javadoc --max-workers=2 --console=plain` checkpoint,
with bounded actual device smoke/skips. No performance campaign.
Planning now checks links/anchors, brief limits and diff only, with no Java reruns.

## Open decisions and risks

- CP0005B is Complete with independent implementation/docs approval and recorded validation;
  successors require their own contract/frontier verification before Ready.
- NN0027 owns semantic recurrent input/output/default-state policy only. CP0005C owns explicit
  versioned wire/save/load adoption; no detailed briefs before those future decisions. Unknown
  recurrent wire types reject now; no registry or recurrent files in CP0005B v1.
- Training loss/signatures remain Proposed, separately from variable NN inference constraints.
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
