# Checkpoint Extension Master Plan

## Goal and authority

Primary proposed product: store a supported declarative model and load it automatically for
inference, or resume a Training-owned run from model + recipe + TrainingState.
Normal usage needs no restore callback, manual model construction, config digest, ArtifactEnvelope
or Arena. Existing explicit APIs remain the advanced foundation.
This is non-authoritative coordination; [root architecture](../../../../ARCHITECTURE.md) wins.
[Proposed ADR 0038](../../../design/decisions/0038-declarative-model-and-training-resume.md)
replaces the withdrawn Proposed snapshot-only direction of
[ADR 0037](../../../design/decisions/0037-stored-training-checkpoint-adapter.md).
The user approved revising direction, not implementing code or accepting a new contract.

## Contracts and common base

- Root: authority/scope index, module ownership, dependency rules and testing requirements.
- Owning [extensions contract](../../../architecture/contracts/extensions-training.md):
  extensions/nn, extensions/training, extensions/checkpoint and Optimizer/training lifecycle.
- Planning base: `52733fcfd32d278b651302e308487e0b6c6afa81`; verify source/contracts again before Ready.
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

Initial support is proposed nested Sequential, Linear/LayerNorm/Embedding/Conv1d/2d/3d and closed
RELU/TANH/SIGMOID steps. Exact attrs/wiring/aliases/input constraints need decision before Ready.
Custom Model.define callbacks/record I/O stay explicit. Declarative branches/contextual layers are
named successors, not a speculative detailed cascade. BPTT/Conv3d gradients and training-mode
BatchNorm buffer alignment remain fail-closed; no changing per-step dropout or exact cursor/RNG claim.
Child ownership is a tree; execution wiring is not inferred from that tree.
Inference aliases must reconstruct identity or reject; training rejects repeated parameter paths/
overlapping stores, not every repeated forward use of one uniquely owned layer.

## Package map

| Package | Delivered or proposed owner |
|---|---|
| `checkpoint.artifact` | Existing ArtifactEnvelope, ArtifactFile, ArtifactLimits and publication. |
| `checkpoint.model` | Existing ModelCheckpoint; proposed private definition codec/composition and whole-model overloads (0005B). |
| `nn.module` | Proposed ModelSpecification/DeclarativeModel (NN 0026); existing callback ModelDefinition unchanged. |
| `training` | Existing snapshots/session; proposed TrainingRecipe/TrainingRun and narrow capture borrow (Training 0006). |
| `training.checkpoint` | Proposed TrainingCheckpoint/Snapshot and private TrainingState/recipe composition codecs in optional project (0006). |

Package names are relative to `io.github.pho001.synaptik`. New names remain proposed until API freeze.

## Dependency DAG and authorized frontier

Completed chain: Checkpoint 0001 → 0002 → 0003 → 0004; CPU 0024 → 0004.
Training 0004 + Checkpoint 0004/CPU 0024 → 0004A → Training-owned 0005 → 0005A.

Proposed capability program (arrows mean prerequisites, not launch permission):

```text
NN 0026 ──> Checkpoint 0005B ─┐
        └─> Training 0006 ───┼─> Checkpoint 0006
completed Checkpoint 0005A ──┘
```

Authorized frontier: None. All new/revised briefs are Draft; the two immediate owner capabilities
join at the adapter. No task waits on unrelated NN scan/Data/Text or broader Training checkpoint.
Each implementation/review uses separate clean contexts; parallel group None for now because
shared contract cutovers need serialization. Integration order: NN → owner capabilities → adapter.
The coordinator verifies dependencies/frontier/common base and records the check before Ready.
Before Ready, approve matrix/owner bridges/wire, cut over root + owning scoped contract +
explanations + ADR + architecture tests as applicable. Build inclusion happens at implementation,
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
| [0005B](tasks/0005b-whole-model-save-load.md) | Whole-model save and automatic inference load | Draft | NN 0026; completed 0005A | Canonical definition + unchanged values component; usable independent NN model, no callback/digest plumbing. |
| [0006](tasks/0006-training-checkpoint-adapter.md) | Optional whole-model TrainingCheckpoint | Draft | NN 0026; Checkpoint 0005B; Training 0006; completed 0005A/Training 0004 | High-level save/loadModel/resume plus advanced explicit foundation. |
| 0007 | Broader recovery/text checkpoint | Draft | 0005B/0006; Text 0003 only for text integration | Later broader corruption/recovery/text validation; not an extra implementation prerequisite or duplicate gate. |

Exact conflict/base/order/validation metadata live in new briefs. No new Ready task is selected.

## Proposed artifact and user flow

The new `whole-model.v1` contains actual declarative configuration under `definition/schema`
and prefixed unchanged `model.v1` entries. Public ModelCheckpoint whole-model envelope operations
use existing ArtifactEnvelope; private value codec parsing is not exposed.
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
review/docs reuse successful executable evidence. Coordinator adds architecture/end-to-end checks
and runs one combined source-matched `test javadoc --max-workers=2 --console=plain` checkpoint,
with bounded actual device smoke/skips. No performance campaign.
Planning now checks links/anchors, brief limits and diff only, with no Java reruns.

## Open decisions and risks

- Approve exact layer/activation/loss matrix, semantic input constraints versus static Training
  signatures, inference alias policy and descriptor-only skeleton proof.
- Freeze Training-owned model-transfer/restore/capture bridge, stable batch admission and cleanup.
- Freeze canonical definition/recipe/Training schema records, default ArtifactLimits and typed
  expected-input-identity overloads. Encoding limits are not peak-heap/large-model evidence.
- Complete independent proposal review and coordinated authoritative cutovers before Ready.
- Preserve atomic live publication (force/revalidate/one atomic move), not portable power-loss
  durability or authentication. Version retention and broader external-state recovery remain later
  decisions; no caller cursor/RNG/scheduler/scaler data is implicitly recovered.

## Related plans and published APIs

- [NN](../nn/master-plan.md), [Training](../training/master-plan.md), [roadmap](../../roadmap.md)
- [Published model checkpoint API](../../../api/model-checkpoint-api.md)
- [Artifact API](../../../api/checkpoint-artifact-api.md)
- [Planning guide](../../planning-guide.md)
