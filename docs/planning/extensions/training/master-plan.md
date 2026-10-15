# Training Master Plan

## Goal

Define backend-agnostic optimizer algorithms, sessions, and training-step orchestration over
parameters declared by `extensions/nn`.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)
- [NN master plan](../nn/master-plan.md)

## Scope

- public backend-neutral optimizer algorithms
- reusable Engine-backed training sessions and steps
- stable path-based optimizer state and gradient accumulation
- immutable optimizer/session snapshot and validate-before-install restore contracts needed by
  downstream exact-resume persistence
- path-based parameter groups and backend-neutral SGD/Adam/AdamW optimizers

## Out of scope

- direct Runtime, Prepare, tuning, or concrete-backend integration
- kernel selection or backend-specific optimizer execution
- `Module`, `Parameter`, `Buffer`, layer behavior, and train/eval mode
- checkpoint file formats, codecs, filesystem publication, tokenizer artifacts, and backend
  materialization

## Module invariants

- Training owns algorithms and orchestration, not an execution pipeline.
- NN owns `Parameter` and `Buffer`; Training consumes declared parameters for optimization.
- Training intentionally depends on the public Engine facade for one compile/prepared session; it
  never imports Engine internals, Runtime, Prepare, tuning, or concrete backend modules.
- NN never depends on Training.

## Allowed dependencies

- modules/model
- extensions/nn
- the public modules/engine facade

## Forbidden dependencies

- backends/cpu
- backends/metal
- backends/cuda

## Dependency DAG and authorized frontier

Completed correctness/recovery chain:
`0001E → 0002A → 0002B → 0002C → 0003A → 0004`.
Training 0004 depends on completed Checkpoint model persistence and CPU 0024; it does not depend
on the future optional adapter. Existing broader Training capability checkpoint 0005 stays Draft.
Training-owned [Checkpoint 0005 reconstruction](../checkpoint/tasks/0005-training-state-reconstruction.md)
and [Checkpoint 0005A envelope composition](../checkpoint/tasks/0005a-model-envelope-composition.md)
are Complete; they do not authorize new execution.

Revised proposed product in [ADR 0038](../../../design/decisions/0038-declarative-model-and-training-resume.md):
automatic usable-model load and optional owned resume, not snapshot-only Model + TrainingState.
[ADR 0037](../../../design/decisions/0037-stored-training-checkpoint-adapter.md) is withdrawn Proposed;
raw state codecs/comparison remain useful. New DAG:
`NN0026 Complete → CP0005B → NN0027 semantics → CP0005C persistence → Training0006 → optional CP0006`;
completed Checkpoint 0005A also precedes the adapter. No cycle or unrelated 0005 prerequisite.

Authorized Training frontier: None. NN0026 and CP0005B are Complete, including final independent
reviews and coordinator gates. NN0027/CP0005C are Draft rows with no detailed briefs;
NN owns recurrent semantics, Checkpoint owns versioned persistence. Initial-state/output selection,
format version and Training gradient admission remain future decisions. Training0006 and adapter
remain Draft/Proposed until their separate owner bridges and contract cutovers. No optimized
RecurrentScan/backend BPTT/Data hard dependency is added by this ordering.
Current scoped lifecycle wins until Training cutover; no Training code/build changes are authorized.

Proposed package map: TrainingRecipe and TrainingRun in existing
`io.github.pho001.synaptik.training`, including a narrow nested checkpoint capture borrow.
Training owns recipe loss/signature/mode meaning, stable batch placeholders and run resource
admission/cleanup; the optional adapter owns only persistence composition.
Recipe excludes duplicate optimizer/groups encoding: saved TrainingState is sole wire authority.
Initial scalar MSE uses SUM/MEAN; objective FLOAT32/FLOAT64, low predictions explicitly cast as needed.
Static dense offset-zero signatures reuse one prepared session; no per-batch recompile or RNG advance.
Inference NN constraints may admit variable leading dimensions separately.

TrainingRun owns private model/session/shared parameter Arena/placeholders, borrows Engine and
batch storage, closes session before arenas and attempts all cleanup with retained failures.
Batch binding uses fixed prototype IDs and borrowed replaceHostStorage/clearHostStorage in finally,
not fresh IDs/list positions; roles resolve once from compiled metadata. Capture admission spans
session.state and downstream model capture. No unrestricted model can outlive dead native storage.
Repeated parameter Tensor identities across recursive paths/overlap reject; repeated forward use
of one uniquely owned parameter path is distinct.

Parallel group None, planning base `52733fcfd32d278b651302e308487e0b6c6afa81` reverified before Ready.
Clean owner implementation and independent targeted Class C review; coordinator owns shared
documents and once-only source-matched combined test/Javadoc validation. Exact metadata live in
the brief. No duplicate pure validator, transaction framework, reverse edge or new runner.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| [0001](tasks/0001-public-training-session-and-sgd-lifecycle.md) | Public Training Session and SGD Lifecycle | Complete | Stable NN parameters; published first gradients; Engine reusable sessions | Public reusable session and SGD lifecycle approved after Class C remediation. |
| [0001A](tasks/0001a-bound-parameter-snapshot-byte-sizes.md) | Bound parameter snapshot byte sizes | Complete | 0001 | Capture now rejects unsnapshotable logical or optimizer-width sizes before that parameter's buffers or graph compilation. |
| [0001B](tasks/0001b-remove-repeated-parameter-overlap-scan.md) | Remove repeated parameter overlap scan | Complete | 0001A | Fixed parameter pairs are checked at open; runs retain binding and dynamic-overlap validation. |
| [0001C](tasks/0001c-explicit-native-parameter-copy.md) | Explicit native parameter copy before training | Complete | 0001A–0001B | Caller-owned shared Arena and exact Tensor identities bridge initialized NN parameters to the existing Training storage domain. |
| [0001D](tasks/0001d-adopt-fresh-training-snapshot-payloads.md) | Adopt fresh Training snapshot payloads | Complete | 0001C | Removed four redundant full-size clones per `state()` parameter while preserving immutable detached snapshots. |
| [0001E](tasks/0001e-demand-driven-optimizer-candidates.md) | Demand-driven optimizer candidate buffers | Complete | 0001D | Inactive staging is demand-driven; active slots and rollback remain provisioned. |
| [0002A](tasks/0002a-sgd-parameter-groups.md) | SGD parameter groups | Complete | 0001E | Disjoint path-based SGD overrides have exact state/restore identity and preserve ungrouped behavior. |
| [0002B](tasks/0002b-adam-and-adamw-optimizers.md) | Adam and AdamW optimizer state | Complete | 0002A | Mixed SGD/Adam/AdamW groups have exact two-moment state/restore; independent review and repository checkpoint passed. |
| [0002C](tasks/0002c-adaptive-update-operation-order.md) | Adaptive-update operation order | Complete | 0002B; project review | Preserve documented multiply-before-divide order and finite-update admission for Adam/AdamW. |
| [0003A](tasks/0003a-exact-resume-state-ownership-decision.md) | Resume-state ownership and claim decision | Complete | 0002C; audit reconciliation | ADR 0030 accepts bounded exact continuation and the future joint restore gate; independent Class C re-review approved. |
| 0003 | Exact-resume progress state | Superseded | 0003A decision | No new Training progress API: caller owns next input, sampler/epoch, external RNG, scheduler, and scaler handoff. |
| [0004](tasks/0004-joint-restore-preflight-and-candidate-commit.md) | Joint restore on an unpublished candidate | Complete | 0003A; Checkpoint 0001 and 0004; CPU 0024 | Bounded same-route in-memory private-candidate restore proved in integration tests; independent Class C review and repository gate passed. No production facade or durable artifact. |
| 0005 | Broader training capability checkpoint | Draft | 0002A–0004 | Validate broader optimizers, state-path stability, exact-resume claims, documentation, architecture boundaries, and a representative peak-memory measurement of parameter, slot, candidate, rollback, and snapshot storage before claiming large-model support. |
| [0006](tasks/0006-supported-recipe-and-owned-run.md) | Supported recipe and owned reusable run | Draft | NN0026/0027; CP0005B/0005C; completed Training0004/Checkpoint0005; future owner contract | Proposed Training loss/signature/batch/resource lifecycle; no file I/O. |


## Milestones

- Public reusable TrainingSession and persistent SGD
- Broader optimizer and parameter-group contracts
- Exact-resume state and optional checkpoint integration

## Current status

Sessions, SGD/Adam/AdamW groups, detached state, defensive reconstruction and bounded same-route
private-candidate restore are implemented. Completed evidence lives in linked briefs.
ModelCheckpoint direct leaf save/re-save and public envelope composition are also Complete.
NN0026's whole-model semantic specification is implemented; no whole-model persistence, durable
adapter or supported recipe/run convenience is implemented.

The user revised checkpoint direction (2026-10-14): normal load returns a usable supported model,
with inference or automatic Training-owned resume. [ADR 0038](../../../design/decisions/0038-declarative-model-and-training-resume.md)
and the new Draft owner capabilities replace the snapshot-only primary UX. External cursor/RNG/
scheduler/scaler remain application-owned, not automatic exact continuation.
BatchNorm training-buffer alignment, fixed RecurrentScan BPTT, Conv3d gradients and changing dropout streams remain
fail-closed. Current native-copy/session APIs remain unchanged and available for advanced use.
Ordinary static unrolling depends on constituent-operation adjoints, not a blanket recurrent ban.

## Open questions

- NN matrix/aliases/descriptor skeleton are accepted and implemented; after CP0005B, NN0027
  decides semantic recurrent reconstruction and CP0005C decides persistence before Training0006. Loss/signature/mode
  and Training run contracts remain Proposed.
- Freeze Training-owned model transfer, coherent capture/restore bridge, batch admission and
  failure cleanup; these are bounded contract decisions before Ready, not a new prerequisite waterfall.
- Coordinate Proposed Training contract scope under ADR 0038; broader Training0005 is separate.
- Preserve existing native-only update domain and duplicate-path/overlap rejection; do not quietly
  split shared parameters or widen backend/gradient capability.
- Rollback storage remains provisioned; optional rollback policy requires a separate contract
  decision and must not be silently disabled as a memory optimization.
- Snapshot payloads remain bounded by individual Java byte arrays; chunking requires a separate
  state-representation decision, not merely larger encoded limits.
- Dynamic compiled-input overlap remains pairwise per run; replacing it requires proof for native,
  heap and sliced segment identities and preserved diagnostics.
- Caller progress/RNG and large-model aggregate memory policy remain outside this artifact.
  Stored encoded-byte limits do not establish a peak-memory budget.

## Decisions made

- [ADR 0018](../../../design/decisions/0018-public-training-session-and-sgd-lifecycle.md) selects
  one borrowed public Engine and one owned reusable inference session rather than a second
  execution pipeline.
- `Parameter` ownership and train/eval behavior belong to `extensions/nn`.
- [ADR 0030](../../../design/decisions/0030-exact-training-resume-state-ownership.md) keeps
  caller next-input/progress and external state outside `TrainingState`; the fixed session does
  not advance graph RNG between runs, and no new Training 0003 public API is selected.
- [ADR 0035](../../../design/decisions/0035-private-candidate-training-restore.md) selects the
  private-candidate validation/publication order for Training 0004.
- Training owns optimizer/session snapshot semantics; Checkpoint owns durable encoding and file
  publication. A separate optional Training Checkpoint adapter preserves model-only checkpoint
  use without forcing a Training dependency.
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Introducing concrete backend bridges into the training extension.
- Saving optimizer slots without stable parameter paths or restoring one owner before another has
  validated could produce a partially resumed training session.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
