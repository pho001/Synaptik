# Training Master Plan

## Goal

Maintain backend-agnostic optimizers/session state and generalize parameter admission independently
of NN. Training0007 implements physical `modules/training` with a small live named-parameter
boundary consumed by NN; final independent approval and the coordinator gate passed.
The [cross-area program](../../model-execution-refactor.md) owns cutover phases and gates.

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

Current ownership follows the root/scoped contract enacted by Training0007.

- Training owns algorithms and orchestration, not an execution pipeline.
- NN owns `Parameter` and `Buffer`; Training consumes generic live `TrainingParameters`.
- Training intentionally depends on the public Engine facade for one compile/prepared session; it
  never imports Engine internals, Runtime, Prepare, tuning, or concrete backend modules.
- NN depends on Training only for its live Module admission adapter, not optimizer ownership.

## Allowed dependencies

Current direct public project edges (later targets remain in the program):

- modules/model
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
raw state codecs/comparison remain useful. Earlier proposed DAG (now held):
`NN0026 Complete → CP0005B → NN0027 semantics → CP0005C persistence → Training0006 → optional CP0006`;
completed Checkpoint 0005A also precedes the adapter. No cycle or unrelated 0005 prerequisite.

[ADR0039](../../../design/decisions/0039-portable-graph-model-persistence.md) now records the five
accepted product choices and approved P2 core/API packet. [CP0005D](../checkpoint/tasks/0005d-portable-graph-persistence-decision.md)
and Engine0020 remain Complete. Engine0021 is now Complete after independent Class C approval
and the native-enabled integrated test/Javadoc gate; its brief owns evidence.
[Training0007](tasks/0007-generic-live-parameter-admission.md) is Complete after Pascal's final
independent APPROVE and the coordinator's native-enabled test/Javadoc gate. Compiler0008 and
Checkpoint0008 are also Complete after final independent reviews and coordinator gates.
Authorized Training implementation frontier: None. [Training0008](tasks/0008-fresh-value-training-session-runs.md)
is Complete after Euclid's definitive code/docs APPROVE and main's source-matched root gate
(2026-10-19); its brief owns evidence. NN0028 is Complete, accepted 2026-10-19.
The [program DAG](../../model-execution-refactor.md#phases-and-owner-dag)
replaces recurrent-first ordering for the new graph path, not the existing flexible declarative API.

CURRENT TrainingSession owns Engine PreparedExecution after Engine0021 mechanically changed its
field/constructor, open to engine.prepare(graph), execution to
engine.run(prepared, inputs), and closed/admission/cleanup checks to prepared.isClosed/close.
Keep public TrainingSession: optimizer/accumulation state, counters, busy admission and atomic
transactions justify it. Preserve live/native/precommit checks, result-before-prepared-before-seed
cleanup and failed-open rollback. No new Training API or project edge in 0021; Training fixture
session() accessors remain. Training0007/0008 below are not pulled into this migration.

Training0007 implementation: modules/training, retaining :extensions:training via projectDir and
the training Java package, api(Model, Engine) only. This complete generic admission/native-copy
capability removes NN imports in TrainingSession/TrainingParameter/TrainingParameterStorage,
adds TrainingParameters and migrates callers. NN's Module.trainingParameters() adapter is included
in that atomic boundary cutover, removing Training→NN with NN→Training. No retained NN-taking
overloads inside generic Training and no compatibility project.

TrainingParameters has ordered Entry(path, Tensor) values and live validateUnchanged().
The NN adapter rechecks current path/order/wrapper/Tensor membership, not a detached map.
Training retains native descriptor/storage/finiteness and precommit checks, optimizer algorithms,
slots/counters, rollback and failure atomicity. copyToSharedNative accepts this live source.
It is still an explicit pre-session copy into a caller-owned shared Arena; the high-level Model
must first acquire its own weight leaves, never rebind external borrowed Tensors.

Training0008 delivered fresh stored values in `inputs()` order, excluding parameters and private seed,
using Engine0021. Existing run remains exact-identity/arbitrary-order. The bounded root/owning
Training lifecycle cutover, implementation, independent review and coordinator gate are Complete.
No per-batch graph build/copy/compile/prepare or autotuning integration was added.
In completed NN0028, configured train(inputs,targets) delegates RESET_AND_STEP over the same owned weights
used by compute/save. Module.train() remains only a mode toggle. Heap-to-native admission may
copy once and invalidate old inference preparations before creating the training objective/session.
Loaded/owned values need no ownership-only duplicate; no automatic optimizer/RNG/cursor resume.

Training0007 depends on reviewed P2 and completed session/native-copy/state capabilities, not
Engine fresh binding or a codec; it builds on 0021's migrated consumer in serial integration.
Training0008 depends on 0007 and Engine0021 and uses Engine.runValues, not a removed session API. Serial integration:
Engine0021 → Training0007 → Compiler0008 → Checkpoint0008 → Training0008 → NN0028.
The shared-document order is not an invented code dependency. Parallel group None; conflicts are
NN0028 and shared root/scoped Training lifecycle and documentation writers. Current common base:
main `7386249479ccda991b96a7e27bd5ed0279ca1ae3`, including the completed Checkpoint0008 serial join.
The Training0008 brief records integration metadata, launch verification, Euclid's final APPROVE
and main's accepted source-matched root gate (2026-10-19). Model-execution coordinator owns shared
documents and combined validation; implementation is complete, with no authorized Training frontier.

Hold NN0027/CP0005C/Training0006/adapter expansion as optional older proposals, not prerequisites.
TrainingRecipe/TrainingRun and nested capture borrow remain held, not accepted additions.
The first new configured graph training API uses a scalar objective builder once; arbitrary Java
callbacks are not persisted. Existing native/gradient/backend admission remains fail-closed.
No RecurrentScan BPTT, Data, changing dropout stream or broad Training0005 gate is added.

The 0007 package map moves all existing production types unchanged in package to
`modules/training/src/main/java/io/github/pho001/synaptik/training`; adds public
`TrainingParameters` and nested `Entry` there; adds only `Module.trainingParameters()` and its
private live implementation in `io.github.pho001.synaptik.nn.module`. The brief enumerates types,
consumers, tests, coordinated authority and documentation scope. No Compiler consumer was found.

Worker validation covers affected modules once; independent targeted Class C review
reuses successful evidence. Coordinator runs one combined repository checkpoint per integrated
boundary-changing capability with the existing configured Metal dylib. The 0007 brief records
approved readiness, implementation and evidence; only its bounded root/scoped cutover is enacted.

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
| [0007](tasks/0007-generic-live-parameter-admission.md) | Generic live parameter admission and native-copy cutover | Complete | Approved P2; completed 0001C/0002C/0004 and Training-owned CP0005; Engine0021 serial base | Generic admission/native copy, relocation and NN adapter delivered; final independent APPROVE and coordinator gate passed. |
| [0008](tasks/0008-fresh-value-training-session-runs.md) | Fresh-value TrainingSession runs | Complete | 0007; Engine0021; Checkpoint0008 serial join | Fresh positional runs delivered; Euclid final APPROVE and main source-matched root gate passed; evidence in brief. |

## Milestones

- Public reusable TrainingSession and persistent SGD
- Broader optimizer and parameter-group contracts
- Exact-resume state and optional checkpoint integration

## Current status

Sessions, SGD/Adam/AdamW groups, detached state, defensive reconstruction and bounded same-route
private-candidate restore are implemented. Completed evidence lives in linked briefs.
ModelCheckpoint direct leaf save/re-save and public envelope composition are also Complete.
NN0026's whole-model semantic specification and CP0005B whole-model persistence are implemented;
no durable adapter or supported recipe/run convenience is implemented.

The user revised checkpoint direction (2026-10-14): normal load returns a usable supported model,
with inference or automatic Training-owned resume. [ADR 0038](../../../design/decisions/0038-declarative-model-and-training-resume.md)
and the new Draft owner capabilities replace the snapshot-only primary UX. External cursor/RNG/
scheduler/scaler remain application-owned, not automatic exact continuation.
BatchNorm training-buffer alignment, fixed RecurrentScan BPTT, Conv3d gradients and changing dropout streams remain
fail-closed. Current native-copy/session APIs remain unchanged and available for advanced use.
Ordinary static unrolling depends on constituent-operation adjoints, not a blanket recurrent ban.

## Open questions

- Training0007/0008 are Complete; P2 and the five product choices are accepted. Authorized Training
  frontier: None. NN0028 and Model/withEngine are Complete, accepted 2026-10-19.
  Durable recipe/restore/capture remains a
  separate held proposal; broader Training0005 is not a prerequisite.
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
  one borrowed public Engine and, after Engine0021's owner reconciliation, one owned prepared execution rather than a second
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
