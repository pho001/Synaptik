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

`0001E -> 0002A -> 0002B -> 0002C -> 0003A -> 0004` is the completed correctness,
decision, and private-candidate recovery chain. Future work is `0004 -> 0005`.
Task 0004 also depends on Checkpoint 0001's accepted architecture, Checkpoint 0004's model
artifact, and CPU 0024's restored-heap execution repair. It does not depend on the future
Training Checkpoint adapter (Checkpoint 0006).
[0002A](tasks/0002a-sgd-parameter-groups.md) is Complete after focused validation, independent
Class B review, and the integration-owner `./gradlew test javadoc` checkpoint.
[0002B](tasks/0002b-adam-and-adamw-optimizers.md) is Complete after independent Class B approval
and the integration-owner repository test/Javadoc checkpoint. Audit repair 0002C is Complete;
0004 is Complete after its executable proof, independent Class C review, and integration gate;
0005 remains Draft. Accepted
[ADR 0035](../../../design/decisions/0035-private-candidate-training-restore.md) selects
caller-owned exclusion and an unpublished replacement using existing public APIs, with no
mandatory pure validator or production transaction facade. Task 0003A is Complete after independent
Class C approval at base `34fb1c8d6a48f9a9352c25597a86045180477fbc`; it selects a bounded
state-complete claim without a new Training 0003 API. Governing
headings are `ARCHITECTURE.md` scope index, module ownership and dependency rules, plus
`docs/architecture/contracts/extensions-training.md` sections `extensions/nn`,
`extensions/training`, and `Optimizer/training lifecycle`.

The accepted contract snapshot, validation, and limitation are recorded in the
[0004 brief](tasks/0004-joint-restore-preflight-and-candidate-commit.md). The test-only Checkpoint
dependency in `testing/integration-tests` passed the integration-owner repository gate.
The user-authorized [Checkpoint 0005](../checkpoint/tasks/0005-training-state-reconstruction.md)
is Complete: defensive reconstruction, external-package tests, final Javadoc/example validation,
and independent API review passed in Training; do not create a duplicate Training task.
Training's broader capability checkpoint 0005 remains Draft and is not this prerequisite.

Authorized frontier: None. [Checkpoint 0006](../checkpoint/tasks/0006-training-checkpoint-adapter.md)
remains Draft awaiting approval of its proposed API/wire/optional gate in
[proposed ADR 0037](../../../design/decisions/0037-stored-training-checkpoint-adapter.md).
Training's broader 0005 retains its separate Draft gate. Checkpoint
[0005A](../checkpoint/tasks/0005a-model-envelope-composition.md) is Complete after implementation,
independent Class C API/documentation approval, and configured coordinator gates. It introduces no
Training work or dependency; 0006 depends on completed 0005A but is not authorized. Reconstruction exposes only
three existing defensive constructors; trusted adoption/helpers remain internal.
The Training/Checkpoint coordinator owns shared-document
integration; exact metadata and evidence live in the linked Checkpoint brief.

The public `ParameterGroup` type in `io.github.pho001.synaptik.training` assigns stable Module
paths to immutable optimizer configuration; 0002B adds `Adam` and `AdamW` in that package and
extends only Training-owned state. NN keeps parameter declaration and traversal ownership.

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


## Milestones

- Public reusable TrainingSession and persistent SGD
- Broader optimizer and parameter-group contracts
- Exact-resume state and optional checkpoint integration

## Current status

Task [0001](tasks/0001-public-training-session-and-sgd-lifecycle.md) is Complete from exact base
`2e110c7688ead480faa4215ff67f9814fa81f4ad`. Mandatory external Class C review first returned
`BLOCK` with six P1 and one P2 findings. After remediation and full validation, external re-review
returned `APPROVE` with findings `0`. The task provides the first public TrainingSession, SGD,
accumulation, detached in-memory state, strict restore, qualified CPU parameter-carrier coverage,
and the existing Metal objective path recorded by ADR 0018. This does not claim BFLOAT16 or
FLOAT16 Metal training execution; low-precision parameters keep private FLOAT32 masters, momentum,
and accumulated gradients, with exact logical and optimizer-width snapshot payloads. Tasks
0001A–0001E are Complete after focused validation and independent review. Task 0001C also passed
a configured full-repository test/Javadoc run. Task 0001E removed idle inactive staging arrays
and was pushed as `cbd14f3b`. Task 0002A added path-based SGD groups after independent approval
and a successful repository test/Javadoc checkpoint, then was pushed as `19b7785c`. Task 0002B
added mixed SGD/Adam/AdamW groups and exact two-moment state at `10d507a3`, with independent
documentation review fix `23d1e39d` and a successful repository checkpoint. Numerical repair
0002C is Complete. Decision task 0003A's independent Class C review initially blocked on
forward-context capture, training-mode BatchNorm buffer publication, and caller-state codec scope;
targeted remediation and final re-review returned APPROVE with no findings. Joint
restore proof at Training 0004 is Complete after independent Class C review and the integration
gate. Checkpoint 0005 is Complete for Training-owned defensive reconstruction only; Checkpoint 0006
has a concrete proposed design awaiting API/wire/optional-gate approval, not Ready. The user selected Model +
TrainingState persistence (optimizer/groups/slots/accumulation/counters); data position, external
RNG, and other state absent from the artifact remain application-owned. The guarantee is restored
stored state, not complete automatic exact continuation. No durable adapter is implemented.
Completed [Checkpoint 0005A](../checkpoint/tasks/0005a-model-envelope-composition.md) now exposes
detached model envelopes, ordered descriptors, independently validated canonical raw bytes, and
strict envelope load for future comparison with TrainingState. Existing private helpers, file
semantics and module edges remain unchanged; proposed ADR 0037 supplies the adapter design,
but neither its wire nor optional build gate is approved.
Complete [Checkpoint 0004A](../checkpoint/tasks/0004a-direct-leaf-model-save.md) removes the
save-time expression-view workaround: the fixture saves the actual quiescent trained tree and
directly re-saves loaded heap state before native parameter copying on the validated CPU route.
This remains a bounded in-memory recovery proof, not durable Training continuation or generic
zero-node Engine support. Completed Checkpoint 0005 authorizes no successor or durable adapter.

## Open questions

- Approve or revise the concrete [0006 proposal](../checkpoint/tasks/0006-training-checkpoint-adapter.md)
  and coordinate its authoritative contract decision before Ready. It uses the implemented
  existing-ArtifactEnvelope capture/inspect/canonical bytes/load boundary recorded by
  [ADR 0036](../../../design/decisions/0036-model-envelope-composition.md) and completed
  [Checkpoint 0005A](../checkpoint/tasks/0005a-model-envelope-composition.md), with no reverse edge.
- Task 0001C established an explicit pre-session copy into caller-owned shared native storage for
  initialized NN parameters. Direct native initialization remains a possible later convenience,
  not an implicit `TrainingSession.open` behavior.
- Task 0001E makes only inactive staging candidates demand-driven. Rollback storage remains
  provisioned; any optional rollback policy needs a separate contract decision. Large snapshots
  remain bounded by one Java byte array per payload; chunking needs a separate state
  representation decision.
- Dynamic compiled-input overlap is still checked pairwise on each run. Replacing that check
  requires a separate proof for native, heap, and sliced segment identities and diagnostics.
- An unavailable next-input cursor or external random state downgrades the claim to partial
  restore/restart. Data 0001–0004 do not add sampling or epoch ownership.
- Saved local Module modes do not establish the effective forward contexts supplied at objective
  construction. Training-mode BatchNorm remains outside today's exact-continuation claim until
  step-aligned detached running-buffer publication is proved. Durable caller-owned state needs
  supported versioned schemas/codecs and owner validation; no generic object encoding is promised.
- Measure and document aggregate parameter/slot/candidate/snapshot capacity before claiming
  large-model support. Per-array limits and demand-driven staging do not constitute a total
  memory budget; do not disable rollback implicitly as a memory optimization.
- Training 0004 proved the selected unpublished-candidate handoff for one bounded same-route CPU
  fixture. Session-specific state is validated after private open, and the caller publishes only
  the complete replacement under exclusion. Checkpoint 0004A now supports direct trained-tree
  save and loaded heap re-save without the private expression view, using Checkpoint-owned raw
  movement requests through existing Engine compute. No generic zero-node Engine support,
  in-place rollback, general transaction API, or durable Training artifact is provided.

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
