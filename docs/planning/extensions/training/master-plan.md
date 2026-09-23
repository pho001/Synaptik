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
- future parameter groups and broader optimizers

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

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| [0001](tasks/0001-public-training-session-and-sgd-lifecycle.md) | Public Training Session and SGD Lifecycle | Complete | Stable NN parameters; published first gradients; Engine reusable sessions | Public reusable session and SGD lifecycle approved after Class C remediation. |
| 0002 | Parameter groups and broader optimizers | Draft | 0001 | Decide stable group identity and add only separately authorized optimizer algorithms. |
| 0003 | Exact-resume progress state | Draft | 0001–0002; Data lifecycle | Define epoch/scheduler/graph RNG/data sampler cursor and mixed-precision scaler state only where an exact-resume claim is supportable. |
| 0004 | Training checkpoint integration boundary | Draft | 0002–0003; Checkpoint model artifact and optional Training adapter | Expose one coordinated prevalidated state candidate; keep bytes, files, checksums, and atomic publication outside Training. |
| 0005 | Broader training capability checkpoint | Draft | 0001–0004 | Validate broader optimizers, state-path stability, exact-resume claims, documentation, and architecture boundaries. |


## Milestones

- Public reusable TrainingSession and persistent SGD
- Broader optimizer and parameter-group contracts
- Exact-resume state and optional checkpoint integration

## Current status

Task [0001](tasks/0001-public-training-session-and-sgd-lifecycle.md) is Complete from exact base
`2e110c7688ead480faa4215ff67f9814fa81f4ad`. Mandatory external Class C review first returned
`BLOCK` with six P1 and one P2 findings. After remediation and full validation, external re-review
returned `APPROVE` with findings `0`. The task provides the first public TrainingSession, SGD,
accumulation, detached in-memory state, strict restore, CPU/Metal coverage, and the intentional
public-Engine dependency recorded by ADR 0018. Parameter groups, broader optimizers, durable
persistence, and exact data-progress resume remain Draft; no later Training task is Ready.

## Open questions

- Define exact-resume scope for data samplers and external data sources; an unavailable sampler
  cursor must downgrade the claim rather than be silently omitted.
- Define a cross-owner validate/install protocol for model state plus optimizer/session state
  without making NN depend on Training or persistence.

## Decisions made

- [ADR 0018](../../../design/decisions/0018-public-training-session-and-sgd-lifecycle.md) selects
  one borrowed public Engine and one owned reusable inference session rather than a second
  execution pipeline.
- `Parameter` ownership and train/eval behavior belong to `extensions/nn`.
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
