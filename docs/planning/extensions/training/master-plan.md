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

- optimizer algorithms
- parameter groups over `extensions/nn` parameters
- training sessions and steps
- backend-neutral optimizer update representation
- immutable optimizer/session snapshot and validate-before-install restore contracts needed by
  downstream exact-resume persistence

## Out of scope

- backend storage access
- kernel selection
- CPU, Metal, or CUDA optimizer bridges
- backend-specific optimizer execution
- `Module`, `Parameter`, `Buffer`, layer behavior, and train/eval mode
- checkpoint file formats, codecs, filesystem publication, tokenizer artifacts, and backend
  materialization

## Module invariants

- Training owns algorithms, not backend execution.
- NN owns `Parameter` and `Buffer`; training consumes declared parameters for optimization.
- Training depends on NN, not the reverse.
- Training never depends on concrete backend modules.

## Allowed dependencies

- modules/model
- extensions/nn
- modules/config
- modules/compiler and other backend-neutral contracts when required

## Forbidden dependencies

- backends/cpu
- backends/metal
- backends/cuda

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | Optimizer and parameter-group contracts | Draft | Stable NN Parameter paths/replacement; published gradients | Define backend-neutral optimizer/group identity, configuration, and update ownership without persistent I/O. |
| 0002 | Optimizer state and strict snapshot/restore | Draft | 0001 | Define stable parameter-path associations, optimizer slots/groups, complete immutable snapshot, and validate-before-install restore without backend storage access. |
| 0003 | Training session progress and exact-resume state | Draft | 0001–0002; Engine training execution | Define step/epoch, scheduler, graph/dropout random-number-generator state, data shuffle/sampler cursor when exact resume is promised, and mixed-precision scaler state when present. |
| 0004 | Training checkpoint integration boundary | Draft | 0002–0003; Checkpoint model artifact and optional Training Checkpoint adapter | Expose one complete prevalidated training-state candidate to the downstream persistence adapter; keep bytes, files, checksums, and atomic publication outside Training. |
| 0005 | Training capability checkpoint | Draft | 0001–0004 | Validate updates, state-path stability, snapshot/restore atomicity, exact-resume claims, documentation, and architecture boundaries. |


## Milestones

- Optimizer contracts over NN-declared parameters
- Initial optimizer steps
- Training session integration

## Current status

Draft. Persistent training checkpoint I/O remains downstream in the proposed Checkpoint program.
Training must first expose a complete immutable state snapshot and a restore operation that can
validate model-associated optimizer/session state before any installation.

This extension is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- Define exact-resume scope for data samplers and external data sources; an unavailable sampler
  cursor must downgrade the claim rather than be silently omitted.
- Define a cross-owner validate/install protocol for model state plus optimizer/session state
  without making NN depend on Training or persistence.

## Decisions made

- The implementation must follow the current architecture contract.
- `Parameter` ownership and train/eval behavior belong to `extensions/nn`.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Training owns optimizer/session snapshot semantics; Checkpoint owns durable encoding and file
  publication. A separate optional Training Checkpoint adapter preserves model-only checkpoint
  use without forcing a Training dependency.

## Risks

- Introducing concrete backend bridges into the training extension.
- Saving optimizer slots without stable parameter paths or restoring one owner before another has
  validated could produce a partially resumed training session.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
