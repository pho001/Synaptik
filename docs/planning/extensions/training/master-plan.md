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

## Out of scope

- backend storage access
- kernel selection
- CPU, Metal, or CUDA optimizer bridges
- backend-specific optimizer execution
- `Module`, `Parameter`, `Buffer`, layer behavior, and train/eval mode

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


## Milestones

- Optimizer contracts over NN-declared parameters
- Initial optimizer steps
- Training session integration

## Current status

Draft.

This extension is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current architecture contract.
- `Parameter` ownership and train/eval behavior belong to `extensions/nn`.
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Introducing concrete backend bridges into the training extension.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
