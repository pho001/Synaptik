# Metal Backend Master Plan

## Goal

Implement Metal capability, MPSGraph and custom-kernel preparation, storage, native integration, and execution.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- Metal capability provider
- MPSGraph and custom-kernel lowering
- Metal executables, storage, workspace, and materialization
- native bridge and typed tracing

## Out of scope

- global autograd
- public Tensor ownership
- engine dependency
- training-owned Metal optimizer bridge

## Module invariants

- Metal prepare owns lowering and route selection.
- Metal-specific optimizer execution remains in backend prepare or kernels.
- Metal backend never depends on engine.

## Allowed dependencies

- modules/model
- modules/config
- modules/planning
- modules/runtime
- modules/prepare
- modules/backend-contract
- modules/trace

## Forbidden dependencies

- modules/engine
- extensions/training implementation dependencies

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|


## Milestones

- Capability and bridge
- MPSGraph preparation
- Custom routes, storage, and conformance

## Current status

Draft.

This backend is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Moving Metal lowering into shared prepare or training.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
