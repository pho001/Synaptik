# CUDA Backend Master Plan

## Goal

Implement CUDA capability, backend-owned lowering, kernels, storage, native integration, and prepared execution.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- CUDA capability provider
- partition lowering, specialization, fusion, and route selection
- CUDA executables, storage, workspace, and transfers
- native bridge and typed tracing

## Out of scope

- global compiler logic
- public Tensor ownership
- engine dependency
- training-owned CUDA optimizer bridge

## Module invariants

- CUDA prepare owns concrete implementation selection.
- CUDA backend never depends on engine.
- Runtime receives only prepared executable contracts.

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

- Capability and native bridge
- Prepared execution and storage
- Kernel coverage and conformance

## Current status

Draft.

This backend is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Leaking concrete CUDA details into planning or runtime.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
