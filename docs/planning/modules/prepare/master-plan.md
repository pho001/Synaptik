# Prepare Master Plan

## Goal

Define and validate the shared transition from immutable compile artifacts to prepared runtime state.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- prepare context
- backend partition preparer contract
- prepared partitions
- coverage, memory, and schedule validation

## Out of scope

- concrete CPU, Metal, or CUDA lowering
- kernel selection
- backend executable and storage implementations

## Module invariants

- Shared prepare coordinates contracts and validation.
- Concrete backends own lowering and executable construction.

## Allowed dependencies

- modules/runtime
- modules/planning
- modules/compiler
- modules/config
- modules/backend-contract
- modules/trace

## Forbidden dependencies

- concrete backend implementations

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|


## Milestones

- Prepare contracts
- Partition coverage validation
- Prepared memory and schedule validation

## Current status

Draft.

This module is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Accumulating concrete backend logic in the shared prepare layer.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
