# Runtime Master Plan

## Goal

Execute prepared schedules and own dynamic per-run state, residency, transfers, resources, and publication.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- prepared execution contracts
- prepared schedules, slots, and memory plans
- run state and runtime resources
- residency, transfer, materialization, publication, and execution runner
- passive runtime profiling translated through typed trace contracts

## Out of scope

- graph optimization
- autograd construction
- backend discovery
- kernel selection and backend lowering
- model-autotuning search, tuning-cache lookup or mutation, hot-path graph inspection, and
  production-setting selection

## Module invariants

- Runtime executes already-prepared work.
- The hot path does not use `Operation` or `CompiledNode`.
- Runtime does not depend on concrete backend implementations.
- Runtime profiling observes execution but never mutates settings.

## Allowed dependencies

- modules/config
- modules/backend-contract
- modules/trace
- architecture-approved model contracts required by runtime

## Forbidden dependencies

- modules/engine
- concrete backend modules

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|


## Milestones

- Prepared execution contracts
- Run state and resources
- Schedule runner and publication

## Current status

Draft.

This module is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Moving backend discovery or implementation selection into the hot path.
- Letting runtime profiling become hidden online tuning.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
