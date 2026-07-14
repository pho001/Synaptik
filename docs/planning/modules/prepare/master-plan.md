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
- a future narrow opaque-candidate orchestration boundary for model tuning, only after compiler,
  planning, concrete backend, and cache contracts are stable
- explicit workload-cache and model-plan artifact load/fallback handoff before runtime

## Out of scope

- concrete CPU, Metal, or CUDA lowering
- kernel selection
- backend executable and storage implementations
- backend-specific candidate vocabulary or search-space generation
- tuning measurement, comparison, and persistence algorithms

## Module invariants

- Shared prepare coordinates contracts and validation.
- Concrete backends own lowering and executable construction.
- Shared prepare exposes complete candidates opaquely and does not interpret private backend
  fields.
- Compatible cache hits and safe heuristics can prepare correct work without a tuning search.
- Model-autotuning results become explicit prepared or cache state before runtime, never hidden
  global state.

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

- The smallest opaque candidate and artifact-lifecycle boundary waits for stable compiler,
  planning, backend, engine, and persistence consumers. No Java declaration or file format is
  selected here.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Compiler, planning, and concrete backends generate candidates for decisions they own. Shared
  prepare coordinates complete candidates and validation without interpreting backend knobs.
- Cache incompatibility or corruption must fail closed to safe heuristics or an explicit miss;
  runtime never performs cache lookup or mutation.

## Risks

- Accumulating concrete backend logic in the shared prepare layer.
- Turning an opaque orchestration boundary into a generic parameter language or central knob
  registry.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
