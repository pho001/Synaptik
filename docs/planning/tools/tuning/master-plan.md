# Tuning Master Plan

## Goal

Produce immutable tuning and platform profiles without owning runtime execution policy.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- candidate and profile tooling
- platform measurement orchestration
- immutable profile output
- profile inspection

## Out of scope

- runtime hot-path decisions
- backend ownership outside planning
- kernel implementations

## Module invariants

- Tuning outputs immutable profiles consumed through configuration.
- Tooling does not become a runtime service locator.

## Allowed dependencies

- modules/config
- modules/planning
- other explicitly required public contracts

## Forbidden dependencies

- private backend internals and runtime service lookup

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|


## Milestones

- Profile schema
- Measurement orchestration
- Profile persistence and inspection

## Current status

Draft.

This tool is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Embedding mutable tuning services into runtime or configuration.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
