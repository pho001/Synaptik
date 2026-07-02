# Config Master Plan

## Goal

Define immutable, declarative configuration for compile, prepare, run, publication, platforms, and tuning.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- compile modes and optimization configuration
- backend intent and partition scoring configuration
- prepare and run configuration
- platform, backend, and tuning profiles

## Out of scope

- live services
- kernel types
- runtime state
- backend implementation logic

## Module invariants

- Configuration is declarative data.
- Concrete backends interpret backend-specific prepare settings.
- Configuration does not select executable implementations.

## Allowed dependencies

- JDK standard library and explicitly justified declarative contract types.

## Forbidden dependencies

- concrete backend implementations
- runtime executable units and mutable runtime state

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|


## Milestones

- Compile configuration
- Prepare and run configuration
- Profiles and validation

## Current status

Draft.

This module is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Embedding service objects or concrete implementation choices in configuration.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
