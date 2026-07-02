# Trace Master Plan

## Goal

Define typed, serializable diagnostic DTOs shared by compile, prepare, run, and backend trace producers.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- trace envelopes and phases
- typed payload families
- trace-local identifiers
- typed backend attribute escape hatch

## Out of scope

- graph traversal
- business logic
- runtime state
- backend execution

## Module invariants

- Trace remains a DTO-only dependency leaf.
- `Map<String,String>` is never the primary trace model.
- Producers translate local state into trace-local types.

## Allowed dependencies

- JDK standard library only.

## Forbidden dependencies

- model, planning, compiler, runtime, prepare, engine, and concrete backend modules

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|


## Milestones

- Core envelope and identifiers
- Lifecycle payload families
- Serialization and schema validation

## Current status

Draft.

This module is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Allowing producer-domain types or unstructured string maps into shared trace contracts.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
