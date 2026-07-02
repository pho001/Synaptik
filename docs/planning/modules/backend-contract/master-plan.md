# Backend Contract Master Plan

## Goal

Define minimal backend identities, device identities, availability snapshots, and declarative requirements.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- backend and device identifiers
- availability snapshots
- declarative backend requirements
- device classes

## Out of scope

- operation support logic
- kernel registries
- prepare services
- runtime storage or physical buffers

## Module invariants

- Compile-time ownership uses backend identity, not live backend services.
- Contracts remain declarative and implementation-free.

## Allowed dependencies

- JDK standard library only.

## Forbidden dependencies

- compiler, planning, runtime, prepare, engine, concrete backend implementation, and Tensor API dependencies

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|


## Milestones

- Identity value types
- Availability and requirement DTOs
- Contract validation

## Current status

Draft.

This module is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Growing the module into a backend service or execution abstraction.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
