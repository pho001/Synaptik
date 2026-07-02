# Engine Master Plan

## Goal

Provide the public lifecycle facade and explicit composition root for compiler, prepare, runtime, and concrete backends.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- public compiled graph facade
- explicit backend registration
- compile and prepare orchestration
- composition of runtime, validators, tracing, and backends

## Out of scope

- kernel implementations
- backend internals
- graph optimizer passes
- runtime service locator and core reflective discovery

## Module invariants

- Engine is the outer composition root.
- Backends are registered explicitly.
- Concrete backends never depend on engine.

## Allowed dependencies

- modules/compiler
- modules/runtime
- modules/prepare
- modules/config
- modules/trace
- concrete backend modules

## Forbidden dependencies

- No inward module may depend on engine.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|


## Milestones

- Explicit backend composition
- Compile facade
- Prepare and run lifecycle facade

## Current status

Draft.

This module is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Becoming a service locator or absorbing backend implementation details.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
