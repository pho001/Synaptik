# CPU Backend Master Plan

## Goal

Implement CPU capability reporting, backend-owned preparation, kernel routes, storage, workspace, and execution.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- CPU capability provider
- partition lowering, specialization, and fusion
- scalar, Vector API, ASM, OpenBLAS, and specialized routes
- CPU executables, storage, workspace, scheduling, and tracing

## Out of scope

- global compiler logic
- public Tensor ownership
- engine dependency
- separate CPU route backends

## Module invariants

- Planning selects CPU ownership; CPU prepare selects the route.
- All CPU routes remain inside one concrete backend.
- CPU backend never depends on engine.

## Allowed dependencies

- modules/model
- modules/config
- modules/planning
- modules/runtime
- modules/prepare
- modules/backend-contract
- modules/trace
- backends/openblas-provider

## Forbidden dependencies

- modules/engine

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|


## Milestones

- Capability and scalar baseline
- Prepared execution and storage
- Optimized routes and conformance

## Current status

Draft.

This backend is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Leaking route selection into planning or splitting CPU routes into false backends.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
