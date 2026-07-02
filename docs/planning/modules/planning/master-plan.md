# Planning Master Plan

## Goal

Make backend-neutral compile-time ownership, partitioning, capability, and logical memory decisions.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- backend intent propagation
- capability query contracts and matrices
- backend-neutral ownership scoring
- maximal same-owner partitioning
- logical materialization and memory requirements

## Out of scope

- fusion implementation
- concrete kernel or route selection
- physical allocation
- prepared execution and runtime residency

## Module invariants

- Planning answers where work runs, not which implementation runs it.
- Scoring uses compile-time information only.
- Scoring output is backend ownership.

## Allowed dependencies

- modules/model
- modules/config
- modules/backend-contract
- modules/trace

## Forbidden dependencies

- runtime, prepare, engine, and concrete backend modules

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|


## Milestones

- Intent and capability
- Ownership scoring
- Partitioning and logical memory

## Current status

Draft.

This module is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Leaking backend route selection or runtime residency into compile-time scoring.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
