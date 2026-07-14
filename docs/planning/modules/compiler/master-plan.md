# Compiler Master Plan

## Goal

Compile tensor expressions into immutable compile artifacts through capture, validation, transformation, autograd, and planning orchestration.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- graph capture and indexing
- shape and data type inference and validation
- canonicalization and graph optimization
- complete valid backend-neutral graph-transformation candidate generation for later bounded
  model tuning
- autograd and backward graph construction
- publication, planning, logical memory orchestration, and diagnostics

## Out of scope

- physical buffers
- prepared schedules and executions
- backend-specific lowering
- concrete kernel selection

## Module invariants

- Compiler output is immutable compile-time state.
- Compiler never constructs runtime execution units.
- Compiler has no concrete backend dependency.
- Compiler owns the semantics and validity of graph candidates; tuning tooling may measure a
  bounded set but does not construct or reinterpret them.

## Allowed dependencies

- modules/model
- modules/config
- modules/planning
- modules/backend-contract
- modules/trace

## Forbidden dependencies

- runtime, prepare, engine, and concrete backend modules

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|


## Milestones

- Capture and validation
- Optimization and autograd
- Planning orchestration and compile artifacts

## Current status

Draft.

This module is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- The candidate boundary remains Draft until graph transformations, compile artifacts, and the
  prepare/tuning orchestration consumer are stable. No public Java declaration is selected here.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Model autotuning does not move backend-neutral graph-transformation ownership out of the
  compiler or backend-specific fusion ownership out of concrete backends.

## Risks

- Creating prepared or backend-specific state during compilation.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
