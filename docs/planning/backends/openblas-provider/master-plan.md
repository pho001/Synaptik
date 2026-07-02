# OpenBLAS Provider Master Plan

## Goal

Provide a low-level OpenBLAS leaf for library loading, symbol binding, GEMM calls, and thread control.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- native library resolution
- symbol binding
- GEMM invocation
- OpenBLAS thread control

## Out of scope

- configuration interpretation
- fallback and backend ownership
- prepared execution
- Tensor API and residency

## Module invariants

- The provider remains a low-level leaf.
- Dependency direction is CPU backend to OpenBLAS provider, never the reverse.

## Allowed dependencies

- JDK standard library and required native interop APIs.

## Forbidden dependencies

- model Tensor API
- planning, compiler, runtime, prepare, engine, and concrete backend dependencies

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|


## Milestones

- Library and symbols
- GEMM contract
- Thread control and native validation

## Current status

Draft.

This backend is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Allowing CPU backend policy or fallback logic into the provider.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
