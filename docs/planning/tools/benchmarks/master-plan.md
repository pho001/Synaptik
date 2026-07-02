# Benchmarks Master Plan

## Goal

Provide repeatable benchmarks for compiler, prepare, kernels, scoring, and end-to-end execution.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- benchmark harnesses
- repeatable workload definitions
- result reporting
- performance regression evidence

## Out of scope

- production runtime logic
- architecture policy
- correctness substitutes for conformance tests

## Module invariants

- Benchmarks consume public or test contracts.
- Benchmark-only shortcuts never enter production modules.

## Allowed dependencies

- modules/engine and public contracts needed by each benchmark

## Forbidden dependencies

- Production modules depending on benchmark code.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|


## Milestones

- Harness and reporting
- Module benchmarks
- End-to-end benchmark suites

## Current status

Draft.

This tool is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Non-reproducible measurements or benchmark code leaking into production.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
