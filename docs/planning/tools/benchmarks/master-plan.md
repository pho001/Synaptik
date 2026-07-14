# Benchmarks Master Plan

## Goal

Provide fixed reproducible workloads and observational performance reports for comparisons among
commits, models, and environments.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- benchmark harnesses
- repeatable workload definitions
- result reporting
- performance regression evidence
- operation, operation-family, model, and end-to-end workload suites

## Out of scope

- production runtime logic
- architecture policy
- correctness substitutes for conformance tests
- model-autotuning, candidate selection, or mutation of production settings and caches

## Module invariants

- Benchmarks consume public or test contracts.
- Benchmark-only shortcuts never enter production modules.
- A benchmark report is evidence only and has no production-setting side effects.

## Allowed dependencies

- modules/engine and public contracts needed by each benchmark

## Forbidden dependencies

- Production modules depending on benchmark code.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | Benchmark report and reproducible harness | Draft | Operational lifecycle and stable workload contracts | Define fixed workload identity, environment/sample evidence, lifecycle isolation, and reporting without model-autotuning or setting mutation. |
| 0002 | Operation and operation-family suites | Draft | 0001, stable workload classification | Add fixed representative workloads without inventing a production `OperationFamily` contract. |
| 0003 | Model and end-to-end suites | Draft | 0001, operational engine paths | Compare complete model and lifecycle behavior with the same report-only boundary. |


## Milestones

- Harness and reporting
- Module benchmarks
- End-to-end benchmark suites

## Current status

Draft.

This tool is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- Exact workload identity and report schema wait for the operational paths and stable
  classification they measure.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Benchmarking never selects or mutates production settings. The separate explicit
  model-autotuning workflow belongs to `tools/tuning`.

## Risks

- Non-reproducible measurements or benchmark code leaking into production.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
