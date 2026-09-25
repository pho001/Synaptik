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
| 0001 | [Benchmark report and reproducible harness](tasks/0001-cpu-baseline.md) | Complete | Operational lifecycle and stable workload contracts | Public CPU Engine lifecycle harness covers six families with smoke/baseline and strict schema-3 evidence profiles; it remains report-only. |
| 0002 | Operation and operation-family suites | Draft | 0001, stable workload classification | Add fixed representative workloads without inventing a production `OperationFamily` contract. |
| 0003 | Model and end-to-end suites | Draft | 0001, operational engine paths | Compare complete model and lifecycle behavior with the same report-only boundary. |

Cross-area [Metal Task 0043](../../backends/metal/tasks/0043-reproducible-metal-route-benchmark.md)
is Complete at remediation `77e6091b` after same-reviewer Class C approval with zero remaining
findings. Metal owns its lifecycle facts and the separate Task 0044 documentation/audit no-change
evaluation; this tools plan owns the report-only boundary and has no duplicate local task.


## Milestones

- Harness and reporting
- Module benchmarks
- End-to-end benchmark suites

## Current status

Complete through benchmark task 0001 and CPU 0010M's evidence hardening. Cross-area Metal 0043 is
Complete after adding one fixed `[1_048_576]` singleton FLOAT32 `NEG` report over both current
opaque Metal candidates, with thread-safe snapshot attestation, ordinary untraced timing, exact
raw-bit checks, paired smoke/baseline sampling, reconstructible generator schema 2, and no winner,
cache I/O, threshold, or retained result artifact. Metal Task 0044 owns only its documentation/audit
no-change evaluation; local tasks 0002 and 0003 remain Draft.

The directly runnable non-evidence profiles are:

```bash
./gradlew :tools:benchmarks:benchmark -Pprofile=smoke
./gradlew :tools:benchmarks:benchmark -Pprofile=baseline
```

`evidence` fixes representative shapes, five warmups, seven retained repeated measurements, three
one-shot observations, a per-sample 25 ms floor, and fixed JVM compilation/heap flags. It also
requires exact source/configuration/hash, fork/order, CPU, species, worker, materialization, and
OpenBLAS metadata. CPU 0010M's manifest retains its exact structured invocations, five regenerated
machine-local scalar reports, and a minimum future-comparison acceptance framework under
`backends/cpu/evidence/cpu-0010m-defaults/`; it is not a sealed measurement protocol.

Public Engine intentionally exposes no backend-private selected plan. Current schema-3 reports
therefore mark themselves ineligible for a production route/default decision. CPU 0010M completed
with the evidence-backed conclusion that production remains scalar/1/1/1; the invalidated
scalar/vector attempt and its pooled aggregates were removed. The benchmark protocol is hardened,
but a future comparison still requires a separately reviewed, fully sealed workload/target/
statistical matrix. Exact procedure and limitations are recorded in
[CPU task 0010M](../../backends/cpu/tasks/0010m-evidence-backed-production-defaults.md).

## Open questions

- A future production-default comparison needs an approved source of verified selected-plan facts
  without moving backend policy into Engine or making benchmark metadata authoritative.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Benchmarking never selects or mutates production settings. The separate explicit
  model-autotuning workflow belongs to `tools/tuning`.

## Risks

- Non-reproducible measurements or benchmark code leaking into production.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
