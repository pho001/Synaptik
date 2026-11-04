# Benchmarks Master Plan

## Goal

Provide fixed reproducible workloads and observational performance reports for comparisons among
[commits](../../../glossary.md#commit-release-distribution-log-and-seed-meanings), models, and environments.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../glossary.md#architecture-contract))</span>
- [Module boundaries](../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../glossary.md#term-module))</span>
- [Dependency rules](../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../glossary.md#term-dependency))</span>

## [Scope](../../../glossary.md#term-scope)

- benchmark harnesses
- repeatable workload definitions
- result reporting
- performance [regression](../../../glossary.md#term-regression) evidence
- [operation](../../../glossary.md#operation), operation-family, [model](../../../glossary.md#term-model), and end-to-end workload suites

## Out of [scope](../../../glossary.md#term-scope)

- production [runtime](../../../glossary.md#term-runtime) logic
- architecture policy
- correctness substitutes for [conformance](../../../glossary.md#term-conformance) tests
- [model](../../../glossary.md#term-model)-[autotuning](../../../glossary.md#autotuning--model-autotuning), candidate selection, or mutation of production settings and caches

## [Module](../../../glossary.md#term-module) invariants

- Benchmarks consume public or test contracts.
- Benchmark-only shortcuts never enter production modules.
- A [benchmark report](../../../glossary.md#term-benchmark-report) is evidence only and has no production-setting side effects.

## Allowed dependencies

- [modules](../../../glossary.md#term-module)/[engine](../../../glossary.md#term-engine) and public contracts needed by each benchmark

## Forbidden dependencies

- Production [modules](../../../glossary.md#term-module) depending on benchmark code.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Benchmark report and reproducible harness](tasks/0001-cpu-baseline.md)<span data-glossary-navigation="true"> ([Benchmark report glossary](../../../glossary.md#term-benchmark-report))</span> | Complete | Operational [lifecycle](../../../glossary.md#lifecycle) and stable workload contracts | Public [CPU](../../../glossary.md#term-cpu) [Engine](../../../glossary.md#term-engine) lifecycle harness covers six families with [smoke](../../../glossary.md#term-smoke-test)/baseline and strict [schema](../../../glossary.md#term-schema)-3 evidence profiles; it remains report-only. |
| 0002 | [Operation](../../../glossary.md#operation) and operation-family suites | Draft | 0001, stable workload classification | Add fixed representative workloads without inventing a production `OperationFamily` contract. |
| 0003 | [Model](../../../glossary.md#term-model) and end-to-end suites | Draft | 0001, operational engine paths | Compare complete model and lifecycle behavior with the same report-only boundary. |

Cross-area [Metal Task 0043](../../backends/metal/tasks/0043-reproducible-metal-route-benchmark.md)
is Complete at remediation `77e6091b` after same-reviewer Class C approval with zero remaining
findings. [Metal](../../../glossary.md#term-metal) owns its lifecycle facts and the separate Task 0044 documentation/audit no-change
[evaluation](../../../glossary.md#term-evaluation); this tools plan owns the report-only boundary and has no duplicate local task.


## Milestones

- Harness and reporting
- [Module](../../../glossary.md#term-module) benchmarks
- End-to-end benchmark suites

## Current status

Complete through benchmark task 0001 and [CPU](../../../glossary.md#term-cpu) 0010M's evidence hardening. Cross-area [Metal](../../../glossary.md#term-metal) 0043 is
Complete after adding one fixed `[1_048_576]` [singleton](../../../glossary.md#term-singleton) [FLOAT32](../../../glossary.md#term-float32) `NEG` report over both current
[opaque](../../../glossary.md#term-opaque) Metal candidates, with [thread-safe](../../../glossary.md#term-thread-safe) [snapshot](../../../glossary.md#term-snapshot) attestation, ordinary untraced timing, exact
raw-bit checks, paired [smoke](../../../glossary.md#term-smoke-test)/baseline [sampling](../../../glossary.md#term-sampling-machine-learning-and-hyperparameters), reconstructible generator [schema](../../../glossary.md#term-schema) 2, and no winner,
[cache](../../../glossary.md#term-cache) I/O, threshold, or retained result [artifact](../../../glossary.md#term-artifact). Metal Task 0044 owns only its documentation/audit
no-change [evaluation](../../../glossary.md#term-evaluation); local tasks 0002 and 0003 remain Draft.

The directly runnable non-evidence profiles are:

```bash
./gradlew :tools:benchmarks:benchmark -Pprofile=smoke
./gradlew :tools:benchmarks:benchmark -Pprofile=baseline
```

`evidence` fixes representative [shapes](../../../glossary.md#shape), five warmups, seven retained repeated measurements, three
one-shot observations, a per-[sample](../../../glossary.md#term-sample-machine-learning-and-hyperparameters) 25 ms floor, and fixed [JVM](../../../glossary.md#term-java-virtual-machine-jvm) [compilation](../../../glossary.md#compiler-engine-and-execution-stage-names)/[heap](../../../glossary.md#term-heap) flags. It also
requires exact source/configuration/[hash](../../../glossary.md#term-hash), [fork](../../../glossary.md#term-fork-meanings)/order, CPU, species, worker, [materialization](../../../glossary.md#materialization), and
[OpenBLAS](../../../glossary.md#term-openblas) [metadata](../../../glossary.md#term-metadata). CPU 0010M's [manifest](../../../glossary.md#term-manifest) retains its exact structured invocations, five regenerated
machine-local [scalar](../../../glossary.md#term-scalar) reports, and a minimum future-comparison acceptance framework under
`backends/cpu/evidence/cpu-0010m-defaults/`; it is not a [sealed](../../../glossary.md#term-sealed) [measurement protocol](../../../glossary.md#term-measurement-protocol).

Public [Engine](../../../glossary.md#term-engine) intentionally exposes no [backend](../../../glossary.md#backend)-private selected plan. Current schema-3 reports
therefore mark themselves ineligible for a production route/default decision. CPU 0010M completed
with the evidence-backed conclusion that production remains scalar/1/1/1; the invalidated
scalar/[vector](../../../glossary.md#term-vector) attempt and its pooled aggregates were removed. The benchmark protocol is hardened,
but a future comparison still requires a separately reviewed, fully sealed workload/[target](../../../glossary.md#term-target)/
statistical [matrix](../../../glossary.md#term-matrix-meanings). Exact procedure and limitations are recorded in
[CPU task 0010M](../../backends/cpu/tasks/0010m-evidence-backed-production-defaults.md).

## Open questions

- A future production-default comparison needs an approved source of verified selected-plan facts
  without moving backend policy into Engine or making benchmark metadata authoritative.<span data-glossary-navigation="true"> ([backend glossary](../../../glossary.md#backend); [Engine glossary](../../../glossary.md#term-engine); [metadata glossary](../../../glossary.md#term-metadata))</span>

## Decisions made

- The implementation must follow the current [architecture contract](../../../glossary.md#architecture-contract).
- Legacy code is capability evidence only; new implementation is written from scratch.
- Benchmarking never selects or mutates production settings. The separate explicit
  model-autotuning workflow belongs to <span data-glossary-navigation="true"> ([Benchmarking glossary](../../../glossary.md#term-benchmarking); [model glossary](../../../glossary.md#term-model); [autotuning glossary](../../../glossary.md#autotuning--model-autotuning))</span>`tools/tuning`.

## Risks

- Non-reproducible measurements or benchmark code leaking into production.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md)<span data-glossary-navigation="true"> ([planning glossary](../../../glossary.md#term-planning))</span>.
