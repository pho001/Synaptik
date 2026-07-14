# Performance evidence and model autotuning

[`ARCHITECTURE.md`](../../ARCHITECTURE.md) is authoritative. This document explains how fixed
benchmark evidence, model autotuning, backend-local candidate generation, planning cost, runtime
profiling, and persistent tuning artifacts remain separate.

## Purpose and mental model

The user supplies one explicit set of inputs to one optional model-autotuning workflow:

```text
model + target + representative input/shape profiles
      + objective + budget + constraints + explicit caches
  -> model autotuning
     1. reuse or measure canonical local workloads
     2. measure bounded complete plan candidates end to end
  -> updated workload cache + model plan result or prepared artifact
  -> runtime executes the selected result
```

Benchmarking remains a separate report-only activity. Planning cost remains a separate
backend-neutral estimate used to prune or rank ownership choices. Runtime profiling remains
passive observation.

All contracts and artifacts on this page are planned. The diagram establishes ownership and
boundaries; it does not declare a current Java API or physical file format.

## Benchmarking is fixed and observational

`tools/benchmarks` runs fixed reproducible operation, operation-family, model, and end-to-end
workloads. A benchmark report records the workload, environment, supplied settings, samples, and
statistics needed to compare commits, models, or environments.

Benchmarking never selects or mutates a production setting, tuning cache, or prepared plan. It
may record which route and configuration were supplied, but it does not install the fastest
result. Correctness and conformance tests remain separate gates.

## One model-autotuning workflow

`tools/tuning` coordinates one explicit workflow with two related phases. The phases share the
same model, target, objective, budget, constraints, and explicit cache inputs, but they answer
different questions.

### Phase 1: model-guided workload and route tuning

The workflow extracts actual tunable workloads and routes from the model. For each one it forms a
canonical workload signature from semantics and attributes, input and output data types, shapes,
layouts, relevant policies, and target compatibility. Identical signatures are deduplicated for
measurement while retaining their occurrence weight and surrounding context for later plan
evaluation.

The workflow loads the explicit persistent workload cache and reuses only compatible hits. It
measures backend-generated configurations only for cache misses, then records the selected local
result. The operation family chooses which candidate generator to ask; it is not itself a cache
key and never receives one universal configuration. Identical signatures can reuse a result
across occurrences and models.

Running this same workflow over a representative model corpus may pre-seed the same workload
cache for a target. That is a use of model autotuning, not a separate platform-calibration
subsystem, baseline workflow, or profile.

### Phase 2: bounded graph and plan tuning

The workflow then measures a bounded set of complete valid candidates end to end. Depending on
the candidate, variation may include graph transformations and fusion, backend ownership and
partitioning, layouts and materializations, or backend routes and their already selected local
configurations. The result is one explicit model plan or prepared artifact.

This phase uses Phase 1 cache results and does not repeat local route-parameter search. A graph or
plan decision may remain occurrence- or subgraph-specific because surrounding fusion, layout,
materialization, and partition context can differ even when two local workloads have identical
signatures.

## Candidate ownership and opaque orchestration

Each architecture owner generates complete candidates only for its own decisions:

| Owner | Candidate responsibility |
|---|---|
| Compiler | Complete valid backend-neutral graph-transformation alternatives |
| Planning | Complete valid backend ownership and partition alternatives, guided or pruned by backend-neutral cost estimates |
| Shared prepare | A future narrow orchestration boundary that coordinates complete candidates without interpreting private fields |
| Concrete backend prepare | Complete valid backend-specific fusion, lowering, route, layout/materialization, and route-configuration alternatives |
| `tools/tuning` | Bounded measurement, cache coordination, comparison against the objective and constraints, and explicit result selection |

The table shows decision ownership. Tuning does not take over graph semantics, compiler
transformations, planning policy, backend lowering, route logic, or private configuration
vocabulary.

Backend candidate generators are typed, version-controlled, tested, and colocated with the
concrete routes they configure. They derive and prune complete valid configurations from target
capabilities, workload facts, and the tuning budget. For CPU matrix multiplication, candidates
might include supported Java Vector API species and strategy, unroll, tile, parallelism, and
OpenBLAS thread configurations. Scalar, vector, and OpenBLAS configurations are distinct typed
route values rather than booleans in a parameter bag.

Candidate discovery does not use `Map<String,Object>`, string dispatch, reflective annotations, a
central knob registry, a generic configuration language, or arbitrary vector-lane promises. The
backend author defines candidates. An ordinary model author supplies only the model,
representative shapes or inputs, objective, budget, constraints, and explicit caches.

## Persistent caches and compatibility

The future workflow uses two explicit file-backed artifact roles:

- a reusable workload tuning cache keyed by canonical workload signature and target
  compatibility; and
- a model-specific plan cache or prepared-plan record keyed by the model, target, representative
  profiles, objective, and constraints.

Loading occurs before measurement. Compatible hits are reused; misses may be tuned and then
atomically persisted. Each artifact carries an explicit artifact-schema version, backend
candidate-schema versions, target and workload or model fingerprints, objective and constraints,
and a compact measurement summary. Incompatible entries are invalidated, and corrupt data is
rejected safely.

The caches are never hidden process-global state, Java object serialization, or assumed
executable payloads. Rich candidate and measurement evidence remains separate from compact cache
state. The physical file encoding is deliberately not selected here. Serialization of a prepared
executable remains dependent on backend and lifecycle contracts and is deferred.

## Planning cost stays backend-neutral

The planning cost model estimates backend ownership from graph, transfer, materialization, and
boundary facts. It may prune or rank complete ownership candidates, but it never interprets a
route, Java Vector API species, thread count, tile, chunk, kernel, or workload-cache value.

Model autotuning may measure bounded complete plans that planning generated. That does not move
backend vocabulary into planning or turn the planning cost model into a tuning parameter set.

## Correctness fallback and runtime boundary

Model autotuning is optional for correctness. When tuning is disabled, a workload-cache entry is
absent, or an artifact is incompatible or corrupt, backend preparation uses safe heuristics and
valid complete candidates. Cache-only preparation may reuse compatible entries without running a
search.

Runtime executes the selected prepared schedule. It performs no search, tuning-cache lookup or
mutation, or hot-path graph inspection. Runtime profiling may passively describe actual execution
through typed trace data-transfer objects, but the observation does not change the current or a
later run's settings.

## Related decisions and documentation

- [ADR 0008: Separate performance evidence, model autotuning, planning cost, and runtime observation](../design/decisions/0008-performance-evidence-and-tuning-boundaries.md)
- [Partition scoring](partition-scoring.md)
- [Runtime, prepare, and backend boundary](runtime-prepare-backend-boundary.md)
- [Benchmarking guide](../developer-guide/benchmarking.md)
