# ADR 0008: Separate performance evidence and model-autotuning concerns

## Status

Accepted — 2026-07-14

## Context

The planned configuration frontier proposed broad `PlatformProfile`, `BackendProfile`, and
`TuningProfile` records with backend-wide averages. A first replacement separated representative
platform calibration from model/workload autotuning. Both designs were unsatisfactory: the broad
profiles mixed backend-neutral ownership estimates with private route configuration, while the
calibration design created a second workflow whose defaults model tuning would later refine.

The architecture instead needs one explicit user-level model-autotuning workflow that can reuse
identical local workload results while still measuring bounded complete graph and preparation
plans. It must preserve compiler, planning, prepare, and concrete-backend ownership; keep
benchmarking observational; keep tuning out of runtime; and make persistent artifacts explicit.

## Decision drivers

- keep fixed-workload benchmarking reproducible and free of production-setting side effects;
- reuse local measurements by exact workload compatibility rather than a coarse operation family;
- let backend authors define typed candidate spaces without exposing private knobs to shared code;
- measure the surrounding graph and plan context that local route timing cannot predict;
- keep tuning optional for correctness and outside runtime execution;
- persist compatible results explicitly and safely; and
- keep planning cost backend-neutral.

## Options considered

### Broad platform, backend, and tuning profiles

Store global transfer, execution, materialization, and route averages together. This is compact,
but it conflates ownership estimation with backend tuning and erases differences among exact data
types, shapes, layouts, attributes, policies, targets, and routes.

### Platform calibration followed by model autotuning

Search representative platform-wide workload buckets, install robust defaults, and later refine
them for a model. This improves granularity, but it creates a separate baseline subsystem and
profile even though the same model-guided workflow and workload cache can pre-seed a target from a
representative model corpus.

### One model-autotuning workflow with two coordinated phases

Reuse or measure canonical local workloads first, then measure a bounded set of complete valid
plans end to end. Keep candidate generation with each decision owner, expose backend candidates
opaquely, and persist workload and model-plan results separately.

## Decision

Synaptik adopts the third option.

`tools/benchmarks` runs fixed reproducible operation, operation-family, model, and end-to-end
workloads and produces rich observational reports. It never selects or mutates production
settings, caches, or prepared plans.

`tools/tuning` coordinates one explicit model-autotuning workflow:

1. It extracts actual tunable workloads and routes from the supplied model, forms canonical
   workload signatures, deduplicates identical signatures while retaining occurrence weight and
   context, reuses compatible entries from an explicit persistent workload cache, and measures
   only misses.
2. It measures a bounded set of complete valid graph, fusion, ownership/partition, layout,
   materialization, route, and configuration candidates end to end, then selects an explicit
   model plan or prepared artifact. It reuses Phase 1 results and does not repeat local parameter
   search.

An operation family selects the appropriate candidate generator. It is not a cache key for one
universal configuration. Canonical local signatures include semantics and attributes, data types,
shapes, layouts, relevant policies, and target compatibility. Identical signatures may be reused
across occurrences and models; graph and plan decisions may remain occurrence- or
subgraph-specific because their surrounding context differs.

Compiler, planning, and concrete backend prepare generate complete valid candidates for decisions
they own; shared prepare coordinates those candidates and owns its validation contracts. A future
narrow prepare/tuning orchestration boundary exposes backend candidates opaquely. Shared
orchestration measures and selects without interpreting private backend fields or taking over
semantics, transformations, ownership rules, lowering, or route logic. This ADR deliberately
declares no Java API.

Concrete backend candidate generators are typed, version-controlled, tested, and colocated with
their routes. They derive and prune candidates from target capabilities, canonical workload
facts, and the tuning budget. CPU matrix-multiplication candidates may include supported Java
Vector API species and strategy, unroll, tile, parallelism, and OpenBLAS thread configurations.
Scalar, vector, and OpenBLAS are route-specific typed configurations, not boolean flags.
Candidate discovery uses no `Map<String,Object>`, string dispatch, reflective annotations,
central knob registry, generic configuration language, or arbitrary vector-lane promise.

The ordinary model author supplies the model, target, representative input or shape profiles,
objective, budget, constraints, and explicit caches. Backend authors supply backend candidate
spaces. Running the same workflow over a representative model corpus may pre-seed the same cache;
there is no separate platform-calibration workflow, abstraction, or profile.

Future persistence uses an explicit file-backed reusable workload tuning cache and a
model-specific plan cache or prepared-plan record. Compatible hits are reused; misses may be tuned
and atomically persisted. Artifacts include explicit schema and backend candidate-schema
versions, target and workload or model fingerprints, objective and constraints, and a compact
measurement summary. Incompatible entries are invalidated and corrupt data is rejected safely.
There is no hidden global cache, Java object serialization, or executable-payload assumption.
Rich evidence stays separate from compact cache state. Physical formats and prepared-executable
serialization remain deferred.

Model autotuning is optional for correctness. Safe backend heuristics and cache-only preparation
remain valid fallbacks. Runtime performs no tuning search, cache lookup or mutation, or hot-path
graph inspection; passive profiling cannot select settings.

Planning cost remains a separate backend-neutral ownership estimate and pruning mechanism. It
does not interpret routes, species, threads, tiles, chunks, kernels, or workload-cache entries.
Model tuning may measure complete planning candidates without moving backend vocabulary into
planning.

## Rationale

Canonical local signatures reuse evidence only when the semantic, data, layout, policy, and
target facts are compatible. Bounded complete-plan measurement then captures fusion, ownership,
layout, materialization, and surrounding graph effects that isolated route timing cannot predict.
Keeping candidate generation with each architecture owner preserves type safety and decision
ownership, while opaque orchestration can apply one objective and budget across the complete
model. Explicit versioned artifacts make reuse visible and invalidatable without introducing
runtime search or hidden state.

## Consequences

### Positive

- One user workflow connects the model and target to explicit cache and plan artifacts.
- Exact compatible local work is measured once and can be reused across models.
- End-to-end plan measurement captures fusion, layout, materialization, and partition context.
- Backends retain typed route ownership while shared orchestration remains generic only at the
  opaque candidate boundary.
- Benchmark reports, planning estimates, tuning caches, and runtime observations have distinct
  meanings.

### Negative and risks

- Candidate generators and compatibility fingerprints must evolve together and be versioned.
- End-to-end plan search must be deliberately bounded by budget and valid candidate generation.
- Two artifact roles and separate rich evidence require explicit lifecycle and corruption tests.
- Prepared-executable persistence may not be portable and remains unresolved until backend and
  lifecycle contracts exist.

### Migration, testing, and follow-up

The unimplemented Config 0004 broad profile specification remains retired. Config 0004 stays
Draft as a narrow backend-neutral planning cost-profile contract after its planning consumer is
stable. The former Draft platform-calibration workflow is removed. Draft-only follow-up planning
belongs to the actual owners: compiler and planning candidate production, shared prepare's opaque
orchestration and artifact lifecycle, concrete backend candidate generators, and `tools/tuning`
model-autotuning/cache orchestration. No task becomes Ready merely to preserve a queue.

Candidate-generator tests must prove that only complete valid typed configurations are exposed.
Cache tests must cover compatibility, invalidation, corruption, and atomic replacement. Tuning
tests must cover deduplication, miss-only measurement, bounded plan comparison, and safe heuristic
fallback. No dependency rule changes, so architecture tests do not change for this decision.

## Related documentation

- [Architecture contract](../../../ARCHITECTURE.md)
- [Performance evidence and model autotuning](../../architecture/performance-evidence-and-tuning.md)
- [Partition scoring](../../architecture/partition-scoring.md)
- [Runtime, prepare, and backend boundary](../../architecture/runtime-prepare-backend-boundary.md)
- [Benchmarking guide](../../developer-guide/benchmarking.md)
- [Config master plan](../../planning/modules/config/master-plan.md)
- [Planning master plan](../../planning/modules/planning/master-plan.md)
