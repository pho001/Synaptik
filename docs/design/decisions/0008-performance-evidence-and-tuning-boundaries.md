# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0008: Separate performance evidence and [model](../../glossary.md#term-model)-[autotuning](../../glossary.md#autotuning--model-autotuning) concerns


## Editorial reading context

The model author requests tuning with representative inputs and a budget; backend authors supply legal implementation candidates. The tuning tool compares those supplied choices and records evidence. A benchmark harness only reports a fixed workload: it does not install a choice. This decision describes the intended workflow; its future statements do not establish which stages are implemented today.

Use the [implemented tuning stages and report limits](../../developer-guide/benchmarking.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted — 2026-07-14

## Context

The planned configuration frontier proposed broad `PlatformProfile`, `BackendProfile`, and
`TuningProfile` records with [backend](../../glossary.md#backend)-wide averages. A first replacement separated representative
platform calibration from [model](../../glossary.md#term-model)/workload [autotuning](../../glossary.md#autotuning--model-autotuning). Both designs were unsatisfactory: the broad
profiles mixed backend-neutral [ownership](../../glossary.md#term-ownership-meanings) estimates with private route configuration, while the
calibration design created a second workflow whose defaults model tuning would later refine.

The architecture instead needs one explicit user-level model-autotuning workflow that can reuse
identical local workload results while still measuring bounded complete [graph](../../glossary.md#graph) and preparation
plans. It must preserve [compiler](../../glossary.md#term-compiler), [planning](../../glossary.md#term-planning), [prepare](../../glossary.md#prepare), and concrete-[backend ownership](../../glossary.md#backend-ownership); keep
[benchmarking](../../glossary.md#term-benchmarking) observational; keep tuning out of [runtime](../../glossary.md#term-runtime); and make persistent [artifacts](../../glossary.md#term-artifact) explicit.

## Decision drivers

- keep fixed-workload [benchmarking](../../glossary.md#term-benchmarking) reproducible and free of production-setting side effects;
- reuse local measurements by exact workload compatibility rather than a coarse [operation](../../glossary.md#operation) family;
- let [backend](../../glossary.md#backend) authors define typed candidate spaces without exposing private knobs to shared code;
- measure the surrounding [graph](../../glossary.md#graph) and plan context that local route timing cannot predict;
- keep tuning optional for correctness and outside [runtime](../../glossary.md#term-runtime) execution;
- persist compatible results explicitly and safely; and
- keep [planning](../../glossary.md#term-planning) cost backend-neutral.

## Options considered

### Broad platform, [backend](../../glossary.md#backend), and tuning profiles

Store global transfer, execution, [materialization](../../glossary.md#materialization), and route averages together. This is compact,
but it conflates [ownership](../../glossary.md#term-ownership-meanings) estimation with backend tuning and erases differences among exact data
types, [shapes](../../glossary.md#shape), [layouts](../../glossary.md#layout), attributes, policies, [targets](../../glossary.md#term-target), and routes.

### Platform calibration followed by [model autotuning](../../glossary.md#autotuning--model-autotuning)

Search representative platform-wide workload buckets, install robust defaults, and later refine
them for a [model](../../glossary.md#term-model). This improves granularity, but it creates a separate baseline subsystem and
profile even though the same model-guided workflow and workload [cache](../../glossary.md#term-cache) can pre-[seed](../../glossary.md#term-seed) a [target](../../glossary.md#term-target) from a
representative model corpus.

### One [model](../../glossary.md#term-model)-[autotuning](../../glossary.md#autotuning--model-autotuning) workflow with two coordinated phases

Reuse or measure [canonical](../../glossary.md#term-canonical-form) local workloads first, then measure a bounded set of complete valid
plans end to end. Keep candidate generation with each decision [owner](../../glossary.md#term-owner), expose [backend](../../glossary.md#backend) candidates
opaquely, and persist workload and model-plan results separately.

## Decision

Synaptik adopts the third option.

`tools/benchmarks` runs fixed reproducible [operation](../../glossary.md#operation), operation-family, [model](../../glossary.md#term-model), and end-to-end
workloads and produces rich observational reports. It never selects or mutates production
settings, caches, or prepared plans.

`tools/tuning` [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) one explicit model-[autotuning](../../glossary.md#autotuning--model-autotuning) workflow:

1. It extracts actual tunable workloads and routes from the supplied model, forms canonical
   workload signatures, deduplicates identical signatures while retaining occurrence weight and
   context, reuses compatible entries from an explicit persistent workload cache, and measures
   only misses.<span data-glossary-navigation="true"> ([weight glossary](../../glossary.md#term-weight))</span>
2. It measures a bounded set of complete valid graph, fusion, ownership/partition, layout,
   materialization, route, and configuration candidates end to end, then selects an explicit
   model plan or prepared artifact. It reuses Phase 1 results and does not repeat local parameter
   search.<span data-glossary-navigation="true"> ([fusion glossary](../../glossary.md#term-fusion); [partition glossary](../../glossary.md#partition); [materialization glossary](../../glossary.md#materialization); [parameter glossary](../../glossary.md#term-parameter))</span>

An operation family selects the appropriate [candidate generator](../../glossary.md#candidate-generator). It is not a [cache key](../../glossary.md#term-cache-key) for one
universal configuration. [Canonical](../../glossary.md#term-canonical-form) local [signatures](../../glossary.md#term-signature) include semantics and attributes, data types,
[shapes](../../glossary.md#shape), [layouts](../../glossary.md#layout), relevant policies, and [target](../../glossary.md#term-target) compatibility. Identical signatures may be reused
across occurrences and models; [graph](../../glossary.md#graph) and plan decisions may remain [occurrence](../../glossary.md#term-occurrence)- or
subgraph-specific because their surrounding context differs.

[Compiler](../../glossary.md#term-compiler), [planning](../../glossary.md#term-planning), and [concrete backend](../../glossary.md#concrete-backend) [prepare](../../glossary.md#prepare) generate complete valid candidates for decisions
they own; shared prepare coordinates those candidates and owns its [validation](../../glossary.md#term-validation) contracts. A future
narrow prepare/tuning orchestration boundary exposes [backend](../../glossary.md#backend) candidates opaquely. Shared
orchestration measures and selects without interpreting private backend fields or taking over
semantics, transformations, [ownership](../../glossary.md#term-ownership-meanings) rules, [lowering](../../glossary.md#lowering), or route logic. This [ADR](../../glossary.md#term-architecture-decision-record-adr) deliberately
declares no Java [API](../../glossary.md#term-application-programming-interface-api).

Concrete backend candidate generators are typed, version-controlled, tested, and colocated with
their routes. They derive and prune candidates from target capabilities, canonical workload
facts, and the tuning budget. [CPU](../../glossary.md#term-cpu) [matrix](../../glossary.md#term-matrix-meanings)-multiplication candidates may include supported Java
[Vector API](../../glossary.md#term-vector-api) species and strategy, unroll, tile, parallelism, and [OpenBLAS](../../glossary.md#term-openblas) thread configurations.
[Scalar](../../glossary.md#term-scalar), [vector](../../glossary.md#term-vector), and OpenBLAS are route-specific typed configurations, not [boolean](../../glossary.md#data-type--datatype) flags.
Candidate discovery uses no `Map<String,Object>`, string [dispatch](../../glossary.md#term-dispatch), reflective annotations,
central knob [registry](../../glossary.md#term-registry), generic configuration language, or arbitrary vector-[lane](../../glossary.md#term-lane) promise.

The ordinary model author supplies the model, target, representative input or shape profiles,
[objective](../../glossary.md#term-objective), budget, constraints, and explicit caches. Backend authors supply backend candidate
spaces. Running the same workflow over a representative model corpus may pre-[seed](../../glossary.md#term-seed) the same [cache](../../glossary.md#term-cache);
there is no separate platform-calibration workflow, abstraction, or profile.

Future [persistence](../../glossary.md#term-persistence) uses an explicit file-backed reusable workload tuning cache and a
model-specific plan cache or prepared-plan record. Compatible hits are reused; misses may be tuned
and atomically persisted. [Artifacts](../../glossary.md#term-artifact) include explicit [schema](../../glossary.md#term-schema) and backend candidate-schema
versions, target and workload or model fingerprints, objective and constraints, and a compact
measurement summary. Incompatible entries are invalidated and corrupt data is rejected safely.
There is no hidden global cache, Java object [serialization](../../glossary.md#term-serialization), or executable-[payload](../../glossary.md#artifact-envelope-and-atomic-publication) assumption.
Rich evidence stays separate from compact cache state. Physical formats and prepared-executable
serialization remain deferred.

[Model autotuning](../../glossary.md#autotuning--model-autotuning) is optional for correctness. Safe backend heuristics and cache-only preparation
remain valid fallbacks. [Runtime](../../glossary.md#term-runtime) performs no tuning search, cache [lookup](../../glossary.md#term-lookup) or mutation, or hot-path
graph inspection; passive profiling cannot select settings.

Planning cost remains a separate backend-neutral ownership estimate and pruning mechanism. It
does not interpret routes, species, threads, tiles, chunks, [kernels](../../glossary.md#kernel), or workload-cache entries.
Model tuning may measure complete planning candidates without moving backend vocabulary into
planning.

## Rationale

[Canonical](../../glossary.md#term-canonical-form) local [signatures](../../glossary.md#term-signature) reuse evidence only when the semantic, data, [layout](../../glossary.md#layout), policy, and
[target](../../glossary.md#term-target) facts are compatible. Bounded complete-plan measurement then [captures](../../glossary.md#term-capture-meanings) [fusion](../../glossary.md#term-fusion), [ownership](../../glossary.md#term-ownership-meanings),
layout, [materialization](../../glossary.md#materialization), and surrounding [graph](../../glossary.md#graph) effects that isolated route timing cannot predict.
Keeping candidate generation with each architecture [owner](../../glossary.md#term-owner) preserves type safety and decision
ownership, while [opaque](../../glossary.md#term-opaque) orchestration can apply one [objective](../../glossary.md#term-objective) and budget across the complete
[model](../../glossary.md#term-model). Explicit versioned [artifacts](../../glossary.md#term-artifact) make reuse visible and invalidatable without introducing
[runtime](../../glossary.md#term-runtime) search or hidden state.

## Consequences

### Positive

- One user workflow connects the [model](../../glossary.md#term-model) and [target](../../glossary.md#term-target) to explicit [cache](../../glossary.md#term-cache) and plan [artifacts](../../glossary.md#term-artifact).
- Exact compatible local work is measured once and can be reused across models.
- End-to-end plan measurement [captures](../../glossary.md#term-capture-meanings) [fusion](../../glossary.md#term-fusion), [layout](../../glossary.md#layout), [materialization](../../glossary.md#materialization), and [partition](../../glossary.md#partition) context.
- Backends retain typed route ownership while shared orchestration remains generic only at the
  opaque candidate boundary.<span data-glossary-navigation="true"> ([Backends glossary](../../glossary.md#backend); [ownership glossary](../../glossary.md#term-ownership-meanings); [opaque glossary](../../glossary.md#term-opaque))</span>
- Benchmark reports, planning estimates, tuning caches, and runtime observations have distinct
  meanings.<span data-glossary-navigation="true"> ([planning glossary](../../glossary.md#term-planning); [runtime glossary](../../glossary.md#term-runtime))</span>

### Negative and risks

- Candidate generators and compatibility fingerprints must evolve together and be versioned.
- End-to-end plan search must be deliberately bounded by budget and valid candidate generation.
- Two [artifact](../../glossary.md#term-artifact) roles and separate rich evidence require explicit [lifecycle](../../glossary.md#lifecycle) and corruption tests.
- Prepared-executable persistence may not be portable and remains unresolved until backend and
  lifecycle contracts exist.<span data-glossary-navigation="true"> ([persistence glossary](../../glossary.md#term-persistence); [backend glossary](../../glossary.md#backend))</span>

### Migration, testing, and follow-up

The unimplemented Config 0004 broad profile specification remains retired. Config 0004 stays
Draft as a narrow [backend](../../glossary.md#backend)-neutral [planning](../../glossary.md#term-planning) cost-profile contract after its planning consumer is
stable. The former Draft platform-calibration workflow is removed. Draft-only follow-up planning
belongs to the actual [owners](../../glossary.md#term-owner): [compiler](../../glossary.md#term-compiler) and planning candidate production, shared [prepare](../../glossary.md#prepare)'s [opaque](../../glossary.md#term-opaque)
orchestration and [artifact](../../glossary.md#term-artifact) [lifecycle](../../glossary.md#lifecycle), [concrete backend](../../glossary.md#concrete-backend) candidate generators, and `tools/tuning`
[model](../../glossary.md#term-model)-[autotuning](../../glossary.md#autotuning--model-autotuning)/[cache](../../glossary.md#term-cache) orchestration. No task becomes [Ready](../../glossary.md#term-ready) merely to preserve a queue.

Candidate-generator tests must prove that only complete valid typed configurations are exposed.
Cache tests must cover compatibility, invalidation, corruption, and [atomic](../../glossary.md#resource-transaction-and-rollback) replacement. Tuning
tests must cover deduplication, miss-only measurement, bounded plan comparison, and safe [heuristic](../../glossary.md#term-heuristic)
fallback. No [dependency](../../glossary.md#term-dependency) rule changes, so architecture tests do not change for this decision.

## Related documentation

- [Architecture contract](../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../glossary.md#architecture-contract))</span>
- [Performance evidence and model autotuning](../../architecture/performance-evidence-and-tuning.md)<span data-glossary-navigation="true"> ([model autotuning glossary](../../glossary.md#autotuning--model-autotuning))</span>
- [Partition scoring](../../architecture/partition-scoring.md)<span data-glossary-navigation="true"> ([Partition glossary](../../glossary.md#partition))</span>
- [Runtime, prepare, and backend boundary](../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime); [prepare glossary](../../glossary.md#prepare); [backend glossary](../../glossary.md#backend))</span>
- [Benchmarking guide](../../developer-guide/benchmarking.md)<span data-glossary-navigation="true"> ([Benchmarking glossary](../../glossary.md#term-benchmarking))</span>
- [Config master plan](../../planning/modules/config/master-plan.md)
- [Planning master plan](../../planning/modules/planning/master-plan.md)<span data-glossary-navigation="true"> ([Planning glossary](../../glossary.md#term-planning))</span>
