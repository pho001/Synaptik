# Backend execution contract

> This scoped contract is incorporated by the [authoritative architecture root](../../../ARCHITECTURE.md).
> It is normative only within the scope stated below and has no independent authority. Root global
> invariants and dependency rules apply everywhere and take precedence. Any overlap, contradiction,
> missing applicable scope, or ambiguity requires an explicit architecture update; do not choose
> between contracts silently.

## Scope

This contract owns concrete backend responsibilities, performance-evidence and optimization-tool
boundaries, CPU routes and generated-code placement, Metal ownership, and the OpenBLAS provider.
It does not own backend-neutral Planning ownership, shared Prepare orchestration, or Runtime policy.

## Concrete backend modules

Concrete backend modules own concrete backend implementation.

Examples:

```text
backends/cpu
backends/metal
backends/cuda
```

Allowed:

- capability provider
- backend-owned prepare/lowering
- backend-specific fusion
- backend-specific specialization
- kernel route selection
- executable units
- backend storage
- backend workspace
- backend trace contribution
- native bridge integration
- typed, version-controlled, tested candidate generators colocated with the routes they configure
- compatible workload-cache lookup during preparation
- safe heuristic selection when model autotuning is disabled or compatible cache entries are
  absent
- deterministic partition analysis and exact shared buffer/workspace requirement declaration
- finalization of an analyzed partition against shared assigned slots

Forbidden:

- public Tensor API ownership
- global graph compiler logic
- engine dependency
- service locator ownership
- runtime plugin discovery ownership
- changing module ownership rules

Concrete backend modules may depend on:

- model
- config
- planning
- runtime
- prepare
- backend-contract
- trace

Concrete backend modules must not depend on engine.

Each concrete backend owns the complete typed configuration vocabulary and candidate generator
for each route it implements. A generator derives and prunes complete valid configurations from
target capabilities, canonical workload facts, and the tuning budget. CPU matrix-multiplication
candidates may, for example, include supported Vector API species and strategy, unroll, tile,
parallelism, or OpenBLAS thread configurations. Vector, scalar, and OpenBLAS choices are distinct
route-specific typed configurations, not booleans or entries in a generic parameter bag.

Operation family selects the appropriate backend candidate generator; it is not a cache key for
one universal family-wide configuration. Local tuning measurements use a canonical workload
signature that includes semantics and attributes, data types, shapes, layouts, relevant policies,
and target compatibility. Identical signatures can reuse one result across occurrences and
models. Physical vector lanes remain constrained by hardware and supported JDK Vector API
species; candidate generation must not promise arbitrary lane counts.

Backend candidate discovery must not use `Map<String,Object>`, string dispatch, reflective
annotations, a central knob registry, or a generic configuration language. Shared preparation and
tuning orchestration sees candidates opaquely and does not interpret private backend fields.

## Performance evidence and optimization tooling

`tools/benchmarks` owns observational, report-oriented benchmarking. It runs fixed reproducible
operation, operation-family, model, and end-to-end workloads to compare commits, models, or
environments. A benchmark produces measurement evidence and reports only. It must not select or
mutate production settings.

`tools/tuning` owns one explicit model-autotuning workflow with two coordinated phases. First, it
extracts actual tunable workloads and routes from the model, forms canonical workload signatures,
deduplicates identical signatures while retaining occurrence weight and context, reuses compatible
entries from an explicit persistent workload cache, and measures only cache misses. Second, it
measures a bounded set of complete valid graph, fusion, ownership/partition, layout,
materialization, route, and configuration candidates end to end and selects an explicit prepared
plan or artifact. This second phase does not repeat local route-parameter search.

Compiler, planning, prepare, and concrete backends generate the candidates for decisions they
own. Tuning tooling coordinates measurement and selection without taking over graph semantics,
transformations, ownership rules, lowering, route logic, or private backend vocabulary. The model
author supplies the model, representative input or shape profiles, objective, budget, constraints,
and explicit cache locations; backend authors define backend candidates. Running this same
workflow over a representative model corpus may pre-seed the same workload cache for a target,
but there is no separate platform-calibration subsystem, workflow, or profile.

`modules/config` may store immutable declarative inputs to this workflow after consumers are
stable, but it does not own runners, search algorithms, live discovery, caches, or mutable
evidence. Model autotuning is optional for correctness. Cache-only or heuristic preparation must
remain safe when it is not requested.

Future tuning artifacts are explicit persistent files: a reusable workload tuning cache and a
model-specific plan cache or prepared-plan record. A load reuses only compatible hits; a miss may
be tuned and atomically persisted. Entries carry explicit schema and backend candidate-schema
versions, target and workload or model fingerprints, objective and constraints, and a measurement
summary. Implementations invalidate incompatible entries and safely reject corrupt data. They do
not use hidden global caches, Java object serialization, or executable payload assumptions. Rich
measurement evidence remains separate from compact caches. Physical file formats and prepared
executable serialization remain deferred to their backend and lifecycle owners.

Runtime profiling is passive observation of actual execution. `modules/runtime` owns the observed
execution context and `modules/trace` owns typed diagnostic DTOs; neither profiling nor tracing
selects settings.

## CPU backend routes

CPU scalar, CPU Vector API, generated JVM-bytecode CPU computation kernels, and OpenBLAS are
routes inside the CPU backend.

They are not separate backends.

A generated CPU computation kernel is backend-internal executable code whose JVM bytecode is
produced for a selected CPU lowering and specialization. CPU backend analysis owns the lowering,
specialization, fusion, route choice, and exact shared-resource declarations. CPU backend
finalization may generate and define the selected kernel, or reuse it from a CPU-owned compatible
generated-artifact cache, only after shared Prepare assigns slots. Runtime receives the resulting
prepared executable and neither generates, caches, selects, nor specializes kernels.

This contract does not prescribe a bytecode-generation library or a particular JDK builder API.
Changing that implementation mechanism within the CPU backend does not change module ownership,
dependency direction, or lifecycle placement.

Planning chooses:

```text
owner = CPU
```

CPU prepare chooses:

```text
scalar route
Vector API route
generated JVM-bytecode CPU computation-kernel route
OpenBLAS route
specialized kernel
fused kernel
```

Do not create separate backend modules such as:

```text
cpu-scalar
cpu-vector
cpu-blas
```

unless this document is updated first.

## Metal backend

Metal backend owns:

- MPSGraph lowering
- MPSGraph executable creation
- custom Metal kernel routes
- Metal storage
- native bridge integration
- Metal-specific materialization
- Metal trace contributions

Metal-specific optimizer execution belongs to Metal backend prepare/kernels, not to training.

Do not add `MetalOptimizerBridge` to `extensions/training`.

## OpenBLAS provider

`backends/openblas-provider` is a low-level leaf provider.

Allowed:

- OpenBLAS library loading
- symbol binding
- GEMM calls
- thread control

Forbidden:

- config interpretation
- planning
- fallback logic
- prepared execution
- Tensor API
- runtime residency
- backend ownership decisions

The dependency direction is:

```text
backends/cpu -> backends/openblas-provider
```

Never the reverse.

## Other tool scopes

The performance-evidence section above owns the detailed boundaries for `tools/benchmarks` and
`tools/tuning`. The intended `tools/cli` location has no additional technical rule in the migrated
contract. Root global invariants and dependency rules therefore apply; if CLI work encounters an
architecture-sensitive ownership, dependency, or lifecycle question not answered there, the
missing scope requires an explicit architecture update.
