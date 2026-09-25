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

For Metal route qualification, supported observation of actual framework-internal dispatches and
temporary resources remains mandatory for every opaque route, including MPSGraph. No graph-node
count, command-buffer count, inferred fusion, framework estimate, or process memory substitutes for
those facts. An explicit custom route may instead attest its compute-dispatch count and route-owned
temporary bytes from an auditable declaration when retained source owns every command encoder
dispatch and every resource used by the measured synchronous hot invocation. The declaration must
enumerate each dispatch and each transient resource, exclude steady prepared inputs and outputs,
and remain tied to the exact measured source identity. Any framework call or unowned encoder or
resource makes that route opaque for this rule. Hot duration is still measured on the named target
device. Qualification fixes one route before production preparation and adds no runtime timing,
fallback, retry, or workload matrix.

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

- the immutable public `MetalBackendConfiguration`, including the required caller-selected
  absolute native library path; any later Metal-private option remains owned here
- `MetalBackendIntegration.open(MetalBackendConfiguration)`, configuration validation and
  snapshotting, native loading, context creation, and partial-construction rollback
- MPSGraph lowering
- MPSGraph executable creation
- custom Metal kernel routes
- Metal storage
- native bridge integration
- Metal-specific materialization
- direct cold binding of exact live Metal buffer representations for prepared host-staged upload
  and download without exposing a native handle or performing backend lookup
- Metal trace contributions

Engine may take ownership of a successfully opened Metal integration, but it must not duplicate
or interpret Metal configuration, discover a library, select a Metal device, or construct native
Metal state itself. Metal must not depend on Engine.

One `MetalBackendIntegration` owns one `MetalDeviceContext` created through the system-default-
device path and one associated command queue. Its configuration selects only the caller-supplied
native library. The availability token `BackendDeviceId(metal, "default")` is the sole abstract
Metal slot, not a stable hardware fingerprint or public selector. Engine registration is keyed by
`BackendId`, so one Engine can own at most one Metal integration. Every Metal buffer, workspace,
route resource, and native handle remains authenticated against its exact context; foreign-context
and foreign-`MTLDevice` use fails closed.

The current Metal invocation boundary is synchronous: native routes commit and wait for completion
before successful return. A completed invocation may publish resources into a `RunResult` whose
ownership outlives that call, so device completion does not imply resource release or authorize a
general output/workspace pool. Concurrent callers do not imply GPU overlap, cross-run ordering,
fairness, queue multiplicity, or throughput. There is no async/cancel/timeout API, physical-device
enumeration or selection, multi-device scheduling, cross-device migration/coherence, failover, or
hot-plug contract. See [ADR 0020](../../design/decisions/0020-synchronous-single-default-device-metal-execution.md).

The current cross-owner transfer capability is deliberately exact: positive rank-1..16 fully
static canonical contiguous `FLOAT32` buffers with checked element and byte geometry may move CPU
to Metal by uploading from a live CPU native representation, or Metal to CPU by downloading into
one. CPU owns the native host staging representation and its per-run lifetime; Metal owns the
device-buffer type check and native copy. Both backends validate descriptor, byte extent, context,
openness, and current-thread access during cold binding. The bound transfer retains direct typed
references and performs exactly one native copy when invoked. Other ranks, zero extents, data
types, layouts, directions, conversion, canonical-byte materialization, and heap staging are
unsupported and must fail before backend analysis.

A locally produced canonical rank-zero `FLOAT32` reduction target may materialize through the
owning Metal integration as exactly four detached canonical bytes. It remains ineligible as a
caller feed or cross-owner transfer value and may not widen the transfer predicate above.

Canonical caller-host ingress into an owned Metal partition accepts exact positive-rank
`FLOAT32` and `INT32` values with no conversion. A locally produced canonical `BOOL` target may
materialize through the owning Metal integration as exact one-byte elements. These local paths do
not widen the cross-owner transfer capability, which remains canonical positive-rank `FLOAT32`.

The current common-profile Metal movement domain also includes bounded canonical FLOAT32
`UNFOLD_AXIS`. Its input rank is 1..15, size is 1..16, step is positive, size does not exceed the
selected extent, and its canonical output has the exact rank-plus-one floor-count Shape. It
materializes addressed input representations in row-major output order, repeats overlaps, omits
incomplete tails, applies no padding, and never aliases or mutates its source. This is forward-only:
Metal admits no FOLD_AXIS, two- or three-dimensional window, or backward window route.

The current common-profile Metal indexing domain is exact positive-rank axis `GATHER` from
canonical `FLOAT32` data with canonical `INT32` indices, positive-rank `INT32`-to-`BOOL`
`ONE_HOT`, and canonical positive-rank `FLOAT32`/`INT32`/`FLOAT32`
`SCATTER_ELEMENTS/NONE` replacement. Scatter has ordered `[data, indices, updates]` inputs, a
data-shaped canonical output, equal index/update Shape, and exact non-axis agreement with data.
Metal validates every logical index in stable node then row-major ordinal order; each scatter
completes bounds before complete-coordinate uniqueness. All checks precede MPSGraph tensor-data
construction, selector dispatch, and target writes. Bounds publish the Model's exact
`IndexOutOfBoundsException`; duplicates publish its exact `IllegalArgumentException`; every target
and input remains unchanged on failure. Selector skip and overlap-winner behavior are never part of
the contract. These are forward backend routes and do not add Compiler production, a complete
backward path, arithmetic scatter, Scatter-ND, INT64 indices, general BOOL consumers, CPU fallback,
or widened transfer.

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

## Numerical-profile backend identity

A concrete backend must qualify capability, preparation plans, specialization, generated
artifacts, tuning candidates and decisions, and cache compatibility by the exact selected
`NumericalProfile`. For the same occurrence domain, strict capability is an accelerator subset and
strict behavior is a subset of accelerator behavior. CPU currently supports both profiles through
one identical exact matrix and unchanged routes.

Metal's common exact occurrence domain under both profiles contains only canonical FLOAT32
`NEG`/`ABS`; `RESHAPE`/`EXPAND`/`PERMUTE`/`EXPAND_DIMS`/`SQUEEZE`; `CONTIGUOUS`; bounded canonical
FLOAT32 `UNFOLD_AXIS`; canonical positive-rank FLOAT32 data `GATHER` with canonical INT32 indices;
positive-rank INT32-to-BOOL `ONE_HOT`; and canonical positive-rank
FLOAT32/INT32/FLOAT32 `SCATTER_ELEMENTS/NONE`. Its accelerator-only set contains only tensor
FLOAT32 `ADD`/`SUB`/`MUL`/`DIV`; canonical FLOAT32 `SUM`, `MEAN`, and binding-resolved
`SUM_TO_SHAPE` over full, normalized single-axis, ordered normalized multi-axis including empty,
and exact keep-dimensions forms; and positive static rank-two FLOAT32 MATMUL with exact contraction
geometry and canonical or authenticated local rank-two-transpose operands. Strict capability is a
subset because every common occurrence has the same answer under accelerator; strict rejects every
accelerator-only addition.

Accelerator operations must produce only results admitted by Model's total recursive `FLOAT32`
exact/discrete, primitive, aggregate, and composite-inheritance floors. The inclusive
ordered-binary32 distance-at-most-five ceiling applies only per irreducible elementary-function
site, not as a final-output comparison rule. Every other profile/operation occurrence fails closed
before route selection; transporting profile identity never authorizes a result outside the
Model-owned set. No Blocked Metal operation has a capability
or schema row.
The package remains ABI v4 with thirteen exports, node schema 11,
operation wires `1..19`, attribute wires `0..6`, and version-twelve workload/policy/candidate/
compatibility/route-policy/codec identities. Schema 12, wire 20, attribute 7, INT64, ABI/export
changes, and version-thirteen identities remain conditional, unimplemented, and unreserved.
