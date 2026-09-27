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

For Metal route qualification, candidate comparison begins only after identical proven semantics
and domain plus numerical acceptance. A single survivor needs no comparative cost gate. Multiple
survivors may be selected without timing only by strict structural dominance: no more compute
dispatches, no more route-owned temporary bytes, and at least one strict improvement. Supported
observation of actual framework-internal dispatches and temporary resources remains mandatory for
every opaque route, including MPSGraph; graph-node counts, command-buffer counts, inferred fusion,
framework estimates, and process memory are not substitutes. An explicit custom route may instead
attest those facts from an auditable source-bound declaration when retained source owns every
command encoder dispatch and resource used by the synchronous hot invocation.

If multiple nondominated survivors remain, selection stays pending an explicitly authorized,
controlled comparative environment and protocol. Uncontrolled local-device timing is diagnostic
only: it never qualifies a candidate, selects a route, changes tuning identity, populates a
production decision, or breaks a structural tie. Qualification fixes an authorized route before
production preparation and adds no runtime timing, fallback, retry, or workload matrix.

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

The current cross-owner transfer capability is deliberately exact: rank-0..16 fully static
descriptors for all six model data types may move CPU to Metal or Metal to CPU when their resolved
layout has positive extents, positive strides, no overlapping logical positions, and one checked
physical referenced span. Canonical contiguous layouts are the simplest member of that domain;
supported SELECT/SLICE layouts may carry a nonzero storage offset and holes. The transfer copies
the complete physical span without conversion, preserving prefix and gap bytes as well as logical
represented bits.

CPU owns the live native host staging representation and its per-run lifetime; Metal owns the
device-buffer type check and native copy. Both backends validate descriptor, exact byte extent,
context, openness, and current-thread access during cold binding. BOOL descriptor-aware download
stages and validates every logical byte before committing any destination mutation; prefix and gap
bytes are not interpreted. The bound transfer retains direct typed references and performs exactly
one native copy when invoked. Unresolved, empty, zero- or negative-stride, overlapping, span-
inconsistent, or byte-overflowing layouts, conversion, heap staging, and unsupported directions
fail before backend analysis.

Local Metal publication uses the same checked geometry for authenticated storage-layout targets
and also supports canonical rank-zero through rank-sixteen targets for all six carriers. Caller
host ingress preserves exact represented storage bytes; descriptor-aware execution, transfer, and
publication paths validate logical BOOL values without treating storage holes as BOOL elements.

The current common-profile Metal movement domain includes bounded canonical FLOAT32
`UNFOLD_AXIS`; all-carrier Task-0059 CAST/indexing/pad/slice/concat/stack/tile and exact
SELECT/SLICE storage layouts; FLOAT64/FLOAT32/BFLOAT16 UNFOLD2D/UNFOLD3D; and Task-0060
non-overlap FOLD_AXIS/FOLD2D/FOLD3D. UNFOLD_AXIS input rank is 1..15, size is 1..16, step is
positive, size does not exceed the selected extent, and its canonical output has the exact
rank-plus-one floor-count Shape. Window folds admit only strides at least the effective dilated
kernel, so each in-bounds contributor is a raw copy and uncovered output cells receive exact
carrier zero. No overlapping fold route exists.

The current common-profile Metal indexing/replacement domain includes exact GATHER,
GATHER_ELEMENTS, GATHER_ND, ONE_HOT, replacement SCATTER_ELEMENTS/SCATTER_ND, SELECT, SLICE, and
SLICE_UPDATE rows over their documented carrier/index/layout subsets. Every logical index is
validated in stable node then row-major ordinal order. Scatter replacement completes bounds and
global destination-uniqueness checks before any initial copy, selector dispatch, or target write.
Bounds publish the Model's exact `IndexOutOfBoundsException`; duplicates publish its exact
`IllegalArgumentException`; every target and input remains unchanged on failure. Selector skip,
arithmetic scatter, colliding replacement, and overlap-winner behavior are never part of the
contract.

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

Metal's common exact occurrence domain under both profiles contains the exact unary, affine,
canonicalization, indexing, classification, BOOL, Task-0059 movement, Task-0060 replacement/fold/
aggregate, unsigned-32-bit-bounded ordering/top-K/numeric arg-extrema, and exact
FLOAT64/FLOAT32/BFLOAT16 maximum Pool2d/Pool3d rows. Task 0059 includes nineteen proved CAST pairs
and all-carrier SELECT/positive-step SLICE over resolved positive-stride non-overlapping layouts.
Task 0060 includes all-carrier replacement SCATTER_ND/SLICE_UPDATE,
FLOAT64/FLOAT32/BFLOAT16 non-overlap folds, modular INT32/INT64 PROD, and BOOL ALL/ANY. Both
profiles also admit no-gradient INT32/INT64 MATMUL pairs with INT64-dominant promotion and modular
result arithmetic. The accelerator-only set adds the documented FLOAT32 tensor/scalar arithmetic,
comparisons, extrema, reductions, and scans; every positive-static FLOAT32 MATMUL vector, matrix,
batched, and right-aligned broadcast geometry; no-gradient BFLOAT16/FLOAT32 or
FLOAT32/BFLOAT16 MATMUL with FLOAT32 result; same-type canonical positive-rank FLOAT32 MSE under
`NONE`, `SUM`, or `MEAN`; FLOAT32-result grouped Conv2d/Conv3d over FLOAT32/BFLOAT16 roles; and
FLOAT32 average Pool2d/Pool3d. MATMUL operands are canonical or authenticated local
identity-prefix, last-two-axis transposes. Conv1d/Pool1d may use only authenticated local
singleton-height views. Existing rank-two FLOAT32 matrix products retain direct MPSGraph; newly
admitted MATMUL forms and all six convolution/pooling rows use the fixed custom program. The five
ordering/arg-extrema kinds are custom-only and TOP_K retains both ordered outputs through one
step. MSE uses one fixed MPSGraph `SUB -> MUL -> qualified full reduction` composition and grants
no generated backward ownership. Metal generated gradients are limited to primitive-closed
all-FLOAT32 Conv2d roles and non-overlapping FLOAT32 average-pool folds; Conv3d, maximum-pool,
mixed/joint convolution, and overlapping-fold gradients remain fail-closed. Attention and
convolution transpose remain unsupported. Strict capability remains a subset because every common
occurrence has the same answer under accelerator; strict rejects every accelerator-only addition.
Direct CPU/Metal transfer supports
all six current data types at ranks `0..16` over canonical or resolved positive-stride
non-overlapping storage layouts; BOOL validation visits logical elements only.

Accelerator operations must produce only results admitted by Model's total recursive FLOAT32
exact/discrete, primitive, aggregate, and composite-inheritance floors. Every other
profile/operation occurrence fails closed before route selection; transporting profile identity
never authorizes a result outside the Model-owned set. The shared custom-program route is realized
by fixed reviewed safe-math/raw-word/integer/movement/convolution/pooling kernels behind one
whole-program native invocation, with declared assigned buffers for every logical value and no
hidden materialization, host staging, hot compilation, retry, or fallback.

The package uses ABI 5 with the same thirteen exports. Node schema 15 is one bounded
self-describing, route-bearing image over type wires `1..6`, operation wires `1..115`, attribute
wires `0..41`, and complete optional storage-layout geometry. Structural execution covers exactly
99 kinds with 16 remaining nonexecutable; production capability is exactly 81 kinds with 34
remaining false. Workload, exact-policy, candidate, compatibility, route-policy, and codec
identities are version twenty; schema 14 and identity 19 fail closed. The complete-plan wrapper
remains version one.
