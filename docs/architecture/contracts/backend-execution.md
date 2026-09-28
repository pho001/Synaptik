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

The current common-profile Metal movement domain uses fixed custom kernels for the selected
occurrences. All six carriers admit exact `RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`,
`SQUEEZE`, `CONTIGUOUS`, affine positive-step `SELECT`/`SLICE`, `GATHER`,
`GATHER_ELEMENTS`, `GATHER_ND`, replacement `SCATTER_ELEMENTS`/`SCATTER_ND`, `PAD`,
`SLICE_UPDATE`, `CONCAT`, `STACK`, `TILE`, and `UNFOLD_AXIS` within their static Shape and role
contracts. `ONE_HOT` accepts INT32 or INT64 indices and publishes canonical BOOL; every other
index role also accepts INT32 or INT64. `FOLD_AXIS` admits
FLOAT64/FLOAT32/BFLOAT16/INT64/INT32 and rejects BOOL; `UNFOLD2D`/`FOLD2D` and
`UNFOLD3D`/`FOLD3D` admit only FLOAT64/FLOAT32/BFLOAT16. Every fold is non-overlapping, and
uncovered cells receive exact carrier zero.

Every logical index is validated in stable node then row-major ordinal order. Replacement scatter
supports `NONE` only and completes bounds and global destination-uniqueness checks before any
initial copy, selector dispatch, or target write. Bounds publish the Model's exact
`IndexOutOfBoundsException`; duplicates publish its exact `IllegalArgumentException`; every target
and input remains unchanged on failure. Arithmetic scatter, colliding replacement, overlapping
folds, and reduction-dependent adjoints are not part of the contract.

Selected affine views retain their exact logical Shape, positive strides, and storage offset.
Preparation and native preflight independently derive and authenticate the separate physical
storage descriptor and complete referenced span before resource creation or mutation. Selected
materializing outputs are canonical. Scalar targets are valid values, not empty buffers; zero
extents, zero/negative external strides, unresolved layouts, overlap, and byte overflow fail closed.

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

Metal's common exact occurrence domain under both profiles contains exact unary operations; all 36
ordered casts; FLOAT64/FLOAT32/BFLOAT16 classification; scalar and right-aligned BOOL logic; all
nine promoted floating `WHERE` signatures; exact six-carrier affine/index/replacement/movement
rows; non-overlapping folds/windows over the carrier subsets above; unsigned-32-bit-bounded
ordering/top-K/numeric arg-extrema; exact FLOAT64/FLOAT32/BFLOAT16 maximum Pool2d/Pool3d; and
zero-input raw INT64[2] `INITIAL_STATE`. The nine floating-to-floating casts preserve legal
gradient metadata. Float-to-integral casts may consume a differentiable input but produce a
non-differentiable result. `WHERE` differentiability is the exact branch-role OR and never includes
its condition.

The bounded CPU-free generated-gradient closure includes floating casts, inverse affine movement,
positive-step SELECT/SLICE/SLICE_UPDATE, base-only negative-step SLICE_UPDATE target subsets, PAD,
CONCAT/STACK/TILE, exact replacement scatter using saved indices, and non-overlapping window
adjoints. Saved condition and index roles remain live until their backward consumers finish. The
Compiler's fully-static target-relative crop inference and narrow layout closure preserve exact
scalar or affine crop layouts; 3D window counterpart validation compares semantic type, Shape, and
gradient properties rather than incidental layout. Dynamic/empty geometry, additive scatter,
overlap accumulation, unlisted arithmetic-reduction ownership, higher-order differentiation, and
every production-false semantic family remain fail-closed. Task-0069 L1 and VARIANCE are
deliberately no-gradient and create no new generated-gradient closure.

Both profiles additionally admit no-gradient INT32/INT64 MATMUL pairs with INT64-dominant
promotion and modular result arithmetic. The accelerator-only set adds the documented FLOAT32
tensor/scalar arithmetic, comparisons, extrema, reductions, and scans; every positive-static
FLOAT32 MATMUL vector, matrix, batched, and right-aligned broadcast geometry; no-gradient
BFLOAT16/FLOAT32 or FLOAT32/BFLOAT16 MATMUL with FLOAT32 result; same-type canonical positive-rank
FLOAT32 MSE under `NONE`, `SUM`, or `MEAN`; exact no-gradient rank-one FLOAT32 `L1_NORM` over
ordered axis `[0]`; exact no-gradient rank-one FLOAT32 `SCATTER_ADD` over axis zero and a
materialized INT32/INT64 index feed; exact singleton no-gradient FLOAT32 `VARIANCE` over axis `[0]`
with correction zero; FLOAT32-result grouped Conv2d/Conv3d over FLOAT32/BFLOAT16 roles; FLOAT32
average Pool2d/Pool3d; and canonical FLOAT32 explicit-state dropout. Conv1d/Pool1d may use only
authenticated local singleton-height views. Strict capability
remains a subset because every common occurrence has the same answer under accelerator; strict
rejects every accelerator-only addition. Attention, recurrent execution, convolution transpose,
overlapping folds, and the other recorded blockers remain unsupported.

Every Task-0066 selected occurrence at wires `6..11,16..19,39..45,51,69,71..84` and all accepted
Task-0069 occurrences at wires `70`, `112`, and `114` select one fixed `CUSTOM_PROGRAM`
whole-partition route. No dtype-, Shape-, or payload-dependent MPSGraph
alternative, nested selected-node fallback, retry, timing, or autotuning exists. Unselected capable
operations may still use their retained routes, including direct rank-two FLOAT32 MATMUL. Variance
outside the exact Task-0069 singleton domain retains its prior direct structural recipe rather than
being rejected or diverted. The shared custom route uses fixed reviewed raw-word/integer/movement/
predicate kernels behind one whole-program invocation with declared assigned buffers for every
logical value and no host repair.

Task-0069 L1 at wire `114`, ScatterAdd at wire `70`, and singleton VARIANCE at wire `112` are fixed
`CUSTOM_PROGRAM` under `CA_0069`; Java and native creation reject their direct MPSGraph routes. L1
requires accelerator profile, FLOAT32, one canonical positive-static rank-one no-gradient input,
ordered multi-axis `[0]`, a canonical scalar or retained `[1]` output, unsigned-32-bit bounds, one
output thread, and distinct buffers. It raw-clears every contributor sign and performs exactly
`N-1` source-ordered safe additions; `N=1` performs none.

ScatterAdd requires accelerator profile, canonical FLOAT32 rank-one base/update/output with positive
static extents, axis zero, no gradient flags, and a materialized canonical rank-one INT32/INT64
index partition feed. The compiler places explicit `CONTIGUOUS` between its generated zero-base
`EXPAND` and Scatter. A complete CPU index scan precedes every command encoding and mutation. One
target thread raw-loads the base once, keeps every matching duplicate in source order, and stores
once; unaddressed targets copy the raw base, and there are no atomics or races. This exact node
closes the existing rank-one Gather data cotangent without admitting gradient-bearing ScatterAdd
or higher-order ownership.

VARIANCE requires accelerator profile, canonical FLOAT32 rank-one `[1]` input, axes exactly `[0]`,
correction zero, no gradient flags, distinct buffers, and a canonical scalar or retained `[1]`
output. One thread evaluates exactly `DIV(x,+1)`, `SUB(x,mean)`, `MUL(difference,difference)`, and
`DIV(square,+1)`, then stores the sole canonical cell once. The task-local Lean proof covers every
DAZ/FTZ choice: all finite inputs yield exact positive zero, while NaN and either infinity sign
yield NaN class. There is no classifier, aggregate add, reciprocal, FMA, clamp, tolerance, fallback,
retry, timing, or alternate publication.

The shared Lean proof, source certificate, and pinned Xcode-27 compiled-MSL/AIR audit establish all
three bounded Model result-set memberships.

Direct CPU/Metal transfer supports all six current data types at ranks `0..16` over canonical or
resolved positive-stride non-overlapping storage layouts; BOOL validation visits logical elements
only. Local selected publication gathers logical elements from authenticated physical storage and
preserves exact target-relative offsets and holes without exposing aliases.

The package uses ABI 5 with the same thirteen exports. Node schema 18 is one bounded
self-describing, route-bearing image over type wires `1..6`, operation wires `1..115`, attribute
wires `0..41`, route wires `1..3`, an exact numerical-profile wire, and complete optional
storage-layout geometry. Its fixed 128-byte `SM18` header authenticates the core image and, only
for `CUSTOM_PROGRAM`, the canonical step/member/binding/materialized-value/typed-instruction/manifest
extension. Every core and extension section is adjacent; references are never padded before
64-bit attributes. The manifest carries only typed execution records and integer source byte
counts; Java neither emits Metal source nor owns fixed, generated, or assembled source hashes.
MPSGraph images set every extension count and flag to zero and physically omit the extension.
Custom execution binds compact materialized slots rather than one slot per logical value.

Schema-2 pointwise fusion may replace each maximal linear chain of eligible canonical FLOAT32
`FLOOR`, `CEIL`, `SIGN`, and `RELU` nodes with deterministic generated units of length `2..8`;
a remainder of one is prevented by taking seven from the preceding unit. Inputs and outputs have
positive fully static rank `1..16` and element count `1..UINT32_MAX`. Generation is bounded by 32
units, 256 instructions, 16,384 bytes per function, 262,144 generated bytes, and 1,048,576 total
source bytes. Native reconstructs exact cap precedence and the first rejected node. The production
emitter first traverses the structured sink in no-allocation count mode, validates all five caps,
allocates the exact byte count, and traverses the same emitter once for compiler input. The exact
fixed corpus is 84,541 UTF-8 bytes: native SHA-256-checks each of its ten ordered components and
their total and independently authenticates the assembled source before compiler entry.

The same schema admits ACCELERATOR-only typed anchor-epilogue instructions. An eligible MATMUL
preserves optional literal scalar multiplication, at most one source-ordered ordinary right-aligned
tensor ADD, and optional terminal RELU or no-gradient CLAMP. Conv2d admits at most ADD followed by
the same terminal pair and never scalar multiplication. Intrinsic rank-one `[C]` remains only the
third convolution input; an external rank-one addend broadcasts over `[W]`, while `[1,C,1,1]`
expresses ordinary channel broadcast. Every absorbed intermediate is private, single-consumer,
non-target, canonical, and absent from the materialized set. The native anchor validator
independently reconstructs those facts before one safe-math dispatch and one final store, with no
intermediate store, retry, or fallback.

The audit fixture uses the same production header in singleton-enabled audit mode, repeats generation,
compares count, bytes, and hash; runtime admission remains length `2..8`. Generated pipelines
require `BindingInfo | BufferTypeInfo`, exact input-read/output-
read-write runtime access, and exact size/alignment/type/name/offset reflection for every member of
the 32-byte, eight-aligned `PointMeta` ABI. Each generated unit has one input load, one output
store, one dispatch, and no intermediate materialization. Barriers, unsupported nodes,
noncanonical layouts, gradients, fan-out, targets, and cap rejection preserve fixed custom or
MPSGraph boundaries; Task-0069 VARIANCE remains custom only for its exact singleton domain.

Structural execution covers exactly 101 kinds with 14 remaining nonexecutable; production
capability is exactly 86 kinds with 29 remaining false. Catalog counts are exactly `75/35/5`
MPSGraph and `73/42/0` custom. Workload, exact-policy, candidate, compatibility, route-policy, and
codec identities are version twenty-seven; every other identity fails closed. The complete-plan
wrapper remains version one.
