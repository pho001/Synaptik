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
modules/backend/cpu
modules/backend/metal
modules/backend/cuda
```

These physical directories retain the separate `:backends:*` Gradle project identities through
explicit `projectDir` mappings, including `:backends` at `modules/backend`. The physical parent is
not the backend-neutral `modules/backend-contract` module and grants no new dependency edge.

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

For accepted general scan, provider support must qualify the complete body and its loop binder,
not infer support from child operations alone. One backend owns the whole scan; global AD remains
Compiler-owned. CPU0030 implements bounded FLOAT32 pointwise execution; CPU0031 extends it with
qualified FLOAT32 broadcast, SUM/sumToShape and bounded shape/layout closure, including actual
independently qualified reverse bodies. CPU0032 adds static homogeneous FLOAT32 rank-two MATMUL
and independently qualified actual reverse bodies, as described under
[CPU backend routes](#cpu-backend-routes). CPU0033 implements homogeneous FLOAT64 across the
complete bounded declared algebra and actual first-order reverse closure under the same
whole-body/binder qualification. The
[scan contract](recurrent-scan.md#general-differentiable-scan) solely owns scan semantics and gates.

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

For Metal route qualification, candidate comparison begins only after candidates cover the same
complete occurrence domain and qualify against Model semantics. Exact structure, stored-value,
discrete, cast, raw-movement, special-value, and domain cases require separate conformance;
ordinary finite arithmetic requires explicit test-only backend/route/family/dtype/size oracles,
metrics, thresholds, and cancellation boundaries. NaN/infinity, signed zero, subnormal/underflow,
empty/domain, and fail-before-mutation checks are not finite-tolerance cases. Qualification does
not infer provider support or license Compiler rewrites. A single survivor needs no comparative
cost gate. Multiple
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

## Low-precision capability evidence

Model `FLOAT16` values/storage/semantics and Metal's qualified low execution remain active.
CPU low arithmetic is subject to the independently reviewed CPU0078 lock below; retained exact/
raw/conversion/storage roles stay active. Canonical ledgers use actual `CpuCapabilityProvider` and
`MetalCapabilityProvider` queries over a stable representative basis. Each row records backend,
occurrence identity, kind and attributes, ordered input/output descriptors, data type,
Shape, layout, gradient metadata, arity, and the boolean provider answer. Checked-in supported and
explicit unsupported rows are both normative drift evidence for that basis.

Target and exclusion columns are architecture mapping decisions, not provider output. The ledger
does not contain route, runtime, device, certificate, or generated-backward ownership, and it does
not change `BackendCapabilityProvider` from a deterministic boolean predicate into an enumeration
API. The representative basis does not replace provider tests over their complete predicate
domains. Canonical manifests use explicit UTF-8 TSV fields, LF endings, stable sorting, and unique
keys; a duplicate key is rejected after exact row-byte comparison, and validation reports the
first missing, added, or changed row.

The 508-row v1 ledger freezes the pre-cutover `ACCELERATOR` and `STRICT_IEEE` provider answers as
historical evidence; its `ACCELERATOR` true and false answers established the profile-free cutover
baseline and must not be rewritten by later capability work. The v2 ledger records actual current
profile-free answers on its representative basis, including independent homogeneous BFLOAT16 and
FLOAT16 queries and explicit negative predicates. It was initially projected from the cutover
`ACCELERATOR` answers, but is not permanently constrained to equal that projection. Reviewed
capability changes may update v2 and the provider while leaving v1 intact. Both ledgers detect drift in
their respective snapshots; neither enumerates the full provider domain or certifies numerics.
Every supported frozen `ACCELERATOR` FLOAT32 Metal occurrence retains its independently queried
homogeneous low counterparts where declared. Existing supported Metal low answers remain true;
CPU's temporary arithmetic lock is the explicit exception to retained CPU availability. Exact
movement, Shape/layout, indexing, selection,
ordering, state, and cast occurrences remain exact substrate. Direct BFLOAT16/FLOAT16 mixed-low
operations are unsupported; an explicit
cast to FLOAT32 is the sole mixed-low boundary. Forward capability remains distinct from
generated-backward ownership.

Every active Metal low family uses the shared typed custom implementation with FLOAT32
working/accumulator values and Model-owned exact discrete/raw/cast/publication rules. Original
per-role FLOAT32, BFLOAT16 and FLOAT16 storage and POD types remain explicit; the common
`synaptik_numeric_codec.h` decodes inputs and narrows each observable low output once with
round-to-nearest, ties-to-even. The Model's named arithmetic-site DAZ/FTZ table and family
exact-zero sign and NaN class/domain rules govern it. Fixed-cell working/publication boundaries
remain distinct from an explicit GeneralScan ordinary-cell composition; sharing arithmetic
owners does not make those rounding schedules interchangeable. Operation-local reassociation/FMA
is permitted only within the qualified implementation; low partitions cannot enter generated-
pointwise or anchor-epilogue fusion. Public, saved, fan-out, predicate, index, mask, selection,
and state boundaries remain barriers.

An opaque or transformed route is eligible only after the same occurrence-scoped exact/special
and finite test qualification as an explicit route. A route name, examples, or generic `allclose`
alone are insufficient; no runtime proof evaluator or numerical certificate replaces those tests.

Every Metal partition containing BFLOAT16 or FLOAT16 values selects `CUSTOM_PROGRAM` and exposes
only that candidate. This applies equally to arithmetic and to raw-preserving RESHAPE, simple
PERMUTE, materializing CONTIGUOUS, SLICE, CONCAT, and TILE. Every operation consuming or producing
a BFLOAT16 or FLOAT16 value uses a custom kernel step; no such operation enters MPSGraph, and no
hidden BFLOAT16-to-FLOAT16 substitution is permitted. Any FLOAT32-only operation in the partition
may use an internal MPSGraph boundary step under existing FLOAT32 policy, whether on an independent
FLOAT32 branch or after an explicit low-to-FLOAT32 cast. This does not change the enclosing
`CUSTOM_PROGRAM` route or candidate set. Generated-pointwise and anchor-epilogue fusion are
unavailable throughout the low-containing partition, including its
FLOAT32-only nodes. The partition policy is independent of device, OS, SDK, compiler, or
loaded-binary identity. MPSGraph is not a low-valued operation or whole-partition route
alternative; classic MPS, MPP, CPU, retry, and fallback are not alternatives either.

Low-precision PREPARE trace metadata names the selected custom route and ordered feed-then-target
logical dtype tuple. It has no accumulator or working-type field; the
Model's FLOAT32 arithmetic working/accumulator guarantee is separate from diagnostic metadata.
The DTO contains no candidate, certificate, accuracy, determinism, or environment state and rejects
a graph selected route.

Metal owns no executable, compilation, or prepared-resource cache: finalization creates fresh
native resources and its outcome reports `NOT_QUERIED`. The only reusable Metal route artifacts are
the tuning compatibility, candidate, and decision values. They are explicitly session-scoped, and
the tools cache never persists or reads a `SESSION` value. Their canonical identity contains the
compatibility/candidate/route-policy/workload/codec versions, native ABI,
per-context random session nonce, versioned program and plan semantics, every descriptor's explicit
dtype wire and Shape/layout/gradient facts, ordered edges, raw constants, and the selected candidate
route wire. Complete-plan identity nests the exact phase-one decision. Dtype, ABI, schema, route, or
semantic changes therefore change the relevant bytes. Every reopened context receives a different
session nonce, so no decision crosses a context boundary.

CPU generated-artifact cache identity was at generator schema 68 before cutover. Both BFLOAT16
and FLOAT16 use the same Java `short` carrier where applicable, but the canonical lowering IR,
ordered `boundaryDataTypes`, carrier pattern, execution strategy, generated-class
identity schema, and code-shaping facts participate in the specialization identity. Consequently
the two raw-16 types have different compatibility bytes and structural keys despite an identical
method descriptor. Persisted envelopes additionally require the current generator schema, exact
structural key, exact compatibility bytes, class-shape validation, and checksum before reuse.

The pre-cutover allocation appended `DataType.FLOAT16` at ordinal 6 after `BOOL`: Metal type wire
7, program schema 19, coordinated Metal candidate/compatibility/route-policy/workload/codec
identity 29, CPU generator schema 68, operation wires 1..115, attribute wires 0..41, route wires
1..3, and native ABI 7. The implementation cutover must append versions for every changed
serialized program, generated artifact, route/candidate/compatibility/workload/codec or tuning
identity, reject stale inputs and cache hits, and retain unaffected wire allocations. Remove the
Metal profile wire atomically in Java and native; review a native ABI bump if an export signature
changes, otherwise retain all thirteen names/signatures. Preserve source/manifest/package
authentication, decoder/preflight, bounds, ownership, compiled-code checks, and fail-before-
mutation rules. No version number is assigned by this documentation stage.

## CPU backend routes

The implemented [CPU0074 exception](#cpu0074-selected-row-region-execution-exception)
qualifies only the selected exec-v1 cases of the scalar-only/no-per-step-dispatch restrictions
below. All other restrictions and implemented-capability statements remain unchanged.

CPU scalar, CPU Vector API, generated JVM-bytecode CPU computation kernels, OpenBLAS and
explicitly configured oneDNN primitive/Graph execution are routes inside the CPU backend.

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
oneDNN primitive or Graph route
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

### Singleton MATMUL correctness cutover

The existing Model contract requires K=0 positive zero and K=1 exactly one typed product,
without an additional identity ADD or FMA. Portable ordinary, ordinary-batch, retained and
ordinary-unit Scan, and selected scalar/vector row regions preserve that singleton result.
K>1 keeps the established ordered producing sites; this correction does not peel a first term
from multi-term regional contractions or change separate bias, carry, partial-merge or SUM ADDs.
Activity gates still dominate both operand loads; inactive singleton contractions retain +0.

The shared scalar/vector contributor seam takes the generation-only `directProduct` fact.
Fixed feature contractions derive it from their complete logical K. Runtime row/cursor owners
select singleton versus ordinary traversal once, outside contributor loops, using the full
logical extent rather than active-row count or worker-range length. Each alternative rebuilds
its generation-local load-cache/last-use state; no callback is reused with stale local definitions.

Both generated compatibility streams append `|matmul-single-product-v1` only for MATMUL-bearing
standalone, retained-Scan or regional bodies. Nested ordinary-unit Scan helpers propagate their
changed specialization keys. This narrowly supersedes earlier literal-retention promises for
those bodies; unaffected pointwise/SUM controls, global envelope 71, family projections, canonical
IR and numeric/helper ABIs remain unchanged. Historical payloads are retired, not repinned.

Production OpenBLAS MATMUL qualification is K>1 for both FLOAT32 and FLOAT64. K=0/K=1 are
ineligible before native candidate/cost/decision selection and retain portable execution.
This is cold shape-based admission, not value inspection, a retry or runtime fallback; raw
provider GEMM bindings and qualified K>1 behavior remain unchanged.


### Qualified oneDNN CPU realization

The current oneDNN route requires explicit loading of an exact external 3.14.0 little-endian LP64
library with sequential CPU runtime (SEQ, code 1). Required loading failures throw; no optional
provider discovery or Runtime fallback may mask them. CPU backend identity, availability and
Model semantics are unchanged. Production qualification is only bare homogeneous FLOAT32
MATMUL with K > 1 over static positive dense/permuted-dense and normalized broadcast-batch
layouts of rank at most 12. Empty/K=0/K=1, gapped or symbolic layouts, Scan bodies, row regions,
convolution, bias, activation and post-operation native forms are not qualified. Wider diagnostic
C bindings do not imply production eligibility. Existing ordinary CPU units preserve convolution
signed zeros, activation special values and bias ordering/rounding.

Both native APIs consume common CPU IR and canonical external buffers; saved/public stores
remain ordinary memory obligations. Graph is not AD or a provider-owned replacement graph
pipeline. API preference does not force native selection, and node count does not force Graph.
CPU analysis compares complete portable/OpenBLAS and explicit ordinary-contraction alternatives,
including longer partitions and constant writes, with conservative dimensionless heuristics
charging arithmetic/remainder, dispatch/control, preparation, memory, ingress and scratch.
These are not measured time or a speedup guarantee. Exact OpenBLAS Phase-1 decisions remain
frozen, selected Phase-2 topology is not replaced, and Phase-2 schema 4 binds oneDNN policy and
load-session identity while rejecting stale decisions.

Analysis retains descriptor facts, not native handles. Primitive metadata queries create no
primitive; Graph's scratch query requires temporary compilation and closure. Finalization
recompiles the exact recipe, verifies retained metadata and returns ordered persistent kernels
as `PreparedResource` objects with transactional rollback. Native caches remain library-owned.
Data spans, padding/alignment, ingress copies and user scratch are declared before shared
assignment. Each run workspace factory creates streams, memory/tensor handles and ABI tables
before binding; the issued workspace retains the kernel and library until close, independently
of prepared-owner closure. Runtime's prepared-resource lease contract remains unchanged.

One RunState owns each shared ingress workspace. Copies share only matching logical value,
origin and referenced span; the first native consumer refreshes that span every run and later
consumers reuse it within the run. No offset prefix or across-run weight snapshot is copied.
Graph inputs are VARIABLE. Cold binding applies direct origins once, copied origins zero, and
rejects output/input/scratch aliasing. Fixed copy/execute/wait runs synchronously under the
shared CPU budget with demand one; there is no hot allocation, selection, retry or fallback.
Existing CNN AD may realize eligible rank-four MATMULs natively while convolution, fold/unfold
and reductions keep ordinary implementations. Metal and CPU low-arithmetic admission are unchanged.

### General Scan CPU realization

The accepted general-scan CPU path is not the existing fixed recurrent Java route. Analysis
prepares one body and declares exact checked staging, carry, temporary, output and demanded-history
resources before shared slot assignment. Finalization, only after assigned slots, emits the outer
time loop and admitted typed body emitters. The complete semantic protocol remains in
[the scan owner](recurrent-scan.md#general-differentiable-scan). CPU0030 implements this path for
bounded same-shape FLOAT32 pointwise bodies and their actual independently qualified
Compiler-generated reverse siblings. CPU0031 adds homogeneous FLOAT32 broadcast, SUM/sumToShape
and bounded shape/layout closure under the same independent whole-occurrence qualification.
CPU0032 adds static homogeneous FLOAT32 rank-two MATMUL within the same emitted time loop,
including independently qualified actual Compiler-generated reverse bodies. CPU0033 implements
homogeneous FLOAT64 across the complete bounded declared algebra and actual first-order reverse
closure, with double producing/accumulator sites and eight-byte resources. CPU0037's implemented
actual-read exception is defined only by the
[scan owner](recurrent-scan.md#traversal-empty-domains-and-declared-qualification);
qualification and main's final capability acceptance are recorded in CPU0037 (2026-10-24).
No low-type, wider-layout, vector, parallel or OpenBLAS scan capability follows;
a forward provider answer alone is not a
backward capability claim.

Conv3d training uses the existing generated movement/fold families, not a backward executor.
`UNFOLD3D` copies FLOAT32/FLOAT64/BFLOAT16/FLOAT16 represented bits from NCDHW into canonical
`[N, C*kD*kH*kW, DOut*HOut*WOut]` columns, with exact configured padding or direct positive zero.
`FOLD3D` admits arithmetic only in FLOAT32/FLOAT64: each output cell starts at positive zero,
inverse-gathers in ascending kernel-coordinate order, includes every in-bounds contributor once
and stores once. Checked cold spatial geometry is shared by extraction/folding; generated loops
use invocation-local primitive facts, not per-element geometry objects or runtime dispatch.
Static public admission and independently proved body `lowerBatch` paths remain distinct;
symbolic Compiler formulas do not widen ordinary public named-B Conv3d admission. Existing
carrier/access/lifetime/alias proofs and legacy family ABIs, keys and generated bytes remain.
Raw low extraction is not low fold/convolution permission: CPU0078's arithmetic lock is unchanged.
The implemented Pool3d completion admits matching input/output gradient flags for FLOAT32/
FLOAT64 MaxPool3d and AveragePool3d, reusing the existing forward kernels, shared Compiler
`PoolingGradientRules`, and `UNFOLD3D`/`FOLD3D` closure. Maximum cotangents retain exact
first-eligible-winner selection; average cotangents retain the fixed kernel-element divisor.
Qualified `lowerBatch` body composition covers both all-valid and runtime-length Scan in both
traversal directions, subject to complete generated-topology qualification and the existing
homogeneous FLOAT32/FLOAT64, static-window and terminal restrictions. This does not widen ordinary
public named-B admission. Legacy low forward domains and CPU0078's low-arithmetic lock remain
unchanged. No standalone backward kernel, runtime fallback, or 3D-specific composition blacklist
is introduced.

Existing CpuLoopEmitter generation callbacks and CpuScalarEmitter typed locals are generation-time
composition seams, not runtime per-step callbacks. CpuKernelIr's pointwise instruction list,
CpuMatmulEmitter's entry-slot/geometry assumptions and CpuCarrierEmitter's fixed entry ABI/local
block are not plug-and-play step bodies. Implementation must explicitly bind invocation versus
step setup, typed carriers/locals, bases, ranges and geometry. Hoist invariant setup outside time
and prefer direct time-dependent offsets/addresses into sequences, emissions and live residuals.
Reuse scalar/vector/carrier arithmetic only where qualified; no vector/parallel capability follows.

Select/materialize only when necessary and justify staging copies, carry banks and body temporaries.
Emission cannot remove live saved histories or all MATMUL storage. Dead residuals, including reverse
residuals, require no history bytes/stores or residual fusion restriction. One RunState owns all
mutable memory, changing addresses, carry and geometry arrays; shared prepared state is immutable.
No runtime graph interpreter, per-operation lookup, per-step dispatch/callback/allocation or T-fold
graph/code expansion is allowed. Technically necessary direct static helpers require an explicit
clean Java oracle rationale; this is not a mandate for one giant method or hidden fallback overhead.

CpuKernelIr canonical identity still excludes extents, assigned slots and run bindings. Separate
baked from runtime scan facts, and key every baked body/role/demand fact in the appropriate CPU
generated compatibility identity. Reject stale format or meaning artifacts. The original scan
decision selected no schema number or CPU type; CPU0030 now uses generator envelope 71 and typed
general-scan family projection 71. CPU0031 preserves both identities and the old pointwise keys/bytes:
its additive shaped-v1 canonical projection is geometry-free, while both existing specialization
identity byte streams conditionally include all baked shaped geometry and resource facts.
CPU0032 preserves schema/family 71 and both old pointwise and shaped keys/bytes. Its additive
MATMUL-v1 canonical projection remains geometry-free; the separate `scan-matmul-v1` payload binds
ordered operands and all baked affine/resource facts in both specialization identity streams.
CPU0033 adds typed FLOAT64 variants without changing schema/family 71, either staged projection or
its versioned payload. Existing arithmetic/boundary dtype and eight-byte baked resource facts
distinguish those variants in both streams; accepted FLOAT32 keys and generated bytes are retained.
CPU0034 subsequently advanced the sole baked shaped/MATMUL payload discriminants to
`scan-shaped-v2` and `scan-matmul-v2` for both dtypes. Typed coordinate induction and proved
post-step emission-carry transitions change staged bytecode, so both specialization streams,
binary names and cache keys advanced and rejected staged v1 artifacts. CPU0035 advanced those
same sole baked payloads to `scan-shaped-v3` and `scan-matmul-v3` in both streams for proved
redundant factor, address and local-copy cleanup. CPU0036 advances the same payloads to
`scan-shaped-v4` and `scan-matmul-v4` for top-tested staged loop control and dominating final-copy
bases. Staged v1/v2/v3 artifacts are incompatible.
Global schema/family 71, geometry-free canonical v1 projections and pointwise/fixed/unrelated
identities remain unchanged;
the CPU0033 retention statement above describes that earlier acceptance, not current staged bytes.
Future CPU briefs must name every added type without speculative
general control-flow infrastructure. Generic no-copy, JIT, speed or CPU0029 timing parity is unproved.
The [scan qualification gates](recurrent-scan.md#qualification-performance-and-migration-gates)
require a complete F32 forward/automatic-reverse milestone, then declared algebra/closure and F64
completion, clean Java algorithm/hot-loop/dataflow comparison, Class-File inspection and short warmed
matched-work evidence with cold/entry/kernel costs separated. No measurement is part of this cutover.

CPU0042's fixed-geometry length-aware execution reuses this generated path, not the fixed executor;
its complete semantics belong to the [scan owner](recurrent-scan.md#length-aware-whole-batch-scan).
CPU independently qualifies FLOAT32/FLOAT64 forward and actual Compiler reverse, mixed INT64/floating boundary
binding, exact predeclared resources, activity-dominated lowering/emission and changed length-mode
identities. Old all-valid/fixed keys and generated bytes stay unchanged and must not depend on
changed ScanAttrs diagnostic text. The length-only branch precedes legacy analysis and adds canonical
`:length-aware-v1` and baked `|scan-length-aware-v1=` in both specialization streams, retaining
schema/family 71, static staged v4, symbolic v3 and unrelated keys/bytes. MAIN accepted CPU0042
on 2026-10-26 after independent review and recorded gates; its
[Result](../../planning/backends/cpu/tasks/0042-length-aware-general-scan-execution.md#result)
owns the evidence and nonstationary timing limits. Prepare0015 implements symbolic-B plus lengths;
its root validation and the sole bounded probe completed (evidence and limits:
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) );
final MAIN acceptance remains pending.
That CPU checkpoint did not qualify low/vector/parallel/OpenBLAS Scan or the independent Metal domain.

The implemented combined scope is fixed T/features, one named B, homogeneous FLOAT32/FLOAT64
generated scalar, single-range forward and actual Compiler-generated reverse. Existing Scan
lowering/IR/preparer/emitter/executable owners carry combined numeric recipes, explicit B and
separate INT64 control; no new type/API/protocol or all-valid symbolic FLOAT64 capability is selected.
The bounded ordinary outer/control/ROW(1) prerequisite below is implemented in the same task.
The [scan owner](recurrent-scan.md#runtime-preflight-and-implementation-gates) retains all semantics,
including original-source AD restrictions, generated mixed roles, saved sites, activity/preflight
and resource algorithms. Owning executable validation and independent code/structural review passed;
Root validation and the sole bounded probe completed;
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)
records evidence and limits. Final MAIN acceptance remains pending.
Use separate canonical `:symbolic-length-v1` and current baked `|scan-symbolic-length-v9=` identities
in both specialization streams, binding dtype, roles, terminal dependencies, demand, control position,
numeric pack and baked recipe facts. Actual B/L, resolved bytes, slots and addresses stay excluded.
CPU0044 advanced only this combined baked payload to v4 for cached-local self-move removal, copy-local
actual splat initializer caching, unused layout/dense-geometry setup removal and coefficient-proved
original-time product sharing. CPU0045 advanced it to v5 for stable own-final-port carries without
a bank: raw initial copying precedes nominal-stride binding, each derived view retains its own
recipe, and identical read/write relative offsets share one cell-scoped local with proved zero
bases omitted. Changing/bank-backed and noncombined bindings retain their selection protocol.
CPU0058 advanced only the combined baked payload to v6: a nonbanked scratch cut gains the existing
zero-base load/store proof only when both coefficients of its complete
`offset + offsetPerBatch * B` recipe are zero. Its entry scratch carrier/base stays valid through
every use in the inclusive cut lifetime; aliases consult the exact origin's proof while keeping
their own relative strides. Port/time-backed and carry-bank cuts retain their original bases,
and relative-offset caching remains cell-scoped. No allocation, lifetime or bank choice changes.
CPU0060 advanced only that payload to v7 for flat scalar initialization, the existing initialized
EMPTY publication path at zero maximum length, coordinate-nested pointwise affine prefixes,
proved direct addresses for active ROW(1) @ ROW(0) contributors and raw-singleton SUM-to-Shape
without a contributor loop. At that version, other matrix paths retained their cursor algorithm.
CPU0064 advances only that payload to v9, skipping withdrawn v8. Stable carriers and temporal
bases bind directly, and affine byte facts derive from admitted explicit B. Combined feature
contractions with proved constant effective K strides also use indexed addresses; the existing
contiguous-feature ROW(1) @ ROW(0) proof retains activity-gated indexed contributors.
Bank-free stage scheduling preserves producing dependencies, physical output read/write hazards
and inclusive assigned-scratch lifetimes; failed proofs retain the declared order. Bank-free
active-time skipping precedes temporal setup, while bank-backed transitions retain raw inactive
copies. Unproved contractions retain their cursor path. Represented products/additions, separate
per-time carry ADD sites, singleton raw movement, padding and saved histories remain unchanged.
Both streams reject combined baked v1 through v8 artifacts. Canonical v1,
the `typed-explicit-b-v1` numeric pack, cuts and scratch remain unchanged.
Retain schema/family 71, static staged v4, all-valid symbolic canonical v1/baked v3, static-length v1
and all other noncombined keys and generated bytes exactly.

The accepted [symbolic Scan domain](recurrent-scan.md#accepted-symbolic-batch-scan)
is implemented by Prepare0014, with final qualification tracked in its
[task](../../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md). Execution is homogeneous FLOAT32, generated
scalar and single-range, fixed T/features/state widths with invocation B; Model source dtype
validity and Compiler's bounded FLOAT32/FLOAT64 AD are not CPU answers. Qualify the complete
forward and actual Compiler-generated reverse occurrences independently, including bounded
rank-three outer closure, not merely supported child kinds or a handwritten reverse body.

Prepare0014 extends existing Scan analysis/IR/preparer/emitter owners and explicit provider dispatch;
symbolic Scan no longer falls through ordinary-batch-only analysis. Analysis fixes the
complete body/resource topology and immutable numeric access/frame/cut/carry-offset recipes before
slots; finalization emits after slots. Use existing `CpuAccessPlan.BatchBinding`, MATMUL/aggregate
numeric binding, `PreparedMemoryPlan.ByteSize` and numeric RunMemoryGeometry handoff. Recipes may
need checked piecewise alignment/empty scratch, not only affine byte sizes. Do not interpret a
sample B, relower on bind, allocate maximum banks or choose per-B routes/slots/topology.
Structural zero-byte workspaces keep their fixed requirement/creator. Prove any cut lifetime and
size dominance for every admitted B; retain disjoint regions otherwise. Complete resource and
empty/saved-site semantics remain solely with the linked Scan owner.

State-aware representation hooks must check actual bytes plus physical type/access/alignment/
overlap. Binding copies numeric facts into fresh run-owned mutable packed/access/carry/address
arrays; no shared currentB or ThreadLocal. The accepted future
[Engine memo](runtime-prepare-engine.md#accepted-engine-geometry-memo) does not authenticate CPU
storage or expose a memo to generated code. Load B/strides/frame offsets once per invocation or
necessary row setup, then use direct time/row/feature loops, cursor increments and typed MATMUL
contributors matching optimal specialized clean Java. No per-element recipe/memo lookup, boxing,
reflection, allocation or dispatch, hidden generic fallback overhead or changed arithmetic is allowed.

Use a distinct symbolic-Scan compatibility branch in both existing specialization streams; bind
all baked dtype/body/role/demand/mapping/recipe facts. Actual B, resolved bytes, assigned slots and
addresses stay excluded. Preserve static Scan/fixed/ordinary keys and generated bytes; diagnostic
text is not identity. Prepare0014 retains schema/family 71 and static canonical v1,
`scan-shaped-v4`/`scan-matmul-v4` identities and bytes. Symbolic mode alone adds canonical
`:symbolic-batch-v1`, which remains unchanged. CPU0039 advanced only the baked symbolic payload
to `|scan-symbolic-batch-v2=` in both existing specialization streams for coefficient-proved fixed
geometry, int-bounded symbolic contractions and generation-only cell-scoped affine offset sharing.
CPU0041 advances only that payload to the current `|scan-symbolic-batch-v3=` for direct immutable
entry-carrier bindings on produced cuts backed by demanded ports. Checked foreign-memory access and
independent formal-carry selection remain unchanged. Symbolic baked v1/v2 artifacts are
incompatible; schema/family 71 and static Scan/fixed/ordinary
keys and generated bytes remain unchanged. Scan's ordinary-batch flag remains false.
Existing View/Cut/Stages facts carry the numeric recipes,
with no new production type or Runtime handoff in that delivered vertical. The later
[CPU0073 accepted mechanism](#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion)
permits only its exact named types and selective identities. Any other new named type or
changed Runtime handoff requires the explicit approval stop gate in the Scan owner.

Affected-module/conformance/integration tests must cover the Scan owner's semantic/empty/
resource/admission matrix and independent actual reverse qualification. Generated Class-File
inspection must compare clean-Java algorithm/hot-loop/dataflow and hidden overhead; bounded warmed
matched-work generated/direct evidence separates setup, admission and kernel costs. Retained static
keys/bytes and read exceptions need regression evidence; symbolic length skipping remains separate.
Prepare0014 records source-matched focused evidence and bounded probe results/limits; final root
qualification remains tracked there. No static/ordinary timing guarantee is inherited.

### CPU0071 approved static MATMUL compatibility cutover

The user approved this append-only cutover on 2026-10-29; CPU0071 implements it in the existing
CPU owners. Independent source/Javadoc review approved the corrected frozen implementation;
MAIN accepted its source and validation gates on 2026-10-29. This is the implemented narrow exception to static/noncombined key-and-byte retention
promises elsewhere in this contract and the root identity history; those historical results remain
unchanged. The exception permits only these baked specialization discriminant updates:

- Static all-valid Scan with a retained MATMUL stage: `|scan-matmul-v4=` → `|scan-matmul-v5=`.
- Static-length Scan with a retained MATMUL stage: `|scan-length-aware-v1=` →
  `|scan-length-aware-matmul-v2=`. Select this branch only when the retained stage order satisfies
  `order.anyMatch(s -> s.right() >= 0)`, not from a MATMUL that lowering has removed.

Both apply to the already-admitted FLOAT32/FLOAT64 forward and actual Compiler-generated reverse
occurrences. `compatibilityBytes()` and `classIdentityBytes()` must both consume
`bakedSpecializationIdentity()` so the changed names/keys advance together and reject old artifacts
in the changed namespaces. No blanket schema bump or invalidation of unchanged controls is allowed.

Retain non-MATMUL shaped-v4 and static-length-v1, all-valid symbolic-v3, combined symbolic-length-v9,
pointwise/fixed, ordinary scalar/vector/tiled/TANH, `ordinary-batch-v1` and
`ordinary-batch-typed-v1` identities and generated bytes. Retained byte identity is a validation
target, not assumed proof. CPU0071's frozen-corpus checks cover the recorded controls; vector-byte
evidence is qualified by matching captured species. An unexpected changed namespace still stops
implementation for review; finite corpus evidence is not proof for every possible specialization.
Schema/family 71, canonical projections, numeric packs, resources and Runtime handoff stay unchanged.

The selected setup change moves static constant MATMUL strides outside the time loop; the evolving
scalar-to-dense factor remains a changing binding. No new production type/public API, RunState,
route, numerics, resource, worker or Job-reuse change is authorized; test fixtures are not new
production types. Scan and Runtime semantics remain solely with
their existing owners. The [CPU0071 brief](../../planning/backends/cpu/tasks/0071-common-matmul-binding-foundation.md)
records implementation, validation and diagnostic limitations, not additional authority or a speed claim.

### CPU0073 accepted row-map and multiple-terminal reduction fusion

CPU0073 scalar v1 remains governed below, with only the implemented
[CPU0075 scalar emission/cache cutover](#cpu0075-selective-scalar-load-and-address-reuse)
and the implemented [CPU0076 staging exception](#cpu0076-prospective-invariant-external-cell-load-staging)
qualifying affected scalar generation and literal retention. The implemented
[CPU0074 exec-v1 exception](#cpu0074-selected-row-region-execution-exception) qualifies its
scalar/range, additional-seam and retention restrictions; CPU0074 itself did not change v1 scalar artifacts.

Implemented by CPU0073 after user approval and authority acceptance on 2026-10-30. Independent
code/tests/Javadoc review approved with no open findings; MAIN accepted the implementation and
named conformance/Engine join. The [task result](../../planning/backends/cpu/tasks/0073-row-map-multi-reduction-fusion.md#result)
records validation, evidence-transfer limits and unmeasured performance.
This subsection owns the exact CPU mechanism and selective identity exception; plans and
[ADR 0052](../../design/decisions/0052-row-map-multi-reduction-fusion.md) do not add authority.
The [Scan owner](recurrent-scan.md#general-differentiable-scan) retains all Scan semantics;
[Prepare and Run lifecycles](runtime-prepare-engine.md#prepare-lifecycle) remain unchanged.

#### Physical region and admission

A row map is a pure ordered row-local directed acyclic graph (DAG) sharing the same proved row
axis R. It may include bounded feature contractions and feed independent row outputs plus
multiple terminal contractions or represented Scan sums. A terminal consumes rows and finishes
after all contributors; it cannot feed row work in the same pass. This region is derived CPU
physical lowering, not a new Tensor operation, Compiler graph, semantic DAG, scheduler or executor.
Ordinary lowering selects one complete execution unit. Scan wraps the same generation-time
region inside its existing unit/time traversal, never a per-step Runtime unit, callback or Job.

Implemented selection requires a genuinely computed row producer reaching a terminal and
another retained or terminal sink. Direct-input or movement sharing alone remains noneligible;
the bounded region is not blanket pointwise or arbitrary-shape fusion.

Admission is scalar, single-range, homogeneous FLOAT32 or FLOAT64 within existing capability:
ordinary static F32/F64, already-admitted ordinary symbolic F32, and independently admitted static,
symbolic, length and combined Scan occurrences. Canonical boundaries, existing dense/scalar read
exceptions and proved internal transpose/expand/feature-only edits remain the limits. No new
operation, dtype, layout or carrier follows: ordinary symbolic F64 MATMUL and all-valid symbolic
F64 Scan remain unqualified. Fixed bounded features, static R or existing named B, and fixed T
do not authorize B/T unrolling, per-B caching, runtime profitability decisions, SIMD, threads,
OpenBLAS, Job reuse or across-time carry promotion.

Ordinary exact SUM/MEAN remain outside the region: keep required producer materialization,
exact workspace slots/creators and empty/singleton behavior. Retain symbolic F32 five-limb/48-byte
and bounded symbolic F64 34-limb/280-byte state; static F64 remains domain-sized.
`CpuExactSumEmitter` is unchanged. Represented Scan SUM is its existing distinct algorithm,
not a relabeling of exact `CpuAggregateIr.Kind.SUM`. Model and existing math owners retain all
numerics: preserve contributor/operand order, typed MUL/ADD/result rounding and raw no-axis/
singleton movement. No new FMA, reassociation, derivative simplification or numerical policy.

#### Cold proof, resources and range

Before cuts or resource declarations, use complete partition/body uses, demanded/public/saved/
external uses and physical storage origins. Prove dependency convexity and an acyclic quotient
of the full DAG, including both contraction operands, views and temporal old/new hazards.
Retain exact producing stores for live external/saved uses or split the region; never replay
activations or remove demanded histories. Prove named row-axis provenance through each view,
injective writes, legal read aliases, no overwritten future reads, and inclusive cut lifetime/
size dominance for every admitted B. Equal extents (including F=B) or sampled B are not proof;
unknown aliases reject. Recheck actual backing aliases at binding.

Activity dominates both operand loads and dependent math, including shared/invariant hoists.
Preserve the Scan owner's B=0/1, T=0, all-inactive, raw carry identity, +0 padding and poisoned-read
rules, producer-ordered all-length preflight before any Scan-local write, and one distinct shared
carry ADD after row reduction at active(t). No carry ADD is folded into per-row contributions.

Every attempted candidate is bounded: at most 32 row sites, 4 terminal outputs, 64 terminal
scalar cells, feature extents at most 16 each, 128 simultaneously live JVM local slots (double
counts two), 32 boundary carriers, 64 candidate attempts per anchor, and a conservative generated
method-size proof at most 48000 bytes. Reject over-limit candidates before resources; do not
truncate work or cap the whole partition. These are cold structural limits, not measured
profitability thresholds; the live-value limit is primary and method size only a Class-File guard.

Ordinary fused work owns both row and shared outputs: prepared work count is 1 and entry [0,1)
executes the full body even at R/B=0, publishing required nonempty shared zeros. [0,0) does no
work; every other range rejects before writes. No rowCount work count, arbitrary pointwise
ranges or parallel splitting. Extend the existing ordinary binder for this case and all-output
alias checks rather than inheriting last-output/count assumptions; no Runtime protocol changes.

Select ordinary regions before decomposition/materialization, Scan regions before
`shapedResources`. Recompute liveness and exact scratch for selected regions, retaining required
banks, histories, old-carry hazards and exact-SUM producer stores. Finalization only binds assigned
slots and emits; it cannot discover requirements, change topology or rerun fusion. Method-size
verification failure then is a defect, not a late fallback. One RunState owns all mutable state.

#### Exact types, signatures and generation lifetime

Only three new top-level production types are admitted, under
`io.github.pho001.synaptik.backend.cpu`: `internal.ir.CpuRowMapReductionIr` (immutable,
implements `CpuPortableKernelIr`), `internal.lowering.CpuRowMapReductionLowering` (shared cold
proof), and `internal.codegen.emit.CpuRowMapReductionEmitter` (generation-only loop/lifetime
orchestration). The IR alone contains the two nested binding records below. Existing pointwise
IR and single-anchor Scan stages do not express multiple terminal accumulators sharing row sites.
These declaration excerpts describe the implemented internal seams, not supported API or runnable Java:

```java
// Nested in CpuRowMapReductionIr; ordered value-ordinal bindings.
public record Contraction(int output, int left, int right) {}
public record Sum(int output, int input, List<Integer> axes, boolean sumToShape) {}
// CpuRowMapReductionIr: immutable snapshots; cold ordinal validation.
public CpuRowMapReductionIr(CpuKernelIr arithmetic, List<Integer> rowSites,
    List<Contraction> contractions, List<Sum> sums, List<Integer> terminalOutputs,
    List<CpuAccessPlan.BatchBinding> accesses, List<Integer> viewOrigins,
    List<Integer> rowAxes, List<Integer> retainedStores);
public CpuKernelIr encodedKernelIr();
String structuralKey();
String bakedSpecializationIdentity();
// CpuRowMapReductionLowering; complete normalized uses and required order/alias edges.
public static Optional<CpuRowMapReductionIr> prove(CpuRowMapReductionIr candidate,
    List<List<Integer>> allUses, List<List<Integer>> requiredOrder);
// CpuRowMapReductionEmitter; package-private, generation-only.
static void emitBound(CodeBuilder code, CpuRowMapReductionIr ir,
    int rowIndex, int rowLimit, int[][] valueLocals, int[][] terminalAccumulatorLocals,
    Runnable init, Consumer<Runnable> activeRow, IntConsumer emitRowSite,
    IntConsumer emitTerminalStep, Runnable finish);
```

`rowSites` interleaves pointwise outputs and row-feature contraction outputs topologically;
each ordinal resolves to exactly one kind and completes its fixed feature cells before consumers.
Terminals are excluded from rowSites and ordered by `terminalOutputs`; operands are dominated.
Use typed contraction loaders, not fabricated ordinary F64 batch MATMUL IR.
`viewOrigins[value]` covers every value with its self-rooted storage-producing ordinal; accesses
encode geometry relative to that origin (zero batch factors for static recipes). BatchBinding
does not encode alias identity. Validate total coverage, only self-root cycles, view composition,
rowAxes provenance and physical hazards. Proof lists contain full normalized uses and must-precede
pairs including conservative alias edges, not candidate-local counts.

The caller allocates locals with the same CodeBuilder: INT rowIndex/rowLimit, valueLocals by
value ordinal/fixed feature cell, terminalAccumulatorLocals by terminal position/cell. Only direct
storage uses -1; validate typed widths, disjoint slots and load dominance. Callbacks close over
these exact immutable handle tables; site callbacks receive ordinals and neither replace locals
nor mutate the tables or rowIndex. Emitter owns control/lifetimes, not allocation: init once,
rowIndex=0, increasing loop to prebound rowLimit, activeRow around all rowSites then terminal
steps, finish once after the loop, including R=0. Finish publishes completed contributions and
separate carry ADDs; Scan's outer active-time gate encloses emission. All callbacks/arrays exist
only during generation, not as runtime/per-step objects.

Existing math owners expose only these additional package-private static void generation seams:

```java
// CpuMatmulEmitter; ordered typed loaders, typed accumulator local.
static void emitInit(CodeBuilder code, DataType dataType, int accumulator);
static void emitContributorStep(CodeBuilder code, DataType dataType, int accumulator,
    boolean directProduct, Runnable left, Runnable right);
static void emitFinish(CodeBuilder code, DataType dataType, int accumulator, IntConsumer sink);
// Geometry-only dispatch, before any contributor traversal; callbacks rebuild branch-local caches.
static void emitSingletonChoice(CodeBuilder code, TypeKind kind, Runnable bound,
    Consumer<Boolean> body);
// CpuAggregateEmitter; represented Scan SUM only, retaining raw singleton movement.
static void emitRepresentedInit(CodeBuilder code, DataType dataType, int accumulator,
    boolean directSingleton);
static void emitRepresentedContributorStep(CodeBuilder code, DataType dataType, int accumulator,
    boolean rawSingleton, Runnable load);
static void emitRepresentedFinish(CodeBuilder code, DataType dataType, int accumulator,
    IntConsumer sink);
```

Reuse these seams in old complete loops and new orchestration, except for the explicit
singleton correctness cutover above. Reuse `CpuClassFileKernelGenerator.emitBoundPointwise`,
`CpuScalarEmitter` and carrier
access; do not clone opcode/math switches or recreate retired Scan math subemitters.
Extend existing portable permits/route, partition lowering/decomposition/recognition/preparation,
Scan lowering/Stages/emitter, specialization, generator and ordinary binder owners only.
`CpuPortableRoutePlan.encoded(CpuPortableKernelIr)` must explicitly return
`region.encodedKernelIr()` before its final aggregate cast. That envelope exposes boundary
stores/family identity with the same structuralKey; typed region facts select emission, not a
second semantic graph. `CpuGeneralScanStageSchedule.order` remains a post-slot reorder proof,
not fusion, and must see region reads/writes/lifetimes. Any extra named production type,
common-seam change beyond the approved signatures, or cross-owner protocol/runtime edge requires
explicit approval; a plan alone cannot authorize it. Ordinary private implementation helpers in
existing owners are permitted within this capability; they are not extra common seams or types.

#### Selective compatibility and retention

Keep global generator `CURRENT_VERSION=71`, existing namespaces and Scan class projection 71.
Reserve ordinary class projection 72 only for selected row regions. Append
`Optional<CpuRowMapReductionIr> rowMapReductionIr` to existing `CpuKernelSpecialization`:
projection 72 iff present, mutually exclusive with matmulIr/generalScanIr. Scan embeds regions
only within generalScanIr, retaining projection 71; keep 54 iff matmulIr and 71 iff generalScanIr.
Preserve every existing constructor overload, including the old canonical signature, delegating
with Optional.empty(). Check fingerprint, boundary dtype/carrier/access, scalar/species-zero,
materializedSourcePosition=-1 and scratch requirements against the region. Ordinary symbolic
regions remain F32 only; this is no bypass of F64 batch-MATMUL rejection.

Use geometry-free ordinary family marker `row-map-multi-reduction-v1`. Append
`|ordinary-row-map-multi-reduction-v1=` only for selected ordinary groups, and
`|scan-row-map-multi-reduction-v1=` only to selected Scan groups' existing baked mode payload.
Both `compatibilityBytes()` and `classIdentityBytes()` consume the same baked payload and reject
old unfused artifacts for selected fused specializations. Bind topology, ordered operands/sites,
dtype/carrier/access, algorithm/singleton/rounding distinctions, row/activity roles, retained
stores/demand, feature/coefficient/resource/lifetime facts, range convention and all code-shaping
limits. Include fixed R/T/strides/resource bytes when baked into ClassFiles. Exclude only
invocation-resolved symbolic B, length values, addresses, assigned slots and invocation-resolved
resource bytes. Diagnostic text is not identity; no blanket schema bump or control namespace change.

This exception and CPU0071's separate implemented exception do not retire noneligible controls.
Noneligible ordinary/static/symbolic/length/combined, vector/tiled/TANH/fixed specializations keep
literal old keys, ClassFiles and resources. Freeze controls before source edits, never derive them
from candidate code. Test both streams, constructor/factory exclusivity, encoded route construction,
artifact generation/load/reuse and stale rejection, plus B/L-independent reuse. Qualify ordinary
and actual Compiler-generated Scan reverse separately with durable specialized clean-Java oracles
and generated-ClassFile inspection; finite controls are not universal preservation proof.
Only after both callers work may superseded eligible whole-domain traversal glue/cuts retire;
keep fallback emitters, scheduling/temporal proofs, exact SUM, live saved stores and old oracles.
No performance claim or measurement permission follows; old timing budgets remain closed.

### CPU0074 selected row-region execution exception

Implemented, independently reviewed and MAIN accepted, 2026-10-30. CPU0074 is Complete after
final CPU/Javadoc, named integration and bounded performance evidence. Steady-state profitability
is not qualified; defaults and rollout are unchanged. This section is the sole CPU mechanism/ABI/identity owner for
this exception. [ADR 0053](../../design/decisions/0053-row-fusion-simd-parallel.md) records rationale;
the ADR alone grants neither implementation launch nor executable capability.

#### Bounded selection and numerical boundary

CPU0073's genuine computed-row-producer, complete-use, convexity, alias, named-axis, liveness,
retained-store and structural-limit proofs still apply. The same ordinary static F32/F64,
ordinary symbolic F32, static all-valid F32/F64 Scan, all-valid symbolic F32 Scan, and static-length/
combined symbolic-length F32/F64 Scan domains may select vector, parallel-scalar or parallel-vector
regions. This is not ordinary symbolic F64 MATMUL or all-valid symbolic F64 Scan admission.
Ordinary array/checked-segment eligibility and Scan's checked-segment-only boundaries remain.
Unproved vector mapping retains scalar compute; unproved worker ownership retains inline execution.
No new dtype/layout/carrier, general pointwise rewrite, native route or automatic all-op Scan SIMD.

Vectorize fixed feature/output cells within one active row, never implicit feature-K splitting.
Use admitted existing vector instructions and masked feature-tail access/publication. Feature K
retains its original ordered MUL/ADD sites; unused vector lanes never enter another result's
reduction. Only selected terminal row-axis contractions and represented Scan SUM may regroup
under the [Scan semantic exception](recurrent-scan.md#cpu0074-selected-terminal-grouping-exception).
Standalone MATMUL-K and ordinary exact SUM/MEAN, including producer stores/workspaces, remain
unchanged. Same compute/accumulator dtype, no implicit FMA/TF32/lower precision or public numerical
mode. Raw movement/casts/indices/masks, singleton/empty identities and selected-realization saved
values remain exact. No cross-strategy/thread-count/device bitwise claim.

Strategy, maximum ranges W and minimum row work m remain in existing preparation
`CpuPartitionPreparationPlan.ExecutionUnitPlan`; no worker policy enters canonical row IR.
Cold run binding resolves N=min(W,ceil(R/m)) for R>0, otherwise zero, with quotient/remainder
partitioning of [0,R) in ascending range index. The rule is fixed before execution and across
time, not length-value profitability selection. N<2 executes inline. Each worker owns disjoint
rows and terminal partials; merge valid partials in fixed ascending range-index order on the caller.
Each bounded partial slot has typed terminal cells and an eight-byte validity flag; skip invalid
empty/inactive partials, never invent +0 contributors or infer singleton geometry from activity.
No shared atomic floating updates or B×parameter bank. Keep one complete ordinary execution unit;
internal range/merge helpers, not arbitrary pointwise ranges, implement new non-scalar realizations.
Scalar v1's work=1/[0,1) and rejection behavior remain unchanged.

Qualify actual F3/H2 saved-primal RNN reverse as a selected vector/parallel region independently
of a genuinely fused forward fixture. An ordinary RNN forward with no eligible terminal region
remains scalar. CPU0010M production scalar/thread-count-one defaults do not change; only existing
explicit eligible configuration/tuning selects new realizations cold.

Cold candidate metadata uses the appended `CpuFusionDecision.WorkspaceRole.ROW_REGION_STATE`
only for ordinary selected row scratch authenticated by matching row IR/emission and no Scan/
recurrent owner. The shared `GENERAL_SCAN_BANKS` preparation tag alone is insufficient. Candidate
construction and retained-plan validation use the same guard; actual Scan/recurrent banks remain
outside generic candidate discovery. Internal row-to-terminal uses stay inside the indivisible
unit, not false cross-unit materializations. Ordinary pointwise accounting and scoring are unchanged.

#### Exact new internal seams

MAIN addendum accepted 2026-10-30 after focused independent APPROVE: the two additional
generation/cold-binding declarations below complete the existing CPU0074 packet. No new type,
runtime protocol or capability-completion claim follows from those declarations.
MAIN/focused-reviewer correction accepted 2026-10-30: the Scan overload also transports
positive minimum row work cold, as specified below; this replaces the unimplemented seven-argument form.
MAIN/Boyle byte-seam addendum accepted 2026-10-30: only the two checked-segment byte-address
generation methods below are added. These accepted seams are implemented and MAIN accepted.

All names are under `io.github.pho001.synaptik.backend.cpu`; no new top-level production type.
The following are non-runnable declaration excerpts, not supported application API:

```java
// Nested in internal.cache.CpuKernelSpecialization: emitted facts, not selection policy.
public record RowFusionEmission(int maximumPartialCount,
    List<Integer> invariantTransposeValues, List<Integer> temporalTransposeValues,
    int stepOperandCount, int stepFrameLongCount) {}
// Append Optional<RowFusionEmission> rowFusionEmission to CpuKernelSpecialization;
// every old constructor, including the former canonical signature, delegates empty.
// Existing internal.codegen.emit.CpuRowMapReductionEmitter; generation only.
static void emitRangeBound(CodeBuilder code, CpuRowMapReductionIr ir,
    int rowIndex, int rowBegin, int rowEnd,
    int[][] valueLocals, int[][] terminalAccumulatorLocals,
    Runnable init, Consumer<Runnable> activeRow, IntConsumer emitRowSite,
    IntConsumer emitTerminalStep, Runnable finish);
// Existing CpuMatmulEmitter; matching FloatVector/DoubleVector reference locals/loaders.
static void emitVectorInit(CodeBuilder code, DataType type, int accumulator);
static void emitVectorContributorStep(CodeBuilder code, DataType type, int accumulator,
    boolean directProduct, Runnable left, Runnable right);
static void emitVectorFinish(CodeBuilder code, DataType type, int accumulator, IntConsumer sink);
// Existing CpuAggregateEmitter; represented Scan SUM, never exact ordinary SUM/MEAN.
static void emitRepresentedVectorInit(CodeBuilder code, DataType type, int accumulator,
    boolean directSingleton);
static void emitRepresentedVectorContributorStep(CodeBuilder code, DataType type,
    int accumulator, boolean rawSingleton, Runnable load);
static void emitRepresentedVectorFinish(CodeBuilder code, DataType type, int accumulator,
    IntConsumer sink);
// Existing CpuCarrierEmitter: append matching VectorMask local to full current forms.
void vectorLoad(DataType type, CarrierAccess access, int parameterSlot, int addressLocal,
    boolean broadcast, boolean intAddress, int maskLocal);
void vectorStore(DataType type, CarrierAccess access, int parameterSlot, int addressLocal,
    int valueLocal, boolean intAddress, int maskLocal);
// Existing CpuCarrierEmitter; package-private generation-only checked-segment byte access.
void vectorLoadBytes(DataType type, int carrierSlot, int byteAddressSlot, int maskLocal);
void vectorStoreBytes(DataType type, int carrierSlot, int byteAddressSlot,
    int valueLocal, int maskLocal);
// Existing CpuGeneratedKernel: explicit cold lookup accessors, never a runtime registry.
public MethodHandle ordinaryRowRangeEntryPoint(); // specialization.entryType()
public MethodHandle ordinaryRowMergeEntryPoint(); // specialization.entryType()
public MethodHandle scanRowRangeEntryPoint();
// Scan range helper: (MemorySegment[], long[], long begin, long end,
//                    long partialByteOffset)void
// Existing package-private CpuGeneralScanEmitter; generation-only access to its private binder.
void emitRowRange(CodeBuilder code, CpuKernelSpecialization specialization, CpuGeneralScanIr ir);
// Existing internal.executable.CpuPreparedExecutable; ordinary symbolic constructor overload.
public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
    CpuGeneratedKernel artifact, CpuPortableRoutePlan route,
    Optional<WorkspaceSelection> workspace, CpuConcurrencyBudget budget,
    int selectedRangeCount, long minimumElementsPerWorker, CpuWorkerGroup workerGroup);
// Existing six-argument constructor delegates with (1, 1, null).
// Nested in internal.executable.CpuGeneralScanPreparedExecutable.
public static final class StepDispatch {
    private StepDispatch(MemorySegment[] operands, long[] frame,
        CpuWorkerGroup.RangeCall[] calls, CpuWorkerGroup workerGroup);
    public MemorySegment[] operands();
    public long[] frame();
    public void execute();
}
// Existing CpuGeneralScanPreparedExecutable: keep old six-argument constructor; delegate (null,1).
public CpuGeneralScanPreparedExecutable(PreparedMemoryPlan plan, List<BufferSelection> buffers,
    Optional<WorkspaceSelection> workspace, Facts facts, CpuGeneratedKernel artifact,
    CpuConcurrencyBudget budget, CpuWorkerGroup workerGroup, long minimumElementsPerWorker);
// No retention of the unimplemented seven-argument overload is required.
// Existing executable package only; RangeCall stays package-private.
void CpuWorkerGroup.execute(RangeCall[] calls, CpuConcurrencyBudget.Lease reservation);
boolean CpuConcurrencyBudget.Lease.covers(CpuConcurrencyBudget budget, int demand);
```

The qualified last two declarations identify methods on their existing owners, not Java source
syntax. emitRowRange generates the already-approved scanRowRange helper using the existing private
temporal/row binder; it is not a runtime callback. The ordinary symbolic constructor overload
transports existing unit-owned range policy and borrowed workers to cold numeric B binding,
without per-B artifact selection or worker policy in canonical row IR. The old six-argument
constructor retains behavior by delegating (1,1,null). The selected parallel Scan entry appends StepDispatch after its existing geometry long[]
and before start/end longs; ordinary and nonparallel entry ABIs remain unchanged. The generated
entry reads context arrays once, then emits bounded frame updates and direct execute calls at
the single selected region. No stage ordinal lookup, callback registry or per-operation dispatch.

vectorLoadBytes/vectorStoreBytes accept only checked MEMORY_SEGMENT FLOAT32/FLOAT64 carriers,
the dtype's target preferred species, a LONG byte-offset local and a matching mask (and matching
value vector for stores). Emit direct native-byte-order masked fromMemorySegment/intoMemorySegment;
disabled lanes access no storage. Scalar broadcasts retain existing loadBytes plus broadcast.
Keep old element-address methods and array paths literally unchanged. Migrate only selected
exec-v1 Scan byte addresses and ordinary private partial scratch. This removes the emitted
byte-offset /width*width roundtrip; JIT elimination is unproved and no slowdown/speed claim follows.
No new type/layer, runtime entry ABI, resource or global schema change. Selected exec-v1 remains
unreleased; final inspection/performance must use a fresh private artifact root so an old
candidate cache cannot hide the changed generated shape.

The Scan overload rejects nonpositive minimumElementsPerWorker with IllegalArgumentException.
CpuPartitionFinalizer passes unit.minimumElementsPerWorker() during cold finalization; policy stays
in the existing unit, not canonical IR, Facts or specialization. The old six-argument Scan
constructor delegates (null,1), preserving its inline behavior. No runtime protocol changes.

RowFusionEmission snapshots nonnull disjoint duplicate-free ordinal lists; each ordinal names a
proved source value. maximumPartialCount is positive; other counts are nonnegative and checked
against emitted frame layout and existing unit facts. Scalar v1 has no payload. Range tables use
typed primitive cells for scalar code or reference locals by fixed feature chunk for vector code;
validate coverage, disjointness and load dominance at generation. Callbacks/handle tables remain
generation-only. Init/finish execute once around [rowBegin,rowEnd), including zero-trip ranges.

Vector init uses the selected target's preferred F32/F64 species, whose bits must match the
specialization, never arbitrary species. Specialization validation must require that dtype's
target preferred species and matching accumulator, load and mask species; arbitrary species
cannot qualify by equal lane count alone. It initializes typed +0 except omitted direct-singleton
reset. Loaders push matching vectors; MATMUL emits a singleton MUL assignment or separate
per-lane MUL then ADD for K>1, never FMA.
Represented SUM emits typed ADD or raw singleton assignment. Finish passes the completed local
to the generation-only sink without arithmetic. Reuse existing pointwise/vector math, loop and
carrier owners, not copied opcode switches or another RNN emitter. All further common signatures,
named production types or cross-owner edges require explicit approval.

#### Resources, dispatch lifetime and compatibility

Analysis fixes all staging/partial/frame facts before resource declarations/slots. F3/H2 reverse
may stage canonical W/U transposes in ten represented cells: 40 F32 / 80 F64 bytes before alignment.
Stage once per run only when invariant across T and inside any-active gating. SHARED histories/
changing cuts stage at their exact active original time after producers, or vector fallback;
never reuse stale data. Reuse that bounded area only after all readers join. Preserve demanded
stores, carry banks, inclusive cut lifetimes and CPU0073 method/local/resource limits. T=0/B=0/
all-inactive executes no such body staging. No per-row transpose arrays, B×parameter banks or
T-fold code/buffers. Run-bound context/calls/arrays are O(body+W), never O(T).

Use the [run-owned synchronous reservation](runtime-prepare-engine.md#cpu0074-run-owned-synchronous-reservation).
StepDispatch is private in purpose, public only for generated cross-package bytecode access.
One RunState owns it and its mutable arrays; accessors expose those run arrays without copying.
Calls/helpers are resolved and bound before execution. Per-step work updates only current
operand/time/base facts, never repacks geometry or rediscovers graph/access/strategy facts.
The exception allows synchronous selected-region dispatch and existing calls.clone/Job/failure/
latch/queue metadata per dispatch, not general callbacks, body allocations or another executor.
Job reuse is optional. Length producers and complete typed/access/alias/value preflight still
precede all Scan-local writes. Worker accessibility includes staging/partial scratch.

Keep envelope 71, ordinary projection 72, Scan projection 71 and canonical row structuralKey.
Only new selected realizations append `|ordinary-row-map-multi-reduction-exec-v1=` or
`|scan-row-map-multi-reduction-exec-v1=` after existing payloads in both specialization streams.
Bind selected strategy/preferred species, mask/transpose/lifetime facts, fixed partition/merge
algorithm, emitted range/frame/partial/helper layout/types, all code-shaping bounds/resource recipes
and baked geometry/bytes. Exclude actual symbolic B/L, addresses/slots and invocation-resolved bytes.
Unit policy stays preparation-owned; key it where it shapes emitted meaning or compatibility.
Scalar v1 and noneligible keys/ClassFiles/resources/ABIs remain literal unchanged except for the
implemented [CPU0075 scalar generation/identity cutover](#cpu0075-selective-scalar-load-and-address-reuse).
Validate explicit
helper shape, artifact stale rejection/reuse and B/L-independent reuse without blanket schema bumps.

Acceptance requires each realized eligible domain, actual F3/H2 reverse and separate fused forward,
optimal same-specialization Java/ClassFile comparison, no hidden elementwise allocation/boxing/
reflection/dispatch, vector tails, exact masks/singletons/empties/saved sites, finite per-op/dtype
tolerances plus cancellation/special/raw checks, and budget/access/error/close/concurrency tests.
Finite performance validation needs its own task budget; no historical campaign reopens or
universal speed/bitwise/JIT guarantee follows. MAIN accepted functional capability after final validation;
the [CPU0074 Result](../../planning/backends/cpu/tasks/0074-row-fusion-simd-parallel.md#result)
records final evidence, concrete JIT compilation-pressure limitations and a separate recommended
investigation, not another authorized task/frontier or timing campaign.

#### CPU0075 selective scalar load and address reuse

Implemented and independently reviewed, 2026-10-30, after explicit user approval. CPU0075 is
Complete after MAIN's final CPU/Javadoc gate and the sole bounded comparison; its
[Result](../../planning/backends/cpu/tasks/0075-row-fusion-scalar-load-address-reuse.md#result)
records evidence and limits. This subsection qualifies only CPU0073/CPU0074 generation and
compatibility retention, not their numerical, eligibility, resource-ownership or worker semantics.
[ADR 0054](../../design/decisions/0054-row-fusion-scalar-load-address-reuse.md) explains the decision.

Replace affected ordinary and general-Scan scalar row emission, including existing parallel-scalar
row bodies, in the existing emitter/binder owners. Reuse only repeated external loads within one
active row and fold/share proved constant affine address terms and row bases. SIMD bodies and
unrelated emitters remain literal controls. No W/U hoisting out of the row, across-T carry promotion,
new fuser, dtype/layout/carrier capability, public policy, runtime object, entry-ABI parameter,
top-level production type or new common helper seam is authorized. Private helpers in existing
owners are permitted; any needed expansion requires approval before implementation.

The implemented [CPU0076 exception](#cpu0076-prospective-invariant-external-cell-load-staging)
qualifies only this within-row-only load-placement prohibition and affected
scalar retention. CPU0075's completed implementation and evidence are not reinterpreted.

A shared load identifies its authenticated source/storage origin and complete effective affine
cell recipe, including symbolic B coefficients, temporal/bank binding and store epoch (the interval
between potentially aliasing writes). A value ordinal, equal extents or sampled B is not proof.
Prove alias/view equivalence, unchanged storage across all intervening retained/public/scratch
writes, activity and first-load dominance on every path. A possibly aliasing store ends that reuse
interval; unknown equivalence keeps a direct checked load. Neither loads nor address-dependent
access move above existing row/time/empty guards. Preserve checked spans, alignment, overflow,
raw inactive identities and all required stores and floating instruction order.

Cold analysis before resource declarations/slots must conservatively bound the replacement's
simultaneously live typed values, cached loads and addresses as well as existing bindings and
accumulators. Retain the 128-slot and method-size limits; doubles/longs count two. Allocate cached
cells only for repeated uses, release at last use and reuse type-compatible local slots; do not
retain every external cell through the row. JVM max_locals is not hardware-register liveness.
If reuse cannot be proved or bounded, use unshared accesses within the replacement or reject the
candidate cold through the existing fallback. Finalization cannot discover resources or choose a
new route. No maintained parallel legacy scalar emitter or runtime discovery follows.

Only affected scalar/parallel-scalar specializations append the scoped
`|ordinary-row-scalar-load-address-reuse-v1=` or `|scan-row-scalar-load-address-reuse-v1=` marker to
both `compatibilityBytes()` and `classIdentityBytes()`. Bind every deterministic code-shaping
reuse/address/proof choice using existing specialization facts; exclude actual symbolic B/L,
addresses, assigned slots and invocation-resolved bytes. Keep envelope 71, ordinary/Scan
projections 72/71, canonical structural keys and all existing namespaces/payload versions unchanged.
Old affected scalar artifacts are stale under the new keys; artifacts outside this cutover keep
literal keys/ClassFiles/resources and entry/helper ABIs. Test both streams and B/L-independent reuse.
Historical `.tsv` and `.gz.b64` controls are not regenerated from candidate code: affected snapshots
prove selective stale rejection, while unaffected SIMD/nonregional snapshots remain literal controls.
Captured old bytecode may be a test-only comparison arm, never a production emitter alternative.
The implemented scalar path ends value reuse at every retained write, excludes changing Scan
carry/scratch bindings, and leaves alternative SUM loads unshared with address-recipe invalidation.
Cold analysis reserves remaining capacity under the existing 128-slot limit; emission caps its
typed pool and single LONG offset within that capacity. Unavailable capacity preserves checked
unshared reads, not a changed route or broader SIMD rejection.
The [CPU0075 brief](../../planning/backends/cpu/tasks/0075-row-fusion-scalar-load-address-reuse.md#result)
records the exhausted separate validation budget. Reduced bytecode load sites are structural
evidence, not a matched-JIT speed guarantee or C1 repair; no material Scan speedup was established.
CPU0074 and all historical campaigns remain closed; defaults and rollout are unchanged.

<a id="cpu0076-prospective-invariant-external-cell-load-staging"></a>

#### CPU0076 invariant external-cell load staging

Accepted after explicit user approval, 2026-10-31; implemented with independent source/protocol,
bounded-result and documentation approval, 0 P1/P2, and MAIN's final CPU/Javadoc/named Engine PASS.
This is the sole owning CPU load-placement/compatibility exception to
CPU0075's within-row-only prohibition. [ADR 0055](../../design/decisions/0055-row-invariant-load-staging.md)
explains the choice; the [CPU0076 brief](../../planning/backends/cpu/tasks/0076-row-invariant-load-staging.md)
owns execution and the separate finite measurement grant. Prior completed results stay historical.

An external cell is a represented value read from an authenticated storage origin, not a row
producer or accumulator local. Existing homogeneous FLOAT32/FLOAT64 ordinary and general-Scan
scalar row regions may stage such cells in typed method locals outside the row loop when the
following proof succeeds. This is generic source/access analysis, never an RNN, F3/H2, ordinal,
root-number or weight-value special case. Retain existing static/named-B/length/combined admission,
ordinary array/checked-segment and Scan checked-segment boundaries; ordinary symbolic F64 MATMUL
and all-valid symbolic F64 Scan gain no qualification.

Authenticate the source, complete view/storage origin and effective affine cell recipe for all
admitted B: fixed and B coefficients, row coefficients, byte base/strides, original-time source,
bank selection and store epoch. Prove structural row independence, nonempty consuming-domain and
first-load dominance, source readiness, and storage/mapping stability throughout the staged
value's complete lifetime. Equality of extents, nominal ordinals, copied history values or sampled
B is not proof. Include every retained/public/saved/carry/scratch write and inclusive cut lifetime;
actual binding still checks backing overlap/access. Unknown alias, mapping, lifetime or dominance
keeps current checked loads. An unrelated write need not end a proved disjoint cell's lifetime;
a possibly aliasing write does. No new alias permission or unchecked access follows.

True invocation invariants load once per generated entry invocation, after existing preflight
and nonempty/any-active guards, before their consuming row loop; they may stay local across T
only with proved temporal invariance. A lazy first-consuming-active-time placement is permitted
if it dominates every use and performs no extra read. A SHARED sequence/history is not an
invocation invariant: stage only after its producer at each consuming active original t, under
active(t), with the exact original-time base. ROW histories, evolving carries and bank/scratch
bindings retain their existing loads in this implementation. Authenticate the SOURCE activity
domain: selecting a constant row from a ROW history does not permit global-active-time staging.
Already-hoisted typed scalar roots are excluded from new placement and its marker. Neither a
scalar-broadcast initializer, zero stride nor SHARED role proves temporal invariance.
The [Scan owner's activity and history rules](recurrent-scan.md#activity-raw-identity-and-retained-histories)
remain unchanged. T=0/B=0/all-inactive/empty-feature paths perform no speculative body staging;
required nonempty shared zero publication and raw inactive carry identities still execute.

Existing parallel-scalar may stage only inside its current method/range bindings when this proof
and local lifetime fit the existing helper ABI. Worker staging additionally requires a consuming
active row in that worker's range: global active(t) alone does not qualify an all-inactive range.
No transport of typed locals between caller and worker, frame extension, new staging bank,
dispatch/reservation change or cross-worker cache is authorized. A helper-preheader load is once
per helper invocation, not once per whole Scan across repeated dispatches. If the intended wider
lifetime cannot fit, retain the current checked path and record that precise qualification.
Inline scalar is the primary acceptance path.

The implemented proof accepts only complete, actually consumed fixed-feature view domains;
rank greater than two and partial/unused/empty domains remain checked. For Scan, every nonformal
self-rooted producer must belong to the region's row sites, terminals or finish epilogue.
Otherwise the binding owners supply empty staging scopes, even when one region in a larger body
would pass an isolated proof. This bounds the full invocation lifetime, not just the pruned
regional stage. Small ordinary F64 staging is qualified; the larger combined F64 primary geometry
retains CPU0075 emission. None of these optimization limits reduces existing execution capability.

Cold proof before cuts, resource declarations and slots bounds staging plus bindings, addresses,
producer values, terminal accumulators and controls, separately for caller/helper phases,
including inline scalar without exec-v1 metadata. Retain all CPU0073 limits, especially 128
simultaneously live JVM slots (DOUBLE/LONG count two) and 48000-byte method bound; use last-use/
type-compatible slot reuse where valid. Do not increase resource limits, scratch/frame recipes
or entry/helper signatures. Finalization emits the fixed proof, not late resource discovery.
Unproved or over-budget staging retains CPU0075 checked emission or existing cold fallback
without shrinking already-supported capability or rejecting an otherwise eligible strategy.
No parallel maintained old emitter or runtime selection is added.

The applicable structural no-duplicate caller/helper floors independently fit conservative 112/116
non-pool reserves; they include rooted stride/address facts, typed producers and terminals,
bindings, controls and CPU0075's LONG offset. Additional wrapper allowances cannot conceal an
excessive floor. Let D count extra demanded wrappers per producing row origin and H count ALL
demanded history ports, including forwarded formals and SHARED histories outside regional row
sites. Every binding owner uses caller base `112 + 3*D + 2*H` and helper base `116 + 3*D`:
each extra wrapper includes a carrier and LONG base in both phases; every history retains its
own LONG caller base across prepublication and staging. Ordinary has no Scan history allowance.
For typed width w and distinct staged-cell slots S, each applicable CPU0075 pool is capped by
both its prior bound and `floor((128 - phaseBase - S - 1) / w)`. The one-slot readiness allowance
is conservative for inline emission and covers the helper's first-active-consumer flag.
Unproved/over-budget selection is empty BEFORE cuts/resources/slots, not a strategy rejection.

Only load placement and proved address work may change: preserve ordered typed MUL/ADD/SUB,
operand/contributor order, retained stores and saved sites, ordinary exact SUM/MEAN, and distinct
memory-backed carry ADDs after each active-time reduction. No carry promotion, FMA, arithmetic
regrouping or SIMD-body rewrite. Add no top-level production type, public API/policy, capability,
runtime object, generic framework or ownership/dependency change. Private helpers in existing
owners are permitted. The following cold method is the ONLY new shared seam/signature exception;
any broader requirement stops for explicit approval.

<a id="cpu0076-shared-cold-proof"></a>

MAIN's accepted request and independent exact-seam APPROVE, 0 P1/P2, 2026-10-31, authorize this
declaration in existing `internal.lowering.CpuRowMapReductionLowering` only. This is the implemented
internal declaration, not a supported public API:

```java
public static List<Integer> invariantLoadStaging(
        CpuRowMapReductionIr candidate,
        List<Integer> invocationInvariantOrigins,
        List<Integer> activeTimeOrigins,
        long callerBaseLiveSlots,
        long helperBaseLiveSlots);
```

Use existing `java.util.List<Integer>`, `CpuRowMapReductionIr` and primitive long only. Public
visibility permits cold internal cross-package access, not a new production type, record
component, emitter, generated helper, entry/helper ABI or runtime protocol. The method is pure,
read-only and deterministic: no runtime storage/input reads, identity parsing, class generation,
allocation-derived eligibility, measured profitability or route/resource decision.

Return an immutable ordered list of actual consumer-view ordinals for complete fixed-feature
cell-domain staging, with deterministic origin/affine deduplication. Every staged cell must have
an actual consumer; partial, unused or unproved domains stay checked. In this one owner, enumerate
exact consuming requests/retained-write epochs and prove source/view origin, all-B effective
affine equality and zero row coefficients under the authenticated temporal/full-lifetime alias
and write scopes. Private helpers here may implement the proof; competing private eligibility
or selection in callers is forbidden.

The binding owner supplies disjoint authenticated read-only source-origin lists: invocation
invariants versus active-original-time sources. Membership is by storage origin, not consumer
ordinal/shape/SHARED alone. Carries and uncertain cut/bank/scratch origins are absent. These same
authenticated scope lists AND conservative structural caller/helper base-live-slot facts must
drive pre-cut lowering (`prove` and existing `CpuGeneralScanLowering` row-region/`shapedResources`
call paths), `CpuPartitionPreparer` analysis, both specialization identity streams, and emission.
No phase substitutes allocation-derived counts or a privately recomputed selection.

Base-live-slot arguments cover all non-pool state: bindings, controls, addresses, row/terminal
locals and fixed CPU0075 state. Zero helper bound means no helper; DOUBLE/LONG count two. Jointly
bound staging, ready/control flags and the remaining CPU0075 typed pool within 128 slots in each
applicable phase; retain the existing method bound. Facts exclude actual B/L, addresses, slots
and measured cost. Unknown proof, checked-arithmetic failure or over-budget optimization returns
empty: retain the CPU0075 checked path without rejecting an otherwise eligible capability or
strategy. Malformed/null metadata keeps existing IllegalArgumentException/NullPointerException
conventions; it is not silently accepted as optimization failure.

Emission binds only this selected complete cell set to locals. The existing binding owner still
authenticates source readiness, temporal scopes, all writes/aliases over the FULL staged lifetime,
activity/nonempty dominance and actual span/alignment/overlap preflight. The method grants none
of those permissions by itself; worker-local staging cannot cross the ABI. Cache appends the
affected-only marker iff this same selection changes emitted scalar placement; empty/no-stage,
SIMD and nonregional cases keep literal controls. No validation/performance grant is added.

Only scalar cases whose emitted placement changes append
`|ordinary-row-invariant-load-staging-v1=` or `|scan-row-invariant-load-staging-v1=` in BOTH
`compatibilityBytes()` and `classIdentityBytes()`, after existing payloads. Bind deterministic
source/affine/temporal/store-lifetime/activity placement and local-budget choices through existing
facts and the scoped algorithm marker. Exclude actual symbolic B/L, addresses, slots and
invocation-resolved bytes. Preserve canonical keys, envelope 71, ordinary/Scan projections 72/71
and prior namespace/payload versions. No-staging cases retain CPU0075 identities/bytes/resources;
SIMD/nonregional controls and immutable historical `.tsv`/`.gz.b64` resources remain literal.
Test affected old-artifact rejection, current generate/load/reuse and B/L-independent reuse in
both streams. Captured old code is test-only evidence, never a production alternative.

The [CPU0076 Result](../../planning/backends/cpu/tasks/0076-row-invariant-load-staging.md#result)
records actual scalar F32 Compiler-generated reverse Scan, generic typed/temporal/helper tests,
resource fallbacks, literal retention and the sole MAIN-released comparison. Its bounded B256
improvement does not imply universal speed, Java parity or C1 repair; B32 is JIT-confounded.
The off-tree Java experiment remains historical input evidence. All measurement grants are now
spent; defaults, rollout and stage/commit/push remain unchanged or unauthorized.

### Accepted ordinary symbolic-batch CPU contract

The implemented [CPU0074 selected-region exception](#cpu0074-selected-row-region-execution-exception)
qualifies only new exec-v1 F32 row regions here. Other scalar-only, typed-control, F64 exclusion
and exact ordinary SUM/MEAN rules remain unchanged; no general symbolic SIMD/worker admission.

Accepted on 2026-10-24 and implemented by
[Prepare0013](../../planning/modules/prepare/tasks/0013-ordinary-dynamic-batch-cpu-vertical.md#result).
Final independent reviews and MAIN's native-enabled integration gate passed;
MAIN accepted the capability on 2026-10-24. The implemented scope is
homogeneous FLOAT32 ordinary generated scalar, single-range CPU, one named B, fixed features/
parameters, rank-at-most-two boundaries plus needed AD views. Exact symbolic capability answers
must qualify every final forward and actual first-order derivative operation. Native/OpenBLAS,
Metal, mixed-owner, other dtype/family extensions, Vector API, parallel and general Scan support
do not follow. Prepare0015 implements only the bounded additional occurrences below.

Prepare0015's bounded outer prerequisite is implemented; owning validation and independent
code/structural review passed, with root validation and the sole bounded probe completed (evidence and limits:
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) );
final MAIN acceptance remains pending.
Retain the existing ordinary FLOAT32 domain. Added floating occurrences use homogeneous FLOAT32
or FLOAT64, at most one exact named B per value, fixed other extents and checked int-bounded
domains. Rank-at-most-two endpoints and the specific `[T,B,F]`, ROW(1) literal/zero `[F,B]` /
`[T,F,B]` mappings plus their required singleton-feature rank edits qualify, not arbitrary rank-three
algebra. T/F are fixed; named axes, not F=B coincidence, determine mappings. Each ordinary provider
predicate must be explicit in family, dtype, shape/rank/axis, attributes and access. An identical
qualifying occurrence gets the same answer inside or outside Scan closure; no origin tag/registry.

The table bounds the added occurrences; all retain existing Model family semantics and flags.

| Occurrence | Required descriptor/attribute/mapping predicate |
|---|---|
| Seed/zero/one/literal constants | Existing registered typed splat ingress for the bounded shapes above, including ROW(1) constants/zeros; no new constant protocol. |
| ADD and required MUL/DIV | Same floating dtype; exact ordinary tensor broadcast/result shape and no-operation attributes, or existing same-dtype scalar-value attributes with shape preserved. Preserve the identified B coordinate under broadcasting; no other ordinary F64 arithmetic family is added. |
| SUM/MEAN and SUM-to-Shape | Existing no-axis, normalized axis/ordered-axes/keep-dimensions or SUM-to-Shape target attributes; exact family result shape, actual contributor count and proved read/write mapping. Use the typed exact ordinary algorithm below, not Scan's activity reduction. |
| EXPAND, feature-only RESHAPE/EXPAND_DIMS/SQUEEZE | Exact target/axis attributes, ordinary broadcast or checked feature-product equality, singleton insertion/removal only; preserve B without flattening/mixing it into features. |
| Transpose/CONTIGUOUS/same-dtype CAST | Valid exact permutation/result shape, no-operation attributes for CONTIGUOUS with shape unchanged, or exact same-dtype CAST target; require a proved canonical/transpose/expand mapping for these bounded shapes and preserve raw semantics. |
| Length CAST control producer | Only non-gradient FLOAT32 `[B]` or FLOAT64 `[B]` to non-gradient INT64 `[B]`, exact CAST target and unchanged shape, under existing cast semantics. No general integral arithmetic/cast family; direct INT64 lengths requires no producer. |

Added cases use checked-segment carriers with exact per-value four/eight-byte spans/alignment,
proved zero-offset canonical/transpose/expand reads and canonical injective writes; preserve existing
qualified read views and reject unresolved/unproved mappings, unsupported carriers and writable
overlap. This qualification requires no new heap-carrier support. Validate fixed-feature products
even behind empty T/B, then actual counts/strides/spans/bytes/aggregate bounds before borrow/allocation;
recheck physical access at binding. The separate eight-byte INT64 control boundary does not make
body arithmetic mixed-dtype. Ordinary symbolic F64 MATMUL, arbitrary rank-three operations and
all-valid symbolic F64 Scan remain unqualified. Scan-internal MATMUL retains its own typed emitter.

The four implemented public nested records live only in existing internal packages under
`io.github.pho001.synaptik.backend.cpu`; declaration excerpts (not runnable Java):

```java
// internal.memory.CpuRepresentationRecipes
public record BatchByteSize(long constantBytes, long bytesPerBatch, int elementByteWidth)
        implements PreparedMemoryPlan.ByteSize { /* fixedByteSize(), byteSize(B) */ }
// internal.ir.CpuAccessPlan
public record BatchBinding(CpuAccessPlan plan, List<Long> extentConstants,
        List<Long> extentBatchFactors, long baseElementOffset,
        List<Long> strideConstants, List<Long> strideBatchFactors) {
    public CpuAccessPlan.Binding bind(long batchSize);
}
// internal.lowering.CpuMatmulLowering
public record BatchGeometry(DataType dataType, CpuAccessPlan.BatchBinding left,
        CpuAccessPlan.BatchBinding right, CpuAccessPlan.BatchBinding result) {
    public CpuMatmulLowering.Geometry bind(long batchSize);
}
// internal.lowering.CpuAggregateLowering
public record BatchGeometry(CpuAggregateIr.Kind kind, DataType dataType,
        CpuAggregateIr.Form form, List<Integer> axes, boolean keepDimensions,
        CpuAccessPlan.BatchBinding input, CpuAccessPlan.BatchBinding output)
        implements PreparedMemoryPlan.ByteSize {
    public CpuAggregateLowering.Geometry bind(long batchSize);
    public OptionalLong fixedByteSize(); // empty: structural run-bound marker
    public long byteSize(long batchSize); // bind(B).workspaceBytes(1)
}
```

References/results are non-null, list membership snapshots reject nulls in encounter order;
null/invalid/overflow/index failures use the corresponding exceptions in the
[resource contract](runtime-prepare-engine.md#accepted-symbolic-batch-resource-contract).
Recipes are immutable/thread-safe/storage-free; all resolved access/packed arrays belong to one
run. BatchBinding validates equal-rank lists and nonnegative coefficients/base; extent/stride
`constant+factor*B` is in elements with checked counts/spans. One B-bearing axis per represented
shape and proved canonical/transpose/expand mappings only. Preserve exact Model empty strides:
`[F,0]` has `[0,1]`, never max(1,B). Regimes/axis roles stay prepared; no representative fake B.
BatchByteSize computes checked `constantBytes+bytesPerBatch*B`, with nonnegative coefficients,
positive supported element width, divisibility and CPU representability limits. Its run-bound
marker remains absent even when bytes happen to be constant.

MATMUL binds existing Geometry/pack once per invocation: forward M/K/N=B/F/H, dX=B/H/F,
dW=F/B/H, preserving actual transposed strides. Analysis fixes route/carrier/code independently
of B. Unsupported carriers reject, never select another artifact at admission. Aggregate geometry
binds actual domains/output counts and conditional workspace, not maximum banks. For the admitted
FLOAT32/int-limited N, signedBits=278+ceil(log2(N)), cardinality term zero at N<=1, is at most
309 bits: retain constant-five-limb CpuExactSumEmitter for all admitted N, including 0/1.
Single-range scratch is 48 bytes (`8+8*5`) when outputCount>0, otherwise zero. SUM-to-Shape
handles B=1 copy/singleton at entry in the same code/route/slot. Zero scratch does not remove or
discover requirements. Preserve actual count division/numerics; long domains remain unqualified.

For implemented bounded ordinary FLOAT64 SUM/MEAN, retain the existing typed exact
sum/rational-mean algorithm: `signedBits=2099+ceil(log2(N))`, with cardinality term zero for N<=1.
Int-bounded N needs at most 2130 bits, so use fixed 34-limb state and 280-byte (`8+8*34`)
single-range scratch when outputCount>0, otherwise zero while retaining its declared slot/creator.
Preserve subnormals, signed zeros, NaN/Inf, empty domains and exact rational-MEAN rounding with no
FLOAT32 narrowing or FLOAT64 DAZ/FTZ. Retained FLOAT32 stays five limbs/48 bytes, including 0/1;
static FLOAT64 sizing remains domain-sized (N=3: 33 limbs/272 bytes). Scan producing-site SUM
remains its distinct typed sequential algorithm; this ordinary qualification changes no Scan numerics.

CPU analysis proves canonical write injectivity during preparation; run binding checks numeric bounds, never Model
lowerers or element-enumerating injectivity. CPU state-aware representation hooks validate actual
bytes from state.geometry(), retaining type/access/alignment/read-only/overlap checks. Existing
bindCompatible/Invocation keeps direct typed carriers and fresh packed/access arrays; no tenth
type, shared currentB/ThreadLocal, per-element recipe dispatch or hidden generic fallback overhead.

Dynamic identities use `ordinary-batch-v1` to distinguish changed baked meaning in both existing
specialization streams; actual B/bytes/slots/addresses are excluded. Retain static keys/generated
bytes and existing IR numeric formats. Migrate inward creator consumers without deleting useful
static/native paths. No type move/removal, new module, dependency or executor is selected.

The implemented [CPU0073 mechanism](#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion)
permits only its exact internal types, optional region and selected-fusion identity suffixes.
It changes neither this symbolic admission nor exact ordinary SUM/MEAN; noneligible identities
and ClassFiles retain the rules above. Its validation and limits are recorded in CPU0073.

Added outer F64/control/new rank-three cases use `ordinary-batch-typed-v1` in IR and BOTH existing
specialization streams, including exact boundary dtype/carrier, mapping and reduction-state facts;
actual B/L/bytes/slots/addresses remain excluded. Existing FLOAT32 `ordinary-batch-v1`, fixed/static
identities and generated bytes remain identical. Positive/negative provider and validated-generation
tests plus artifact load/reuse/stale-rejection tests must qualify the typed namespace, independently
from combined Scan identities. These tests pass for the implemented typed branch; an identity alone
does not grant any wider family or dtype capability.

Qualification must compile/query genuinely symbolic B and execute actual first-order closure
before general Scan: reuse one compilation/preparation for 64→40→1→64; check actual resources,
descriptors and fixed parameter gradients, B=0/1, constant-byte batch-required mode, mismatch,
overflow, failure/cleanup/retry/concurrency and static regressions. The reference is optimal clean
Java for the same specialized algorithm/hot-loop/dataflow, including five-limb SUM. Inspect generated
Class-Files for hidden allocation/boxing/reflection/per-element binding or dispatch. Bounded warmed
matched-work A/B uses invocation B on both sides and separates cold generation/classload/JIT,
admission and kernel costs. Disclose entry/harness costs when generated MethodHandle entry is
compared with a direct Java call; a measured ratio is not an emitted-inner-loop-only measurement.
Diagnose/justify overhead; no identical-JIT or zero-penalty promise.
Prepare0013 records semantic and Class-File checks against specialized clean Java and bounded
generated/direct and dynamic/static measurements. The original harness measurements precede the
final MATMUL loop repair; one separately authorized private post-fix probe also completed. Short-window
variance, the unmatched medium Engine route policy and a remaining exact-MEAN probe gap are
disclosed, not evidence of steady-state parity, negligible overhead or an all-B speed guarantee.
Final code/Javadoc/performance and shared-document reviews approve with zero open P1/P2.
MAIN's native-enabled architecture/conformance/integration join passed and MAIN accepted the
capability on 2026-10-24; this status update runs no measurement.

### CPU0077 finite-softmax admission parity

The [CPU0077 checkpoint](../../planning/backends/cpu/tasks/0077-f64-finite-softmax-admission-parity.md#result)
records the earlier finite-input correction and its bounded verification. The subsequent
interchangeability repair supersedes its finite-only admission and unchanged-class statement.

Current SOFTMAX/LOG_SOFTMAX bodies preserve the Model's finite, nonfinite and empty-slice rules.
All-finite slices retain maximum-shifted arithmetic and compensation, including represented
negative-infinity shifts caused by finite subtraction overflow. A NaN input makes the complete
slice NaN. With positive infinity but no NaN, positive-infinity positions are NaN; other positions
are positive zero for SOFTMAX and negative infinity for LOG_SOFTMAX. Only slices with negative
infinity but neither NaN nor positive infinity use the literal unshifted Model formula.
Any zero extent gives an empty shape-preserving output with no input read or output write.
Checked geometry, carriers, aliases, thread access and resource ownership remain independently
validated. No finite-only validator is bound before ordinary or Scan execution; producer values
are consumed directly by the numerical body.

CPU-private softmax algorithm version 3 invalidates the former finite-only generated bodies,
including retained composition helpers. It does not change loss/attention formulas, invent a
numerical policy, enable runtime fallback, or unlock BFLOAT16/FLOAT16 numerical occurrences under
the current CPU low-arithmetic restriction.

### CPU0078 temporary low-arithmetic lock

CPU0078 records the historical retain/freeze/lock decision and its unchanged historical
qualification in [ADR0060](../../design/decisions/0060-temporary-cpu-low-arithmetic-admission-lock.md)
and [CPU0078](../../planning/backends/cpu/tasks/0078-temporary-low-arithmetic-lock.md#result).
The subsequent user-approved CPU low-precision parity decision supersedes its admission lock;
its old negative capability answers are not current authority.

### CPU low-precision parity cutover

BFLOAT16, FLOAT16, FLOAT8_E4M3 and FLOAT8_E5M2 numerical execution must reuse the ordinary
CPU capability, lowering, IR, preparation, generated bodies, retained plans and invocation owners.
Remove obsolete CPU0078 admission methods/callers and lock-only tests rather than retaining
an always-true shim or a runtime policy switch. Keep exact geometry, layout, gradient, alias,
resource and lifecycle checks. The Model low-precision and OFP8 contracts own arithmetic and
conversion; restored code is not qualified merely because it was retained behind the lock.

The target is functional parity with existing CPU F32/F64 operation families, including actual
Compiler first-order AD, ordinary/fixed/length-aware Scan, supported symbolic geometry, repeated
training and source/value persistence. FP8 stores bytes, not BOOL or short aliases. Working
arithmetic and accumulators are F32, with one final declared output narrowing. Existing exact
and discrete paths preserve all raw bits. No separate interpreter, helper-dispatch executor,
vendor AD, backend registry, numerical selector or execution-time fallback is authorized.

oneDNN remains a cold-selected CPU realization. Typed native recipes authenticate each input
and output dtype, referenced span, origin, carrier alignment and scratch requirement. Strict F32
working/accumulation is required for low inputs. Native destination conversion is usable only
when it proves Model rounding; otherwise declare an F32 result workspace and perform the exact
final narrowing without erasing a logical output or saved boundary. This is not permission to
promote FP8 inputs into F32 inputs and advertise native FP8 execution.

Primitive and Graph availability are independent. In pinned oneDNN 3.14, Graph MATMUL does not
admit FP8, and CPU native F64 is unavailable. FP8 primitives require a supported x86 target;
unavailable native configurations select the ordinary CPU candidate before execution. No native
FP8 execution is claimed on Arm, no native availability is inferred from dtype enum tags, and no
relaxed/low accumulator mode is used. Existing bare-MATMUL native geometry restrictions remain
until independently qualified; OpenBLAS stays F32/F64-only.

Affected low numerical artifacts and new FP8 recipes need selective append-only compatibility
identities. Preserve unchanged F32/F64 and exact raw recipes, retained ABI and generated bytes.
Qualify representations, ties/overflow/underflow, NaN/infinity/zero signs, contributor and mask
domains, native/ordinary execution, actual AD, resource lifetimes and transaction rollback.
Positive native evidence and unsupported-target rejection evidence are separate claims.

### General Scan ordinary-unit realization

Implemented coordinated Class C realization, accepted with Compiler0017/CPU0081 after independent
authority/interface and exact-source implementation reviews, actual ordinary forward/selected-
backward/typed/activity/boundary/raw consumers and the unfiltered affected test/Javadoc gate.
It realizes the [Scan semantics](recurrent-scan.md#general-ordinary-unit-scan-cutover) through
ordinary owners, without widening cold/wire, Metal Scan, low numerical or universal performance
capabilities. Genuine qualified old recipes retain their exact key/ABI/generated-byte replay.

Use the actual local CompiledScanBody graph to build a local PrepareContext/DAG/requirements and
reuse CpuPartitionDagDecomposer.decompose/unitContext and CpuPartitionLowering.lower. Do not
subset the outer one-node Scan DAG or create a second semantic graph/math inventory. Reuse
ordinary focused family emitters, the same CPU0080 emitMethods owner and existing emitted Scan
time loop. Preserve real typed physical layouts/carriers/cuts and live old/new carry/history.

Resolve inferred computed local descriptors centrally in the existing composition `bodyContext`
through Model `ScanBody.closeOrdinaryLayouts(Operation, List<TensorDescriptor>,
List<TensorDescriptor>)`, shared with flat Compiler closure. Overlay projected values and matching
logical requirements only; keep the captured body and exact activity graph immutable. Analysis
and helper resource preparation use this same projection. Retain strict actual/formal guards,
actual-only symbol authority and existing symbolic numeric mapping without sampled B. Exact
views keep backing offsets/strides and ordinary fusion; only existing materialization decisions
may create canonical cuts. Unproved/negative/empty extraction layouts do not acquire fabricated
storage. Do not copy the Compiler classifier/geometry, import Compiler, or add a fallback resolver.

The reviewed complete-graph source-publication prerequisite reuses the existing CPU source
assembler, geometry and representation owners. Expose one descriptor-level checked size
factory on the existing cold geometry owner; use Model referenced span for static affine/empty
layouts and existing batchCanonical(READ)/BatchBinding proof for sole-named-B sources.
If needed, the one nested numeric adapter
`CpuRepresentationRecipes.BatchLayoutByteSize(CpuAccessPlan.BatchBinding, DataType)` implements
existing Runtime ByteSize through actual binding span and checked width multiplication, with a
structurally empty fixedByteSize marker. It is not a new top-level owner, IR or executor.
Reuse existing caller borrowing/constant initialization and raw representation policy.

Every affected concrete CPU unit/container, including generic, general/fixed recurrent and
OpenBLAS paths, must cold-authenticate genuinely fixed selected geometry inside a complete
batch-required source plan. Keep Runtime defaults strict, actual physical/access checks,
static code shape and qualified keys/bytes/ABIs unchanged. Never make a fixed unit adopt an
unrelated caller B, choose a new route or bypass its typed representation validation.

The following focused additional internal production owners are selected, if needed, rather
than growing a duplicate executor or validator framework:
`internal.lowering.CpuGeneralScanCompositionLowering` for local physical composition,
`internal.codegen.emit.CpuGeneralScanBodyEmitter` for fixed ordinary helper-call orchestration
inside the existing time loop, and `internal.ir.CpuScanControl` for CPU numeric control layout/
code-shaping facts. Private nested physical records remain within existing IR/recipe owners.
No other top-level production type or cross-owner protocol is selected by a task brief alone.

Extend existing family emit/geometry owners for controlled emission, active-contributor hooks
and pack rebind/reset. Controlled ordinary helpers append one MemorySegment control after the
old ABI; composite Scan entries may receive numeric long[][] packs. Standalone/retained entry
descriptors and empty-prefix bytes stay unchanged. Derive local-slot widths from actual JVM
descriptors, forward control through subsidiary calls, and verify the ACTUAL complete method
inventory plus class/code/local/parameter/constant-pool limits for fresh and cached artifacts.
One declaration owner supplies emission and verification, not a second family/method registry.

All unit calls inside the time loop are fixed invokestatic calls. Bind wrappers/packs/cursors
once per invocation and emit owner-defined temporal-base/cursor resets, including retry.
No per-step pack/clone/slice/ofArray, unit-object loop, MethodHandle lookup, BoundInvocation body
children, callback, scheduler or new runtime interpreter. Native control is declared RunState
workspace; simultaneous ranges and numeric/control/scratch storage have proved disjoint lifetime.
Initialize roots from the local compiled plan and this occurrence's own time/row domain. A
retained external Tensor seed remains present; do not recognize numeric zeros or consult another
occurrence's control. This clarifies no new helper ABI, storage or selective identity field.
Shared scoped Model declaration proof precedes analysis but grants no backend capability. Retain
independent physical geometry/resource and actual length-contributor checks; do not duplicate the
symbolic proof or impose length ROW provenance on ordinary all-valid reshape.

Qualify static and one-named-B/fixed-feature geometry without sampled descriptors or actual-B-
baked classes. Per-value widths/alignment and arbitrary proved history ranks/axes remain real;
do not replace obsolete first-carry/rank-three assumptions with another permanent body-family
list. Fuse only compatible structural scopes, physical domains, value-validation boundaries
and optional-merge/temporal-ADD meaning. Existing standalone/native/SIMD/parallel eligibility
does not automatically expand to controlled units; portable semantic support is separate.

Reuse represented SUM and MATMUL active-contributor seams and factor existing scatter traversal/
coordinate/reduction ownership for skip-before-index/load behavior. Factor existing softmax/
loss/BOOL/attention-mask/index/scatter validator loops into nonallocating temporal/activity/range
entries shared with ordinary bind. Complete the current unit's entire eligible validation domain
before its first numerical output/history/scratch write. No duplicated predicates, validator
registry or new normalization/categorical finite policy.

Every nested IR/specialization independently passes CPU0078 before resource/cache/generation/
definition. Control/pack/typed/provenance/lifetime/method-inventory code-shaping facts have a
selective composite identity; absent activity adds no old-route empty suffix. Preserve unchanged
frozen recipe bytes/keys, distinguishing genuinely new compact AD graphs. No Metal Scan, native
step route, numerical-policy selector, low arithmetic unlock or broader cold/wire grant follows.
Actual all-valid/length forward AND selected backward, empty/no-ROW/typed/family/poison/guard/
retry/lifetime/identity evidence and independent source/Javadoc review are mandatory.

## Metal backend

The Java project lives in `modules/backend/metal`; its macOS arm64 native bridge is colocated at
`modules/backend/metal/macos-arm64`, without a nested `native/` directory. The current
[audit-driven completion](#audit-driven-backend-completion) governs the qualified FLOAT32,
BFLOAT16 and FLOAT16 ordinary, fixed-cell first-order and native Scan domains.
Task-era qualification paragraphs retain their explicit checkpoint scope;
they neither certify nor veto subsequent independently qualified occurrences.

The interchangeability repair uses coordinated backend-local identity 44, retaining native ABI7
and schema22/header124. Its authenticated fixed corpus is 190,688 UTF-8 bytes across 23 components,
SHA-256 `f6a560fd381664d01e0ec7f65f2148646d9d9837015133c5571ebadfb2c776fc`.
Physical unary/comparison/numeric-MATMUL reads share original-carrier layout authentication;
ordinary empty outputs and zero-grid fused anchors preserve the family identities without reads
or dispatch. Numeric GeneralScan may map proved strided ingress through ordinary CONTIGUOUS.
Raw F64 Scan is canonical, non-gradient and source-only; no F64 body arithmetic, AD, implicit
conversion, dynamic-B Metal or runtime fallback is authorized.


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
descriptors for all seven model data types may move CPU to Metal or Metal to CPU when their resolved
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
and also supports canonical rank-zero through rank-sixteen targets for all seven carriers. Caller
host ingress preserves exact represented storage bytes; descriptor-aware execution, transfer, and
publication paths validate logical BOOL values without treating storage holes as BOOL elements.

A fully static empty descriptor may be materialized to the host without a layout hint.
There are no source addresses to resolve: `RunResult` supplies canonical zero geometry to the
existing selected-backend copy. Nonempty descriptors with unresolved geometry remain rejected.
This empty publication rule changes neither cross-owner transfer admission above nor lease,
identity, budget, lifetime or selected-backend ownership checks.

The exact movement domain uses existing fixed custom owners and original typed descriptors.
All seven carriers retain qualified reshape/rank edits, affine extraction, canonicalization,
gather, padding, slice update, CONCAT/STACK/TILE and UNFOLD_AXIS. Signed-step SLICE materializes
a canonical destination for all seven carriers; negative logical coordinates do not authorize
negative physical storage strides. Index roles accept INT32/INT64; ONE_HOT publishes canonical BOOL.

CONCAT/STACK no longer have a sixteen-operand limit. Dynamic operand references feed one
disjoint direct-copy child per operand through the shared ordinary/Scan owner, without N-wide
argument binding or an intermediate pairwise tensor chain. Original-carrier views and selected
operand cotangents retain their descriptors. Common rank-sixteen, checked geometry, byte-span,
dispatch and workspace bounds still apply; absence of an operand-sixteen guard is not an
unbounded-resource promise.

FOLD_AXIS accepts the six numeric carriers; image unfold/fold accepts the four floating carriers.
FLOAT32, BFLOAT16 and FLOAT16 folds may add overlapping contributors, using deterministic
output-owned FLOAT32 accumulation in declared order and per-output typed publication. No atomics,
host repair, scratch or software FP64 is introduced. Uncovered cells receive exact carrier zero.
FLOAT64 and integral folds retain their qualified non-overlap boundaries. Java and native
preflight independently validate this domain using original carrier facts, never a FLOAT32 proxy.

Shared Compiler convolution/pooling adjoints reuse unfold, MATMUL, reduction and fold.
FLOAT32/BFLOAT16/FLOAT16 Pool2d/Pool3d overlap preserves existing forward winner/divisor semantics,
and Conv3d training retains groups and selected input/weight/bias roles. These are shared typed
owners, not standalone backward kernels or permission for CPU low arithmetic.

Every logical index is validated in stable node then row-major ordinal order. Replacement
scatter completes bounds and global destination-uniqueness checks before mutation; admitted
arithmetic modes retain colliding contributors in declared order. Public prepared execution
reproduces Model bounds/duplicate exception classes, while the raw ABI reports typed status.
Validation failure preserves inputs and the affected operation's write targets, not already
completed producers. Colliding replacement, unqualified rows and overlapping FLOAT64/integral folds remain false.

Selected affine views retain their exact logical shape, nonnegative read strides and offset.
Preparation/native preflight separately authenticate represented physical storage and complete
referenced spans. Materializing outputs are canonical. Qualified empty source/publication,
EXPAND, normalization and Scan paths detect zero extents before product/stride checks; they
do not bypass semantic guards. Rank zero is one stored element. Unproved layouts/extents,
negative physical strides, unsafe overlap and overflow fail closed.

### Accepted Metal dense-view MATMUL extension

MAIN implements the bounded [Metal0080](../../planning/backends/metal/tasks/0080-dense-view-matmul-composition.md)
extension for first-production consumers. Its identity38 source-matched GPU/package/unchanged
consumer proof predates identity39's FLOAT32 image-fold addition, identity40's pooling-overlap
admission, identity41's audit extension, identity42's typed cutover and identity43's singleton
MATMUL product repair; release acceptance remains separate.

MATMUL input geometry may be exact canonical row-major strides, zero offset and referenced span
equal to logical element count with logical isView true or false. Retain positive fully static
rank 1..16, exact contraction/vector promotion/batch broadcasting and checked byte/dispatch
bounds. Outputs remain canonical. No general strided, offset, overlap or zero-stride operand
storage is admitted; legal MATMUL batch broadcasting is separate from storage broadcasting.

Non-dense transpose still requires an actual earlier local PERMUTE with identity prefix and the
last two axes exchanged, rooted in an authenticated dense value. Every source/descriptor/carrier/
gradient/span relation stays checked. A materialized affine result uses its represented dense
buffer; an already-elided transpose uses its authenticated source plus one transpose application.
Do not follow arbitrary logical ancestry, add elision/copies, fabricate canonical metadata or
change Model/Compiler views. Preserve transpose publication/non-MATMUL fan-out.

Provider, preparer, route planner, independent Java/native validation and MPSGraph value-state
initialization authenticate actual selected descriptors/physical sources. Dense MATERIALIZED_LAYOUT
feeds and local AFFINE_VIEW results are admitted only with this MATMUL proof, not blanket state
admission or clearing physical facts. Generic canonicalStorage and canonical-only anchors remain.

Original homogeneous FLOAT32, BFLOAT16 and FLOAT16 forward/generated-gradient rows require
independent qualification. Keep existing no-gradient integral/promotion/widening bounds; no F64
MATMUL, mixed BF16/F16, F16/F32 widening or integral/widening gradients. Low-containing partitions
remain CUSTOM_PROGRAM only, with no generated-pointwise/anchor fusion or fallback. No Java CPU
low-numeric permission follows.

Metal0080 advanced all six coordinated workload/exact-policy/candidate/compatibility/route/codec
identities from36 to37; Metal0081 advanced them to38. Conv3d training advanced them to39 for
overlapping FLOAT32 image folds; Pool3d completion advances them to40 for shared FLOAT32 pooling
overlap. Stale handoffs/decisions reject even with valid outer integrity.
The checkpoint retained schema20/header124/ABI7/thirteen exports. Native/source/package
authentication remains mandatory under the current protocol below; old packages or historical
v1 capability evidence do not substitute for rebuilt current source and original-typed facts.


Metal-specific optimizer execution belongs to Metal backend prepare/kernels, not to training.

Do not add `MetalOptimizerBridge` to `extensions/training`.

## OpenBLAS provider

`modules/backend/openblas-provider` is a low-level leaf provider.

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
modules/backend/cpu -> modules/backend/openblas-provider
```

Never the reverse.

## Other tool scopes

The performance-evidence section above owns the detailed boundaries for `tools/benchmarks` and
`tools/tuning`. The intended `tools/cli` location has no additional technical rule in the migrated
contract. Root global invariants and dependency rules therefore apply; if CLI work encounters an
architecture-sensitive ownership, dependency, or lifecycle question not answered there, the
missing scope requires an explicit architecture update.

## Profile-free backend capability and identity

<a id="numerical-profile-backend-identity"></a>

A concrete backend must use profile-free, occurrence-specific provider capability and qualified
routes. Plans, specialization, generated artifacts, tuning candidates and decisions, and cache
compatibility retain their non-profile semantic, type, target, ABI, source, and version identities;
changed formats reject stale inputs. CPU's retained routes obey its CPU0078 arithmetic lock;
there is no profile selector.

After the cutover, a new supported occurrence must have an independently true provider query for
its exact dtype, descriptors, attributes, and gradient metadata; a route qualified against Model
semantics and the applicable exact, special-value, finite, and failure conformance tests; and an
updated current ledger and provider/conformance tests. A FLOAT32 true answer never implies a
BFLOAT16 or FLOAT16 true answer. Metal checks original per-role carriers and scalar metadata
without a FLOAT32 descriptor proxy. Each admitted low occurrence requires its qualified shared
custom step under the fixed low-containing CUSTOM_PROGRAM rule. CPU-only arithmetic availability
follows CPU0078 above. Advance the relevant compatibility identity append-only when changed
route, program, plan, serialized, or generated behavior makes old inputs incompatible; reject
stale inputs. Capability evidence and route qualification remain cold backend/Prepare concerns,
not a Runtime certificate, tolerance policy, retry, or fallback layer.

Current generated CPU artifact envelopes use schema 71. Current Metal images use schema 22
(`SM22`) with a 124-byte header, and coordinated workload, exact-policy, candidate, compatibility,
route and codec identities use version 43. Native ABI7 retains thirteen exports. CPU schema68
and Metal schema19/identity29 are historical pre-cutover baselines. Metal identities30–38 record
prior profile-free, elementary, boundary and view repairs;39 added FLOAT32 image folds,40 shared
pooling overlap,41 the joined ordinary/native-Scan audit extension,42 shared original-typed
FLOAT32/BFLOAT16/FLOAT16 execution and dynamic composition references, and43 the singleton
MATMUL product repair without changing K>1 arithmetic. Stale identities reject.
CPU schema 70
corrected activation extremes and named low FLOAT32 working sites; those repairs changed neither
Model semantics nor provider admission. CPU schema 71 retains those repairs and adds the bounded
FLOAT32 pointwise general-scan family. CPU0031's distinct shaped specialization retains schema 71
and old pointwise compatibility. CPU0032's MATMUL-only specialization preserves both old scan
forms within schema 71; baked shaped/MATMUL facts enter both existing specialization identity
streams, not the geometry-free canonical family envelope. CPU0033's additive FLOAT64 scan variants
retain those projections/payloads and accepted FLOAT32 keys/bytes; existing dtype and baked
resource facts distinguish their identities. CPU0034 advanced the staged payloads to v2;
CPU0035 advanced them to v3; CPU0036's current staged specialization payloads are v4 in both
existing streams for top-tested loop control and dominating final-copy bases. The global
envelope/family stays 71 and canonical shaped/MATMUL projections stay geometry-free v1.
Staged v1/v2/v3 artifacts are incompatible, while pointwise, fixed
and unrelated identities are unchanged. Current identities reject stale inputs, including Metal
identity 38 and CPU schema 70 and earlier. Existing CPU family projections and historical
versioned evidence remain distinct from this current global envelope identity.
Native source, manifest, and package authentication remain safety gates; historical schema-19
source certificates do not certify current schema-22 programs.

Metal's current exact occurrence domain contains exact unary operations; all 49
ordered casts; FLOAT64/FLOAT32/BFLOAT16/FLOAT16 classification; scalar and right-aligned BOOL
logic; the fourteen supported promoted floating `WHERE` signatures; exact seven-carrier
affine/index/replacement/movement rows; image windows and the fold domain specified above;
unsigned-32-bit-bounded ordering/top-K/numeric arg-extrema; exact maximum Pool2d/Pool3d;
strict comparisons; and zero-input raw INT64[2] `INITIAL_STATE`. Direct BFLOAT16/FLOAT16 mixed-low
operations, including the two remaining promoted WHERE signatures, require explicit casts through
FLOAT32 and have no mixed-low kernel. Floating-to-floating casts preserve legal gradient metadata.
Float-to-integral casts may consume a differentiable input but produce a non-differentiable result.
`WHERE` differentiability is the exact branch-role OR and never includes its condition.

The CPU-free generated-gradient closure includes floating casts, inverse affine movement,
signed-step materializing extraction and selected updates, PAD, CONCAT/STACK/TILE, exact
replacement and admitted arithmetic scatter, broadcast reductions, ordinary arithmetic/
statistical/normalization/loss/attention formulas, saved-mask dropout and qualified window
adjoints. Saved condition/index/mask/forward roles remain live until consumed. FLOAT32,
BFLOAT16 and FLOAT16 use original typed roles through these shared owners; qualification is
still occurrence-specific, not inferred from a FLOAT32 proxy. Conv3d/overlapping Pool3d,
general Scan and fixed RNN/GRU/LSTM have actual selected-first-order qualification in all three
types on a Metal-only Engine. Fixed-cell rules cover selected floating input/state/weight/bias
roles, not higher-order or arbitrary geometry. Entire generated topology must be eligible;
unsupported geometry, dtype and order remain false.

The current provider also admits no-gradient INT32/INT64 MATMUL pairs with INT64-dominant
promotion and modular result arithmetic. For the frozen supported ACCELERATOR FLOAT32 occurrences,
Metal's independently queried homogeneous BFLOAT16 and FLOAT16 counterparts remain true. This
covers the qualified tensor/scalar arithmetic, comparisons, extrema, reductions, cumulative
operations, positive-static MATMUL geometries, normalization, losses, attention, convolution,
pooling, explicit-state dropout and recurrent domains. Each low input is decoded exactly,
every numerical working value and accumulator is FLOAT32, and each observable low result is
narrowed once at its declared publication boundary with round-to-nearest, ties-to-even.
INT32/INT64 cumulative operations retain exact modular arithmetic, not floating emulation.
Conv1d/Pool1d may use only authenticated local singleton-height views. This paragraph does not
widen unlisted or false provider occurrences.

### Historical EXP/SIGMOID qualification checkpoints

These checkpoints establish the separately preserved low domains and historical evidence.
The later audit below supersedes earlier FLOAT32 scalar/gradient exclusions, not low restrictions.

At the Task-0074 checkpoint (2026-10-09), the Metal provider admitted `EXP` only for one canonical,
fully static, positive-rank `1..16` FLOAT32 input and equal-Shape output with no gradients, no
attributes, and checked four-byte span and dispatch geometry. Prepare selected its fixed direct
MPSGraph exponent route. At that checkpoint, independently queried BFLOAT16 and FLOAT16 `EXP`
answers were false, as were FLOAT64, gradient-bearing, scalar-rank, noncanonical, and over-limit
answers. The historical v1 `ACCELERATOR` EXP answer stays false; at that checkpoint v2 recorded
FLOAT32 true and both low answers false.

Task 0075 independently qualifies homogeneous BFLOAT16 and FLOAT16 `EXP` in that bounded canonical,
positive-static-rank `1..16`, no-gradient, no-attribute, same-type/same-Shape domain with checked
two-byte span and dispatch geometry. Current v2 records true for both low types while historical
v1 stays false. Each low type has a distinct typed custom kernel step, evaluates in FLOAT32 working
values, and narrows once to its original low type under the Model's `EXP` special-value and
arithmetic-site DAZ/FTZ rules. The current FLOAT32 occurrence retains its direct MPSGraph route.
FLOAT64, gradient-bearing, scalar-rank, noncanonical, mixed-low, and over-limit `EXP` occurrences
remain false. A low-valued `EXP` step is custom, never MPSGraph.
An explicit low-to-FLOAT32 cast followed by FLOAT32 `EXP` may separately use an internal MPSGraph
boundary step in the enclosing low-containing `CUSTOM_PROGRAM` route; the cast does not itself
grant low-valued `EXP` support or change the partition candidate set.

Task 0076 admits one canonical fully static positive-rank `1..16` FLOAT32 `SIGMOID` input and
equal-Shape output with no attributes, no gradients, and checked four-byte span and dispatch
geometry. Wire 64 uses a fixed composed MPSGraph route: a predicate over reinterpreted stored
FLOAT32 bits selects the Model's sign branch before its `EXP`, typed-one addition, and division.
The guard classifies signed zero and subnormals without floating-comparison DAZ. Historical v1
stays false; at that checkpoint v2 recorded only FLOAT32 true. BFLOAT16, FLOAT16, FLOAT64,
gradient-bearing, scalar-rank, noncanonical, and over-limit SIGMOID occurrences were false. A
FLOAT32 SIGMOID after an explicit low-to-FLOAT32 cast may run as an internal MPSGraph step in a
low-containing `CUSTOM_PROGRAM` partition; that composition alone did not grant a low-valued
SIGMOID or second partition route.

Task 0077 independently admits homogeneous BFLOAT16 and FLOAT16 `SIGMOID` in the same bounded
canonical, positive-static-rank `1..16`, no-gradient, no-attribute, same-type/same-Shape domain,
with checked two-byte spans and dispatch geometry. Current v2 records true for both low types;
historical v1 remains false. The two typed custom steps use the exact stored low sign guard,
one FLOAT32 `EXP` on the selected branch input, branch-selected numerator, FLOAT32 addition and
division, and one final ties-to-even narrowing to the original low type. The FLOAT32 wire-64
step remains composed MPSGraph, including after an explicit low-to-FLOAT32 cast inside a custom
partition. A low-valued wire-64 step is always custom, never MPSGraph. FLOAT64,
gradient-bearing, scalar-rank, noncanonical, mixed-low, and over-limit `SIGMOID` remain false.

Every partition containing BFLOAT16 or FLOAT16 values selects the fixed `CUSTOM_PROGRAM`
whole-partition route. This includes homogeneous no-gradient raw-preserving `RESHAPE`, simple
`PERMUTE`, materializing `CONTIGUOUS`, `SLICE`, `CONCAT`, and `TILE` schema-20 images.
`CUSTOM_PROGRAM` is their only candidate, and the selected route is immutable before shared
declarations escape analysis. Existing eligible FLOAT32 occurrences retain their qualified direct
or composed routes, including direct rank-two FLOAT32 MATMUL. Any FLOAT32-only node, including one
on an independent branch or following an explicit low-to-FLOAT32 cast, may therefore use an
internal MPSGraph boundary step in the same custom partition; no BFLOAT16/FLOAT16-valued node may
do so, and no hidden BFLOAT16-to-FLOAT16 substitution is allowed. Generated-pointwise and
anchor-epilogue fusion remain disabled for the whole low-containing partition. The shared custom
route uses fixed reviewed raw-word/integer/movement/predicate kernels behind one whole-program invocation with
declared assigned buffers for every logical value and no host repair.

The Task-0069 specialized FLOAT32 L1 at wire `114`, ScatterAdd at wire `70`, and singleton
VARIANCE at wire `112` select `CUSTOM_PROGRAM` under `CA_0069`; Java and native creation reject
their direct MPSGraph routes. These FLOAT32 kernel restrictions do not remove the independently
admitted homogeneous BFLOAT16/FLOAT16 occurrences, which use the general low custom program and
its `lp_l1_norm`, `lp_scatter_add`, and `lp_variance` operations under their own occurrence checks.
The specialized FLOAT32 L1 slice requires one canonical positive-static rank-one no-gradient input,
ordered multi-axis `[0]`, a canonical scalar or retained `[1]` output, unsigned-32-bit bounds, one
output thread, and distinct buffers. It raw-clears every contributor sign and performs exactly
`N-1` source-ordered safe additions; `N=1` performs none.

The specialized FLOAT32 ScatterAdd slice requires canonical rank-one base/update/output with positive
static extents, axis zero, no gradient flags, and a materialized canonical rank-one INT32/INT64
index partition feed. The compiler places explicit `CONTIGUOUS` between its generated zero-base
`EXPAND` and Scatter. A complete CPU index scan precedes every command encoding and mutation. One
target thread raw-loads the base once, keeps every matching duplicate in source order, and stores
once; unaddressed targets copy the raw base, and there are no atomics or races. This exact node
closes the existing rank-one Gather data cotangent without admitting gradient-bearing ScatterAdd
or higher-order ownership.

The specialized FLOAT32 VARIANCE slice requires canonical rank-one `[1]` input, axes exactly `[0]`,
correction zero, no gradient flags, distinct buffers, and a canonical scalar or retained `[1]`
output. One thread evaluates exactly `DIV(x,+1)`, `SUB(x,mean)`, `MUL(difference,difference)`, and
`DIV(square,+1)`, then stores the sole canonical cell once. The task-local Lean proof covers every
DAZ/FTZ choice: all finite inputs yield exact positive zero, while NaN and either infinity sign
yield NaN class. There is no classifier, aggregate add, reciprocal, FMA, clamp, fallback,
retry, timing, or alternate publication.

The retained shared Lean proof, source-bound declaration, and pinned Xcode-27 compiled-MSL/AIR
audit remain source and compiled-code review evidence for the three bounded implementations, not a
runtime numerical certificate or a substitute for post-cutover route conformance.

Direct CPU/Metal transfer supports all seven current data types at ranks `0..16` over canonical or
resolved positive-stride non-overlapping storage layouts; BOOL validation visits logical elements
only. Local selected publication gathers logical elements from authenticated physical storage and
preserves exact target-relative offsets and holes without exposing aliases.

The pre-cutover package uses ABI 7 with thirteen exports. Node schema 19 is the frozen bounded self-describing,
route-bearing image over type wires `1..7`, operation wires `1..115`, attribute wires `0..41`,
route wires `1..3` and complete optional storage-layout geometry. The changed header/decoder
format receives an append-only schema identity and rejects profile-bearing stale images.
Its fixed 128-byte `SM19` header authenticates the core image and, only for `CUSTOM_PROGRAM`, the
canonical step/member/binding/materialized-value/typed-instruction/manifest extension. Every core
and extension section is adjacent; references are never padded before 64-bit attributes. The
manifest carries only typed execution records and integer source byte counts; Java neither emits
Metal source nor owns fixed, generated, or assembled source hashes. MPSGraph images set every
extension count and flag to zero and physically omit the extension. Custom execution binds compact
materialized slots rather than one slot per logical value.

Schema-2 pointwise fusion may replace each maximal linear chain of eligible canonical FLOAT32
`FLOOR`, `CEIL`, `SIGN`, and `RELU` nodes with deterministic generated units of length `2..8`;
a remainder of one is prevented by taking seven from the preceding unit. Inputs and outputs have
positive fully static rank `1..16` and element count `1..UINT32_MAX`. Generation is bounded by 32
units, 256 instructions, 16,384 bytes per function, 262,144 generated bytes, and 1,048,576 total
source bytes. Native reconstructs exact cap precedence and the first rejected node. The production
emitter first traverses the structured sink in no-allocation count mode, validates all five caps,
allocates the exact byte count, and traverses the same emitter once for compiler input. The exact
fixed corpus is 84,593 UTF-8 bytes: native SHA-256-checks each of its ten ordered components and
their total and independently authenticates the assembled source before compiler entry.

The versioned program image admits the existing qualified typed anchor-epilogue instructions. An eligible MATMUL
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

At the frozen cutover baseline, structural execution covered 101 kinds with 14 nonexecutable and
production capability covered 86 kinds with 29 false. The FLOAT32 `EXP` addition raised
those counts to 102/13 structural and 87/28 production; the later BFLOAT16/FLOAT16 `EXP` additions
do not change kind counts. FLOAT32 SIGMOID raised that checkpoint to103/12 structural and88/27
production; these historical counts do not describe the current live inventory. Workload,
exact-policy, candidate, compatibility, route-policy, and codec identities were version 29 before
cutover, version 30 at cutover, version 31 for FLOAT32 `EXP`, version 32 for low `EXP`, version 33
for FLOAT32 `SIGMOID`, version 34 for low `SIGMOID`, and version 35 for the native-boundary and
exact low-value repairs. Version36 corrected standalone scalar admission and37 added dense-view
MATMUL;38 added selected materializing RESHAPE;39 added overlapping FLOAT32 image folds;
40 added shared FLOAT32 pooling overlap;41/schema21 added the ordinary/native-Scan audit;
42/schema22 added shared typed execution;43/schema22 preserves singleton MATMUL product semantics.
Stale versions fail closed. The complete-plan wrapper retains its unrelated version-one format.

## Accepted ordinary materializing RESHAPE extension

[ADR0062](../../design/decisions/0062-ordinary-materializing-reshape-closure.md) is implemented through
Complete Compiler0019 final geometry and Complete CPU0079 existing affine ordinal copy, without
unlocking low arithmetic. Integrated Metal0081 source has actual source-matched package/device
and unchanged typed-backward consumer proof; final current-ledger/docs/release acceptance remains
separate. Ordinary validated static RESHAPE selects canonical non-view geometry only when its
supported inferred view remains unproved. Raw source/wire inference stays unchanged.

Metal0081 independently retains the exact AFFINE_VIEW branch and admits the selected materializing
branch only with canonical non-view output/CANONICAL state, offset0, span=count and
densePhysical=false. Source/output offset/span equality remains mandatory for actual views,
not for compact materialization of a valid offset/holed source. Neither source flags nor actual
physical geometry may be erased. Materialization receives its ordinary distinct destination and
lifetime, not an alias or new elision.

Reuse the selected exact_move_unary_0066 operation6 kernel and existing metadata/copy helpers,
with the representation actually bound. Preserve existing movement dtype/gradient/storage bounds,
fan-out/publication, runtime alias/size/context checks and CUSTOM_PROGRAM-only low partitions.
No new shader, numeric math, consumer CONTIGUOUS or older0059-kernel substitution is authorized.
Schema20/header124/ABI7/thirteen exports remain; all six coordinated Metal identities advance
37 to38 and valid stale37 evidence rejects. This explicitly extends the former noncontiguous-
RESHAPE exclusion; it does not broaden MATMUL storage or other affine operations.

## Audit-driven backend completion

The user-requested CPU/Metal operation audit implements the following bounded extensions.
Every occurrence still requires complete independent logical, physical, native and selected-
adjoint qualification. The [production ledger](../../backend-guide/production-support.md)
records actual numerical/device evidence separately from final candidate acceptance.

CPU reuses its existing typed IR, conversions, exact comparisons and movement/layout owners for
Model-promoted pointwise operands, signed-step materializing slices and their existing
pullbacks, variadic composition beyond the old sixteen-input guard, and sound larger injective
output proofs. Signed logical coordinates do not authorize negative physical storage strides.
Preserve checked resource/classfile limits, exact stored-value comparisons, promoted computation
and original cotangent carrier conversion. CPU0078's low-computation lock and independent
ordinary named-B predicates remain unchanged.

Metal implements the qualified FLOAT32/BFLOAT16/FLOAT16 elementary/activation and
statistical/masked-reduction signatures, affine/scalar reduction reads, colliding arithmetic
scatter and Gather adjoints, softmax/normalization/loss/attention, spatial operations and fixed
recurrent forward/selected first-order rules through existing Model/Compiler equations and
shared prepared Metal owners. The separate low-precision library is retired; common typed
arithmetic and numeric codecs retain each original role's storage/POD type and each output's
publication boundary. There is no FLOAT32 tensor proxy or duplicated low implementation.
Exact original-typed FLOAT64/integral operations use represented semantics without software
floating arithmetic; INT32/INT64 cumulative arithmetic remains exactly modular. Original
descriptors determine promotion, optional operands, gradient bits and canonical siblings.
Cancellation, signed zero, nonfinite classes, exclusions and named working sites follow Model,
not old structural recipes. Fixed-cell and explicit GeneralScan ordinary-cell rounding remain
distinct. No CPU low arithmetic, FP64 arithmetic/emulation or CPU fallback is enabled.

General Metal Scan uses an authenticated bounded native body/binder plan inside the existing
executable owner. It reuses neutral compiled-body/activity/demand transport and once-prepared
ordinary typed Metal steps in a native time loop, preserving traversal, carry/history bindings,
original demanded output roles, structural inactivity, pre-write validation, empty identities
and actual reverse-body semantics. FLOAT32/BFLOAT16/FLOAT16 explicit ordinary-cell bodies use
that same owner, not a fixed-cell compatibility fallback. There is no Runtime graph interpreter,
Java child loop, T-fold unrolling, copied CPU evaluator or CPU fallback. Symbolic/cold/order
admission is separate.

Cold reconstruction accepts Model-legal per-role floating descriptors and casts in the existing
recipe domain. Public checkpoint wire versions and strict descriptor rebind remain unchanged;
this does not admit arbitrary PURE bodies or optimizer/history/compiled-plan continuation.

Ordinary operation/attribute identities are reused where their grammar suffices. Scan added wire
116/attribute42 at schema21/identity41; current schema22/`SM22`/header124/identity43 retains the
shared typed execution and dynamic composition-reference grammar. Every caller uses that grammar,
shader/component/source authentication is renewed and stale inputs reject. ABI7/thirteen exports
remain. Metal general FLOAT64 numerical arithmetic, unsupported mixed-low domains, CPU low
arithmetic and unqualified higher-order recurrent derivatives remain excluded. Source-matched
device execution and independent source review precede production-support claims; current
component evidence is not final full-catalog or sealed-release qualification.
