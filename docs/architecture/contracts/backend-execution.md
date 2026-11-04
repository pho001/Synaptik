# [Backend](../../glossary.md#backend) execution contract

> This scoped contract is incorporated by the [authoritative architecture root](../../../ARCHITECTURE.md).
> It is normative only within the scope stated below and has no independent authority. Root global
> invariants and dependency rules apply everywhere and take precedence. Any overlap, contradiction,
> missing applicable scope, or ambiguity requires an explicit architecture update; do not choose
> between contracts silently.<span data-glossary-navigation="true"> ([normative glossary](../../glossary.md#term-normative); [scope glossary](../../glossary.md#term-scope); [dependency glossary](../../glossary.md#term-dependency))</span>

## [Scope](../../glossary.md#term-scope)

This contract owns [concrete backend](../../glossary.md#concrete-backend) responsibilities, performance-evidence and optimization-tool
boundaries, [CPU](../../glossary.md#term-cpu) routes and generated-code placement, [Metal](../../glossary.md#term-metal) [ownership](../../glossary.md#term-ownership-meanings), and the [OpenBLAS](../../glossary.md#term-openblas) provider.
It does not own [backend](../../glossary.md#backend)-neutral [Planning](../../glossary.md#term-planning) ownership, shared [Prepare](../../glossary.md#prepare) orchestration, or [Runtime](../../glossary.md#term-runtime) policy.

## [Concrete backend](../../glossary.md#concrete-backend) [modules](../../glossary.md#term-module)

Concrete backend modules own concrete backend implementation.

Examples:

```text
modules/backend/cpu
modules/backend/metal
modules/backend/cuda
```

These physical directories retain the separate `:backends:*` [Gradle](../../glossary.md#term-gradle) project [identities](../../glossary.md#term-identity-meanings) through
explicit `projectDir` mappings, including `:backends` at `modules/backend`. The physical parent is
not the [backend](../../glossary.md#backend)-neutral `modules/backend-contract` module and grants no new [dependency edge](../../glossary.md#term-dependency-edge).

Allowed:

- capability provider
- backend-owned [prepare](../../glossary.md#prepare)/[lowering](../../glossary.md#lowering)
- backend-specific [fusion](../../glossary.md#term-fusion)
- backend-specific [specialization](../../glossary.md#term-specialization)
- [kernel](../../glossary.md#kernel) route selection
- executable units
- backend storage
- backend workspace
- backend [trace](../../glossary.md#trace) contribution
- [native bridge](../../glossary.md#term-native-bridge) integration
- typed, version-controlled, tested candidate generators colocated with the routes they configure
- compatible workload-[cache](../../glossary.md#term-cache) [lookup](../../glossary.md#term-lookup) during preparation
- safe heuristic selection when model autotuning is disabled or compatible cache entries are
  absent<span data-glossary-navigation="true"> ([heuristic glossary](../../glossary.md#term-heuristic); [model autotuning glossary](../../glossary.md#autotuning--model-autotuning))</span>
- deterministic [partition](../../glossary.md#partition) analysis and exact shared [buffer](../../glossary.md#term-buffer)/workspace requirement declaration
- finalization of an analyzed partition against shared assigned slots

Forbidden:

- public [Tensor](../../glossary.md#tensor) [API](../../glossary.md#term-application-programming-interface-api) [ownership](../../glossary.md#term-ownership-meanings)
- global [graph](../../glossary.md#graph) [compiler](../../glossary.md#term-compiler) logic
- [engine](../../glossary.md#term-engine) [dependency](../../glossary.md#term-dependency)
- [service locator](../../glossary.md#term-service-locator) ownership
- [runtime](../../glossary.md#term-runtime) [plugin](../../glossary.md#term-plugin) discovery ownership
- changing module ownership rules

Concrete backend modules may depend on:

- [model](../../glossary.md#term-model)
- config
- [planning](../../glossary.md#term-planning)
- runtime
- prepare
- backend-contract
- trace

Concrete backend modules must not depend on engine.

For accepted [general scan](../../glossary.md#general-differentiable-scan), provider support must qualify the complete body and its loop binder,
not infer support from child operations alone. One backend owns the whole scan; global [AD](../../glossary.md#autograd) remains
Compiler-owned. CPU0030 implements bounded [FLOAT32](../../glossary.md#term-float32) [pointwise](../../glossary.md#term-pointwise) execution; CPU0031 extends it with
qualified FLOAT32 broadcast, SUM/sumToShape and bounded [shape](../../glossary.md#shape)/[layout](../../glossary.md#layout) closure, including actual
independently qualified reverse bodies. CPU0032 adds static [homogeneous](../../glossary.md#term-homogeneous) FLOAT32 [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../glossary.md#matrix-multiplication--matmul)
and independently qualified actual reverse bodies, as described under
[CPU backend routes](#cpu-backend-routes). CPU0033 implements homogeneous [FLOAT64](../../glossary.md#term-float64) across the
complete bounded declared algebra and actual first-order reverse closure under the same
whole-body/binder [qualification](../../glossary.md#term-qualification). The
[scan contract](recurrent-scan.md#general-differentiable-scan) solely owns scan semantics and gates.

Each concrete backend owns the complete typed configuration vocabulary and [candidate generator](../../glossary.md#candidate-generator)
for each route it implements. A generator derives and prunes complete valid configurations from
[target](../../glossary.md#term-target) capabilities, [canonical](../../glossary.md#term-canonical-form) workload facts, and the tuning budget. [CPU](../../glossary.md#term-cpu) [matrix](../../glossary.md#term-matrix-meanings)-multiplication
candidates may, for example, include supported [Vector API](../../glossary.md#term-vector-api) species and strategy, unroll, tile,
parallelism, or [OpenBLAS](../../glossary.md#term-openblas) thread configurations. [Vector](../../glossary.md#term-vector), [scalar](../../glossary.md#term-scalar), and OpenBLAS choices are distinct
route-specific typed configurations, not booleans or entries in a generic [parameter](../../glossary.md#term-parameter) bag.

[Operation](../../glossary.md#operation) family selects the appropriate backend candidate generator; it is not a [cache key](../../glossary.md#term-cache-key) for
one universal family-wide configuration. Local tuning measurements use a [canonical workload
signature](../../glossary.md#canonical-workload-signature) that includes semantics and attributes, data types, shapes, layouts, relevant policies,
and target compatibility. Identical [signatures](../../glossary.md#term-signature) can reuse one result across occurrences and
models. Physical vector [lanes](../../glossary.md#term-lane) remain constrained by hardware and supported [JDK](../../glossary.md#term-java-development-kit-jdk) Vector API
species; candidate generation must not promise arbitrary lane counts.

Backend candidate discovery must not use `Map<String,Object>`, string [dispatch](../../glossary.md#term-dispatch), reflective
annotations, a central knob [registry](../../glossary.md#term-registry), or a generic configuration language. Shared preparation and
tuning orchestration sees candidates opaquely and does not interpret private backend fields.

## Performance evidence and optimization tooling

`tools/benchmarks` owns observational, report-oriented [benchmarking](../../glossary.md#term-benchmarking). It runs fixed reproducible
[operation](../../glossary.md#operation), operation-family, [model](../../glossary.md#term-model), and end-to-end workloads to compare [commits](../../glossary.md#commit-release-distribution-log-and-seed-meanings), models, or
environments. A benchmark produces measurement evidence and reports only. It must not select or
mutate production settings.

`tools/tuning` owns one explicit model-[autotuning](../../glossary.md#autotuning--model-autotuning) workflow with two coordinated phases. First, it
extracts actual tunable workloads and routes from the model, forms [canonical](../../glossary.md#term-canonical-form) workload [signatures](../../glossary.md#term-signature),
deduplicates identical signatures while retaining [occurrence](../../glossary.md#term-occurrence) [weight](../../glossary.md#term-weight) and context, reuses compatible
entries from an explicit persistent workload [cache](../../glossary.md#term-cache), and measures only cache misses. Second, it
measures a bounded set of complete valid [graph](../../glossary.md#graph), [fusion](../../glossary.md#term-fusion), [ownership](../../glossary.md#term-ownership-meanings)/[partition](../../glossary.md#partition), [layout](../../glossary.md#layout),
[materialization](../../glossary.md#materialization), route, and configuration candidates end to end and selects an explicit prepared
plan or [artifact](../../glossary.md#term-artifact). This second phase does not repeat local route-[parameter](../../glossary.md#term-parameter) search.

[Compiler](../../glossary.md#term-compiler), [planning](../../glossary.md#term-planning), [prepare](../../glossary.md#prepare), and concrete [backends](../../glossary.md#backend) generate the candidates for decisions they
own. Tuning tooling coordinates measurement and selection without taking over graph semantics,
transformations, ownership rules, [lowering](../../glossary.md#lowering), route logic, or private backend vocabulary. The model
author supplies the model, representative input or [shape](../../glossary.md#shape) profiles, [objective](../../glossary.md#term-objective), budget, constraints,
and explicit cache locations; backend authors define backend candidates. Running this same
workflow over a representative model corpus may pre-[seed](../../glossary.md#term-seed) the same workload cache for a [target](../../glossary.md#term-target),
but there is no separate platform-calibration subsystem, workflow, or profile.

`modules/config` may store [immutable](../../glossary.md#term-immutable) declarative inputs to this workflow after consumers are
stable, but it does not own runners, search algorithms, live discovery, caches, or [mutable](../../glossary.md#term-mutable)
evidence. [Model autotuning](../../glossary.md#autotuning--model-autotuning) is optional for correctness. Cache-only or [heuristic](../../glossary.md#term-heuristic) preparation must
remain safe when it is not requested.

Future tuning artifacts are explicit persistent files: a reusable workload tuning cache and a
model-specific plan cache or prepared-plan record. A load reuses only compatible hits; a miss may
be tuned and atomically persisted. Entries carry explicit [schema](../../glossary.md#term-schema) and backend candidate-schema
versions, target and workload or model fingerprints, objective and constraints, and a measurement
summary. Implementations invalidate incompatible entries and safely reject corrupt data. They do
not use hidden global caches, Java object [serialization](../../glossary.md#term-serialization), or executable [payload](../../glossary.md#artifact-envelope-and-atomic-publication) assumptions. Rich
measurement evidence remains separate from compact caches. Physical file formats and [prepared
executable](../../glossary.md#prepared-executable--preparedexecutable) serialization remain deferred to their backend and [lifecycle](../../glossary.md#lifecycle) [owners](../../glossary.md#term-owner).

[Runtime profiling](../../glossary.md#runtime-profiling) is passive observation of actual execution. `modules/runtime` owns the observed
[execution context](../../glossary.md#term-execution-context) and `modules/trace` owns typed diagnostic DTOs; neither profiling nor tracing
selects settings.

For [Metal](../../glossary.md#term-metal) route [qualification](../../glossary.md#term-qualification), candidate comparison begins only after candidates cover the same
complete occurrence domain and qualify against Model semantics. Exact structure, stored-value,
discrete, cast, raw-movement, special-value, and domain cases require separate [conformance](../../glossary.md#term-conformance);
ordinary [finite](../../glossary.md#term-finite) arithmetic requires explicit test-only backend/route/family/[dtype](../../glossary.md#data-type--datatype)/size oracles,
[metrics](../../glossary.md#machine-learning-and-hyperparameters), thresholds, and [cancellation](../../glossary.md#term-cancellation) boundaries. [NaN](../../glossary.md#term-nan)/[infinity](../../glossary.md#term-infinity), [signed zero](../../glossary.md#term-signed-zero), [subnormal](../../glossary.md#term-subnormal)/[underflow](../../glossary.md#term-underflow),
empty/domain, and fail-before-mutation checks are not finite-[tolerance](../../glossary.md#term-tolerance) cases. Qualification does
not infer provider support or license Compiler rewrites. A single survivor needs no comparative
cost gate. Multiple
survivors may be selected without timing only by strict [structural dominance](../../glossary.md#term-structural-dominance-index-mappings-and-cost-dominance): no more compute
dispatches, no more route-owned temporary bytes, and at least one strict improvement. Supported
observation of actual framework-internal dispatches and temporary resources remains mandatory for
every [opaque](../../glossary.md#term-opaque) route, including [MPSGraph](../../glossary.md#term-mpsgraph); graph-[node](../../glossary.md#node) counts, command-[buffer](../../glossary.md#term-buffer) counts, inferred fusion,
framework estimates, and process memory are not substitutes. An explicit custom route may instead
attest those facts from an auditable source-bound declaration when retained source owns every
command encoder [dispatch](../../glossary.md#term-dispatch) and resource used by the [synchronous](../../glossary.md#term-synchronous) hot invocation.

If multiple [nondominated](../../glossary.md#compiler-transformations-and-dominance) survivors remain, selection stays pending an explicitly authorized,
controlled comparative environment and protocol. Uncontrolled local-[device](../../glossary.md#term-device) timing is diagnostic
only: it never qualifies a candidate, selects a route, changes tuning [identity](../../glossary.md#term-identity-meanings), populates a
production decision, or breaks a structural tie. Qualification fixes an authorized route before
production preparation and adds no [runtime](../../glossary.md#term-runtime) timing, fallback, retry, or workload [matrix](../../glossary.md#term-matrix-meanings).

## Low-[precision](../../glossary.md#term-precision) capability evidence

[Model](../../glossary.md#term-model) floating values/storage/semantics and [Metal](../../glossary.md#term-metal)'s independently qualified low execution remain
active. [CPU](../../glossary.md#term-cpu) numerical [admission](../../glossary.md#term-admission) follows the [low-precision parity cutover](#cpu-low-precision-parity-cutover),
not the historical CPU0078 lock. [Canonical](../../glossary.md#term-canonical-form) ledgers use actual `CpuCapabilityProvider` and
`MetalCapabilityProvider` queries over a stable representative basis. Each row records [backend](../../glossary.md#backend),
[occurrence](../../glossary.md#term-occurrence) [identity](../../glossary.md#term-identity-meanings), kind and attributes, ordered input/output [descriptors](../../glossary.md#term-descriptor), [data type](../../glossary.md#data-type--datatype),
[Shape](../../glossary.md#shape), [layout](../../glossary.md#layout), [gradient](../../glossary.md#term-gradient) [metadata](../../glossary.md#term-metadata), [arity](../../glossary.md#term-arity), and the [boolean](../../glossary.md#data-type--datatype) provider answer. Checked-in supported and
explicit unsupported rows are both [normative](../../glossary.md#term-normative) drift evidence for that basis.

[Target](../../glossary.md#term-target) and exclusion columns are architecture mapping decisions, not provider output. The ledger
does not contain route, [runtime](../../glossary.md#term-runtime), [device](../../glossary.md#term-device), [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions), or generated-backward [ownership](../../glossary.md#term-ownership-meanings), and it does
not change [`BackendCapabilityProvider`](../../glossary.md#backend-capability-provider--backendcapabilityprovider) from a deterministic boolean predicate into an enumeration
[API](../../glossary.md#term-application-programming-interface-api). The representative basis does not replace provider tests over their complete predicate
domains. Canonical manifests use explicit [UTF-8](../../glossary.md#term-utf-8) TSV fields, LF endings, stable sorting, and unique
keys; a duplicate key is rejected after exact row-byte comparison, and [validation](../../glossary.md#term-validation) reports the
first missing, added, or changed row.

The 508-row v1 ledger freezes the pre-cutover [`ACCELERATOR`](../../glossary.md#term-accelerator) and `STRICT_IEEE` provider answers as
historical evidence; its `ACCELERATOR` true and false answers established the profile-free cutover
baseline and must not be rewritten by later capability work. The v2 ledger records actual current
profile-free answers on its representative basis, including independent [homogeneous](../../glossary.md#term-homogeneous) [BFLOAT16](../../glossary.md#term-bfloat16) and
[FLOAT16](../../glossary.md#term-float16) queries and explicit negative predicates. It was initially projected from the cutover
`ACCELERATOR` answers, but is not permanently constrained to equal that projection. Reviewed
capability changes may update v2 and the provider while leaving v1 intact. Both ledgers detect drift in
their respective snapshots; neither enumerates the full provider domain or certifies numerics.
Every supported frozen `ACCELERATOR` [FLOAT32](../../glossary.md#term-float32) Metal occurrence retains its independently queried
homogeneous low counterparts where declared. Existing supported Metal low answers remain true;
CPU's temporary arithmetic lock is the explicit exception to retained CPU availability. Exact
movement, Shape/layout, indexing, selection,
ordering, state, and cast occurrences remain exact substrate. Direct BFLOAT16/FLOAT16 mixed-low
operations are unsupported; an explicit
cast to FLOAT32 is the sole mixed-low boundary. Forward capability remains distinct from
generated-backward ownership.

Every active Metal low family uses the shared typed custom implementation with FLOAT32
working/[accumulator](../../glossary.md#accumulator-type) values and Model-owned exact discrete/raw/cast/[publication](../../glossary.md#term-publication) rules. Original
per-role FLOAT32, BFLOAT16 and FLOAT16 storage and [POD](../../glossary.md#term-pod) types remain explicit; the common
`synaptik_numeric_codec.h` decodes inputs and narrows each observable low output once with
[round-to-nearest, ties-to-even](../../glossary.md#floating-point-precision-and-rounding). The Model's named arithmetic-site [DAZ](../../glossary.md#term-daz)/[FTZ](../../glossary.md#term-ftz) table and family
exact-zero sign and [NaN](../../glossary.md#term-nan) class/domain rules govern it. Fixed-cell working/publication boundaries
remain distinct from an explicit GeneralScan ordinary-cell composition; sharing arithmetic
[owners](../../glossary.md#term-owner) does not make those [rounding](../../glossary.md#term-rounding) schedules interchangeable. [Operation](../../glossary.md#operation)-local [reassociation](../../glossary.md#term-reassociation)/[FMA](../../glossary.md#term-fused-multiply-add-fma)
is permitted only within the qualified implementation; low partitions cannot enter generated-
[pointwise](../../glossary.md#term-pointwise) or anchor-[epilogue](../../glossary.md#term-epilogue) [fusion](../../glossary.md#term-fusion). Public, saved, [fan-out](../../glossary.md#term-fan-out), predicate, index, [mask](../../glossary.md#term-boolean-mask), selection,
and state boundaries remain barriers.

An [opaque](../../glossary.md#term-opaque) or transformed route is eligible only after the same occurrence-scoped exact/special
and [finite](../../glossary.md#term-finite) test [qualification](../../glossary.md#term-qualification) as an explicit route. A route name, examples, or generic `allclose`
alone are insufficient; no runtime proof evaluator or numerical certificate replaces those tests.

Every Metal [partition](../../glossary.md#partition) containing BFLOAT16 or FLOAT16 values selects `CUSTOM_PROGRAM` and exposes
only that candidate. This applies equally to arithmetic and to raw-preserving RESHAPE, simple
PERMUTE, materializing [CONTIGUOUS](../../glossary.md#term-contiguous), [SLICE](../../glossary.md#slice), CONCAT, and TILE. Every operation consuming or producing
a BFLOAT16 or FLOAT16 value uses a custom [kernel](../../glossary.md#kernel) step; no such operation enters [MPSGraph](../../glossary.md#term-mpsgraph), and no
hidden BFLOAT16-to-FLOAT16 substitution is permitted. Any FLOAT32-only operation in the partition
may use an internal MPSGraph boundary step under existing FLOAT32 policy, whether on an independent
FLOAT32 branch or after an explicit low-to-FLOAT32 cast. This does not change the enclosing
`CUSTOM_PROGRAM` route or candidate set. Generated-pointwise and anchor-epilogue fusion are
unavailable throughout the low-containing partition, including its
FLOAT32-only [nodes](../../glossary.md#node). The partition policy is independent of device, OS, [SDK](../../glossary.md#term-sdk), [compiler](../../glossary.md#term-compiler), or
loaded-[binary](../../glossary.md#term-binary-descriptor-and-metadata) identity. MPSGraph is not a low-valued operation or whole-partition route
alternative; classic [MPS](../../glossary.md#term-mps), [MPP](../../glossary.md#term-mpp), CPU, retry, and fallback are not alternatives either.

Low-precision [PREPARE](../../glossary.md#prepare) [trace](../../glossary.md#trace) metadata names the selected custom route and ordered feed-then-target
logical [dtype](../../glossary.md#data-type--datatype) tuple. It has no accumulator or working-type field; the
Model's FLOAT32 arithmetic working/accumulator guarantee is separate from diagnostic metadata.
The [DTO](../../glossary.md#data-transfer-object--dto) contains no candidate, certificate, accuracy, determinism, or environment state and rejects
a [graph](../../glossary.md#graph) selected route.

Metal owns no executable, [compilation](../../glossary.md#compiler-engine-and-execution-stage-names), or prepared-resource [cache](../../glossary.md#term-cache): finalization creates fresh
native resources and its outcome reports `NOT_QUERIED`. The only reusable Metal route [artifacts](../../glossary.md#term-artifact) are
the tuning compatibility, candidate, and decision values. They are explicitly session-scoped, and
the tools cache never persists or reads a `SESSION` value. Their canonical identity contains the
compatibility/candidate/route-policy/workload/[codec](../../glossary.md#term-codec) versions, native [ABI](../../glossary.md#term-application-binary-interface-abi),
per-context random session nonce, versioned program and plan semantics, every descriptor's explicit
dtype wire and Shape/layout/gradient facts, ordered edges, raw [constants](../../glossary.md#term-constant), and the selected candidate
route wire. Complete-plan identity nests the exact phase-one decision. Dtype, ABI, [schema](../../glossary.md#term-schema), route, or
semantic changes therefore change the relevant bytes. Every reopened context receives a different
session nonce, so no decision crosses a context boundary.

CPU generated-artifact cache identity was at generator schema 68 before cutover. Both BFLOAT16
and FLOAT16 use the same Java `short` [carrier](../../glossary.md#term-storage-carrier) where applicable, but the canonical [lowering](../../glossary.md#lowering) [IR](../../glossary.md#term-intermediate-representation-ir),
ordered `boundaryDataTypes`, carrier pattern, execution strategy, generated-class
identity schema, and code-shaping facts participate in the [specialization](../../glossary.md#term-specialization) identity. Consequently
the two raw-16 types have different compatibility bytes and structural keys despite an identical
method descriptor. Persisted envelopes additionally require the current generator schema, exact
structural key, exact compatibility bytes, class-shape validation, and [checksum](../../glossary.md#term-checksum) before reuse.

The pre-cutover allocation appended `DataType.FLOAT16` at ordinal 6 after [`BOOL`](../../glossary.md#data-type--datatype): Metal type wire
7, program schema 19, coordinated Metal candidate/compatibility/route-policy/workload/codec
identity 29, CPU generator schema 68, operation wires 1..115, attribute wires 0..41, route wires
1..3, and native ABI 7. The implementation cutover must append versions for every changed
serialized program, generated artifact, route/candidate/compatibility/workload/codec or tuning
identity, reject stale inputs and cache hits, and retain unaffected wire allocations. Remove the
Metal profile wire atomically in Java and native; review a native ABI bump if an export [signature](../../glossary.md#term-signature)
changes, otherwise retain all thirteen names/signatures. Preserve source/[manifest](../../glossary.md#term-manifest)/package
[authentication](../../glossary.md#term-authentication), decoder/[preflight](../../glossary.md#term-preflight), bounds, ownership, compiled-code checks, and fail-before-
mutation rules. No version number is assigned by this documentation stage.

## [CPU](../../glossary.md#term-cpu) [backend](../../glossary.md#backend) routes

The implemented [CPU0074 exception](#cpu0074-selected-row-region-execution-exception)
qualifies only the selected exec-v1 cases of the [scalar](../../glossary.md#term-scalar)-only/no-per-step-[dispatch](../../glossary.md#term-dispatch) restrictions
below. All other restrictions and implemented-capability statements remain unchanged.

CPU scalar, CPU [Vector API](../../glossary.md#term-vector-api), generated [JVM](../../glossary.md#term-java-virtual-machine-jvm)-[bytecode](../../glossary.md#term-bytecode) CPU computation [kernels](../../glossary.md#kernel), [OpenBLAS](../../glossary.md#term-openblas) and
explicitly configured [oneDNN](../../glossary.md#term-onednn) primitive/[Graph](../../glossary.md#graph) execution are routes inside the CPU backend.

They are not separate backends.

A [generated CPU computation kernel](../../glossary.md#generated-cpu-computation-kernel) is backend-internal executable code whose JVM bytecode is
produced for a selected CPU [lowering](../../glossary.md#lowering) and [specialization](../../glossary.md#term-specialization). CPU backend analysis owns the lowering,
specialization, [fusion](../../glossary.md#term-fusion), route choice, and exact shared-resource declarations. CPU backend
finalization may generate and define the selected [kernel](../../glossary.md#kernel), or reuse it from a CPU-owned compatible
generated-[artifact](../../glossary.md#term-artifact) [cache](../../glossary.md#term-cache), only after shared [Prepare](../../glossary.md#prepare) assigns slots. [Runtime](../../glossary.md#term-runtime) receives the resulting
[prepared executable](../../glossary.md#prepared-executable--preparedexecutable) and neither generates, caches, selects, nor specializes kernels.

This contract does not prescribe a bytecode-generation library or a particular [JDK](../../glossary.md#term-java-development-kit-jdk) builder [API](../../glossary.md#term-application-programming-interface-api).
Changing that implementation mechanism within the CPU backend does not change [module](../../glossary.md#term-module) [ownership](../../glossary.md#term-ownership-meanings),
[dependency direction](../../glossary.md#java-platform-and-build-tools), or [lifecycle](../../glossary.md#lifecycle) placement.

[Planning](../../glossary.md#term-planning) chooses:

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

### [Singleton](../../glossary.md#term-singleton) [MATMUL](../../glossary.md#matrix-multiplication--matmul) correctness cutover

The existing [Model](../../glossary.md#term-model) contract requires K=0 [positive zero](../../glossary.md#nan-infinity-and-signed-zero) and K=1 exactly one typed product,
without an additional [identity](../../glossary.md#term-identity-meanings) ADD or [FMA](../../glossary.md#term-fused-multiply-add-fma). Portable ordinary, ordinary-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters), retained and
ordinary-unit Scan, and selected [scalar](../../glossary.md#term-scalar)/[vector](../../glossary.md#term-vector) row regions preserve that singleton result.
K>1 keeps the established ordered producing sites; this correction does not peel a first term
from multi-term regional contractions or change separate [bias](../../glossary.md#term-bias), [carry](../../glossary.md#term-carry), partial-merge or SUM ADDs.
Activity gates still dominate both operand loads; inactive singleton contractions retain +0.

The shared scalar/vector contributor seam takes the generation-only `directProduct` fact.
Fixed [feature](../../glossary.md#term-feature) contractions derive it from their complete logical K. [Runtime](../../glossary.md#term-runtime) row/cursor [owners](../../glossary.md#term-owner)
select singleton versus ordinary traversal once, outside contributor loops, using the full
logical [extent](../../glossary.md#term-extent) rather than active-row count or worker-range length. Each alternative rebuilds
its generation-local load-[cache](../../glossary.md#term-cache)/last-use state; no [callback](../../glossary.md#term-callback) is reused with stale local definitions.

Both generated compatibility streams append `|matmul-single-product-v1` only for MATMUL-bearing
standalone, retained-Scan or regional bodies. Nested ordinary-unit Scan helpers propagate their
changed [specialization](../../glossary.md#term-specialization) keys. This narrowly supersedes earlier literal-retention promises for
those bodies; unaffected [pointwise](../../glossary.md#term-pointwise)/SUM controls, global envelope 71, family projections, [canonical](../../glossary.md#term-canonical-form)
[IR](../../glossary.md#term-intermediate-representation-ir) and numeric/helper ABIs remain unchanged. Historical payloads are retired, not repinned.

Production [OpenBLAS](../../glossary.md#term-openblas) MATMUL [qualification](../../glossary.md#term-qualification) is K>1 for both [FLOAT32](../../glossary.md#term-float32) and [FLOAT64](../../glossary.md#term-float64). K=0/K=1 are
ineligible before native candidate/cost/decision selection and retain portable execution.
This is cold [shape](../../glossary.md#shape)-based [admission](../../glossary.md#term-admission), not value inspection, a retry or runtime fallback; raw
provider [GEMM](../../glossary.md#general-matrix-multiplication--gemm) [bindings](../../glossary.md#term-binding) and qualified K>1 behavior remain unchanged.


### Qualified [oneDNN](../../glossary.md#term-onednn) [CPU](../../glossary.md#term-cpu) realization

The current oneDNN route requires explicit loading of an exact external 3.14.0 [little-endian](../../glossary.md#text-encoding-and-file-boundaries) [LP64](../../glossary.md#term-lp64)
library with sequential CPU [runtime](../../glossary.md#term-runtime) (SEQ, code 1). Required loading failures throw; no optional
provider discovery or Runtime fallback may mask them. CPU [backend identity](../../glossary.md#backend-identity--backendid), availability and
[Model](../../glossary.md#term-model) semantics are unchanged. Bare [homogeneous](../../glossary.md#term-homogeneous) [FLOAT32](../../glossary.md#term-float32) [MATMUL](../../glossary.md#matrix-multiplication--matmul) with K > 1 over static positive
dense/permuted-dense and normalized broadcast-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [layouts](../../glossary.md#layout) of [rank](../../glossary.md#term-rank) at most 12 is the existing
qualified baseline. The user-authorized [F32](../../glossary.md#floating-point-precision-and-rounding) expansion may additionally realize common [pointwise](../../glossary.md#term-pointwise),
reduction and movement units, and literal native substeps of an existing composite unit, only
where the actual implementation preserves every Model arithmetic site, contributor domain,
special value and saved/public boundary. Native [descriptor](../../glossary.md#term-descriptor) availability alone is not [qualification](../../glossary.md#term-qualification).
Unsupported or unqualified units retain their already planned ordinary CPU realization; no
value-based route choice, numerical repair, [tolerance](../../glossary.md#term-tolerance) relaxation or runtime fallback is allowed.
Empty/K=0/K=1 contractions, symbolic layouts, Scan orchestration and [opaque](../../glossary.md#term-opaque) native
[convolution](../../glossary.md#term-channels-first-convolution-layers)/[normalization](../../glossary.md#term-normalization)/[pooling](../../glossary.md#term-pooling)/[activation](../../glossary.md#term-activation) replacements are not thereby qualified.

Both native APIs consume common CPU [IR](../../glossary.md#term-intermediate-representation-ir) and [canonical](../../glossary.md#term-canonical-form) external [buffers](../../glossary.md#term-buffer); saved/public stores
remain ordinary memory obligations. [Graph](../../glossary.md#graph) is not [AD](../../glossary.md#autograd) or a provider-owned replacement graph
pipeline. [API](../../glossary.md#term-application-programming-interface-api) preference does not force native selection, and [node](../../glossary.md#node) count does not force Graph.
CPU analysis compares complete portable/[OpenBLAS](../../glossary.md#term-openblas) and explicit ordinary-[contraction](../../glossary.md#contraction-dimension) alternatives,
including longer partitions and [constant](../../glossary.md#term-constant) writes, with conservative dimensionless heuristics
charging arithmetic/remainder, [dispatch](../../glossary.md#term-dispatch)/control, preparation, memory, ingress and scratch.
These are not measured time or a speedup guarantee. Exact OpenBLAS Phase-1 decisions remain
frozen, selected Phase-2 [topology](../../glossary.md#term-topology) is not replaced, and Phase-2 [schema](../../glossary.md#term-schema) 4 binds oneDNN policy and
load-session [identity](../../glossary.md#term-identity-meanings) while rejecting stale decisions.

Analysis retains descriptor facts, not native handles. Primitive [metadata](../../glossary.md#term-metadata) queries create no
primitive; Graph's scratch query requires temporary [compilation](../../glossary.md#compiler-engine-and-execution-stage-names) and closure. Finalization
recompiles the exact recipe, verifies retained metadata and returns ordered persistent [kernels](../../glossary.md#kernel)
as [`PreparedResource`](../../glossary.md#prepared-resource--preparedresource) objects with transactional [rollback](../../glossary.md#term-rollback). Native caches remain library-owned.
Data spans, [padding](../../glossary.md#padding)/[alignment](../../glossary.md#term-alignment), ingress copies, literal native intermediates and user scratch are
declared before shared assignment. A composite unit's native substeps remain one cold-fixed
realization of the existing common unit, not a replacement semantic graph or interpreter.
Each [run](../../glossary.md#term-run) workspace factory creates streams, memory/[tensor](../../glossary.md#tensor) handles and [ABI](../../glossary.md#term-application-binary-interface-abi) tables before [binding](../../glossary.md#term-binding);
the issued workspace retains the [kernel](../../glossary.md#kernel) and library until close, independently of prepared-[owner](../../glossary.md#term-owner)
closure. Runtime's prepared-[resource lease](../../glossary.md#ownership-borrowing-and-lease) contract remains unchanged.

One [RunState](../../glossary.md#run-state--runstate) owns each shared ingress workspace. Copies share only matching logical value,
origin and [referenced span](../../glossary.md#referenced-element-span); the first native consumer refreshes that span every run and later
consumers reuse it within the run. No offset prefix or across-run [weight](../../glossary.md#term-weight) [snapshot](../../glossary.md#term-snapshot) is copied.
Graph inputs are VARIABLE. Cold binding applies direct origins once, copied origins zero, and
rejects output/input/scratch [aliasing](../../glossary.md#term-aliasing). Fixed copy/execute/wait runs synchronously under the
shared CPU budget with demand one; there is no hot allocation, selection, retry or fallback.
Existing [Compiler](../../glossary.md#term-compiler) AD uses these same qualified ordinary native units; there is no vendor backward
or second differentiation path. [F64](../../glossary.md#floating-point-precision-and-rounding) CPU/OpenBLAS support and [Metal](../../glossary.md#term-metal) [admission](../../glossary.md#term-admission) remain unchanged.
The separately authorized CPU low-[precision](../../glossary.md#term-precision) cutover follows F32 expansion and retains the same
Model-owned arithmetic, resource and qualification obligations.

### [General Scan](../../glossary.md#general-differentiable-scan) [CPU](../../glossary.md#term-cpu) realization

The accepted general-scan CPU path is not the existing fixed recurrent Java route. Analysis
prepares one body and declares exact checked staging, [carry](../../glossary.md#term-carry), temporary, output and demanded-history
resources before shared slot assignment. Finalization, only after assigned slots, emits the outer
time loop and admitted typed body emitters. The complete semantic protocol remains in
[the scan owner](recurrent-scan.md#general-differentiable-scan). CPU0030 implements this path for
bounded same-[shape](../../glossary.md#shape) [FLOAT32](../../glossary.md#term-float32) [pointwise](../../glossary.md#term-pointwise) bodies and their actual independently qualified
[Compiler](../../glossary.md#term-compiler)-generated reverse siblings. CPU0031 adds [homogeneous](../../glossary.md#term-homogeneous) FLOAT32 broadcast, SUM/sumToShape
and bounded shape/[layout](../../glossary.md#layout) closure under the same independent whole-[occurrence](../../glossary.md#term-occurrence) [qualification](../../glossary.md#term-qualification).
CPU0032 adds static homogeneous FLOAT32 [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../glossary.md#matrix-multiplication--matmul) within the same emitted time loop,
including independently qualified actual Compiler-generated reverse bodies. CPU0033 implements
homogeneous [FLOAT64](../../glossary.md#term-float64) across the complete bounded declared algebra and actual first-order reverse
closure, with double producing/[accumulator](../../glossary.md#accumulator-type) sites and eight-byte resources. CPU0037's implemented
actual-read exception is defined only by the
[scan owner](recurrent-scan.md#traversal-empty-domains-and-declared-qualification);
qualification and main's final capability acceptance are recorded in CPU0037 (2026-10-24).
No low-type, wider-layout, [vector](../../glossary.md#term-vector), parallel or [OpenBLAS](../../glossary.md#term-openblas) scan capability follows;
a forward provider answer alone is not a
backward capability claim.

[Conv3d](../../glossary.md#conv3d--ncdhw) training uses the existing generated movement/fold families, not a backward executor.
`UNFOLD3D` copies represented floating bits from [NCDHW](../../glossary.md#conv3d--ncdhw) into [canonical](../../glossary.md#term-canonical-form)
`[N, C*kD*kH*kW, DOut*HOut*WOut]` columns, with exact configured [padding](../../glossary.md#padding) or direct [positive zero](../../glossary.md#nan-infinity-and-signed-zero).
`FOLD3D` uses the declared floating working type: each output cell starts at positive zero,
inverse-gathers in ascending [kernel](../../glossary.md#kernel)-[coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) order, includes every in-bounds contributor once
and stores once. Checked cold [spatial](../../glossary.md#term-spatial-axes) geometry is shared by extraction/folding; generated loops
use invocation-local primitive facts, not per-[element](../../glossary.md#term-element) geometry objects or [runtime](../../glossary.md#term-runtime) [dispatch](../../glossary.md#term-dispatch).
Static public [admission](../../glossary.md#term-admission) and independently proved body `lowerBatch` paths remain distinct;
symbolic Compiler formulas do not widen ordinary public named-B Conv3d admission. Existing
[carrier](../../glossary.md#term-storage-carrier)/access/[lifetime](../../glossary.md#term-lifetime)/[alias](../../glossary.md#aliasing-and-arena) proofs and legacy family ABIs, keys and generated bytes remain.
Low fold/[convolution](../../glossary.md#term-channels-first-convolution-layers) admission follows the [parity cutover](#cpu-low-precision-parity-cutover),
not raw extraction alone. [Pool3d](../../glossary.md#pool3d--ncdhw-pooling) admits matching input/output [gradient](../../glossary.md#term-gradient) flags for independently
qualified floating MaxPool3d and AveragePool3d, reusing the existing forward [kernels](../../glossary.md#kernel), shared Compiler
`PoolingGradientRules`, and `UNFOLD3D`/`FOLD3D` closure. Maximum cotangents retain exact
first-eligible-winner selection; average cotangents retain the fixed kernel-element divisor.
Qualified `lowerBatch` body composition covers both all-valid and runtime-length Scan in both
traversal directions, subject to complete generated-[topology](../../glossary.md#term-topology) qualification and the existing
homogeneous [dtype](../../glossary.md#data-type--datatype), static-window and terminal restrictions. This does not widen ordinary
public named-B admission. Low arithmetic retains [F32](../../glossary.md#floating-point-precision-and-rounding) working accumulators and one final [narrowing](../../glossary.md#term-narrowing).
No standalone backward kernel, runtime fallback, or 3D-specific composition blacklist
is introduced.

Existing CpuLoopEmitter generation callbacks and CpuScalarEmitter typed locals are generation-time
composition seams, not runtime per-step callbacks. CpuKernelIr's pointwise instruction list,
CpuMatmulEmitter's entry-slot/geometry assumptions and CpuCarrierEmitter's fixed entry [ABI](../../glossary.md#term-application-binary-interface-abi)/local
block are not plug-and-play step bodies. Implementation must explicitly bind invocation versus
step setup, typed carriers/locals, bases, ranges and geometry. Hoist [invariant](../../glossary.md#term-invariant) setup outside time
and prefer direct time-dependent offsets/addresses into sequences, emissions and live residuals.
Reuse [scalar](../../glossary.md#term-scalar)/vector/carrier arithmetic only where qualified; no vector/parallel capability follows.

Select/materialize only when necessary and justify staging copies, carry banks and body temporaries.
Emission cannot remove live saved histories or all MATMUL storage. Dead residuals, including reverse
residuals, require no history bytes/stores or [residual](../../glossary.md#scan-residual-port-and-history-demand) [fusion](../../glossary.md#term-fusion) restriction. One [RunState](../../glossary.md#run-state--runstate) owns all
[mutable](../../glossary.md#term-mutable) memory, changing addresses, carry and geometry arrays; shared prepared state is [immutable](../../glossary.md#term-immutable).
No runtime [graph](../../glossary.md#graph) interpreter, per-[operation](../../glossary.md#operation) [lookup](../../glossary.md#term-lookup), per-step dispatch/[callback](../../glossary.md#term-callback)/allocation or T-fold
graph/code expansion is allowed. Technically necessary direct static helpers require an explicit
clean Java [oracle](../../glossary.md#term-test-oracle) rationale; this is not a mandate for one giant method or hidden fallback overhead.

CpuKernelIr canonical [identity](../../glossary.md#term-identity-meanings) still excludes extents, assigned slots and [run](../../glossary.md#term-run) [bindings](../../glossary.md#term-binding). Separate
baked from runtime scan facts, and key every baked body/role/demand fact in the appropriate CPU
generated [compatibility identity](../../glossary.md#identity-meanings-and-algebraic-identities). Reject stale format or meaning [artifacts](../../glossary.md#term-artifact). The original scan
decision selected no [schema](../../glossary.md#term-schema) number or CPU type; CPU0030 now uses generator envelope 71 and typed
general-scan family projection 71. CPU0031 preserves both identities and the old pointwise keys/bytes:
its additive shaped-v1 canonical projection is geometry-free, while both existing [specialization](../../glossary.md#term-specialization)
identity byte streams conditionally include all baked shaped geometry and resource facts.
CPU0032 preserves schema/family 71 and both old pointwise and shaped keys/bytes. Its additive
MATMUL-v1 canonical projection remains geometry-free; the separate `scan-matmul-v1` [payload](../../glossary.md#artifact-envelope-and-atomic-publication) binds
ordered operands and all baked affine/resource facts in both specialization identity streams.
CPU0033 adds typed FLOAT64 variants without changing schema/family 71, either staged projection or
its versioned payload. Existing arithmetic/boundary dtype and eight-byte baked resource facts
distinguish those variants in both streams; accepted FLOAT32 keys and generated bytes are retained.
CPU0034 subsequently advanced the sole baked shaped/MATMUL payload discriminants to
`scan-shaped-v2` and `scan-matmul-v2` for both dtypes. Typed coordinate induction and proved
post-step emission-carry transitions change staged [bytecode](../../glossary.md#term-bytecode), so both specialization streams,
[binary](../../glossary.md#term-binary-descriptor-and-metadata) names and [cache](../../glossary.md#term-cache) keys advanced and rejected staged v1 artifacts. CPU0035 advanced those
same sole baked payloads to `scan-shaped-v3` and `scan-matmul-v3` in both streams for proved
redundant factor, address and local-copy cleanup. CPU0036 advances the same payloads to
`scan-shaped-v4` and `scan-matmul-v4` for top-tested staged loop control and dominating final-copy
bases. Staged v1/v2/v3 artifacts are incompatible.
Global schema/family 71, geometry-free canonical v1 projections and pointwise/fixed/unrelated
identities remain unchanged;
the CPU0033 retention statement above describes that earlier acceptance, not current staged bytes.
Future CPU briefs must name every added type without speculative
general control-flow infrastructure. Generic no-copy, [JIT](../../glossary.md#term-just-in-time-compilation-jit), speed or CPU0029 timing parity is unproved.
The [scan qualification gates](recurrent-scan.md#qualification-performance-and-migration-gates)
require a complete F32 forward/automatic-reverse milestone, then declared algebra/closure and [F64](../../glossary.md#floating-point-precision-and-rounding)
completion, clean Java algorithm/hot-loop/[dataflow](../../glossary.md#term-dataflow) comparison, Class-File inspection and short warmed
matched-work evidence with cold/entry/kernel costs separated. No measurement is part of this cutover.

CPU0042's fixed-geometry length-aware execution reuses this generated path, not the fixed executor;
its complete semantics belong to the [scan owner](recurrent-scan.md#length-aware-whole-batch-scan).
CPU independently qualifies FLOAT32/FLOAT64 forward and actual Compiler reverse, mixed [INT64](../../glossary.md#data-type--datatype)/floating boundary
binding, exact predeclared resources, activity-dominated [lowering](../../glossary.md#lowering)/emission and changed length-mode
identities. Old all-valid/fixed keys and generated bytes stay unchanged and must not depend on
changed ScanAttrs diagnostic text. The length-only branch precedes legacy analysis and adds canonical
`:length-aware-v1` and baked `|scan-length-aware-v1=` in both specialization streams, retaining
schema/family 71, static staged v4, symbolic v3 and unrelated keys/bytes. MAIN accepted CPU0042
on 2026-10-26 after independent review and recorded gates; its
[Result](../../planning/backends/cpu/tasks/0042-length-aware-general-scan-execution.md#result)
owns the evidence and [nonstationary](../../glossary.md#term-nonstationary) timing limits. Prepare0015 implements symbolic-B plus lengths;
its root [validation](../../glossary.md#term-validation) and the sole bounded probe completed (evidence and limits:
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) );
final MAIN acceptance remains pending.
That CPU [checkpoint](../../glossary.md#term-checkpoint) did not qualify low/vector/parallel/OpenBLAS Scan or the independent [Metal](../../glossary.md#term-metal) domain.

The implemented combined [scope](../../glossary.md#term-scope) is fixed T/[features](../../glossary.md#term-feature), one named B, homogeneous FLOAT32/FLOAT64
generated scalar, single-range forward and actual Compiler-generated reverse. Existing Scan
lowering/[IR](../../glossary.md#term-intermediate-representation-ir)/preparer/emitter/executable [owners](../../glossary.md#term-owner) carry combined numeric recipes, explicit B and
separate INT64 control; no new type/[API](../../glossary.md#term-application-programming-interface-api)/protocol or all-valid symbolic FLOAT64 capability is selected.
The bounded ordinary outer/control/ROW(1) prerequisite below is implemented in the same task.
The [scan owner](recurrent-scan.md#runtime-preflight-and-implementation-gates) retains all semantics,
including original-source [AD](../../glossary.md#autograd) restrictions, generated mixed roles, saved sites, activity/[preflight](../../glossary.md#term-preflight)
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
a bank: raw initial copying precedes [nominal](../../glossary.md#term-nominal)-[stride](../../glossary.md#term-stride) binding, each derived [view](../../glossary.md#term-view-meanings) retains its own
recipe, and identical read/write relative offsets share one cell-scoped local with proved zero
bases omitted. Changing/bank-backed and noncombined bindings retain their selection protocol.
CPU0058 advanced only the combined baked payload to v6: a nonbanked scratch cut gains the existing
zero-base load/store proof only when both coefficients of its complete
`offset + offsetPerBatch * B` recipe are zero. Its entry scratch carrier/base stays valid through
every use in the inclusive cut lifetime; aliases consult the exact origin's proof while keeping
their own relative strides. Port/time-backed and carry-bank cuts retain their original bases,
and relative-offset caching remains cell-scoped. No allocation, lifetime or bank choice changes.
CPU0060 advanced only that payload to v7 for flat scalar initialization, the existing initialized
EMPTY [publication](../../glossary.md#term-publication) path at zero maximum length, coordinate-nested pointwise affine prefixes,
proved direct addresses for active ROW(1) @ ROW(0) contributors and raw-[singleton](../../glossary.md#term-singleton) [SUM-to-Shape](../../glossary.md#sum-to-shape)
without a contributor loop. At that version, other [matrix](../../glossary.md#term-matrix-meanings) paths retained their cursor algorithm.
CPU0064 advances only that payload to v9, skipping withdrawn v8. Stable carriers and temporal
bases bind directly, and affine byte facts derive from admitted explicit B. Combined feature
contractions with proved [constant](../../glossary.md#term-constant) effective K strides also use indexed addresses; the existing
[contiguous](../../glossary.md#term-contiguous)-feature ROW(1) @ ROW(0) proof retains activity-gated indexed contributors.
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
scalar and single-range, fixed T/[features](../../glossary.md#term-feature)/state widths with invocation B; [Model](../../glossary.md#term-model) source dtype
validity and Compiler's bounded FLOAT32/FLOAT64 AD are not CPU answers. Qualify the complete
forward and actual Compiler-generated reverse occurrences independently, including bounded
[rank](../../glossary.md#term-rank)-three outer closure, not merely supported child kinds or a handwritten reverse body.

Prepare0014 extends existing Scan analysis/IR/preparer/emitter owners and explicit provider dispatch;
[symbolic Scan](../../glossary.md#symbolic-scan) no longer falls through ordinary-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-only analysis. Analysis fixes the
complete body/resource topology and immutable numeric access/frame/cut/carry-offset recipes before
slots; finalization emits after slots. Use existing `CpuAccessPlan.BatchBinding`, MATMUL/aggregate
numeric binding, `PreparedMemoryPlan.ByteSize` and numeric [RunMemoryGeometry](../../glossary.md#run-memory-geometry--runmemorygeometry) [handoff](../../glossary.md#term-handoff). Recipes may
need checked piecewise [alignment](../../glossary.md#term-alignment)/empty scratch, not only affine byte sizes. Do not interpret a
[sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) B, relower on bind, allocate maximum banks or choose per-B routes/slots/topology.
Structural zero-byte workspaces keep their fixed requirement/creator. Prove any cut lifetime and
size dominance for every admitted B; retain disjoint regions otherwise. Complete resource and
empty/saved-site semantics remain solely with the linked Scan [owner](../../glossary.md#term-owner).

State-aware representation hooks must check actual bytes plus physical type/access/alignment/
overlap. Binding copies numeric facts into fresh run-owned mutable packed/access/carry/address
arrays; no shared currentB or ThreadLocal. The accepted future
[Engine memo](runtime-prepare-engine.md#accepted-engine-geometry-memo)<span data-glossary-navigation="true"> ([Engine glossary](../../glossary.md#term-engine))</span> does not authenticate CPU
storage or expose a [memo](../../glossary.md#cache-and-memoization) to generated code. Load B/strides/frame offsets once per invocation or
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

Affected-[module](../../glossary.md#term-module)/[conformance](../../glossary.md#term-conformance)/integration tests must cover the Scan owner's semantic/empty/
resource/admission matrix and independent actual reverse qualification. Generated Class-File
inspection must compare clean-Java algorithm/hot-loop/dataflow and hidden overhead; bounded warmed
matched-work generated/direct evidence separates setup, admission and kernel costs. Retained static
keys/bytes and read exceptions need [regression](../../glossary.md#term-regression) evidence; symbolic length skipping remains separate.
Prepare0014 records source-matched focused evidence and bounded probe results/limits; final root
qualification remains tracked there. No static/ordinary timing guarantee is inherited.

### CPU0071 approved static [MATMUL](../../glossary.md#matrix-multiplication--matmul) compatibility cutover

The user approved this append-only cutover on 2026-10-29; CPU0071 implements it in the existing
[CPU](../../glossary.md#term-cpu) [owners](../../glossary.md#term-owner). Independent source/[Javadoc](../../glossary.md#term-javadoc) review approved the corrected frozen implementation;
MAIN accepted its source and [validation](../../glossary.md#term-validation) gates on 2026-10-29. This is the implemented narrow exception to static/noncombined key-and-byte retention
promises elsewhere in this contract and the root [identity](../../glossary.md#term-identity-meanings) history; those historical results remain
unchanged. The exception permits only these baked [specialization](../../glossary.md#term-specialization) discriminant updates:

- Static all-valid Scan with a retained MATMUL stage: `|scan-matmul-v4=` → `|scan-matmul-v5=`.
- Static-length Scan with a retained MATMUL stage: `|scan-length-aware-v1=` →
  `|scan-length-aware-matmul-v2=`. Select this branch only when the retained stage order satisfies
  `order.anyMatch(s -> s.right() >= 0)`, not from a MATMUL that [lowering](../../glossary.md#lowering) has removed.

Both apply to the already-admitted [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) forward and actual [Compiler](../../glossary.md#term-compiler)-generated reverse
occurrences. `compatibilityBytes()` and `classIdentityBytes()` must both consume
`bakedSpecializationIdentity()` so the changed names/keys advance together and reject old [artifacts](../../glossary.md#term-artifact)
in the changed namespaces. No blanket [schema](../../glossary.md#term-schema) bump or invalidation of unchanged controls is allowed.

Retain non-MATMUL shaped-v4 and static-length-v1, all-valid symbolic-v3, combined symbolic-length-v9,
[pointwise](../../glossary.md#term-pointwise)/fixed, ordinary [scalar](../../glossary.md#term-scalar)/[vector](../../glossary.md#term-vector)/tiled/[TANH](../../glossary.md#term-tanh), `ordinary-batch-v1` and
`ordinary-batch-typed-v1` identities and generated bytes. Retained byte identity is a validation
[target](../../glossary.md#term-target), not assumed proof. CPU0071's frozen-corpus checks cover the recorded controls; vector-byte
evidence is qualified by matching captured species. An unexpected changed namespace still stops
implementation for review; [finite](../../glossary.md#term-finite) corpus evidence is not proof for every possible specialization.
Schema/family 71, [canonical](../../glossary.md#term-canonical-form) projections, numeric packs, resources and [Runtime](../../glossary.md#term-runtime) [handoff](../../glossary.md#term-handoff) stay unchanged.

The selected setup change moves static [constant](../../glossary.md#term-constant) MATMUL [strides](../../glossary.md#term-stride) outside the time loop; the evolving
scalar-to-dense factor remains a changing [binding](../../glossary.md#term-binding). No new production type/public [API](../../glossary.md#term-application-programming-interface-api), [RunState](../../glossary.md#run-state--runstate),
route, numerics, resource, worker or Job-reuse change is authorized; test fixtures are not new
production types. Scan and Runtime semantics remain solely with
their existing owners. The [CPU0071 brief](../../planning/backends/cpu/tasks/0071-common-matmul-binding-foundation.md)
records implementation, validation and diagnostic limitations, not additional authority or a speed claim.

### CPU0073 accepted row-map and multiple-terminal reduction [fusion](../../glossary.md#term-fusion)

CPU0073 [scalar](../../glossary.md#term-scalar) v1 remains governed below, with only the implemented
[CPU0075 scalar emission/cache cutover](#cpu0075-selective-scalar-load-and-address-reuse)<span data-glossary-navigation="true"> ([cache glossary](../../glossary.md#term-cache))</span>
and the implemented [CPU0076 staging exception](#cpu0076-prospective-invariant-external-cell-load-staging)
qualifying affected scalar generation and literal retention. The implemented
[CPU0074 exec-v1 exception](#cpu0074-selected-row-region-execution-exception) qualifies its
scalar/range, additional-seam and retention restrictions; CPU0074 itself did not change v1 scalar [artifacts](../../glossary.md#term-artifact).

Implemented by CPU0073 after user approval and authority acceptance on 2026-10-30. Independent
code/tests/[Javadoc](../../glossary.md#term-javadoc) review approved with no open findings; MAIN accepted the implementation and
named [conformance](../../glossary.md#term-conformance)/[Engine](../../glossary.md#term-engine) join. The [task result](../../planning/backends/cpu/tasks/0073-row-map-multi-reduction-fusion.md#result)
records [validation](../../glossary.md#term-validation), evidence-transfer limits and unmeasured performance.
This subsection owns the exact [CPU](../../glossary.md#term-cpu) mechanism and selective [identity](../../glossary.md#term-identity-meanings) exception; plans and
[ADR 0052](../../design/decisions/0052-row-map-multi-reduction-fusion.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span> do not add authority.
The [Scan owner](recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([owner glossary](../../glossary.md#term-owner))</span> retains all Scan semantics;
[Prepare and Run lifecycles](runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../glossary.md#prepare); [Run glossary](../../glossary.md#term-run))</span> remain unchanged.

#### Physical region and [admission](../../glossary.md#term-admission)

A row map is a pure ordered row-local [directed acyclic graph](../../glossary.md#term-directed-acyclic-graph-dag) ([DAG](../../glossary.md#intermediate-representation-dag-and-ssa)) sharing the same proved row
[axis](../../glossary.md#term-axis) R. It may include bounded [feature](../../glossary.md#term-feature) contractions and feed independent row outputs plus
multiple terminal contractions or represented Scan sums. A terminal consumes rows and finishes
after all contributors; it cannot feed row work in the same pass. This region is derived [CPU](../../glossary.md#term-cpu)
physical [lowering](../../glossary.md#lowering), not a new [Tensor](../../glossary.md#tensor) [operation](../../glossary.md#operation), [Compiler](../../glossary.md#term-compiler) [graph](../../glossary.md#graph), semantic DAG, scheduler or executor.
Ordinary lowering selects one complete execution unit. Scan wraps the same generation-time
region inside its existing unit/time traversal, never a per-step [Runtime](../../glossary.md#term-runtime) unit, [callback](../../glossary.md#term-callback) or Job.

Implemented selection requires a genuinely computed row [producer](../../glossary.md#term-producer) reaching a terminal and
another retained or terminal sink. Direct-input or movement sharing alone remains noneligible;
the bounded region is not blanket [pointwise](../../glossary.md#term-pointwise) or arbitrary-[shape](../../glossary.md#shape) [fusion](../../glossary.md#term-fusion).

Admission is [scalar](../../glossary.md#term-scalar), single-range, [homogeneous](../../glossary.md#term-homogeneous) [FLOAT32](../../glossary.md#term-float32) or [FLOAT64](../../glossary.md#term-float64) within existing capability:
ordinary static [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding), already-admitted ordinary symbolic F32, and independently admitted static,
symbolic, length and combined Scan occurrences. [Canonical](../../glossary.md#term-canonical-form) boundaries, existing dense/scalar read
exceptions and proved internal [transpose](../../glossary.md#term-transpose)/expand/[feature](../../glossary.md#term-feature)-only edits remain the limits. No new
operation, [dtype](../../glossary.md#data-type--datatype), [layout](../../glossary.md#layout) or [carrier](../../glossary.md#term-storage-carrier) follows: ordinary symbolic F64 [MATMUL](../../glossary.md#matrix-multiplication--matmul) and all-valid symbolic
F64 Scan remain unqualified. Fixed bounded features, static R or existing named B, and fixed T
do not authorize B/T [unrolling](../../glossary.md#term-unrolling), per-B caching, runtime profitability decisions, [SIMD](../../glossary.md#simd), threads,
[OpenBLAS](../../glossary.md#term-openblas), Job reuse or across-time [carry](../../glossary.md#term-carry) promotion.

Ordinary exact SUM/[MEAN](../../glossary.md#term-mean) remain outside the region: keep required producer [materialization](../../glossary.md#materialization),
exact workspace slots/creators and empty/[singleton](../../glossary.md#term-singleton) behavior. Retain symbolic F32 five-[limb](../../glossary.md#exact-superaccumulator-and-limbs)/48-byte
and bounded symbolic F64 34-limb/280-byte state; static F64 remains domain-sized.
`CpuExactSumEmitter` is unchanged. Represented Scan SUM is its existing distinct algorithm,
not a relabeling of exact `CpuAggregateIr.Kind.SUM`. [Model](../../glossary.md#term-model) and existing math [owners](../../glossary.md#term-owner) retain all
numerics: preserve contributor/operand order, typed MUL/ADD/result [rounding](../../glossary.md#term-rounding) and raw no-axis/
singleton movement. No new [FMA](../../glossary.md#term-fused-multiply-add-fma), [reassociation](../../glossary.md#term-reassociation), [derivative](../../glossary.md#term-derivative) simplification or numerical policy.

#### Cold proof, resources and range

Before cuts or resource declarations, use complete [partition](../../glossary.md#partition)/body uses, demanded/public/saved/
external uses and physical storage origins. Prove [dependency](../../glossary.md#term-dependency) convexity and an acyclic quotient
of the full [DAG](../../glossary.md#term-directed-acyclic-graph-dag), including both [contraction](../../glossary.md#contraction-dimension) operands, [views](../../glossary.md#term-view-meanings) and temporal old/new hazards.
Retain exact producing stores for live external/saved uses or split the region; never replay
[activations](../../glossary.md#term-activation) or remove demanded histories. Prove named row-[axis](../../glossary.md#term-axis) [provenance](../../glossary.md#provenance) through each view,
[injective](../../glossary.md#term-injective) writes, legal read [aliases](../../glossary.md#aliasing-and-arena), no overwritten future reads, and inclusive cut [lifetime](../../glossary.md#term-lifetime)/
size dominance for every admitted B. Equal extents (including F=B) or sampled B are not proof;
unknown aliases reject. Recheck actual backing aliases at [binding](../../glossary.md#term-binding).

Activity dominates both operand loads and dependent math, including shared/[invariant](../../glossary.md#term-invariant) hoists.
Preserve the Scan [owner](../../glossary.md#term-owner)'s B=0/1, T=0, all-inactive, raw [carry](../../glossary.md#term-carry) [identity](../../glossary.md#term-identity-meanings), +0 [padding](../../glossary.md#padding) and poisoned-read
rules, [producer](../../glossary.md#term-producer)-ordered all-length [preflight](../../glossary.md#term-preflight) before any Scan-local write, and one distinct shared
carry ADD after row reduction at active(t). No carry ADD is folded into per-row contributions.

Every attempted candidate is bounded: at most 32 row sites, 4 terminal outputs, 64 terminal
[scalar](../../glossary.md#term-scalar) cells, [feature](../../glossary.md#term-feature) extents at most 16 each, 128 simultaneously live [JVM](../../glossary.md#term-java-virtual-machine-jvm) local slots (double
counts two), 32 boundary carriers, 64 candidate attempts per anchor, and a conservative generated
method-size proof at most 48000 bytes. Reject over-limit candidates before resources; do not
truncate work or cap the whole partition. These are cold structural limits, not measured
profitability thresholds; the live-value limit is primary and method size only a Class-File [guard](../../glossary.md#term-guard).

Ordinary [fused](../../glossary.md#specialization-and-fusion) work owns both row and shared outputs: prepared work count is 1 and entry [0,1)
executes the full body even at R/B=0, publishing required nonempty shared zeros. [0,0) does no
work; every other range rejects before writes. No rowCount work count, arbitrary [pointwise](../../glossary.md#term-pointwise)
ranges or parallel splitting. Extend the existing ordinary binder for this case and all-output
alias checks rather than inheriting last-output/count assumptions; no [Runtime](../../glossary.md#term-runtime) protocol changes.

Select ordinary regions before [decomposition](../../glossary.md#term-decomposition)/[materialization](../../glossary.md#materialization), Scan regions before
`shapedResources`. Recompute [liveness](../../glossary.md#term-liveness) and exact scratch for selected regions, retaining required
banks, histories, old-carry hazards and exact-SUM producer stores. Finalization only binds assigned
slots and emits; it cannot discover requirements, change [topology](../../glossary.md#term-topology) or rerun [fusion](../../glossary.md#term-fusion). Method-size
verification failure then is a defect, not a late fallback. One [RunState](../../glossary.md#run-state--runstate) owns all [mutable](../../glossary.md#term-mutable) state.

#### Exact types, [signatures](../../glossary.md#term-signature) and generation [lifetime](../../glossary.md#term-lifetime)

Only three new top-level production types are admitted, under
`io.github.pho001.synaptik.backend.cpu`: `internal.ir.CpuRowMapReductionIr` ([immutable](../../glossary.md#term-immutable),
implements `CpuPortableKernelIr`), `internal.lowering.CpuRowMapReductionLowering` (shared cold
proof), and `internal.codegen.emit.CpuRowMapReductionEmitter` (generation-only loop/lifetime
orchestration). The [IR](../../glossary.md#term-intermediate-representation-ir) alone contains the two nested [binding](../../glossary.md#term-binding) records below. Existing [pointwise](../../glossary.md#term-pointwise)
IR and single-anchor Scan stages do not express multiple terminal accumulators sharing row sites.
These declaration excerpts describe the implemented internal seams, not supported [API](../../glossary.md#term-application-programming-interface-api) or runnable Java:

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

`rowSites` interleaves pointwise outputs and row-[feature](../../glossary.md#term-feature) [contraction](../../glossary.md#contraction-dimension) outputs topologically;
each ordinal resolves to exactly one kind and completes its fixed feature cells before consumers.
Terminals are excluded from rowSites and ordered by `terminalOutputs`; operands are dominated.
Use typed contraction loaders, not fabricated ordinary [F64](../../glossary.md#floating-point-precision-and-rounding) [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [MATMUL](../../glossary.md#matrix-multiplication--matmul) IR.
`viewOrigins[value]` covers every value with its self-rooted storage-producing ordinal; accesses
encode geometry relative to that origin (zero batch factors for static recipes). BatchBinding
does not encode [alias](../../glossary.md#aliasing-and-arena) [identity](../../glossary.md#term-identity-meanings). Validate total coverage, only self-root cycles, [view](../../glossary.md#term-view-meanings) composition,
rowAxes [provenance](../../glossary.md#provenance) and physical hazards. Proof lists contain full normalized uses and must-precede
pairs including conservative alias edges, not candidate-local counts.

The caller allocates locals with the same CodeBuilder: INT rowIndex/rowLimit, valueLocals by
value ordinal/fixed feature cell, terminalAccumulatorLocals by terminal position/cell. Only direct
storage uses -1; validate typed widths, disjoint slots and load dominance. Callbacks close over
these exact immutable handle tables; site callbacks receive ordinals and neither replace locals
nor mutate the tables or rowIndex. Emitter owns control/lifetimes, not allocation: init once,
rowIndex=0, increasing loop to prebound rowLimit, activeRow around all rowSites then terminal
steps, finish once after the loop, including R=0. Finish publishes completed contributions and
separate [carry](../../glossary.md#term-carry) ADDs; Scan's outer active-time gate encloses emission. All callbacks/arrays exist
only during generation, not as [runtime](../../glossary.md#term-runtime)/per-step objects.

Existing math [owners](../../glossary.md#term-owner) expose only these additional package-private static void generation seams:

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
[singleton](../../glossary.md#term-singleton) correctness cutover above. Reuse `CpuClassFileKernelGenerator.emitBoundPointwise`,
`CpuScalarEmitter` and [carrier](../../glossary.md#term-storage-carrier)
access; do not clone [opcode](../../glossary.md#term-opcode)/math switches or recreate retired Scan math subemitters.
Extend existing portable permits/route, [partition](../../glossary.md#partition) [lowering](../../glossary.md#lowering)/[decomposition](../../glossary.md#term-decomposition)/recognition/preparation,
Scan lowering/Stages/emitter, [specialization](../../glossary.md#term-specialization), generator and ordinary binder owners only.
`CpuPortableRoutePlan.encoded(CpuPortableKernelIr)` must explicitly return
`region.encodedKernelIr()` before its final aggregate cast. That envelope exposes boundary
stores/family identity with the same structuralKey; typed region facts select emission, not a
second semantic [graph](../../glossary.md#graph). `CpuGeneralScanStageSchedule.order` remains a post-slot reorder proof,
not [fusion](../../glossary.md#term-fusion), and must see region reads/writes/lifetimes. Any extra named production type,
common-seam change beyond the approved signatures, or cross-[owner](../../glossary.md#term-owner) protocol/runtime edge requires
explicit approval; a plan alone cannot authorize it. Ordinary private implementation helpers in
existing owners are permitted within this capability; they are not extra common seams or types.

#### Selective compatibility and retention

Keep global generator `CURRENT_VERSION=71`, existing namespaces and Scan class projection 71.
Reserve ordinary class projection 72 only for selected row regions. Append
`Optional<CpuRowMapReductionIr> rowMapReductionIr` to existing `CpuKernelSpecialization`:
projection 72 iff present, mutually exclusive with matmulIr/generalScanIr. Scan embeds regions
only within generalScanIr, retaining projection 71; keep 54 iff matmulIr and 71 iff generalScanIr.
Preserve every existing constructor overload, including the old [canonical](../../glossary.md#term-canonical-form) [signature](../../glossary.md#term-signature), delegating
with Optional.empty(). Check [fingerprint](../../glossary.md#term-fingerprint), boundary [dtype](../../glossary.md#data-type--datatype)/[carrier](../../glossary.md#term-storage-carrier)/access, [scalar](../../glossary.md#term-scalar)/species-zero,
materializedSourcePosition=-1 and scratch requirements against the region. Ordinary symbolic
regions remain [F32](../../glossary.md#floating-point-precision-and-rounding) only; this is no bypass of [F64](../../glossary.md#floating-point-precision-and-rounding) [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[MATMUL](../../glossary.md#matrix-multiplication--matmul) rejection.

Use geometry-free ordinary family marker `row-map-multi-reduction-v1`. Append
`|ordinary-row-map-multi-reduction-v1=` only for selected ordinary groups, and
`|scan-row-map-multi-reduction-v1=` only to selected Scan groups' existing baked mode [payload](../../glossary.md#artifact-envelope-and-atomic-publication).
Both `compatibilityBytes()` and `classIdentityBytes()` consume the same baked payload and reject
old unfused [artifacts](../../glossary.md#term-artifact) for selected [fused](../../glossary.md#specialization-and-fusion) specializations. Bind [topology](../../glossary.md#term-topology), ordered operands/sites,
dtype/carrier/access, algorithm/[singleton](../../glossary.md#term-singleton)/[rounding](../../glossary.md#term-rounding) distinctions, row/activity roles, retained
stores/demand, [feature](../../glossary.md#term-feature)/coefficient/resource/[lifetime](../../glossary.md#term-lifetime) facts, range convention and all code-shaping
limits. Include fixed R/T/[strides](../../glossary.md#term-stride)/resource bytes when baked into ClassFiles. Exclude only
invocation-resolved symbolic B, length values, addresses, assigned slots and invocation-resolved
resource bytes. Diagnostic text is not [identity](../../glossary.md#term-identity-meanings); no blanket [schema](../../glossary.md#term-schema) bump or control namespace change.

This exception and CPU0071's separate implemented exception do not retire noneligible controls.
Noneligible ordinary/static/symbolic/length/combined, [vector](../../glossary.md#term-vector)/tiled/[TANH](../../glossary.md#term-tanh)/fixed specializations keep
literal old keys, ClassFiles and resources. Freeze controls before source edits, never derive them
from candidate code. Test both streams, constructor/factory exclusivity, encoded route construction,
artifact generation/load/reuse and stale rejection, plus B/L-independent reuse. Qualify ordinary
and actual [Compiler](../../glossary.md#term-compiler)-generated Scan reverse separately with durable specialized clean-Java oracles
and generated-ClassFile inspection; [finite](../../glossary.md#term-finite) controls are not universal preservation proof.
Only after both callers work may superseded eligible whole-domain traversal glue/cuts retire;
keep fallback emitters, scheduling/temporal proofs, exact SUM, live saved stores and old oracles.
No performance claim or measurement permission follows; old timing budgets remain closed.

### CPU0074 selected row-region execution exception

Implemented, independently reviewed and MAIN accepted, 2026-10-30. CPU0074 is Complete after
final [CPU](../../glossary.md#term-cpu)/[Javadoc](../../glossary.md#term-javadoc), named integration and bounded performance evidence. Steady-state profitability
is not qualified; defaults and rollout are unchanged. This section is the sole CPU mechanism/[ABI](../../glossary.md#term-application-binary-interface-abi)/[identity](../../glossary.md#term-identity-meanings) [owner](../../glossary.md#term-owner) for
this exception. [ADR 0053](../../design/decisions/0053-row-fusion-simd-parallel.md) records rationale;
the [ADR](../../glossary.md#term-architecture-decision-record-adr) alone grants neither implementation launch nor executable capability.

#### Bounded selection and numerical boundary

CPU0073's genuine computed-row-[producer](../../glossary.md#term-producer), complete-use, convexity, [alias](../../glossary.md#aliasing-and-arena), named-[axis](../../glossary.md#term-axis), [liveness](../../glossary.md#term-liveness),
retained-store and structural-limit proofs still apply. The same ordinary static [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding),
ordinary symbolic F32, static all-valid F32/F64 Scan, all-valid symbolic F32 Scan, and static-length/
combined symbolic-length F32/F64 Scan domains may select [vector](../../glossary.md#term-vector), parallel-[scalar](../../glossary.md#term-scalar) or parallel-vector
regions. This is not ordinary symbolic F64 [MATMUL](../../glossary.md#matrix-multiplication--matmul) or all-valid symbolic F64 Scan [admission](../../glossary.md#term-admission).
Ordinary array/checked-segment eligibility and Scan's checked-segment-only boundaries remain.
Unproved vector mapping retains scalar compute; unproved worker [ownership](../../glossary.md#term-ownership-meanings) retains inline execution.
No new [dtype](../../glossary.md#data-type--datatype)/[layout](../../glossary.md#layout)/[carrier](../../glossary.md#term-storage-carrier), general [pointwise](../../glossary.md#term-pointwise) rewrite, native route or automatic all-op Scan [SIMD](../../glossary.md#simd).

Vectorize fixed [feature](../../glossary.md#term-feature)/output cells within one active row, never implicit feature-K splitting.
Use admitted existing vector instructions and masked feature-tail access/[publication](../../glossary.md#term-publication). Feature K
retains its original ordered MUL/ADD sites; unused vector [lanes](../../glossary.md#term-lane) never enter another result's
reduction. Only selected terminal row-axis contractions and represented Scan SUM may regroup
under the [Scan semantic exception](recurrent-scan.md#cpu0074-selected-terminal-grouping-exception).
Standalone MATMUL-K and ordinary exact SUM/[MEAN](../../glossary.md#term-mean), including producer stores/workspaces, remain
unchanged. Same compute/[accumulator](../../glossary.md#accumulator-type) dtype, no implicit [FMA](../../glossary.md#term-fused-multiply-add-fma)/[TF32](../../glossary.md#term-tf32)/lower [precision](../../glossary.md#term-precision) or public numerical
mode. Raw movement/casts/indices/[masks](../../glossary.md#term-boolean-mask), [singleton](../../glossary.md#term-singleton)/empty [identities](../../glossary.md#term-identity-meanings) and selected-realization saved
values remain exact. No cross-strategy/thread-count/[device](../../glossary.md#term-device) bitwise claim.

Strategy, maximum ranges W and minimum row work m remain in existing preparation
`CpuPartitionPreparationPlan.ExecutionUnitPlan`; no worker policy enters [canonical](../../glossary.md#term-canonical-form) row [IR](../../glossary.md#term-intermediate-representation-ir).
Cold [run](../../glossary.md#term-run) [binding](../../glossary.md#term-binding) resolves N=min(W,ceil(R/m)) for R>0, otherwise zero, with quotient/remainder
partitioning of [0,R) in ascending range index. The rule is fixed before execution and across
time, not length-value profitability selection. N<2 executes inline. Each worker owns disjoint
rows and terminal partials; merge valid partials in fixed ascending range-index order on the caller.
Each bounded partial slot has typed terminal cells and an eight-byte validity flag; skip invalid
empty/inactive partials, never invent +0 contributors or infer singleton geometry from activity.
No shared [atomic](../../glossary.md#resource-transaction-and-rollback) floating updates or B×[parameter](../../glossary.md#term-parameter) bank. Keep one complete ordinary execution unit;
internal range/merge helpers, not arbitrary pointwise ranges, implement new non-scalar realizations.
Scalar v1's work=1/[0,1) and rejection behavior remain unchanged.

Qualify actual F3/H2 saved-[primal](../../glossary.md#term-primal) [RNN](../../glossary.md#term-recurrent-neural-network-rnn) reverse as a selected vector/parallel region independently
of a genuinely [fused](../../glossary.md#specialization-and-fusion) forward [fixture](../../glossary.md#term-fixture). An ordinary RNN forward with no eligible terminal region
remains scalar. CPU0010M production scalar/thread-count-one defaults do not change; only existing
explicit eligible configuration/tuning selects new realizations cold.

Cold candidate [metadata](../../glossary.md#term-metadata) uses the appended `CpuFusionDecision.WorkspaceRole.ROW_REGION_STATE`
only for ordinary selected row scratch authenticated by matching row IR/emission and no Scan/
recurrent [owner](../../glossary.md#term-owner). The shared `GENERAL_SCAN_BANKS` preparation tag alone is insufficient. Candidate
construction and retained-plan [validation](../../glossary.md#term-validation) use the same [guard](../../glossary.md#term-guard); actual Scan/recurrent banks remain
outside generic candidate discovery. Internal row-to-terminal uses stay inside the indivisible
unit, not false cross-unit materializations. Ordinary pointwise accounting and scoring are unchanged.

#### Exact new internal seams

MAIN addendum accepted 2026-10-30 after focused independent APPROVE: the two additional
generation/cold-[binding](../../glossary.md#term-binding) declarations below complete the existing CPU0074 packet. No new type,
[runtime](../../glossary.md#term-runtime) protocol or capability-completion claim follows from those declarations.
MAIN/focused-reviewer correction accepted 2026-10-30: the Scan overload also transports
positive minimum row work cold, as specified below; this replaces the unimplemented seven-[argument](../../glossary.md#term-argument) form.
MAIN/Boyle byte-seam addendum accepted 2026-10-30: only the two checked-segment byte-address
generation methods below are added. These accepted seams are implemented and MAIN accepted.

All names are under `io.github.pho001.synaptik.backend.cpu`; no new top-level production type.
The following are non-runnable declaration excerpts, not supported application [API](../../glossary.md#term-application-programming-interface-api):

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

The qualified last two declarations identify methods on their existing [owners](../../glossary.md#term-owner), not Java source
syntax. emitRowRange generates the already-approved scanRowRange helper using the existing private
temporal/row binder; it is not a runtime [callback](../../glossary.md#term-callback). The ordinary symbolic constructor overload
transports existing unit-owned range policy and [borrowed](../../glossary.md#ownership-borrowing-and-lease) workers to cold numeric B binding,
without per-B [artifact](../../glossary.md#term-artifact) selection or worker policy in [canonical](../../glossary.md#term-canonical-form) row [IR](../../glossary.md#term-intermediate-representation-ir). The old six-argument
constructor retains behavior by delegating (1,1,null). The selected parallel Scan entry appends StepDispatch after its existing geometry long[]
and before start/end longs; ordinary and nonparallel entry ABIs remain unchanged. The generated
entry reads context arrays once, then emits bounded frame updates and direct execute calls at
the single selected region. No stage ordinal [lookup](../../glossary.md#term-lookup), callback [registry](../../glossary.md#term-registry) or per-[operation](../../glossary.md#operation) [dispatch](../../glossary.md#term-dispatch).

vectorLoadBytes/vectorStoreBytes accept only checked MEMORY_SEGMENT [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) carriers,
the [dtype](../../glossary.md#data-type--datatype)'s [target](../../glossary.md#term-target) preferred species, a LONG byte-offset local and a matching [mask](../../glossary.md#term-boolean-mask) (and matching
value [vector](../../glossary.md#term-vector) for stores). Emit direct native-byte-order masked fromMemorySegment/intoMemorySegment;
disabled [lanes](../../glossary.md#term-lane) access no storage. [Scalar](../../glossary.md#term-scalar) broadcasts retain existing loadBytes plus broadcast.
Keep old [element](../../glossary.md#term-element)-address methods and array paths literally unchanged. Migrate only selected
exec-v1 Scan byte addresses and ordinary private partial scratch. This removes the emitted
byte-offset /width*width roundtrip; [JIT](../../glossary.md#term-just-in-time-compilation-jit) elimination is unproved and no slowdown/speed claim follows.
No new type/layer, runtime entry [ABI](../../glossary.md#term-application-binary-interface-abi), resource or global [schema](../../glossary.md#term-schema) change. Selected exec-v1 remains
unreleased; final inspection/performance must use a fresh private artifact root so an old
candidate [cache](../../glossary.md#term-cache) cannot hide the changed generated shape.

The Scan overload rejects nonpositive minimumElementsPerWorker with IllegalArgumentException.
CpuPartitionFinalizer passes unit.minimumElementsPerWorker() during cold finalization; policy stays
in the existing unit, not canonical IR, Facts or [specialization](../../glossary.md#term-specialization). The old six-argument Scan
constructor delegates (null,1), preserving its inline behavior. No runtime protocol changes.

RowFusionEmission snapshots nonnull disjoint duplicate-free ordinal lists; each ordinal names a
proved source value. maximumPartialCount is positive; other counts are nonnegative and checked
against emitted frame [layout](../../glossary.md#layout) and existing unit facts. Scalar v1 has no [payload](../../glossary.md#artifact-envelope-and-atomic-publication). Range tables use
typed primitive cells for scalar code or reference locals by fixed [feature](../../glossary.md#term-feature) chunk for vector code;
validate coverage, disjointness and load dominance at generation. Callbacks/handle tables remain
generation-only. Init/finish execute once around [rowBegin,rowEnd), including zero-trip ranges.

Vector init uses the selected target's preferred [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) species, whose bits must match the
specialization, never arbitrary species. Specialization [validation](../../glossary.md#term-validation) must require that dtype's
target preferred species and matching [accumulator](../../glossary.md#accumulator-type), load and mask species; arbitrary species
cannot qualify by equal lane count alone. It initializes typed +0 except omitted direct-[singleton](../../glossary.md#term-singleton)
reset. Loaders push matching vectors; [MATMUL](../../glossary.md#matrix-multiplication--matmul) emits a singleton MUL assignment or separate
per-lane MUL then ADD for K>1, never [FMA](../../glossary.md#term-fused-multiply-add-fma).
Represented SUM emits typed ADD or raw singleton assignment. Finish passes the completed local
to the generation-only sink without arithmetic. Reuse existing [pointwise](../../glossary.md#term-pointwise)/vector math, loop and
[carrier](../../glossary.md#term-storage-carrier) owners, not copied [opcode](../../glossary.md#term-opcode) switches or another [RNN](../../glossary.md#term-recurrent-neural-network-rnn) emitter. All further common [signatures](../../glossary.md#term-signature),
named production types or cross-[owner](../../glossary.md#term-owner) edges require explicit approval.

#### Resources, [dispatch](../../glossary.md#term-dispatch) [lifetime](../../glossary.md#term-lifetime) and compatibility

Analysis fixes all staging/partial/frame facts before resource declarations/slots. F3/H2 reverse
may stage [canonical](../../glossary.md#term-canonical-form) W/U transposes in ten represented cells: 40 [F32](../../glossary.md#floating-point-precision-and-rounding) / 80 [F64](../../glossary.md#floating-point-precision-and-rounding) bytes before [alignment](../../glossary.md#term-alignment).
Stage once per [run](../../glossary.md#term-run) only when [invariant](../../glossary.md#term-invariant) across T and inside any-active gating. SHARED histories/
changing cuts stage at their exact active original time after [producers](../../glossary.md#term-producer), or [vector](../../glossary.md#term-vector) fallback;
never reuse stale data. Reuse that bounded area only after all readers join. Preserve demanded
stores, [carry](../../glossary.md#term-carry) banks, inclusive cut lifetimes and CPU0073 method/local/resource limits. T=0/B=0/
all-inactive executes no such body staging. No per-row [transpose](../../glossary.md#term-transpose) arrays, B×[parameter](../../glossary.md#term-parameter) banks or
T-fold code/[buffers](../../glossary.md#term-buffer). Run-bound context/calls/arrays are O(body+W), never O(T).

Use the [run-owned synchronous reservation](runtime-prepare-engine.md#cpu0074-run-owned-synchronous-reservation).
StepDispatch is private in purpose, public only for generated cross-package [bytecode](../../glossary.md#term-bytecode) access.
One [RunState](../../glossary.md#run-state--runstate) owns it and its [mutable](../../glossary.md#term-mutable) arrays; accessors expose those run arrays without copying.
Calls/helpers are resolved and bound before execution. Per-step work updates only current
operand/time/base facts, never repacks geometry or rediscovers [graph](../../glossary.md#graph)/access/strategy facts.
The exception allows [synchronous](../../glossary.md#term-synchronous) selected-region dispatch and existing calls.clone/Job/failure/
[latch](../../glossary.md#control-flow-graph)/queue [metadata](../../glossary.md#term-metadata) per dispatch, not general callbacks, body allocations or another executor.
Job reuse is optional. Length producers and complete typed/access/[alias](../../glossary.md#aliasing-and-arena)/value [preflight](../../glossary.md#term-preflight) still
precede all Scan-local writes. Worker accessibility includes staging/partial scratch.

Keep envelope 71, ordinary projection 72, Scan projection 71 and canonical row structuralKey.
Only new selected realizations append `|ordinary-row-map-multi-reduction-exec-v1=` or
`|scan-row-map-multi-reduction-exec-v1=` after existing payloads in both [specialization](../../glossary.md#term-specialization) streams.
Bind selected strategy/preferred species, [mask](../../glossary.md#term-boolean-mask)/transpose/lifetime facts, fixed [partition](../../glossary.md#partition)/merge
algorithm, emitted range/frame/partial/helper [layout](../../glossary.md#layout)/types, all code-shaping bounds/resource recipes
and baked geometry/bytes. Exclude actual symbolic B/L, addresses/slots and invocation-resolved bytes.
Unit policy stays preparation-owned; key it where it shapes emitted meaning or compatibility.
[Scalar](../../glossary.md#term-scalar) v1 and noneligible keys/ClassFiles/resources/ABIs remain literal unchanged except for the
implemented [CPU0075 scalar generation/identity cutover](#cpu0075-selective-scalar-load-and-address-reuse)<span data-glossary-navigation="true"> ([identity glossary](../../glossary.md#term-identity-meanings))</span>.
Validate explicit
helper shape, [artifact](../../glossary.md#term-artifact) stale rejection/reuse and B/L-independent reuse without blanket [schema](../../glossary.md#term-schema) bumps.

Acceptance requires each realized eligible domain, actual F3/H2 reverse and separate [fused](../../glossary.md#specialization-and-fusion) forward,
optimal same-specialization Java/ClassFile comparison, no hidden [elementwise](../../glossary.md#term-elementwise) allocation/boxing/
reflection/dispatch, vector tails, exact masks/singletons/empties/saved sites, [finite](../../glossary.md#term-finite) per-op/[dtype](../../glossary.md#data-type--datatype)
tolerances plus [cancellation](../../glossary.md#term-cancellation)/special/raw checks, and budget/access/error/close/[concurrency](../../glossary.md#term-concurrency) tests.
Finite performance [validation](../../glossary.md#term-validation) needs its own task budget; no historical campaign reopens or
universal speed/bitwise/[JIT](../../glossary.md#term-just-in-time-compilation-jit) guarantee follows. MAIN accepted functional capability after final validation;
the [CPU0074 Result](../../planning/backends/cpu/tasks/0074-row-fusion-simd-parallel.md#result)
records final evidence, concrete JIT [compilation](../../glossary.md#compiler-engine-and-execution-stage-names)-pressure limitations and a separate recommended
investigation, not another authorized task/frontier or timing campaign.

#### CPU0075 selective [scalar](../../glossary.md#term-scalar) load and address reuse

Implemented and independently reviewed, 2026-10-30, after explicit user approval. CPU0075 is
Complete after MAIN's final [CPU](../../glossary.md#term-cpu)/[Javadoc](../../glossary.md#term-javadoc) gate and the sole bounded comparison; its
[Result](../../planning/backends/cpu/tasks/0075-row-fusion-scalar-load-address-reuse.md#result)
records evidence and limits. This subsection qualifies only CPU0073/CPU0074 generation and
compatibility retention, not their numerical, eligibility, resource-[ownership](../../glossary.md#term-ownership) or worker semantics.
[ADR 0054](../../design/decisions/0054-row-fusion-scalar-load-address-reuse.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span> explains the decision.

Replace affected ordinary and general-Scan scalar row emission, including existing parallel-scalar
row bodies, in the existing emitter/binder [owners](../../glossary.md#term-owner). Reuse only repeated external loads within one
active row and fold/share proved [constant](../../glossary.md#term-constant) affine address terms and row bases. [SIMD](../../glossary.md#simd) bodies and
unrelated emitters remain literal controls. No W/U hoisting out of the row, across-T [carry](../../glossary.md#term-carry) promotion,
new fuser, [dtype](../../glossary.md#data-type--datatype)/[layout](../../glossary.md#layout)/[carrier](../../glossary.md#term-storage-carrier) capability, public policy, [runtime](../../glossary.md#term-runtime) object, entry-[ABI](../../glossary.md#term-application-binary-interface-abi) [parameter](../../glossary.md#term-parameter),
top-level production type or new common helper seam is authorized. Private helpers in existing
owners are permitted; any needed expansion requires approval before implementation.

The implemented [CPU0076 exception](#cpu0076-prospective-invariant-external-cell-load-staging)
qualifies only this within-row-only load-placement prohibition and affected
scalar retention. CPU0075's completed implementation and evidence are not reinterpreted.

A shared load identifies its authenticated source/storage origin and complete effective affine
cell recipe, including symbolic B coefficients, temporal/bank [binding](../../glossary.md#term-binding) and store [epoch](../../glossary.md#term-epoch-meanings) (the interval
between potentially [aliasing](../../glossary.md#term-aliasing) writes). A value ordinal, equal extents or sampled B is not proof.
Prove [alias](../../glossary.md#aliasing-and-arena)/[view](../../glossary.md#term-view-meanings) equivalence, unchanged storage across all intervening retained/public/scratch
writes, activity and first-load dominance on every path. A possibly aliasing store ends that reuse
interval; unknown equivalence keeps a direct checked load. Neither loads nor address-dependent
access move above existing row/time/empty guards. Preserve checked spans, [alignment](../../glossary.md#term-alignment), [overflow](../../glossary.md#term-overflow),
raw inactive [identities](../../glossary.md#term-identity-meanings) and all required stores and floating instruction order.

Cold analysis before resource declarations/slots must conservatively bound the replacement's
simultaneously live typed values, cached loads and addresses as well as existing bindings and
accumulators. Retain the 128-slot and method-size limits; doubles/longs count two. Allocate cached
cells only for repeated uses, [release](../../glossary.md#term-release) at last use and reuse type-compatible local slots; do not
retain every external cell through the row. [JVM](../../glossary.md#term-java-virtual-machine-jvm) max_locals is not hardware-register [liveness](../../glossary.md#term-liveness).
If reuse cannot be proved or bounded, use unshared accesses within the replacement or reject the
candidate cold through the existing fallback. Finalization cannot discover resources or choose a
new route. No maintained parallel legacy scalar emitter or runtime discovery follows.

Only affected scalar/parallel-scalar specializations append the scoped
`|ordinary-row-scalar-load-address-reuse-v1=` or `|scan-row-scalar-load-address-reuse-v1=` marker to
both `compatibilityBytes()` and `classIdentityBytes()`. Bind every deterministic code-shaping
reuse/address/proof choice using existing [specialization](../../glossary.md#term-specialization) facts; exclude actual symbolic B/L,
addresses, assigned slots and invocation-resolved bytes. Keep envelope 71, ordinary/Scan
projections 72/71, [canonical](../../glossary.md#term-canonical-form) structural keys and all existing namespaces/[payload](../../glossary.md#artifact-envelope-and-atomic-publication) versions unchanged.
Old affected scalar [artifacts](../../glossary.md#term-artifact) are stale under the new keys; artifacts outside this cutover keep
literal keys/ClassFiles/resources and entry/helper ABIs. Test both streams and B/L-independent reuse.
Historical `.tsv` and `.gz.b64` controls are not regenerated from candidate code: affected snapshots
prove selective stale rejection, while unaffected SIMD/nonregional snapshots remain literal controls.
Captured old [bytecode](../../glossary.md#term-bytecode) may be a test-only comparison arm, never a production emitter alternative.
The implemented scalar path ends value reuse at every retained write, excludes changing Scan
carry/scratch bindings, and leaves alternative SUM loads unshared with address-recipe invalidation.
Cold analysis reserves remaining capacity under the existing 128-slot limit; emission caps its
typed pool and single LONG offset within that capacity. Unavailable capacity preserves checked
unshared reads, not a changed route or broader SIMD rejection.
The [CPU0075 brief](../../planning/backends/cpu/tasks/0075-row-fusion-scalar-load-address-reuse.md#result)
records the exhausted separate [validation](../../glossary.md#term-validation) budget. Reduced bytecode load sites are structural
evidence, not a matched-[JIT](../../glossary.md#term-just-in-time-compilation-jit) speed guarantee or [C1](../../glossary.md#term-c1) repair; no material Scan speedup was established.
CPU0074 and all historical campaigns remain closed; defaults and rollout are unchanged.

<a id="cpu0076-prospective-invariant-external-cell-load-staging"></a>

#### CPU0076 [invariant](../../glossary.md#term-invariant) external-cell load staging

Accepted after explicit user approval, 2026-10-31; implemented with independent source/protocol,
bounded-result and documentation approval, 0 P1/P2, and MAIN's final [CPU](../../glossary.md#term-cpu)/[Javadoc](../../glossary.md#term-javadoc)/named [Engine](../../glossary.md#term-engine) PASS.
This is the sole owning CPU load-placement/compatibility exception to
CPU0075's within-row-only prohibition. [ADR 0055](../../design/decisions/0055-row-invariant-load-staging.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>
explains the choice; the [CPU0076 brief](../../planning/backends/cpu/tasks/0076-row-invariant-load-staging.md)
owns execution and the separate [finite](../../glossary.md#term-finite) measurement grant. Prior completed results stay historical.

An external cell is a represented value read from an authenticated storage origin, not a row
[producer](../../glossary.md#term-producer) or [accumulator](../../glossary.md#accumulator-type) local. Existing [homogeneous](../../glossary.md#term-homogeneous) [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) ordinary and general-Scan
[scalar](../../glossary.md#term-scalar) row regions may stage such cells in typed method locals outside the row loop when the
following proof succeeds. This is generic source/access analysis, never an [RNN](../../glossary.md#term-recurrent-neural-network-rnn), F3/H2, ordinal,
root-number or [weight](../../glossary.md#term-weight)-value special case. Retain existing static/named-B/length/combined [admission](../../glossary.md#term-admission),
ordinary array/checked-segment and Scan checked-segment boundaries; ordinary symbolic [F64](../../glossary.md#floating-point-precision-and-rounding) [MATMUL](../../glossary.md#matrix-multiplication--matmul)
and all-valid symbolic F64 Scan gain no [qualification](../../glossary.md#term-qualification).

Authenticate the source, complete [view](../../glossary.md#term-view-meanings)/storage origin and effective affine cell recipe for all
admitted B: fixed and B coefficients, row coefficients, byte base/[strides](../../glossary.md#term-stride), original-time source,
bank selection and store [epoch](../../glossary.md#term-epoch-meanings). Prove structural row independence, nonempty consuming-domain and
first-load dominance, source readiness, and storage/mapping stability throughout the staged
value's complete [lifetime](../../glossary.md#term-lifetime). Equality of extents, [nominal](../../glossary.md#term-nominal) ordinals, copied history values or sampled
B is not proof. Include every retained/public/saved/[carry](../../glossary.md#term-carry)/scratch write and inclusive cut lifetime;
actual [binding](../../glossary.md#term-binding) still checks backing overlap/access. Unknown [alias](../../glossary.md#aliasing-and-arena), mapping, lifetime or dominance
keeps current checked loads. An unrelated write need not end a proved disjoint cell's lifetime;
a possibly [aliasing](../../glossary.md#term-aliasing) write does. No new alias permission or unchecked access follows.

True invocation invariants load once per generated entry invocation, after existing [preflight](../../glossary.md#term-preflight)
and nonempty/any-active guards, before their consuming row loop; they may stay local across T
only with proved temporal invariance. A [lazy](../../glossary.md#term-lazy-evaluation) first-consuming-active-time placement is permitted
if it dominates every use and performs no extra read. A SHARED sequence/history is not an
invocation invariant: stage only after its producer at each consuming active original t, under
active(t), with the exact original-time base. ROW histories, evolving carries and bank/scratch
bindings retain their existing loads in this implementation. Authenticate the SOURCE activity
domain: selecting a [constant](../../glossary.md#term-constant) row from a ROW history does not permit global-active-time staging.
Already-hoisted typed scalar roots are excluded from new placement and its marker. Neither a
scalar-broadcast initializer, zero stride nor SHARED role proves temporal invariance.
The [Scan owner's activity and history rules](recurrent-scan.md#activity-raw-identity-and-retained-histories)
remain unchanged. T=0/B=0/all-inactive/empty-[feature](../../glossary.md#term-feature) paths perform no speculative body staging;
required nonempty shared zero [publication](../../glossary.md#term-publication) and raw inactive carry [identities](../../glossary.md#term-identity-meanings) still execute.

Existing parallel-scalar may stage only inside its current method/range bindings when this proof
and local lifetime fit the existing helper [ABI](../../glossary.md#term-application-binary-interface-abi). Worker staging additionally requires a consuming
active row in that worker's range: global active(t) alone does not qualify an all-inactive range.
No transport of typed locals between caller and worker, frame extension, new staging bank,
[dispatch](../../glossary.md#term-dispatch)/reservation change or cross-worker [cache](../../glossary.md#term-cache) is authorized. A helper-preheader load is once
per helper invocation, not once per whole Scan across repeated dispatches. If the intended wider
lifetime cannot fit, retain the current checked path and record that precise qualification.
Inline scalar is the primary acceptance path.

The implemented proof accepts only complete, actually consumed fixed-feature view domains;
[rank](../../glossary.md#term-rank) greater than two and partial/unused/empty domains remain checked. For Scan, every nonformal
self-rooted producer must belong to the region's row sites, terminals or finish [epilogue](../../glossary.md#term-epilogue).
Otherwise the binding [owners](../../glossary.md#term-owner) supply empty staging scopes, even when one region in a larger body
would pass an isolated proof. This bounds the full invocation lifetime, not just the pruned
regional stage. Small ordinary F64 staging is qualified; the larger combined F64 primary geometry
retains CPU0075 emission. None of these optimization limits reduces existing execution capability.

Cold proof before cuts, resource declarations and slots bounds staging plus bindings, addresses,
producer values, terminal accumulators and controls, separately for caller/helper phases,
including inline scalar without exec-v1 [metadata](../../glossary.md#term-metadata). Retain all CPU0073 limits, especially 128
simultaneously live [JVM](../../glossary.md#term-java-virtual-machine-jvm) slots (DOUBLE/LONG count two) and 48000-byte method bound; use last-use/
type-compatible slot reuse where valid. Do not increase resource limits, scratch/frame recipes
or entry/helper [signatures](../../glossary.md#term-signature). Finalization emits the fixed proof, not late resource discovery.
Unproved or over-budget staging retains CPU0075 checked emission or existing cold fallback
without shrinking already-supported capability or rejecting an otherwise eligible strategy.
No parallel maintained old emitter or [runtime](../../glossary.md#term-runtime) selection is added.

The applicable structural no-duplicate caller/helper floors independently fit conservative 112/116
non-pool reserves; they include rooted stride/address facts, typed producers and terminals,
bindings, controls and CPU0075's LONG offset. Additional wrapper allowances cannot conceal an
excessive floor. Let D count extra demanded wrappers per producing row origin and H count ALL
demanded history ports, including forwarded formals and SHARED histories outside regional row
sites. Every binding [owner](../../glossary.md#term-owner) uses caller base `112 + 3*D + 2*H` and helper base `116 + 3*D`:
each extra wrapper includes a [carrier](../../glossary.md#term-storage-carrier) and LONG base in both phases; every history retains its
own LONG caller base across prepublication and staging. Ordinary has no Scan history allowance.
For typed width w and distinct staged-cell slots S, each applicable CPU0075 pool is capped by
both its prior bound and `floor((128 - phaseBase - S - 1) / w)`. The one-slot readiness allowance
is conservative for inline emission and covers the helper's first-active-consumer flag.
Unproved/over-budget selection is empty BEFORE cuts/resources/slots, not a strategy rejection.

Only load placement and proved address work may change: preserve ordered typed MUL/ADD/SUB,
operand/contributor order, retained stores and saved sites, ordinary exact SUM/[MEAN](../../glossary.md#term-mean), and distinct
memory-backed carry ADDs after each active-time reduction. No carry promotion, [FMA](../../glossary.md#term-fused-multiply-add-fma), arithmetic
regrouping or [SIMD](../../glossary.md#simd)-body rewrite. Add no top-level production type, public [API](../../glossary.md#term-application-programming-interface-api)/policy, capability,
runtime object, generic framework or [ownership](../../glossary.md#term-ownership-meanings)/[dependency](../../glossary.md#term-dependency) change. Private helpers in existing
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
read-only and deterministic: no runtime storage/input reads, [identity](../../glossary.md#term-identity-meanings) parsing, class generation,
allocation-derived eligibility, measured profitability or route/resource decision.

Return an [immutable](../../glossary.md#term-immutable) ordered list of actual consumer-view ordinals for complete fixed-feature
cell-domain staging, with deterministic origin/affine deduplication. Every staged cell must have
an actual consumer; partial, unused or unproved domains stay checked. In this one owner, enumerate
exact consuming requests/retained-write epochs and prove source/view origin, all-B effective
affine equality and zero row coefficients under the authenticated temporal/full-lifetime alias
and write scopes. Private helpers here may implement the proof; competing private eligibility
or selection in callers is forbidden.

The binding owner supplies disjoint authenticated read-only source-origin lists: invocation
invariants versus active-original-time sources. Membership is by storage origin, not consumer
ordinal/[shape](../../glossary.md#shape)/SHARED alone. Carries and uncertain cut/bank/scratch origins are absent. These same
authenticated [scope](../../glossary.md#term-scope) lists AND conservative structural caller/helper base-live-slot facts must
drive pre-cut [lowering](../../glossary.md#lowering) (`prove` and existing `CpuGeneralScanLowering` row-region/`shapedResources`
call paths), `CpuPartitionPreparer` analysis, both [specialization](../../glossary.md#term-specialization) identity streams, and emission.
No phase substitutes allocation-derived counts or a privately recomputed selection.

Base-live-slot [arguments](../../glossary.md#term-argument) cover all non-pool state: bindings, controls, addresses, row/terminal
locals and fixed CPU0075 state. Zero helper bound means no helper; DOUBLE/LONG count two. Jointly
bound staging, [ready](../../glossary.md#term-ready)/control flags and the remaining CPU0075 typed pool within 128 slots in each
applicable phase; retain the existing method bound. Facts exclude actual B/L, addresses, slots
and measured cost. Unknown proof, checked-arithmetic failure or over-budget optimization returns
empty: retain the CPU0075 checked path without rejecting an otherwise eligible capability or
strategy. Malformed/null metadata keeps existing IllegalArgumentException/NullPointerException
conventions; it is not silently accepted as optimization failure.

Emission binds only this selected complete cell set to locals. The existing binding owner still
authenticates source readiness, temporal scopes, all writes/aliases over the FULL staged lifetime,
activity/nonempty dominance and actual span/[alignment](../../glossary.md#term-alignment)/overlap preflight. The method grants none
of those permissions by itself; worker-local staging cannot cross the ABI. Cache appends the
affected-only marker iff this same selection changes emitted scalar placement; empty/no-stage,
SIMD and nonregional cases keep literal controls. No [validation](../../glossary.md#term-validation)/performance grant is added.

Only scalar cases whose emitted placement changes append
`|ordinary-row-invariant-load-staging-v1=` or `|scan-row-invariant-load-staging-v1=` in BOTH
`compatibilityBytes()` and `classIdentityBytes()`, after existing payloads. Bind deterministic
source/affine/temporal/store-lifetime/activity placement and local-budget choices through existing
facts and the scoped algorithm marker. Exclude actual symbolic B/L, addresses, slots and
invocation-resolved bytes. Preserve [canonical](../../glossary.md#term-canonical-form) keys, envelope 71, ordinary/Scan projections 72/71
and prior namespace/[payload](../../glossary.md#artifact-envelope-and-atomic-publication) versions. No-staging cases retain CPU0075 identities/bytes/resources;
SIMD/nonregional controls and immutable historical `.tsv`/`.gz.b64` resources remain literal.
Test affected old-[artifact](../../glossary.md#term-artifact) rejection, current generate/load/reuse and B/L-independent reuse in
both streams. Captured old code is test-only evidence, never a production alternative.

The [CPU0076 Result](../../planning/backends/cpu/tasks/0076-row-invariant-load-staging.md#result)
records actual scalar [F32](../../glossary.md#floating-point-precision-and-rounding) [Compiler](../../glossary.md#term-compiler)-generated reverse Scan, generic typed/temporal/helper tests,
resource fallbacks, literal retention and the sole MAIN-released comparison. Its bounded B256
improvement does not imply universal speed, Java parity or [C1](../../glossary.md#term-c1) repair; B32 is [JIT](../../glossary.md#term-just-in-time-compilation-jit)-confounded.
The off-tree Java experiment remains historical input evidence. All measurement grants are now
spent; defaults, rollout and stage/[commit](../../glossary.md#term-commit)/push remain unchanged or unauthorized.

### Accepted ordinary symbolic-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [CPU](../../glossary.md#term-cpu) contract

The implemented [CPU0074 selected-region exception](#cpu0074-selected-row-region-execution-exception)
qualifies only new exec-v1 [F32](../../glossary.md#floating-point-precision-and-rounding) row regions here. Other [scalar](../../glossary.md#term-scalar)-only, typed-control, [F64](../../glossary.md#floating-point-precision-and-rounding) exclusion
and exact ordinary SUM/[MEAN](../../glossary.md#term-mean) rules remain unchanged; no general symbolic [SIMD](../../glossary.md#simd)/worker [admission](../../glossary.md#term-admission).

Accepted on 2026-10-24 and implemented by
[Prepare0013](../../planning/modules/prepare/tasks/0013-ordinary-dynamic-batch-cpu-vertical.md#result).
Final independent reviews and MAIN's native-enabled integration gate passed;
MAIN accepted the capability on 2026-10-24. The implemented [scope](../../glossary.md#term-scope) is
[homogeneous](../../glossary.md#term-homogeneous) [FLOAT32](../../glossary.md#term-float32) ordinary generated scalar, single-range CPU, one named B, fixed [features](../../glossary.md#term-feature)/
[parameters](../../glossary.md#term-parameter), [rank](../../glossary.md#term-rank)-at-most-two boundaries plus needed [AD](../../glossary.md#autograd) [views](../../glossary.md#term-view-meanings). Exact symbolic capability answers
must qualify every final forward and actual first-order [derivative](../../glossary.md#term-derivative) [operation](../../glossary.md#operation). Native/[OpenBLAS](../../glossary.md#term-openblas),
[Metal](../../glossary.md#term-metal), mixed-[owner](../../glossary.md#term-owner), other [dtype](../../glossary.md#data-type--datatype)/family extensions, [Vector API](../../glossary.md#term-vector-api), parallel and [general Scan](../../glossary.md#general-differentiable-scan) support
do not follow. Prepare0015 implements only the bounded additional occurrences below.

Prepare0015's bounded outer prerequisite is implemented; owning [validation](../../glossary.md#term-validation) and independent
code/structural review passed, with root validation and the sole bounded probe completed (evidence and limits:
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) );
final MAIN acceptance remains pending.
Retain the existing ordinary FLOAT32 domain. Added floating occurrences use homogeneous FLOAT32
or [FLOAT64](../../glossary.md#term-float64), at most one exact named B per value, fixed other extents and checked int-bounded
domains. Rank-at-most-two endpoints and the specific `[T,B,F]`, ROW(1) literal/zero `[F,B]` /
`[T,F,B]` mappings plus their required [singleton](../../glossary.md#term-singleton)-feature rank edits qualify, not arbitrary rank-three
algebra. T/F are fixed; named [axes](../../glossary.md#term-axis), not F=B coincidence, determine mappings. Each ordinary provider
predicate must be explicit in family, dtype, [shape](../../glossary.md#shape)/rank/axis, attributes and access. An identical
qualifying [occurrence](../../glossary.md#term-occurrence) gets the same answer inside or outside Scan closure; no origin tag/[registry](../../glossary.md#term-registry).

The table bounds the added occurrences; all retain existing [Model](../../glossary.md#term-model) family semantics and flags.

| Occurrence | Required [descriptor](../../glossary.md#term-descriptor)/attribute/mapping predicate |
|---|---|
| [Seed](../../glossary.md#term-seed)/zero/one/literal [constants](../../glossary.md#term-constant) | Existing registered typed splat ingress for the bounded shapes above, including ROW(1) constants/zeros; no new constant protocol. |
| ADD and required MUL/DIV | Same floating dtype; exact ordinary [tensor](../../glossary.md#tensor) broadcast/result shape and no-operation attributes, or existing same-dtype scalar-value attributes with shape preserved. Preserve the identified B [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) under [broadcasting](../../glossary.md#broadcasting); no other ordinary F64 arithmetic family is added. |
| SUM/MEAN and [SUM-to-Shape](../../glossary.md#sum-to-shape) | Existing no-axis, [normalized axis](../../glossary.md#normalized-axis)/ordered-axes/keep-[dimensions](../../glossary.md#dimension) or SUM-to-Shape [target](../../glossary.md#term-target) attributes; exact family result shape, actual contributor count and proved read/write mapping. Use the typed exact ordinary algorithm below, not Scan's activity reduction. |
| EXPAND, feature-only RESHAPE/EXPAND_DIMS/SQUEEZE | Exact target/axis attributes, ordinary broadcast or checked feature-product equality, singleton insertion/removal only; preserve B without flattening/mixing it into features. |
| [Transpose](../../glossary.md#term-transpose)/[CONTIGUOUS](../../glossary.md#term-contiguous)/same-dtype CAST | Valid exact [permutation](../../glossary.md#permutation)/result shape, no-operation attributes for CONTIGUOUS with shape unchanged, or exact same-dtype CAST target; require a proved [canonical](../../glossary.md#term-canonical-form)/transpose/expand mapping for these bounded shapes and preserve raw semantics. |
| Length CAST control [producer](../../glossary.md#term-producer) | Only non-[gradient](../../glossary.md#term-gradient) FLOAT32 `[B]` or FLOAT64 `[B]` to non-gradient [INT64](../../glossary.md#data-type--datatype) `[B]`, exact CAST target and unchanged shape, under existing cast semantics. No general integral arithmetic/cast family; direct INT64 lengths requires no producer. |

Added cases use checked-segment carriers with exact per-value four/eight-byte spans/[alignment](../../glossary.md#term-alignment),
proved zero-offset canonical/transpose/expand reads and canonical [injective](../../glossary.md#term-injective) writes; preserve existing
qualified read views and reject unresolved/unproved mappings, unsupported carriers and writable
overlap. This [qualification](../../glossary.md#term-qualification) requires no new [heap](../../glossary.md#term-heap)-[carrier](../../glossary.md#term-storage-carrier) support. Validate fixed-feature products
even behind empty T/B, then actual counts/[strides](../../glossary.md#term-stride)/spans/bytes/aggregate bounds before borrow/allocation;
recheck physical access at [binding](../../glossary.md#term-binding). The separate eight-byte INT64 control boundary does not make
body arithmetic mixed-dtype. Ordinary symbolic F64 [MATMUL](../../glossary.md#matrix-multiplication--matmul), arbitrary rank-three operations and
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
null/invalid/[overflow](../../glossary.md#term-overflow)/index failures use the corresponding exceptions in the
[resource contract](runtime-prepare-engine.md#accepted-symbolic-batch-resource-contract).
Recipes are [immutable](../../glossary.md#term-immutable)/[thread-safe](../../glossary.md#term-thread-safe)/storage-free; all resolved access/packed arrays belong to one
[run](../../glossary.md#term-run). BatchBinding validates equal-rank lists and nonnegative coefficients/base; [extent](../../glossary.md#term-extent)/stride
`constant+factor*B` is in [elements](../../glossary.md#term-element) with checked counts/spans. One B-bearing axis per represented
shape and proved canonical/transpose/expand mappings only. Preserve exact Model empty strides:
`[F,0]` has `[0,1]`, never max(1,B). Regimes/axis roles stay prepared; no representative fake B.
BatchByteSize computes checked `constantBytes+bytesPerBatch*B`, with nonnegative coefficients,
positive supported element width, divisibility and CPU representability limits. Its run-bound
marker remains absent even when bytes happen to be constant.

MATMUL binds existing Geometry/pack once per invocation: forward M/K/N=B/F/H, dX=B/H/F,
dW=F/B/H, preserving actual transposed strides. Analysis fixes route/carrier/code independently
of B. Unsupported carriers reject, never select another [artifact](../../glossary.md#term-artifact) at admission. Aggregate geometry
binds actual domains/output counts and conditional workspace, not maximum banks. For the admitted
FLOAT32/int-limited N, signedBits=278+ceil(log2(N)), [cardinality](../../glossary.md#term-cardinality) term zero at N<=1, is at most
309 bits: retain constant-five-[limb](../../glossary.md#exact-superaccumulator-and-limbs) CpuExactSumEmitter for all admitted N, including 0/1.
Single-range scratch is 48 bytes (`8+8*5`) when outputCount>0, otherwise zero. SUM-to-Shape
handles B=1 copy/singleton at entry in the same code/route/slot. Zero scratch does not remove or
discover requirements. Preserve actual count division/numerics; long domains remain unqualified.

For implemented bounded ordinary FLOAT64 SUM/MEAN, retain the existing typed exact
sum/rational-mean algorithm: `signedBits=2099+ceil(log2(N))`, with cardinality term zero for N<=1.
Int-bounded N needs at most 2130 bits, so use fixed 34-limb state and 280-byte (`8+8*34`)
single-range scratch when outputCount>0, otherwise zero while retaining its declared slot/creator.
Preserve [subnormals](../../glossary.md#term-subnormal), signed zeros, [NaN](../../glossary.md#term-nan)/Inf, empty domains and exact rational-MEAN [rounding](../../glossary.md#term-rounding) with no
FLOAT32 [narrowing](../../glossary.md#term-narrowing) or FLOAT64 [DAZ](../../glossary.md#term-daz)/[FTZ](../../glossary.md#term-ftz). Retained FLOAT32 stays five limbs/48 bytes, including 0/1;
static FLOAT64 sizing remains domain-sized (N=3: 33 limbs/272 bytes). Scan producing-site SUM
remains its distinct typed sequential algorithm; this ordinary qualification changes no Scan numerics.

CPU analysis proves canonical write [injectivity](../../glossary.md#storage-mapping-contiguity-and-injectivity) during preparation; run binding checks numeric bounds, never Model
lowerers or element-enumerating injectivity. CPU state-aware representation hooks validate actual
bytes from state.geometry(), retaining type/access/alignment/read-only/overlap checks. Existing
bindCompatible/Invocation keeps direct typed carriers and fresh packed/access arrays; no tenth
type, shared currentB/ThreadLocal, per-element recipe [dispatch](../../glossary.md#term-dispatch) or hidden generic fallback overhead.

Dynamic [identities](../../glossary.md#term-identity-meanings) use `ordinary-batch-v1` to distinguish changed baked meaning in both existing
[specialization](../../glossary.md#term-specialization) streams; actual B/bytes/slots/addresses are excluded. Retain static keys/generated
bytes and existing [IR](../../glossary.md#term-intermediate-representation-ir) numeric formats. Migrate inward creator consumers without deleting useful
static/native paths. No type move/removal, new [module](../../glossary.md#term-module), [dependency](../../glossary.md#term-dependency) or executor is selected.

The implemented [CPU0073 mechanism](#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion)
permits only its exact internal types, optional region and selected-[fusion](../../glossary.md#term-fusion) [identity](../../glossary.md#term-identity-meanings) suffixes.
It changes neither this symbolic admission nor exact ordinary SUM/MEAN; noneligible identities
and ClassFiles retain the rules above. Its validation and limits are recorded in CPU0073.

Added outer F64/control/new rank-three cases use `ordinary-batch-typed-v1` in IR and BOTH existing
specialization streams, including exact boundary dtype/carrier, mapping and reduction-state facts;
actual B/L/bytes/slots/addresses remain excluded. Existing FLOAT32 `ordinary-batch-v1`, fixed/static
identities and generated bytes remain identical. Positive/negative provider and validated-generation
tests plus artifact load/reuse/stale-rejection tests must qualify the typed namespace, independently
from combined Scan identities. These tests pass for the implemented typed branch; an identity alone
does not grant any wider family or dtype capability.

Qualification must [compile](../../glossary.md#compile)/query genuinely symbolic B and execute actual first-order closure
before general Scan: reuse one [compilation](../../glossary.md#compiler-engine-and-execution-stage-names)/preparation for 64→40→1→64; check actual resources,
descriptors and fixed parameter gradients, B=0/1, constant-byte batch-required mode, mismatch,
overflow, failure/cleanup/retry/[concurrency](../../glossary.md#term-concurrency) and static regressions. The reference is optimal clean
Java for the same specialized algorithm/hot-loop/[dataflow](../../glossary.md#term-dataflow), including five-limb SUM. Inspect generated
Class-Files for hidden allocation/boxing/reflection/per-element binding or dispatch. Bounded warmed
matched-work A/B uses invocation B on both sides and separates cold generation/classload/[JIT](../../glossary.md#term-just-in-time-compilation-jit),
admission and [kernel](../../glossary.md#kernel) costs. Disclose entry/harness costs when generated [MethodHandle](../../glossary.md#native-numerical-library-and-instruction-architecture) entry is
compared with a direct Java call; a measured ratio is not an emitted-inner-loop-only measurement.
Diagnose/justify overhead; no identical-JIT or zero-penalty promise.
Prepare0013 records semantic and Class-File checks against specialized clean Java and bounded
generated/direct and dynamic/static measurements. The original harness measurements precede the
final MATMUL loop repair; one separately authorized private post-fix probe also completed. Short-window
[variance](../../glossary.md#term-variance), the unmatched medium [Engine](../../glossary.md#term-engine) route policy and a remaining exact-MEAN probe gap are
disclosed, not evidence of steady-state parity, negligible overhead or an all-B speed guarantee.
Final code/[Javadoc](../../glossary.md#term-javadoc)/performance and shared-document reviews approve with zero open P1/P2.
MAIN's native-enabled architecture/[conformance](../../glossary.md#term-conformance)/integration join passed and MAIN accepted the
capability on 2026-10-24; this status update runs no measurement.

### CPU0077 [finite](../../glossary.md#term-finite)-[softmax](../../glossary.md#softmax--log-softmax) [admission](../../glossary.md#term-admission) parity

The [CPU0077 checkpoint](../../planning/backends/cpu/tasks/0077-f64-finite-softmax-admission-parity.md#result)<span data-glossary-navigation="true"> ([checkpoint glossary](../../glossary.md#term-checkpoint))</span>
records the earlier finite-input correction and its bounded verification. The subsequent
interchangeability repair supersedes its finite-only admission and unchanged-class statement.

Current SOFTMAX/LOG_SOFTMAX bodies preserve the [Model](../../glossary.md#term-model)'s finite, nonfinite and empty-[slice](../../glossary.md#slice) rules.
All-finite slices retain maximum-shifted arithmetic and compensation, including represented
negative-[infinity](../../glossary.md#term-infinity) shifts caused by finite subtraction [overflow](../../glossary.md#term-overflow). A [NaN](../../glossary.md#term-nan) input makes the complete
slice NaN. With [positive infinity](../../glossary.md#nan-infinity-and-signed-zero) but no NaN, positive-infinity positions are NaN; other positions
are [positive zero](../../glossary.md#nan-infinity-and-signed-zero) for SOFTMAX and [negative infinity](../../glossary.md#nan-infinity-and-signed-zero) for LOG_SOFTMAX. Only slices with negative
infinity but neither NaN nor positive infinity use the literal unshifted Model formula.
Any zero [extent](../../glossary.md#term-extent) gives an empty [shape](../../glossary.md#shape)-preserving output with no input read or output write.
Checked geometry, carriers, [aliases](../../glossary.md#aliasing-and-arena), thread access and resource [ownership](../../glossary.md#term-ownership) remain independently
validated. No finite-only validator is bound before ordinary or Scan execution; [producer](../../glossary.md#term-producer) values
are consumed directly by the numerical body.

[CPU](../../glossary.md#term-cpu)-private softmax algorithm version 3 invalidates the former finite-only generated bodies,
including retained composition helpers. It does not change [loss](../../glossary.md#loss)/attention formulas, invent a
numerical policy or [runtime](../../glossary.md#term-runtime) fallback. Low arithmetic separately follows the parity cutover below.

### CPU0078 temporary low-arithmetic lock

CPU0078 records the historical retain/freeze/lock decision and its unchanged historical
[qualification](../../glossary.md#term-qualification) in [ADR0060](../../design/decisions/0060-temporary-cpu-low-arithmetic-admission-lock.md)
and [CPU0078](../../planning/backends/cpu/tasks/0078-temporary-low-arithmetic-lock.md#result).
The subsequent user-approved [CPU](../../glossary.md#term-cpu) low-[precision](../../glossary.md#term-precision) parity decision supersedes its [admission](../../glossary.md#term-admission) lock;
its old negative capability answers are not current authority.

### [CPU](../../glossary.md#term-cpu) low-[precision](../../glossary.md#term-precision) parity cutover

[BFLOAT16](../../glossary.md#term-bfloat16), [FLOAT16](../../glossary.md#term-float16), FLOAT8_E4M3 and FLOAT8_E5M2 numerical execution must reuse the ordinary
CPU capability, [lowering](../../glossary.md#lowering), [IR](../../glossary.md#term-intermediate-representation-ir), preparation, generated bodies, retained plans and invocation [owners](../../glossary.md#term-owner).
Remove obsolete CPU0078 [admission](../../glossary.md#term-admission) methods/callers and lock-only tests rather than retaining
an always-true shim or a [runtime](../../glossary.md#term-runtime) policy switch. Keep exact geometry, [layout](../../glossary.md#layout), [gradient](../../glossary.md#term-gradient), [alias](../../glossary.md#aliasing-and-arena),
resource and [lifecycle](../../glossary.md#lifecycle) checks. The [Model](../../glossary.md#term-model) low-precision and [OFP8](../../glossary.md#term-ofp8) contracts own arithmetic and
conversion; restored code is not qualified merely because it was retained behind the lock.

The [target](../../glossary.md#term-target) is functional parity with existing CPU [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) [operation](../../glossary.md#operation) families, including actual
[Compiler](../../glossary.md#term-compiler) first-order [AD](../../glossary.md#autograd), ordinary/fixed/length-aware Scan, supported symbolic geometry, repeated
training and source/value [persistence](../../glossary.md#term-persistence). [FP8](../../glossary.md#term-fp8) stores bytes, not [BOOL](../../glossary.md#data-type--datatype) or short aliases. [Working
arithmetic](../../glossary.md#accumulator-type) and accumulators are F32, with one final declared output [narrowing](../../glossary.md#term-narrowing). Existing exact
and discrete paths preserve all [raw bits](../../glossary.md#storage-mapping-contiguity-and-injectivity). No separate interpreter, helper-[dispatch](../../glossary.md#term-dispatch) executor,
vendor AD, [backend](../../glossary.md#backend) [registry](../../glossary.md#term-registry), numerical selector or execution-time fallback is authorized.

[oneDNN](../../glossary.md#term-onednn) remains a cold-selected CPU realization. Typed native recipes authenticate each input
and output [dtype](../../glossary.md#data-type--datatype), [referenced span](../../glossary.md#referenced-element-span), origin, [carrier](../../glossary.md#term-storage-carrier) [alignment](../../glossary.md#term-alignment) and scratch requirement. Strict F32
working/accumulation is required for low inputs. Native destination conversion is usable only
when it proves Model [rounding](../../glossary.md#term-rounding); otherwise declare an F32 result workspace and perform the exact
final narrowing without erasing a logical output or saved boundary. This is not permission to
promote FP8 inputs into F32 inputs and advertise native FP8 execution.

Primitive and [Graph](../../glossary.md#graph) availability are independent. In pinned oneDNN 3.14, Graph [MATMUL](../../glossary.md#matrix-multiplication--matmul) does not
admit FP8, and CPU native F64 is unavailable. FP8 primitives require a supported x86 target;
unavailable native configurations select the ordinary CPU candidate before execution. No native
FP8 execution is claimed on Arm, no native availability is inferred from dtype enum tags, and no
relaxed/low [accumulator](../../glossary.md#accumulator-type) mode is used. Existing bare-MATMUL native geometry restrictions remain
until independently qualified; [OpenBLAS](../../glossary.md#term-openblas) stays F32/F64-only.

Affected low numerical [artifacts](../../glossary.md#term-artifact) and new FP8 recipes need selective append-only compatibility
[identities](../../glossary.md#term-identity-meanings). Preserve unchanged F32/F64 and exact raw recipes, retained [ABI](../../glossary.md#term-application-binary-interface-abi) and generated bytes.
Qualify representations, ties/[overflow](../../glossary.md#term-overflow)/[underflow](../../glossary.md#term-underflow), [NaN](../../glossary.md#term-nan)/[infinity](../../glossary.md#term-infinity)/zero signs, contributor and [mask](../../glossary.md#term-boolean-mask)
domains, native/ordinary execution, actual AD, resource lifetimes and [transaction](../../glossary.md#term-transaction) [rollback](../../glossary.md#term-rollback).
Positive native evidence and unsupported-target rejection evidence are separate claims.

### [General Scan](../../glossary.md#general-differentiable-scan) ordinary-unit realization

Implemented coordinated Class C realization, accepted with Compiler0017/CPU0081 after independent
authority/interface and exact-source implementation reviews, actual ordinary forward/selected-
backward/typed/activity/boundary/raw consumers and the unfiltered affected test/[Javadoc](../../glossary.md#term-javadoc) gate.
It realizes the [Scan semantics](recurrent-scan.md#general-ordinary-unit-scan-cutover) through
ordinary [owners](../../glossary.md#term-owner), without [widening](../../glossary.md#term-widening) cold/wire, [Metal](../../glossary.md#term-metal) Scan, low numerical or universal performance
capabilities. Genuine qualified old recipes retain their exact key/[ABI](../../glossary.md#term-application-binary-interface-abi)/generated-byte replay.

Use the actual local CompiledScanBody [graph](../../glossary.md#graph) to build a local [PrepareContext](../../glossary.md#prepare-context-preparecontext)/[DAG](../../glossary.md#term-directed-acyclic-graph-dag)/requirements and
reuse CpuPartitionDagDecomposer.decompose/unitContext and CpuPartitionLowering.lower. Do not
subset the outer one-[node](../../glossary.md#node) Scan DAG or create a second semantic graph/math inventory. Reuse
ordinary focused family emitters, the same CPU0080 emitMethods [owner](../../glossary.md#term-owner) and existing emitted Scan
time loop. Preserve real typed physical [layouts](../../glossary.md#layout)/carriers/cuts and live old/new [carry](../../glossary.md#term-carry)/history.

Resolve inferred computed local [descriptors](../../glossary.md#term-descriptor) centrally in the existing composition `bodyContext`
through [Model](../../glossary.md#term-model) `ScanBody.closeOrdinaryLayouts(Operation, List<TensorDescriptor>,
List<TensorDescriptor>)`, shared with flat [Compiler](../../glossary.md#term-compiler) closure. Overlay projected values and matching
logical requirements only; keep the captured body and exact activity graph [immutable](../../glossary.md#term-immutable). Analysis
and helper resource preparation use this same projection. Retain strict actual/formal guards,
actual-only [symbol](../../glossary.md#term-symbol) authority and existing symbolic numeric mapping without sampled B. Exact
[views](../../glossary.md#term-view-meanings) keep backing offsets/[strides](../../glossary.md#term-stride) and ordinary [fusion](../../glossary.md#term-fusion); only existing [materialization](../../glossary.md#materialization) decisions
may create [canonical](../../glossary.md#term-canonical-form) cuts. Unproved/negative/empty extraction layouts do not acquire fabricated
storage. Do not copy the Compiler classifier/geometry, import Compiler, or add a fallback resolver.

The reviewed complete-graph source-[publication](../../glossary.md#term-publication) prerequisite reuses the existing [CPU](../../glossary.md#term-cpu) source
assembler, geometry and representation owners. Expose one descriptor-level checked size
factory on the existing cold geometry owner; use Model [referenced span](../../glossary.md#referenced-element-span) for static affine/empty
layouts and existing batchCanonical(READ)/BatchBinding proof for sole-named-B sources.
If needed, the one nested numeric adapter
`CpuRepresentationRecipes.BatchLayoutByteSize(CpuAccessPlan.BatchBinding, DataType)` implements
existing [Runtime](../../glossary.md#term-runtime) ByteSize through actual [binding](../../glossary.md#term-binding) span and checked width multiplication, with a
structurally empty fixedByteSize marker. It is not a new top-level owner, [IR](../../glossary.md#term-intermediate-representation-ir) or executor.
Reuse existing caller [borrowing](../../glossary.md#term-borrowing)/[constant](../../glossary.md#term-constant) initialization and raw representation policy.

Every affected concrete CPU unit/container, including generic, general/fixed recurrent and
[OpenBLAS](../../glossary.md#term-openblas) paths, must cold-authenticate genuinely fixed selected geometry inside a complete
[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-required source plan. Keep Runtime defaults strict, actual physical/access checks,
static code shape and qualified keys/bytes/ABIs unchanged. Never make a fixed unit adopt an
unrelated caller B, choose a new route or bypass its typed representation [validation](../../glossary.md#term-validation).

The following focused additional internal production owners are selected, if needed, rather
than growing a duplicate executor or validator framework:
`internal.lowering.CpuGeneralScanCompositionLowering` for local physical composition,
`internal.codegen.emit.CpuGeneralScanBodyEmitter` for fixed ordinary helper-call orchestration
inside the existing time loop, and `internal.ir.CpuScanControl` for CPU numeric control layout/
code-shaping facts. Private nested physical records remain within existing IR/recipe owners.
No other top-level production type or cross-owner protocol is selected by a task brief alone.

Extend existing family emit/geometry owners for controlled emission, active-contributor hooks
and pack [rebind](../../glossary.md#capture-graph-ports-and-reconstruction)/reset. Controlled ordinary helpers append one [MemorySegment](../../glossary.md#term-memorysegment) control after the
old ABI; composite Scan entries may receive numeric long[][] packs. Standalone/retained entry
descriptors and empty-prefix bytes stay unchanged. Derive local-slot widths from actual [JVM](../../glossary.md#term-java-virtual-machine-jvm)
descriptors, forward control through subsidiary calls, and verify the ACTUAL complete method
inventory plus class/code/local/[parameter](../../glossary.md#term-parameter)/constant-pool limits for fresh and cached [artifacts](../../glossary.md#term-artifact).
One declaration owner supplies emission and verification, not a second family/method [registry](../../glossary.md#term-registry).

All unit calls inside the time loop are fixed invokestatic calls. Bind wrappers/packs/cursors
once per invocation and emit owner-defined temporal-base/cursor resets, including retry.
No per-step pack/clone/[slice](../../glossary.md#slice)/ofArray, unit-object loop, [MethodHandle](../../glossary.md#native-numerical-library-and-instruction-architecture) [lookup](../../glossary.md#term-lookup), [BoundInvocation](../../glossary.md#bound-invocation--boundinvocation) body
children, [callback](../../glossary.md#term-callback), scheduler or new runtime interpreter. Native control is declared [RunState](../../glossary.md#run-state--runstate)
workspace; simultaneous ranges and numeric/control/scratch storage have proved disjoint [lifetime](../../glossary.md#term-lifetime).
Initialize roots from the local compiled plan and this [occurrence](../../glossary.md#term-occurrence)'s own time/row domain. A
retained external [Tensor](../../glossary.md#tensor) [seed](../../glossary.md#term-seed) remains present; do not recognize numeric zeros or consult another
occurrence's control. This clarifies no new helper ABI, storage or selective [identity](../../glossary.md#term-identity-meanings) field.
Shared scoped Model declaration proof precedes analysis but grants no [backend capability](../../glossary.md#backend-capability). Retain
independent physical geometry/resource and actual length-contributor checks; do not duplicate the
symbolic proof or impose length ROW [provenance](../../glossary.md#provenance) on ordinary all-valid reshape.

Qualify static and one-named-B/fixed-[feature](../../glossary.md#term-feature) geometry without sampled descriptors or actual-B-
baked classes. Per-value widths/[alignment](../../glossary.md#term-alignment) and arbitrary proved history ranks/[axes](../../glossary.md#term-axis) remain real;
do not replace obsolete first-carry/[rank](../../glossary.md#term-rank)-three assumptions with another permanent body-family
list. Fuse only compatible structural scopes, physical domains, value-validation boundaries
and optional-merge/temporal-ADD meaning. Existing standalone/native/[SIMD](../../glossary.md#simd)/parallel eligibility
does not automatically expand to controlled units; portable semantic support is separate.

Reuse represented SUM and [MATMUL](../../glossary.md#matrix-multiplication--matmul) active-contributor seams and factor existing scatter traversal/
[coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix)/reduction [ownership](../../glossary.md#term-ownership-meanings) for skip-before-index/load behavior. Factor existing [softmax](../../glossary.md#softmax--log-softmax)/
[loss](../../glossary.md#loss)/[BOOL](../../glossary.md#data-type--datatype)/attention-[mask](../../glossary.md#term-boolean-mask)/index/scatter validator loops into nonallocating temporal/activity/range
entries shared with ordinary bind. Complete the current unit's entire eligible validation domain
before its first numerical output/history/scratch write. No duplicated predicates, validator
registry or new [normalization](../../glossary.md#term-normalization)/categorical [finite](../../glossary.md#term-finite) policy.

Every nested IR/[specialization](../../glossary.md#term-specialization) independently passes its exact numerical, geometry and resource
[admission](../../glossary.md#term-admission) before [cache](../../glossary.md#term-cache)/generation/definition. Control/pack/typed/provenance/lifetime/method-inventory code-shaping facts have a
selective composite identity; absent activity adds no old-route empty suffix. Preserve unchanged
frozen recipe bytes/keys, distinguishing genuinely new compact [AD](../../glossary.md#autograd) graphs. No Metal Scan, native
step route, numerical-policy selector or broader cold/wire grant follows from Scan composition alone.
Actual all-valid/length forward AND selected backward, empty/no-ROW/typed/family/poison/[guard](../../glossary.md#term-guard)/
retry/lifetime/identity evidence and independent source/Javadoc review are mandatory.

## [Metal](../../glossary.md#term-metal) [backend](../../glossary.md#backend)

The Java project lives in `modules/backend/metal`; its macOS [arm64](../../glossary.md#native-packaging-and-platform-vocabulary) [native bridge](../../glossary.md#term-native-bridge) is colocated at
`modules/backend/metal/macos-arm64`, without a nested `native/` directory. The current
[audit-driven completion](#audit-driven-backend-completion) governs the qualified [FLOAT32](../../glossary.md#term-float32),
[BFLOAT16](../../glossary.md#term-bfloat16) and [FLOAT16](../../glossary.md#term-float16) ordinary, fixed-cell first-order and native Scan domains.
Task-era [qualification](../../glossary.md#term-qualification) paragraphs retain their explicit [checkpoint](../../glossary.md#term-checkpoint) [scope](../../glossary.md#term-scope);
they neither certify nor veto subsequent independently qualified occurrences.

The public-readiness repair uses coordinated backend-local [identity](../../glossary.md#term-identity-meanings) 45, retaining native ABI7
and schema22/header124. Its authenticated fixed corpus is 190,781 [UTF-8](../../glossary.md#term-utf-8) bytes across 23 components,
[SHA-256](../../glossary.md#term-sha-256) `0c936f49500dd5022ce55f5b4c2276fd000407520f3b6a11d50fc18e687b2568`.
Physical [unary](../../glossary.md#term-unary)/comparison/numeric-[MATMUL](../../glossary.md#matrix-multiplication--matmul) reads share original-[carrier](../../glossary.md#term-storage-carrier) [layout](../../glossary.md#layout) [authentication](../../glossary.md#term-authentication);
ordinary empty outputs and zero-grid [fused](../../glossary.md#specialization-and-fusion) anchors preserve the family identities without reads
or [dispatch](../../glossary.md#term-dispatch). Numeric GeneralScan may map proved strided ingress through ordinary [CONTIGUOUS](../../glossary.md#term-contiguous).
Raw [F64](../../glossary.md#floating-point-precision-and-rounding) Scan is [canonical](../../glossary.md#term-canonical-form), non-[gradient](../../glossary.md#term-gradient) and source-only; no F64 body arithmetic, [AD](../../glossary.md#autograd), implicit
conversion, dynamic-B Metal or [runtime](../../glossary.md#term-runtime) fallback is authorized.
Scan attribute grammar v2 authenticates a fifteen-word control prefix, including a five-word
[run](../../glossary.md#term-run)-private first-failure record; v1 is incompatible. The added forty bytes per Scan are declared
before shared assignment. Failure context reports category, original time, unit, operand and
logical ordinal without capturing [tensor](../../glossary.md#tensor) data or replaying the body. Source-only active [BOOL](../../glossary.md#data-type--datatype)
roles validate before [publication](../../glossary.md#term-publication); inactive sequence cells remain unread. External typed controls
require canonical [descriptors](../../glossary.md#term-descriptor); independently proved local dense [views](../../glossary.md#term-view-meanings) retain their own [admission](../../glossary.md#term-admission).
[FP8](../../glossary.md#term-fp8) remains rejected at typed Metal storage and publication boundaries, including empty cases.


Metal backend owns:

- the [immutable](../../glossary.md#term-immutable) public `MetalBackendConfiguration`, including the required caller-selected
  absolute native library path; any later Metal-private option remains owned here
- `MetalBackendIntegration.open(MetalBackendConfiguration)`, configuration validation and
  snapshotting, native loading, context creation, and partial-construction rollback<span data-glossary-navigation="true"> ([rollback glossary](../../glossary.md#term-rollback))</span>
- [MPSGraph](../../glossary.md#term-mpsgraph) [lowering](../../glossary.md#lowering)
- MPSGraph executable creation
- custom Metal [kernel](../../glossary.md#kernel) routes
- Metal storage
- native bridge integration
- Metal-specific [materialization](../../glossary.md#materialization)
- direct cold binding of exact live Metal buffer representations for prepared host-staged upload
  and download without exposing a native handle or performing backend lookup<span data-glossary-navigation="true"> ([lookup glossary](../../glossary.md#term-lookup))</span>
- Metal [trace](../../glossary.md#trace) contributions

[Engine](../../glossary.md#term-engine) may take [ownership](../../glossary.md#term-ownership-meanings) of a successfully opened Metal integration, but it must not duplicate
or interpret Metal configuration, discover a library, select a Metal [device](../../glossary.md#term-device), or construct native
Metal state itself. Metal must not depend on Engine.

One `MetalBackendIntegration` owns one `MetalDeviceContext` created through the system-default-
device path and one associated [command queue](../../glossary.md#term-command-queue). Its configuration selects only the caller-supplied
[native library](../../glossary.md#term-native-library). The availability [token](../../glossary.md#term-token) `BackendDeviceId(metal, "default")` is the sole abstract
Metal slot, not a stable hardware [fingerprint](../../glossary.md#term-fingerprint) or public selector. Engine registration is keyed by
[`BackendId`](../../glossary.md#backend-identity--backendid), so one Engine can own at most one Metal integration. Every Metal [buffer](../../glossary.md#term-buffer), workspace,
route resource, and native handle remains authenticated against its exact context; foreign-context
and foreign-`MTLDevice` use fails closed.

The current Metal invocation boundary is [synchronous](../../glossary.md#term-synchronous): native routes [commit](../../glossary.md#term-commit) and wait for completion
before successful return. A completed invocation may publish resources into a [`RunResult`](../../glossary.md#run-result--runresult) whose
ownership outlives that call, so device completion does not imply resource [release](../../glossary.md#term-release) or authorize a
general output/workspace pool. Concurrent callers do not imply [GPU](../../glossary.md#term-gpu) overlap, cross-run ordering,
fairness, queue multiplicity, or [throughput](../../glossary.md#term-throughput). There is no async/cancel/timeout [API](../../glossary.md#term-application-programming-interface-api), physical-device
enumeration or selection, multi-device scheduling, cross-device migration/coherence, failover, or
hot-plug contract. See [ADR 0020](../../design/decisions/0020-synchronous-single-default-device-metal-execution.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>.

The current cross-[owner](../../glossary.md#term-owner) transfer capability is deliberately exact: [rank](../../glossary.md#term-rank)-0..16 fully static
descriptors for all seven [model](../../glossary.md#term-model) data types may move [CPU](../../glossary.md#term-cpu) to Metal or Metal to CPU when their resolved
layout has positive extents, positive [strides](../../glossary.md#term-stride), no overlapping logical positions, and one checked
physical [referenced span](../../glossary.md#referenced-element-span). Canonical contiguous layouts are the simplest member of that domain;
supported SELECT/[SLICE](../../glossary.md#slice) layouts may carry a nonzero [storage offset](../../glossary.md#term-storage-offset) and holes. The transfer copies
the complete physical span without conversion, preserving prefix and gap bytes as well as logical
[represented bits](../../glossary.md#term-represented-bits).

CPU owns the live native [host](../../glossary.md#term-host) staging representation and its per-run [lifetime](../../glossary.md#term-lifetime); Metal owns the
device-buffer type check and native copy. Both backends validate descriptor, exact byte [extent](../../glossary.md#term-extent),
context, openness, and current-thread access during cold [binding](../../glossary.md#term-binding). BOOL descriptor-aware download
stages and validates every logical byte before committing any destination mutation; prefix and gap
bytes are not interpreted. The bound transfer retains direct typed references and performs exactly
one native copy when invoked. Unresolved, empty, zero- or negative-stride, overlapping, span-
inconsistent, or byte-overflowing layouts, conversion, [heap](../../glossary.md#term-heap) staging, and unsupported directions
fail before backend analysis.

Local Metal publication uses the same checked geometry for authenticated storage-layout [targets](../../glossary.md#term-target)
and also supports canonical [rank-zero](../../glossary.md#axis-rank-scalar-vector-and-matrix) through rank-sixteen targets for all seven carriers. Caller
host ingress preserves exact represented storage bytes; descriptor-aware execution, transfer, and
publication paths validate logical BOOL values without treating storage holes as BOOL [elements](../../glossary.md#term-element).

A fully static empty descriptor may be materialized to the host without a layout hint.
There are no source addresses to resolve: `RunResult` supplies canonical zero geometry to the
existing selected-backend copy. Nonempty descriptors with unresolved geometry remain rejected.
This empty publication rule changes neither cross-owner transfer admission above nor [lease](../../glossary.md#term-lease),
identity, budget, lifetime or selected-[backend ownership](../../glossary.md#backend-ownership) checks.

The exact movement domain uses existing fixed custom owners and original typed descriptors.
All seven carriers retain qualified reshape/rank edits, affine extraction, [canonicalization](../../glossary.md#occurrence-identity-and-canonical-form),
[gather](../../glossary.md#gather), [padding](../../glossary.md#padding), [slice update](../../glossary.md#slice-update), CONCAT/STACK/TILE and UNFOLD_AXIS. Signed-step SLICE materializes
a canonical destination for all seven carriers; negative logical [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) do not authorize
negative physical storage strides. Index roles accept [INT32](../../glossary.md#data-type--datatype)/[INT64](../../glossary.md#data-type--datatype); ONE_HOT publishes canonical BOOL.

CONCAT/STACK no longer have a sixteen-operand limit. Dynamic operand references feed one
disjoint direct-copy child per operand through the shared ordinary/Scan owner, without N-wide
[argument](../../glossary.md#term-argument) binding or an intermediate pairwise tensor chain. Original-carrier views and selected
operand cotangents retain their descriptors. Common rank-sixteen, checked geometry, byte-span,
dispatch and workspace bounds still apply; absence of an operand-sixteen [guard](../../glossary.md#term-guard) is not an
unbounded-resource promise.

FOLD_AXIS accepts the six numeric carriers; image unfold/fold accepts the four floating carriers.
FLOAT32, BFLOAT16 and FLOAT16 folds may add overlapping contributors, using deterministic
output-owned FLOAT32 accumulation in declared order and per-output typed publication. No atomics,
host repair, scratch or software [FP64](../../glossary.md#term-fp64) is introduced. Uncovered cells receive exact carrier zero.
[FLOAT64](../../glossary.md#term-float64) and integral folds retain their qualified non-overlap boundaries. Java and native
[preflight](../../glossary.md#term-preflight) independently validate this domain using original carrier facts, never a FLOAT32 proxy.

Shared [Compiler](../../glossary.md#term-compiler) [convolution](../../glossary.md#term-channels-first-convolution-layers)/[pooling](../../glossary.md#term-pooling) adjoints reuse unfold, MATMUL, reduction and fold.
FLOAT32/BFLOAT16/FLOAT16 Pool2d/[Pool3d](../../glossary.md#pool3d--ncdhw-pooling) overlap preserves existing forward winner/divisor semantics,
and [Conv3d](../../glossary.md#conv3d--ncdhw) training retains groups and selected input/[weight](../../glossary.md#term-weight)/[bias](../../glossary.md#term-bias) roles. These are shared typed
owners, not standalone backward [kernels](../../glossary.md#kernel) or permission for CPU low arithmetic.

Every [logical index](../../glossary.md#axis-rank-scalar-vector-and-matrix) is validated in stable [node](../../glossary.md#node) then [row-major](../../glossary.md#term-row-major) ordinal order. Replacement
scatter completes bounds and global destination-uniqueness checks before mutation; admitted
arithmetic modes retain colliding contributors in declared order. Public [prepared execution](../../glossary.md#prepared-execution--preparedexecution)
reproduces Model bounds/duplicate exception classes, while the raw [ABI](../../glossary.md#term-application-binary-interface-abi) reports typed status.
[Validation](../../glossary.md#term-validation) failure preserves inputs and the affected [operation](../../glossary.md#operation)'s write targets, not already
completed [producers](../../glossary.md#term-producer). Colliding replacement, unqualified rows and overlapping FLOAT64/integral folds remain false.

Selected affine views retain their exact logical [shape](../../glossary.md#shape), nonnegative read strides and offset.
Preparation/native preflight separately authenticate represented physical storage and complete
referenced spans. Materializing outputs are canonical. Qualified empty source/publication,
EXPAND, [normalization](../../glossary.md#term-normalization) and Scan paths detect zero extents before product/stride checks; they
do not bypass semantic guards. Rank zero is one stored element. Unproved layouts/extents,
negative physical strides, unsafe overlap and [overflow](../../glossary.md#term-overflow) [fail closed](../../glossary.md#admission-preflight-and-fail-closed-validation).

### Accepted [Metal](../../glossary.md#term-metal) dense-[view](../../glossary.md#term-view-meanings) [MATMUL](../../glossary.md#matrix-multiplication--matmul) extension

MAIN implements the bounded [Metal0080](../../planning/backends/metal/tasks/0080-dense-view-matmul-composition.md)
extension for first-production consumers. Its identity38 source-matched [GPU](../../glossary.md#term-gpu)/package/unchanged
consumer proof predates identity39's [FLOAT32](../../glossary.md#term-float32) image-fold addition, identity40's [pooling](../../glossary.md#term-pooling)-overlap
[admission](../../glossary.md#term-admission), identity41's audit extension, identity42's typed cutover and identity43's [singleton](../../glossary.md#term-singleton)
MATMUL product repair; [release](../../glossary.md#term-release) acceptance remains separate.

MATMUL input geometry may be exact [canonical](../../glossary.md#term-canonical-form) [row-major](../../glossary.md#term-row-major) [strides](../../glossary.md#term-stride), zero offset and [referenced span](../../glossary.md#referenced-element-span)
equal to logical [element](../../glossary.md#term-element) count with logical isView true or false. Retain positive fully static
[rank](../../glossary.md#term-rank) 1..16, exact [contraction](../../glossary.md#contraction-dimension)/[vector](../../glossary.md#term-vector) promotion/[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [broadcasting](../../glossary.md#broadcasting) and checked byte/[dispatch](../../glossary.md#term-dispatch)
bounds. Outputs remain canonical. No general strided, offset, overlap or zero-stride operand
storage is admitted; legal MATMUL batch broadcasting is separate from storage broadcasting.

Non-dense [transpose](../../glossary.md#term-transpose) still requires an actual earlier local PERMUTE with [identity](../../glossary.md#term-identity-meanings) prefix and the
last two [axes](../../glossary.md#term-axis) exchanged, rooted in an authenticated dense value. Every source/[descriptor](../../glossary.md#term-descriptor)/[carrier](../../glossary.md#term-storage-carrier)/
[gradient](../../glossary.md#term-gradient)/span relation stays checked. A materialized affine result uses its represented dense
[buffer](../../glossary.md#term-buffer); an already-elided transpose uses its authenticated source plus one transpose application.
Do not follow arbitrary logical ancestry, add elision/copies, fabricate canonical [metadata](../../glossary.md#term-metadata) or
change [Model](../../glossary.md#term-model)/[Compiler](../../glossary.md#term-compiler) views. Preserve transpose [publication](../../glossary.md#term-publication)/non-MATMUL [fan-out](../../glossary.md#term-fan-out).

Provider, preparer, route planner, independent Java/native [validation](../../glossary.md#term-validation) and [MPSGraph](../../glossary.md#term-mpsgraph) value-state
initialization authenticate actual selected descriptors/physical sources. Dense MATERIALIZED_LAYOUT
feeds and local AFFINE_VIEW results are admitted only with this MATMUL proof, not blanket state
admission or clearing physical facts. Generic canonicalStorage and canonical-only anchors remain.

Original [homogeneous](../../glossary.md#term-homogeneous) FLOAT32, [BFLOAT16](../../glossary.md#term-bfloat16) and [FLOAT16](../../glossary.md#term-float16) forward/generated-gradient rows require
independent [qualification](../../glossary.md#term-qualification). Keep existing no-gradient integral/promotion/[widening](../../glossary.md#term-widening) bounds; no [F64](../../glossary.md#floating-point-precision-and-rounding)
MATMUL, mixed [BF16](../../glossary.md#floating-point-precision-and-rounding)/[F16](../../glossary.md#floating-point-precision-and-rounding), F16/[F32](../../glossary.md#floating-point-precision-and-rounding) widening or integral/widening gradients. Low-containing partitions
remain CUSTOM_PROGRAM only, with no generated-[pointwise](../../glossary.md#term-pointwise)/anchor [fusion](../../glossary.md#term-fusion) or fallback. No Java [CPU](../../glossary.md#term-cpu)
low-numeric permission follows.

Metal0080 advanced all six coordinated workload/exact-policy/candidate/compatibility/route/[codec](../../glossary.md#term-codec)
identities from36 to37; Metal0081 advanced them to38. [Conv3d](../../glossary.md#conv3d--ncdhw) training advanced them to39 for
overlapping FLOAT32 image folds; [Pool3d](../../glossary.md#pool3d--ncdhw-pooling) completion advances them to40 for shared FLOAT32 pooling
overlap. Stale handoffs/decisions reject even with valid outer [integrity](../../glossary.md#term-integrity).
The [checkpoint](../../glossary.md#term-checkpoint) retained schema20/header124/ABI7/thirteen exports. Native/source/package
[authentication](../../glossary.md#term-authentication) remains mandatory under the current protocol below; old packages or historical
v1 capability evidence do not substitute for rebuilt current source and original-typed facts.


Metal-specific [optimizer](../../glossary.md#term-optimizer) execution belongs to Metal [backend](../../glossary.md#backend) [prepare](../../glossary.md#prepare)/[kernels](../../glossary.md#kernel), not to training.

Do not add `MetalOptimizerBridge` to `extensions/training`.

## [OpenBLAS](../../glossary.md#term-openblas) provider

`modules/backend/openblas-provider` is a low-level [leaf](../../glossary.md#term-leaf) provider.

Allowed:

- OpenBLAS library loading
- [symbol](../../glossary.md#term-symbol) [binding](../../glossary.md#term-binding)
- [GEMM](../../glossary.md#general-matrix-multiplication--gemm) calls
- thread control

Forbidden:

- config interpretation
- [planning](../../glossary.md#term-planning)
- fallback logic
- [prepared execution](../../glossary.md#prepared-execution--preparedexecution)
- [Tensor](../../glossary.md#tensor) [API](../../glossary.md#term-application-programming-interface-api)
- [runtime](../../glossary.md#term-runtime) [residency](../../glossary.md#residency)
- [backend ownership](../../glossary.md#backend-ownership) decisions

The [dependency direction](../../glossary.md#java-platform-and-build-tools) is:

```text
modules/backend/cpu -> modules/backend/openblas-provider
```

Never the reverse.

## Other tool scopes

The performance-evidence section above owns the detailed boundaries for `tools/benchmarks` and
`tools/tuning`. The intended `tools/cli` location has no additional technical rule in the migrated
contract. Root global invariants and [dependency](../../glossary.md#term-dependency) rules therefore apply; if [CLI](../../glossary.md#term-cli) work encounters an
architecture-sensitive [ownership](../../glossary.md#term-ownership-meanings), dependency, or [lifecycle](../../glossary.md#lifecycle) question not answered there, the
missing [scope](../../glossary.md#term-scope) requires an explicit architecture update.

## Profile-free [backend capability](../../glossary.md#backend-capability) and [identity](../../glossary.md#term-identity-meanings)

<a id="numerical-profile-backend-identity"></a>

A [concrete backend](../../glossary.md#concrete-backend) must use profile-free, [occurrence](../../glossary.md#term-occurrence)-specific provider capability and qualified
routes. Plans, [specialization](../../glossary.md#term-specialization), generated [artifacts](../../glossary.md#term-artifact), tuning candidates and decisions, and [cache](../../glossary.md#term-cache)
compatibility retain their non-profile semantic, type, [target](../../glossary.md#term-target), [ABI](../../glossary.md#term-application-binary-interface-abi), source, and version identities;
changed formats reject stale inputs. [CPU](../../glossary.md#term-cpu) routes obey the low-[precision](../../glossary.md#term-precision) parity cutover above;
there is no profile selector.

After the cutover, a new supported occurrence must have an independently true provider query for
its exact [dtype](../../glossary.md#data-type--datatype), [descriptors](../../glossary.md#term-descriptor), attributes, and [gradient](../../glossary.md#term-gradient) [metadata](../../glossary.md#term-metadata); a route qualified against [Model](../../glossary.md#term-model)
semantics and the applicable exact, special-value, [finite](../../glossary.md#term-finite), and failure [conformance](../../glossary.md#term-conformance) tests; and an
updated current ledger and provider/conformance tests. A [FLOAT32](../../glossary.md#term-float32) true answer never implies a
[BFLOAT16](../../glossary.md#term-bfloat16) or [FLOAT16](../../glossary.md#term-float16) true answer. [Metal](../../glossary.md#term-metal) checks original per-role carriers and [scalar](../../glossary.md#term-scalar) metadata
without a FLOAT32 descriptor proxy. Each admitted low occurrence requires its qualified shared
custom step under the fixed low-containing CUSTOM_PROGRAM rule. CPU-only arithmetic availability
follows its independently qualified ordinary families. Advance the relevant [compatibility identity](../../glossary.md#identity-meanings-and-algebraic-identities) append-only when changed
route, program, plan, serialized, or generated behavior makes old inputs incompatible; reject
stale inputs. Capability evidence and route [qualification](../../glossary.md#term-qualification) remain cold [backend](../../glossary.md#backend)/[Prepare](../../glossary.md#prepare) concerns,
not a [Runtime](../../glossary.md#term-runtime) [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions), [tolerance](../../glossary.md#term-tolerance) policy, retry, or fallback layer.

Current generated CPU artifact envelopes use [schema](../../glossary.md#term-schema) 71. Current Metal images use schema 22
(`SM22`) with a 124-byte header, and coordinated workload, exact-policy, candidate, compatibility,
route and [codec](../../glossary.md#term-codec) identities use version 45. Native ABI7 retains thirteen exports. CPU schema68
and Metal schema19/identity29 are historical pre-cutover baselines. Metal identities30–38 record
prior profile-free, elementary, boundary and [view](../../glossary.md#term-view-meanings) repairs;39 added FLOAT32 image folds,40 shared
[pooling](../../glossary.md#term-pooling) overlap,41 the joined ordinary/native-Scan audit extension,42 shared original-typed
FLOAT32/BFLOAT16/FLOAT16 execution and dynamic composition references;43 the [singleton](../../glossary.md#term-singleton)
[MATMUL](../../glossary.md#matrix-multiplication--matmul) product repair;44 affine/empty ingress;45 [contraction](../../glossary.md#contraction-dimension)-before-[bias](../../glossary.md#term-bias) [convolution](../../glossary.md#term-channels-first-convolution-layers) and Scan v2
[validation](../../glossary.md#term-validation)/workspace/exact [admission](../../glossary.md#term-admission). Stale identities reject.
CPU schema 70
corrected [activation](../../glossary.md#term-activation) extremes and named low FLOAT32 working sites; those repairs changed neither
Model semantics nor provider admission. CPU schema 71 retains those repairs and adds the bounded
FLOAT32 [pointwise](../../glossary.md#term-pointwise) general-scan family. CPU0031's distinct shaped specialization retains schema 71
and old pointwise compatibility. CPU0032's MATMUL-only specialization preserves both old scan
forms within schema 71; baked shaped/MATMUL facts enter both existing specialization identity
streams, not the geometry-free [canonical](../../glossary.md#term-canonical-form) family envelope. CPU0033's additive [FLOAT64](../../glossary.md#term-float64) scan variants
retain those projections/payloads and accepted FLOAT32 keys/bytes; existing dtype and baked
resource facts distinguish their identities. CPU0034 advanced the staged payloads to v2;
CPU0035 advanced them to v3; CPU0036's current staged specialization payloads are v4 in both
existing streams for top-tested loop control and dominating final-copy bases. The global
envelope/family stays 71 and canonical shaped/MATMUL projections stay geometry-free v1.
Staged v1/v2/v3 artifacts are incompatible, while pointwise, fixed
and unrelated identities are unchanged. Current identities reject stale inputs, including Metal
identity 38 and CPU schema 70 and earlier. Existing CPU family projections and historical
versioned evidence remain distinct from this current global envelope identity.
Native source, [manifest](../../glossary.md#term-manifest), and package [authentication](../../glossary.md#term-authentication) remain safety gates; historical schema-19
source certificates do not certify current schema-22 programs.

Metal's current exact occurrence domain contains exact [unary](../../glossary.md#term-unary) operations; all 49
ordered casts; FLOAT64/FLOAT32/BFLOAT16/FLOAT16 classification; scalar and right-aligned [BOOL](../../glossary.md#data-type--datatype)
logic; the fourteen supported promoted floating `WHERE` [signatures](../../glossary.md#term-signature); exact seven-[carrier](../../glossary.md#term-storage-carrier)
affine/index/replacement/movement rows; image windows and the fold domain specified above;
unsigned-32-bit-bounded ordering/top-K/numeric arg-[extrema](../../glossary.md#saturation-pole-and-extremum); exact maximum Pool2d/[Pool3d](../../glossary.md#pool3d--ncdhw-pooling);
strict comparisons; and zero-input raw [INT64](../../glossary.md#data-type--datatype)[2] `INITIAL_STATE`. Direct BFLOAT16/FLOAT16 mixed-low
operations, including the two remaining promoted WHERE signatures, require explicit casts through
FLOAT32 and have no mixed-low [kernel](../../glossary.md#kernel). Floating-to-floating casts preserve legal gradient metadata.
Float-to-integral casts may consume a [differentiable](../../glossary.md#derivative-and-chain-rule) input but produce a non-differentiable result.
`WHERE` differentiability is the exact branch-role OR and never includes its condition.

The CPU-free generated-gradient closure includes floating casts, inverse affine movement,
signed-step materializing extraction and selected updates, [PAD](../../glossary.md#padding), CONCAT/STACK/TILE, exact
replacement and admitted arithmetic scatter, broadcast reductions, ordinary arithmetic/
statistical/[normalization](../../glossary.md#term-normalization)/[loss](../../glossary.md#loss)/attention formulas, saved-[mask](../../glossary.md#term-boolean-mask) [dropout](../../glossary.md#dropout--inverted-dropout) and qualified window
adjoints. Saved condition/index/mask/forward roles remain live until consumed. FLOAT32,
BFLOAT16 and FLOAT16 use original typed roles through these shared [owners](../../glossary.md#term-owner); qualification is
still occurrence-specific, not inferred from a FLOAT32 proxy. [Conv3d](../../glossary.md#conv3d--ncdhw)/overlapping Pool3d,
[general Scan](../../glossary.md#general-differentiable-scan) and fixed [RNN](../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../glossary.md#term-long-short-term-memory-lstm) have actual selected-first-order qualification in all three
types on a Metal-only [Engine](../../glossary.md#term-engine). Fixed-cell rules cover selected floating input/state/[weight](../../glossary.md#term-weight)/bias
roles, not higher-order or arbitrary geometry. Entire generated [topology](../../glossary.md#term-topology) must be eligible;
unsupported geometry, dtype and order remain false.

The current provider also admits no-gradient [INT32](../../glossary.md#data-type--datatype)/INT64 MATMUL pairs with INT64-dominant
promotion and modular result arithmetic. For the frozen supported [ACCELERATOR](../../glossary.md#term-accelerator) FLOAT32 occurrences,
Metal's independently queried [homogeneous](../../glossary.md#term-homogeneous) BFLOAT16 and FLOAT16 counterparts remain true. This
covers the qualified [tensor](../../glossary.md#tensor)/scalar arithmetic, comparisons, extrema, reductions, cumulative
operations, positive-static MATMUL geometries, normalization, losses, attention, convolution,
pooling, explicit-state dropout and recurrent domains. Each low input is decoded exactly,
every numerical working value and [accumulator](../../glossary.md#accumulator-type) is FLOAT32, and each observable low result is
narrowed once at its declared [publication](../../glossary.md#term-publication) boundary with [round-to-nearest, ties-to-even](../../glossary.md#floating-point-precision-and-rounding).
INT32/INT64 cumulative operations retain exact [modular arithmetic](../../glossary.md#term-modular-arithmetic), not floating emulation.
Conv1d/Pool1d may use only authenticated local singleton-height views. This paragraph does not
widen unlisted or false provider occurrences.

### Historical [EXP](../../glossary.md#term-expx)/[SIGMOID](../../glossary.md#term-sigmoid) [qualification](../../glossary.md#term-qualification) [checkpoints](../../glossary.md#term-checkpoint)

These checkpoints establish the separately preserved low domains and historical evidence.
The later audit below supersedes earlier [FLOAT32](../../glossary.md#term-float32) [scalar](../../glossary.md#term-scalar)/[gradient](../../glossary.md#term-gradient) exclusions, not low restrictions.

At the Task-0074 checkpoint (2026-10-09), the [Metal](../../glossary.md#term-metal) provider admitted `EXP` only for one [canonical](../../glossary.md#term-canonical-form),
fully static, positive-[rank](../../glossary.md#term-rank) `1..16` FLOAT32 input and equal-[Shape](../../glossary.md#shape) output with no gradients, no
attributes, and checked four-byte span and [dispatch](../../glossary.md#term-dispatch) geometry. [Prepare](../../glossary.md#prepare) selected its fixed direct
[MPSGraph](../../glossary.md#term-mpsgraph) [exponent](../../glossary.md#term-exponent) route. At that checkpoint, independently queried [BFLOAT16](../../glossary.md#term-bfloat16) and [FLOAT16](../../glossary.md#term-float16) `EXP`
answers were false, as were [FLOAT64](../../glossary.md#term-float64), gradient-bearing, scalar-rank, noncanonical, and over-limit
answers. The historical v1 [`ACCELERATOR`](../../glossary.md#term-accelerator) EXP answer stays false; at that checkpoint v2 recorded
FLOAT32 true and both low answers false.

Task 0075 independently qualifies [homogeneous](../../glossary.md#term-homogeneous) BFLOAT16 and FLOAT16 `EXP` in that bounded canonical,
positive-static-rank `1..16`, no-gradient, no-attribute, same-type/same-Shape domain with checked
two-byte span and dispatch geometry. Current v2 records true for both low types while historical
v1 stays false. Each low type has a distinct typed custom [kernel](../../glossary.md#kernel) step, evaluates in FLOAT32 working
values, and narrows once to its original low type under the [Model](../../glossary.md#term-model)'s `EXP` special-value and
arithmetic-site [DAZ](../../glossary.md#term-daz)/[FTZ](../../glossary.md#term-ftz) rules. The current FLOAT32 [occurrence](../../glossary.md#term-occurrence) retains its direct MPSGraph route.
FLOAT64, gradient-bearing, scalar-rank, noncanonical, mixed-low, and over-limit `EXP` occurrences
remain false. A low-valued `EXP` step is custom, never MPSGraph.
An explicit low-to-FLOAT32 cast followed by FLOAT32 `EXP` may separately use an internal MPSGraph
boundary step in the enclosing low-containing `CUSTOM_PROGRAM` route; the cast does not itself
grant low-valued `EXP` support or change the [partition](../../glossary.md#partition) candidate set.

Task 0076 admits one canonical fully static positive-rank `1..16` FLOAT32 `SIGMOID` input and
equal-Shape output with no attributes, no gradients, and checked four-byte span and dispatch
geometry. Wire 64 uses a fixed composed MPSGraph route: a predicate over reinterpreted stored
FLOAT32 bits selects the Model's sign branch before its `EXP`, typed-one addition, and division.
The [guard](../../glossary.md#term-guard) classifies [signed zero](../../glossary.md#term-signed-zero) and [subnormals](../../glossary.md#term-subnormal) without floating-comparison DAZ. Historical v1
stays false; at that checkpoint v2 recorded only FLOAT32 true. BFLOAT16, FLOAT16, FLOAT64,
gradient-bearing, scalar-rank, noncanonical, and over-limit SIGMOID occurrences were false. A
FLOAT32 SIGMOID after an explicit low-to-FLOAT32 cast may [run](../../glossary.md#term-run) as an internal MPSGraph step in a
low-containing `CUSTOM_PROGRAM` partition; that composition alone did not grant a low-valued
SIGMOID or second partition route.

Task 0077 independently admits homogeneous BFLOAT16 and FLOAT16 `SIGMOID` in the same bounded
canonical, positive-static-rank `1..16`, no-gradient, no-attribute, same-type/same-Shape domain,
with checked two-byte spans and dispatch geometry. Current v2 records true for both low types;
historical v1 remains false. The two typed custom steps use the exact stored low sign guard,
one FLOAT32 `EXP` on the selected branch input, branch-selected numerator, FLOAT32 addition and
division, and one final [ties-to-even](../../glossary.md#term-ties-to-even) [narrowing](../../glossary.md#term-narrowing) to the original low type. The FLOAT32 wire-64
step remains composed MPSGraph, including after an explicit low-to-FLOAT32 cast inside a custom
partition. A low-valued wire-64 step is always custom, never MPSGraph. FLOAT64,
gradient-bearing, scalar-rank, noncanonical, mixed-low, and over-limit `SIGMOID` remain false.

Every partition containing BFLOAT16 or FLOAT16 values selects the fixed `CUSTOM_PROGRAM`
whole-partition route. This includes homogeneous no-gradient raw-preserving `RESHAPE`, simple
`PERMUTE`, materializing [`CONTIGUOUS`](../../glossary.md#term-contiguous), [`SLICE`](../../glossary.md#slice), `CONCAT`, and `TILE` [schema](../../glossary.md#term-schema)-20 images.
`CUSTOM_PROGRAM` is their only candidate, and the selected route is [immutable](../../glossary.md#term-immutable) before shared
declarations escape analysis. Existing eligible FLOAT32 occurrences retain their qualified direct
or composed routes, including direct [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) FLOAT32 [MATMUL](../../glossary.md#matrix-multiplication--matmul). Any FLOAT32-only [node](../../glossary.md#node), including one
on an independent branch or following an explicit low-to-FLOAT32 cast, may therefore use an
internal MPSGraph boundary step in the same custom partition; no BFLOAT16/FLOAT16-valued node may
do so, and no hidden BFLOAT16-to-FLOAT16 substitution is allowed. Generated-[pointwise](../../glossary.md#term-pointwise) and
anchor-[epilogue](../../glossary.md#term-epilogue) [fusion](../../glossary.md#term-fusion) remain disabled for the whole low-containing partition. The shared custom
route uses fixed reviewed raw-word/integer/movement/predicate [kernels](../../glossary.md#kernel) behind one whole-program invocation with
declared assigned [buffers](../../glossary.md#term-buffer) for every logical value and no [host](../../glossary.md#term-host) repair.

The Task-0069 specialized FLOAT32 [L1](../../glossary.md#term-l1) at wire `114`, ScatterAdd at wire `70`, and [singleton](../../glossary.md#term-singleton)
[VARIANCE](../../glossary.md#term-variance) at wire `112` select `CUSTOM_PROGRAM` under `CA_0069`; Java and native creation reject
their direct MPSGraph routes. These FLOAT32 kernel restrictions do not remove the independently
admitted homogeneous BFLOAT16/FLOAT16 occurrences, which use the general low custom program and
its `lp_l1_norm`, `lp_scatter_add`, and `lp_variance` operations under their own occurrence checks.
The specialized FLOAT32 L1 slice requires one canonical positive-static [rank-one](../../glossary.md#axis-rank-scalar-vector-and-matrix) no-gradient input,
ordered multi-[axis](../../glossary.md#term-axis) `[0]`, a canonical scalar or retained `[1]` output, unsigned-32-bit bounds, one
output thread, and distinct buffers. It raw-clears every contributor sign and performs exactly
`N-1` source-ordered safe additions; `N=1` performs none.

The specialized FLOAT32 ScatterAdd slice requires canonical rank-one base/update/output with positive
static extents, axis zero, no gradient flags, and a materialized canonical rank-one [INT32](../../glossary.md#data-type--datatype)/[INT64](../../glossary.md#data-type--datatype)
index partition feed. The [compiler](../../glossary.md#term-compiler) places explicit `CONTIGUOUS` between its generated zero-base
`EXPAND` and Scatter. A complete [CPU](../../glossary.md#term-cpu) index scan precedes every command encoding and mutation. One
[target](../../glossary.md#term-target) thread raw-loads the base once, keeps every matching duplicate in source order, and stores
once; unaddressed targets copy the raw base, and there are no atomics or races. This exact node
closes the existing rank-one [Gather](../../glossary.md#gather) data [cotangent](../../glossary.md#cotangent) without admitting gradient-bearing ScatterAdd
or higher-order [ownership](../../glossary.md#term-ownership-meanings).

The specialized FLOAT32 VARIANCE slice requires canonical rank-one `[1]` input, axes exactly `[0]`,
correction zero, no gradient flags, distinct buffers, and a canonical scalar or retained `[1]`
output. One thread evaluates exactly `DIV(x,+1)`, `SUB(x,mean)`, `MUL(difference,difference)`, and
`DIV(square,+1)`, then stores the sole canonical cell once. The task-local [Lean](../../glossary.md#proof-assistant-and-model-checking-tools) proof covers every
DAZ/FTZ choice: all [finite](../../glossary.md#term-finite) inputs yield exact [positive zero](../../glossary.md#nan-infinity-and-signed-zero), while [NaN](../../glossary.md#term-nan) and either [infinity](../../glossary.md#term-infinity) sign
yield NaN class. There is no classifier, aggregate add, reciprocal, [FMA](../../glossary.md#term-fused-multiply-add-fma), clamp, fallback,
retry, timing, or alternate [publication](../../glossary.md#term-publication).

The retained shared Lean proof, source-bound declaration, and pinned Xcode-27 compiled-[MSL](../../glossary.md#term-metal-shading-language-msl)/[AIR](../../glossary.md#term-air)
audit remain source and compiled-code review evidence for the three bounded implementations, not a
[runtime](../../glossary.md#term-runtime) numerical [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions) or a substitute for post-cutover route [conformance](../../glossary.md#term-conformance).

Direct CPU/Metal transfer supports all seven current data types at ranks `0..16` over canonical or
resolved positive-[stride](../../glossary.md#term-stride) non-overlapping storage [layouts](../../glossary.md#layout); [BOOL](../../glossary.md#data-type--datatype) [validation](../../glossary.md#term-validation) visits logical [elements](../../glossary.md#term-element)
only. Local selected publication gathers logical elements from authenticated physical storage and
preserves exact target-relative offsets and holes without exposing [aliases](../../glossary.md#aliasing-and-arena).

The pre-cutover package uses [ABI](../../glossary.md#term-application-binary-interface-abi) 7 with thirteen exports. Node schema 19 is the frozen bounded self-describing,
route-bearing image over type wires `1..7`, [operation](../../glossary.md#operation) wires `1..115`, attribute wires `0..41`,
route wires `1..3` and complete optional storage-layout geometry. The changed header/decoder
format receives an append-only schema [identity](../../glossary.md#term-identity-meanings) and rejects profile-bearing stale images.
Its fixed 128-byte `SM19` header authenticates the core image and, only for `CUSTOM_PROGRAM`, the
canonical step/member/[binding](../../glossary.md#term-binding)/materialized-value/typed-instruction/[manifest](../../glossary.md#term-manifest) extension. Every core
and extension section is adjacent; references are never padded before 64-bit attributes. The
manifest carries only typed execution records and integer source byte counts; Java neither emits
Metal source nor owns fixed, generated, or assembled source [hashes](../../glossary.md#term-hash). MPSGraph images set every
extension count and flag to zero and physically omit the extension. Custom execution binds compact
materialized slots rather than one slot per logical value.

Schema-2 pointwise fusion may replace each maximal [linear](../../glossary.md#term-linear-meanings) chain of eligible canonical FLOAT32
`FLOOR`, `CEIL`, `SIGN`, and `RELU` nodes with deterministic generated units of length `2..8`;
a remainder of one is prevented by taking seven from the preceding unit. Inputs and outputs have
positive fully static rank `1..16` and element count `1..UINT32_MAX`. Generation is bounded by 32
units, 256 instructions, 16,384 bytes per function, 262,144 generated bytes, and 1,048,576 total
source bytes. Native reconstructs exact cap precedence and the first rejected node. The production
emitter first traverses the structured sink in no-allocation count mode, validates all five caps,
allocates the exact byte count, and traverses the same emitter once for compiler input. The exact
fixed corpus is 84,593 [UTF-8](../../glossary.md#term-utf-8) bytes: native [SHA-256](../../glossary.md#term-sha-256)-checks each of its ten ordered components and
their total and independently authenticates the assembled source before compiler entry.

The versioned program image admits the existing qualified typed anchor-epilogue instructions. An eligible MATMUL
preserves optional literal scalar multiplication, at most one source-ordered ordinary right-aligned
[tensor](../../glossary.md#tensor) ADD, and optional terminal RELU or no-gradient CLAMP. [Conv2d](../../glossary.md#convolution--conv2d) admits at most ADD followed by
the same terminal pair and never scalar multiplication. Intrinsic rank-one `[C]` remains only the
third [convolution](../../glossary.md#term-channels-first-convolution-layers) input; an external rank-one addend broadcasts over `[W]`, while `[1,C,1,1]`
expresses ordinary [channel](../../glossary.md#term-channel) broadcast. Every absorbed intermediate is private, single-consumer,
non-target, canonical, and absent from the materialized set. The native anchor validator
independently reconstructs those facts before one safe-math dispatch and one final store, with no
intermediate store, retry, or fallback.

The audit [fixture](../../glossary.md#term-fixture) uses the same production header in singleton-enabled audit mode, repeats generation,
compares count, bytes, and [hash](../../glossary.md#term-hash); runtime [admission](../../glossary.md#term-admission) remains length `2..8`. Generated pipelines
require `BindingInfo | BufferTypeInfo`, exact input-read/output-
read-write runtime access, and exact size/[alignment](../../glossary.md#term-alignment)/type/name/offset reflection for every member of
the 32-byte, eight-aligned `PointMeta` ABI. Each generated unit has one input load, one output
store, one dispatch, and no intermediate [materialization](../../glossary.md#materialization). Barriers, unsupported nodes,
noncanonical layouts, gradients, [fan-out](../../glossary.md#term-fan-out), targets, and cap rejection preserve fixed custom or
MPSGraph boundaries; Task-0069 VARIANCE remains custom only for its exact singleton domain.

At the frozen cutover baseline, structural execution covered 101 kinds with 14 nonexecutable and
production capability covered 86 kinds with 29 false. The FLOAT32 `EXP` addition raised
those counts to 102/13 structural and 87/28 production; the later BFLOAT16/FLOAT16 `EXP` additions
do not change kind counts. FLOAT32 SIGMOID raised that checkpoint to103/12 structural and88/27
production; these historical counts do not describe the current live inventory. Workload,
exact-policy, candidate, compatibility, route-policy, and [codec](../../glossary.md#term-codec) identities were version 29 before
cutover, version 30 at cutover, version 31 for FLOAT32 `EXP`, version 32 for low `EXP`, version 33
for FLOAT32 `SIGMOID`, version 34 for low `SIGMOID`, and version 35 for the native-boundary and
exact low-value repairs. Version36 corrected standalone scalar admission and37 added dense-[view](../../glossary.md#term-view-meanings)
MATMUL;38 added selected materializing RESHAPE;39 added overlapping FLOAT32 image folds;
40 added shared FLOAT32 [pooling](../../glossary.md#term-pooling) overlap;41/schema21 added the ordinary/native-Scan audit;
42/schema22 added shared typed execution;43/schema22 preserves singleton MATMUL product semantics.
Stale versions [fail closed](../../glossary.md#admission-preflight-and-fail-closed-validation). The complete-plan wrapper retains its unrelated version-one format.

## Accepted ordinary materializing RESHAPE extension

[ADR0062](../../design/decisions/0062-ordinary-materializing-reshape-closure.md) is implemented through
Complete Compiler0019 final geometry and Complete CPU0079 existing affine ordinal copy, without
unlocking low arithmetic. Integrated Metal0081 source has actual source-matched package/[device](../../glossary.md#term-device)
and unchanged typed-backward consumer proof; final current-ledger/docs/[release](../../glossary.md#term-release) acceptance remains
separate. Ordinary validated static RESHAPE selects [canonical](../../glossary.md#term-canonical-form) non-[view](../../glossary.md#term-view-meanings) geometry only when its
supported inferred view remains unproved. Raw source/wire [inference](../../glossary.md#term-inference-meanings) stays unchanged.

Metal0081 independently retains the exact AFFINE_VIEW branch and admits the selected materializing
branch only with canonical non-view output/CANONICAL state, offset0, span=count and
densePhysical=false. Source/output offset/span equality remains mandatory for actual views,
not for compact [materialization](../../glossary.md#materialization) of a valid offset/holed source. Neither source flags nor actual
physical geometry may be erased. Materialization receives its ordinary distinct destination and
[lifetime](../../glossary.md#term-lifetime), not an [alias](../../glossary.md#aliasing-and-arena) or new elision.

Reuse the selected exact_move_unary_0066 operation6 [kernel](../../glossary.md#kernel) and existing [metadata](../../glossary.md#term-metadata)/copy helpers,
with the representation actually bound. Preserve existing movement [dtype](../../glossary.md#data-type--datatype)/[gradient](../../glossary.md#term-gradient)/storage bounds,
[fan-out](../../glossary.md#term-fan-out)/[publication](../../glossary.md#term-publication), [runtime](../../glossary.md#term-runtime) alias/size/context checks and CUSTOM_PROGRAM-only low partitions.
No new [shader](../../glossary.md#term-shader), numeric math, consumer [CONTIGUOUS](../../glossary.md#term-contiguous) or older0059-kernel substitution is authorized.
Schema20/header124/ABI7/thirteen exports remain; all six coordinated [Metal](../../glossary.md#term-metal) [identities](../../glossary.md#term-identity-meanings) advance
37 to38 and valid stale37 evidence rejects. This explicitly extends the former noncontiguous-
RESHAPE exclusion; it does not broaden [MATMUL](../../glossary.md#matrix-multiplication--matmul) storage or other affine operations.

## Audit-driven [backend](../../glossary.md#backend) completion

The user-requested [CPU](../../glossary.md#term-cpu)/[Metal](../../glossary.md#term-metal) [operation](../../glossary.md#operation) audit implements the following bounded extensions.
Every [occurrence](../../glossary.md#term-occurrence) still requires complete independent logical, physical, native and selected-
[adjoint](../../glossary.md#term-adjoint) [qualification](../../glossary.md#term-qualification). The [production ledger](../../backend-guide/production-support.md)
records actual numerical/[device](../../glossary.md#term-device) evidence separately from final candidate acceptance.

CPU reuses its existing typed [IR](../../glossary.md#term-intermediate-representation-ir), conversions, exact comparisons and movement/[layout](../../glossary.md#layout) [owners](../../glossary.md#term-owner) for
[Model](../../glossary.md#term-model)-promoted [pointwise](../../glossary.md#term-pointwise) operands, signed-step materializing slices and their existing
pullbacks, [variadic](../../glossary.md#term-variadic) composition beyond the old sixteen-input [guard](../../glossary.md#term-guard), and sound larger [injective](../../glossary.md#term-injective)
output proofs. Signed logical [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) do not authorize negative physical storage [strides](../../glossary.md#term-stride).
Preserve checked resource/classfile limits, exact stored-value comparisons, promoted computation
and original [cotangent](../../glossary.md#cotangent) [carrier](../../glossary.md#term-storage-carrier) conversion. The low-[precision](../../glossary.md#term-precision) parity cutover and independent
ordinary named-B predicates apply.

Metal implements the qualified [FLOAT32](../../glossary.md#term-float32)/[BFLOAT16](../../glossary.md#term-bfloat16)/[FLOAT16](../../glossary.md#term-float16) elementary/[activation](../../glossary.md#term-activation) and
statistical/masked-reduction [signatures](../../glossary.md#term-signature), affine/[scalar](../../glossary.md#term-scalar) reduction reads, colliding arithmetic
scatter and [Gather](../../glossary.md#gather) adjoints, [softmax](../../glossary.md#softmax--log-softmax)/[normalization](../../glossary.md#term-normalization)/[loss](../../glossary.md#loss)/attention, [spatial](../../glossary.md#term-spatial-axes) operations and fixed
recurrent forward/selected first-order rules through existing Model/[Compiler](../../glossary.md#term-compiler) equations and
shared prepared Metal owners. The separate low-precision library is retired; common typed
arithmetic and numeric codecs retain each original role's storage/[POD](../../glossary.md#term-pod) type and each output's
[publication](../../glossary.md#term-publication) boundary. There is no FLOAT32 [tensor](../../glossary.md#tensor) proxy or duplicated low implementation.
Exact original-typed [FLOAT64](../../glossary.md#term-float64)/integral operations use represented semantics without software
floating arithmetic; [INT32](../../glossary.md#data-type--datatype)/[INT64](../../glossary.md#data-type--datatype) cumulative arithmetic remains exactly modular. Original
[descriptors](../../glossary.md#term-descriptor) determine promotion, optional operands, [gradient](../../glossary.md#term-gradient) bits and [canonical](../../glossary.md#term-canonical-form) siblings.
[Cancellation](../../glossary.md#term-cancellation), [signed zero](../../glossary.md#term-signed-zero), nonfinite classes, exclusions and named working sites follow Model,
not old structural recipes. Fixed-cell and explicit GeneralScan ordinary-cell [rounding](../../glossary.md#term-rounding) remain
distinct. No CPU low arithmetic, [FP64](../../glossary.md#term-fp64) arithmetic/emulation or CPU fallback is enabled.

General Metal Scan uses an authenticated bounded native body/binder plan inside the existing
executable [owner](../../glossary.md#term-owner). It reuses neutral compiled-body/activity/demand transport and once-prepared
ordinary typed Metal steps in a native time loop, preserving traversal, [carry](../../glossary.md#term-carry)/history [bindings](../../glossary.md#term-binding),
original demanded output roles, structural inactivity, pre-write [validation](../../glossary.md#term-validation), empty [identities](../../glossary.md#term-identity-meanings)
and actual reverse-body semantics. FLOAT32/BFLOAT16/FLOAT16 explicit ordinary-cell bodies use
that same owner, not a fixed-cell compatibility fallback. There is no [Runtime](../../glossary.md#term-runtime) [graph](../../glossary.md#graph) interpreter,
Java child loop, T-fold [unrolling](../../glossary.md#term-unrolling), copied CPU evaluator or CPU fallback. Symbolic/cold/order
[admission](../../glossary.md#term-admission) is separate.

[Cold reconstruction](../../glossary.md#term-cold-reconstruction) accepts Model-legal per-role floating descriptors and casts in the existing
recipe domain. Public [checkpoint](../../glossary.md#term-checkpoint) wire versions and strict descriptor [rebind](../../glossary.md#capture-graph-ports-and-reconstruction) remain unchanged;
this does not admit arbitrary [PURE](../../glossary.md#term-pure) bodies or [optimizer](../../glossary.md#term-optimizer)/history/compiled-plan continuation.

Ordinary operation/attribute identities are reused where their grammar suffices. Scan added wire
116/attribute42 at schema21/identity41; current schema22/`SM22`/header124/identity45 retains the
shared typed execution and dynamic composition-reference grammar. Every caller uses that grammar,
[shader](../../glossary.md#term-shader)/component/source [authentication](../../glossary.md#term-authentication) is renewed and stale inputs reject. ABI7/thirteen exports
remain. Metal general FLOAT64 numerical arithmetic, unsupported mixed-low domains and
unqualified higher-order recurrent [derivatives](../../glossary.md#term-derivative) remain excluded. CPU low arithmetic follows its
separate parity cutover, not this Metal qualification. Source-matched
device execution and independent source review precede production-support claims; current
component evidence is not final full-catalog or [sealed](../../glossary.md#term-sealed)-[release](../../glossary.md#term-release) qualification.
