# Metal backend

## Outcome and supported scope

The Metal backend executes one whole maximal Metal-owned partition. Its active protocol is
native ABI 7, program schema 20 with a 124-byte header, data-type wires `1..7`, operation wires
`1..115`, attribute wires `0..41`, route wires `1..3`, and coordinated backend-local identity 30.
Schema 19 and identity 29 are historical pre-cutover allocations, not current decoder inputs.

The profile-free exact domain contains represented-bit unary operations, affine
movement and canonicalization, indexing, replacement scatter, non-overlapping folds, classification,
BOOL logic, casts, WHERE, stable ordering, top-K, numeric arg-extrema, strict comparisons, maximum
pooling, promoted integral MATMUL, and raw RNG state. All seven carriers participate where the
operation's Model contract permits them. In particular:

- `RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`, `CONTIGUOUS`, `SELECT`, `SLICE`,
  `GATHER`, `GATHER_ELEMENTS`, `GATHER_ND`, replacement `SCATTER_ELEMENTS`/`SCATTER_ND`, `PAD`,
  `SLICE_UPDATE`, `CONCAT`, `STACK`, `TILE`, and `UNFOLD_AXIS` preserve all seven carriers;
- all 49 ordered casts use the exact Model conversion; all four floating carriers support the
  admitted classification rows; the fourteen non-mixed-low promoted floating WHERE signatures
  preserve exact branch-gradient OR metadata;
- `SORT`, `ARGSORT`, and positive-K `TOP_K` accept all seven carriers, while `ARG_MIN` and
  `ARG_MAX` accept the six numeric carriers and publish INT64;
- `FOLD_AXIS` accepts every non-BOOL carrier, and image unfold/fold accepts
  FLOAT64/FLOAT32/BFLOAT16/FLOAT16; admitted folds are non-overlapping;
- maximum Pool2d/Pool3d accepts FLOAT64 plus every homogeneous FLOAT32/BFLOAT16/FLOAT16 occurrence;
  and
- no-gradient INT32/INT64 MATMUL uses INT64-dominant promotion and modular result arithmetic.

Every Shape is fully static within the operation-specific rank domain. Selected affine views retain
their logical Shape, positive strides, and storage offset. Java preparation and native preflight
independently derive and authenticate physical spans, unsigned-32-bit geometry, index bounds, and
replacement destination uniqueness before resource creation or mutation. Zero extents, unresolved
layouts, unsafe external strides, overlap, out-of-span descriptors, and byte overflow fail closed.
BOOL validation visits logical elements only.

Every supported current homogeneous FLOAT32 occurrence has independently queried
admitted BFLOAT16 and FLOAT16 counterparts. This closes tensor and scalar arithmetic, comparisons,
extrema, SUM/MEAN/MIN/MAX and sum-to-Shape reductions, cumulative SUM/PROD, positive-static MATMUL,
MSE, convolution, average pooling, dropout, and the qualified L1, ScatterAdd, and singleton
VARIANCE slices. Low inputs are decoded exactly to FLOAT32; working values and accumulators remain
FLOAT32; each observable low result is narrowed exactly once with round-to-nearest, ties-to-even.
The same gradient metadata relationships as the corresponding FLOAT32 occurrence are required:
binary arithmetic uses operand OR, scalar and reciprocal preserve the input bit, reductions and
scans preserve differentiability, comparisons publish no-gradient BOOL, and dropout preserves its
value role while saving a no-gradient mask and state.

Direct BFLOAT16/FLOAT16 mixed-low operations are unsupported. Explicit casts to FLOAT32 are the
only mixed-low boundary. Existing one-low-plus-FLOAT32 widening occurrences remain permitted where
the Model result is FLOAT32. Every partition containing BFLOAT16 or FLOAT16 values selects the
fixed `CUSTOM_PROGRAM` route. It cannot select MPSGraph, classic MPS, MPP, generated pointwise
source, anchor-epilogue fusion, CPU fallback, retry, timing, or autotuning.

Generated backward graphs reuse the same admitted occurrence predicates. Floating casts, inverse
affine movement, broadcast sum-to-Shape, arithmetic/scalar/reduction/scan formulas, saved-mask
dropout, stable ordering/top-K value paths, replacement scatter, and non-overlapping window
adjoints close for BFLOAT16 and FLOAT16 exactly as for FLOAT32. Compiler-generated first order and
owned higher order therefore remain on Metal for admitted differentiable rows. Maximum-pool
backward is admitted only when every stride is at least the effective dilated kernel. Overlapping
window accumulation and Conv3d backward remain compiler/backend fail-closed.

Convolution and pooling validate exact symmetric padding, stride, dilation, groups, result geometry,
and a 65,536-position pooling-kernel bound before resource creation. Conv1d/Pool1d affine views must
be authenticated local singleton-height compositions. Maximum pooling uses NaN-first,
positive-zero-over-negative-zero, first-logical-winner order with negative-infinity padding.
Average pooling includes conceptual positive-zero padding in the full-kernel divisor. Conv3d is
forward-only until its compiler gradient owner exists.

`INITIAL_STATE` is canonical no-gradient INT64[2]. Metal dropout accepts
canonical rank `0..16`, publishes the value, canonical BOOL saved mask, and next state, and advances
the counter modulo $2^{64}$. Its private SplitMix64 V1 mapping and threshold are deterministic
Metal replay facts, not a portable random stream. Generated backward consumes the saved mask
without advancing state. Recurrent operations remain unsupported.

L1, rank-one ScatterAdd, and singleton VARIANCE retain their deliberately narrow no-gradient
domains. LOG_SUM_EXP, STANDARD_DEVIATION, L2_NORM, attention, recurrent execution,
convolution transpose, arithmetic scatter, colliding replacement scatter, unsupported layouts,
dynamic or over-limit geometry, and every unlisted occurrence remain fail-closed.

```text
capability -> Planning ownership -> Metal analysis and typed candidates
           -> optional decision authentication and private route selection
           -> exact declarations
           -> shared slot assignment -> Metal finalization
           -> PreparedExecution -> isolated Runtime run
```

Capability applies per occurrence. Preparation authenticates the complete maximal partition.
Existing eligible rank-two FLOAT32 MATMUL may retain direct MPSGraph. Every partition containing
BFLOAT16 or FLOAT16 values uses the shared custom whole-program route. The capability catalog
remains `86 true / 29 false` over operation kinds, with structural execution `101 / 14`. These kind
inventories do not count additional dtype occurrences.

Within that capability domain, Metal analysis generates a typed complete candidate batch and
selects one of three private routes:

- `CUSTOM_SINGLE_NEG` for exactly one FLOAT32 NEG occurrence, one unique feed,
  one unique target, and a checked element count in `1..UINT32_MAX`;
- `CUSTOM_PROGRAM` for every partition containing BFLOAT16/FLOAT16 values, any Task-0052 node, any
  of the seven BOOL-domain nodes, exact raw `FLOOR`/`CEIL`/`SIGN`/`RELU`, an admitted
  Task-0059/Task-0060 exact custom node, any Task-0063 ordering/arg-extrema node, any Task-0064
  convolution/pooling node, either Task-0065 RNG node, or MATMUL outside the retained all-FLOAT32
  rank-two MPSGraph slice; or
- `MPSGRAPH` for every other supported partition.

Route choice is fixed during cold preparation before shared declarations escape. There is no later
retry, fallback, or route replacement. The seven BOOL-domain and four exact raw unary nodes retain
their package-private structural checks without production graph promotion. The singleton NEG
decision seam is unchanged. Route choice is a prepare-time implementation-domain boundary, not
capability narrowing, CPU fallback, retry, repartitioning, or a performance claim.

The three identities are one closed package-private prepared-route enum. It owns candidate wires
`1`, `2`, and `3` and the two-value `CUSTOM_KERNEL`/`MPSGRAPH` family; the candidate batch delegates
wire encoding to that identity rather than maintaining another wire table. Every returned plan
retains one non-null final identity. Finalization, trace metadata, cold binding, and execution read
that retained identity and cannot replace it.

### Custom-only low-precision route

Every BFLOAT16/FLOAT16 partition exposes exactly one candidate: `CUSTOM_PROGRAM`. Raw-preserving
RESHAPE, simple PERMUTE, materializing CONTIGUOUS, SLICE, CONCAT, and TILE follow the same policy as
low arithmetic. Route eligibility no longer depends on a device/toolchain environment record,
loaded-dylib hash, certificate store, program certificate, or vendor probe.
This removes numerical certification only: source/manifest checks, verified package loading,
native ABI checks, and Java/native preflight remain current safety gates. Task-0071 schema-19
source certificates are historical evidence, not schema-20 certification.

Within that custom partition, each BFLOAT16/FLOAT16-valued operation uses a custom kernel step;
there is no hidden BFLOAT16-to-FLOAT16 substitution. Any FLOAT32-only node, including an
independent FLOAT32 branch, may use an internal MPSGraph boundary step under the existing FLOAT32
policy. For example, a BFLOAT16 feed cast explicitly to FLOAT32 uses a fixed custom cast step,
then a FLOAT32-only NEG may use an internal MPSGraph boundary step. The partition still has only
the `CUSTOM_PROGRAM` candidate. No generated-pointwise or anchor-epilogue fusion is allowed
anywhere in the partition.

The low-precision `LowPrecisionTraceMetadata` emitted in PREPARE is selected-route execution
evidence, not a candidate inventory. It reports `CUSTOM_KERNEL`, the ordered feed-then-target
logical dtype tuple. It has no numerical-profile, accumulator, or working-type field: the
FLOAT32 working/accumulator rule above remains a Model arithmetic guarantee, not a diagnostic
field. It carries no certificate status, key, accuracy, determinism, or environment state. While
tracing remains enabled, the separate structural PREPARE event and outcome also apply to
low-containing programs.

Metal performs no executable, compilation, or preparation-cache lookup; each finalization builds
fresh native resources and trace outcomes retain `NOT_QUERIED`. Its tuning identities are
session-only. Compatibility bytes bind identity versions, ABI, a random context nonce, the
schema-20 program/plan digest, explicit dtype wires, Shapes/layouts/gradients, ordered boundaries,
and raw constants; candidate bytes bind the route wire, and complete-plan identity embeds the exact
phase-one decision. The tuning tools do not persist session-scoped values. Thus
BFLOAT16/FLOAT16, route, ABI/schema/policy/semantics, and context boundaries cannot reuse one
another's decision.

Task 0053 retains proposed integer-only raw-word `EXP` and stable `SIGMOID` sources plus independent
MPFR/Sollya/Lean evidence. They are evidence-only artifacts: production imports no candidate header,
contains no Task-0053 gate, appends no candidate source, and carries no Task-0053 function or
dispatch metadata. Java capability and catalog availability remain false, and the native parser
returns unsupported operation for the registered structural wires. Reintroducing the candidate
requires a separately approved production cutover rather than a runtime boolean gate.

## Prerequisites

Java uses JDK 26 Foreign Function and Memory (FFM) APIs. The native bridge requires an
Apple-silicon macOS host, Xcode Command Line Tools, and Foundation, Metal, and
MetalPerformanceShadersGraph. Build and ABI instructions are in the
[native Metal guide](../../native/metal-macos-arm64/README.md).

Backend-local and public integration tests supply the dylib's absolute path directly. The backend
does not discover, extract, package, sign, notarize, or cache the library.
`MetalBackendConfiguration` snapshots that caller-selected absolute path, and either
`MetalBackendIntegration.open(configuration)` or the traced
`open(configuration, observer)` overload loads it, validates the ABI, opens the default device
context, and rolls partial construction back before returning. Planning separately receives the
captured Metal availability snapshot; the capability provider performs no native loading or
device discovery.

## Contracts and ownership

| Stage or resource | Owner and current behavior |
|---|---|
| Capability truth | Public `MetalCapabilityProvider` reports occurrence-specific true and false answers under one Model semantic contract. Supported homogeneous FLOAT32 rows have independently queried BFLOAT16 and FLOAT16 counterparts; direct BFLOAT16/FLOAT16 mixed-low operations remain false. |
| Native configuration and integration | Public `MetalBackendConfiguration` and `MetalBackendIntegration` belong to Metal. Metal validates configuration, opens native ownership, and rolls partial construction back before Engine can take the completed integration. |
| Backend ownership | Planning chooses `owner = metal` and groups consecutive equal owners; it never selects MPSGraph or a custom kernel. |
| Analysis | Package-private Metal code validates the complete partition, assigns stable structural value order, regenerates typed route candidates and session compatibility, authenticates any supplied decision, fixes one route, and declares that route's exact resources. |
| Opaque tuning handoff | Metal can construct the Prepare-owned marker-role handoff with no decision or one Metal decision. Shared Prepare does not inspect private candidates; the Metal preparer treats a present decision as untrusted. |
| Shared preparation | `GraphPreparation` projects facts, assigns slots, validates the result, and transfers persistent resources transactionally. It does not inspect the Metal plan or route. |
| Finalization | Metal validates exact assignments, compiles the fixed singleton pipeline, one shared custom-program recipe, or one shape-specialized MPSGraph executable, then allocates/uploads each source-owned immutable splat in feed order. It never reselects the route. |
| Persistent resources | The route resource and zero or more immutable splat resources are returned in physical acquisition order. `PreparedExecution` uniquely owns each prepared owner. `MetalProgramExecutableResource` owns the native executable and context child lease for either a `CUSTOM_PROGRAM` or an `MPSGRAPH` route; the custom-singleton pipeline has a separate resource type. A splat owner may have already-issued read-only run-binding child leases; owner close rejects new bindings and the last child performs exact-once physical buffer release. These native handles are not interchangeable across routes. |
| Per-run state | Runtime borrows exact typed caller buffers, owns one fresh read-only binding object for each persistent source splat, and owns fresh typed output buffers. MPSGraph and custom-program runs own one fresh native-address workspace; singleton custom runs do not. Mutable outputs/workspaces are never shared or pooled. The scatter executable owns bounded primitive coordinate/uniqueness scratch rebuilt under its synchronized run. SELECT/SLICE buffers cover their full physical referenced span; kernels touch only logical positions. A reduction target receives its declared carrier width, including one byte per BOOL element. |
| Hot invocation | Cold binding authenticates and unwraps immutable splat bindings, retains direct typed buffers/handles, validates every resource and then every indexing or BOOL element before destination mutation, and makes one synchronous typed native call into assigned output destinations only on success. |
| Optional trace observation | While tracing remains enabled, a traced integration emits advisory `MetalPreparationStructure` facts in PREPARE, selected-route `LowPrecisionTraceMetadata` for low-containing programs, and a `BackendPreparationOutcome` after finalization. A bound invocation reaching the native-call path emits a pre-native `MetalInvocationPlan` in RUN and then a `BackendInvocationOutcome` on success or failure; Java-side failures have no native status. PREPARE succeeds only after route and splat acquisition. The ordinary open overload creates no producer or trace work. Observation never selects a route, retries, falls back, or changes ownership or results. |
| Publication | Runtime leases the already-resident output representation. Canonical outputs, local rank-zero reduction outputs, authenticated affine targets, and readable persistent splat bindings may be downloaded as detached canonical host bytes. SELECT/SLICE publication gathers all seven carriers from exact physical storage while preserving holes. CPU/Metal transfer accepts canonical or supported positive-stride non-overlapping storage layouts and validates logical BOOL bytes before any destination mutation. |

The public Java surface contains `MetalCapabilityProvider`, `MetalBackendConfiguration`,
`MetalBackendIntegration`, and `MetalTraceObserver`. Contexts, physical storage, preparers,
finalizers, schedules, executable recipes, the structural route catalog, prepared-route identity,
test-only route forcing, native handles, Objective-C objects, and MPSGraph types remain internal.

### Completion and device topology

Each `MetalBackendIntegration` owns exactly one `MetalDeviceContext` created with the system-
default Metal device and one associated command queue. `MetalBackendConfiguration` selects only the
native library; it exposes no physical-device selector. One Engine can own at most one Metal
integration because its registry rejects a duplicate Metal `BackendId`. Planning may prove an
exact-device requirement against `BackendDeviceId(metal, "default")`, but it retains only the
owning backend. The `"default"` token is an abstract availability slot, not a stable device
fingerprint or selection handle.

Both native routes commit work and wait for completion before returning successfully. Public and
Runtime `run(...)` therefore form a synchronous completion barrier: every schedule action and
native operation for that invocation has completed, and the returned `RunResult` leases completed
publication/resources rather than representing in-flight work. Failure publishes no result.
Transfers and materialization remain synchronous at their existing call boundaries. Each route
resource currently serializes its native run method; this implementation detail and the one queue
do not promise device overlap, submission or completion order, fairness, queue multiplicity, or
throughput to concurrent callers. Their mutable run state remains isolated.

Every Metal buffer, workspace, executable, splat, and route resource is authenticated against its
exact context, and native code rejects a foreign `MTLDevice`. Separate Engines may open separate
contexts, but those contexts may select the same default device and do not form a coordinated
multi-device system. There is no async/cancel/timeout API, enumeration, explicit selection, cross-
device transfer, migration, replication, coherence, sharding, scheduling, failover, or hot-plug
refresh.

`BackendDeviceId(metal, "default")`, fixed `TraceDeviceId(0)`, and the tuning `SessionNonce` are
distinct non-physical identities. The first is an abstract availability slot, the second a Metal
trace correlation token, and the third a session-compatibility identity that makes no stable-device
claim. No mapping among them is implied. ABI 7 and the thirteen native exports are fixed. The
route-bearing image is schema 20 with a 124-byte header, and every workload/exact-policy/candidate/compatibility/
route/codec identity is version 30; every other schema or identity fails closed. Tuning remains
session-scoped and non-persistent.

Future asynchronous execution requires a separate cross-module contract for completion/failure,
cancellation/timeout, input borrowing, result/workspace ownership, prepared leases through device
completion, close races, cross-action dependencies, trace/tuning timing, and only then pooling.
Future explicit-device or multi-device work requires a real supported environment and workflow,
stable selection/enumeration, truthful ABI and availability, planner-retained device identity,
device-qualified resources, cross-device transfer/coherence/topology/lifetime, trace/tuning
identity, and real-device evidence. Selecting one device per integration should precede any
scheduler. See [ADR 0020](../design/decisions/0020-synchronous-single-default-device-metal-execution.md).

## Integration lifecycle

### Optional typed tracing

`MetalBackendIntegration.open(configuration, observer)` retains a non-null caller-owned observer
for that integration without taking ownership of it. The callback may be concurrent and therefore
must be thread-safe. Metal never closes the observer. The existing one-argument overload remains
the exact no-trace fast path: it creates no producer or payload, allocates no trace ID, reads no
clock, and makes no callback.

One traced integration is one producer-defined stream. Backend and device correlations are fixed
to zero; event, prepared-unit, and invocation IDs use independent thread-safe non-negative
sequences beginning at zero. Each event uses `System.nanoTime()`. After final route and anchor
recognition, but before native creation, Metal emits one bounded `MetalPreparationStructure`
`PREPARE` event with neutral route, fused/composed anchor family and suffix order,
schema/generator identities, bounded counts, canonical digest, and bounded custom-step summaries.
For a low-containing program, it then emits `LowPrecisionTraceMetadata` with the selected custom
route and ordered feed-then-target logical dtype tuple. Resource finalization emits the
preparation outcome. Immediately before each native
run call Metal emits one `MetalInvocationPlan` `RUN` event with invocation identity, route, slot
counts, aggregate input/output/internal/splat/workspace bytes, splat/workspace counts, and the same
bounded step summaries; the
existing invocation outcome follows native completion or failure.
Splat count and bytes are derived from the exact validated invocation bindings: a shared splat
prepared by an earlier source owner is counted when bound, while a source-owned resource not bound
by that invocation is not. After an observer failure disables its producer, later preparations
skip trace anchor diagnostics, step summarization, and digest construction entirely.
If the structural PREPARE callback disables tracing, analysis reuses the heuristic plan when its
route matches and otherwise builds exactly one selected no-trace plan. An enabled forced fallback
builds only the heuristic and traced selected plans, never an eager no-trace third copy.

Preparation always reports `NOT_QUERIED`: Metal authenticates an optional outer tuning decision
but performs no cache lookup and cannot claim an outer hit or miss. Native codes `0..12` map to
`SUCCESS`, `INVALID_ARGUMENT`, `DEVICE_UNAVAILABLE`, `COMMAND_QUEUE_UNAVAILABLE`,
`ALLOCATION_FAILED`, `RANGE_OUT_OF_BOUNDS`, `COPY_FAILED`, `INTERNAL_ERROR`,
`UNSUPPORTED_SHAPE`, `COMPILATION_FAILED`, `INCOMPATIBLE_RESOURCE`, `EXECUTION_FAILED`, and
`COMPILATION_FAILED`, respectively. Other signed codes map to `UNKNOWN` without losing the code;
a Java-side failure before a native return has no native status. MPSGraph range status is captured
before the existing Java indexing rescan translates the outward exception.

Structural PREPARE and pre-run planning callback failures, including observer `Error`, atomically
disable tracing and are contained before native finalization or execution. ID exhaustion,
trace-object construction failure, and observer `RuntimeException` from outcome reporting also
disable later tracing without changing backend work, failure, rollback, suppression, or outward
exception. An observer `Error` propagates only from successful outcome reporting. During failure
outcome reporting, the existing finalization or route-specific run failure remains primary; a
distinct observer `Error` is attached as an acyclic suppressed failure after any rollback
suppression, without disabling tracing. Payloads include only Trace-owned IDs and bounded closed
structural or
aggregate facts. They never include source text, tensor data, scalar bits, per-buffer byte
geometry, pointers, native handles, paths, names, secrets, exceptions, free-form strings, or
generic maps. Close tracing remains deliberately deferred: prepared units are logical plans while
native executables are shared reference-counted resources, so a per-unit close event would
misstate native lifetime and add noise. Tracing changes no capability, route, native call, native
ABI/schema/export, result, lifecycle, or Engine production behavior.

### Capability, whole-partition analysis, and route selection

`MetalCapabilityProvider.supports` checks one occurrence and its exact
descriptors. It admits only the operation-specific canonical or authenticated affine input domain;
no general view capability follows from an occurrence query.

All 49 ordered casts validate source/result carrier and Model conversion semantics.
Floating-to-floating casts preserve legal gradient metadata; float-to-integral results are no-grad.
Classification accepts FLOAT64, FLOAT32, BFLOAT16, or FLOAT16 and publishes equal-Shape no-grad
BOOL. AND/OR use exact right-aligned BOOL broadcast, NOT preserves Shape, and WHERE supports every
floating promotion except direct BFLOAT16/FLOAT16 mixing while propagating exactly the branch
gradient OR.

All-seven-carrier affine and movement occurrences validate exact rank, axis, Shape, layout,
attribute, and gradient relations. GATHER/GATHER_ELEMENTS/GATHER_ND and replacement
SCATTER_ELEMENTS/SCATTER_ND accept INT32 or INT64 indices; replacement requires
`ScatterReduction.NONE`. Complete index bounds and destination-uniqueness validation precede
mutation. `UNFOLD_AXIS` accepts all seven carriers; `FOLD_AXIS` accepts the six numeric carriers and
rejects BOOL; 2D/3D unfold/fold accepts FLOAT64/FLOAT32/BFLOAT16/FLOAT16 and requires non-overlap.
SELECT/SLICE/Crop can produce scalar or positive-rank logical views. Capability recomputes exact
positive strides, storage offset, and referenced span and rejects unresolved, empty, unsafe, or
out-of-span geometry.

Admitted arithmetic, reduction, MSE, MATMUL, convolution, pooling, and dropout rows accept every
qualified homogeneous FLOAT32, BFLOAT16, or FLOAT16 occurrence. The provider derives each low
answer by projecting the exact descriptors and scalar attributes to its current FLOAT32
counterpart, never by an operation-name allowlist. Binary output differentiability is the operand
OR; scalar/reduction/scan output differentiability preserves the source relationship. Both
The provider also admits no-gradient INT32/INT64 MATMUL with exact promotion and modular arithmetic.
MATMUL operands are canonical or exact authenticated local identity-prefix, last-two-axis
transposes; output is canonical.

Convolution requires rank-four NCHW/OIHW or rank-five NCDHW/OIDHW, exact grouped channel/bias
relationships, symmetric nonnegative padding, positive stride/dilation/groups/dimensions, exact
floor output geometry, and canonical output. Maximum and average pooling require exact
rank-four/rank-five geometry, matching input/output gradient metadata, and at most 65,536 kernel
positions. Generated data cotangents require non-overlapping effective windows. The exact
singleton-height Conv1d/Pool1d forms remain authenticated local compositions. Availability and hard
backend requirements remain separate Planning facts.

Generated pointwise units and anchor epilogues require a partition whose complete value table
contains no BFLOAT16 or FLOAT16 descriptor. Even eligible FLOAT32 nodes cannot form those fused
steps when another value in their partition has a low-precision type.

Metal recognizes a bounded anchor epilogue only when every suffix intermediate is private,
single-consumer, non-target, canonical, and shape/type/gradient compatible. MATMUL admits optional
literal `SCALAR_MUL`, at most one source-ordered tensor `ADD`, and optional terminal `RELU` or
no-gradient `CLAMP`. Conv2d admits its intrinsic optional rank-one channel bias, at most one
external tensor `ADD`, and optional `RELU` or no-gradient `CLAMP`; it never absorbs scalar
multiplication. External ADD is a distinct canonical FLOAT32 value and follows ordinary
right-aligned broadcasting. Consequently Conv2d rank-one `[W]` broadcasts over width, never
channels; channel broadcasting uses `[1,C,1,1]`, while intrinsic `[C]` remains only the third
Conv2d input. Any publication, fanout, unsupported order, aliasing role, type/layout/gradient
mismatch or instruction/cap mismatch preserves the original composition.
Rank-zero `SCALAR_MUL` and `RELU` are accepted only at their authenticated ordered positions in a
validated anchor step; the same nodes as standalone or unrelated custom-program steps fail closed.
An admitted suffix becomes one custom step, one physical dispatch, and one final store with no
intermediate materialized slot and no native retry or fallback.

The package-private `MetalOperationRouteCatalog` separately describes every one of the 115
current schema-20 `NodeKind` values. Exhaustive enum switching yields shared immutable entries with
closed MPSGraph state/reason and custom-kernel state/reason values: MPSGraph totals are
`75 DIRECT / 35 COMPOSED / 5 UNAVAILABLE`; custom totals are
`73 AVAILABLE / 42 PENDING / 0 UNAVAILABLE_WITH_PROOF`. Task 0066 moves exactly wires `6..11` and
`16..19` from pending to available under reason `CA_0066`. Task-0069 Slices 1 through 3 move wires
`114=L1_NORM`, `70=SCATTER_ADD`, and `112=VARIANCE` from pending to available under reason
`CA_0069` without changing their MPSGraph records. This catalog performs no capability admission
and no selection. It is never consulted by Runtime; 29 kinds remain capability-false even when the
catalog records a structurally direct or composed MPSGraph realization. Structural executable
status separately covers 101 wires with 14 nonexecutable rows and never grants production
ownership.

After Planning creates one maximal Metal partition, analysis walks nodes in partition order with
explicit unavailable, canonical, affine-view, and materialized-layout states. For selected view
operations it retains the logical descriptor separately from an independently derived physical
descriptor; selected materializing operations produce canonical state. Selected consumers index
the authenticated physical offset/strides rather than assuming logical contiguity. A selected
scalar is a real one-element value and may feed another admitted selected operation or publish
directly.

Task-0066 casts, predicates, WHERE, affine transforms, raw movement, and replacement/window rows
retain their exact ordered roles, carrier matrix, gradient flags, Shape, layout, and checked
physical byte span. Index and replacement rows also retain index width, complete bounds, and
uniqueness state. BOOL custom outputs are exact zero or one. Stable value indexing follows first
encounter, repeated use names the same value, and only the authenticated materialized-value set
owns declared assigned buffers. Eligible maximal linear canonical no-gradient FLOAT32 chains of
`FLOOR`, `CEIL`, `SIGN`, and `RELU` are split deterministically into generated units of length
`2..8`; an eight-node take that would leave one instead takes seven. Feeds, targets, fan-out,
gradients, layout changes, unsupported nodes, fixed custom nodes, MPSGraph boundaries, and source
caps stop fusion.

Unselected rows retain their established state rules. Their FLOAT32 MPSGraph binary, reduction,
and MSE variants consume canonical values. MATMUL produces canonical state and may consume affine state only
when analysis authenticates the exact local identity-prefix last-two-axis transpose. Convolution
and pooling normally consume canonical state and always produce canonical state, with only the
exact authenticated singleton-height Conv1d/Pool1d input rewrite. Feeds and published boundary
targets receive Runtime slots; authenticated custom-program boundaries receive compact run-owned
slots, while generated-unit interiors and ordinary MPSGraph intermediates remain unmaterialized.

Analysis receives compile-time constant sources through `PrepareContext.constants()`. A boundary
constant must be an exact type-matching splat in one of the seven schema carriers. Analysis records
whether the current Metal partition is that producer-free value's first ordered consumer. Only
that source partition prepares physical storage; later Metal consumers reuse the same logical Metal
representation, while a value first owned by CPU retains the ordinary transfer destination.
Runtime uses an `InitializedBuffer` fresh read-only binding for a prepared Metal source rather than
consuming a caller position. Existing shared `GraphPreparation` tests independently enforce the
chain `CompileConstantPlan.ConstantSource -> PrepareContext.constants() -> InitializedBuffer`.

Once stable values, states, feeds, targets, checked byte geometry, typed node records, and the
canonical execution plan are known, analysis creates a version-twenty-nine candidate batch and
workload fingerprint. Every partition containing BFLOAT16/FLOAT16 values or a selected
Task-0066/Task-0069 occurrence forces the fixed `CUSTOM_PROGRAM` production candidate; the batch
rejects a selected-node MPSGraph route before native allocation. Other FLOAT32 partitions retain
their established candidate sets, including the bounded singleton-NEG dual-route case and direct
rank-two FLOAT32 MATMUL where no selected custom-only node requires the whole-program route.
Package-private structural forcing remains available only for a route already present in the fresh
authenticated batch.

The version-thirty canonical workload fingerprint covers the schema-twenty route-bearing program
image and authenticated execution extension, ordered
input and output references, all typed attributes, logical and physical value states,
authenticated local provenance, complete tensor descriptors and storage layouts, compact
materialized and target sets, exact scalar/splat bits, logical-boundary roles,
policy/candidate/route schemas, and ABI version. It encodes structural positions rather than graph
object identity. Target compatibility also contains a fresh private nonce from the exact
`MetalDeviceContext`; only identity 30 is accepted, while every other decision fails closed.
Metal can construct an absent- or present-decision `BackendPartitionTuningHandoff`. Fresh analysis
always regenerates the current batch. Every supplied handoff is accepted only when the exact
partition, candidate schema, workload fingerprint, and context session match. An absent decision
then uses the existing heuristic without cache lookup or measurement; a present decision must
additionally name a current candidate. Stale or foreign values fail closed rather than reverting
to the heuristic.

After authentication, `CUSTOM_SINGLE_NEG` declares one feed buffer, one target buffer, and no
workspace. `CUSTOM_PROGRAM` declares the compact materialized set plus one address workspace.
MPSGraph declares its feed and target buffers plus one address workspace. A larger supported
singleton has only its fixed production candidate unless it belongs to a previously audited
dual-route structural set.

Package-private tests can require one exact route only after ordinary lowering, capability
admission, semantic validation, candidate regeneration, and any supplied handoff authentication
have succeeded. The required route must already be a member of that fresh exact batch. Thus the
approved singleton-NEG routes and previously audited forceable structural routes can traverse the
same lifecycle independently, while every Task-0066 or Task-0069 selected occurrence, stale or
foreign handoff, and unsupported operation rejects an unauthorized route before native creation.
No public configuration, integration, Engine, tuning, or Runtime input
exposes this test seam.

### Session decision codec and limitations

The package-private version-thirty Metal codec produces bounded canonical compatibility,
candidate, and checksummed decision bytes. Decode rejects wrong magic, schema, session scope,
malformed or truncated content, trailing or corrupt bytes, changed workload or
context, and unknown or pruned candidates. The bytes contain no native handle or executable. Only
identity 30 is accepted; every other codec or stale decision fails closed even when its
trailing checksum is otherwise valid.

The public `MetalLocalWorkloadTuning` retained by `MetalBackendIntegration` wraps this codec and
candidate generator without changing their version-thirty bytes. It returns a handoff only
for an exact singleton NEG whose complete ordered candidate list is `[CUSTOM_SINGLE_NEG, MPSGRAPH]`.
`MetalCompletePlanTuning` authenticates the exact Phase-1 association and exposes one complete-plan
candidate fixed to that selected route. Both collaborations use opaque exact-owner/batch values,
defensive bytes, fresh authoritative analysis for trial and selected preparation, and
`SESSION`-only compatibility. They perform no execution, measurement, winner selection, or cache
input/output. Validation analysis omits tracing; returned preparations retain the integration's
real trace producer, so only preparations that are actually finalized allocate and report units.
MPSGraph-only, mixed-owner, and multiple-partition plans remain ineligible. Metal cannot publish
or hit a persistent workload-cache entry, and its complete-plan phase never touches the supplied
model-plan path. Cross-session Metal reuse still requires a separately authorized stable
device/library fingerprint.

Schema-twenty workload bytes and workload compatibility include the fixed route,
canonical execution extension, and manifest digest. Candidate and decision bytes retain route
wires `1..3`, route-policy version thirty, and the target session.
Prepared plans and native resources are route-specific. A future executable-cache key would
therefore require the tuple `(workload compatibility, route wire, route-policy version, target
session)` rather than a workload digest alone. The repository has no persistent Metal executable
cache, and this cutover adds none.

### Finalization and persistent ownership

Shared Prepare assigns all declared slots before Metal finalization. The finalizer checks
declaration identity, order, geometry, slot uniqueness, exact `MetalDeviceContext` identity,
source-owner representation position, and route/workspace agreement before native work. It creates
the selected route resource first, then each source-owned splat resource in stable feed order, and
constructs one immutable `PreparedExecutable` recipe.

For the singleton-NEG custom route, native creation compiles the fixed branch-free
`synaptik_neg_f32` source and creates one pipeline. Shared custom-program creation compiles the
fixed reviewed safe-math/raw-word/integer/movement corpus, authenticated bounded pointwise source,
and any interleaved existing-node MPSGraph executables before publication. Generated pipelines use
the strongest supported Xcode path, `MTLPipelineOptionBindingInfo |
MTLPipelineOptionBufferTypeInfo`. They require exactly three named buffers: `input` is read-only
`uint*`, `output` is reflected as read-write `uint*`, and `meta` is a read-only `PointMeta`
pointer/struct. Metal does not report the output device pointer as write-only even though compiled
AIR proves it write-only, so runtime validation requires the actual read-write reflection contract
rather than fabricating a stronger access value. `PointMeta` must expose exactly five ordered
members with types/offsets `ulong@0`, `ulong@8`, `ulong@16`, `uint@24`, and `uint@28`, size 32,
and alignment eight. For the
ordinary MPSGraph route, native creation validates one canonical bounded schema-20 route-bearing
program image with no extension and compiles one fixed-shape whole-partition executable. MATMUL
compilation requires reduced-precision-fast-math read-back `None`. All compilation happens during
finalization, never invocation.

Each source splat owns one exact-sized Metal buffer and context child lease. Finalization fills and
uploads the exact same-type `FLOAT64`, `FLOAT32`, `BFLOAT16`, `FLOAT16`, `INT32`, `INT64`, or
`BOOL` raw carrier bits once, before PREPARE success. The result returns the route resource followed by source
splats in acquisition order. A malformed native route result
fails closed: success with a null handle is rejected, while failure with a non-null handle releases
that handle once and preserves a distinct cleanup failure as suppressed evidence. Allocation,
upload, executable construction, result construction, and observer failure close the current
buffer where applicable, then prior splats and the route resource in exact reverse order while
preserving the original failure.

After finalizer return, shared Prepare owns rollback until a validated `PreparedExecution` accepts
each resource exactly once. `PreparedExecution.close()` rejects new runs without waiting.
Previously admitted synchronous runs may finish. Route-owner release occurs after their execution
leases end. Splat-owner close also rejects new bindings, but an already-returned `RunResult` keeps
its exact read-only binding and physical buffer resident until that child closes. Cleanup is
idempotent, reverse-order, and attempt-all; the first failure remains primary and later distinct
failures are suppressed.

### Cold run setup and binding

Each run receives isolated mutable state and fresh nominal representation identity:

1. caller input positions borrow caller-owned Metal buffers;
2. each source-owned constant feed receives a fresh run-owned read-only binding to its exact
   prepared immutable buffer, with no allocation, fill, or upload;
3. each target allocates a fresh run-owned output buffer; affine-view targets use the full positive
   logical element count and retain exact finalized-route authentication, scalar reduction targets
   use exactly four bytes, and other canonical targets use the ordinary canonical buffer path;
4. an MPSGraph or `CUSTOM_PROGRAM` run allocates one native address-array workspace, while a
   `CUSTOM_SINGLE_NEG` run allocates no workspace; and
5. cold binding validates context, byte extent, scalar type/raw bits, read/write role, and physical
   input/output non-aliasing before unwrapping direct handles.

MPSGraph and custom-program cold binding write ordered native handles into their address workspace
once and retain direct input/output slices. Singleton-custom cold binding retains the exact
underlying input and output buffers and their typed native handles directly. A later run-setup
failure closes its fresh child
binding and other created run resources without closing the prepared splat owner. Metal-to-CPU
transfer and canonical materialization may unwrap a live splat binding as a read source;
CPU-to-Metal upload and executable output binding reject it before native mutation.

The address-workspace alias contracts are route-specific. Ordinary `MPSGRAPH` permits repeated
input handles but requires outputs to be pairwise distinct and distinct from every input.
`CUSTOM_PROGRAM` instead requires the complete compact materialized-slot table to be pairwise
distinct, then requires output position `i` to repeat exactly the authenticated
`slots[targetMaterializedSlots[i]]` handle. Duplicate slot handles and wrong, swapped, or
duplicated target handles fail before command encoding or mutation. `CUSTOM_SINGLE_NEG` requires
distinct input and output handles. Java binding repeats these checks before the downcall, and the
boundary treats every handle as untrusted.

This is deliberately not a general buffer pool. Mutable outputs remain owned by `RunResult` beyond
the synchronous call and concurrent results require distinct writable buffers. MPSGraph address
workspaces are mutable per-run pointer arrays. Safe pooling would first require an explicit
async/result-lifetime contract for exclusive return, reset and validity, bounded capacity/eviction,
context-close deferral, and cleanup failure. No backend-global constant cache, pool, registry, or
cross-preparation reuse is added.

### Custom singleton hot execution

The custom bound invocation makes one typed native call with the persistent pipeline, input
buffer, and assigned output buffer. Java performs no allocation, boxing, reflection, lookup,
address-array marshalling, cast, operation dispatch, route branch, retry, fallback, upload, or
download in that hot method.

Native execution validates the typed buffer family, originating device, byte extents, and
non-aliasing. It creates one command buffer and one compute encoder, binds the direct input
`MTLBuffer` at index `0` and assigned output `MTLBuffer` at index `1`, performs one exact
`dispatchThreads`, commits once, and waits once. Success requires completed command-buffer status
and no error. The output buffer is written directly; Synaptik performs no explicit host staging or
intermediate output copy.

The custom ABI carries `element_count` as `uint64_t`, but native creation accepts only
`1..UINT32_MAX`. A grid that cannot represent the count maps to `UNSUPPORTED_SHAPE`; unusable
threadgroup geometry maps to `EXECUTION_FAILED`. Compilation, function lookup, target support,
and pipeline creation failures map to `KERNEL_COMPILATION_FAILED`. No failure switches the
prepared route to MPSGraph.

### MPSGraph hot execution

The MPSGraph bound invocation holds direct slices of the address workspace plus the persistent
executable. Its hot method makes one native call and performs no Java allocation, address
marshalling, slot or map lookup, representation cast, graph traversal, operation dispatch, route
choice, reflection, string dispatch, host copy, retry, or fallback.

Native execution enters a local `@autoreleasepool` inside the Objective-C exception boundary,
creates the bounded framework binding objects required by MPSGraph, binds the ordered input and
supplied output `MTLBuffer` values, and invokes the executable once with
`waitUntilCompleted = YES`. Every success and early-failure return drains transient framework
objects before crossing the C ABI. Success additionally requires no completion error and the exact
ordered usable result count. Synaptik performs no explicit output copy and does not request hidden
result materialization; this is not a claim that MPSGraph uses no internal temporary storage.

### Fixed report-only route benchmark

`tools/benchmarks` contains one fixed Metal route benchmark for canonical no-grad `FLOAT32 NEG`
with Shape `[1_048_576]` under the one Model numerical contract. It uses the public
`MetalLocalWorkloadTuning` collaboration to enumerate and pin the complete current ordered pair,
but it never invokes `prepareTuned(...)`, chooses a winner, uses complete-plan tuning, accesses a
cache, changes production preparation, or applies a performance threshold.

An untimed integration with a `MetalTraceObserver` first attests successful
`CUSTOM_KERNEL`/`GRAPH_EXECUTABLE` PREPARE and RUN outcomes with `NOT_QUERIED` and exact raw-bit
outputs. Timed work opens a fresh ordinary integration, requires the same opaque candidate
identities, and therefore pays no trace producer, event, clock, or callback cost. The retained hot
boundary is `PreparedExecutionRunner.run(...) + RunResult.close()`; input upload, preparation,
materialization, and checksums stay outside it.

Run the bounded smoke profile after building the explicit bridge:

```bash
./native/metal-macos-arm64/build.sh
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew -q :tools:benchmarks:metalBenchmark -Pprofile=smoke
```

The schema-2 report retains all raw samples and exact correctness hashes, records only the
`smoke`/`baseline` sampling profile (no `numericalProfile` field), and is always marked
ineligible for a production decision and not autotuning evidence. The controlled-host `baseline`
profile adds fixed JVM flags, required source/machine/device/native-build/power/thermal/fork
metadata, paired alternating route order, and a 25 ms floor for each of eight retained batches per
route. The harness creates no result artifact. Default `build`/`check` never run the Metal task;
see the [benchmarking guide](../developer-guide/benchmarking.md) for the full protocol and CI
boundary.

## Examples

### Exact singleton custom route

For caller input `x = [1.0, -2.0, +0.0, -0.0]`, this supported partition:

```text
y = NEG(x)
publish y
```

has one feed, one target, and four elements, so it selects the custom route. Finalization compiles
one persistent pipeline. Each run borrows `x`, creates a fresh output, invokes the pipeline once,
and publishes `[-1.0, 2.0, -0.0, +0.0]`. A second run reuses the pipeline but owns a different
output buffer. Replacing the caller feed with an exact positive-rank `FLOAT32` splat keeps the same
route and creates one fresh initialized feed buffer per run.

### Multi-NEG MPSGraph route

For canonical caller input `x`, this supported graph:

```text
n = NEG(x)
z = NEG(n)
publish n, z
```

forms one maximal typed MPSGraph partition because it is not an eligible custom singleton. `n` is
both an internal value and a boundary target. A run creates fresh supplied destinations for both
targets, executes the graph once, and publishes them directly. Repeated runs of one prepared
session reuse its persistent executable while owning different outputs and address workspaces.
Opening another session performs preparation again and compiles a separate executable.

### Composed views and canonicalization

For canonical `FLOAT32` input `x` with Shape `[6]`, one maximal Metal partition may be:

```text
r = RESHAPE(x, [2, 1, 3])
e = EXPAND(r, [2, 4, 3])
p = PERMUTE(e, [1, 0, 2])
d = EXPAND_DIMS(p, 2)
s = SQUEEZE(d, 2)
c = CONTIGUOUS(s)
n = NEG(c)
publish r, e, p, d, s, c, n
```

Preparation proves each affine input came from the preceding admitted local producer. The first
five publications preserve exact logical view descriptors but use authenticated dense
represented-order buffers. `c` and `n` are ordinary canonical publications. Omitting `c` makes
the view-to-NEG edge invalid before native work.

### Reductions and scalar publication

For canonical `FLOAT32` input `x` with Shape `[2, 3, 4]` and values `1..24`, one Metal
partition may be:

```text
m = MEAN(x, axis=1, keepDimensions=false)
a = ABS(m)
s = SUM_TO_SHAPE(x, [1, 4])
c = ADD(a, s)
t = SUM(x)
publish c, t
```

Preparation lowers the normalized axis and binding-resolved sum-to-Shape target into the bounded
schema-20 route-bearing program image. `m`, `a`, `s`, and `c` remain positive-rank
canonical, so they may compose inside the partition. The scalar `t` is a direct target only.
One run publishes `c = [[71, 78, 85, 92], [83, 90, 97, 104]]` plus `t = 300`; local materialization of
`t` copies exactly four bytes. Scalar operation results do not become composable inputs in this
route.

### General static MATMUL and generated gradients

MATMUL follows the Model's vector promotion and batch broadcast rules: `[K]@[K]` produces a scalar,
`[K]@[K,N]` produces `[N]`, `[M,K]@[K]` produces `[M]`, and rank-two-or-greater leading axes
broadcast right-aligned. Metal executes every no-gradient INT32/INT64 ordered pair with
INT64-dominant output promotion and exact two's-complement modular accumulation. It also
executes homogeneous FLOAT32/BFLOAT16/FLOAT16 with gradient eligibility equal to the operand OR and
the admitted one-low-plus-FLOAT32 widening rows with FLOAT32 output. Each logical operand may be
canonical or the exact local identity-prefix, last-two-axis transpose of a canonical source.

All-FLOAT32 rank-two matrix products retain the pre-existing MPSGraph MATMUL route, including exact
authenticated local transposes. Every geometry containing BFLOAT16 or FLOAT16 and every other
newly admitted geometry uses the fixed safe-math custom kernel, one output writer per logical cell,
increasing contraction order, exact broadcast coordinates, physical offset/stride addressing, and
canonical output storage. Homogeneous low signatures decode to FLOAT32, retain FLOAT32
accumulation, and narrow once; widening signatures publish FLOAT32. The integer signatures operate
on raw unsigned words so overflow is defined modularly. There is no tiling, atomics, fallback,
runtime selection, or hidden transpose materialization.

Compiler-generated explicitly seeded FLOAT32/BFLOAT16/FLOAT16 gradients use the same general domain.
For matrix operands they retain the familiar formulas:

```text
left gradient  = seed @ transpose(right)
right gradient = transpose(left) @ seed
```

Vector/vector multiplies the scalar seed by the opposite operand. Vector/matrix and matrix/vector
insert the missing matrix axes, explicitly materialize those rank edits, apply the corresponding
MATMUL/last-two transpose, squeeze the promoted vector result when required, and `sumToShape` each
broadcast operand. Matrix/matrix gradients also use `sumToShape`. The local multiplication,
transposes, rank edits, reductions, and MATMUL nodes execute in one CPU-free Metal partition for
vector, matrix, batched, and broadcast Shapes.

This remains generated differentiation, not implicit seeding or an unrestricted Metal training
claim. FLOAT64-result pairs, direct BFLOAT16/FLOAT16 mixing, and gradients
on integral or widening mixed-carrier rows fail before native creation.

### Convolution, pooling, and bounded generated gradients

Task-0064 convolution iterates each output cell in N/C/spatial order, then each grouped input
channel and kernel coordinate in fixed logical order. Positive-zero padding remains an ordinary
multiplicand, including against infinity, and optional bias initializes the accumulator. Mixed
BFLOAT16/FLOAT32 source words widen before FLOAT32 multiplication. Task-0064 maximum pooling uses
an integer-key comparator and copies the winning raw word; NaNs dominate numbers, positive zero
dominates negative zero, and the first logical coordinate wins a tie. Average pooling visits the
entire dilated kernel rectangle/box, supplies positive zero for out-of-bounds positions, divides
once by the full kernel-position count, and applies the specified all-negative-zero rule.

Compiler-generated FLOAT32/BFLOAT16/FLOAT16 Conv2d gradients are owned when the
requested roles and geometry form a primitive-closed partition. Explicit `CONTIGUOUS` nodes
materialize group reshapes/permutations, matrix operands/results, unfolded columns, and fold inputs.
The input cotangent requires a non-overlapping `FOLD2D`. Average-pool and maximum-pool cotangents
similarly close only for non-overlapping effective windows; maximum pooling reconstructs the exact
first winning candidate and materializes its final permute before reshape/fold. Pool3d
constructibility compares semantic type/Shape/gradient metadata after final materialization rather
than requiring the original input layout to be canonical. Conv3d gradients and every
overlap-accumulating fold remain fail-closed. These bounded cases add no implicit seed, backend
autograd, or unrestricted training ownership.

### Exact GATHER and ONE_HOT

For canonical `FLOAT32 data:[2,4]`, canonical `INT32 gatherIndices:[3]`, and canonical
`INT32 classIndices:[3]`, the profile-free backend may execute:

```text
g = GATHER(data, gatherIndices, axis=1)   // [2,3] FLOAT32
h = ONE_HOT(classIndices, depth=4)       // [3,4] BOOL
publish g, h
```

`GATHER=16` moves the represented FLOAT32 bits without conversion, including signed zero,
subnormal, infinity, and NaN payload encodings. `ONE_HOT=17` writes exact BOOL bytes zero and one.
Before creating MPSGraph tensor data or touching a target, native execution scans each retained
INT32 input in stable node order and row-major logical ordinal. Java translates the first failure
to the exact Model `IndexOutOfBoundsException` text. There is no negative-index normalization,
selector-out-of-bounds reliance, INT64 route, CPU fallback, transfer widening, backward claim, or
general BOOL consumer.

### Exact replacement Scatter Elements

For canonical `FLOAT32 data:[2,3]`, canonical `INT32 indices:[2,2]`, and canonical
`FLOAT32 updates:[2,2]`, the profile-free backend may execute:

```text
s = SCATTER_ELEMENTS(data, indices, updates, axis=1, reduction=NONE) // [2,3] FLOAT32
n = NEG(s)
publish s, n
```

`SCATTER_ELEMENTS=18` uses only the data-taking
`scatterAlongAxis:withDataTensor:updatesTensor:indicesTensor:mode:name:` selector with
`MPSGraphScatterModeSet`. Each update replaces the selected coordinate with its non-negative
index; unique addressed cells copy exact update bits and unaddressed cells preserve exact data
bits. Data, indices, and updates remain unchanged. Native execution first validates every resource,
then completes the scatter's row-major bounds pass, then derives complete target coordinates and
sorts bounded executable-owned primitive scratch to reject collisions before creating tensor data,
dispatching, or touching a target. Java's exceptional rescan publishes the exact Model bounds or
smallest-later/smallest-earlier duplicate message. Equal indices at different non-axis coordinates
are not duplicates. There is no skip, overlap winner, arithmetic scatter, Scatter-ND, INT64, alias,
fallback, transfer widening, or complete backward claim.

### Exact general-axis unfold

For canonical `FLOAT32 input:[2,6]`, the profile-free backend may execute:

```text
u = UNFOLD_AXIS(input, axis=1, size=3, step=2) // [2,2,3] FLOAT32
n = NEG(u)
publish u, n
```

`UNFOLD_AXIS=19` creates one documented half-open strided slice for each window offset `k=0..2`,
inserts a singleton final axis into each slice, and concatenates them in ascending `k`. The source
coordinate is `p*2+k`, so the windows overlap at source coordinate two and an incomplete suffix is
omitted. Native creation independently rederives the count and every selector bound, authenticates
canonical states and exact Shapes, and proves selector integers representable before graph
compilation. The result preserves signed zero, subnormal, infinity, and quiet/signaling NaN payload
bits and leaves the input unchanged. There is no padding, dilation, alias, fold, two- or
three-dimensional window, transfer widening, CPU fallback, or backward claim.

The public Engine path for a supported Metal graph uses the same contracts:

```java
MetalBackendConfiguration configuration =
        new MetalBackendConfiguration(nativeLibrary.toAbsolutePath());
try (Engine.Builder builder = Engine.builder()) {
    builder.takeOwnership(MetalBackendIntegration.open(configuration));
    try (Engine engine = builder.build()) {
        CompiledGraph graph = engine.compile(List.of(input.neg()));
        try (PreparedExecution prepared = engine.prepare(graph);
                RunResult result = engine.run(prepared, List.of(input))) {
            HostTensorValue value =
                    result.materialize(result.publications().getFirst(), maximumBytes);
        }
    }
}
```

For `input = [1.0, -2.0]`, the detached canonical value represents `[-1.0, 2.0]`. Opening and
preparation require the configured Apple-silicon host; there is no `Engine.standard()` Metal
variant or automatic Metal selection outside the explicitly registered inventory.

## ABI and failures

Private ABI version `7` exports exactly thirteen symbols: the version/context/buffer foundation,
`synaptik_metal_mpsgraph_executable_create`, executable release/run, and the three typed custom
singleton-NEG pipeline operations. The create function accepts a pointer plus unsigned byte count
for one schema-20 program image. The old
`synaptik_metal_context_certification_environment` and
`synaptik_metal_mpsgraph_neg_executable_create` symbols are absent. Statuses `0..12` retain their
documented meanings, status `13` reports a structurally valid registered operation without a
current native route, and unknown integers fail closed with the raw value retained.

The schema-20 image is little-endian, at least 124 bytes, and at most `Integer.MAX_VALUE` bytes on
both sides. Its core contains a 124-byte/31-word header, 40-byte value descriptors, 32-byte node
descriptors, 64-bit dimensions, 64-bit element strides, 32-bit value references, and 64-bit
attribute words. Every section follows its predecessor immediately; in particular, the reference
pool is never padded before attributes. The header begins with `SM20=0x30324d53`, schema `20`,
header bytes `124`, total bytes, and fixed route; it has no numerical-profile wire. Each value descriptor carries layout
presence, kind, view and dense-physical flags, stride-pool offset, storage offset, and referenced
span.

A `CUSTOM_PROGRAM` image additionally carries authoritative 40-byte step records, 32-bit member
positions, 24-byte bindings, the sorted 32-bit materialized-value set, 64-byte typed execution
instructions, the canonical ASCII manifest, and its 32-byte SHA-256 digest. Generator schema 2
admits fixed, graph-boundary, generated-pointwise, and anchor-epilogue steps; the generated Metal
source carries the matching `synaptik pointwise fusion generator schema 2` marker. Predecessor
generator schema 1 remains rejected. Anchor instructions bind source order, ADD side and external
role, and raw scalar/clamp bits. The manifest binds typed plan agreement and integer source byte
counts only; Java neither emits Metal source nor owns fixed, generated, or assembled source hashes.
Header counts, source sizes, rejection/cap fields, record ranges, materialized slot mappings,
manifest text, and the manifest digest must recompute exactly. An MPSGraph image clears its
extension flag, generator schema, counts, sizes, and caps and physically omits every extension
section. Pre-cutover schema-19 images, route zero, and every other schema or route fail closed.

Production capability admits exactly 86 operation kinds while 29 remain false; structural native
execution covers 101 kinds and leaves 14 nonexecutable. Attribute wires `0..41` and type wires
`1..7` cover all current Model signatures and carriers.

Java and native code independently require exact operation/attribute/type/cardinality agreement,
ordered feeds, targets, node inputs and outputs, exact Shapes, checked physical byte geometry,
validated layout kind/offset/positive-stride/span relationships, explicit
unavailable/canonical/affine-view/materialized-layout state transitions, topological availability,
fresh outputs, unique produced targets, unpadded canonical packed sections, extension lockstep,
and signed-32-bit image bounds. Pointwise admission additionally requires fully static positive
ranks `1..16` and element counts in `1..UINT32_MAX`. Java and native both reject stale schema-19
images and invalid occurrence/route combinations before native entry. Java owns typed handle
liveness and pointer-region preconditions that a raw C boundary cannot prove.
Index-domain and buffer-copy bounds map to status `5`; input/output aliasing and wrong device or
insufficient extent map to status `10`; grid representability maps to status `8`; unusable
threadgroup geometry and custom command failures map to status `11`; Objective-C exceptions map to
status `7`. MPSGraph compilation remains status `9`, while custom compilation and target proof use
status `12`.

## Evidence composition

The task-era probes and certificates below are historical evidence for earlier profile and
schema allocations. They do not certify current schema-20 routes by themselves. Current
qualification must check the profile-free Model contract, current source/package authentication,
native preflight, and focused route conformance; a passing old sample is not capability authority.

Task 0005 originally delivered MPSGraph binary arithmetic. A subsequent independent real-device
audit showed that MPSGraph arithmetic does not preserve strict `FLOAT32` subnormal behavior, so
the operations remained excluded from `STRICT_IEEE`. Task 0015 performed a new mandatory
disposable M3 Max probe at optimization levels 0 and 1, with reduced-precision fast math disabled,
independently compiled executables, repeated fresh contexts, separate operation gates,
equal/row/column/scalar-tensor broadcasting, fan-out, repeated operands, direct guarded targets,
permuted bindings, and exact input/canary controls. All control and bounded-profile
oracle comparisons passed for `ADD`, `SUB`, `MUL`, and `DIV`, enabling only the explicit
`ACCELERATOR` route.

The historical accelerator contract was Model's total recursive `FLOAT32` result set, not IEEE
equality, an `allclose` oracle, or a performance claim. For the implemented binary routes, each
subnormal input may be consumed exactly or as same-signed zero, a finite subnormal result may be
exact or FTZ, and each basic arithmetic site performs one FLOAT32 round-to-nearest-even operation.
NaN remains NaN and ordinary finite, infinity, and required signed-zero classes remain governed by
the Model formula and existing publication freedoms. Reduced precision, reciprocal substitution,
cross-node contraction, algebraic rewrites, and tolerance acceptance remain forbidden. The
historical 0015 probe evidence qualified earlier routes under that earlier set; it is not a
schema-20 certificate.

Task 0019 reused the still-current 0016 exact ABS selector gate because only planning documents
changed after the recorded executable baseline and the same Apple M3 Max, SDK/toolchain, selector,
oracle, and corpus remained in use. The reused gate covered 108 independently compiled ABS
executables and 864 runs across optimization levels, Shapes through rank sixteen, two contexts,
repeated/fan-out/direct-target topology, and exact raw-bit controls. Fresh post-implementation
real-dylib and CPU-free Engine evidence then exercised both profiles. `ABS` preserves every
non-NaN magnitude bit, maps negative zero to positive zero and both infinities to positive
infinity, and preserves NaN classification.

Task 0020 reran the corrected reduction gate from a fresh disposable Objective-C source before
production edits. After correcting one invalid one-term sum-to-Shape control case in the probe
itself, the full matrix passed: 300 independently compiled executables and 2,400 runs across two
fresh contexts and optimization levels zero and one, with reduced-precision fast math disabled.
The oracle enumerated every declared-term permutation and full binary tree, applied `FLOAT32`
rounding plus permitted DAZ/FTZ after every arithmetic step, and admitted zero-sign freedom only
at a final multi-term SUM/SUM_TO_SHAPE root or the mandatory final MEAN quotient. Empty-axis
identities and one-term SUM/SUM_TO_SHAPE identities remained exact copies. MEAN always performs
the mandatory positive-count FLOAT32 quotient and retains the row's DAZ/FTZ/NaN freedoms, even
when the selected count is one. The disposable source and binary were removed before production
edits.

Task 0062 needs no new empirical selector qualification because wire `85` is not lowered through an
opaque MSE selector. The reviewed native graph-construction switch emits exactly one
`subtractionWithPrimaryTensor:secondaryTensor:`, reuses that exact difference as both inputs to one
`multiplicationWithPrimaryTensor:secondaryTensor:`, and then either publishes the square or applies
the already qualified full-axis `reductionSumWithTensor:axes:` or `meanOfTensor:axes:`. The only
following operation is a Shape-only scalar reshape when MPSGraph's reduction rank differs from the
declared scalar. There is no algebraic expansion, reciprocal replacement, reduced-precision
intermediate, hidden contributor, dropped contributor, or alternate vendor MSE selector.

That finite source composition was the task-era authorization proof for the admitted domain. Model's
loss formula still names the same `SUB`, `delta*delta`, and exact reduction sites. Tasks 0015 and
0020 supplied earlier accelerator-route evidence, but that historical recursive argument does
not alone qualify every current schema-20 result. Focused raw-bit
tests independently enumerate the same per-site DAZ/FTZ, signed-zero, NaN-class, all-binary-tree,
all-contributors-once, and final-quotient choices and exercise both quiet and signaling NaNs,
infinities, maximum finite values, normal/subnormal boundaries, repeated inputs, direct targets,
reuse, isolated sessions, and input preservation. Those runs corroborate the fixed composition;
they do not authorize it by sampling. Java/native parity tests separately reject wrong reduction,
type, rank, Shape, and gradient metadata before resource creation. The other loss and
normalization wires remain structurally cataloged but production-false.

Task 0021 ran a completely fresh disposable M3 Max MATMUL probe before production edits. The
matrix passed 384 independently compiled executables and 3,072 runs across two contexts,
optimization levels zero and one, direct/left-transpose/right-transpose/both-transpose forms,
eight Shape/data cells per form, stable/permuted bindings, and guarded direct targets. Its oracle
enumerated legal DAZ/FTZ choices, all declared-term reassociations and corresponding FMA choices,
required every term exactly once, and admitted either zero sign only at final publication of an
exact-zero complete contraction. Reduced-precision fast math was set to and read back as `None`;
all selector, Shape, canary, input-preservation, all-term, and precision sentinels passed. The
disposable source and binary were removed before production edits.

Task 0023 ran one fresh disposable M3 Max Objective-C smoke before production edits using the
documented MPSGraph selectors. One context and one execution proved axis GATHER preserved raw
FLOAT32 signed-zero, subnormal, infinity, and NaN-payload bits, and ONE_HOT produced the exact
`[3,4]` BOOL shape and zero/one bytes. Input buffers were unchanged. The disposable source and
binary were removed before production edits; no optimization, context, or repetition matrix was
run.

Task 0024 ran exactly one fresh disposable M3 Max Objective-C smoke before production edits using
the documented data-taking scatter selector with Set mode. One context, graph, executable,
optimization choice, and execution proved addressed updates and unaddressed base cells preserved
raw signed-zero, subnormal, infinity, and signaling-NaN payload bits, and that all three inputs were
unchanged. The disposable source and binary were removed immediately; no bounds, duplicate,
Shape/type/profile/context/optimization, or repetition matrix was run.

Task 0025 ran exactly one fresh disposable M3 Max Objective-C smoke before production edits using
the documented slice/expand/concat composition. One context, graph, executable at optimization
level zero, and execution used input `[2,6]`, axis one, size three, and step two. The exact
`[2,2,3]` output proved overlap, omitted tail, and unchanged signed-zero, subnormal, infinity, and
quiet/signaling-NaN payload bits; the input remained unchanged. The disposable source and binary
were removed immediately; no Shape, rank, axis, size, step, profile, context, optimization, or
repetition matrix ran.

Task 0070 binds generated pointwise execution to four universal Lean result-set certificates over
the shared binary32 `Word`/`RawClass` substrate. A checked two-sided codec covers every one of the
`2^32` raw words; each `FLOOR`, `CEIL`, `SIGN`, and `RELU` helper result belongs to its model
relation, whose permitted set is proved unique. The source certificate authenticates the Java
integer-size planner/image writer, native parser and sole source emitter, compact slot mapping,
strongest available reflection, and test-only dispatch observation site. Native validates frozen
SHA-256 values for all ten fixed components and their ordered 84,541-byte total, then validates the
assembled total independently before compiler entry. One structured sink implements both a
no-allocation count traversal and the exact byte traversal: production validates all five caps
before allocating the counted byte size and emits once for compile. A standalone native fixture includes that exact
production header, enables audit-only singleton generation, repeats generation and compares count,
bytes, and hash, and is the sole writer of the compiled audit source; Python only parses that
source and writes metadata. A second native fixture and the Java formula cover every decimal-width
boundary from `9/10` through `999999999/1000000000` plus `UINT32_MAX` without constructing large
programs. Schema-19 hostile images place otherwise valid NEG and ABS nodes in generated-member
ranges, independently of malformed generated or anchor instruction records and predecessor-schema
images.

A pinned Xcode 27 audit first bootstraps the Metal toolchain and records the exact available
`clang`, `metal`, `metallib`, `metal-objdump`, and `metal-nm` paths, resolved identities, versions,
byte counts, and SHA-256 values. It compiles the exact 84,541-byte fixed corpus plus audit-only
singleton sites for `FLOOR`, `CEIL`, `SIGN`, and `RELU` and valid runtime-grammar chain sites at
every length `2..8`, using `metal3.2`, no fast math, exact `-Wall -Wextra -Werror`, and an absolute
SDK isysroot. An authenticated structured LLVM/AIR instruction parser, rather than text opcode
matching, produces the semantic reproducibility ledger for every generated function. Separately
compiled `fadd`, `fmul`, `fdiv`, and `air.fma`/`llvm.fma` negative sites prove that provider rejects
each forbidden class. The generated source, AIR, metallib, canonical disassembly, symbol table,
provider inventory, semantic ledger, and negative ledger remain in the ignored evidence build
directory, with every actual byte count and hash recorded. AIR and metallib also retain observed
reference hashes; their compiler-generated container identifiers may vary, so the pinned canonical
structured instruction ledger is the reproducibility authority. Real `metal-objdump`, `metal-nm`,
and runtime reflection evidence verify every generated function and the three-buffer 32-byte
`PointMeta` ABI.
A separately linked test-only observer records the manifest digest and metadata immediately before
each real custom dispatch. It proves generated `7+2` and three-instruction plans and one mixed
generated/MPSGraph/fixed plan containing L1_NORM, singleton VARIANCE, and SCATTER_ADD; both public
and observer proof runners force Gradle task reruns rather than accepting cached test results.

Task 0071 adds a separate source/compiler-site certificate and a checked Lean model of literal
scalar/add/terminal order, one physical dispatch, zero intermediate stores, one final store,
right-aligned rank-one indexing, and exhaustive DAZ/FTZ boundary alternatives. MATMUL anchor
recognition and native authentication reuse the general exact geometry accepted by the existing
custom kernel: vector/matrix, matrix/vector, vector dot with scalar output, batched geometry, and
right-aligned batch broadcasting. Exact-output residuals and ordinary right-aligned addends,
including trailing `[N]` bias, retain their existing semantics. Malformed contractions, output
shapes, and batch broadcasts fail closed.

The pinned `metal3.2 -fno-fast-math` audit compiles the exact ten-component runtime source through
a location-independent stdin virtual source and inspects all eight anchor kernels plus the retained
Conv2d helper. It records seven unflagged FMA sites, six unflagged `fmul` sites, four unflagged
`fadd` sites, no unsafe flags, and exactly one static final byte store in each kernel. Real-device
observer coverage proves one dispatch for fused general-geometry MATMUL and Conv2d, exact width
broadcasting for a rank-one Conv2d addend, and NaN, signed-zero, infinity, subnormal FTZ, and
clamp-endpoint behavior. A public `Engine` smoke exercises both fused families and the general
MATMUL cases and observes the new structural PREPARE and planned RUN payloads. Fusion admission is
all-or-none at 64 anchors; structural trace scans only through cap-plus-one and reports count 65 as
the exceeded-cap sentinel with bounded composed member facts.

Historical task validation comprised the following evidence; the current schema-20 integration
gate is separate:

- focused capability, historical schema-nineteen/native-preflight, raw-bit native execution, route-identity,
  backend-conformance, and CPU-free public Engine tests for the bounded former two-profile
  UNFOLD_AXIS domain, wire `19`, typed `WINDOW_AXIS=6`, schema rejection, overlap/tail mapping,
  input preservation, and same-partition composition;
- current generic lifecycle, prepared-byte-geometry, target publication, reuse, concurrency, close,
  malformed-image, and indexing diagnostics without duplicating their matrices for one wire;
- a rebuilt arm64 dylib inspected for exactly thirteen exports, ABI `7`, required framework
  linkage, absence of the removed certification-environment export, and absence of the old
  NEG-only create symbol;
- focused real-device native and Engine proof of exact signed-zero, subnormal, infinity, and
  quiet/signaling-NaN raw-bit movement with unchanged input;
- real-device BFLOAT16/FLOAT16 arithmetic and raw-preserving execution through the custom program,
  plus native and Java rejection of low MPSGraph images;
- architecture tests, Metal Javadoc, the one final repository build, Markdown validation, and diff
  validation.
- historical real-device ABS execution under both profiles across signed zeros, subnormal and normal
  boundaries, ordinary finite values, maximum finite values, infinities, and multiple signed
  quiet/signaling NaNs, plus strict and accelerator compositions, published intermediates, direct
  targets, reuse, concurrency, independent contexts, input preservation, and close rejection;
- retained real-device binary execution of each operation with broadcasting, operand-order checks,
  chains, published intermediates, fan-out, repeated operands, input preservation, repeated runs,
  and the independent bounded raw-bit oracle;
- real-device reduction execution for full, single-axis, ordered multi-axis, empty identity,
  keep-dimensions, and sum-to-Shape forms, including exact copy identities, scalar publication,
  positive-rank ABS/binary composition, direct targets, repetitions, and input preservation;
- real-device MATMUL execution for direct, left-transposed, right-transposed, and both-transposed
  operands across executable reuse, exact outputs, direct targets, and input preservation;
- the retained CPU-free Metal-only public Engine indexing smoke that publishes exact GATHER bits and
  BOOL bytes, proves sole Metal ownership, mutates retained index buffers for deterministic GATHER
  and ONE_HOT exceptions, and uses no CPU owner;
- a CPU-free Metal-only public Engine reduction run that compiles, prepares, reuses and
  concurrently opens sessions, publishes and materializes exact scalar/vector raw bits, rejects
  strict ownership and post-close work, and has no CPU owner;
- CPU-free Metal-only public Engine proofs for FLOAT32/BFLOAT16/FLOAT16 MATMUL and broadcast
  gradients, mixed low execution through explicit FLOAT32 casts, gradient-bearing scalar
  arithmetic and reciprocal, homogeneous low cumulative-product backward, non-overlapping
  maximum-pool backward, deterministic dropout saved-mask replay, and FLOAT16 second-order
  differentiation; and
- real-device MSE execution for `NONE`, `SUM`, and `MEAN` through the fixed
  `SUB -> MUL -> qualified full reduction` composition, with source-derived recursive
  floating-result oracles, direct and nested custom-program execution, reuse, isolated sessions,
  input preservation, and malformed-image parity.

Compiler contract coverage checks exact forward view layouts, explicit rank-edit materialization,
inverse first-order operations, saved-value lifetime, and higher-order request publication. No
capability here implies implicit seeding or unrestricted training support.

The public `GraphCompilationPort` intentionally supplies no explicit positive-rank forward
constant ingress, so CPU-free Engine scenarios use caller inputs rather than claiming a public
compile-time splat. Backend-local coverage passes exact type-matching logical splats in all seven
schema carriers through `PrepareContext.constants()` and verifies fresh initialized Metal feed
buffers.

## Registration and composition

The integration is supplied explicitly to `Engine.Builder`; there is no `ServiceLoader`, global
registry, or runtime service locator. `Engine.standard()` remains CPU-only, while an explicitly
composed Engine may register Metal alone or beside CPU.

`MetalBackendIntegration.open(MetalBackendConfiguration)` and its optional traced overload own
configuration validation, native library loading, context construction, availability production,
host ingress and materialization, partition preparation, the retained
`MetalLocalWorkloadTuning`/`MetalCompletePlanTuning` collaborations, physical contribution, direct
upload/download endpoints, and partial-open rollback. `Engine.Builder.takeOwnership(...)`
transfers that complete opened owner into Engine. Engine owns registration, duplicate-ID
validation, compile-time inventory, complete owner/transfer preflight, shared preparation, global
schedule composition, tuning execution and evidence, per-occurrence outer adapter capture, and
closure. Metal host ingress and materialization occur outside Runtime without re-querying the
registry. The dependency remains one-way: Engine may depend on Metal; Metal production never
depends on Engine.

An explicitly composed Engine may mix CPU and Metal partitions. Shared Prepare assigns one
representation position per participating owner; Metal contributes its exact buffer/workspace
creators. Engine inserts a direct CPU-to-Metal upload or Metal-to-CPU download once for each
distinct destination owner immediately before its first consumer. The current path accepts all
seven carriers over fully static rank-0..16 canonical or positive-stride non-overlapping layouts
with checked physical spans, without conversion, retry, fallback, or on-demand discovery. A
persistent Metal splat binding is accepted only as a Metal read source; it cannot be an upload
destination. Runtime executes only the resulting direct prepared references.

For one exact singleton NEG Metal plan, public `prepareTuned(...)` measures the complete two-route
local batch, authenticates the selected version-twenty-nine decision, then correctness-checks and
times one complete-plan candidate fixed to that route. The returned production recipe is freshly
prepared after representative cleanup. Metal policy, producer, and decision-codec identities are
adapter-owned and distinct from CPU identities; existing CPU bytes and behavior remain unchanged.
Both Metal phases are `SESSION`, so workload-cache entries cannot hit or publish and the
model-plan path receives no input/output. An allowed recoverable fallback performs ordinary
preparation through the already-selected Metal adapter and never switches owner.

Mixed-owner plans, plans with multiple partitions, and MPSGraph-only partitions remain ineligible
for Metal tuning. Metal production has no Compiler or Engine dependency. Architecture tests lock
that direction and the API-visible Engine dependency on Metal. Builder lifecycle tests cover
entry-time transfer, snapshot and order freezing, duplicate-ID rejection, terminal failed build,
reverse cleanup, and composition through `CONTIGUOUS`, historical strict
binary/reduction fail-closed ownership, CPU ownership when CPU is explicitly registered,
and historical accelerator binary and reduction execution. Separate CPU/Metal tests cover custom
singleton execution, exact mixed classification/logic/WHERE execution, canonical BOOL ingress,
fixed custom-program routing, both transfer directions, adapter use after registry lookup is
poisoned, unchanged CPU tuning, public bounded Metal tuning, and selected-owner fallback.
These implement the construction boundary in
[ADR 0015](../design/decisions/0015-explicit-engine-backend-composition.md) and the current
owner-indexed mixed schedule in
[ADR 0016](../design/decisions/0016-cpu-metal-mixed-owner-schedule.md).

## Limitations and related documentation

Current Metal support remains limited to the admitted occurrence rows above; the historical
`STRICT_IEEE` and `ACCELERATOR` columns no longer select capability. Masked and floating product
reductions, unqualified statistical/norm/log-sum-exp families, attention, convolution transpose,
and recurrent execution remain unsupported. MATMUL accepts positive-static homogeneous
FLOAT32/BFLOAT16/FLOAT16 vector, matrix, batched, and right-aligned broadcast geometry plus
qualified one-low-plus-FLOAT32 widening rows and no-gradient promoted INT32/INT64 pairs.
FLOAT64-result, direct BFLOAT16/FLOAT16 mixed,
zero-extent, dynamic, malformed, and disallowed-gradient MATMUL forms remain false.

The low-precision closure adds no generic cross-carrier or layout promise beyond the documented
occurrence predicates. It admits all 49 casts, seven-carrier raw movement, exact
classification/BOOL/WHERE rows, and the indexed/fold subsets above. Dynamic or empty geometry,
unresolved or unsafe external strides, arithmetic scatter, duplicate replacement destinations,
overlapping folds, Conv3d backward, and every production-false kind remain rejected. Variadic
support is limited to one-through-sixteen CONCAT/STACK. Rank zero is a real one-element value in
the admitted cast, affine, movement, predicate/WHERE, publication, and generated-gradient
compositions; it never means an empty buffer.

The fixed shared custom route is not a general custom-kernel framework: cold preparation owns the
reviewed fixed corpus, authenticated bounded pointwise generator, and any interleaved existing-node
executables. Only the compact materialized set has declared assigned buffers, and hot Java
execution makes one synchronous native call with no host staging, retry, fallback, or hidden
materialization. Selected affine logical views retain their exact Shape/strides/offset while all
resource sizing and native addressing use independently authenticated physical spans.

There is no asynchronous API, general output/workspace pool, executable serialization, discovery,
persistent route cache, alias promise, portable RNG stream, entropy source, random-quality claim,
or performance claim. Immutable source-owned splat buffers persist only within their exact
`PreparedExecution`; mutable outputs, masks, next-state values, and workspaces remain fresh per
run. Executable backward and higher-order paths include explicitly seeded homogeneous floating
MATMUL, bounded Conv2d/pooling formulas, primitive-closed dropout, floating casts, inverse affine
movement, broadcast reduction, arithmetic/scalar/reduction/scan formulas, exact replacement,
saved-role, and non-overlapping window closure. They add no implicit seeding or inferred operation
ownership. RNN, GRU, and LSTM remain unavailable under the current profile-free contract, including zero-length/no-work
special cases and all gradients.

Related documentation:

- [Metal backend contract](../architecture/contracts/backend-execution.md#metal-backend)
- [Runtime / Prepare / Backend boundary](../architecture/runtime-prepare-backend-boundary.md)
- [Prepared-resource lifecycle ADR](../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)
- [Metal strategy note](../design/notes/metal-backend-strategy.md)
- [Metal task 0003](../planning/backends/metal/tasks/0003-single-neg-custom-metal-kernel-route.md)
- [Metal task 0004](../planning/backends/metal/tasks/0004-typed-metal-route-candidate-generators-and-cache-compatibility.md)
- [Metal task 0005](../planning/backends/metal/tasks/0005-mpsgraph-mixed-binary-whole-partition.md)
- [Metal task 0008](../planning/backends/metal/tasks/0008-mpsgraph-float32-affine-transforms.md)
- [Metal task 0014](../planning/backends/metal/tasks/0014-mpsgraph-float32-affine-layout-composition.md)
- [Metal task 0015](../planning/backends/metal/tasks/0015-accelerator-float32-tensor-binary-arithmetic.md)
- [Metal task 0019](../planning/backends/metal/tasks/0019-exact-profile-qualified-float32-abs.md)
- [Metal task 0020](../planning/backends/metal/tasks/0020-accelerator-float32-reductions-after-zero-sign-refinement.md)
- [Metal task 0021](../planning/backends/metal/tasks/0021-accelerator-float32-rank2-matmul-after-zero-sign-refinement.md)
- [Metal task 0023](../planning/backends/metal/tasks/0023-exact-int32-gather-and-one-hot.md)
- [Metal task 0024](../planning/backends/metal/tasks/0024-exact-int32-scatter-elements-replacement.md)
- [Metal task 0025](../planning/backends/metal/tasks/0025-exact-float32-unfold-axis-materialization.md)
- [Metal task 0058](../planning/backends/metal/tasks/0058-remaining-elementwise-arithmetic.md)
- [Metal task 0061](../planning/backends/metal/tasks/0061-general-static-matmul-domain.md)
- [Metal task 0062](../planning/backends/metal/tasks/0062-accelerator-mse-and-normalization-loss-boundary.md)
- [Metal task 0063](../planning/backends/metal/tasks/0063-exact-ordering-and-arg-extrema.md)
- [Metal task 0064](../planning/backends/metal/tasks/0064-convolution-pooling-and-attention-boundary.md)
- [Metal task 0065](../planning/backends/metal/tasks/0065-explicit-state-rng-dropout-and-recurrent-boundary.md)
- [Metal task 0066](../planning/backends/metal/tasks/0066-dtype-layout-gradient-gap-closure.md)
- [Native ABI and build guide](../../native/metal-macos-arm64/README.md)

## Numerical semantics and capability

Metal capability and preparation have no numerical-profile selector. They admit the listed exact
unary, all-49-cast, affine, indexing,
classification, scalar/broadcast BOOL, promoted WHERE, raw movement/replacement, non-overlapping
fold/window, no-gradient promoted INT32/INT64 MATMUL, ordering/top-K/numeric arg-extrema, maximum-
pool, and initial-state occurrences. Metal also admits the documented FLOAT32 tensor/scalar
arithmetic, comparisons, reductions, scans, general
MATMUL, MSE, grouped convolution, average pooling, explicit-state dropout, and the narrow
no-gradient L1/ScatterAdd plus singleton VARIANCE occurrences. Homogeneous BFLOAT16/FLOAT16
counterparts use the general low custom route; the Task-0069 specialized kernels below are
FLOAT32-only implementations, not operation-wide dtype restrictions.

P0 freezes these current F32 answers through the canonical provider-derived capability ledger and
independently queries each corresponding BF16 witness. Existing BF16 true rows are preserved; only
actual BF16 false rows may map to future backend-relative parity. It activates no Metal type wire,
schema, identity, capability, route, execution, native, or evidence behavior. The former
`STRICT_IEEE`/`ACCELERATOR` columns are historical provider-baseline evidence, not current selectors.
Route, generated-backward, and
runtime/device facts remain outside provider output.

Every Task-0066 selected occurrence is custom-only. Exact authenticated local logical views may
depart from canonical layout only where their physical descriptors and spans are independently
proved; all selected materializing outputs are canonical. These routes remain inside Model's
exact/discrete or recursive primitive/aggregate floors and gain no generic final-output tolerance.
The bounded generated-gradient closure is the exact capable graph, not a new semantic formula or
general backward promise.

The Task-0069 specialized FLOAT32 L1, ScatterAdd, and singleton VARIANCE slices select the fixed
custom program. The specialized FLOAT32 L1 slice independently authenticates unsigned-32-bit
element and four-byte span bounds, one-thread dispatch, distinct buffers, raw ABS leaves,
contributor order, and one logical publication. The specialized FLOAT32 ScatterAdd slice admits
only axis zero, positive canonical rank-one FLOAT32 base/update/output, a materialized canonical
rank-one INT32/INT64 index feed, and no gradient flags. A complete CPU index scan precedes every
command encoding and mutation. One output thread raw-loads each base word once, retains duplicate
matching updates in source order, and stores once; unaddressed cells select the unchanged raw word,
and no atomic operation exists. The compiler explicitly canonicalizes its generated zero base, so
the same domain closes the existing Gather data cotangent end to end. The specialized FLOAT32
VARIANCE slice admits only
canonical FLOAT32 `[1]`, axes `[0]`, correction zero, no gradients, distinct buffers, and scalar or
retained `[1]` output. One writer evaluates its four literal sites and stores the shared canonical
cell once, with no fallback, classifier, retry, timing, clamp, reciprocal, FMA, or tolerance.
Other VARIANCE shapes/corrections remain outside production capability but retain their established
direct MPSGraph structural recipe; the singleton cutover neither rejects nor diverts them.

The historical Task-0069 task-local Lean proof and source certificate were paired with a compiled-MSL/AIR audit of the
exact assembled runtime source under Xcode 27.0 build 27A266a, Metal 32023.921, macOS SDK 27.0,
`metal3.2`, no-fast-math, warnings as errors, and the explicit SDK isysroot. AIR contains exactly
one unflagged `fadd` in each L1/ScatterAdd kernel. The VARIANCE kernel contains exactly two
unflagged `fdiv`, one `fsub`, one `fmul`, no `fadd` or FMA, and one static store; the scatter body
also retains its ordinal-zero filter, raw-base unaddressed select, one static store, and no atomics.

There is no profile in partition plans or route/tuning/codec/workload identities. Java
enforces the boundary before native entry. ABI version `7` exposes exactly thirteen export names
and accepts one bounded schema-20 route-bearing image with a 124-byte header; operation wires `1..115`, attribute wires
`0..41`, route wires `1..3`, and type wires `1..7` cover the current structural registry without
widening operation-kind capability. Route, candidate, compatibility, workload, exact-policy, and
codec identities are version `30`; every other identity fails closed. Capability/structural counts
remain `86/29` and `101/14`.
The complete-plan wrapper remains version `1`.
