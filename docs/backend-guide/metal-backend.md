# Metal backend

## Outcome and supported scope

The Metal backend executes one whole maximal profile-homogeneous Metal partition. Under both
profiles the common exact domain contains parameterless `NEG` and `ABS`, `RESHAPE`, `EXPAND`,
`PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`, the explicit `CONTIGUOUS` canonicalization barrier, bounded
canonical `FLOAT32` `UNFOLD_AXIS`, canonical positive-rank `FLOAT32` data `GATHER` with canonical
`INT32` indices, canonical positive-rank `INT32`-to-`BOOL` `ONE_HOT`, and canonical positive-rank
`FLOAT32`/`INT32`/`FLOAT32` `SCATTER_ELEMENTS/NONE` replacement. UNFOLD_AXIS accepts canonical
input rank `1..15`, size `1..16`, positive step, and size no larger than the selected extent. Its
fresh canonical output has the exact floor-count rank-plus-one Shape, repeats overlapping source
representations, omits incomplete tails, applies no padding, and does not mutate its source. Graph
feeds and direct `NEG` or `ABS` operands are canonical dense, zero-offset non-views. An affine or
`CONTIGUOUS` input may also be an exact resolved zero-offset view produced by an earlier admitted
affine node in the same partition. Affine outputs retain exact Model/Compiler logical view geometry.
`CONTIGUOUS` has unchanged Shape and canonical output geometry; it must separate an affine view
from `NEG` or `ABS`.

`ACCELERATOR` additionally admits tensor `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, and `MAX`; all six
binary comparisons; exact FLOAT32 scalar `MIN`, `MAX`, and fused `CLAMP`; canonical FLOAT32
reductions; `CUM_SUM` and `CUM_PROD`; and positive static rank-two FLOAT32 `MATMUL`. Comparisons
publish canonical one-byte BOOL. Extrema reductions join `SUM` and `MEAN` across full, normalized
single-axis, ordered normalized multi-axis including the empty identity, and exact
keep-dimensions forms; `SUM_TO_SHAPE` remains SUM-only. Scans accept every exclusive/reverse mode
and preserve Shape. Binary operations have two canonical dense, zero-offset non-view inputs, and
their output Shape is the exact right-aligned broadcast. Scalar operations and scans preserve
Shape and retain exact FLOAT32 attribute words. MATMUL accepts canonical inputs or authenticated
local transposes of canonical sources, requires exact `[M,K] @ [K,N] -> [M,N]` geometry, and
produces a canonical output. Reduction and ordinary inputs are positive-rank `1..16`; a locally
produced reduction target may be scalar. Caller ingress accepts exact canonical FLOAT32 and INT32,
and a locally produced canonical BOOL result may publish as exact one-byte zero/one values.
BOOL feeds, consumers, and CPU/Metal transfer remain unsupported; transfer stays canonical
FLOAT32-only. Every descriptor is fully static and has the operation-specific exact type, layout,
Shape, and `requiresGrad == false` relationship. The common exact operations have the same Model
result contract in both profiles; Task-0052 operations exist only under ACCELERATOR.

```text
capability -> Planning ownership -> Metal analysis and typed candidates
           -> optional decision authentication and private route selection
           -> exact declarations
           -> shared slot assignment -> Metal finalization
           -> PreparedExecution -> isolated Runtime run
```

Capability applies per occurrence. Preparation then authenticates the complete maximal partition
that Planning forms: common NEG/ABS/affine/canonicalization/indexing graphs and, under accelerator,
valid compositions with binary, comparison, extrema, scalar, reduction, scan, and MATMUL nodes,
stable fan-out, repeated and ordered inputs, internal publications, and multiple feeds and targets.
A positive-rank reduction result may compose with other supported accelerator nodes; a scalar
reduction result is a direct target only. A comparison BOOL result is a direct local publication
only. A local transpose accepted as a MATMUL operand must be produced inside the same partition
from a canonical source, but its other valid affine consumers and boundary publication remain
available. Scalar kinds other than exact FLOAT32 MIN/MAX/CLAMP; masked, product, and other
reductions; every unary operation other than listed NEG and ABS; logical operations and BOOL
consumers; strict Task-0052 operations and strict MATMUL; vector or batched MATMUL; every window
kind except bounded canonical FLOAT32 UNFOLD_AXIS; indexing outside the listed exact rows; INT64
indexing, profile-crossing operations, unsupported attributes/types, zero or dynamic extents,
unresolved layout, noncanonical graph feeds, foreign views, mismatched descriptors, and
multi-output forms remain fail-closed.

Within that capability domain, Metal analysis generates a typed complete candidate batch
and selects one of three private routes:

- `CUSTOM_SINGLE_NEG` for exactly one NEG occurrence under either profile, one unique feed, one
  unique target, and a checked element count in `1..UINT32_MAX`;
- `CUSTOM_TASK0052` for a complete ACCELERATOR partition containing any Task-0052 node; or
- `MPSGRAPH` for every other supported partition.

With no selected decision, the first candidate preserves this exact heuristic. A compatible
session-local decision can select the other valid route for an eligible singleton. This remains a
prepare-time implementation-domain boundary: it is not capability narrowing, CPU fallback,
retry, repartitioning, or a performance-superiority claim.

The three identities are one closed package-private prepared-route enum. It owns candidate wires
`1`, `2`, and `3` and the two-value `CUSTOM_KERNEL`/`MPSGRAPH` family; the candidate batch delegates
wire encoding to that identity rather than maintaining another wire table. Every returned plan
retains one non-null final identity. Finalization, trace metadata, cold binding, and execution read
that retained identity and cannot replace it.

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
| Capability truth | Public `MetalCapabilityProvider` reports the exact canonical FLOAT32 NEG/ABS/affine/`CONTIGUOUS`/bounded `UNFOLD_AXIS`, FLOAT32+INT32 GATHER, INT32-to-BOOL ONE_HOT, and FLOAT32/INT32/FLOAT32 `SCATTER_ELEMENTS/NONE` domains under both profiles and the accelerator-only tensor-binary, reduction, and positive static rank-two MATMUL domains above. |
| Native configuration and integration | Public `MetalBackendConfiguration` and `MetalBackendIntegration` belong to Metal. Metal validates configuration, opens native ownership, and rolls partial construction back before Engine can take the completed integration. |
| Backend ownership | Planning chooses `owner = metal` and groups consecutive equal owners; it never selects MPSGraph or a custom kernel. |
| Analysis | Package-private Metal code validates the complete partition, assigns stable structural value order, regenerates typed route candidates and session compatibility, authenticates any supplied decision, fixes one route, and declares that route's exact resources. |
| Opaque tuning handoff | Metal can construct the Prepare-owned marker-role handoff with no decision or one Metal decision. Shared Prepare does not inspect private candidates; the Metal preparer treats a present decision as untrusted. |
| Shared preparation | `GraphPreparation` projects facts, assigns slots, validates the result, and transfers persistent resources transactionally. It does not inspect the Metal plan or route. |
| Finalization | Metal validates exact assignments, compiles either one typed custom pipeline or one shape-specialized MPSGraph executable, then allocates/uploads each source-owned immutable splat in feed order. It never reselects the route. |
| Persistent resources | The route resource and zero or more immutable splat resources are returned in physical acquisition order. `PreparedExecution` uniquely owns each prepared owner. A splat owner may have already-issued read-only run-binding child leases; owner close rejects new bindings and the last child performs exact-once physical buffer release. Custom-pipeline and MPSGraph handles are never interchangeable. |
| Per-run state | Runtime borrows exact canonical FLOAT32/INT32 caller buffers, owns one fresh read-only binding object for each persistent source splat, and owns fresh typed output buffers. Only MPSGraph runs also own one fresh native-address workspace. Mutable outputs/workspaces are never shared or pooled. The scatter executable owns bounded primitive coordinate/uniqueness scratch rebuilt under its synchronized run. Affine targets receive full logical-byte buffers; a scalar reduction target receives four bytes and a BOOL target one byte per element. |
| Hot invocation | Cold binding authenticates and unwraps immutable splat bindings, retains direct typed buffers/handles, validates every resource and then every indexing element in stable node/ordinal order, completes each scatter's bounds before complete-coordinate uniqueness, and makes one synchronous typed native call into assigned output destinations only on success. |
| Optional trace observation | A traced integration emits only final PREPARE and native RUN outcomes through a caller-owned `MetalTraceObserver`. PREPARE succeeds only after route and splat acquisition. The ordinary open overload creates no producer or trace work. Observation never selects a route, retries, falls back, or changes ownership or results. |
| Publication | Runtime leases the already-resident output representation. Canonical FLOAT32 outputs, local rank-zero reduction outputs, local canonical BOOL outputs, exact authenticated dense affine targets, and a readable persistent splat binding may be downloaded as detached canonical host bytes. Ordinary CPU/Metal transfer remains positive-rank canonical FLOAT32 and rejects logical views; a persistent splat is read-source-only. |

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
claim. No mapping among them is implied. ABI 5, the thirteen native exports, schema 13, and every
version-14 workload/exact-policy/candidate/compatibility/route/codec identity remain fixed; tuning
remains session-scoped and non-persistent.

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
sequences beginning at zero. Each event uses `System.nanoTime()`. After analysis fixes a route and
before finalization, Metal retains the prepared-unit ID, requested profile, and neutral route.
Successful or failed executable/resource finalization emits one `PREPARE` event while tracing is
enabled. Each custom-kernel or MPSGraph native invocation similarly emits one `RUN` event. There
are no start, transfer, allocation, binding, materialization, publication, or close events.

Preparation always reports `NOT_QUERIED`: Metal authenticates an optional outer tuning decision
but performs no cache lookup and cannot claim an outer hit or miss. Native codes `0..12` map to
`SUCCESS`, `INVALID_ARGUMENT`, `DEVICE_UNAVAILABLE`, `COMMAND_QUEUE_UNAVAILABLE`,
`ALLOCATION_FAILED`, `RANGE_OUT_OF_BOUNDS`, `COPY_FAILED`, `INTERNAL_ERROR`,
`UNSUPPORTED_SHAPE`, `COMPILATION_FAILED`, `INCOMPATIBLE_RESOURCE`, `EXECUTION_FAILED`, and
`COMPILATION_FAILED`, respectively. Other signed codes map to `UNKNOWN` without losing the code;
a Java-side failure before a native return has no native status. MPSGraph range status is captured
before the existing Java indexing rescan translates the outward exception.

ID exhaustion, trace-object construction failure, or an observer `RuntimeException` atomically
disables later tracing without changing backend work, failure, rollback, suppression, or outward
exception. An observer `Error` from success reporting propagates normally. During failure
reporting, the existing finalization or route-specific run failure remains primary; a distinct
observer `Error` is attached as an acyclic suppressed failure after any rollback suppression,
without disabling tracing. Payloads include only Trace-owned IDs and closed facts; they never
include the library path, device/session token, address, handle, Tensor/storage value, scalar,
shape, byte extent, workload/candidate fingerprint, thread identity, exception, free-form string,
or generic map. Tracing changes no capability, route, native call, native ABI/schema/export,
result, lifecycle, or Engine production behavior.

### Capability, whole-partition analysis, and route selection

`MetalCapabilityProvider.supports` checks one occurrence, its numerical profile, and its exact
descriptors. Under either profile it admits a resolved zero-offset view input only for affine or
`CONTIGUOUS` occurrences because the query contains no graph closure; `NEG` and `ABS` require one
canonical input and an equal canonical output. Exact `UNFOLD_AXIS` requires a canonical FLOAT32
input of rank `1..15`, normalized axis, size `1..16`, positive step, size no larger than the selected
extent, equal input/output gradient eligibility, and the exact canonical rank-plus-one output with
position count `floor((D-S)/T)+1`. Exact `GATHER` requires canonical positive-rank `FLOAT32` data,
canonical positive-rank `INT32` indices, one normalized data axis, the exact axis-replacement output
Shape, and matching forward gradient eligibility. Exact `ONE_HOT` requires canonical positive-rank
`INT32` input, a positive depth, exact appended-axis canonical `BOOL` output, and no gradient
request. Exact `SCATTER_ELEMENTS` requires ordered canonical
`FLOAT32`/`INT32`/`FLOAT32` data, indices, and updates, `ScatterReduction.NONE`, equal positive
rank and exact indices/update Shape, non-axis agreement with data, normalized axis, a canonical
data-shaped output, and output gradient eligibility equal to the data/update OR. In accelerator
mode each binary requires two canonical inputs and the exact broadcast output. A reduction requires
one positive-rank canonical input and its exact full, normalized-axis, keep-dimensions, empty-axis
identity, or binding-resolved sum-to-Shape output; only a reduction output may be rank zero.
Accelerator MATMUL requires two positive static rank-two FLOAT32 inputs, exact contraction and
output Shapes, canonical output, and canonical or exact transpose layout operands. Strict MATMUL
is false. Availability and hard backend requirements remain separate Planning facts.

The package-private `MetalOperationRouteCatalog` separately describes every one of the 115
schema-thirteen `NodeKind` values. Exhaustive enum switching yields shared immutable entries with
closed MPSGraph state/reason and custom-kernel state/reason values: MPSGraph totals are
`76 DIRECT / 34 COMPOSED / 5 UNAVAILABLE`; custom totals are
`16 AVAILABLE / 99 PENDING / 0 UNAVAILABLE_WITH_PROOF`. The normative
[per-wire evidence audit](../planning/backends/metal/tasks/0056-route-evidence-audit.md) supplies
the exact installed-SDK selector or finite composition and current Model source for every row.
This catalog performs no capability admission and no selection. It is never consulted by Runtime;
all 81 registered-but-nonexecutable kinds remain capability-false even when the catalog records a
structurally direct or composed MPSGraph realization.

After Planning creates one maximal Metal partition, analysis walks nodes in partition order with
explicit unavailable, canonical, and affine-view states. Every view input must resolve to an
earlier admitted affine producer in that exact partition; every graph feed is canonical;
`CONTIGUOUS` produces canonical state; and `NEG` and `ABS` reject affine-view state.
`UNFOLD_AXIS` consumes and produces canonical FLOAT32 state and retains normalized axis, size, and
step in its typed node. `GATHER` consumes canonical FLOAT32 data and canonical INT32 indices and
produces canonical FLOAT32; `ONE_HOT` consumes canonical INT32 and produces canonical BOOL.
`SCATTER_ELEMENTS` consumes ordered canonical FLOAT32 data, canonical INT32 indices, and canonical
FLOAT32 updates and produces canonical FLOAT32. Accelerator binary and reduction nodes consume
only canonical FLOAT32 values and produce canonical FLOAT32; binary nodes preserve the exact
broadcast Shape, while reduction lowering preserves its typed form, ordered axes, keep-dimensions
state, sum-to-Shape target, and checked term geometry. MATMUL produces canonical state and may
consume affine state only when analysis authenticates that exact operand as a local rank-two
transpose of a canonical source. That authentication constrains only the MATMUL input edge; the
view may be published or used by another admitted affine operation. A positive-rank FLOAT32 result
can feed a later compatible node; BOOL is locally publishable but has no general consumer, and a
scalar result must be a direct target. Stable value indexing follows first encounter. Repeated use
names the same value. Only feeds and published boundary targets receive Runtime slots; other
intermediates remain symbolic MPSGraph tensors.

Analysis receives compile-time constant sources through `PrepareContext.constants()`. A boundary
constant must be an exact type-matching `FLOAT32` or `INT32` splat. Analysis records whether the
current Metal partition is that producer-free value's first ordered consumer. Only that source
partition prepares physical storage; later Metal consumers reuse the same logical Metal
representation, while a value first owned by CPU retains the ordinary transfer destination.
Runtime uses an `InitializedBuffer` fresh read-only binding for a prepared Metal source rather than
consuming a caller position. Existing shared `GraphPreparation` tests independently enforce the
chain `CompileConstantPlan.ConstantSource -> PrepareContext.constants() -> InitializedBuffer`.

Once stable values, states, feeds, targets, checked byte geometry, and typed node records are known,
analysis creates a version-fourteen candidate batch and workload fingerprint. Partitions without a
Task-0052 node retain the ordinary MPSGraph candidate, with the existing bounded singleton-NEG
alternative where eligible. Any partition containing a Task-0052 node has exactly the fixed
`CUSTOM_TASK0052` whole-program candidate; tuning cannot replace it with MPSGraph or split it.

The version-fourteen canonical workload fingerprint covers the explicit numerical-profile wire,
the schema-thirteen program image, ordered input and output references, all typed attributes,
affine/indexing states, authenticated local-transpose provenance, complete tensor descriptors and
logical layouts, target and internal-value sets, exact scalar/splat bits, logical-boundary roles,
policy/candidate/route schemas, and ABI version. It encodes structural positions rather than graph
object identity. Target compatibility also contains a fresh private nonce from the exact
`MetalDeviceContext`; version-thirteen and earlier decisions fail closed.

Metal can construct an absent- or present-decision `BackendPartitionTuningHandoff`. Fresh analysis
always regenerates the current batch. Every supplied handoff is accepted only when the exact
partition, candidate schema, workload fingerprint, and context session match. An absent decision
then uses the existing heuristic without cache lookup or measurement; a present decision must
additionally name a current candidate. Stale or foreign values fail closed rather than reverting
to the heuristic.

After authentication, `CUSTOM_SINGLE_NEG` declares one feed buffer, one target buffer, and no
workspace. `CUSTOM_TASK0052` declares every feed, target, and internal logical-value buffer plus one
address workspace. MPSGraph declares its feed and target buffers plus one address workspace. A
larger supported singleton has only the MPSGraph candidate; analysis does not reject or split it.

Package-private tests can require one exact route only after ordinary lowering, capability
admission, semantic validation, candidate regeneration, and any supplied handoff authentication
have succeeded. The required route must already be a member of that fresh exact batch. Thus both
approved singleton-NEG routes can traverse the same finalization/lifecycle/publication path
independently, while Task-0052 MPSGraph forcing, singleton Task-0052 forcing, stale or foreign
handoffs, and unsupported operations fail before native creation. No public configuration,
integration, Engine, tuning, or Runtime input exposes this test seam.

### Session decision codec and limitations

The package-private version-fourteen Metal codec produces bounded canonical compatibility,
candidate, and checksummed decision bytes. Decode rejects wrong magic, schema, session scope,
numerical profile, malformed or truncated content, trailing or corrupt bytes, changed workload or
context, and unknown or pruned candidates. The bytes contain no native handle or executable.
Version-thirteen and earlier codec bytes and cross-profile decisions fail closed even when their
trailing checksum is otherwise valid.

The public `MetalLocalWorkloadTuning` retained by `MetalBackendIntegration` wraps this codec and
candidate generator without changing their version-fourteen bytes. It returns a handoff only for an
exact singleton NEG whose complete ordered candidate list is `[CUSTOM_SINGLE_NEG, MPSGRAPH]`.
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

Schema-thirteen workload bytes and workload compatibility remain route-neutral. Candidate and
decision bytes retain route wires `1..3`, route-policy version fourteen, and the target session.
Prepared plans and native resources are route-specific. A future executable-cache key would
therefore require the tuple `(workload compatibility, route wire, route-policy version, target
session)` rather than a workload digest alone. The repository has no persistent Metal executable
cache, and this catalog/identity refactor adds none.

### Finalization and persistent ownership

Shared Prepare assigns all declared slots before Metal finalization. The finalizer checks
declaration identity, order, geometry, slot uniqueness, exact `MetalDeviceContext` identity,
source-owner representation position, and route/workspace agreement before native work. It creates
the selected route resource first, then each source-owned splat resource in stable feed order, and
constructs one immutable `PreparedExecutable` recipe.

For the singleton-NEG custom route, native creation compiles the fixed branch-free
`synaptik_neg_f32` source and creates one pipeline. For a Task-0052 custom program, creation
compiles the fifteen fixed reviewed safe-math kernels and any interleaved existing-node MPSGraph
executables before publication. For the ordinary MPSGraph route, native creation validates one
canonical bounded schema-13 program image and compiles one fixed-shape whole-partition executable.
MATMUL compilation requires reduced-precision-fast-math read-back `None`. All compilation happens
during finalization, never invocation.

Each source splat owns one exact-sized Metal buffer and context child lease. Finalization fills and
uploads the exact `FLOAT32` or `INT32` raw bits once, before PREPARE success. The result returns the
route resource followed by source splats in acquisition order. A malformed native route result
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
4. an MPSGraph run allocates one native address-array workspace, while a custom run allocates no
   workspace; and
5. cold binding validates context, byte extent, scalar type/raw bits, read/write role, and physical
   input/output non-aliasing before unwrapping direct handles.

MPSGraph cold binding writes ordered native handles into its address workspace once and retains
direct input/output slices. Custom cold binding retains the exact underlying input and output
buffers and their typed native handles directly. A later run-setup failure closes its fresh child
binding and other created run resources without closing the prepared splat owner. Metal-to-CPU
transfer and canonical materialization may unwrap a live splat binding as a read source;
CPU-to-Metal upload and executable output binding reject it before native mutation.

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
with Shape `[1_048_576]` under `STRICT_IEEE`. It uses the public
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

The schema-1 report retains all raw samples and exact correctness hashes and is always marked
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

### Accelerator reductions and scalar publication

For canonical `FLOAT32` input `x` with Shape `[2, 3, 4]` and values `1..24`, one accelerator
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
schema-thirteen program image. `m`, `a`, `s`, and `c` remain positive-rank canonical values, so they
may compose inside the partition. The scalar `t` is a direct target and cannot feed another node.
One run publishes `c = [[71, 78, 85, 92], [83, 90, 97, 104]]` plus `t = 300`; local materialization of
`t` copies exactly four bytes. Strict ownership rejects the same reduction graph before native
creation, and CPU/Metal transfer does not become scalar-capable.

### Accelerator rank-two MATMUL and seeded gradients

For canonical `left:[M,K]` and `right:[K,N]`, accelerator Metal lowers `left.matmul(right)` to one
`MATMUL=15` node. No-bias `left.linear(weight:[N,K])` retains the visible local
`PERMUTE [1,0] -> MATMUL` chain; the backend does not add a `LINEAR` operation or materialization.
An explicitly seeded first-order compile for both operands retains the existing shape-restoration
boundaries around exactly:

```text
left gradient  = seed @ transpose(right)
right gradient = transpose(left) @ seed
```

The local transposes and MATMUL nodes execute in one CPU-free Metal partition. This is a narrowly
proved explicitly seeded rank-two backward path, not scalar-loss training, batched MATMUL,
implicit seeding, or a general Metal training claim. Strict Metal rejects the same MATMUL graph
during ownership selection.

### Exact GATHER and ONE_HOT

For canonical `FLOAT32 data:[2,4]`, canonical `INT32 gatherIndices:[3]`, and canonical
`INT32 classIndices:[3]`, either profile may execute:

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
`FLOAT32 updates:[2,2]`, either profile may execute:

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

For canonical `FLOAT32 input:[2,6]`, either profile may execute:

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

Private ABI version `5` exports exactly thirteen symbols: the version/context/buffer foundation,
`synaptik_metal_mpsgraph_executable_create`, executable release/run, and the three typed custom
singleton-NEG pipeline operations. The create function accepts a pointer plus unsigned byte count
for one schema-13 program image. The old `synaptik_metal_mpsgraph_neg_executable_create` symbol is
absent. Statuses `0..12` retain their documented meanings, status `13` reports a structurally valid
registered operation without a current native route, and unknown integers fail closed with the raw
value retained.

The schema-13 image is little-endian, at least 64 bytes, at most `Integer.MAX_VALUE` bytes on both
sides, and consists of a 64-byte header, 16-byte value descriptors, 32-byte node descriptors,
64-bit dimensions, 32-bit value references, canonical alignment padding, and 64-bit attribute
words. Header and create-argument byte counts must match exactly. Operation wires `1..34` retain
their executable meaning; wires `35..115` are registered structural vocabulary that return status
`13` only after complete image validation. Attribute wires `0..41` and type wires `1..6` cover all
current Model signatures and carriers.

Java and native code independently require exact operation/attribute/type/cardinality agreement,
ordered feeds, targets, node inputs and outputs, exact Shapes and checked byte geometry, explicit
unavailable/canonical/affine-view state transitions, topological availability, fresh outputs,
unique produced targets, canonical packed sections, zero reserved/padding bytes, and signed-32-bit
image bounds. Java additionally rejects profile-incompatible programs before native entry and owns
typed handle liveness and pointer-region preconditions that a raw C boundary cannot prove.
Index-domain and buffer-copy bounds map to status `5`; input/output aliasing and wrong device or
insufficient extent map to status `10`; grid representability maps to status `8`; unusable
threadgroup geometry and custom command failures map to status `11`; Objective-C exceptions map to
status `7`. MPSGraph compilation remains status `9`, while custom compilation and target proof use
status `12`.

## Evidence composition

Task 0005 originally delivered MPSGraph binary arithmetic. A subsequent independent real-device
audit showed that MPSGraph arithmetic does not preserve strict `FLOAT32` subnormal behavior, so
the operations remained excluded from `STRICT_IEEE`. Task 0015 performed a new mandatory
disposable M3 Max probe at optimization levels 0 and 1, with reduced-precision fast math disabled,
independently compiled executables, repeated fresh contexts, separate operation gates,
equal/row/column/scalar-tensor broadcasting, fan-out, repeated operands, direct guarded targets,
permuted bindings, and exact input/canary controls. All control and bounded-profile
oracle comparisons passed for `ADD`, `SUB`, `MUL`, and `DIV`, enabling only the explicit
`ACCELERATOR` route.

The current accelerator contract is Model's total recursive `FLOAT32` result set, not IEEE
equality, an `allclose` oracle, or a performance claim. For the implemented binary routes, each
subnormal input may be consumed exactly or as same-signed zero, a finite subnormal result may be
exact or FTZ, and each basic arithmetic site performs one FLOAT32 round-to-nearest-even operation.
NaN remains NaN and ordinary finite, infinity, and required signed-zero classes remain governed by
the Model formula and existing publication freedoms. Reduced precision, reciprocal substitution,
cross-node contraction, algebraic rewrites, and tolerance acceptance remain forbidden. The
historical 0015 probe evidence certifies these current routes as a subset of that recursive set.

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

Current validation composes:

- focused capability, schema-thirteen/native-preflight, raw-bit native execution, route-identity,
  backend-conformance, and CPU-free public Engine tests proving the exact bounded two-profile
  UNFOLD_AXIS domain, wire `19`, typed `WINDOW_AXIS=6`, schema rejection, overlap/tail mapping,
  input preservation, and same-partition composition;
- current generic lifecycle, prepared-byte-geometry, target publication, reuse, concurrency, close,
  malformed-image, and indexing diagnostics without duplicating their matrices for one wire;
- a rebuilt arm64 dylib inspected for exactly thirteen exports, ABI `5`, required framework
  linkage, and absence of the old NEG-only create symbol;
- focused real-device native and Engine proof of exact signed-zero, subnormal, infinity, and
  quiet/signaling-NaN raw-bit movement with unchanged input;
- architecture tests, Metal Javadoc, the one final repository build, Markdown validation, and diff
  validation.
- real-device ABS execution under both profiles across signed zeros, subnormal and normal
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
  strict ownership and post-close work, and has no CPU owner; and
- a CPU-free Metal-only public Engine proof for direct rank-two MATMUL, no-bias linear's visible
  right transpose, and explicitly seeded gradients for both operands, including graph/formula,
  owner, raw-bit publication, reuse, independent-session, input-preservation, close, and strict
  rejection assertions.

Compiler contract coverage checks exact forward view layouts and inverse first-order operations.
The existing indexing formula guard remains unchanged; this task adds no Compiler production or
Metal indexing backward claim. Metal execution includes only the explicitly seeded rank-two
MATMUL gradient formulas above and does not imply broader backward or training support.

The public `GraphCompilationPort` intentionally supplies no explicit positive-rank forward
constant ingress, so CPU-free Engine scenarios use caller inputs rather than claiming a public
compile-time splat. Backend-local coverage passes exact type-matching `FLOAT32` and `INT32` logical
splats through `PrepareContext.constants()` and verifies fresh initialized Metal feed buffers.

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
distinct destination owner immediately before its first consumer. The current path accepts fully
static canonical contiguous `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, `INT64`, and `BOOL` without
conversion, retry, fallback, or on-demand discovery. A persistent Metal splat binding is accepted
only as a Metal read source; it cannot be an upload destination. Runtime executes only the
resulting direct prepared references.

For one exact singleton NEG Metal plan, public `prepareTuned(...)` measures the complete two-route
local batch, authenticates the selected version-fourteen decision, then correctness-checks and times
one complete-plan candidate fixed to that route. The returned production recipe is freshly
prepared after representative cleanup. Metal policy, producer, and decision-codec identities are
adapter-owned and distinct from CPU identities; existing CPU bytes and behavior remain unchanged.
Both Metal phases are `SESSION`, so workload-cache entries cannot hit or publish and the
model-plan path receives no input/output. An allowed recoverable fallback performs ordinary
preparation through the already-selected Metal adapter and never switches owner.

Mixed-owner plans, plans with multiple partitions, and MPSGraph-only partitions remain ineligible
for Metal tuning. Metal production has no Compiler or Engine dependency. Architecture tests lock
that direction and the API-visible Engine dependency on Metal. Builder lifecycle tests cover
entry-time transfer, snapshot and order freezing, duplicate-ID rejection, terminal failed build,
reverse cleanup, and composition through `CONTIGUOUS`, the default strict
binary/reduction fail-closed ownership boundary, CPU ownership when CPU is explicitly registered,
and CPU-free accelerator binary and reduction execution. Separate CPU/Metal tests cover custom
singleton execution, both transfer directions, adapter use after registry lookup is poisoned,
unchanged CPU tuning, public bounded Metal tuning, and selected-owner fallback.
These implement the construction boundary in
[ADR 0015](../design/decisions/0015-explicit-engine-backend-composition.md) and the current
owner-indexed mixed schedule in
[ADR 0016](../design/decisions/0016-cpu-metal-mixed-owner-schedule.md).

## Limitations and related documentation

Accelerator support is limited to the exact canonical FLOAT32 rows above. Strict binary
arithmetic, every Task-0052 comparison/extrema/scalar/reduction-extrema/scan row, and MATMUL remain
unsupported. Product, masked, statistical, Boolean, arg-extrema, and other reductions remain
unsupported. Accelerator MATMUL is limited to positive static same-type rank two, exact
contraction/output geometry, and canonical or authenticated local rank-two-transpose operands.
Exact canonical ABS is the only accelerator unary operation and receives no numerical relaxation.
Indexing under both profiles is limited to the listed GATHER, ONE_HOT, replacement Scatter
Elements, and UNFOLD_AXIS rows; general BOOL consumers, BOOL ingress/transfer, INT64 indices, and
other indexing remain unsupported. Metal also has no FLOAT16, BFLOAT16, FLOAT64, other integer
computation, scalar graph ingress, integer transfer, zero-extent, dynamic-shape, unresolved-layout,
noncanonical graph-ingress, foreign-view, variadic, or multi-output support. A scalar rank is
admitted only for a locally produced direct reduction target and four-byte local materialization.
The fixed Task-0052 custom route is not a general custom-kernel framework: cold preparation owns
the fifteen reviewed kernels and interleaved existing-node executables, every logical value has a
declared assigned buffer, and hot Java execution makes one synchronous native call with no host
staging, retry, fallback, or hidden materialization. There is no asynchronous API, general
output/workspace pool, executable serialization, discovery, persistent route cache, alias promise,
general backward route, or performance claim. Immutable source-owned splat buffers persist only
within their exact `PreparedExecution`; mutable outputs and address workspaces remain fresh per run.
The only executable backward path is the explicitly seeded rank-two MATMUL formula described
above. Model task 0026 must define FLOAT16
semantics before any backend can advertise it.

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
- [Native ABI and build guide](../../native/metal-macos-arm64/README.md)

## Numerical profiles

Metal capability and preparation make the listed exact canonical movement, affine, indexing, and
NEG/ABS rows common to both profile matrices. `ACCELERATOR` additionally admits tensor FLOAT32
`ADD`/`SUB`/`MUL`/`DIV`/`MIN`/`MAX`, all six comparisons, scalar MIN/MAX/CLAMP, canonical
FLOAT32 SUM/MEAN/MIN/MAX/SUM_TO_SHAPE, every CUM_SUM/CUM_PROD scan mode, and positive static
rank-two FLOAT32 MATMUL with authenticated local transposes. These arithmetic routes remain inside
Model's exact/discrete or recursive primitive/aggregate floors; they gain no generic final-output
tolerance. The profile is retained in partition plans and every route/tuning/codec/workload
identity. Java enforces the boundary before native entry. ABI version `5` retains thirteen export
names and accepts one bounded schema-13 image; operation wires `1..115`, attribute wires `0..41`,
and type wires `1..6` cover the current structural registry without widening capability. Route,
candidate, compatibility, workload, exact-policy, and codec identities are version `14`; the
complete-plan wrapper remains version `1`.
