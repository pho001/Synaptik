# Metal backend

## Outcome and supported scope

The Metal backend executes a whole maximal Metal-owned partition when every occurrence is
parameterless `NEG`, one of `RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE`, or the
explicit `CONTIGUOUS` canonicalization barrier. Every descriptor uses `FLOAT32`, has a fully
static positive Shape of rank `1..16`, and preserves the occurrence's `requiresGrad` flag. Graph
feeds and direct `NEG` operands are canonical dense, zero-offset non-views. An affine or
`CONTIGUOUS` input may also be an exact resolved zero-offset view produced by an earlier admitted
affine node in the same partition. Affine outputs retain exact Model/Compiler logical view
geometry. `CONTIGUOUS` has unchanged Shape and canonical output geometry; it must separate an
affine view from `NEG`. `ADD`, `SUB`, `MUL`, and `DIV` remain deliberately outside the Metal
capability domain.

```text
capability -> Planning ownership -> Metal analysis and typed candidates
           -> optional decision authentication and private route selection
           -> exact declarations
           -> shared slot assignment -> Metal finalization
           -> PreparedExecution -> isolated Runtime run
```

Capability applies per occurrence. Preparation then authenticates the complete maximal partition
that Planning forms: NEG chains, composed affine views, explicit canonicalization, stable fan-out,
repeated inputs, internal publications, and multiple feeds and targets. Every binary arithmetic
operation and every other operation or attribute family, type, rank, zero extent, dynamic or
unresolved layout, noncanonical graph feed, foreign view, affine-to-NEG edge without
`CONTIGUOUS`, mismatched descriptor, and multi-output form remains fail-closed.

Within that capability domain, Metal analysis generates a typed complete candidate batch
and selects one of two private routes:

- `CUSTOM_SINGLE_NEG` for exactly one NEG occurrence, one unique feed, one unique target, and a
  checked element count in `1..UINT32_MAX`; or
- `MPSGRAPH` for every other supported partition, including affine or mixed NEG-and-affine
  partitions and an otherwise matching singleton above `UINT32_MAX`.

With no selected decision, the first candidate preserves this exact heuristic. A compatible
session-local decision can select the other valid route for an eligible singleton. This remains a
prepare-time implementation-domain boundary: it is not capability narrowing, CPU fallback,
retry, repartitioning, or a performance-superiority claim.

## Prerequisites

Java uses JDK 26 Foreign Function and Memory (FFM) APIs. The native bridge requires an
Apple-silicon macOS host, Xcode Command Line Tools, and Foundation, Metal, and
MetalPerformanceShadersGraph. Build and ABI instructions are in the
[native Metal guide](../../native/metal-macos-arm64/README.md).

Backend-local and public integration tests supply the dylib's absolute path directly. The backend
does not discover, extract, package, sign, notarize, or cache the library.
`MetalBackendConfiguration` snapshots that caller-selected absolute path, and
`MetalBackendIntegration.open(configuration)` loads it, validates the ABI, opens the default
device context, and rolls partial construction back before returning. Planning separately receives
the captured Metal availability snapshot; the capability provider performs no native loading or
device discovery.

## Contracts and ownership

| Stage or resource | Owner and current behavior |
|---|---|
| Capability truth | Public `MetalCapabilityProvider` reports only the exact typed `FLOAT32` NEG, affine-composition, and `CONTIGUOUS` domain above. |
| Native configuration and integration | Public `MetalBackendConfiguration` and `MetalBackendIntegration` belong to Metal. Metal validates configuration, opens native ownership, and rolls partial construction back before Engine can take the completed integration. |
| Backend ownership | Planning chooses `owner = metal` and groups consecutive equal owners; it never selects MPSGraph or a custom kernel. |
| Analysis | Package-private Metal code validates the complete partition, assigns stable structural value order, regenerates typed route candidates and session compatibility, authenticates any supplied decision, fixes one route, and declares that route's exact resources. |
| Opaque tuning handoff | Metal can construct the Prepare-owned marker-role handoff with no decision or one Metal decision. Shared Prepare does not inspect private candidates; the Metal preparer treats a present decision as untrusted. |
| Shared preparation | `GraphPreparation` projects facts, assigns slots, validates the result, and transfers persistent resources transactionally. It does not inspect the Metal plan or route. |
| Finalization | Metal validates exact assignments and compiles either one typed custom pipeline or one shape-specialized MPSGraph executable after slots exist. It never reselects the route. |
| Persistent resource | One typed `PreparedResource` owns the selected native handle and context child lease; `PreparedExecution` becomes its sole owner. Custom-pipeline and MPSGraph handles are never interchangeable. |
| Per-run state | Runtime borrows caller buffers and owns fresh initialized-constant buffers and output buffers. Only MPSGraph runs also own one native-address workspace. Affine targets receive full logical-byte buffers, not referenced-span-sized buffers. |
| Hot invocation | A cold-bound route-specific invocation retains direct references and makes one synchronous typed native call into assigned output destinations. |
| Publication | Runtime leases the already-resident output representation. Canonical outputs and exact authenticated dense affine targets may be downloaded as detached canonical host bytes; ordinary CPU/Metal transfer still rejects logical views. |

The public Java surface contains `MetalCapabilityProvider`, `MetalBackendConfiguration`, and
`MetalBackendIntegration`. Contexts, physical storage, preparers, finalizers, schedules, executable
recipes, native handles, Objective-C objects, MPSGraph types, and the custom route remain internal.

## Integration lifecycle

### Capability, whole-partition analysis, and route selection

`MetalCapabilityProvider.supports` checks one occurrence and its exact descriptors. It admits a
resolved zero-offset view input for affine or `CONTIGUOUS` occurrences because the query contains
no graph closure. Availability and hard backend requirements remain separate Planning facts.

After Planning creates one maximal Metal partition, analysis walks nodes in partition order with
explicit unavailable, canonical, and affine-view states. Every view input must resolve to an
earlier admitted affine producer in that exact partition; every graph feed must be canonical;
`CONTIGUOUS` produces canonical state; and `NEG` rejects affine-view state. Stable value indexing
follows first encounter. Repeated use names the same value. Only feeds and published boundary
targets receive Runtime slots; other intermediates remain symbolic MPSGraph tensors.

Analysis receives compile-time constant sources through `PrepareContext.constants()`. A boundary
constant must be an exact `FLOAT32` splat. The run uses an `InitializedBuffer` for that feed rather
than consuming a caller position. Existing shared `GraphPreparation` tests independently enforce
the chain `CompileConstantPlan.ConstantSource -> PrepareContext.constants() -> InitializedBuffer`.

Once stable values, states, feeds, targets, checked byte geometry, and typed node records are
known, analysis creates a version-four candidate batch and workload fingerprint. MPSGraph is valid
for every supported partition. The custom candidate exists only for one `NEG` node, one feed, one
target, and an element count in `1..UINT32_MAX`; affine and `CONTIGUOUS` nodes never select it.
Candidate order is the current safe heuristic first and then the other valid route, so a positive
budget returns a stable prefix and budget one cannot change ordinary preparation.

The version-four canonical workload fingerprint covers typed node kinds and attributes, ordered
edges, explicit value states, complete tensor descriptors and logical layouts, dense represented-
order geometry, target set, exact `FLOAT32` splat bits, logical-boundary roles, exact/default
policy, candidate and route-policy schemas, native node schema, and ABI version. It encodes
structural positions rather than `NodeId`, `ValueId`, or partition object identity, so equal
occurrences within one live context compare equally. Target compatibility also contains a fresh
private nonce from the exact `MetalDeviceContext`; stale version-three decisions fail closed.

Metal can construct an absent- or present-decision `BackendPartitionTuningHandoff`. Fresh analysis
always regenerates the current batch. A present decision is accepted only when the exact partition,
candidate schema, workload fingerprint, context session, and candidate membership all match;
stale or foreign values fail closed rather than reverting to the heuristic. Absence uses the
existing heuristic without cache lookup or measurement.

After authentication, the custom route declares one feed buffer, one target buffer, and no
workspace. MPSGraph declares its feed and target buffers plus one address workspace. A larger
supported singleton has only the MPSGraph candidate; analysis does not reject or split it.

### Session decision codec and limitations

The package-private version-four Metal codec produces bounded canonical compatibility, candidate,
and checksummed decision bytes. Decode rejects wrong magic, schema, session scope, malformed or
truncated content, trailing or corrupt bytes, changed workload or context, and unknown or pruned
candidates. The bytes contain no native handle or executable. Earlier codec versions fail closed
even when their trailing checksum is otherwise valid.

This codec is only the backend-side authentication foundation. It performs no file input/output,
measurement, winner selection, or persistent reuse, and there is no `tools/tuning`, Engine, or
outer `WorkloadCompatibility` adapter. The tools-owned cache artifact version, checksum, atomic
replacement, objective, sampling, and cache lifecycle remain separate. Cross-session Metal reuse
requires a separately authorized stable device/library fingerprint.

### Finalization and persistent ownership

Shared Prepare assigns all declared slots before Metal finalization. The finalizer checks
declaration identity, order, geometry, slot uniqueness, exact `MetalDeviceContext` identity, and
route/workspace agreement. It then performs exactly the native create operation for the selected
route and constructs one immutable `PreparedExecutable` recipe.

For the custom route, native creation compiles the fixed branch-free `synaptik_neg_f32` Metal
Shading Language source and creates one `MTLComputePipelineState`. For MPSGraph, native creation
validates a fixed-width version-four typed node table and compiles one fixed-shape
`MPSGraphExecutable` for the whole partition. Compilation happens during prepare finalization,
never during invocation.

The typed resource owns the selected native handle and one context child lease. A provisional
lease prevents concurrent context close from invalidating native creation. A malformed native
create result fails closed: success with a null handle is rejected, while failure with a non-null
handle releases that handle once and preserves a distinct cleanup failure as suppressed evidence.
Failed finalization similarly rolls its resource back locally. After finalizer return, shared
Prepare owns rollback until a validated `PreparedExecution` accepts the resource exactly once.

`PreparedExecution.close()` rejects new runs without waiting. Previously admitted synchronous
runs may finish, and physical resource release is deferred until the last admitted run lease
ends. Context owner close follows the same child-lease boundary. Cleanup is idempotent,
reverse-order, and attempt-all; the first failure remains primary and later distinct failures are
suppressed.

### Cold run setup and binding

Each run receives isolated mutable state:

1. caller input positions borrow caller-owned Metal buffers;
2. each constant feed allocates a fresh run-owned Metal buffer and uploads the exact raw
   `FLOAT32` splat bits once;
3. each target allocates a fresh run-owned output buffer; affine-view targets use the full positive
   logical element count and retain exact finalized-route authentication, while canonical
   `CONTIGUOUS` and NEG targets use the ordinary canonical buffer path;
4. an MPSGraph run allocates one native address-array workspace, while a custom run allocates no
   workspace; and
5. cold binding validates context identity and byte extents and rejects input/output aliasing.

MPSGraph cold binding writes ordered native handles into its address workspace once and retains
direct input/output slices. Custom cold binding retains the exact input and output representations
and their typed native handles directly. An allocation or upload failure closes the current and
previously created run-owned resources through Runtime rollback. No constant buffer is prepared
once, shared between runs, or owned by a backend-global cache. Materialization accepts a logical
affine view only when the live buffer authenticates the exact finalized executable, preparation
plan, context, target position, value identity, producer kind, descriptor, and full logical byte
extent.

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

The public Engine path for a supported NEG uses the same contracts:

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

Private ABI version `4` exports exactly thirteen symbols: the version/context/buffer foundation,
`synaptik_metal_mpsgraph_executable_create`, executable release/run, and the three typed custom
singleton-NEG pipeline operations. The old
`synaptik_metal_mpsgraph_neg_executable_create` symbol is absent. Statuses `0..12` retain their
documented meanings; unknown integers fail closed with the raw value retained.

The create ABI requires node schema `4` and one 160-byte discriminated record per node. Accepted
operations are `NEG=1`, `RESHAPE=6`, `EXPAND=7`, `PERMUTE=8`, `EXPAND_DIMS=9`, `SQUEEZE=10`, and
`CONTIGUOUS=11`. Former binary operation wires `2..5` are withdrawn and rejected. Attribute kinds
are none, target Shape, permutation, and normalized axis; target dimensions and permutations
occupy a bounded sixteen-`uint64_t` payload. Java and native code require exact
operation/attribute pairing, explicit unavailable/canonical/affine-view state transitions,
`UINT32_MAX` absent-input/axis sentinels, zero reserved and unused payload fields, positive ranks
and dimensions, complete permutations, valid axes, topological availability, fresh outputs,
unique produced targets, exact declared Shapes, and checked full logical byte geometry. Java
additionally owns typed handle liveness and pointer-region preconditions that a raw C boundary
cannot prove. Input/output aliasing and wrong device or insufficient extent map to status `10`;
grid representability maps to status `8`; unusable threadgroup geometry and custom command
failures map to status `11`; Objective-C exceptions map to status `7`. MPSGraph compilation
remains status `9`, while custom compilation and target proof use status `12`.

## Evidence composition

Task 0005 originally delivered MPSGraph binary arithmetic. A subsequent independent real-device
audit on Apple M3 Max exercised default and reduced optimization settings across 576 runs and
79,488 arithmetic comparisons. All identity, permutation, input-preservation, and canary controls
passed, while 12,096 arithmetic comparisons failed in the subnormal domain. Basic MSL float
arithmetic, including safe math mode, showed the same flush behavior. Because the Model requires
ordinary IEEE `FLOAT32` underflow semantics, the accepted remediation withdraws all four binary
operations rather than adding value-dependent dispatch, fallback, or an unproved custom float
implementation.

Current validation composes:

- the mandatory disposable Apple-silicon probe at MPSGraph optimization levels 0 and 1, with
  three independently compiled executables per topology and level, 32 corpus rotations plus a
  concurrent execution per executable, independent process sessions, direct canary-filled targets
  for every requested intermediate/final value, permuted bindings, rank-one/rank-sixteen
  boundaries, fan-out, same-Shape `CONTIGUOUS`, and exact raw-bit equality;
- focused backend tests proving all four binary occurrences remain rejected, withdrawn schema
  wires `2..5` fail closed, schema-v4 state transitions accept local affine composition and reject
  unavailable/view-to-NEG paths, and preparation retains stable ordering, authentication, and
  lifecycle behavior;
- backend-conformance coverage for exact public truth and maximal composed partition closure;
- a rebuilt arm64 dylib inspected for exactly thirteen exports, ABI `4`, required framework
  linkage, and absence of the old NEG-only create symbol;
- real-device backend execution for custom NEG, backend-local typed logical splats, retained
  affine selector/direct-target evidence, and view-to-`CONTIGUOUS`-to-NEG composition; and
- a Metal-only public Engine run publishing composed intermediate/final views and canonical
  results with exact raw bits, plus the subnormal binary regression that rejects Metal ownership
  and leaves the explicitly registered CPU as the sole owner.

Compiler contract coverage still checks exact forward view layouts and inverse first-order
operations, including `SUM_TO_SHAPE` for `EXPAND`. Metal execution evidence remains forward-only;
it adds no Metal backward or training claim.

The public `GraphCompilationPort` intentionally supplies no explicit positive-rank forward
constant ingress, so the CPU-free Engine scenario uses caller inputs rather than claiming a public
compile-time splat. Backend-local real-device coverage passes an exact `FLOAT32` logical splat
through `PrepareContext.constants()` and verifies fresh initialized Metal feed buffers on repeated
runs.

## Registration and composition

The integration is supplied explicitly to `Engine.Builder`; there is no `ServiceLoader`, global
registry, or runtime service locator. `Engine.standard()` remains CPU-only, while an explicitly
composed Engine may register Metal alone or beside CPU.

`MetalBackendIntegration.open(MetalBackendConfiguration)` owns configuration validation, native
library loading, context construction, availability production, host ingress and materialization,
partition preparation, physical contribution, direct upload/download endpoints, and partial-open
rollback. `Engine.Builder.takeOwnership(MetalBackendIntegration)` transfers that complete opened
owner into Engine. Engine owns registration, duplicate-ID validation, compile-time inventory,
complete owner/transfer preflight, shared preparation, global schedule composition, per-occurrence
outer adapter capture, and closure. Metal host ingress and materialization occur outside Runtime
without re-querying the registry. The dependency remains one-way: Engine may depend on Metal;
Metal production never depends on Engine.

An explicitly composed Engine may mix CPU and Metal partitions. Shared Prepare assigns one
representation position per participating owner; Metal contributes its exact buffer/workspace
creators. Engine inserts a direct CPU-to-Metal upload or Metal-to-CPU download once for each
distinct destination owner immediately before its first consumer. The current path accepts only
fully static canonical contiguous `FLOAT32` and performs no conversion, retry, fallback, or
on-demand discovery. Runtime executes only the resulting direct prepared references.

The builder does not make Metal's current backend-local candidates available to public
`prepareTuned(...)`. A Metal-owned plan rejects that CPU-only workflow before representative
input borrowing, candidate generation, or trial work; an allowed CPU safe-heuristic fallback
cannot select or prepare Metal.

Metal production has no Compiler or Engine dependency. Architecture tests lock that direction and
the API-visible Engine dependency on Metal. Builder lifecycle tests cover entry-time transfer,
snapshot and order freezing, duplicate-ID rejection, terminal failed build, reverse cleanup, and
pre-analysis transfer-domain rejection. Public integration covers Metal-only NEG, affine
composition through `CONTIGUOUS`, the subnormal binary fail-closed ownership boundary, and CPU
ownership when CPU is explicitly registered. A separate CPU/Metal test covers custom singleton
execution, both transfer directions, adapter use after registry lookup is poisoned, CPU tuning
with Metal registered, and early Metal tuning rejection.
These implement the construction boundary in
[ADR 0015](../design/decisions/0015-explicit-engine-backend-composition.md) and the current
owner-indexed mixed schedule in
[ADR 0016](../design/decisions/0016-cpu-metal-mixed-owner-schedule.md).

## Limitations and related documentation

The current routes have no binary arithmetic because available Metal arithmetic paths do not
preserve required `FLOAT32` subnormals. They also have no FLOAT16, BFLOAT16, FLOAT64, integer,
BOOL, scalar-rank, zero-extent, dynamic-shape, unresolved-layout, noncanonical graph-ingress,
foreign-view, variadic, or multi-output support. There is no general custom-kernel framework,
asynchronous API, cross-run overlap guarantee, buffer pool, persistent constant buffer, executable
serialization, packaging, discovery, persistent route cache, current tuning integration, alias
promise, backward/training route, or performance claim. Model task 0026 must define FLOAT16
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
- [Native ABI and build guide](../../native/metal-macos-arm64/README.md)
