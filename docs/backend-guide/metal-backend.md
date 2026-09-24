# Metal backend

## Outcome and supported scope

The Metal backend executes one whole maximal profile-homogeneous Metal partition. Under
`STRICT_IEEE`, every occurrence is parameterless `NEG` or `ABS`, one of `RESHAPE`, `EXPAND`,
`PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE`, or the explicit `CONTIGUOUS` canonicalization barrier.
Strict graph feeds and direct `NEG` or `ABS` operands are canonical dense, zero-offset non-views. A
strict affine or `CONTIGUOUS` input may also be an exact resolved zero-offset view produced by an
earlier admitted affine node in the same partition. Affine outputs retain exact Model/Compiler
logical view geometry. `CONTIGUOUS` has unchanged Shape and canonical output geometry; it must
separate an affine view from `NEG` or `ABS`.

Under `ACCELERATOR`, every occurrence is exact `ABS`, tensor `ADD`, `SUB`, `MUL`, or `DIV`, a
canonical `FLOAT32` reduction, positive static rank-two `FLOAT32` `MATMUL`, or the exact local
rank-two `PERMUTE [1,0]` that supplies a MATMUL operand. The reduction set is full, normalized
single-axis, ordered normalized multi-axis including the empty identity, and exact keep-dimensions
`SUM` or `MEAN`, plus binding-resolved `SUM_TO_SHAPE`. `ABS` has one canonical dense, zero-offset
non-view input and an equal canonical output. Binary operations have two such ordered inputs, and
their output Shape is the exact right-aligned broadcast of the input Shapes. MATMUL accepts
canonical inputs or authenticated local transposes of canonical sources, requires exact
`[M,K] @ [K,N] -> [M,N]` geometry, and produces a canonical output. Reduction and ordinary inputs
are positive-rank `1..16`; a locally produced reduction target may be scalar. Every descriptor is
fully static, checked `FLOAT32`, and has the operation-specific exact layout and `requiresGrad`
relationship. `ABS` has the same exact Model result contract in both profiles; it receives no
DAZ/FTZ, approximation, or profile-specific arithmetic freedom.

```text
capability -> Planning ownership -> Metal analysis and typed candidates
           -> optional decision authentication and private route selection
           -> exact declarations
           -> shared slot assignment -> Metal finalization
           -> PreparedExecution -> isolated Runtime run
```

Capability applies per occurrence. Preparation then authenticates the complete maximal partition
that Planning forms: strict NEG/ABS/affine/canonicalization graphs or accelerator
ABS/binary/reduction/MATMUL chains with stable fan-out, repeated and ordered inputs, internal
publications, and multiple feeds and targets. A positive-rank reduction result may compose with
other supported accelerator nodes; a scalar reduction result is a direct target only. A local
transpose accepted as a MATMUL operand must be produced inside the same partition from a canonical
source and cannot itself be a boundary target. Scalar pointwise operation families; masked,
extrema, and product reductions; every unary operation other than listed `NEG` and `ABS`, including
`EXP`, `SIGMOID`, `RELU`, and `TANH`; comparison/logical operations; strict MATMUL; vector or
batched MATMUL; profile-crossing operations; attributes outside the listed forms; and every other
type, zero extent, dynamic or unresolved layout, noncanonical graph feed, foreign view,
affine-to-NEG/ABS edge without `CONTIGUOUS`, mismatched descriptor, and multi-output form remain
fail-closed.

Within that capability domain, Metal analysis generates a typed complete candidate batch
and selects one of two private routes:

- `CUSTOM_SINGLE_NEG` for exactly one NEG occurrence, one unique feed, one unique target, and a
  checked element count in `1..UINT32_MAX`; or
- `MPSGRAPH` for every other supported strict partition and every accelerator partition.

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
| Capability truth | Public `MetalCapabilityProvider` reports only exact canonical FLOAT32 ABS under both profiles, the strict NEG/affine/`CONTIGUOUS` domain, and the accelerator tensor-binary, reduction, and positive static rank-two MATMUL domains above. |
| Native configuration and integration | Public `MetalBackendConfiguration` and `MetalBackendIntegration` belong to Metal. Metal validates configuration, opens native ownership, and rolls partial construction back before Engine can take the completed integration. |
| Backend ownership | Planning chooses `owner = metal` and groups consecutive equal owners; it never selects MPSGraph or a custom kernel. |
| Analysis | Package-private Metal code validates the complete partition, assigns stable structural value order, regenerates typed route candidates and session compatibility, authenticates any supplied decision, fixes one route, and declares that route's exact resources. |
| Opaque tuning handoff | Metal can construct the Prepare-owned marker-role handoff with no decision or one Metal decision. Shared Prepare does not inspect private candidates; the Metal preparer treats a present decision as untrusted. |
| Shared preparation | `GraphPreparation` projects facts, assigns slots, validates the result, and transfers persistent resources transactionally. It does not inspect the Metal plan or route. |
| Finalization | Metal validates exact assignments and compiles either one typed custom pipeline or one shape-specialized MPSGraph executable after slots exist. It never reselects the route. |
| Persistent resource | One typed `PreparedResource` owns the selected native handle and context child lease; `PreparedExecution` becomes its sole owner. Custom-pipeline and MPSGraph handles are never interchangeable. |
| Per-run state | Runtime borrows caller buffers and owns fresh initialized-constant buffers and output buffers. Only MPSGraph runs also own one native-address workspace. Affine targets receive full logical-byte buffers, not referenced-span-sized buffers; a scalar reduction target receives exactly four bytes. |
| Hot invocation | A cold-bound route-specific invocation retains direct references and makes one synchronous typed native call into assigned output destinations. |
| Publication | Runtime leases the already-resident output representation. Canonical outputs, local rank-zero reduction outputs, and exact authenticated dense affine targets may be downloaded as detached canonical host bytes; ordinary CPU/Metal transfer remains positive-rank and rejects logical views. |

The public Java surface contains `MetalCapabilityProvider`, `MetalBackendConfiguration`, and
`MetalBackendIntegration`. Contexts, physical storage, preparers, finalizers, schedules, executable
recipes, native handles, Objective-C objects, MPSGraph types, and the custom route remain internal.

## Integration lifecycle

### Capability, whole-partition analysis, and route selection

`MetalCapabilityProvider.supports` checks one occurrence, its numerical profile, and its exact
descriptors. In strict mode it admits a resolved zero-offset view input only for affine or
`CONTIGUOUS` occurrences because the query contains no graph closure. In either profile `ABS`
requires one canonical input and equal canonical output. In accelerator mode each binary requires
two canonical inputs and the exact broadcast output. A reduction requires one positive-rank
canonical input and its exact full, normalized-axis, keep-dimensions, empty-axis identity, or
binding-resolved sum-to-Shape output; only a reduction output may be rank zero. Accelerator
MATMUL requires two positive static rank-two FLOAT32 inputs, exact contraction and output Shapes,
canonical output, and canonical or exact transpose layout operands. Strict MATMUL is false.
Availability and hard backend requirements remain separate Planning facts.

After Planning creates one maximal Metal partition, analysis walks nodes in partition order with
explicit unavailable, canonical, and affine-view states. Every strict view input must resolve to
an earlier admitted affine producer in that exact partition; every graph feed is canonical;
`CONTIGUOUS` produces canonical state; and `NEG` and `ABS` reject affine-view state. Accelerator
ABS, binary, and reduction nodes consume only canonical values and produce canonical values;
binary nodes preserve the exact broadcast Shape, while reduction lowering preserves its typed
form, ordered axes, keep-dimensions state, sum-to-Shape target, and checked term geometry.
MATMUL produces canonical state and may consume affine state only when analysis authenticates the
exact value as a local rank-two transpose of a canonical source. Every such transpose must feed a
MATMUL and cannot be a partition boundary. A positive-rank result can feed later admitted nodes;
a scalar result must be a direct target. Stable value indexing follows first encounter. Repeated
use names the same value. Only feeds and published boundary targets receive Runtime slots; other
intermediates remain symbolic MPSGraph tensors.

Analysis receives compile-time constant sources through `PrepareContext.constants()`. A boundary
constant must be an exact `FLOAT32` splat. The run uses an `InitializedBuffer` for that feed rather
than consuming a caller position. Existing shared `GraphPreparation` tests independently enforce
the chain `CompileConstantPlan.ConstantSource -> PrepareContext.constants() -> InitializedBuffer`.

Once stable values, states, feeds, targets, checked byte geometry, and typed node records are
known, analysis creates a version-nine candidate batch and workload fingerprint. MPSGraph is
valid for every supported partition. The custom candidate exists only for one strict `NEG` node,
one feed, one target, and an element count in `1..UINT32_MAX`; ABS, affine, `CONTIGUOUS`,
accelerator binary, reduction, and MATMUL nodes never select it. Candidate order is the current
safe heuristic first and then the other valid route, so a positive budget returns a stable prefix
and budget one cannot change ordinary preparation.

The version-nine canonical workload fingerprint covers the explicit numerical-profile wire value,
schema-eight typed node kinds and attributes, ordered first/second input edges, reduction form,
ordered axes, keep-dimensions state and sum-to-Shape target, authenticated local-transpose
provenance, explicit value states, complete tensor descriptors and logical layouts, dense
represented-order geometry, target set, exact `FLOAT32` splat bits, logical-boundary roles,
exact/default policy, candidate and route-policy schemas, native node schema, and ABI version. It
encodes structural positions rather than `NodeId`, `ValueId`, or partition object identity, so
equal occurrences under the same profile and live context compare equally. Target compatibility
also contains a fresh private nonce from the exact `MetalDeviceContext`; version-eight and earlier
decisions fail closed.

Metal can construct an absent- or present-decision `BackendPartitionTuningHandoff`. Fresh analysis
always regenerates the current batch. A present decision is accepted only when the exact partition,
candidate schema, workload fingerprint, context session, and candidate membership all match;
stale or foreign values fail closed rather than reverting to the heuristic. Absence uses the
existing heuristic without cache lookup or measurement.

After authentication, the custom route declares one feed buffer, one target buffer, and no
workspace. MPSGraph declares its feed and target buffers plus one address workspace. A larger
supported singleton has only the MPSGraph candidate; analysis does not reject or split it.

### Session decision codec and limitations

The package-private version-nine Metal codec produces bounded canonical compatibility, candidate,
and checksummed decision bytes. Decode rejects wrong magic, schema, session scope, numerical
profile, malformed or truncated content, trailing or corrupt bytes, changed workload or context,
and unknown or pruned candidates. The bytes contain no native handle or executable. Version-eight
and earlier codec bytes and cross-profile decisions fail closed even when their trailing checksum
is otherwise valid.

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
validates a fixed-width version-eight typed node table and compiles one fixed-shape
`MPSGraphExecutable` for the whole partition. MATMUL compilation requires the available
reduced-precision-fast-math control to set and read back `None`. Compilation happens during
prepare finalization, never during invocation.

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
   logical element count and retain exact finalized-route authentication, scalar reduction targets
   use exactly four bytes, and other canonical targets use the ordinary canonical buffer path;
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
extent. Local canonical materialization accepts a produced scalar reduction result, but transfer
and caller ingress retain their positive-rank boundary.

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

Preparation lowers the normalized axis and binding-resolved sum-to-Shape target into typed
schema-eight records. `m`, `a`, `s`, and `c` remain positive-rank canonical values, so they may
compose inside the partition. The scalar `t` is a direct target and cannot feed another node. One
run publishes `c = [[71, 78, 85, 92], [83, 90, 97, 104]]` plus `t = 300`; local materialization of
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

The create ABI requires node schema `8` and one 160-byte discriminated record per node. Accepted
operations are `NEG=1`, `ADD=2`, `SUB=3`, `MUL=4`, `DIV=5`, `RESHAPE=6`, `EXPAND=7`,
`PERMUTE=8`, `EXPAND_DIMS=9`, `SQUEEZE=10`, `CONTIGUOUS=11`, `ABS=12`, `SUM=13`,
`MEAN=14`, and `MATMUL=15`. Attribute kinds are none, target Shape, permutation, normalized axis,
and reduction. Reduction records use typed full, single-axis, multi-axis, and sum-to-Shape forms;
their bounded payload holds ordered axes or target dimensions, and the reserved cell holds the
exact keep-dimensions bit. Java and native code require exact operation/attribute pairing, ordered
binary/MATMUL inputs, exact broadcast or contraction output, exact unary Shape equality, exact
derived reduction output and positive term count, explicit unavailable/canonical/affine-view state
transitions, authenticated local transpose provenance, `UINT32_MAX` absent-input/axis sentinels,
zero unused fields, positive input ranks and dimensions, rank-zero reduction direct targets only,
complete permutations, valid axes, topological availability, fresh outputs, unique produced
targets, exact declared Shapes, and checked full logical byte geometry. Java also rejects
profile-incompatible programs before native entry and owns typed handle liveness and pointer-region
preconditions that a raw C boundary cannot prove.
Input/output aliasing and wrong device or insufficient extent map to status `10`; grid
representability maps to status `8`; unusable threadgroup geometry and custom command failures map
to status `11`; Objective-C exceptions map to status `7`. MPSGraph compilation remains status `9`,
while custom compilation and target proof use status `12`.

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

The accelerator contract is a bounded result set, not IEEE equality or a performance claim. Each
subnormal input may be consumed either exactly or as signed zero; a finite subnormal result may be
returned exactly or as either signed zero; exact-zero `ADD`/`SUB` may use either sign; and NaN
payload/sign are unspecified while NaN classification is preserved. No tolerance-based comparison
is used. Ordinary exact finite, infinity, and required signed-zero outcomes outside those freedoms
remain exact.

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

Current validation composes:

- focused backend tests proving the complete closed profile matrices, schema-eight retained wires
  `1..14` and `MATMUL=15`, strict/profile-incompatible rejection before native entry, exact typed
  reduction and rank-two contraction Shapes, authenticated local transpose topology, rank-zero
  direct-target restrictions, malformed Java and raw-native record failure, and stable
  whole-partition lowering;
- backend-conformance coverage for exact public truth and maximal profile-qualified closure;
- a rebuilt arm64 dylib inspected for exactly thirteen exports, ABI `4`, required framework
  linkage, and absence of the old NEG-only create symbol;
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
- a CPU-free Metal-only public Engine reduction run that compiles, prepares, reuses and
  concurrently opens sessions, publishes and materializes exact scalar/vector raw bits, rejects
  strict ownership and post-close work, and has no CPU owner; and
- a CPU-free Metal-only public Engine proof for direct rank-two MATMUL, no-bias linear's visible
  right transpose, and explicitly seeded gradients for both operands, including graph/formula,
  owner, raw-bit publication, reuse, independent-session, input-preservation, close, and strict
  rejection assertions.

Compiler contract coverage checks exact forward view layouts and inverse first-order operations.
Metal execution now includes the explicitly seeded rank-two MATMUL gradient formulas above; it
does not imply broader backward or training support.

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
composition through `CONTIGUOUS`, the default strict binary/reduction fail-closed ownership
boundary, CPU ownership when CPU is explicitly registered, and CPU-free accelerator binary and
reduction execution. Separate CPU/Metal tests cover custom singleton execution, both transfer
directions, adapter use after registry lookup is poisoned, CPU tuning with Metal registered, and
early Metal tuning rejection.
These implement the construction boundary in
[ADR 0015](../design/decisions/0015-explicit-engine-backend-composition.md) and the current
owner-indexed mixed schedule in
[ADR 0016](../design/decisions/0016-cpu-metal-mixed-owner-schedule.md).

## Limitations and related documentation

Accelerator binary support is limited to tensor `FLOAT32` `ADD`, `SUB`, `MUL`, and `DIV` under the
bounded semantics above; strict binary remains unsupported. Accelerator reduction support is
limited to canonical `FLOAT32` `SUM`, `MEAN`, and binding-resolved `SUM_TO_SHAPE` under the exact
listed forms; strict, masked, extrema, product, and other reductions remain unsupported.
Accelerator MATMUL is limited to positive static same-type rank two, exact contraction/output
geometry, and canonical or authenticated local rank-two-transpose operands; strict, vector,
batched, broadcast, promoted, and arbitrary-view MATMUL remain unsupported. Exact canonical `ABS`
is the only accelerator unary operation and receives no numerical relaxation. Metal also has no
FLOAT16, BFLOAT16, FLOAT64, integer, BOOL, scalar graph ingress or transfer, zero-extent,
dynamic-shape, unresolved-layout, noncanonical graph-ingress, foreign-view, variadic, or
multi-output support. A scalar rank is admitted only for a locally produced direct reduction
target and four-byte local materialization. There is no scalar pointwise or other unary
accelerator route, comparison or logical operation, general custom-kernel framework, asynchronous
API, cross-run overlap guarantee, buffer pool, persistent constant buffer, executable
serialization, packaging, discovery, persistent route cache, current tuning integration, alias
promise, scalar-loss or implicit-seed training route, general backward route, or performance
claim. The one executable backward path is the explicitly seeded rank-two MATMUL formula described
above. Model task 0026 must define FLOAT16 semantics before any backend can advertise it.

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
- [Native ABI and build guide](../../native/metal-macos-arm64/README.md)

## Numerical profiles

Metal capability and preparation use two exact profile matrices. Both admit canonical `FLOAT32`
`ABS` with no relaxed result. `STRICT_IEEE` additionally retains the NEG/affine/`CONTIGUOUS`
baseline; `ACCELERATOR` additionally admits tensor `FLOAT32` `ADD`/`SUB`/`MUL`/`DIV` with exact
broadcasting, canonical `FLOAT32` `SUM`/`MEAN`/`SUM_TO_SHAPE`, and positive static rank-two
`FLOAT32` MATMUL with authenticated local transposes. Accelerator arithmetic uses only its
operation-specific Model-owned bounded DAZ/FTZ, reassociation/FMA, NaN, and final-result exact-zero
sign contracts. The profile is retained in partition plans and participates in route-candidate,
tuning-compatibility, decision-codec, and workload identity. Java enforces the profile boundary
before native entry. Native ABI version `4` remains unchanged; MPSGraph node schema `8` retains
wires `1..14` and appends `MATMUL=15`.
