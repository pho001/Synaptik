# Metal backend

## Outcome and supported scope

The Metal backend currently executes a whole maximal Metal-owned partition when every occurrence
is unary `NEG` with equal input and output descriptors satisfying all of these conditions:

- data type is `FLOAT32`;
- shape is fully static with rank `1..16` and every dimension is positive;
- input and output shapes are equal;
- layout is resolved dense-contiguous, non-view, and zero-offset; and
- input and output `requiresGrad` flags are equal, with either shared value accepted.

```text
capability -> Planning ownership -> Metal analysis and private route selection
           -> shared slot assignment -> Metal finalization
           -> PreparedExecution -> isolated Runtime run
```

Capability applies per occurrence. Preparation accepts the complete maximal partition that
Planning forms: chains, independent nodes, fan-out, repeated consumption, internal values,
multiple boundary values, and different valid shapes. Every other operation, type, rank, zero
extent, dynamic shape, unresolved or view layout, broadcast, and multi-output form remains
fail-closed.

Within that unchanged capability domain, Metal analysis selects one of two private routes:

- `CUSTOM_SINGLE_NEG` for exactly one NEG occurrence, one unique feed, one unique target, and a
  checked element count in `1..UINT32_MAX`; or
- `MPSGRAPH` for every other supported partition, including an otherwise matching singleton
  above `UINT32_MAX`.

This is a deterministic implementation-domain boundary. It is not tuning, capability narrowing,
CPU fallback, retry, or repartitioning, and it carries no performance-superiority claim.

## Prerequisites

Java uses JDK 26 Foreign Function and Memory (FFM) APIs. The native bridge requires an
Apple-silicon macOS host, Xcode Command Line Tools, and Foundation, Metal, and
MetalPerformanceShadersGraph. Build and ABI instructions are in the
[native Metal guide](../../native/metal-macos-arm64/README.md).

The caller supplies the dylib's absolute path. The backend does not discover, extract, package,
sign, notarize, or cache the library. Planning separately receives a Metal availability snapshot;
the capability provider performs no native loading or device discovery.

## Contracts and ownership

| Stage or resource | Owner and current behavior |
|---|---|
| Capability truth | Public `MetalCapabilityProvider` reports only the exact `FLOAT32` NEG domain above. Task 0003 does not change it. |
| Backend ownership | Planning chooses `owner = metal` and groups consecutive equal owners; it never selects MPSGraph or a custom kernel. |
| Analysis | Package-private Metal code validates the complete partition, assigns stable value order, derives feeds and targets, selects one closed private route, and declares that route's exact resources. |
| Shared preparation | `GraphPreparation` projects facts, assigns slots, validates the result, and transfers persistent resources transactionally. It does not inspect the Metal plan or route. |
| Finalization | Metal validates exact assignments and compiles either one typed custom pipeline or one shape-specialized MPSGraph executable after slots exist. It never reselects the route. |
| Persistent resource | One typed `PreparedResource` owns the selected native handle and context child lease; `PreparedExecution` becomes its sole owner. Custom-pipeline and MPSGraph handles are never interchangeable. |
| Per-run state | Runtime borrows caller buffers and owns fresh initialized-constant buffers and output buffers. Only MPSGraph runs also own one native-address workspace. |
| Hot invocation | A cold-bound route-specific invocation retains direct references and makes one synchronous typed native call into assigned output destinations. |
| Publication | Runtime leases the already-resident output representation; host observation is an explicit later download. |

The public Java surface remains only `MetalCapabilityProvider`. Contexts, physical storage,
preparers, finalizers, schedules, executable recipes, native handles, Objective-C objects,
MPSGraph types, and the custom route remain package-private.

## Integration lifecycle

### Capability, whole-partition analysis, and route selection

`MetalCapabilityProvider.supports` checks descriptors only. It cannot see whether an eligible
feed originated as a caller input or a compile-time constant, so equal eligible descriptors
receive the same answer. Availability and hard backend requirements remain separate Planning
facts.

After Planning creates one maximal Metal partition, analysis walks nodes in partition order and
indexes each input then output on first encounter. External feeds follow first-consumer order;
boundary targets follow producer-node and output-port order. Repeated use names the same indexed
value. Values internal to the MPSGraph route remain graph tensors and receive no Runtime slot.

Analysis receives compile-time constant sources through `PrepareContext.constants()`. A boundary
constant must be an exact `FLOAT32` splat. The run uses an `InitializedBuffer` for that feed rather
than consuming a caller position. Existing shared `GraphPreparation` tests independently enforce
the chain `CompileConstantPlan.ConstantSource -> PrepareContext.constants() -> InitializedBuffer`.

Once stable feeds and targets and checked byte geometry are known, analysis applies the exact
singleton/count predicate. The custom route declares one feed buffer, one target buffer, and no
workspace. MPSGraph declares its feed and target buffers plus one address workspace. A larger
supported singleton is still valid Metal work and therefore selects MPSGraph; analysis does not
reject it or split it.

### Finalization and persistent ownership

Shared Prepare assigns all declared slots before Metal finalization. The finalizer checks
declaration identity, order, geometry, slot uniqueness, exact `MetalDeviceContext` identity, and
route/workspace agreement. It then performs exactly the native create operation for the selected
route and constructs one immutable `PreparedExecutable` recipe.

For the custom route, native creation compiles the fixed branch-free `synaptik_neg_f32` Metal
Shading Language source and creates one `MTLComputePipelineState`. For MPSGraph, native creation
compiles one fixed-shape `MPSGraphExecutable` for the whole partition. Compilation happens during
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
3. each target allocates a fresh run-owned output buffer;
4. an MPSGraph run allocates one native address-array workspace, while a custom run allocates no
   workspace; and
5. cold binding validates context identity and byte extents and rejects input/output aliasing.

MPSGraph cold binding writes ordered native handles into its address workspace once and retains
direct input/output slices. Custom cold binding retains the exact input and output representations
and their typed native handles directly. An allocation or upload failure closes the current and
previously created run-owned resources through Runtime rollback. No constant buffer is prepared
once, shared between runs, or owned by a backend-global cache.

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

Native execution creates the bounded framework binding objects required by MPSGraph, binds the
ordered input and supplied output `MTLBuffer` values, and invokes the executable once with
`waitUntilCompleted = YES`. Success requires no completion error and the exact ordered usable
result count. Synaptik performs no explicit output copy and does not request hidden result
materialization; this is not a claim that MPSGraph uses no internal temporary storage.

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

### Multi-node MPSGraph route

For input `x = [1.0, -2.0]`, this supported graph:

```text
y = NEG(x)
z = NEG(y)
publish y, z
```

forms one maximal partition but has two NEG occurrences, so it remains MPSGraph. `x` is one feed,
`y` is both an internal value and boundary target, and `z` is a boundary target. A run creates
fresh supplied destinations for `y` and `z`, executes once, and publishes `y = [-1.0, 2.0]` and
`z = [1.0, -2.0]`. A second run reuses the executable but receives different outputs and address
workspace.

These are explanatory scenarios over current backend-local contracts, not public Engine samples.
There is no public Metal composition or standard `compute` entry yet.

## ABI and failures

Private ABI version `3` exports exactly thirteen symbols: the version/context/buffer foundation,
the three MPSGraph executable operations, and the three typed custom-pipeline operations. Statuses
`0..11` retain their earlier meanings; status `12` is
`KERNEL_COMPILATION_FAILED`. Unknown integers fail closed with the raw value retained.

Java and native code validate every safely inspectable count, rank, dimension, index, topology,
equal NEG shape, and byte geometry. Java additionally owns typed handle liveness and pointer-region
preconditions that a raw C boundary cannot prove. Input/output aliasing and wrong device or
insufficient buffer extent map to status `10`; grid representability maps to status `8`;
unusable threadgroup geometry and custom command failures map to status `11`; Objective-C
exceptions map to status `7`. MPSGraph compilation remains status `9`, while custom compilation
and target proof use status `12`. No status string or framework object crosses the ABI.

Deterministic fake/native-seam tests are the accepted status and error matrix. Source-contract
assertions cover the native fail-closed branches and mappings. Real-device tests validate
successful native behavior rather than trying to induce undocumented framework failures.

## Evidence composition

The stabilized task-0003 evidence is:

- the focused `MetalFoundationTest` run reported 14 total, 13 passed, and one expected native
  opt-in skip;
- the final ordinary Metal suite reported 45 total, 42 passed, three expected native opt-in
  skips, and no failures or errors;
- an earlier focused `MetalNegPreparedExecutionTest` run reported 28 total, 26 passed, and two
  expected native opt-in skips before the final test-only/native mapping refinement; the final
  full suite supersedes that focused run;
- the native dylib build passed as Mach-O arm64, with exactly thirteen `synaptik_metal_*` exports
  and expected Foundation, Metal, and MetalPerformanceShadersGraph linkage;
- both `nativeFoundationRoundTrip` and `nativePreparedNegRoundTripAndReuse` passed against the
  final rebuilt dylib on the real device;
- generated Javadoc and custom hot-shape `javap` inspection passed, and source assertions cover
  the specified native fail-closed branches and status mappings; and
- code-review blockers were fixed and independent re-review reported no blocking findings.

`nativePreparedNegRoundTripAndReuse` proves repeated custom caller and splat execution, fresh
direct outputs, and the retained multi-node MPSGraph path. Backend-local typed tests separately
prove declaration timing, assignment validation, malformed output-cell cleanup, typed handle
separation, rollback, close/run lifecycle, and concurrent admitted-run isolation. Existing shared
`GraphPreparation` contract tests prove that a `ConstantSource` reaches the owning partition's
`PrepareContext.constants()` and must use `InitializedBuffer`.

The public `GraphCompilationPort` intentionally supplies no explicit positive-rank forward-
constant ingress. Therefore the repository does not claim one public-port positive-rank-splat
test. Combining the real caller-input route, backend-local typed splat route, and existing shared
constant contracts is the truthful evidence.

## Registration and composition

The provider is supplied explicitly to Planning; there is no `ServiceLoader`, registry, or
runtime service locator. Current standard Engine composition remains CPU-only, and its supported
CPU adapter rejects mixed or multiple partitions. Metal therefore composes the public shared
Compiler, Prepare, and Runtime contracts inside backend tests without publishing a Metal Engine
adapter. It makes no standard Engine, mixed-owner, cross-region transfer, or fallback claim.

The module has a test-only Compiler dependency so its typed integration test can create public
`CompileArtifacts`; Metal production has no Compiler or Engine dependency. No backend-conformance,
integration, architecture-test, or Gradle change is required because capability, public
composition, dependencies, and module boundaries are unchanged.

## Limitations and related documentation

The current routes have no FLOAT16, BFLOAT16, FLOAT64, integer, BOOL, scalar-rank, zero-extent,
dynamic-shape, strided/view/offset, broadcast, or multi-output support. There is no general custom
kernel framework, asynchronous API, cross-run overlap guarantee, buffer pool, persistent constant
buffer, serialization, packaging, discovery, route cache, tuning, or performance claim. Model
task 0026 must define FLOAT16 semantics before any backend can advertise it. Metal 0004 remains
Draft for typed route candidates and cache compatibility; it has no detailed task specification.

Related documentation:

- [Architecture contract](../../ARCHITECTURE.md#metal-backend)
- [Runtime / Prepare / Backend boundary](../architecture/runtime-prepare-backend-boundary.md)
- [Prepared-resource lifecycle ADR](../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)
- [Metal strategy note](../design/notes/metal-backend-strategy.md)
- [Metal task 0003](../planning/backends/metal/tasks/0003-single-neg-custom-metal-kernel-route.md)
- [Native ABI and build guide](../../native/metal-macos-arm64/README.md)
