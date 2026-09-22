# Metal backend

## Outcome and supported scope

This guide explains Synaptik's first executable Metal route and the boundaries a backend
contributor must preserve. The route executes a whole maximal Metal-owned partition when every
occurrence is unary `NEG` with equal input/output descriptors satisfying all of these conditions:

- data type is `FLOAT32`;
- shape is fully static with rank `1..16` and every dimension is positive;
- input and output shapes are equal;
- layout is resolved dense-contiguous, non-view, and zero-offset; and
- input and output `requiresGrad` flags are equal, with either shared value accepted.

```text
capability -> Planning ownership -> Metal analysis -> shared slot assignment
           -> Metal finalization -> PreparedExecution -> isolated Runtime run
```

Capability applies per occurrence, so preparation accepts the complete maximal partition that
Planning forms: chains, independent nodes, fan-out, repeated consumption, internal values,
multiple boundary values, and different valid shapes all lower into one executable. Every other
operation, type, rank, zero extent, dynamic shape, unresolved or view layout, broadcast, and
multi-output form remains fail-closed.

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
| Capability truth | Public `MetalCapabilityProvider` reports only the exact `FLOAT32` NEG domain. |
| Backend ownership | Planning chooses `owner = metal` and groups consecutive equal owners; it never selects MPSGraph. |
| Analysis | Package-private Metal code validates the entire partition, assigns stable value order, classifies caller and constant feeds, and declares boundary buffers plus one address workspace. |
| Shared preparation | `GraphPreparation` projects facts, assigns slots, validates the result, and transfers persistent resources transactionally. It does not inspect the Metal plan. |
| Finalization | Metal validates exact assignments and compiles one shape-specialized MPSGraph executable after slots exist. |
| Persistent executable | One `PreparedResource` owns the native executable and context lease; the resulting `PreparedExecution` becomes its sole owner. |
| Per-run state | Runtime borrows caller buffers and owns fresh initialized-constant buffers, output buffers, and one native-address workspace. |
| Hot invocation | A cold-bound invocation retains direct references and makes one synchronous native downcall into supplied output destinations. |
| Publication | Runtime leases the already-resident output representation; host observation is an explicit later download. |

The public Java surface remains only `MetalCapabilityProvider`. Contexts, physical storage,
preparers, finalizers, schedules, executables, native handles, Objective-C objects, and MPSGraph
types remain package-private.

## Integration lifecycle

### Capability and whole-partition analysis

`MetalCapabilityProvider.supports` checks descriptors only. It cannot see whether an eligible
feed originated as a caller input or a compile-time constant, so equal eligible descriptors
receive the same answer. Availability and hard backend requirements remain separate Planning
facts.

After Planning creates one maximal Metal partition, analysis walks nodes in partition order and
indexes each input then output on first encounter. External feeds follow first-consumer order;
boundary targets follow producer-node and output-port order. Internal values remain MPSGraph
tensors and receive no Runtime slot. Repeated use names the same indexed value.

Analysis receives compile-time constant sources through `PrepareContext.constants()`. A boundary
constant must be an exact `FLOAT32` splat. The route declares it like any other feed, but the run
uses an `InitializedBuffer` instead of consuming a caller position. Existing shared
`GraphPreparation` tests independently enforce the chain
`CompileConstantPlan.ConstantSource -> PrepareContext.constants() -> InitializedBuffer`.

### Finalization and persistent ownership

Shared Prepare assigns all declared buffer and workspace slots before Metal finalization. The
finalizer checks declaration identity, order, geometry, slot uniqueness, and exact
`MetalDeviceContext` identity. It then calls native creation once and constructs one immutable
`PreparedExecutable` recipe.

Native creation compiles a fixed-shape `MPSGraphExecutable` for the whole partition. The returned
resource owns that executable and a context child lease. A provisional lease prevents concurrent
context close from invalidating native creation. Failed creation or wrapper construction releases
the executable when present, then the lease; failed finalization rolls the resource back locally.
After finalizer return, shared Prepare owns rollback until a fully validated `PreparedExecution`
accepts the resource exactly once.

`PreparedExecution.close()` rejects new runs without waiting. A previously admitted synchronous
run may finish, after which the last run lease releases persistent resources. Cleanup is
idempotent, reverse-order, and attempt-all; the first failure remains primary and later distinct
failures are suppressed.

### Cold run setup and binding

Each run receives isolated mutable state:

1. caller input positions borrow caller-owned Metal buffers;
2. each constant feed allocates a fresh run-owned Metal buffer and uploads the exact raw
   `FLOAT32` splat bits once;
3. each target allocates a fresh run-owned output buffer;
4. one run-owned workspace allocates native address-array storage; and
5. cold binding validates context identity and byte extents, rejects input/output aliasing, and
   writes all ordered native handles into that workspace once.

An allocation or upload failure closes the current and previously created run-owned resources
through Runtime rollback. No constant buffer is prepared once, shared between runs, or owned by a
backend-global cache.

### Hot execution and supplied destinations

The bound invocation holds direct slices of the address workspace plus the persistent executable.
Its hot method makes one native call. It performs no Java allocation, address marshalling, slot or
map lookup, representation cast, graph traversal, operation dispatch, route choice, reflection,
string dispatch, host copy, retry, or fallback.

Native execution creates the bounded framework binding objects required by MPSGraph, binds the
ordered input and caller-supplied output `MTLBuffer` values, and invokes the executable once with
`waitUntilCompleted = YES`. Success requires no completion error and the exact ordered usable
result count. Synaptik performs no explicit output copy and does not request hidden result
materialization; this is not a claim that MPSGraph uses no internal temporary storage.

## Example: two NEG occurrences

For inputs `x = [1.0, -2.0]`, this supported graph:

```text
y = NEG(x)
z = NEG(y)
publish y, z
```

forms one maximal Metal partition and one reusable executable. `x` is one feed, `y` is both an
internal value and boundary target, and `z` is a boundary target. A run uploads `x` into a
caller-owned Metal buffer before Runtime, creates fresh supplied destinations for `y` and `z`,
executes once, and publishes `y = [-1.0, 2.0]` and `z = [1.0, -2.0]`. A second run reuses the
executable but receives different output buffers and workspace.

This is an explanatory scenario over current contracts, not a public Engine sample. There is no
public Metal composition or standard `compute` entry yet.

## ABI and failures

ABI version 2 exports ten symbols: the original version/context/buffer functions plus executable
create, release, and run. Statuses `0..7` retain their foundation meanings; `8` is unsupported
shape, `9` graph compilation failure, `10` incompatible resource, and `11` execution failure.
Unknown integers fail closed with the exact raw value retained.

Java and native code both validate safely inspectable counts, ranks, dimensions, indices,
topology, equal NEG shapes, and byte geometry. Java additionally owns typed handle liveness and
pointer-region preconditions that a raw C boundary cannot prove. Input/output aliasing and wrong
device or insufficient buffer extent map to status `10`; completion errors and malformed results
map to `11`; Objective-C exceptions map to `7`. No status string or framework object crosses the
ABI.

Deterministic fake/native-seam tests are the accepted status and error matrix. The real-device
test validates successful native behavior rather than trying to induce undocumented MPSGraph
failures.

## Evidence composition

No single test overstates the available public surface:

- `nativePreparedNegRoundTripAndReuse` uses public `GraphCompilationPort`, shared
  `GraphPreparation`, Runtime, and real MPSGraph with ordinary caller inputs. It proves one
  maximal partition, two-run executable reuse, fresh outputs, supplied destinations, publication,
  host observation, and finite/infinite/NaN/signed-zero NEG behavior.
- Backend-local typed tests construct `PrepareContext` directly to prove explicit positive-rank
  `FLOAT32` splats flow through analysis, finalization, schedule assembly, `InitializedBuffer`,
  and Runtime with exact raw bits, per-run allocation/upload, rollback, and concurrent-run
  isolation.
- Existing shared `GraphPreparation` contract tests prove that a `ConstantSource` reaches the
  owning partition's `PrepareContext.constants()` and must use `InitializedBuffer`, while a
  non-constant must not.

The public `GraphCompilationPort` intentionally supplies no explicit positive-rank forward-
constant ingress. Therefore the repository does not claim one public-port positive-rank-splat
test. Combining the real caller-input route, backend-local typed splat route, and existing shared
constant contracts is the truthful end-to-end evidence.

## Registration and composition

The provider is supplied explicitly to Planning; there is no `ServiceLoader`, registry, or
runtime service locator. Current standard Engine composition remains CPU-only, and its supported
CPU adapter rejects mixed or multiple partitions. Metal 0002 therefore composes the public shared
Compiler, Prepare, and Runtime contracts inside backend tests without publishing a Metal Engine
adapter. It makes no standard Engine, mixed-owner, cross-region transfer, or fallback claim.

## Conformance and validation

The stabilized implementation evidence is:

- the final ordinary Metal suite passed 35 tests, with three opt-in tests skipped because the
  required environment variables were absent;
- the focused prepared-execution class passed 20 tests, with its two opt-in real-device tests
  skipped in the ordinary sandboxed run;
- the native build succeeded; symbol inspection found exactly ten `synaptik_metal_*` exports,
  and link inspection found Foundation, Metal, and MetalPerformanceShadersGraph;
- the exact opt-in command in the native guide passed both
  `nativeFoundationRoundTrip` and `nativePreparedNegRoundTripAndReuse` outside the sandbox;
- focused backend-conformance passed two tests and focused dependency architecture validation
  passed one test; and
- source plus `javap` inspection confirmed the package-private surface and direct one-downcall hot
  path across the exact 20 implementation paths present before this documentation pass.

The module has a test-only Compiler dependency so its typed integration test can create public
`CompileArtifacts`; Metal production has no Compiler or Engine dependency. Backend conformance
has a test-only Metal dependency. No generic Engine integration test was added because no
supported public Metal Engine composition exists.

## Limitations and related documentation

The current route has no FLOAT16, BFLOAT16, FLOAT64, integer, BOOL, scalar-rank, zero-extent,
dynamic-shape, strided/view/offset, broadcast, or multi-output support. It adds no custom kernel,
asynchronous API, cross-run overlap guarantee, buffer pool, persistent constant buffer,
serialization, packaging, discovery, tuning, or performance-superiority claim. Model task 0026
must define FLOAT16 semantics before any backend can advertise it.

Related documentation:

- [Architecture contract](../../ARCHITECTURE.md#metal-backend)
- [Runtime / Prepare / Backend boundary](../architecture/runtime-prepare-backend-boundary.md)
- [Prepared-resource lifecycle ADR](../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)
- [Metal strategy note](../design/notes/metal-backend-strategy.md)
- [Metal task 0002](../planning/backends/metal/tasks/0002-mpsgraph-prepared-execution-route.md)
- [Native ABI and build guide](../../native/metal-macos-arm64/README.md)
