# Task 0002: MPSGraph Prepared Execution Route

## Status

Complete

## Goal

Deliver the first truthful executable Metal capability: every non-empty maximal Metal-owned
partition composed solely of individually supported fully static, positive-shape, resolved
dense-contiguous `FLOAT32` unary `NEG` occurrences.

Compiler 0006B7 is a mandatory prerequisite. Occurrence capability queries contain descriptors
but no compile-time source provenance, so this route must accept an otherwise eligible NEG whose
feed is an explicit compile-time splat. Compiler 0006B7 first closes only the fully static
still-unresolved NEG outputs and direct splat inputs needed for truthful capability admission.

The route must lower and compile one shape-specialized Metal Performance Shaders Graph
(`MPSGraph`) executable during backend finalization, transfer that persistent native resource
transactionally through Prepare, bind per-run Metal buffers without exposing Objective-C values
to Java callers, execute the reusable native executable synchronously, and write the result
directly into the assigned Metal output buffer.

```text
Planning: eligible NEG occurrence -> owner = metal
Prepare analysis: validate the complete maximal NEG partition -> declare boundary buffers
Prepare finalization: compile reusable MPSGraphExecutable -> transfer PreparedResource
Runtime cold setup: caller-supplied inputs + fresh initialized splat inputs + fresh outputs
Runtime cold bind: checked representations + one run-owned native-address workspace
Runtime execute: one synchronous region submission -> assigned output buffers
Runtime publication: lease the already-resident output; copy only for host observation
```

This task establishes a backend-local prepared execution route through the current shared
Planning, Prepare, and Runtime contracts. It does not add a public Engine Metal composition or
claim mixed-owner execution.

## Pre-implementation audit

### Repository evidence

- Metal 0001 currently supplies a fail-closed provider, one default-device/command-queue native
  context, fresh shared-storage `MTLBuffer` wrappers, checked host upload/download, context child
  leases, and exact once-only cleanup. It submits no command and uses ABI version `1` with seven
  symbols.
- Planning asks capability per operation occurrence and retains only `BackendId`; it cannot and
  must not select an MPSGraph route.
- `PrepareContext` supplies the exact immutable partition-local DAG, fully static value
  descriptors, logical requirements, constants, and opaque backend inputs. The candidate needs
  no new shared semantic projection.
- `BackendPartitionAnalysis` can declare the candidate input and output buffers before slot
  assignment. `BackendPartitionFinalization` then supplies their exact assigned dense plan
  positions.
- `BackendPartitionFinalizationResult` can atomically return the prepared executable and an
  acquisition-ordered `PreparedResource`. Shared Prepare rolls that resource back on every later
  failure and transfers it once to Runtime.
- `PreparedExecutable` already performs one checked cold representation-compatibility pass and
  produces a backend-owned `BoundInvocation` with direct typed references. Its hot invocation
  may call the native executable without graph, operation, slot, map, or route lookup.
- `PreparedRepresentationPlan` can borrow caller-supplied Metal input representations and create
  fresh run-owned Metal output representations. Its `InitializedBuffer` variant can create one
  fresh run-owned representation whose logical value is already valid. This is the exact existing
  contract needed to allocate and upload each eligible FLOAT32 splat once during cold run-state
  creation for each run.
- `BoundInvocation` has no close hook, and `PreparedExecutable.bind(...)` explicitly forbids an
  independently closeable auxiliary binding resource. Native pointer-array storage therefore
  cannot be allocated and retained directly by the invocation. One declared per-run workspace
  can own that storage, receive exact buffer addresses once during cold binding, and close with
  `RunState`; hot execution can then perform one downcall without Java allocation or marshalling.
- Runtime 0016, Prepare 0006, and Engine 0010 complete the inward resource owner, transactional
  handoff, close/run lease, and outward prepared-handle lifecycle. Metal does not need to change
  any of them.
- The current Engine composition remains CPU-only and rejects mixed/multi-partition artifacts.
  This task therefore proves the route through backend-local composition of the public shared
  Prepare and Runtime contracts and makes no public Engine or mixed-backend claim.

### Installed SDK evidence

The planning audit inspected the installed macOS SDK at
`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk` on an arm64 macOS 26.6.2 host. `xcrun`
reported SDK version 27.0 and Apple clang 21.0.0.

The installed framework headers establish the required route:

- `MPSGraph.h` exposes
  `compileWithDevice:feeds:targetTensors:targetOperations:compilationDescriptor:` and states that
  compilation blocks and returns a graph executable specialized by the supplied shaped feeds.
- `MPSGraphMemoryOps.h` exposes a typed fixed-shape placeholder.
- `MPSGraphArithmeticOps.h` exposes `negativeWithTensor:name:`.
- `MPSGraphTensorData.h` exposes
  `initWithMTLBuffer:shape:dataType:`; the buffer's device supplies the graph device.
- `MPSGraphExecutable.h` exposes the synchronous
  `runWithMTLCommandQueue:inputsArray:resultsArray:executionDescriptor:` call. Inputs follow the
  executable feed order, and caller-supplied result tensor data follows target order.
- `MPSGraphExecutableExecutionDescriptor` exposes `waitUntilCompleted` and a completion handler
  whose nullable `NSError` reports execution failure.
- `MPSGraphExecutable` is the compiled reusable graph object and reports ordered feed and target
  tensors. It is available from macOS 12.0.
- `MPSGraphDevice.h` exposes construction from the context's exact `MTLDevice`.
- `MPSGraphOptionsNone` avoids the optional end-of-execution result synchronization blit. The
  executable call itself remains synchronous, so the route waits once at its region boundary.

A temporary Objective-C probe containing a FLOAT32 fixed-shape placeholder, NEG, compilation,
`MTLBuffer` tensor-data binding, a supplied result array, completion-error capture, and two calls
against one executable compiled successfully against those installed headers and frameworks. In
the original sandbox it observed `NO_DEVICE`. The coordinator then executed that exact
`/tmp/synaptik_mpsgraph_audit` binary outside the sandbox; it exited `0` and printed exactly:

```text
PASS executable_reused=2 direct_output_buffer=1 completion_errors=0 results=1
```

This is preimplementation feasibility evidence only. It supports two sequential uses of one
executable, observation of the supplied destination buffer, one returned result, and the values
checked by that probe's source. It does not prove the repository ABI, Java/Prepare/Runtime
integration, cleanup, concurrency, or failure delivery, and `completion_errors=0` does not test
completion-error mapping. Repository fake/native-seam tests must deterministically cover every
failure status; the real-device repository test covers successful route execution, reuse, and
assigned output without requiring an artificial MPSGraph failure.

### Audit decision

The repository and SDK evidence closes the Metal architecture and route-design questions. The
separate Compiler 0006B7 descriptor prerequisite is Complete. Capability is
queried per occurrence, while `MaximalSameOwnerPartitioning` joins consecutive equal owners.
Truthful support therefore requires one route for every maximal partition whose occurrences all
meet the exact positive-static-shape contiguous `FLOAT32` unary `NEG` domain:

- one or more boundary inputs prove explicit caller upload or exact run-owned splat
  initialization without adding broadcasting;
- one or more boundary outputs prove assignment, direct caller-supplied result binding, Runtime
  validity, result publication, and host observation;
- finalization-time compilation proves cold specialization and transactional persistent-resource
  ownership;
- two executions of one prepared recipe prove executable reuse and isolated per-run buffers;
- chains, independent nodes, fan-out, repeated consumption, internal values, multiple boundary
  inputs/outputs, and differing positive static shapes prove that occurrence capability remains
  closed under the actual partitioner without inventing a partition split;
- one executable for the whole maximal partition makes the only synchronization point the
  compiled region boundary; and
- the exact capability row can change from fail-closed to true only with the complete route.

`ADD` would add broadcasting and two-input binding without proving a different lifecycle.
Metadata-only or zero-element work would not prove native execution. Another operation, another
data type, or an unresolved layout would enlarge lowering and capability truth without being
necessary for the first route.

## Scope

- Evolve the same native library from ABI version `1` to version `2`, preserving the seven
  foundation functions and adding exactly the three MPSGraph functions specified below.
- Link the native library with `MetalPerformanceShadersGraph` in addition to Foundation and
  Metal.
- Extend the native context box only as needed to let compilation use its exact device and
  execution use its exact command queue.
- Add one opaque persistent native executable box retaining the compiled
  `MPSGraphExecutable`, ordered feed/target descriptors and byte extents, and exact context
  prerequisite.
- Add one package-private Java `PreparedResource` owner for the native executable. It holds a
  provisionally acquired context child lease, adopts it only after native creation succeeds,
  closes idempotently, consumes the native handle once, and never exposes the handle.
- Add backend-private immutable analysis inputs and preparation plan for the complete maximal
  partition route. The preparation plan retains the exact `MetalDeviceContext` identity supplied
  to analysis; finalization and schedule assembly reject any different context.
- Analyze every non-empty partition whose nodes are all unary `NEG` occurrences in exact
  partition order and whose individual input/output descriptors satisfy the capability domain.
  Accept every topology admitted by the compiled graph: chains, independent nodes, fan-out,
  repeated consumption, multiple boundary inputs and outputs, differing positive static shapes
  where each unary occurrence is semantically valid, and values internal to the partition.
- Derive unique boundary inputs in first external-consumer occurrence order (partition node order,
  then input-port order) and unique boundary outputs in producer node/output-port order when the
  logical requirement records a graph output or an out-of-partition consumer. Declare those
  ordered boundary inputs followed by ordered boundary outputs as shared buffers, each with exact
  checked `elementCount * Float.BYTES` geometry and four-byte alignment. Internal values remain
  native graph tensors and receive no shared buffer declaration or Runtime slot. Retain exact
  FLOAT32 splat facts for boundary inputs present in `PrepareContext.constants()` and reject a
  constant on any non-feed value or any mismatched type/source fact.
- Declare exactly one backend-local workspace for the fixed feed-plus-target native address
  arrays, with checked exact pointer-count byte geometry and address alignment. Its per-run Metal
  representation owns closeable native storage, is filled once during cold binding, and is
  released by `RunState`; it is not element scratch, a graph value, or persistent prepared state.
- During finalization, map those exact declarations to assigned plan positions, compile the
  whole partition into one native executable once, construct the immutable prepared executable,
  and return the native executable resource in acquisition order.
- Roll back the native executable locally on every failure before a successful finalizer return,
  preserving the exact primary and suppression rules of Prepare 0006.
- Add one package-private schedule assembler for the exact sole-partition all-Metal graph route.
  It creates a representation plan with each bindable graph input borrowed in boundary-input
  order, each compile-time splat feed represented by `InitializedBuffer` whose backend creator
  allocates a fresh run-owned Metal buffer and uploads the exact repeated FLOAT32 bits once during
  that run's cold state creation, each boundary output freshly run-owned in boundary-output order,
  and the exact run-owned address workspace, then one execution step and ordered publication
  steps. Mixed-owner sourcing and generic constant materialization remain out of scope.
- Add one `PreparedExecutable` whose selections are all boundary inputs as `READ_ONLY` followed
  by all boundary outputs as `WRITE_ONLY`. Cold binding accepts only compatible live Metal buffer
  representations with sufficient logical bytes and the same device context.
- Add one bound invocation retaining direct references to the native executable resource and the
  run-owned address workspace after cold binding has written the exact ordered input/output
  native handles once. Its hot method performs exactly one native execution downcall with no Java
  allocation, address-array marshalling, map, slot, or representation lookup.
- Change `MetalCapabilityProvider.supports(...)` to return `true` only for the exact occurrence
  domain above and `false` for every other current query.
- Add deterministic fake-native Java tests for topology/order analysis, assignments, native
  descriptor marshalling, every failure-status mapping, finalizer rollback, provisional-lease
  adoption/rollback, resource ownership, binding, alias rejection, capability/partition closure,
  execution status, close/run races delegated to Runtime, and absence of hot-path lookup.
- Add an opt-in real-native test that prepares a multi-node/multi-boundary route including both a
  caller input and an eligible compile-time splat feed through `GraphPreparation`, uploads each
  host caller value once into its caller-owned Metal input buffer, proves each run allocates and
  uploads its own exact initialized splat buffer during cold setup, runs the same prepared
  execution twice with isolated run state, verifies direct writes in every
  assigned output buffer, downloads only for observation, and closes results, inputs, prepared
  execution, and context in ownership order.
- Add `testImplementation(project(":modules:compiler"))` to `backends/metal/build.gradle.kts` so
  the backend-local typed integration test can construct the public `CompileArtifacts` consumed
  by `GraphPreparation`. This is test-only and does not alter Metal production dependency
  direction.
- Add the narrow backend-conformance coverage required for new backend behavior: public
  capability truth plus maximal-partition closure for adjacent eligible NEG occurrences. Add only
  a test dependency from `testing/backend-conformance` to `backends/metal`, and update the
  existing exact dependency architecture test. Do not add an Engine integration test because the
  supported Engine composition remains CPU-only.
- Finalize affected Javadocs, the Metal backend guide, native README, glossary impact, and
  planning evidence in a separate clean documentation-focused context after executable behavior
  stabilizes.

## Exact capability domain

`MetalCapabilityProvider.supports(query)` returns `true` if and only if all conditions hold:

1. `query.operation().kind()` is `UnaryElementwiseKind.NEG` and its attributes are exactly
   `NoOperationAttrs.INSTANCE`.
2. There is exactly one input descriptor and one output descriptor.
3. Both data types are `FLOAT32`.
4. Both shapes are fully static, equal, have rank in the inclusive range `1..16`, every dimension
   is strictly positive, and checked element/byte arithmetic succeeds.
5. Both layouts are present and exactly resolved dense-contiguous, non-view, zero-offset layouts
   for their shapes.
6. Input and output `requiresGrad` flags are equal, matching the current Model NEG contract; the
   route accepts either flag value because this task executes only the forward occurrence.

Null retains the existing `NullPointerException("query")` contract. The provider performs no
native loading, device discovery, compilation, allocation, caching, or route selection.
Availability and hard requirements remain separate Planning inputs.

Because capability is occurrence-scoped, any consecutive eligible NEG occurrences may be
assigned to one maximal Metal partition. Analysis must accept and lower every such non-empty
partition as one reusable executable; it must not reject a supported topology, split the
partition, inspect graph context in the provider, or weaken Planning's maximal same-owner rule.
Preparation rejection is limited to a fact excluded by the occurrence capability domain or a
malformed/inconsistent projected DAG, descriptor, logical requirement, declaration, assignment,
or native handoff. Broader operations, types, layouts, and shapes remain fail-closed.

## Stable whole-partition lowering and binding

Analysis assigns native value indices by walking nodes in exact `PartitionDag.nodes()` order,
visiting each node's input ports and then output ports, and retaining the first occurrence of each
`ValueId`. Each node contributes one input-value index and one output-value index in that same
node order. The descriptor for each indexed value comes from the exact projected `GraphValue`;
each unary occurrence requires equal input/output shapes, while unrelated values may have
different positive static shapes.

Feeds are the unique values with no local producer, ordered by their first external consumer
occurrence. Targets are the unique locally produced values whose logical requirement records a
graph output or consumer partition outside this partition, ordered by producer node then output
port. A fan-out or repeated input therefore names one indexed tensor repeatedly rather than
creating duplicate placeholders. A locally produced value consumed only inside the partition is
lowered and retained in the MPSGraph value table but has no feed, target, shared declaration, or
Runtime buffer.

Each feed is classified from the exact projected facts as either caller-bindable or an explicit
compile-time splat. A splat feed must have an exact FLOAT32 `ScalarValue`; its final descriptor
must independently satisfy the same capability domain as every other feed. Source provenance does
not alter occurrence capability. The schedule supplies no caller entry for a splat: its
`InitializedBuffer` creator allocates a fresh context-local buffer, fills every logical element
with the scalar's exact raw FLOAT32 bits, and returns it already logically valid during cold
run-state creation. Creation/upload failure closes the fresh buffer under existing Runtime
rollback. No prepared/shared constant buffer survives across runs.

The shared buffer declarations, assigned plan positions, `PreparedExecutable` selections, native
run arrays, and required-byte arrays preserve one alignment:

```text
unique feeds in feed order       -> READ_ONLY assigned buffers -> native inputsArray
unique targets in target order   -> WRITE_ONLY assigned buffers -> supplied resultsArray
one address workspace            -> cold-marshalled input/output pointer arrays
```

Finalization rejects a missing, repeated, foreign, reordered, or geometry-inconsistent
declaration/assignment instead of reconstructing order. The schedule assembler resolves each
bindable graph input to its feed buffer and each published result to its target buffer by
`ValueId`; publication steps retain `PublicationPlan` order, so repeated publication aliases may
lease the same one target buffer without adding another native result. These rules make the
placeholder/feed and target/result orders stable across analysis, compilation, cold binding, and
every run.

The preparation plan retains the exact `MetalDeviceContext` object from analysis. The finalizer
and schedule assembler both require identity equality with their supplied context before native
work or recipe construction. Equality by backend/device label is insufficient.

## Exact native ABI version 2

The existing version symbol keeps its name and returns unsigned value `2`. The six existing
context/buffer functions keep their exact version-1 signatures and semantics. Version `2`
exports exactly ten `synaptik_metal_*` symbols by adding:

```c
int32_t synaptik_metal_mpsgraph_neg_executable_create(
        void *context,
        uint32_t value_count,
        const uint32_t *value_ranks,
        const uint64_t *value_dimensions,
        uint32_t node_count,
        const uint32_t *node_input_value_indices,
        const uint32_t *node_output_value_indices,
        uint32_t feed_count,
        const uint32_t *feed_value_indices,
        uint32_t target_count,
        const uint32_t *target_value_indices,
        void **out_executable);

int32_t synaptik_metal_mpsgraph_executable_release(void *executable);

int32_t synaptik_metal_mpsgraph_executable_run(
        void *executable,
        uint32_t input_count,
        void *const *input_buffers,
        uint32_t output_count,
        void *const *output_buffers);
```

Java maps `uint32_t` and `int32_t` to `JAVA_INT`, each `uint64_t` element to `JAVA_LONG`, and
every pointer position to `ADDRESS`/`MemorySegment`. `value_dimensions` is a row-major
`value_count * 16` table; for value `v`, only the first `value_ranks[v]` entries are dimensions
and the remaining entries are zero. Node input/output arrays are parallel unary-node arrays in
partition order. Feed and target indices use the stable boundary orders defined above. Java
rejects every cardinality relationship, rank, used dimension, zero-padded unused dimension cell,
node/feed/target index, producer order, duplicate role, input/output Shape mismatch for each NEG,
and checked element/byte/address-workspace geometry before the create downcall. Native
independently validates nulls, every scalar/count/index, unused dimension cell, producer order,
matching NEG input/output Shapes, checked products and byte extents, and every other safely
inspectable structure. Java performs the corresponding exact input/output counts, handle
cardinality, alias, context identity, logical extent, and pointer-workspace geometry validation
before the run downcall; native independently validates all matching facts safely inspectable
from valid live boxes and readable arrays.

The ABI remains an opaque-pointer boundary, not a general memory-safety or handle-introspection
service. Java guarantees that context, executable, and buffer handles are live handles of the
correct native box type, are consumed once, and that every array/dimensions storage region is
readable for the count or rank implied by the scalar arguments. Native cannot independently prove
that a non-null raw pointer is live, type-correct, or sufficiently sized before dereferencing it;
stale, wrong-type, or undersized pointers are outside the ABI contract. Native still validates
nulls, scalar geometry, the contents/device/logical extents of valid live boxes, and every other
fact that can be inspected safely after Java satisfies that boundary.

Version `2` retains status values `0..7` unchanged and appends:

| Value | C name | Meaning |
|---:|---|---|
| `8` | `SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE` | Rank, a dimension, or checked FLOAT32 byte geometry is outside this route. |
| `9` | `SYNAPTIK_METAL_STATUS_GRAPH_COMPILATION_FAILED` | MPSGraph construction or compilation produced no usable executable. |
| `10` | `SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE` | A buffer has the wrong device, insufficient logical byte extent, or aliases an input/output handle. |
| `11` | `SYNAPTIK_METAL_STATUS_EXECUTION_FAILED` | Synchronous execution failed, reported an `NSError`, or did not return the exact ordered usable result count. |

Unknown status integers remain unknown and fail closed with the raw integer retained. No status
string, Objective-C object, `NSError`, block, callback, graph, tensor, queue, device, raw address,
or framework type crosses the C ABI.

Status classification is exact:

- null output cells, handles, required array pointers, zero counts, count mismatches, invalid or
  duplicate value indices, invalid producer order/topology, nonzero unused dimension cells, and a
  null release handle map to `INVALID_ARGUMENT`;
- rank outside `1..16`, nonpositive used dimensions, or overflowing element/byte geometry maps to
  `UNSUPPORTED_SHAPE`;
- nil graph/placeholder/NEG/shaped-type/device/compilation-descriptor/executable objects or a
  non-bijective executable feed/target report maps to `GRAPH_COMPILATION_FAILED`;
- executable-box allocation maps to `ALLOCATION_FAILED`;
- a valid live buffer box with wrong device, insufficient logical extent, or identical input and
  output native handle maps to `INCOMPATIBLE_RESOURCE`;
- nil run tensor-data/array/execution-descriptor/result objects, completion `NSError`, wrong result
  count, or nil result entry maps to `EXECUTION_FAILED`; and
- an Objective-C exception in create, run, or release maps to `INTERNAL_ERROR`.

The ABI does not attempt to classify stale, wrong-type, or undersized raw pointers because those
calls are outside its preconditions.

### Create and ownership

- `out_executable` is validated first and set to null before any other validation or framework
  work. Every failure leaves it null.
- `context`, every required array pointer, and every required dimensions row must be non-null;
  `value_count`, `node_count`, `feed_count`, and `target_count` must be positive; every rank must
  be `1..16`; every used dimension must be positive; every unused dimension cell must be zero;
  and checked element/byte counts must fit both `uint64_t` and `NSUInteger`. Every node input must
  name a feed or an earlier node output, every node output must be unique, and feed/target indices
  must be unique values with the required producer boundary role.
- Native code creates one fixed-shape FLOAT32 placeholder per ordered feed, lowers every ordered
  node to one NEG tensor using the indexed feed or earlier result, and collects ordered targets.
  Values internal to the partition remain MPSGraph tensors. It creates shaped feed types in feed
  order, one graph device from the context device, and one specialized `MPSGraphExecutable` with
  `MPSGraphOptionsNone`.
- Because the SDK compile entry receives a feed dictionary rather than promising dictionary
  encounter order, native compilation compares the executable's reported feed and target tensor
  arrays by Objective-C object identity against the stable placeholders/targets. It requires an
  exact bijection and stores immutable executable-order-to-stable-order permutations. A missing,
  duplicate, foreign, or unmatched reported tensor maps to `GRAPH_COMPILATION_FAILED`. The Java
  and C ABI order remains stable; SDK arrays are permuted internally at run time.
- Compilation occurs synchronously in this function. A nil graph, placeholder, NEG tensor,
  shaped type, graph device, compilation descriptor, or executable maps exactly to
  `GRAPH_COMPILATION_FAILED`; executable-box allocation maps to `ALLOCATION_FAILED`; an
  Objective-C exception maps to `INTERNAL_ERROR`.
- On success, one retained opaque executable box is published. It strongly owns the executable,
  immutable copied ordered feed/target shapes and byte extents, and context box needed for its
  device/queue.
- Java acquires a provisional child lease atomically under `MetalDeviceContext`'s existing
  lifecycle lock before the native create/compile call. Native creation runs while that lease
  keeps the context handle alive even if owner close begins; successful wrapper construction
  adopts the same lease without another increment. If native creation fails, Java releases the
  provisional lease. If wrapper construction fails after native success, Java releases the
  native executable first and the provisional lease second. The construction failure remains
  primary; distinct executable-release and then deferred-context-release failures are suppressed
  in that cleanup encounter order, and the exact primary object is never self-suppressed. Thus
  owner close cannot release the native context between creation and lease publication.
- `synaptik_metal_mpsgraph_executable_release` consumes one live non-null handle exactly once.
  Java supplies idempotency and must never retry a failed native release. Resource close marks
  the wrapper closed, attempts executable release, and then releases the adopted child lease;
  executable-release failure is primary and a distinct deferred-context-release failure is
  suppressed, matching `MetalBufferRepresentation`. Prepare rollback retains its existing
  preparation failure as primary and suppresses this resource-close failure in reverse resource
  cleanup order.

### Execution, supplied destinations, errors, and synchronization

- Run validates the executable handle, exact input/output counts, non-null arrays, and every
  buffer handle. Each input and output must be a live buffer box from the executable context's
  exact Metal device with logical byte size at least its ordered descriptor's required extent.
  Identical native buffer handles across the input and output sets are rejected as
  `INCOMPATIBLE_RESOURCE`; this route makes no in-place or alias-safety claim.
- Run creates `MPSGraphTensorData` bindings directly over the ordered input and output
  `MTLBuffer` objects, using each retained shape and `MPSDataTypeFloat32`. The input array follows
  stable feed order and the caller-supplied results array follows stable target order at the ABI;
  native code uses the stored permutations to present both arrays in executable-reported order to
  the SDK. Java passes the run-owned workspace's already populated native address segments
  directly; it allocates no Arena, array, collection, segment, or wrapper and performs no handle
  copy or address marshalling in `executeBound()`. Shared assigned buffer positions are resolved
  once during finalization and cold binding, so repeated internal consumption of a value reuses
  its one MPSGraph tensor rather than adding another feed.
- The executable and graph use `MPSGraphOptionsNone`; no implicit CPU-result synchronization
  option or hidden output materialization is requested.
- One fresh execution descriptor has `waitUntilCompleted = YES` and a completion handler that
  captures a nullable execution `NSError` inside the native call. The synchronous executable
  method is invoked on the context command queue exactly once.
- A nil input or result tensor-data object, input/results array, or execution descriptor maps
  exactly to `EXECUTION_FAILED`. The call returns `OK` only after the synchronous method returns,
  the completion handler reports no error, and the returned results count equals `target_count`
  with no nil result. A completion `NSError`, nil run result, malformed count, or nil returned
  result maps to `EXECUTION_FAILED`; an Objective-C exception maps to `INTERNAL_ERROR`.
- Success means the supplied output buffers contain their ordered results. Synaptik performs no
  explicit subsequent device-to-device or host copy and requests no hidden output
  materialization. This does not claim that MPSGraph internally uses no temporary storage. Native
  autorelease pools and framework wrapper allocation are bounded per invocation; there is no
  per-element allocation, graph compilation, route choice, map lookup, reflection, retry, or
  fallback.
- The synchronous wait is the boundary after the one compiled maximal region. No wait or host
  copy occurs between its nodes. General asynchronous
  completion and cross-region event chaining remain out of scope.

## Zero-copy and minimal-copy semantics

- Each caller-owned host value is copied at most once into its caller-owned Metal input buffer
  before Runtime execution. Runtime then borrows those Metal representations for the run.
- Each eligible compile-time splat feed is expanded and uploaded exactly once into a newly
  allocated run-owned Metal buffer during that run's cold state creation through
  `InitializedBuffer`. It is not uploaded during prepare and not touched by hot execution.
- The native executable reads the assigned input `MTLBuffer` instances directly.
- Each boundary output is a fresh run-owned Metal representation created for its assigned output
  slot. Synaptik supplies those `MTLBuffer` destinations in the caller-supplied results array and
  performs no explicit post-execution copy or requested hidden output materialization.
- There is no intermediate host representation or Synaptik-requested output copy inside the
  compiled region. MPSGraph remains free to use undocumented internal temporary storage.
- Adjacent Metal work would retain Metal residency, but this task accepts only one exact Metal
  partition and makes no general multi-region or mixed-owner composition claim.
- The route waits once when its synchronous region completes. A later CPU consumer or explicit
  host observation would require a boundary transfer; this task has no CPU/Metal schedule and
  downloads only in the opt-in test while the publication lease remains open.
- “Zero-copy” here means Synaptik requests no intermediate output materialization and performs no
  explicit copy after host upload and before optional host observation. It does not mean caller
  host storage aliases `MTLBuffer.contents` or that MPSGraph never uses temporary storage.

## Out of scope

- FLOAT16, BFLOAT16, FLOAT64, integer, BOOL, zero-element, scalar-rank, unresolved-layout,
  strided, view, offset, dynamic-shape, broadcast, or multi-output execution
- any operation other than unary NEG
- custom Metal kernels, shader libraries, pipeline state, fusion, optimizer routes, or route
  candidate generation
- public Engine composition, `CpuEngineBackendComposition` changes, automatic host input
  binding, standard Engine `compute`, mixed-owner scheduling, CPU fallback, or backend discovery
- a generic Metal integration/factory, public Metal storage, raw native handles, Objective-C or
  MPSGraph Java types, or a backend registry
- general asynchronous API, futures, events, cancellation, timeout, cross-run overlap policy, or
  command-buffer exposure
- bounded buffer pooling, representation reuse, aliasing, lease-bound public `MemorySegment`
  views, persistent/shared prepared constant buffers, generic constant materialization, or
  whole-plan/local route autotuning
- native executable serialization, cache persistence, packaging, discovery, extraction, signing,
  notarization, publishing, Intel macOS, iOS, tvOS, or non-Apple platforms
- architecture-contract, ADR, shared production-module, production dependency, or global
  integration-test changes; the exact test-only Compiler/conformance dependencies and dependency
  architecture-test update in scope do not change production direction
- legacy source or package reuse; the legacy branch remains read-only capability evidence only

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially core invariants, Planning,
  Prepare, Runtime, concrete backends, Metal, lifecycle, and dependency rules
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime resource ownership](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [ADR 0013: Prepared-execution persistent-resource lifecycle](../../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)
- [Metal strategy note](../../../../design/notes/metal-backend-strategy.md)
- [Metal master plan](../master-plan.md)
- [Metal 0001](0001-metal-capability-storage-and-native-foundation.md)
- [Compiler 0006B7](../../../modules/compiler/tasks/0006b7-final-neg-logical-layout-closure.md)
- [Runtime 0016](../../../modules/runtime/tasks/0016-persistent-prepared-resource-lifecycle.md)
- [Prepare 0006](../../../modules/prepare/tasks/0006-persistent-prepared-resource-finalization-transaction.md)
- [Engine 0010](../../../modules/engine/tasks/0010-prepared-handle-ownership-and-closure.md)

## Architecture constraints

- Planning evaluates only occurrence capability and selects `owner = metal`; it does not name,
  select, configure, compile, or execute MPSGraph.
- Metal analysis owns exact route validation, whole-partition lowering, stable boundary ordering,
  complete boundary input/output declarations, exact splat-source classification, and the one
  address-workspace declaration.
- Shared Prepare owns projection, staged invocation, slot assignment, validation, and
  transactional handoff but never inspects the Metal plan or native resource.
- Metal finalization compiles the executable only after assignment and cannot add a declaration,
  change route, or allocate per-run buffers.
- Runtime sees only one immutable partition `PreparedExecutable`, one nominal
  `PreparedResource`, ordinary
  representations, and a prepared schedule. It sees no operation, graph node, MPSGraph type,
  native handle, backend lookup, or route choice on the hot path.
- `PreparedExecution` is the unique persistent-resource owner. The executable may retain a direct
  reference for invocation but does not close it or create another ownership occurrence.
- Each run receives isolated output storage. The caller-owned input representation is borrowed;
  each splat input and output is fresh and run-owned; the result owns the published output lease
  under current Runtime rules. The run-owned workspace gives native address arrays an explicit
  close lifecycle absent from `BoundInvocation`.
- Engine remains the composition root, but no Engine source changes here. Metal does not depend
  on Engine, and this route is not presented as a supported public Engine composition.
- No architecture authority changes are expected. Stop if implementation requires a shared
  contract, dependency direction, public Engine seam, mixed-backend rule, or ownership change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.metal` — exact public capability truth only.
- `io.github.pho001.synaptik.backend.metal.internal` — package-private ABI, context, storage,
  MPSGraph persistent resource, route analysis/finalization, prepared executable, binding, and
  exact schedule assembly.

No package is added. The master-plan package map is narrowed for this first route: the planned
`.internal.prepare` and `.internal.route.mpsgraph` packages remain deferred because splitting this
small package-private vertical slice would require widening cross-package implementation
visibility before a second route proves a stable seam.

Type placement:

- `MetalMpsGraphExecutableResource` — package-private `PreparedResource` owning one opaque native
  executable and context lease.
- `MetalNegAnalysisInputs` — package-private immutable context-bearing analysis inputs.
- `MetalNegPreparationPlan` — package-private immutable selected-route facts: exact device-context
  identity, partition DAG, stable value/tensor indices, ordered boundary source roles/descriptors/
  byte geometry, one address-workspace declaration, and no executable state.
- `MetalNegPartitionPreparer` — package-private complete maximal-NEG-partition analysis and
  declarations.
- `MetalNegPartitionFinalizer` — package-private assignment validation, native compilation,
  executable construction, local rollback, and atomic result handoff.
- `MetalNegPreparedExecutable` — package-private immutable Runtime recipe; its bound invocation
  and run-owned closeable native-address workspace may be private/package-visible nested types in
  this same allowlisted path because neither is reused as a public contract.
- `MetalNegPreparedScheduleAssembler` — package-private sole-partition representation, execution,
  and publication recipe assembly.

No new public Java type, constructor, method, package, service, factory, registry, or Engine-facing
surface is introduced. The existing public `MetalCapabilityProvider` changes behavior only for
the exact capability domain.

## Affected files

Exact implementation allowlist:

Production and native paths:

1. `backends/metal/build.gradle.kts`
2. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalCapabilityProvider.java`
3. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/package-info.java`
4. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNativeApi.java`
5. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalDeviceContext.java`
6. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalBufferRepresentation.java`
7. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalMpsGraphExecutableResource.java` (new)
8. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegAnalysisInputs.java` (new)
9. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPreparationPlan.java` (new)
10. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPartitionPreparer.java` (new)
11. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPartitionFinalizer.java` (new)
12. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPreparedExecutable.java` (new)
13. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPreparedScheduleAssembler.java` (new)
14. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/package-info.java`
15. `native/metal-macos-arm64/src/synaptik_metal_foundation.m`
16. `native/metal-macos-arm64/build.sh`
17. `native/metal-macos-arm64/README.md`

Test paths:

18. `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalCapabilityProviderTest.java`
19. `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/internal/MetalFoundationTest.java`
20. `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPreparedExecutionTest.java` (new)

Conformance and dependency-enforcement paths:

21. `testing/backend-conformance/build.gradle.kts`
22. `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/MetalNegCapabilityPartitionConformanceTest.java` (new)
23. `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/BackendConformanceDependencyContractTest.java`

Documentation and planning paths:

24. `docs/backend-guide/metal-backend.md`
25. `docs/glossary.md`
26. this task
27. `docs/planning/backends/metal/master-plan.md`
28. `docs/planning/roadmap.md`

## Maximum scope

At most the exact 28 paths above: the Metal build file, 13 Metal Java production/Javadoc paths,
three native-layout paths, three Metal test paths, two backend-conformance paths, one architecture-
test path, two explanatory documentation paths, and three planning paths.

This exceeds the usual source-file guardrail because the first route is an indivisible
Java/native vertical slice: capability, staged analysis/finalization, persistent ownership,
cold binding, hot invocation, schedule integration, deterministic fakes, real native evidence,
and documentation must agree before any capability becomes true. The separate small types avoid
a route-owning god class while keeping every implementation detail package-private.

The only Metal build change is the test-only Compiler dependency, and the native build script owns
framework linkage. The only conformance/build enforcement changes are paths 21--23. Do not edit
shared production Java, Engine, CPU, build logic, architecture authority, or integration-test
source. If another path, package, public type, production module edge, or shared contract is
required, stop and return the task to planning.

## Acceptance criteria

- Capability is true exactly for the specified NEG occurrence domain and false for all other
  kinds, attributes, types, shapes, ranks, zero dimensions, and layouts. It neither receives nor
  infers source provenance and therefore gives the same answer for equal eligible descriptors
  whether the input is caller-bound or an explicit compile-time splat.
- Metal analysis accepts every arbitrary non-empty matching maximal partition, uses only the
  supplied partition-local DAG/value/logical-requirement facts, handles chains, independent
  nodes, fan-out, repeated consumption, internal values, multiple ordered boundary inputs and
  outputs, and differing positive static shapes, declares each unique boundary in the specified
  stable order with exact geometry, classifies caller versus exact FLOAT32 splat feeds, and
  declares exactly one checked address workspace.
- Automated topology cases include a multi-node chain with an internal value, independent NEG
  nodes of different valid shapes, two consumers of one input or intermediate, multiple graph
  inputs/outputs, and logical requirements representing values entering from or leaving for
  another partition. The backend-local executable integration remains a sole all-Metal graph so
  it does not imply unsupported mixed-owner schedule composition.
- Preparation rejects only occurrence-domain exclusions or malformed/inconsistent handoff facts;
  no supported topology rejection, fallback, repartitioning, hidden CPU execution, or Planning
  route logic is introduced.
- The plan retains the exact analysis `MetalDeviceContext`; finalization and schedule assembly
  reject an identity mismatch before resource acquisition or recipe construction. Finalization
  validates exact declaration/assignment identity, including the address workspace, and compiles
  one reusable native executable only after slots exist.
- Provisional context lease acquisition is atomic with context owner close. Every create/wrapper
  failure releases the executable when present and then the provisional lease exactly once;
  every finalizer-local failure after resource acquisition closes it once; all paths preserve the
  specified primary and distinct suppressed-failure order. Successful return lists exactly the
  adopted resource once.
- Shared Prepare transfers the resource to one `PreparedExecution`; close/run races use Runtime
  0016 unchanged, and repeated schedule occurrences do not duplicate ownership.
- ABI version is exactly `2`; the dylib exports exactly ten required symbols; the original seven
  signatures/semantics remain intact; the three added signatures, statuses, output-cell rules,
  ownership, validation, and error mapping match this specification.
- Java resolves and binds all ten symbols eagerly after validating version `2`, in documented
  order, before context creation. Malformed native create results are cleaned up once.
- One whole-partition prepared recipe executes twice without recompilation. Each run uses the
  exact ordered caller inputs, fresh initialized splat buffers, fresh ordered output buffers, and
  a fresh closeable address workspace selected during cold setup/binding.
- Native execution binds input and result `MPSGraphTensorData` directly over assigned
  `MTLBuffer` objects. Output bytes prove writing into the supplied destinations with no explicit
  Synaptik post-execution copy or requested hidden output materialization; no claim is made about
  MPSGraph internal temporary storage.
- The real route preserves FLOAT32 NEG semantics including finite values, infinities, NaN sign
  irrelevance, and signed zero (`+0.0 -> -0.0`, `-0.0 -> +0.0`) with raw-bit assertions where
  applicable.
- Deterministic fake/native-seam tests prove compilation-object, tensor-data/descriptor, alias,
  completion error, malformed result, Objective-C exception, and all other status mappings.
  Completion/run failures map to stable status `11`, input/output aliasing maps to `10`,
  compilation-object failures map to `9`, exceptions map to `7`, and Java retains operation/raw
  status with no retry or fallback. The real-device test need not induce an artificial failure.
- Deterministic tests prove exact context identity retention and finalizer/assembler mismatch
  rejection; Java-side cardinality/rank/padding/index/producer/Shape/geometry rejection before
  downcall; independent native rejection of matching safely inspectable malformed facts;
  caller/splat feed ordering and repeated consumption; exact FLOAT32 splat bits including signed
  zero, infinities, and NaN payload preservation; one fresh allocation/upload per splat per run;
  initialized validity, rollback/cleanup, concurrent-run isolation, and no prepared/shared
  constant buffer; and address-workspace identity, cold marshalling, exact close, and absence of
  hot Java allocation/marshalling.
- Runtime hot execution performs no graph compilation, operation inspection, slot lookup,
  representation cast, map lookup, reflection, string dispatch, route selection, allocation of
  Java collections, host copy, or per-element allocation.
- The only wait is the synchronous compiled-region boundary. `MPSGraphOptionsSynchronizeResults`
  is not enabled.
- Each caller host input requires exactly one explicit upload before execution. Each eligible
  splat is uploaded once per run by its `InitializedBuffer` creator during cold state creation,
  never during prepare or hot execution. Output download occurs only in the test's observable
  host-consumption step while the result lease is open.
- Concurrent runs use isolated buffers and may reuse the immutable executable. Tests use bounded
  latches, not sleeps, for Java lifecycle races; the native route makes no throughput or overlap
  claim.
- Public Java surface remains `MetalCapabilityProvider` only. No Objective-C, MPSGraph, native
  handle, storage, preparer, finalizer, executable, or context type becomes public.
- No Engine source changes and no public standard-compute claim occur. Documentation explicitly
  states that current Engine composition is still CPU-only and mixed-owner execution is absent.
- `backends/metal` adds Compiler only as `testImplementation`; its production dependencies and
  direction remain unchanged. Backend conformance proves public capability/maximal-partition
  closure, and the architecture test locks `testImplementation(project(":backends:metal"))`
  in backend conformance and `testImplementation(project(":modules:compiler"))` in Metal while
  rejecting production forms of both edges and continuing to reject Engine and OpenBLAS-provider
  edges. Generic Engine integration tests remain unchanged because no supported Metal Engine
  composition exists.
- Focused fake-native tests, ordinary Metal tests, native build/symbol/link inspection, real
  native prepared execution, Javadoc, Markdown, exact scope/status/order, staged-file, and
  whitespace validation all pass.
- A separate clean documentation-focused pass finalizes every affected Javadoc, Metal guide,
  native README, glossary impact, task evidence, master plan, and roadmap before completion.

## Performance constraints and evidence

- Compilation, shape conversion, graph construction, target selection, and resource acquisition
  happen only during finalization.
- Cold binding writes ordered buffer addresses once into a run-owned closeable workspace. Bound
  execution retains direct resource/workspace references and makes one native call. There is no
  Java allocation, address-array marshalling, lookup, cast, graph traversal, operation dispatch,
  or route decision in `executeBound()`.
- Native invocation may create the bounded framework tensor-data/array/descriptor wrappers
  required by the SDK, but performs no allocation proportional to element count and requests no
  hidden intermediate output materialization. No claim is made about MPSGraph internal temporary
  storage.
- The route performs exactly one region submission and one completion wait per invocation.
- The implementation must inspect native source and Java bytecode/source for these properties.
  No latency or throughput threshold is required for this first correctness route, and no
  performance superiority claim may be made.

## Tests / validation

Implementation development may run focused classes. After executable Java and native source
stabilize, run one final ordinary Metal module suite:

```bash
./gradlew :backends:metal:test
```

Build and inspect the native library:

```bash
./native/metal-macos-arm64/build.sh
file native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
otool -L native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
```

The symbol inspection must show exactly the ten specified `synaptik_metal_*` exports. Link
inspection must show Foundation, Metal, and MetalPerformanceShadersGraph.

On the current macOS arm64 host, run the exact opt-in native cases with the freshly built dylib:

```bash
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew :backends:metal:test \
  --tests '*MetalFoundationTest.nativeFoundationRoundTrip' \
  --tests '*MetalNegPreparedExecutionTest.nativePreparedNegRoundTripAndReuse'
```

The task cannot become Complete by skipping the real-device case on this eligible host. If the
environment sandbox hides the Metal device, rerun the exact test with approved unsandboxed device
access and record both outcomes. If supplied result binding, executable reuse, or signed-zero
semantics fail, keep the capability fail-closed and return the task to planning. Failure-status
mapping is gated by deterministic fake/native-seam tests, not by an unspecified artificial
real-device MPSGraph failure.

Run the narrow conformance and dependency-enforcement suites necessitated by the new backend
behavior and test-only conformance edge:

```bash
./gradlew :testing:backend-conformance:test \
  --tests '*MetalNegCapabilityPartitionConformanceTest'
./gradlew :testing:architecture-tests:test \
  --tests '*BackendConformanceDependencyContractTest'
```

Inspect the compiled/public/internal shape and hot path:

```bash
javap -classpath backends/metal/build/classes/java/main -p \
  io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider \
  io.github.pho001.synaptik.backend.metal.internal.MetalNegPreparedExecutable
```

Manually confirm no new public Metal type, no forbidden Engine import, no native handle accessor,
and a direct bound invocation. Stable recurring shape/capability rules belong in automated tests;
the manual inspection addresses only package/private and hot-path risks not represented by
ordinary tests.

The native-backed Metal module test is the truthful typed backend-local prepared integration
evidence: its test-only Compiler edge lets it construct `CompileArtifacts` and traverse shared
`GraphPreparation`, Runtime representation creation/cold binding/execution/publication, and native
MPSGraph without widening production dependencies. `ARCHITECTURE.md` expects backend behavior
changes to update backend conformance, so the narrow public-surface conformance test proves exact
NEG capability and that adjacent eligible occurrences form one maximal Metal partition; it does
not reach package-private execution or create a public Metal integration seam. The focused
architecture test reads both affected build files and enforces that their new edges are test-only.

`testing/integration-tests` exercises supported end-to-end Engine composition. That composition
remains CPU-only, so adding a Metal test there would be impossible without out-of-scope Engine
work and would falsely imply public Metal support. It remains unchanged for a concrete
architecture reason, not as a waiver of backend validation.

Documentation-focused pass, after all Javadocs and explanatory text stabilize:

```bash
./gradlew :backends:metal:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  native/metal-macos-arm64/README.md \
  docs/backend-guide/metal-backend.md docs/glossary.md \
  docs/planning/backends/metal/tasks/0002-mpsgraph-prepared-execution-route.md \
  docs/planning/backends/metal/master-plan.md docs/planning/roadmap.md
git diff --check
git diff --cached --check
git status --short -uall
```

If the temporary validator is absent, use an equivalent tool outside the repository. Validate
local links and heading anchors, unique effective anchors, balanced fences, final newlines,
trailing whitespace, exact 28-path scope, package/type placement, ABI/version/status/symbol
consistency, task/master/roadmap status and order, no staged files, and no later detailed Metal
task specification.

Repository-wide Java validation is deferred to the Metal capability checkpoint or continuous
integration because the production change remains inside one concrete backend and changes no
shared contract or production edge. The focused architecture test above is required because the
conformance build's exact test dependency list changes; a full architecture or repository suite
would duplicate unrelated coverage and is not promised by this task.

## Dependencies

- Metal 0001 — Complete; supplies the native context, shared-storage buffers, host byte access,
  child leases, failure model, and current seven-symbol ABI to evolve.
- Compiler 0006B7 — Complete; supplies final canonical logical layouts only for fully static
  still-unresolved exact NEG outputs and directly consumed compile-time splat inputs before
  capability queries. Its initial focused 41-test run, pre-correction full Compiler
  40-suite/281-test run, Compiler Javadoc, independent documentation review, and corrected focused
  closure-test rerun passed.
- Runtime 0016 — Complete; supplies unique persistent-resource ownership and close/run leases.
- Prepare 0006 — Complete; supplies finalizer-local obligation, shared transactional rollback,
  and ownership transfer.
- Engine 0010 — Complete; supplies outward prepared-handle closure for a future composition. It
  is a lifecycle prerequisite, not an authorization to change Engine here.
- Installed macOS SDK 27.0 headers and frameworks plus JDK 26 FFM.

Model 0026 is not a dependency because this task supports only existing `FLOAT32`. No tuning,
custom-kernel, packaging, or mixed-owner prerequisite applies.

## Follow-up tasks

- Metal 0003 remains Draft for custom Metal kernels and broader route work.
- Metal 0004 remains Draft for typed route candidates and cache compatibility.
- A later separately planned composition task must decide how Engine receives Metal
  collaborations, uploads ordinary `HostTensorStorage`, materializes host results, and composes
  mixed owners. This task does not create that detailed specification.
- More operations/types/layouts, device selection, asynchronous event chaining, buffer pooling,
  and packaging remain future work. Whole-partition lowering for the supported NEG domain is
  complete task-0002 scope, not a follow-up.

## Architecture impact

Expected impact: None.

The task implements responsibilities already assigned to Metal and uses the accepted staged
Prepare and persistent-resource lifecycle exactly. It changes no authority, ownership boundary,
dependency direction, or shared API. If implementation requires an architecture, shared-module,
Engine, dependency, or package-boundary change, stop and report the exact conflict instead of
editing architecture authority.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the clean implementation agent for Synaptik Metal task 0002. Work in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/backends/metal/master-plan.md, and
docs/planning/backends/metal/tasks/0002-mpsgraph-prepared-execution-route.md in full. Read the
task's directly referenced Compiler 0006B7, Runtime 0016, Prepare 0006, Engine 0010, Metal 0001, ADR 0013, Metal
strategy, installed SDK headers, current Metal Java/native/tests, and affected shared contracts.

Do not start until Compiler 0006B7 is Complete and this task has been returned to Ready.

Implement exactly the 28-path specification. Preserve the exact ABI, capability domain,
whole-maximal-partition boundary, eligible per-run splat initialization, exact context identity,
run-owned address workspace, allocation-free/marshalling-free one-downcall hot execution, supplied
destinations, synchronization, ownership, rollback, failure, package/private surface, and
validation rules. Do not add Engine composition, mixed-owner work,
another operation/type/layout, custom kernels, async APIs, packaging, discovery, tuning, or any
out-of-scope change. Stop for an architecture, shared-contract, dependency, package, ABI, SDK,
direct-binding, or scope conflict rather than inventing a boundary.

Run the focused/ordinary/native validation and record exact evidence. Then hand the stabilized
diff and evidence to a distinct clean documentation-focused context. That pass must follow
docs/developer-guide/documentation-rules.md, apply the General/API-Javadoc/Backend-Guide/Planning
profiles, finalize Javadocs/docs/glossary/planning, validate the exact scope and status, and avoid
repeating stable Java tests unless executable behavior changes or a concrete risk requires it.
Mark Complete only after the real-device route and every acceptance gate pass.
```

## Stop conditions

Stop and return the task to planning if any of these occurs:

- the SDK does not write the supplied result tensor data directly or cannot reuse one executable;
- synchronous completion cannot report native execution failure deterministically;
- the route needs graph compilation, allocation proportional to element count, a host copy, or
  route selection during hot execution;
- finalization cannot return the executable as one identity-unique `PreparedResource`;
- the whole-partition route needs a workspace beyond the one exact declared address workspace or
  changes declarations after assignment;
- Runtime binding would require raw `Object`, unchecked casts, a native auxiliary binding owner,
  lookup, or a shared-contract change;
- a public Metal type other than the provider, Engine source, mixed-owner composition, production
  module dependency, Gradle change beyond the exact test-only edges, new package, architecture
  change, or twenty-ninth path is required;
  or
- the unsandboxed real-device test cannot prove two-run reuse, supplied output destinations, and
  exact FLOAT32 NEG behavior, or deterministic seam tests cannot prove error mapping.

## Local decisions

- Keep the first route in the existing `.internal` package. Seven focused package-private types
  make the lifecycle visible without widening implementation visibility before a second route
  proves a stable subpackage seam.
- Use one executable for the complete maximal Metal partition. Stable first-encounter value/feed
  order and producer-order targets cover chains, independent nodes, fan-out, repeated
  consumption, internal values, multiple boundaries, and differing valid shapes without
  repartitioning.
- Retain the exact analysis `MetalDeviceContext` by identity. Finalization and schedule assembly
  reject an equal-looking but distinct context before resource acquisition or creator execution.
- Represent explicit positive-rank `FLOAT32` splats with per-run `InitializedBuffer` creators.
  Preparation retains typed scalar facts but allocates and uploads no constant buffer; each run
  gets fresh storage containing the exact raw bits.
- Declare one per-run native-address workspace. Cold binding fills its feed-then-target pointer
  array once; the bound hot method retains direct slices and performs one downcall.
- Treat deterministic fake/native-seam coverage as the accepted failure-status matrix. The
  real-device gate proves successful execution, supplied destinations, and reuse without trying
  to manufacture undocumented MPSGraph failures.
- Compose evidence rather than claim an impossible public-port test. Public
  `GraphCompilationPort` plus shared `GraphPreparation` proves the real caller-input route;
  backend-local typed `PrepareContext` tests prove positive-rank splats; existing shared Prepare
  tests prove `ConstantSource -> PrepareContext.constants() -> InitializedBuffer`.

## Known limitations

- Capability remains limited to equal-shape/equal-gradient-flag, positive fully static rank
  `1..16`, resolved dense-contiguous non-view zero-offset `FLOAT32` unary `NEG` occurrences.
- The schedule assembler accepts one sole all-Metal partition. There is no public Metal Engine
  adapter, standard `compute` composition, mixed-owner schedule, CPU fallback, or cross-region
  transfer claim.
- The public compilation port has no explicit positive-rank forward-constant ingress. The task
  therefore does not claim one public `GraphCompilationPort` positive-rank-splat test.
- Execution is synchronous and the native executable resource serializes its invocation boundary.
  No asynchronous, cancellation, timeout, cross-run throughput, or overlap guarantee is made.
- No FLOAT16, BFLOAT16, FLOAT64, integer, BOOL, scalar-rank, zero-extent, dynamic-shape,
  strided/view/offset, broadcast, multi-output, custom-kernel, pooling, persistent constant,
  serialization, packaging, discovery, tuning, or performance-superiority capability is added.
- MPSGraph may use internal temporary storage. The no-copy statement is limited to Synaptik not
  requesting hidden result materialization or performing an explicit post-execution copy.

## Validation evidence

- Clean implementation context stabilized exactly the first 20 paths in the task allowlist. Its
  final ordinary `./gradlew :backends:metal:test` run passed 35 tests with zero failures/errors
  and three expected opt-in skips. The focused prepared-execution run passed 20 tests with zero
  failures/errors and two expected opt-in skips. No executable Java or native source changed in
  the documentation-focused completion context, so these successful suites were not repeated.
- `./native/metal-macos-arm64/build.sh` passed. `file` identified the output as an arm64 Mach-O
  dynamic library. `nm -gU` showed exactly the ten required `synaptik_metal_*` exports, and
  `otool -L` showed Foundation, Metal, and MetalPerformanceShadersGraph linkage.
- The exact command
  `SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test --tests '*MetalFoundationTest.nativeFoundationRoundTrip' --tests '*MetalNegPreparedExecutionTest.nativePreparedNegRoundTripAndReuse'`
  passed outside the sandbox with both real-device tests executed and no skips.
- `./gradlew :testing:backend-conformance:test --tests '*MetalNegCapabilityPartitionConformanceTest'`
  passed two tests. `./gradlew :testing:architecture-tests:test --tests '*BackendConformanceDependencyContractTest'`
  passed one test. These are the required focused conformance and dependency gates; no generic
  Engine integration test applies because supported Engine composition remains CPU-only.
- Manual source and `javap -p` inspection covered `MetalCapabilityProvider` and
  `MetalNegPreparedExecutable`: the provider is the only public Metal type; no native handle
  accessor or Engine import was introduced; the cold-bound invocation directly calls
  `resource.run(...)`; and hot execution performs one native downcall with no Java address
  marshalling, allocation, lookup, or dispatch.
- Evidence composition was reviewed against the source and tests. The real
  `nativePreparedNegRoundTripAndReuse` path uses `GraphCompilationPort`, `GraphPreparation`,
  Runtime, and MPSGraph with caller inputs. `positiveRankSplatsAreColdInitializedInStableOrderAndIsolatedAcrossRuns`
  covers typed `PrepareContext` analysis/finalization/schedule/Runtime splats. Existing
  `GraphPreparationTest` coverage proves constant-source projection into
  `PrepareContext.constants()` and rejects a constant not backed by `InitializedBuffer` or a
  non-constant backed by one.
- Documentation-focused completion context `/root` applied the General, API-Javadoc,
  Backend-Guide, and Planning profiles. It reviewed all 13 affected production/Javadoc paths,
  the Objective-C ABI and build script, the three Metal tests, conformance and architecture
  tests, SDK/strategy/ADR boundaries, and implementation evidence. It finalized both package
  Javadocs, the native README, Metal guide, glossary, this task, master plan, and roadmap without
  changing executable behavior.
- `./gradlew :backends:metal:javadoc` passed after final package-Javadoc edits. Targeted Markdown
  validation passed local file links, heading anchors and duplicate effective anchors, balanced
  fences, final newlines, and trailing-whitespace checks for the six Markdown files. ABI version,
  all twelve status values, exact ten symbols, package/type placement, task/master/roadmap status
  and ordering, and absence of a detailed later Metal task were consistent.
- Final scope validation found exactly the authorized 28 paths: the stabilized 20 implementation
  paths plus the eight documentation allowlist paths. `git status --short -uall` showed no staged
  files. `git diff --check` and `git diff --cached --check` passed.

## Implementation notes

- ABI version 2 preserves the original seven functions and statuses `0..7`, adds exactly the
  three executable functions and statuses `8..11`, eagerly resolves all ten symbols, and links
  MetalPerformanceShadersGraph in addition to Foundation and Metal.
- Analysis produces stable value indices, ordered unique feeds and targets, exact buffer geometry,
  typed optional splats, and one pointer workspace without physical allocation. Finalization
  validates assignment identity/order/geometry and compiles the persistent executable only after
  slots exist.
- Native compilation checks the SDK-reported feed/target tensors by identity and retains
  permutations between stable ABI order and framework order. Native execution binds
  `MPSGraphTensorData` directly over ordered input and supplied output `MTLBuffer` instances,
  waits once at the region boundary, and validates completion plus returned results.
- The executable resource adopts one provisional context lease and implements exactly-once,
  idempotent release. `PreparedExecution` owns it across runs. Caller buffers are borrowed;
  splats, outputs, and address workspace are run-owned; published outputs remain leased through
  `RunResult`.
- Gradle changes are test-only: Metal tests depend on Compiler to construct public compile
  artifacts, and backend conformance depends on Metal. The focused architecture test rejects
  production versions of those edges. No production dependency or architecture rule changed.

## Completion summary

- Completed changes: implemented exact Metal capability truth, whole-maximal-partition NEG
  analysis/finalization, ABI-v2 MPSGraph compilation and synchronous execution, persistent
  executable ownership, per-run splat/output/address storage, direct supplied destinations,
  deterministic failure mapping, and focused conformance/dependency enforcement.
- Files changed or created: exactly the 28 paths listed in this task; no path outside the
  allowlist and no staged file.
- Tests and validation: final Metal suite had 35 total tests with three expected opt-in skips;
  focused execution had 20 total tests with two expected opt-in skips; native build, exact
  exports/framework links, both exact
  real-device tests outside the sandbox, two-test conformance, one-test architecture, hot-path
  inspection, Metal Javadoc, Markdown, scope, consistency, and whitespace gates passed.
- Documentation-agent review: clean documentation-focused context `/root` completed the required
  independent review without changing executable behavior or rerunning stable Java suites.
- Documentation impact: package summaries, native ABI/build guide, backend lifecycle guide,
  glossary, task evidence, master plan, and roadmap now describe the current capability and its
  boundaries.
- Javadoc review: every changed production Javadoc was inspected against implementation/tests;
  member Javadocs remained accurate, while both stale package Javadocs were finalized.
- Glossary impact: capability-provider and physical-representation entries were corrected, and a
  reusable MPSGraph prepared-executable distinction was added.
- Unresolved issues: None within task scope.
- Follow-up required: None. Metal 0003 and 0004 remain Draft master-plan rows without detailed
  specifications.

Status: Complete
