# Task 0002: [MPSGraph](../../../../glossary.md#term-mpsgraph) [Prepared Execution](../../../../glossary.md#prepared-execution--preparedexecution) Route

## Status

Complete

## Goal

Deliver the first truthful executable [Metal](../../../../glossary.md#term-metal) capability: every non-empty maximal Metal-owned
[partition](../../../../glossary.md#partition) composed solely of individually supported fully static, positive-[shape](../../../../glossary.md#shape), resolved
dense-[contiguous](../../../../glossary.md#term-contiguous) [`FLOAT32`](../../../../glossary.md#term-float32) [unary](../../../../glossary.md#term-unary) `NEG` occurrences.

[Compiler](../../../../glossary.md#term-compiler) 0006B7 is a mandatory prerequisite. [Occurrence](../../../../glossary.md#term-occurrence) capability queries contain [descriptors](../../../../glossary.md#term-descriptor)
but no [compile](../../../../glossary.md#compile)-time [source provenance](../../../../glossary.md#term-source-provenance), so this route must accept an otherwise eligible NEG whose
feed is an explicit compile-time splat. Compiler 0006B7 first closes only the fully static
still-unresolved NEG outputs and direct splat inputs needed for truthful capability [admission](../../../../glossary.md#term-admission).

The route must lower and compile one shape-specialized [Metal Performance Shaders Graph](../../../../glossary.md#mps-mpsgraph-and-openblas)
([`MPSGraph`](../../../../glossary.md#term-mpsgraph)) executable during [backend](../../../../glossary.md#backend) finalization, transfer that persistent native resource
transactionally through [Prepare](../../../../glossary.md#prepare), bind per-[run](../../../../glossary.md#term-run) Metal [buffers](../../../../glossary.md#term-buffer) without exposing [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) values
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

This task establishes a backend-local [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution) route through the current shared
[Planning](../../../../glossary.md#term-planning), Prepare, and [Runtime](../../../../glossary.md#term-runtime) contracts. It does not add a public [Engine](../../../../glossary.md#term-engine) Metal composition or
claim mixed-[owner](../../../../glossary.md#term-owner) execution.

## Pre-implementation audit

### Repository evidence

- Metal 0001 currently supplies a fail-closed provider, one default-device/command-queue native
  context, fresh shared-storage <span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [device glossary](../../../../glossary.md#term-device))</span>`MTLBuffer` wrappers, checked host upload/download, context child
  leases, and exact once-only cleanup. It submits no command and uses ABI version <span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [leases glossary](../../../../glossary.md#term-lease); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>`1` with seven
  symbols.
- [Planning](../../../../glossary.md#term-planning) asks capability per [operation](../../../../glossary.md#operation) [occurrence](../../../../glossary.md#term-occurrence) and retains only [`BackendId`](../../../../glossary.md#backend-identity--backendid); it cannot and
  must not select an MPSGraph route.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
- [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) supplies the exact immutable partition-local DAG, fully static value
  descriptors, logical requirements, constants, and opaque backend inputs. The candidate needs
  no new shared semantic projection.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [descriptors glossary](../../../../glossary.md#term-descriptor); [constants glossary](../../../../glossary.md#term-constant); [opaque glossary](../../../../glossary.md#term-opaque); [backend glossary](../../../../glossary.md#backend))</span>
- [`BackendPartitionAnalysis`](../../../../glossary.md#backend-partition-analysis-backendpartitionanalysis) can declare the candidate input and output buffers before slot
  assignment. <span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer))</span>`BackendPartitionFinalization` then supplies their exact assigned dense plan
  positions.
- `BackendPartitionFinalizationResult` can atomically return the prepared executable and an
  acquisition-ordered <span data-glossary-navigation="true"> ([prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable))</span>[`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource). Shared Prepare rolls that resource back on every later
  failure and transfers it once to Runtime.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) already performs one checked cold representation-compatibility pass and
  produces a backend-owned [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation) with direct typed references. Its hot invocation
  may call the native executable without graph, operation, slot, map, or route lookup.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- [`PreparedRepresentationPlan`](../../../../glossary.md#prepared-representation-plan--preparedrepresentationplan) can borrow caller-supplied Metal input representations and create
  fresh run-owned Metal output representations. Its <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>`InitializedBuffer` variant can create one
  fresh run-owned representation whose logical value is already valid. This is the exact existing
  contract needed to allocate and upload each eligible FLOAT32 splat once during cold run-state
  creation for each run.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32))</span>
- `BoundInvocation` has no close hook, and `PreparedExecutable.bind(...)` explicitly forbids an
  independently closeable auxiliary binding resource. Native pointer-array storage therefore
  cannot be allocated and retained directly by the invocation. One declared per-run workspace
  can own that storage, receive exact buffer addresses once during cold binding, and close with<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
  [`RunState`](../../../../glossary.md#run-state--runstate); hot execution can then perform one [downcall](../../../../glossary.md#term-downcall) without Java allocation or [marshalling](../../../../glossary.md#term-marshalling).
- Runtime 0016, Prepare 0006, and Engine 0010 complete the inward resource owner, transactional
  handoff, close/run lease, and outward prepared-handle lifecycle. Metal does not need to change
  any of them.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [resource owner glossary](../../../../glossary.md#ownership-borrowing-and-lease); [handoff glossary](../../../../glossary.md#term-handoff); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- At task 0002's completion frontier, Engine composition was CPU-only and rejected
  mixed/multi-partition artifacts. This task therefore proved the route through backend-local
  composition of the public shared Prepare and Runtime contracts and made no public Engine or
  mixed-backend claim. Engine task 0015 later superseded only that public-composition status by
  adding explicit CPU/Metal registration and current single-owner Metal execution; mixed-owner
  execution remains unsupported.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [artifacts glossary](../../../../glossary.md#term-artifact); [owner glossary](../../../../glossary.md#term-owner))</span>

### Installed [SDK](../../../../glossary.md#term-sdk) evidence

The [planning](../../../../glossary.md#term-planning) audit inspected the installed macOS SDK at
`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk` on an [arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary) macOS 26.6.2 [host](../../../../glossary.md#term-host). `xcrun`
reported SDK version 27.0 and Apple clang 21.0.0.

The installed framework headers establish the required route:

- `MPSGraph.h` exposes
  `compileWithDevice:feeds:targetTensors:targetOperations:compilationDescriptor:` and states that
  compilation blocks and returns a graph executable specialized by the supplied shaped feeds.
- `MPSGraphMemoryOps.h` exposes a typed fixed-[shape](../../../../glossary.md#shape) placeholder.
- `MPSGraphArithmeticOps.h` exposes `negativeWithTensor:name:`.
- `MPSGraphTensorData.h` exposes
  `initWithMTLBuffer:shape:dataType:`; the [buffer](../../../../glossary.md#term-buffer)'s [device](../../../../glossary.md#term-device) supplies the [graph](../../../../glossary.md#graph) device.
- `MPSGraphExecutable.h` exposes the [synchronous](../../../../glossary.md#term-synchronous)
  `runWithMTLCommandQueue:inputsArray:resultsArray:executionDescriptor:` call. Inputs follow the
  executable feed order, and caller-supplied result tensor data follows target order.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- `MPSGraphExecutableExecutionDescriptor` exposes `waitUntilCompleted` and a completion handler
  whose nullable `NSError` reports execution failure.
- `MPSGraphExecutable` is the compiled reusable graph object and reports ordered feed and target
  tensors. It is available from macOS 12.0.
- `MPSGraphDevice.h` exposes construction from the context's exact `MTLDevice`.
- `MPSGraphOptionsNone` avoids the optional end-of-execution result synchronization blit. The
  executable call itself remains synchronous, so the route waits once at its region boundary.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>

A temporary [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) probe containing a [FLOAT32](../../../../glossary.md#term-float32) fixed-shape placeholder, NEG, [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names),
`MTLBuffer` [tensor](../../../../glossary.md#tensor)-data [binding](../../../../glossary.md#term-binding), a supplied result array, completion-error [capture](../../../../glossary.md#term-capture-meanings), and two calls
against one executable compiled successfully against those installed headers and frameworks. In
the original sandbox it observed `NO_DEVICE`. The coordinator then executed that exact
`/tmp/synaptik_mpsgraph_audit` [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) outside the sandbox; it exited `0` and printed exactly:

```text
PASS executable_reused=2 direct_output_buffer=1 completion_errors=0 results=1
```

This is preimplementation feasibility evidence only. It supports two sequential uses of one
executable, observation of the supplied destination buffer, one returned result, and the values
checked by that probe's source. It does not prove the repository [ABI](../../../../glossary.md#term-application-binary-interface-abi), Java/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)
integration, cleanup, [concurrency](../../../../glossary.md#term-concurrency), or failure delivery, and `completion_errors=0` does not test
completion-error mapping. Repository fake/native-seam tests must deterministically cover every
failure status; the real-device repository test covers successful route execution, reuse, and
assigned output without requiring an artificial [MPSGraph](../../../../glossary.md#term-mpsgraph) failure.

### Audit decision

The repository and [SDK](../../../../glossary.md#term-sdk) evidence closes the [Metal](../../../../glossary.md#term-metal) architecture and route-design questions. The
separate [Compiler](../../../../glossary.md#term-compiler) 0006B7 [descriptor](../../../../glossary.md#term-descriptor) prerequisite is Complete. Capability is
queried per [occurrence](../../../../glossary.md#term-occurrence), while `MaximalSameOwnerPartitioning` joins consecutive equal [owners](../../../../glossary.md#term-owner).
Truthful support therefore requires one route for every maximal [partition](../../../../glossary.md#partition) whose occurrences all
meet the exact positive-static-[shape](../../../../glossary.md#shape) [contiguous](../../../../glossary.md#term-contiguous) [`FLOAT32`](../../../../glossary.md#term-float32) [unary](../../../../glossary.md#term-unary) `NEG` domain:

- one or more boundary inputs prove explicit caller upload or exact run-owned splat
  initialization without adding broadcasting;
- one or more boundary outputs prove assignment, direct caller-supplied result binding, Runtime
  validity, result publication, and host observation;<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [publication glossary](../../../../glossary.md#term-publication); [host glossary](../../../../glossary.md#term-host))</span>
- finalization-time compilation proves cold specialization and transactional persistent-resource
  ownership;<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [specialization glossary](../../../../glossary.md#term-specialization); [ownership glossary](../../../../glossary.md#term-ownership))</span>
- two executions of one prepared recipe prove executable reuse and isolated per-[run](../../../../glossary.md#term-run) [buffers](../../../../glossary.md#term-buffer);
- chains, independent nodes, fan-out, repeated consumption, internal values, multiple boundary
  inputs/outputs, and differing positive static shapes prove that occurrence capability remains
  closed under the actual partitioner without inventing a partition split;<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node); [fan-out glossary](../../../../glossary.md#term-fan-out))</span>
- one executable for the whole maximal partition makes the only synchronization point the
  compiled region boundary; and<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- the exact capability row can change from [fail-closed](../../../../glossary.md#term-fail-closed) to true only with the complete route.

`ADD` would add [broadcasting](../../../../glossary.md#broadcasting) and two-input [binding](../../../../glossary.md#term-binding) without proving a different [lifecycle](../../../../glossary.md#lifecycle).
[Metadata](../../../../glossary.md#term-metadata)-only or zero-[element](../../../../glossary.md#term-element) work would not prove native execution. Another [operation](../../../../glossary.md#operation), another
[data type](../../../../glossary.md#data-type--datatype), or an unresolved [layout](../../../../glossary.md#layout) would enlarge [lowering](../../../../glossary.md#lowering) and capability truth without being
necessary for the first route.

## [Scope](../../../../glossary.md#term-scope)

- Evolve the same [native library](../../../../glossary.md#term-native-library) from [ABI](../../../../glossary.md#term-application-binary-interface-abi) version `1` to version `2`, preserving the seven
  foundation functions and adding exactly the three MPSGraph functions specified below.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
- Link the native library with `MetalPerformanceShadersGraph` in addition to Foundation and
  Metal.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>
- Extend the native context box only as needed to let compilation use its exact device and
  execution use its exact command queue.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [device glossary](../../../../glossary.md#term-device); [command queue glossary](../../../../glossary.md#term-command-queue))</span>
- Add one [opaque](../../../../glossary.md#term-opaque) persistent native executable box retaining the compiled
  `MPSGraphExecutable`, ordered feed/target descriptors and byte extents, and exact context
  prerequisite.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- Add one package-private Java [`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource) owner for the native executable. It holds a
  provisionally acquired context child lease, adopts it only after native creation succeeds,
  closes idempotently, consumes the native handle once, and never exposes the handle.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [lease glossary](../../../../glossary.md#term-lease))</span>
- Add backend-private immutable analysis inputs and preparation plan for the complete maximal
  partition route. The preparation plan retains the exact <span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [immutable glossary](../../../../glossary.md#term-immutable))</span>`MetalDeviceContext` identity supplied
  to analysis; finalization and schedule assembly reject any different context.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Analyze every non-empty [partition](../../../../glossary.md#partition) whose [nodes](../../../../glossary.md#node) are all [unary](../../../../glossary.md#term-unary) `NEG` occurrences in exact
  partition order and whose individual input/output descriptors satisfy the capability domain.
  Accept every topology admitted by the compiled graph: chains, independent nodes, fan-out,
  repeated consumption, multiple boundary inputs and outputs, differing positive static shapes
  where each unary occurrence is semantically valid, and values internal to the partition.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [fan-out glossary](../../../../glossary.md#term-fan-out); [shapes glossary](../../../../glossary.md#shape); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Derive unique boundary inputs in first external-consumer occurrence order (partition node order,
  then input-port order) and unique boundary outputs in producer node/output-port order when the
  logical requirement records a graph output or an out-of-partition consumer. Declare those
  ordered boundary inputs followed by ordered boundary outputs as shared buffers, each with exact
  checked <span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [graph glossary](../../../../glossary.md#graph); [buffers glossary](../../../../glossary.md#term-buffer))</span>`elementCount * Float.BYTES` geometry and four-byte alignment. Internal values remain
  native graph tensors and receive no shared buffer declaration or Runtime slot. Retain exact
  FLOAT32 splat facts for boundary inputs present in <span data-glossary-navigation="true"> ([byte alignment glossary](../../../../glossary.md#term-byte-alignment); [tensors glossary](../../../../glossary.md#tensor); [Runtime glossary](../../../../glossary.md#term-runtime); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>`PrepareContext.constants()` and reject a
  constant on any non-feed value or any mismatched type/source fact.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- Declare exactly one backend-local workspace for the fixed feed-plus-target native address
  arrays, with checked exact pointer-count byte geometry and address alignment. Its per-run Metal
  representation owns closeable native storage, is filled once during cold binding, and is
  released by <span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span><span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [binding glossary](../../../../glossary.md#term-binding))</span>[`RunState`](../../../../glossary.md#run-state--runstate); it is not [element](../../../../glossary.md#term-element) scratch, a [graph value](../../../../glossary.md#graph-value), or persistent prepared state.
- During finalization, map those exact declarations to assigned plan positions, compile the
  whole partition into one native executable once, construct the immutable prepared executable,
  and return the native executable resource in acquisition order.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable))</span>
- Roll back the native executable locally on every failure before a successful finalizer return,
  preserving the exact primary and suppression rules of Prepare 0006.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
- Add one package-private schedule assembler for the exact sole-partition all-Metal graph route.
  It creates a representation plan with each bindable graph input borrowed in boundary-input
  order, each compile-time splat feed represented by <span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>`InitializedBuffer` whose backend creator
  allocates a fresh run-owned Metal buffer and uploads the exact repeated FLOAT32 bits once during
  that run's cold state creation, each boundary output freshly run-owned in boundary-output order,
  and the exact run-owned address workspace, then one execution step and ordered publication
  steps. Mixed-owner sourcing and generic constant materialization remain out of scope.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [materialization glossary](../../../../glossary.md#materialization))</span>
- Add one [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) whose selections are all boundary inputs as `READ_ONLY` followed
  by all boundary outputs as `WRITE_ONLY`. Cold binding accepts only compatible live Metal buffer
  representations with sufficient logical bytes and the same device context.<span data-glossary-navigation="true"> ([device context glossary](../../../../glossary.md#device-submission-synchronization-and-execution-context))</span>
- Add one bound invocation retaining direct references to the native executable resource and the
  run-owned address workspace after cold binding has written the exact ordered input/output
  native handles once. Its hot method performs exactly one native execution downcall with no Java
  allocation, address-array marshalling, map, slot, or representation lookup.<span data-glossary-navigation="true"> ([bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [downcall glossary](../../../../glossary.md#term-downcall); [marshalling glossary](../../../../glossary.md#term-marshalling); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Change `MetalCapabilityProvider.supports(...)` to return `true` only for the exact occurrence
  domain above and `false` for every other current query.
- Add deterministic fake-native Java tests for topology/order analysis, assignments, native
  descriptor marshalling, every failure-status mapping, finalizer rollback, provisional-lease
  adoption/rollback, resource ownership, binding, alias rejection, capability/partition closure,
  execution status, close/run races delegated to Runtime, and absence of hot-path lookup.<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback); [ownership glossary](../../../../glossary.md#term-ownership); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Add an opt-in real-native test that prepares a multi-node/multi-boundary route including both a
  caller input and an eligible compile-time splat feed through [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation), uploads each
  host caller value once into its caller-owned Metal input buffer, proves each run allocates and
  uploads its own exact initialized splat buffer during cold setup, runs the same prepared
  execution twice with isolated run state, verifies direct writes in every
  assigned output buffer, downloads only for observation, and closes results, inputs, prepared
  execution, and context in ownership order.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [run state glossary](../../../../glossary.md#run-state--runstate))</span>
- Add `testImplementation(project(":modules:compiler"))` to `backends/metal/build.gradle.kts` so
  the backend-local typed integration test can construct the public <span data-glossary-navigation="true"> ([integration test glossary](../../../../glossary.md#term-integration-test))</span>`CompileArtifacts` consumed
  by `GraphPreparation`. This is test-only and does not alter Metal production dependency
  direction.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
- Add the narrow backend-conformance coverage required for new backend behavior: public
  capability truth plus maximal-partition closure for adjacent eligible NEG occurrences. Add only
  a test dependency from <span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [dependency glossary](../../../../glossary.md#term-dependency))</span>`testing/backend-conformance` to `backends/metal`, and update the
  existing exact dependency architecture test. Under task 0002's completion-time CPU-only Engine
  frontier, do not add an Engine integration test.<span data-glossary-navigation="true"> ([architecture test glossary](../../../../glossary.md#term-architecture-test); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Finalize affected Javadocs, the Metal backend guide, native README, glossary impact, and
  planning evidence in a separate clean documentation-focused context after executable behavior
  stabilizes.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Exact capability domain

`MetalCapabilityProvider.supports(query)` returns `true` if and only if all conditions hold:

1. `query.operation().kind()` is `UnaryElementwiseKind.NEG` and its attributes are exactly
   `NoOperationAttrs.INSTANCE`.
2. There is exactly one input [descriptor](../../../../glossary.md#term-descriptor) and one output descriptor.
3. Both data types are [`FLOAT32`](../../../../glossary.md#term-float32).
4. Both [shapes](../../../../glossary.md#shape) are fully static, equal, have [rank](../../../../glossary.md#term-rank) in the inclusive range `1..16`, every dimension
   is strictly positive, and checked element/byte arithmetic succeeds.<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension); [element glossary](../../../../glossary.md#term-element))</span>
5. Both layouts are present and exactly resolved dense-contiguous, non-view, zero-offset layouts
   for their shapes.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [view glossary](../../../../glossary.md#term-view-meanings))</span>
6. Input and output `requiresGrad` flags are equal, matching the current Model NEG contract; the
   route accepts either flag value because this task executes only the forward occurrence.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>

Null retains the existing `NullPointerException("query")` contract. The provider performs no
native loading, [device](../../../../glossary.md#term-device) discovery, [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), allocation, caching, or route selection.
Availability and hard requirements remain separate [Planning](../../../../glossary.md#term-planning) inputs.

Because capability is [occurrence](../../../../glossary.md#term-occurrence)-scoped, any consecutive eligible NEG occurrences may be
assigned to one maximal [Metal](../../../../glossary.md#term-metal) [partition](../../../../glossary.md#partition). Analysis must accept and lower every such non-empty
partition as one reusable executable; it must not reject a supported [topology](../../../../glossary.md#term-topology), split the
partition, inspect [graph](../../../../glossary.md#graph) context in the provider, or weaken Planning's maximal same-[owner](../../../../glossary.md#term-owner) rule.
Preparation rejection is limited to a fact excluded by the occurrence capability domain or a
malformed/inconsistent projected [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag), descriptor, logical requirement, declaration, assignment,
or native [handoff](../../../../glossary.md#term-handoff). Broader operations, types, [layouts](../../../../glossary.md#layout), and shapes remain [fail-closed](../../../../glossary.md#term-fail-closed).

## Stable whole-[partition](../../../../glossary.md#partition) [lowering](../../../../glossary.md#lowering) and [binding](../../../../glossary.md#term-binding)

Analysis assigns native value indices by walking [nodes](../../../../glossary.md#node) in exact `PartitionDag.nodes()` order,
visiting each node's input ports and then output ports, and retaining the first [occurrence](../../../../glossary.md#term-occurrence) of each
[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot). Each node contributes one input-value index and one output-value index in that same
node order. The [descriptor](../../../../glossary.md#term-descriptor) for each indexed value comes from the exact projected `GraphValue`;
each [unary](../../../../glossary.md#term-unary) occurrence requires equal input/output [shapes](../../../../glossary.md#shape), while unrelated values may have
different positive static shapes.

Feeds are the unique values with no local [producer](../../../../glossary.md#term-producer), ordered by their first external consumer
occurrence. [Targets](../../../../glossary.md#term-target) are the unique locally produced values whose logical requirement records a
[graph](../../../../glossary.md#graph) output or consumer partition outside this partition, ordered by producer node then [output
port](../../../../glossary.md#capture-graph-ports-and-reconstruction). A [fan-out](../../../../glossary.md#term-fan-out) or repeated input therefore names one indexed [tensor](../../../../glossary.md#tensor) repeatedly rather than
creating duplicate placeholders. A locally produced value consumed only inside the partition is
lowered and retained in the [MPSGraph](../../../../glossary.md#term-mpsgraph) value table but has no feed, target, shared declaration, or
[Runtime](../../../../glossary.md#term-runtime) [buffer](../../../../glossary.md#term-buffer).

Each feed is classified from the exact projected facts as either caller-bindable or an explicit
[compile](../../../../glossary.md#compile)-time splat. A splat feed must have an exact [FLOAT32](../../../../glossary.md#term-float32) [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue); its final descriptor
must independently satisfy the same capability domain as every other feed. [Source provenance](../../../../glossary.md#term-source-provenance) does
not alter occurrence capability. The schedule supplies no caller entry for a splat: its
`InitializedBuffer` creator allocates a fresh context-local buffer, fills every logical [element](../../../../glossary.md#term-element)
with the [scalar](../../../../glossary.md#term-scalar)'s exact raw FLOAT32 bits, and returns it already logically valid during cold
[run](../../../../glossary.md#term-run)-state creation. Creation/upload failure closes the fresh buffer under existing Runtime
[rollback](../../../../glossary.md#term-rollback). No prepared/shared [constant](../../../../glossary.md#term-constant) buffer survives across runs.

The shared buffer declarations, assigned plan positions, [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) selections, native
run arrays, and required-byte arrays preserve one [alignment](../../../../glossary.md#term-alignment):

```text
unique feeds in feed order       -> READ_ONLY assigned buffers -> native inputsArray
unique targets in target order   -> WRITE_ONLY assigned buffers -> supplied resultsArray
one address workspace            -> cold-marshalled input/output pointer arrays
```

Finalization rejects a missing, repeated, foreign, reordered, or geometry-inconsistent
declaration/assignment instead of reconstructing order. The schedule assembler resolves each
bindable graph input to its feed buffer and each published result to its target buffer by
`ValueId`; [publication](../../../../glossary.md#term-publication) steps retain `PublicationPlan` order, so repeated publication [aliases](../../../../glossary.md#aliasing-and-arena) may
[lease](../../../../glossary.md#term-lease) the same one target buffer without adding another native result. These rules make the
placeholder/feed and target/result orders stable across analysis, [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), cold binding, and
every run.

The preparation plan retains the exact `MetalDeviceContext` object from analysis. The finalizer
and schedule assembler both require [identity](../../../../glossary.md#term-identity-meanings) equality with their supplied context before native
work or recipe construction. Equality by [backend](../../../../glossary.md#backend)/[device](../../../../glossary.md#term-device) label is insufficient.

## Exact native [ABI](../../../../glossary.md#term-application-binary-interface-abi) version 2

The existing version [symbol](../../../../glossary.md#term-symbol) keeps its name and returns unsigned value `2`. The six existing
context/[buffer](../../../../glossary.md#term-buffer) functions keep their exact version-1 [signatures](../../../../glossary.md#term-signature) and semantics. Version `2`
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

Java maps [`uint32_t`](../../../../glossary.md#index-mappings-and-cost-dominance) and `int32_t` to `JAVA_INT`, each [`uint64_t`](../../../../glossary.md#index-mappings-and-cost-dominance) [element](../../../../glossary.md#term-element) to `JAVA_LONG`, and
every pointer position to `ADDRESS`/[`MemorySegment`](../../../../glossary.md#term-memorysegment). `value_dimensions` is a [row-major](../../../../glossary.md#term-row-major)
`value_count * 16` table; for value `v`, only the first `value_ranks[v]` entries are [dimensions](../../../../glossary.md#dimension)
and the remaining entries are zero. [Node](../../../../glossary.md#node) input/output arrays are parallel [unary](../../../../glossary.md#term-unary)-node arrays in
[partition](../../../../glossary.md#partition) order. Feed and [target](../../../../glossary.md#term-target) indices use the stable boundary orders defined above. Java
rejects every [cardinality](../../../../glossary.md#term-cardinality) relationship, [rank](../../../../glossary.md#term-rank), used dimension, zero-padded unused dimension cell,
node/feed/target index, [producer](../../../../glossary.md#term-producer) order, duplicate role, input/output [Shape](../../../../glossary.md#shape) mismatch for each NEG,
and checked element/byte/address-workspace geometry before the create [downcall](../../../../glossary.md#term-downcall). Native
independently validates nulls, every [scalar](../../../../glossary.md#term-scalar)/count/index, unused dimension cell, producer order,
matching NEG input/output Shapes, checked products and byte extents, and every other safely
inspectable structure. Java performs the corresponding exact input/output counts, handle
cardinality, [alias](../../../../glossary.md#aliasing-and-arena), context [identity](../../../../glossary.md#term-identity-meanings), logical [extent](../../../../glossary.md#term-extent), and pointer-workspace geometry [validation](../../../../glossary.md#term-validation)
before the [run](../../../../glossary.md#term-run) downcall; native independently validates all matching facts safely inspectable
from valid live boxes and readable arrays.

The ABI remains an [opaque](../../../../glossary.md#term-opaque)-pointer boundary, not a general memory-safety or handle-introspection
service. Java guarantees that context, executable, and buffer handles are live handles of the
correct native box type, are consumed once, and that every array/dimensions storage region is
readable for the count or rank implied by the scalar [arguments](../../../../glossary.md#term-argument). Native cannot independently prove
that a non-null raw pointer is live, type-correct, or sufficiently sized before dereferencing it;
stale, wrong-type, or undersized pointers are outside the ABI contract. Native still validates
nulls, scalar geometry, the contents/[device](../../../../glossary.md#term-device)/logical extents of valid live boxes, and every other
fact that can be inspected safely after Java satisfies that boundary.

Version `2` retains status values `0..7` unchanged and appends:

| Value | C name | Meaning |
|---:|---|---|
| `8` | `SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE` | Rank, a dimension, or checked [FLOAT32](../../../../glossary.md#term-float32) byte geometry is outside this route. |
| `9` | `SYNAPTIK_METAL_STATUS_GRAPH_COMPILATION_FAILED` | [MPSGraph](../../../../glossary.md#term-mpsgraph) construction or [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) produced no usable executable. |
| `10` | `SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE` | A buffer has the wrong device, insufficient logical byte extent, or aliases an input/output handle. |
| `11` | `SYNAPTIK_METAL_STATUS_EXECUTION_FAILED` | [Synchronous](../../../../glossary.md#term-synchronous) execution failed, reported an `NSError`, or did not return the exact ordered usable result count. |

Unknown status integers remain unknown and [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) with the raw integer retained. No status
string, [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) object, `NSError`, block, [callback](../../../../glossary.md#term-callback), [graph](../../../../glossary.md#graph), [tensor](../../../../glossary.md#tensor), queue, device, raw address,
or framework type crosses the C ABI.

Status classification is exact:

- null output cells, handles, required array pointers, zero counts, count mismatches, invalid or
  duplicate value indices, invalid producer order/topology, nonzero unused dimension cells, and a
  null release handle map to <span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology))</span>`INVALID_ARGUMENT`;
- rank outside `1..16`, nonpositive used dimensions, or overflowing element/byte geometry maps to
  `UNSUPPORTED_SHAPE`;
- nil graph/placeholder/NEG/shaped-type/device/compilation-descriptor/executable objects or a
  non-bijective executable feed/target report maps to <span data-glossary-navigation="true"> ([bijective glossary](../../../../glossary.md#term-bijective))</span>`GRAPH_COMPILATION_FAILED`;
- executable-box allocation maps to `ALLOCATION_FAILED`;
- a valid live buffer box with wrong device, insufficient logical extent, or identical input and
  output native handle maps to `INCOMPATIBLE_RESOURCE`;
- nil run tensor-data/array/execution-[descriptor](../../../../glossary.md#term-descriptor)/result objects, completion `NSError`, wrong result
  count, or nil result entry maps to `EXECUTION_FAILED`; and
- an Objective-C exception in create, run, or [release](../../../../glossary.md#term-release) maps to `INTERNAL_ERROR`.

The ABI does not attempt to classify stale, wrong-type, or undersized raw pointers because those
calls are outside its preconditions.

### Create and [ownership](../../../../glossary.md#term-ownership-meanings)

- `out_executable` is validated first and set to null before any other validation or framework
  work. Every failure leaves it null.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- `context`, every required array pointer, and every required [dimensions](../../../../glossary.md#dimension) row must be non-null;
  `value_count`, `node_count`, `feed_count`, and `target_count` must be positive; every rank must
  be <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>`1..16`; every used dimension must be positive; every unused dimension cell must be zero;
  and checked element/byte counts must fit both <span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>[`uint64_t`](../../../../glossary.md#index-mappings-and-cost-dominance) and `NSUInteger`. Every node input must
  name a feed or an earlier node output, every node output must be unique, and feed/target indices
  must be unique values with the required producer boundary role.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [target glossary](../../../../glossary.md#term-target); [producer glossary](../../../../glossary.md#term-producer))</span>
- Native code creates one fixed-shape FLOAT32 placeholder per ordered feed, lowers every ordered
  node to one NEG tensor using the indexed feed or earlier result, and collects ordered targets.
  Values internal to the partition remain MPSGraph tensors. It creates shaped feed types in feed
  order, one graph device from the context device, and one specialized <span data-glossary-navigation="true"> ([Native code glossary](../../../../glossary.md#native-numerical-library-and-instruction-architecture); [shape glossary](../../../../glossary.md#shape); [FLOAT32 glossary](../../../../glossary.md#term-float32); [tensor glossary](../../../../glossary.md#tensor); [partition glossary](../../../../glossary.md#partition); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [graph glossary](../../../../glossary.md#graph); [device glossary](../../../../glossary.md#term-device))</span>`MPSGraphExecutable` with
  `MPSGraphOptionsNone`.
- Because the SDK compile entry receives a feed dictionary rather than promising dictionary
  encounter order, native compilation compares the executable's reported feed and target tensor
  arrays by Objective-C object identity against the stable placeholders/targets. It requires an
  exact bijection and stores immutable executable-order-to-stable-order permutations. A missing,
  duplicate, foreign, or unmatched reported tensor maps to <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([SDK glossary](../../../../glossary.md#term-sdk); [compile glossary](../../../../glossary.md#compile); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [Objective-C glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary); [bijection glossary](../../../../glossary.md#term-bijection); [immutable glossary](../../../../glossary.md#term-immutable))</span>`GRAPH_COMPILATION_FAILED`. The Java
  and C ABI order remains stable; SDK arrays are permuted internally at run time.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [run glossary](../../../../glossary.md#term-run))</span>
- Compilation occurs synchronously in this function. A nil graph, placeholder, NEG tensor,
  shaped type, graph device, compilation descriptor, or executable maps exactly to<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
  `GRAPH_COMPILATION_FAILED`; executable-box allocation maps to `ALLOCATION_FAILED`; an
  Objective-C exception maps to `INTERNAL_ERROR`.
- On success, one retained opaque executable box is published. It strongly owns the executable,
  immutable copied ordered feed/target shapes and byte extents, and context box needed for its
  device/queue.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque))</span>
- Java acquires a provisional child [lease](../../../../glossary.md#term-lease) atomically under `MetalDeviceContext`'s existing
  lifecycle lock before the native create/compile call. Native creation runs while that lease
  keeps the context handle alive even if owner close begins; successful wrapper construction
  adopts the same lease without another increment. If native creation fails, Java releases the
  provisional lease. If wrapper construction fails after native success, Java releases the
  native executable first and the provisional lease second. The construction failure remains
  primary; distinct executable-release and then deferred-context-release failures are suppressed
  in that cleanup encounter order, and the exact primary object is never self-suppressed. Thus
  owner close cannot release the native context between creation and lease publication.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [owner glossary](../../../../glossary.md#term-owner); [releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [release glossary](../../../../glossary.md#term-release); [publication glossary](../../../../glossary.md#term-publication))</span>
- `synaptik_metal_mpsgraph_executable_release` consumes one live non-null handle exactly once.
  Java supplies idempotency and must never retry a failed native release. Resource close marks
  the wrapper closed, attempts executable release, and then releases the adopted child lease;
  executable-release failure is primary and a distinct deferred-context-release failure is
  suppressed, matching `MetalBufferRepresentation`. Prepare rollback retains its existing
  preparation failure as primary and suppresses this resource-close failure in reverse resource
  cleanup order.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [rollback glossary](../../../../glossary.md#term-rollback))</span>

### Execution, supplied destinations, errors, and [synchronization](../../../../glossary.md#term-synchronization)

- Run validates the executable handle, exact input/output counts, non-null arrays, and every
  buffer handle. Each input and output must be a live buffer box from the executable context's
  exact Metal device with logical byte size at least its ordered descriptor's required extent.
  Identical native buffer handles across the input and output sets are rejected as<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [Metal glossary](../../../../glossary.md#term-metal); [device glossary](../../../../glossary.md#term-device); [extent glossary](../../../../glossary.md#term-extent))</span>
  `INCOMPATIBLE_RESOURCE`; this route makes no in-place or [alias](../../../../glossary.md#aliasing-and-arena)-safety claim.
- [Run](../../../../glossary.md#term-run) creates `MPSGraphTensorData` [bindings](../../../../glossary.md#term-binding) directly over the ordered input and output
  `MTLBuffer` objects, using each retained [shape](../../../../glossary.md#shape) and `MPSDataTypeFloat32`. The input array follows
  stable feed order and the caller-supplied results array follows stable target order at the ABI;
  native code uses the stored permutations to present both arrays in executable-reported order to
  the SDK. Java passes the run-owned workspace's already populated native address segments
  directly; it allocates no Arena, array, collection, segment, or wrapper and performs no handle
  copy or address marshalling in <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [native code glossary](../../../../glossary.md#native-numerical-library-and-instruction-architecture); [SDK glossary](../../../../glossary.md#term-sdk); [Arena glossary](../../../../glossary.md#term-arena); [marshalling glossary](../../../../glossary.md#term-marshalling))</span>`executeBound()`. Shared assigned buffer positions are resolved
  once during finalization and cold binding, so repeated internal consumption of a value reuses
  its one MPSGraph tensor rather than adding another feed.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [tensor glossary](../../../../glossary.md#tensor))</span>
- The executable and [graph](../../../../glossary.md#graph) use `MPSGraphOptionsNone`; no implicit CPU-result synchronization
  option or hidden output materialization is requested.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [materialization glossary](../../../../glossary.md#materialization))</span>
- One fresh execution [descriptor](../../../../glossary.md#term-descriptor) has `waitUntilCompleted = YES` and a completion handler that
  captures a nullable execution <span data-glossary-navigation="true"> ([captures glossary](../../../../glossary.md#term-capture-meanings))</span>`NSError` inside the native call. The synchronous executable
  method is invoked on the context command queue exactly once.<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous); [command queue glossary](../../../../glossary.md#term-command-queue))</span>
- A nil input or result tensor-data object, input/results array, or execution descriptor maps
  exactly to `EXECUTION_FAILED`. The call returns `OK` only after the synchronous method returns,
  the completion handler reports no error, and the returned results count equals `target_count`
  with no nil result. A completion `NSError`, nil run result, malformed count, or nil returned
  result maps to <span data-glossary-navigation="true"> ([run result glossary](../../../../glossary.md#run-result--runresult))</span>`EXECUTION_FAILED`; an [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) exception maps to `INTERNAL_ERROR`.
- Success means the supplied output buffers contain their ordered results. Synaptik performs no
  explicit subsequent device-to-device or host copy and requests no hidden output
  materialization. This does not claim that MPSGraph internally uses no temporary storage. Native
  autorelease pools and framework wrapper allocation are bounded per invocation; there is no
  per-element allocation, graph compilation, route choice, map lookup, reflection, retry, or
  fallback.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [element glossary](../../../../glossary.md#term-element); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- The synchronous wait is the boundary after the one compiled maximal region. No wait or host
  copy occurs between its nodes. General asynchronous
  completion and cross-region event chaining remain out of scope.<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node); [asynchronous glossary](../../../../glossary.md#term-asynchronous); [scope glossary](../../../../glossary.md#term-scope))</span>

## Zero-copy and minimal-copy semantics

- Each caller-owned host value is copied at most once into its caller-owned Metal input buffer
  before Runtime execution. Runtime then borrows those Metal representations for the run.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [Metal glossary](../../../../glossary.md#term-metal); [buffer glossary](../../../../glossary.md#term-buffer); [Runtime glossary](../../../../glossary.md#term-runtime); [run glossary](../../../../glossary.md#term-run))</span>
- Each eligible compile-time splat feed is expanded and uploaded exactly once into a newly
  allocated run-owned Metal buffer during that run's cold state creation through<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>
  `InitializedBuffer`. It is not uploaded during [prepare](../../../../glossary.md#prepare) and not touched by hot execution.
- The native executable reads the assigned input `MTLBuffer` instances directly.
- Each boundary output is a fresh run-owned Metal representation created for its assigned output
  slot. Synaptik supplies those `MTLBuffer` destinations in the caller-supplied results array and
  performs no explicit post-execution copy or requested hidden output materialization.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- There is no intermediate host representation or Synaptik-requested output copy inside the
  compiled region. MPSGraph remains free to use undocumented internal temporary storage.
- Adjacent Metal work would retain Metal residency, but this task accepts only one exact Metal
  partition and makes no general multi-region or mixed-owner composition claim.<span data-glossary-navigation="true"> ([residency glossary](../../../../glossary.md#residency); [partition glossary](../../../../glossary.md#partition); [owner glossary](../../../../glossary.md#term-owner))</span>
- The route waits once when its synchronous region completes. A later CPU consumer or explicit
  host observation would require a boundary transfer; this task has no CPU/Metal schedule and
  downloads only in the opt-in test while the publication lease remains open.<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous); [CPU glossary](../../../../glossary.md#term-cpu); [publication glossary](../../../../glossary.md#term-publication); [lease glossary](../../../../glossary.md#term-lease))</span>
- “Zero-copy” here means Synaptik requests no intermediate output materialization and performs no
  explicit copy after host upload and before optional host observation. It does not mean caller
  host storage aliases <span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>`MTLBuffer.contents` or that [MPSGraph](../../../../glossary.md#term-mpsgraph) never uses temporary storage.

## Out of [scope](../../../../glossary.md#term-scope)

- FLOAT16, BFLOAT16, FLOAT64, integer, BOOL, zero-element, scalar-rank, unresolved-layout,
  strided, view, offset, dynamic-shape, broadcast, or multi-output execution<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT64 glossary](../../../../glossary.md#term-float64); [BOOL glossary](../../../../glossary.md#data-type--datatype); [element glossary](../../../../glossary.md#term-element); [scalar glossary](../../../../glossary.md#term-scalar); [rank glossary](../../../../glossary.md#term-rank); [layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [shape glossary](../../../../glossary.md#shape))</span>
- any [operation](../../../../glossary.md#operation) other than [unary](../../../../glossary.md#term-unary) NEG
- custom Metal kernels, shader libraries, pipeline state, fusion, optimizer routes, or route
  candidate generation<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [kernels glossary](../../../../glossary.md#kernel); [shader glossary](../../../../glossary.md#term-shader); [fusion glossary](../../../../glossary.md#term-fusion); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- public [Engine](../../../../glossary.md#term-engine) composition, `CpuEngineBackendComposition` changes, automatic host input
  binding, standard Engine <span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [binding glossary](../../../../glossary.md#term-binding))</span>`compute`, mixed-[owner](../../../../glossary.md#term-owner) scheduling, [CPU](../../../../glossary.md#term-cpu) fallback, or [backend](../../../../glossary.md#backend) discovery
- a generic Metal integration/factory, public Metal storage, raw native handles, Objective-C or
  MPSGraph Java types, or a backend registry<span data-glossary-navigation="true"> ([Objective-C glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [registry glossary](../../../../glossary.md#term-registry))</span>
- general asynchronous API, futures, events, cancellation, timeout, cross-run overlap policy, or
  command-buffer exposure<span data-glossary-navigation="true"> ([asynchronous glossary](../../../../glossary.md#term-asynchronous); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [cancellation glossary](../../../../glossary.md#term-cancellation); [run glossary](../../../../glossary.md#term-run))</span>
- bounded [buffer](../../../../glossary.md#term-buffer) [pooling](../../../../glossary.md#term-pooling), representation reuse, [aliasing](../../../../glossary.md#term-aliasing), [lease](../../../../glossary.md#term-lease)-bound public [`MemorySegment`](../../../../glossary.md#term-memorysegment)
  views, persistent/shared prepared constant buffers, generic constant materialization, or
  whole-plan/local route autotuning<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [materialization glossary](../../../../glossary.md#materialization); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- native executable serialization, cache persistence, packaging, discovery, extraction, signing,
  notarization, publishing, Intel macOS, iOS, tvOS, or non-Apple platforms<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [cache glossary](../../../../glossary.md#term-cache); [persistence glossary](../../../../glossary.md#term-persistence); [notarization glossary](../../../../glossary.md#term-notarization))</span>
- architecture-contract, ADR, shared production-module, production dependency, or global
  integration-test changes; the exact test-only Compiler/conformance dependencies and dependency
  architecture-test update in scope do not change production direction<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Compiler glossary](../../../../glossary.md#term-compiler); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- legacy source or package reuse; the legacy branch remains read-only capability evidence only

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially core invariants, Planning,
  Prepare, Runtime, concrete backends, Metal, lifecycle, and dependency rules<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [backends glossary](../../../../glossary.md#backend); [Metal glossary](../../../../glossary.md#term-metal); [lifecycle glossary](../../../../glossary.md#lifecycle); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [ADR 0011: Per-run Runtime resource ownership](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [ownership glossary](../../../../glossary.md#term-ownership))</span>
- [ADR 0013: Prepared-execution persistent-resource lifecycle](../../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)
- [Metal strategy note](../../../../design/notes/metal-backend-strategy.md)
- [Metal master plan](../master-plan.md)
- [Metal 0001](0001-metal-capability-storage-and-native-foundation.md)
- [Compiler 0006B7](../../../modules/compiler/tasks/0006b7-final-neg-logical-layout-closure.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- [Runtime 0016](../../../modules/runtime/tasks/0016-persistent-prepared-resource-lifecycle.md)
- [Prepare 0006](../../../modules/prepare/tasks/0006-persistent-prepared-resource-finalization-transaction.md)
- [Engine 0010](../../../modules/engine/tasks/0010-prepared-handle-ownership-and-closure.md)<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>

## Architecture constraints

- [Planning](../../../../glossary.md#term-planning) evaluates only [occurrence](../../../../glossary.md#term-occurrence) capability and selects `owner = metal`; it does not name,
  select, configure, compile, or execute MPSGraph.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
- Metal analysis owns exact route validation, whole-partition lowering, stable boundary ordering,
  complete boundary input/output declarations, exact splat-source classification, and the one
  address-workspace declaration.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [validation glossary](../../../../glossary.md#term-validation); [lowering glossary](../../../../glossary.md#lowering))</span>
- Shared Prepare owns projection, staged invocation, slot assignment, validation, and
  transactional handoff but never inspects the Metal plan or native resource.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- Metal finalization compiles the executable only after assignment and cannot add a declaration,
  change route, or allocate per-run buffers.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- [Runtime](../../../../glossary.md#term-runtime) sees only one [immutable](../../../../glossary.md#term-immutable) [partition](../../../../glossary.md#partition) [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable), one [nominal](../../../../glossary.md#term-nominal)
  [`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource), ordinary
  representations, and a prepared schedule. It sees no operation, graph node, MPSGraph type,
  native handle, backend lookup, or route choice on the hot path.<span data-glossary-navigation="true"> ([prepared schedule glossary](../../../../glossary.md#prepared-schedule--preparedschedule); [operation glossary](../../../../glossary.md#operation); [graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [backend glossary](../../../../glossary.md#backend); [lookup glossary](../../../../glossary.md#term-lookup); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) is the unique persistent-resource owner. The executable may retain a direct
  reference for invocation but does not close it or create another ownership occurrence.<span data-glossary-navigation="true"> ([resource owner glossary](../../../../glossary.md#ownership-borrowing-and-lease); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Each run receives isolated output storage. The caller-owned input representation is borrowed;
  each splat input and output is fresh and run-owned; the result owns the published output lease
  under current Runtime rules. The run-owned workspace gives native address arrays an explicit
  close lifecycle absent from <span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [lease glossary](../../../../glossary.md#term-lease); [close lifecycle glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>[`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation).
- Engine remains the composition root, but no Engine source changes here. Metal does not depend
  on Engine, and this route is not presented as a supported public Engine composition.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [composition root glossary](../../../../glossary.md#term-composition-root))</span>
- No architecture authority changes are expected. Stop if implementation requires a shared
  contract, dependency direction, public Engine seam, mixed-backend rule, or ownership change.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.metal` — exact public capability truth only.
- `io.github.pho001.synaptik.backend.metal.internal` — package-private ABI, context, storage,
  MPSGraph persistent resource, route analysis/finalization, prepared executable, binding, and
  exact schedule assembly.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [binding glossary](../../../../glossary.md#term-binding))</span>

No package is added. The master-plan package map is narrowed for this first route: the planned
`.internal.prepare` and `.internal.route.mpsgraph` packages remain deferred because splitting this
small package-private vertical [slice](../../../../glossary.md#slice) would require [widening](../../../../glossary.md#term-widening) cross-package implementation
visibility before a second route proves a stable seam.

Type placement:

- `MetalMpsGraphExecutableResource` — package-private [`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource) owning one opaque native
  executable and context lease.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [lease glossary](../../../../glossary.md#term-lease))</span>
- `MetalNegAnalysisInputs` — package-private [immutable](../../../../glossary.md#term-immutable) context-bearing analysis inputs.
- `MetalNegPreparationPlan` — package-private immutable selected-route facts: exact device-context
  identity, partition DAG, stable value/tensor indices, ordered boundary source roles/descriptors/
  byte geometry, one address-workspace declaration, and no executable state.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [identity glossary](../../../../glossary.md#term-identity-meanings); [partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [tensor glossary](../../../../glossary.md#tensor); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- `MetalNegPartitionPreparer` — package-private complete maximal-NEG-partition analysis and
  declarations.
- `MetalNegPartitionFinalizer` — package-private assignment validation, native compilation,
  executable construction, local rollback, and atomic result handoff.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [rollback glossary](../../../../glossary.md#term-rollback); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- `MetalNegPreparedExecutable` — package-private immutable Runtime recipe; its bound invocation
  and run-owned closeable native-address workspace may be private/package-visible nested types in
  this same allowlisted path because neither is reused as a public contract.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [run glossary](../../../../glossary.md#term-run))</span>
- `MetalNegPreparedScheduleAssembler` — package-private sole-partition representation, execution,
  and publication recipe assembly.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>

No new public Java type, constructor, method, package, service, factory, [registry](../../../../glossary.md#term-registry), or [Engine](../../../../glossary.md#term-engine)-facing
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

[Conformance](../../../../glossary.md#term-conformance) and [dependency](../../../../glossary.md#term-dependency)-enforcement paths:

21. `testing/backend-conformance/build.gradle.kts`
22. `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/MetalNegCapabilityPartitionConformanceTest.java` (new)
23. `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/BackendConformanceDependencyContractTest.java`

Documentation and [planning](../../../../glossary.md#term-planning) paths:

24. `docs/backend-guide/metal-backend.md`
25. `docs/glossary.md`
26. this task
27. `docs/planning/backends/metal/master-plan.md`
28. `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

At most the exact 28 paths above: the [Metal](../../../../glossary.md#term-metal) build file, 13 Metal Java production/[Javadoc](../../../../glossary.md#term-javadoc) paths,
three native-[layout](../../../../glossary.md#layout) paths, three Metal test paths, two [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) paths, one architecture-
test path, two explanatory documentation paths, and three [planning](../../../../glossary.md#term-planning) paths.

This exceeds the usual source-file guardrail because the first route is an indivisible
Java/native vertical [slice](../../../../glossary.md#slice): capability, staged analysis/finalization, persistent [ownership](../../../../glossary.md#term-ownership-meanings),
cold [binding](../../../../glossary.md#term-binding), hot invocation, schedule integration, deterministic fakes, real native evidence,
and documentation must agree before any capability becomes true. The separate small types avoid
a route-owning god class while keeping every implementation detail package-private.

The only Metal build change is the test-only [Compiler](../../../../glossary.md#term-compiler) [dependency](../../../../glossary.md#term-dependency), and the native build script owns
framework linkage. The only conformance/build enforcement changes are paths 21--23. Do not edit
shared production Java, [Engine](../../../../glossary.md#term-engine), [CPU](../../../../glossary.md#term-cpu), build logic, architecture authority, or integration-test
source. If another path, package, public type, production [module](../../../../glossary.md#term-module) edge, or shared contract is
required, stop and return the task to planning.

## Acceptance criteria

- Capability is true exactly for the specified NEG occurrence domain and false for all other
  kinds, attributes, types, shapes, ranks, zero dimensions, and layouts. It neither receives nor
  infers source provenance and therefore gives the same answer for equal eligible descriptors
  whether the input is caller-bound or an explicit compile-time splat.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [shapes glossary](../../../../glossary.md#shape); [dimensions glossary](../../../../glossary.md#dimension); [layouts glossary](../../../../glossary.md#layout); [source provenance glossary](../../../../glossary.md#term-source-provenance); [descriptors glossary](../../../../glossary.md#term-descriptor); [compile glossary](../../../../glossary.md#compile))</span>
- Metal analysis accepts every arbitrary non-empty matching maximal partition, uses only the
  supplied partition-local DAG/value/logical-requirement facts, handles chains, independent
  nodes, fan-out, repeated consumption, internal values, multiple ordered boundary inputs and
  outputs, and differing positive static shapes, declares each unique boundary in the specified
  stable order with exact geometry, classifies caller versus exact FLOAT32 splat feeds, and
  declares exactly one checked address workspace.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [nodes glossary](../../../../glossary.md#node); [fan-out glossary](../../../../glossary.md#term-fan-out); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>
- Automated topology cases include a multi-node chain with an internal value, independent NEG
  nodes of different valid shapes, two consumers of one input or intermediate, multiple graph
  inputs/outputs, and logical requirements representing values entering from or leaving for
  another partition. The backend-local executable integration remains a sole all-Metal graph so
  it does not imply unsupported mixed-owner schedule composition.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend); [owner glossary](../../../../glossary.md#term-owner))</span>
- Preparation rejects only occurrence-domain exclusions or malformed/inconsistent handoff facts;
  no supported topology rejection, fallback, repartitioning, hidden CPU execution, or Planning
  route logic is introduced.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff); [CPU glossary](../../../../glossary.md#term-cpu); [Planning glossary](../../../../glossary.md#term-planning))</span>
- The plan retains the exact analysis `MetalDeviceContext`; finalization and schedule assembly
  reject an identity mismatch before resource acquisition or recipe construction. Finalization
  validates exact declaration/assignment identity, including the address workspace, and compiles
  one reusable native executable only after slots exist.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Provisional context lease acquisition is atomic with context owner close. Every create/wrapper
  failure releases the executable when present and then the provisional lease exactly once;
  every finalizer-local failure after resource acquisition closes it once; all paths preserve the
  specified primary and distinct suppressed-failure order. Successful return lists exactly the
  adopted resource once.<span data-glossary-navigation="true"> ([lease glossary](../../../../glossary.md#term-lease); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- Shared [Prepare](../../../../glossary.md#prepare) transfers the resource to one [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution); close/run races use Runtime
  0016 unchanged, and repeated schedule occurrences do not duplicate ownership.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [Runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [ABI](../../../../glossary.md#term-application-binary-interface-abi) version is exactly `2`; the dylib exports exactly ten required symbols; the original seven
  signatures/semantics remain intact; the three added signatures, statuses, output-cell rules,
  ownership, validation, and error mapping match this specification.<span data-glossary-navigation="true"> ([dylib glossary](../../../../glossary.md#term-dylib); [signatures glossary](../../../../glossary.md#term-signature); [validation glossary](../../../../glossary.md#term-validation))</span>
- Java resolves and binds all ten symbols eagerly after validating version `2`, in documented
  order, before context creation. Malformed native create results are cleaned up once.
- One whole-partition prepared recipe executes twice without recompilation. Each run uses the
  exact ordered caller inputs, fresh initialized splat buffers, fresh ordered output buffers, and
  a fresh closeable address workspace selected during cold setup/binding.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [binding glossary](../../../../glossary.md#term-binding))</span>
- Native execution binds input and result `MPSGraphTensorData` directly over assigned
  `MTLBuffer` objects. Output bytes prove writing into the supplied destinations with no explicit
  Synaptik post-execution copy or requested hidden output materialization; no claim is made about
  MPSGraph internal temporary storage.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
- The real route preserves FLOAT32 NEG semantics including finite values, infinities, NaN sign
  irrelevance, and signed zero (<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [signed zero glossary](../../../../glossary.md#term-signed-zero))</span>`+0.0 -> -0.0`, `-0.0 -> +0.0`) with raw-bit assertions where
  applicable.
- Deterministic fake/native-seam tests prove compilation-object, tensor-data/descriptor, alias,
  completion error, malformed result, Objective-C exception, and all other status mappings.
  Completion/run failures map to stable status <span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [tensor glossary](../../../../glossary.md#tensor); [alias glossary](../../../../glossary.md#aliasing-and-arena); [Objective-C glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary))</span>`11`, input/output [aliasing](../../../../glossary.md#term-aliasing) maps to `10`,
  compilation-object failures map to `9`, exceptions map to `7`, and Java retains operation/raw
  status with no retry or fallback. The real-device test need not induce an artificial failure.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [device glossary](../../../../glossary.md#term-device))</span>
- Deterministic tests prove exact context identity retention and finalizer/assembler mismatch
  rejection; Java-side cardinality/rank/padding/index/producer/Shape/geometry rejection before
  downcall; independent native rejection of matching safely inspectable malformed facts;
  caller/splat feed ordering and repeated consumption; exact FLOAT32 splat bits including signed
  zero, infinities, and NaN payload preservation; one fresh allocation/upload per splat per run;
  initialized validity, rollback/cleanup, concurrent-run isolation, and no prepared/shared
  constant buffer; and address-workspace identity, cold marshalling, exact close, and absence of
  hot Java allocation/marshalling.<span data-glossary-navigation="true"> ([cardinality glossary](../../../../glossary.md#term-cardinality); [rank glossary](../../../../glossary.md#term-rank); [padding glossary](../../../../glossary.md#padding); [producer glossary](../../../../glossary.md#term-producer); [downcall glossary](../../../../glossary.md#term-downcall); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [rollback glossary](../../../../glossary.md#term-rollback); [constant glossary](../../../../glossary.md#term-constant); [marshalling glossary](../../../../glossary.md#term-marshalling))</span>
- Runtime hot execution performs no graph compilation, operation inspection, slot lookup,
  representation cast, map lookup, reflection, string dispatch, route selection, allocation of
  Java collections, host copy, or per-element allocation.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [dispatch glossary](../../../../glossary.md#term-dispatch); [host glossary](../../../../glossary.md#term-host); [element glossary](../../../../glossary.md#term-element))</span>
- The only wait is the [synchronous](../../../../glossary.md#term-synchronous) compiled-region boundary. `MPSGraphOptionsSynchronizeResults`
  is not enabled.
- Each caller host input requires exactly one explicit upload before execution. Each eligible
  splat is uploaded once per run by its `InitializedBuffer` creator during cold state creation,
  never during prepare or hot execution. Output download occurs only in the test's observable
  host-consumption step while the result lease is open.
- Concurrent runs use isolated buffers and may reuse the immutable executable. Tests use bounded
  latches, not sleeps, for Java lifecycle races; the native route makes no throughput or overlap
  claim.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [lifecycle glossary](../../../../glossary.md#lifecycle); [throughput glossary](../../../../glossary.md#term-throughput))</span>
- At task completion, the public Java surface was `MetalCapabilityProvider` only. No Objective-C,
  MPSGraph, native handle, storage, preparer, finalizer, executable, or context type became public.
- No Engine source changes and no public standard-compute claim occurred. Documentation stated
  that the then-current Engine composition was CPU-only and mixed-owner execution was absent.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- `backends/metal` added [Compiler](../../../../glossary.md#term-compiler) only as `testImplementation`; its production dependencies and
  direction remained unchanged. Backend conformance proved public capability/maximal-partition
  closure, and the architecture test locked <span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [architecture test glossary](../../../../glossary.md#term-architecture-test))</span>`testImplementation(project(":backends:metal"))`
  in backend conformance and `testImplementation(project(":modules:compiler"))` in Metal while
  rejecting production forms of both edges and continuing to reject Engine and OpenBLAS-provider
  edges. Generic Engine integration tests remained unchanged because no supported Metal Engine
  composition existed at this task's completion frontier.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- Focused fake-native tests, ordinary Metal tests, native build/symbol/link inspection, real
  native prepared execution, Javadoc, Markdown, exact scope/status/order, staged-file, and
  whitespace validation all pass.<span data-glossary-navigation="true"> ([symbol glossary](../../../../glossary.md#term-symbol); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- A separate clean documentation-focused pass finalizes every affected Javadoc, Metal guide,
  native README, glossary impact, task evidence, master plan, and roadmap before completion.

## Performance constraints and evidence

- Compilation, shape conversion, graph construction, target selection, and resource acquisition
  happen only during finalization.<span data-glossary-navigation="true"> ([Compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [shape glossary](../../../../glossary.md#shape); [graph glossary](../../../../glossary.md#graph); [target glossary](../../../../glossary.md#term-target))</span>
- Cold binding writes ordered buffer addresses once into a run-owned closeable workspace. Bound
  execution retains direct resource/workspace references and makes one native call. There is no
  Java allocation, address-array marshalling, lookup, cast, graph traversal, operation dispatch,
  or route decision in <span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [buffer glossary](../../../../glossary.md#term-buffer); [run glossary](../../../../glossary.md#term-run); [marshalling glossary](../../../../glossary.md#term-marshalling); [lookup glossary](../../../../glossary.md#term-lookup); [operation glossary](../../../../glossary.md#operation); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>`executeBound()`.
- Native invocation may create the bounded framework tensor-data/array/descriptor wrappers
  required by the SDK, but performs no allocation proportional to element count and requests no
  hidden intermediate output materialization. No claim is made about MPSGraph internal temporary
  storage.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor); [descriptor glossary](../../../../glossary.md#term-descriptor); [SDK glossary](../../../../glossary.md#term-sdk); [element glossary](../../../../glossary.md#term-element); [materialization glossary](../../../../glossary.md#materialization); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
- The route performs exactly one region submission and one completion wait per invocation.
- The implementation must inspect native source and Java bytecode/source for these properties.
  No latency or throughput threshold is required for this first correctness route, and no
  performance superiority claim may be made.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [latency glossary](../../../../glossary.md#term-latency); [throughput glossary](../../../../glossary.md#term-throughput))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation development may [run](../../../../glossary.md#term-run) focused classes. After executable Java and native source
stabilize, run one final ordinary [Metal](../../../../glossary.md#term-metal) [module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :backends:metal:test
```

Build and inspect the [native library](../../../../glossary.md#term-native-library):

```bash
./native/metal-macos-arm64/build.sh
file native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
otool -L native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
```

The [symbol](../../../../glossary.md#term-symbol) inspection must show exactly the ten specified `synaptik_metal_*` exports. Link
inspection must show Foundation, Metal, and MetalPerformanceShadersGraph.

On the current macOS [arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary) [host](../../../../glossary.md#term-host), run the exact opt-in native cases with the freshly built [dylib](../../../../glossary.md#term-dylib):

```bash
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew :backends:metal:test \
  --tests '*MetalFoundationTest.nativeFoundationRoundTrip' \
  --tests '*MetalNegPreparedExecutionTest.nativePreparedNegRoundTripAndReuse'
```

The task cannot become Complete by skipping the real-[device](../../../../glossary.md#term-device) case on this eligible host. If the
environment sandbox hides the Metal device, rerun the exact test with approved unsandboxed device
access and record both outcomes. If supplied result [binding](../../../../glossary.md#term-binding), executable reuse, or [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero)
semantics fail, keep the capability [fail-closed](../../../../glossary.md#term-fail-closed) and return the task to [planning](../../../../glossary.md#term-planning). Failure-status
mapping is gated by deterministic fake/native-seam tests, not by an unspecified artificial
real-device [MPSGraph](../../../../glossary.md#term-mpsgraph) failure.

Run the narrow [conformance](../../../../glossary.md#term-conformance) and [dependency](../../../../glossary.md#term-dependency)-enforcement suites necessitated by the new [backend](../../../../glossary.md#backend)
behavior and test-only conformance edge:

```bash
./gradlew :testing:backend-conformance:test \
  --tests '*MetalNegCapabilityPartitionConformanceTest'
./gradlew :testing:architecture-tests:test \
  --tests '*BackendConformanceDependencyContractTest'
```

Inspect the compiled/public/internal [shape](../../../../glossary.md#shape) and [hot path](../../../../glossary.md#term-hot-path):

```bash
javap -classpath backends/metal/build/classes/java/main -p \
  io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider \
  io.github.pho001.synaptik.backend.metal.internal.MetalNegPreparedExecutable
```

Manually confirm no new public Metal type, no forbidden [Engine](../../../../glossary.md#term-engine) import, no native handle accessor,
and a direct [bound invocation](../../../../glossary.md#bound-invocation--boundinvocation). Stable recurring shape/capability rules belong in automated tests;
the manual inspection addresses only package/private and hot-path risks not represented by
ordinary tests.

The native-backed Metal module test is the truthful typed backend-local prepared integration
evidence: its test-only [Compiler](../../../../glossary.md#term-compiler) edge lets it construct `CompileArtifacts` and traverse shared
[`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation), [Runtime](../../../../glossary.md#term-runtime) representation creation/cold binding/execution/[publication](../../../../glossary.md#term-publication), and native
MPSGraph without [widening](../../../../glossary.md#term-widening) production dependencies. `ARCHITECTURE.md` expects backend behavior
changes to update backend conformance, so the narrow public-surface conformance test proves exact
NEG capability and that adjacent eligible occurrences form one maximal Metal [partition](../../../../glossary.md#partition); it does
not reach package-private execution or create a public Metal integration seam. The focused
[architecture test](../../../../glossary.md#term-architecture-test) reads both affected build files and enforces that their new edges are test-only.

`testing/integration-tests` exercised the supported end-to-end Engine composition at task 0002's
completion frontier. That composition was [CPU](../../../../glossary.md#term-cpu)-only, so adding a Metal test there would have
required out-of-[scope](../../../../glossary.md#term-scope) Engine work and would have falsely implied public Metal support at that
time. It remained unchanged for a concrete architecture reason, not as a waiver of backend
validation.

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
trailing whitespace, exact 28-path scope, package/type placement, [ABI](../../../../glossary.md#term-application-binary-interface-abi)/version/status/symbol
consistency, task/master/roadmap status and order, no staged files, and no later detailed Metal
task specification.

Repository-wide Java validation is deferred to the Metal capability [checkpoint](../../../../glossary.md#term-checkpoint) or [continuous
integration](../../../../glossary.md#term-continuous-integration-ci) because the production change remains inside one [concrete backend](../../../../glossary.md#concrete-backend) and changes no
shared contract or production edge. The focused architecture test above is required because the
conformance build's exact test dependency list changes; a full architecture or repository suite
would duplicate unrelated coverage and is not promised by this task.

## Dependencies

- Metal 0001 — Complete; supplies the native context, shared-storage buffers, host byte access,
  child leases, failure model, and current seven-symbol ABI to evolve.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [buffers glossary](../../../../glossary.md#term-buffer); [host glossary](../../../../glossary.md#term-host); [symbol glossary](../../../../glossary.md#term-symbol); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Compiler 0006B7 — Complete; supplies final canonical logical layouts only for fully static
  still-unresolved exact NEG outputs and directly consumed compile-time splat inputs before
  capability queries. Its initial focused 41-test run, pre-correction full Compiler
  40-suite/281-test run, Compiler Javadoc, independent documentation review, and corrected focused
  closure-test rerun passed.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [canonical glossary](../../../../glossary.md#term-canonical-form); [layouts glossary](../../../../glossary.md#layout); [compile glossary](../../../../glossary.md#compile); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Runtime](../../../../glossary.md#term-runtime) 0016 — Complete; supplies unique persistent-resource [ownership](../../../../glossary.md#term-ownership) and close/[run](../../../../glossary.md#term-run) [leases](../../../../glossary.md#term-lease).
- Prepare 0006 — Complete; supplies finalizer-local obligation, shared transactional rollback,
  and ownership transfer.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span><span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Engine 0010 — Complete; supplies outward prepared-handle closure for a future composition. It
  is a lifecycle prerequisite, not an authorization to change Engine here.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Installed macOS [SDK](../../../../glossary.md#term-sdk) 27.0 headers and frameworks plus [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26 [FFM](../../../../glossary.md#term-ffm).

[Model](../../../../glossary.md#term-model) 0026 is not a [dependency](../../../../glossary.md#term-dependency) because this task supports only existing [`FLOAT32`](../../../../glossary.md#term-float32). No tuning,
custom-[kernel](../../../../glossary.md#kernel), packaging, or mixed-[owner](../../../../glossary.md#term-owner) prerequisite applies.

## Follow-up tasks

- [Metal](../../../../glossary.md#term-metal) 0003 remains Draft for custom Metal [kernels](../../../../glossary.md#kernel) and broader route work.
- Metal 0004 remains Draft for typed route candidates and [cache](../../../../glossary.md#term-cache) compatibility.
- A later separately planned composition task must decide how Engine receives Metal
  collaborations, uploads ordinary <span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>`HostTensorStorage`, materializes host results, and composes
  mixed owners. This task does not create that detailed specification.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host))</span>
- More operations/types/layouts, device selection, asynchronous event chaining, buffer pooling,
  and packaging remain future work. Whole-partition lowering for the supported NEG domain is
  complete task-0002 scope, not a follow-up.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [device glossary](../../../../glossary.md#term-device); [asynchronous glossary](../../../../glossary.md#term-asynchronous); [buffer glossary](../../../../glossary.md#term-buffer); [pooling glossary](../../../../glossary.md#term-pooling); [partition glossary](../../../../glossary.md#partition); [lowering glossary](../../../../glossary.md#lowering); [scope glossary](../../../../glossary.md#term-scope))</span>

## Architecture impact

Expected impact: None.

The task implements responsibilities already assigned to [Metal](../../../../glossary.md#term-metal) and uses the accepted staged
[Prepare](../../../../glossary.md#prepare) and persistent-resource [lifecycle](../../../../glossary.md#lifecycle) exactly. It changes no authority, [ownership](../../../../glossary.md#term-ownership-meanings) boundary,
[dependency direction](../../../../glossary.md#java-platform-and-build-tools), or shared [API](../../../../glossary.md#term-application-programming-interface-api). If implementation requires an architecture, shared-[module](../../../../glossary.md#term-module),
[Engine](../../../../glossary.md#term-engine), [dependency](../../../../glossary.md#term-dependency), or package-boundary change, stop and report the exact conflict instead of
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

Stop and return the task to [planning](../../../../glossary.md#term-planning) if any of these occurs:

- the [SDK](../../../../glossary.md#term-sdk) does not write the supplied result [tensor](../../../../glossary.md#tensor) data directly or cannot reuse one executable;
- [synchronous](../../../../glossary.md#term-synchronous) completion cannot report native execution failure deterministically;
- the route needs graph compilation, allocation proportional to element count, a host copy, or
  route selection during hot execution;<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [element glossary](../../../../glossary.md#term-element); [host glossary](../../../../glossary.md#term-host))</span>
- finalization cannot return the executable as one [identity](../../../../glossary.md#term-identity-meanings)-unique [`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource);
- the whole-partition route needs a workspace beyond the one exact declared address workspace or
  changes declarations after assignment;<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition))</span>
- [Runtime](../../../../glossary.md#term-runtime) [binding](../../../../glossary.md#term-binding) would require raw `Object`, unchecked casts, a native auxiliary binding owner,
  lookup, or a shared-contract change;<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- a public Metal type other than the provider, Engine source, mixed-owner composition, production
  module dependency, Gradle change beyond the exact test-only edges, new package, architecture
  change, or twenty-ninth path is required;
  or<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Engine glossary](../../../../glossary.md#term-engine); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- the unsandboxed real-device test cannot prove two-run reuse, supplied output destinations, and
  exact FLOAT32 NEG behavior, or deterministic seam tests cannot prove error mapping.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [run glossary](../../../../glossary.md#term-run); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>

## Local decisions

- Keep the first route in the existing `.internal` package. Seven focused package-private types
  make the lifecycle visible without widening implementation visibility before a second route
  proves a stable subpackage seam.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [widening glossary](../../../../glossary.md#term-widening))</span>
- Use one executable for the complete maximal Metal partition. Stable first-encounter value/feed
  order and producer-order targets cover chains, independent nodes, fan-out, repeated
  consumption, internal values, multiple boundaries, and differing valid shapes without
  repartitioning.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [partition glossary](../../../../glossary.md#partition); [producer glossary](../../../../glossary.md#term-producer); [targets glossary](../../../../glossary.md#term-target); [nodes glossary](../../../../glossary.md#node); [fan-out glossary](../../../../glossary.md#term-fan-out); [shapes glossary](../../../../glossary.md#shape))</span>
- Retain the exact analysis `MetalDeviceContext` by identity. Finalization and schedule assembly
  reject an equal-looking but distinct context before resource acquisition or creator execution.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Represent explicit positive-[rank](../../../../glossary.md#term-rank) [`FLOAT32`](../../../../glossary.md#term-float32) splats with per-[run](../../../../glossary.md#term-run) `InitializedBuffer` creators.
  Preparation retains typed scalar facts but allocates and uploads no constant buffer; each run
  gets fresh storage containing the exact raw bits.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [constant glossary](../../../../glossary.md#term-constant); [buffer glossary](../../../../glossary.md#term-buffer); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- Declare one per-run native-address workspace. Cold binding fills its feed-then-target pointer
  array once; the bound hot method retains direct slices and performs one downcall.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [downcall glossary](../../../../glossary.md#term-downcall))</span>
- Treat deterministic fake/native-seam coverage as the accepted failure-status matrix. The
  real-device gate proves successful execution, supplied destinations, and reuse without trying
  to manufacture undocumented MPSGraph failures.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [device glossary](../../../../glossary.md#term-device); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
- Compose evidence rather than claim an impossible public-port test. Public
  `GraphCompilationPort` plus shared [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) proves the real caller-input route;
  backend-local typed <span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>[`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) tests prove positive-rank splats; existing shared Prepare
  tests prove <span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>`ConstantSource -> PrepareContext.constants() -> InitializedBuffer`.

## Known limitations

- Capability remains limited to equal-[shape](../../../../glossary.md#shape)/equal-[gradient](../../../../glossary.md#term-gradient)-flag, positive fully static [rank](../../../../glossary.md#term-rank)
  `1..16`, resolved dense-[contiguous](../../../../glossary.md#term-contiguous) non-[view](../../../../glossary.md#term-view-meanings) zero-offset [`FLOAT32`](../../../../glossary.md#term-float32) [unary](../../../../glossary.md#term-unary) `NEG` occurrences.
- At task 0002 completion, the schedule assembler accepted one sole all-Metal partition and no
  public Metal Engine adapter or standard <span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [partition glossary](../../../../glossary.md#partition); [Engine glossary](../../../../glossary.md#term-engine))</span>`compute` composition existed. The task made no
  mixed-owner schedule, CPU fallback, or cross-region transfer claim.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- The public compilation port has no explicit positive-rank forward-constant ingress. The task
  therefore does not claim one public <span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [constant glossary](../../../../glossary.md#term-constant))</span>`GraphCompilationPort` positive-rank-splat test.
- Execution is synchronous and the native executable resource serializes its invocation boundary.
  No asynchronous, cancellation, timeout, cross-run throughput, or overlap guarantee is made.<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous); [asynchronous glossary](../../../../glossary.md#term-asynchronous); [cancellation glossary](../../../../glossary.md#term-cancellation); [run glossary](../../../../glossary.md#term-run); [throughput glossary](../../../../glossary.md#term-throughput))</span>
- No FLOAT16, BFLOAT16, FLOAT64, integer, BOOL, scalar-rank, zero-extent, dynamic-shape,
  strided/view/offset, broadcast, multi-output, custom-kernel, pooling, persistent constant,
  serialization, packaging, discovery, tuning, or performance-superiority capability is added.<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT64 glossary](../../../../glossary.md#term-float64); [BOOL glossary](../../../../glossary.md#data-type--datatype); [scalar glossary](../../../../glossary.md#term-scalar); [extent glossary](../../../../glossary.md#term-extent); [kernel glossary](../../../../glossary.md#kernel); [pooling glossary](../../../../glossary.md#term-pooling); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- MPSGraph may use internal temporary storage. The no-copy statement is limited to Synaptik not
  requesting hidden result materialization or performing an explicit post-execution copy.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [materialization glossary](../../../../glossary.md#materialization))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Clean implementation context stabilized exactly the first 20 paths in the task allowlist. Its
  final ordinary `./gradlew :backends:metal:test` run passed 35 tests with zero failures/errors
  and three expected opt-in skips. The focused prepared-execution run passed 20 tests with zero
  failures/errors and two expected opt-in skips. No executable Java or native source changed in
  the documentation-focused completion context, so these successful suites were not repeated.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- `./native/metal-macos-arm64/build.sh` passed. `file` identified the output as an arm64 Mach-O
  dynamic library. <span data-glossary-navigation="true"> ([arm64 glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary); [Mach-O glossary](../../../../glossary.md#term-mach-o); [dynamic library glossary](../../../../glossary.md#term-dynamic-library))</span>`nm -gU` showed exactly the ten required `synaptik_metal_*` exports, and
  `otool -L` showed Foundation, [Metal](../../../../glossary.md#term-metal), and MetalPerformanceShadersGraph linkage.
- The exact command
  `SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test --tests '*MetalFoundationTest.nativeFoundationRoundTrip' --tests '*MetalNegPreparedExecutionTest.nativePreparedNegRoundTripAndReuse'`
  passed outside the sandbox with both real-device tests executed and no skips.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>
- `./gradlew :testing:backend-conformance:test --tests '*MetalNegCapabilityPartitionConformanceTest'`
  passed two tests. `./gradlew :testing:architecture-tests:test --tests '*BackendConformanceDependencyContractTest'`
  passed one test. These were the required focused conformance and dependency gates; under the
  completion-time CPU-only Engine frontier, no generic Engine integration test applied.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [dependency glossary](../../../../glossary.md#term-dependency); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [integration test glossary](../../../../glossary.md#term-integration-test))</span>
- Manual source and `javap -p` inspection covered `MetalCapabilityProvider` and
  `MetalNegPreparedExecutable`: the provider was the only public Metal type delivered at that
  frontier; no native handle accessor or Engine import was introduced; the cold-bound invocation
  directly called<span data-glossary-navigation="true"> ([bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation))</span>
  `resource.run(...)`; and hot execution performs one native downcall with no Java address
  marshalling, allocation, lookup, or dispatch.<span data-glossary-navigation="true"> ([downcall glossary](../../../../glossary.md#term-downcall); [marshalling glossary](../../../../glossary.md#term-marshalling); [lookup glossary](../../../../glossary.md#term-lookup); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Evidence composition was reviewed against the source and tests. The real
  `nativePreparedNegRoundTripAndReuse` path uses `GraphCompilationPort`, [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation),
  Runtime, and MPSGraph with caller inputs. <span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>`positiveRankSplatsAreColdInitializedInStableOrderAndIsolatedAcrossRuns`
  covers typed [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) analysis/finalization/schedule/[Runtime](../../../../glossary.md#term-runtime) splats. Existing
  `GraphPreparationTest` coverage proves [constant](../../../../glossary.md#term-constant)-source projection into
  `PrepareContext.constants()` and rejects a constant not backed by `InitializedBuffer` or a
  non-constant backed by one.
- Documentation-focused completion context `/root` applied the General, API-Javadoc,
  Backend-Guide, and Planning profiles. It reviewed all 13 affected production/Javadoc paths,
  the Objective-C ABI and build script, the three Metal tests, conformance and architecture
  tests, SDK/strategy/ADR boundaries, and implementation evidence. It finalized both package
  Javadocs, the native README, Metal guide, glossary, this task, master plan, and roadmap without
  changing executable behavior.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Backend glossary](../../../../glossary.md#backend); [Planning glossary](../../../../glossary.md#term-planning); [Objective-C glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [SDK glossary](../../../../glossary.md#term-sdk); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- `./gradlew :backends:metal:javadoc` passed after final package-Javadoc edits. Targeted Markdown
  validation passed local file links, heading anchors and duplicate effective anchors, balanced
  fences, final newlines, and trailing-whitespace checks for the six Markdown files. ABI version,
  all twelve status values, exact ten symbols, package/type placement, task/master/roadmap status
  and ordering, and absence of a detailed later Metal task were consistent.
- Final scope validation found exactly the authorized 28 paths: the stabilized 20 implementation
  paths plus the eight documentation allowlist paths. <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git status --short -uall` showed no staged
  files. `git diff --check` and `git diff --cached --check` passed.

## Implementation notes

- [ABI](../../../../glossary.md#term-application-binary-interface-abi) version 2 preserves the original seven functions and statuses `0..7`, adds exactly the
  three executable functions and statuses `8..11`, eagerly resolves all ten symbols, and links
  MetalPerformanceShadersGraph in addition to Foundation and Metal.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>
- Analysis produces stable value indices, ordered unique feeds and targets, exact buffer geometry,
  typed optional splats, and one pointer workspace without physical allocation. Finalization
  validates assignment identity/order/geometry and compiles the persistent executable only after
  slots exist.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target); [buffer glossary](../../../../glossary.md#term-buffer); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Native compilation checks the SDK-reported feed/target tensors by identity and retains
  permutations between stable ABI order and framework order. Native execution binds<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [SDK glossary](../../../../glossary.md#term-sdk); [tensors glossary](../../../../glossary.md#tensor))</span>
  `MPSGraphTensorData` directly over ordered input and supplied output `MTLBuffer` instances,
  waits once at the region boundary, and validates completion plus returned results.
- The executable resource adopts one provisional context lease and implements exactly-once,
  idempotent release. <span data-glossary-navigation="true"> ([lease glossary](../../../../glossary.md#term-lease); [idempotent glossary](../../../../glossary.md#term-idempotent); [release glossary](../../../../glossary.md#term-release))</span>[`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) owns it across runs. Caller buffers are borrowed;
  splats, outputs, and address workspace are run-owned; published outputs remain leased through<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [run glossary](../../../../glossary.md#term-run))</span>
  [`RunResult`](../../../../glossary.md#run-result--runresult).
- Gradle changes are test-only: Metal tests depend on Compiler to construct public compile
  artifacts, and backend conformance depends on Metal. The focused architecture test rejects
  production versions of those edges. No production dependency or architecture rule changed.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [Compiler glossary](../../../../glossary.md#term-compiler); [compile artifacts glossary](../../../../glossary.md#compile-artifacts); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [architecture test glossary](../../../../glossary.md#term-architecture-test); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Completion summary

- Completed changes: implemented exact Metal capability truth, whole-maximal-partition NEG
  analysis/finalization, ABI-v2 MPSGraph compilation and synchronous execution, persistent
  executable ownership, per-run splat/output/address storage, direct supplied destinations,
  deterministic failure mapping, and focused conformance/dependency enforcement.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [partition glossary](../../../../glossary.md#partition); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [synchronous glossary](../../../../glossary.md#term-synchronous); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [run glossary](../../../../glossary.md#term-run); [conformance glossary](../../../../glossary.md#term-conformance); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Files changed or created: exactly the 28 paths listed in this task; no path outside the
  allowlist and no staged file.
- Tests and validation: final Metal suite had 35 total tests with three expected opt-in skips;
  focused execution had 20 total tests with two expected opt-in skips; native build, exact
  exports/framework links, both exact
  real-device tests outside the sandbox, two-test conformance, one-test architecture, hot-path
  inspection, Metal Javadoc, Markdown, scope, consistency, and whitespace gates passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [device glossary](../../../../glossary.md#term-device); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Documentation-agent review: clean documentation-focused context `/root` completed the required
  independent review without changing executable behavior or rerunning stable Java suites.
- Documentation impact: package summaries, native ABI/build guide, backend lifecycle guide,
  glossary, task evidence, master plan, and roadmap now describe the current capability and its
  boundaries.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Javadoc review: every changed production Javadoc was inspected against implementation/tests;
  member Javadocs remained accurate, while both stale package Javadocs were finalized.
- Glossary impact: capability-provider and physical-representation entries were corrected, and a
  reusable MPSGraph prepared-executable distinction was added.
- Unresolved issues: None within task [scope](../../../../glossary.md#term-scope).
- Follow-up required: None. Metal 0003 and 0004 remain Draft master-plan rows without detailed
  specifications.

Status: Complete
