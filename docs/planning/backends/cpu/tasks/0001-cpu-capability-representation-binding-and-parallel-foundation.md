# Task 0001: [CPU](../../../../glossary.md#term-cpu) Capability, Representation, [Binding](../../../../glossary.md#term-binding), and Parallel Foundation

## Status

Superseded

## Goal

Replace the [CPU](../../../../glossary.md#term-cpu) marker with the smallest truthful concrete-[backend](../../../../glossary.md#backend) foundation needed before any
CPU [operation](../../../../glossary.md#operation) route is implemented:

- one explicitly constructible CPU capability provider with the stable [backend identity](../../../../glossary.md#backend-identity--backendid) `cpu`;
- a [fail-closed](../../../../glossary.md#term-fail-closed) capability answer that advertises no executable operation semantics yet;
- one [run](../../../../glossary.md#term-run)-owned aligned native [off-heap](../../../../glossary.md#term-off-heap-memory) [`MemorySegment`](../../../../glossary.md#term-memorysegment) representation for internal CPU buffers
  and workspaces;<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer))</span>
- package-private cold [binding](../../../../glossary.md#term-binding) of [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) `MemorySegment` storage into direct,
  data-type-correct heap-carrier or exact-segment arguments;<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [carrier glossary](../../../../glossary.md#term-storage-carrier); [arguments glossary](../../../../glossary.md#term-argument))</span>
- a narrow package-private CPU prepared-executable/bound-invocation seam for later generated and
  native routes; and
- one package-private shared worker foundation for later scalar-parallel and Vector API-parallel
  kernels.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [Vector API glossary](../../../../glossary.md#term-vector-api); [kernels glossary](../../../../glossary.md#kernel))</span>

The result establishes [identity](../../../../glossary.md#term-identity-meanings), memory, binding, [lifetime](../../../../glossary.md#term-lifetime), threading, and failure contracts. It
does not claim that CPU can execute any [Model](../../../../glossary.md#term-model) operation.

## Rationale and [mental model](../../../../glossary.md#term-mental-model)

```text
Planning capability query
  -> CpuCapabilityProvider(cpu)
  -> false until a later task delivers and tests that exact semantic coverage

Prepare-selected Runtime geometry
  -> CPU creator callback
  -> shared Arena + aligned native MemorySegment
  -> run-owned CPU representation
  -> Runtime RunState owns cleanup orchestration

borrowed or run-owned CPU representation
  -> Runtime PreparedExecutable.bind(...)
  -> one checked CPU cold-binding pass
  -> direct typed heap-carrier or MemorySegment arguments
  -> later CPU BoundInvocation

later parallel CPU invocation
  -> shared CPU worker group
  -> bounded deterministic ranges
  -> synchronous completion or fail/cancel/interrupt propagation
```

Capability truth comes first: the current task delivers infrastructure but no computation, so
`supports` must return `false` for every non-null query. [Runtime](../../../../glossary.md#term-runtime) already owns representation
creation ordering, [rollback](../../../../glossary.md#term-rollback), per-[run](../../../../glossary.md#term-run) [ownership](../../../../glossary.md#term-ownership-meanings), validity, and cleanup orchestration. [CPU](../../../../glossary.md#term-cpu) owns only
its physical allocation/access/close mechanics and its checked [specialization](../../../../glossary.md#term-specialization) at the cold [binding](../../../../glossary.md#term-binding)
boundary. The worker group is [backend](../../../../glossary.md#backend)-private infrastructure, not a public executor or a route.

## [Scope](../../../../glossary.md#term-scope)

- Remove `CpuBackendModule`.
- Add one public final `CpuCapabilityProvider` implementing [Planning](../../../../glossary.md#term-planning)'s existing
  [`BackendCapabilityProvider`](../../../../glossary.md#backend-capability-provider--backendcapabilityprovider).
- Define one public [immutable](../../../../glossary.md#term-immutable) [`BackendId`](../../../../glossary.md#backend-identity--backendid) [constant](../../../../glossary.md#term-constant) with exact value `cpu`, returned by exact
  reference from every provider instance.
- Validate `supports(query)` with `NullPointerException("query")`, then return `false` without
  inspecting, switching on, or retaining the operation occurrence.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Add package documentation that distinguishes delivered foundation from later semantic coverage.
- Add package-private CPU buffer and workspace representation implementations under the CPU
  execution package.
- Allocate every [run](../../../../glossary.md#term-run)-owned internal [buffer](../../../../glossary.md#term-buffer)/workspace with `Arena.ofShared()` and
  `Arena.allocate(byteSize, byteAlignment)` so later CPU worker threads and Foreign Function and
  Memory (FFM) calls can use the same native segment with at least the requested alignment.<span data-glossary-navigation="true"> ([Foreign Function and Memory glossary](../../../../glossary.md#application-binary-interface-and-native-bridge); [FFM glossary](../../../../glossary.md#term-ffm); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- Treat zero-byte geometry as a real run-owned representation with a live shared arena and a
  zero-byte segment allocated using the declared alignment.
- Close the owning [arena](../../../../glossary.md#term-arena) exactly once through an [idempotent](../../../../glossary.md#term-idempotent) representation close [operation](../../../../glossary.md#operation).
- Add package-private [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) [CPU](../../../../glossary.md#term-cpu) buffer construction from `HostTensorStorage`, retaining the
  exact storage and exact [`MemorySegment`](../../../../glossary.md#term-memorysegment) while acquiring no [ownership](../../../../glossary.md#term-ownership-meanings) and making close a no-op.
- During cold binding, validate data type, exact byte size, liveness, current-thread
  accessibility, required writability, heap-carrier compatibility, and native alignment before
  constructing a direct typed argument.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [data type glossary](../../../../glossary.md#data-type--datatype); [liveness glossary](../../../../glossary.md#term-liveness); [argument glossary](../../../../glossary.md#term-argument))</span>
- Recognize the six exact [heap](../../../../glossary.md#term-heap) [carrier](../../../../glossary.md#term-storage-carrier) mappings already used by [Model](../../../../glossary.md#term-model):
  `FLOAT64 -> double[]`, `FLOAT32 -> float[]`, `BFLOAT16 -> short[]`, `INT32 -> int[]`,
  `INT64 -> long[]`, and `BOOL -> byte[]`.
- Preserve an observable heap carrier's carrier-relative byte offset and exact byte length so
  slices bind without copying. When the exact matching primitive carrier is not observable,
  retain the exact selected segment or slice without copying; this includes both genuine native
  segments and JDK 26 read-only heap segments whose <span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>`heapBase()` is empty.
- Permit each selection independently to use a typed array argument or an exact-segment argument,
  including mixed input/output signatures. Perform no materialization merely because argument
  forms or representation provenance differ.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [materialization glossary](../../../../glossary.md#materialization); [provenance glossary](../../../../glossary.md#provenance))</span>
- Add a package-private abstract CPU prepared-executable base over [Runtime](../../../../glossary.md#term-runtime)'s exact
  [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) contract. It performs only the CPU representation/type/access checks and
  hands fresh cold-binding arrays to a route-specific subclass, which must build a bound
  invocation retaining direct typed fields rather than those arrays.<span data-glossary-navigation="true"> ([bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation))</span>
- Add a package-private CPU workspace representation/access role for native shared-arena scratch.
- Add a package-private fixed worker group and range-body contract for later scalar and Vector
  routes.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [Vector glossary](../../../../glossary.md#term-vector))</span>
- Split one non-empty half-open element range into at most the configured worker count, with
  deterministic contiguous boundaries and no empty range. Empty work completes without dispatch.<span data-glossary-navigation="true"> ([half-open glossary](../../../../glossary.md#term-half-open-interval); [contiguous glossary](../../../../glossary.md#term-contiguous); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Allocate coordination only once per submitted parallel call, never per [element](../../../../glossary.md#term-element) or [graph](../../../../glossary.md#graph) [node](../../../../glossary.md#node).
- Define synchronous completion, cancellation at range boundaries, interruption, first/suppressed
  failure, worker shutdown, and idempotent close semantics.<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous); [cancellation glossary](../../../../glossary.md#term-cancellation))</span>
- Add focused tests for exact public shape, fail-closed capability, native allocation/alignment,
  zero size, lifetime and rollback compatibility, all six heap carriers, heap slices, native and
  mixed binding, access rejection, direct bound fields, range partitioning, cancellation,
  interruption, failure suppression, concurrent submissions, and close.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [lifetime glossary](../../../../glossary.md#term-lifetime); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Finalize affected Javadocs, CPU backend guidance, glossary impact, and planning records through
  the required separate clean documentation-focused pass.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- support or execution of any Tensor operation, operation family, metadata-only occurrence, view,
  scalar kernel, Vector API kernel, generated kernel, native kernel, or fused partition<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [metadata glossary](../../../../glossary.md#term-metadata); [occurrence glossary](../../../../glossary.md#term-occurrence); [scalar glossary](../../../../glossary.md#term-scalar); [kernel glossary](../../../../glossary.md#kernel); [Vector API glossary](../../../../glossary.md#term-vector-api); [fused glossary](../../../../glossary.md#specialization-and-fusion); [partition glossary](../../../../glossary.md#partition))</span>
- changing `supports` to `true` for any query, including zero-[element](../../../../glossary.md#term-element) or [view](../../../../glossary.md#term-view-meanings)-like occurrences
- `java.lang.classfile.CodeBuilder`, ASM use or replacement, class generation, hidden classes,
  method-handle entry points, generator schemas, generated-artifact keys, or caches<span data-glossary-navigation="true"> ([schemas glossary](../../../../glossary.md#term-schema); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- task 0002 or any other generated-kernel work before the separately authorized architecture
  synchronization recorded in the CPU master plan<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- CPU partition analysis, lowering, route candidates, route selection, representation-plan
  reconciliation, schedule assembly, finalization of a real operation, or a production executable<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
- OpenBLAS invocation integration, library discovery, provider fallback, thread-count mutation, or
  any oneMKL, oneDNN, Accelerate, AOCL, ZenDNN, Metal, or CUDA integration<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [oneMKL glossary](../../../../glossary.md#term-onemkl); [oneDNN glossary](../../../../glossary.md#term-onednn); [AOCL glossary](../../../../glossary.md#term-aocl); [ZenDNN glossary](../../../../glossary.md#term-zendnn); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda))</span>
- planning orchestration, ownership scoring, compiler behavior, Engine composition, public Tensor
  API, publication/output access, or benchmark/model-autotuning behavior<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [compiler glossary](../../../../glossary.md#term-compiler); [Engine glossary](../../../../glossary.md#term-engine); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [publication glossary](../../../../glossary.md#term-publication); [model glossary](../../../../glossary.md#term-model); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- a public CPU buffer, workspace, executor, scheduler, range, route, config-map, target-discovery,
  registry, manager, service-locator, or generic native-memory API<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [target glossary](../../../../glossary.md#term-target); [registry glossary](../../../../glossary.md#term-registry))</span>
- pooling, reuse, alias analysis, persistent prepared resources, mapped memory, pinning, NUMA
  policy, device memory, automatic materialization, packing, layout conversion, or coherence<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [alias glossary](../../../../glossary.md#aliasing-and-arena); [NUMA glossary](../../../../glossary.md#term-numa); [device glossary](../../../../glossary.md#term-device); [materialization glossary](../../../../glossary.md#materialization); [layout glossary](../../../../glossary.md#layout))</span>
- per-[operation](../../../../glossary.md#operation) [trace](../../../../glossary.md#trace) payloads or event emission; the current [Trace payload](../../../../glossary.md#trace-payload--tracepayload) families remain Draft
- Config, Planning, Prepare, Runtime, Backend Contract, Trace, Model, OpenBLAS provider, Engine,
  Gradle, dependency, architecture-contract, architecture-explanation, ADR, architecture-test,
  backend-conformance, or integration-test changes<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Backend glossary](../../../../glossary.md#backend); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- detailed specifications for [CPU](../../../../glossary.md#term-cpu) tasks 0002–0016

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core invariants
  - `modules/runtime`
  - `modules/prepare`
  - [Concrete backend](../../../../glossary.md#concrete-backend) [modules](../../../../glossary.md#term-module)
  - [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) routes
  - [Prepare](../../../../glossary.md#prepare) [lifecycle](../../../../glossary.md#lifecycle)
  - [Run](../../../../glossary.md#term-run) lifecycle
  - [Dependency](../../../../glossary.md#term-dependency) rules
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Tracing](../../../../architecture/tracing.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)<span data-glossary-navigation="true"> ([model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership); [binding glossary](../../../../glossary.md#term-binding))</span>
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)

## Architecture constraints

- Planning selects CPU ownership; CPU prepare will later select one implementation route. This
  task supplies only the Planning-owned capability-provider collaboration and cannot select a
  route.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [prepare glossary](../../../../glossary.md#prepare))</span>
- Capability is semantic ownership truth, not availability, registration, target discovery,
  preparation success, or execution readiness. Infrastructure alone justifies no <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>`true` answer.
- The authoritative contract still names CPU ASM and separately gates bytecode-generated CPU
  kernels on an architecture update. This task does not edit or reinterpret that contract, use
  ASM, use <span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [kernels glossary](../../../../glossary.md#kernel))</span>`CodeBuilder`, or create generated code.
- Concrete backends own physical representation implementations and allocation, release,
  transfer, and access mechanics. Runtime owns prepared geometry, per-run logical state,
  ownership transitions, validity/residency, rollback, and cleanup orchestration.<span data-glossary-navigation="true"> ([backends glossary](../../../../glossary.md#backend); [release glossary](../../../../glossary.md#term-release); [run glossary](../../../../glossary.md#term-run); [residency glossary](../../../../glossary.md#residency); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Caller inputs are borrowed for one run. CPU wrappers never close or extend a borrowed Model
  storage lifetime. Internal buffers and workspaces are run-owned and own their shared arena.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [Model glossary](../../../../glossary.md#term-model); [lifetime glossary](../../../../glossary.md#term-lifetime); [buffers glossary](../../../../glossary.md#term-buffer); [arena glossary](../../../../glossary.md#term-arena))</span>
- `HostTensorStorage` and `MemorySegmentStorage` remain unchanged borrowed Model contracts. CPU
  adds no alignment, allocation, route, or Runtime ownership meaning to them.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span>
- Prepare analysis declares exact byte size/alignment before slot assignment. CPU allocation
  consumes the final Runtime geometry only through backend creator callbacks; it does not infer
  geometry from a Tensor, graph value, or operation in the run path.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [graph value glossary](../../../../glossary.md#graph-value); [operation glossary](../../../../glossary.md#operation))</span>
- Backend finalization may construct immutable executable recipes but cannot allocate closeable
  physical resources. The native CPU allocation remains per-run representation creation.
- [Runtime](../../../../glossary.md#term-runtime)'s existing `PreparedExecutable.bind` is the only heterogeneous compatibility boundary.
  Later bound CPU invocations retain direct typed fields and perform no representation lookup,
  heap-base discovery, cast, data-type switch, route selection, or allocation in their hot call.<span data-glossary-navigation="true"> ([heterogeneous glossary](../../../../glossary.md#term-heterogeneous); [lookup glossary](../../../../glossary.md#term-lookup); [heap glossary](../../../../glossary.md#term-heap))</span>
- One [immutable](../../../../glossary.md#term-immutable) executable may bind concurrently to distinct [`RunState`](../../../../glossary.md#run-state--runstate) instances. A bound
  invocation remains non-thread-safe and cannot race its <span data-glossary-navigation="true"> ([bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [thread-safe glossary](../../../../glossary.md#term-thread-safe); [race glossary](../../../../glossary.md#term-race))</span>`RunState` closure.
- The CPU worker group owns workers and coordination. Generated classes and later kernels do not
  create pools, submit one task per element/node, or own scheduler lifecycle.<span data-glossary-navigation="true"> ([CPU worker group glossary](../../../../glossary.md#cpu-worker-group--cpuworkergroup); [element glossary](../../../../glossary.md#term-element); [node glossary](../../../../glossary.md#node); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- No dependency direction, public shared contract, module boundary, or architecture rule changes.
  If implementation requires one, stop and request a separate planning/architecture decision.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.backend.cpu` — replaces its marker with the only task-0001 public
  surface, `CpuCapabilityProvider`, plus package documentation.

Package added:

- `io.github.pho001.synaptik.backend.cpu.execution` — package-private CPU physical
  representations, cold typed arguments, prepared/bound route seam, and shared worker/range
  foundation. The package is implementation-facing inside the CPU backend and exports no public
  type.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [arguments glossary](../../../../glossary.md#term-argument); [backend glossary](../../../../glossary.md#backend))</span>

No [`buffer`](../../../../glossary.md#term-buffer), `storage`, `executor`, `service`, `manager`, [`registry`](../../../../glossary.md#term-registry), `config`, `route`, [`kernel`](../../../../glossary.md#kernel),
`generated`, or `native` public package is added. Keeping the closely collaborating internal
foundation in one package permits package-private visibility without manufacturing public
cross-package APIs.

## Exact public and package-private surface

The exact public task-0001 source surface is:

```java
package io.github.pho001.synaptik.backend.cpu;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;

public final class CpuCapabilityProvider implements BackendCapabilityProvider {
    public static final BackendId CPU_BACKEND_ID = new BackendId("cpu");

    public CpuCapabilityProvider();

    @Override
    public BackendId backendId();

    @Override
    public boolean supports(OperationCapabilityQuery query);
}
```

The constructor performs no discovery, allocation, registration, caching, or native loading.
`backendId()` returns `CPU_BACKEND_ID` by exact reference. `supports` checks `query` for null with
message `query` and otherwise returns `false`. The class has no [singleton](../../../../glossary.md#term-singleton) accessor, availability
[snapshot](../../../../glossary.md#term-snapshot), [device](../../../../glossary.md#term-device) [identity](../../../../glossary.md#term-identity-meanings), [target](../../../../glossary.md#term-target) probe, configuration, preparer, finalizer, executable factory,
route list, diagnostic-reason result, or [mutable](../../../../glossary.md#term-mutable) field.

All declarations in `io.github.pho001.synaptik.backend.cpu.execution` are package-private. The
implementation must contain the following narrow roles and source-level [shape](../../../../glossary.md#shape); a [signature](../../../../glossary.md#term-signature) change
requires updating this task before implementation begins:

```java
abstract class CpuBufferRepresentation implements BufferRepresentation {
    final DataType dataType();
    final long byteSize();
    final MemorySegment segment();
    final CpuBufferArgument argument();
    final boolean isClosed();
}

final class CpuBorrowedBuffer extends CpuBufferRepresentation {
    static CpuBorrowedBuffer borrow(HostTensorStorage storage);
}

final class CpuNativeBuffer extends CpuBufferRepresentation {
    static CpuNativeBuffer allocate(DataType dataType, long byteSize, long byteAlignment);
}

final class CpuNativeWorkspace implements WorkspaceRepresentation {
    static CpuNativeWorkspace allocate(long byteSize, long byteAlignment);
    long byteSize();
    long byteAlignment();
    MemorySegment segment();
    boolean isClosed();
}

sealed interface CpuBufferArgument {
    long byteOffset();
    long byteSize();
    boolean readOnly();

    record Doubles(double[] carrier, long byteOffset, long byteSize, boolean readOnly)
            implements CpuBufferArgument {}
    record Floats(float[] carrier, long byteOffset, long byteSize, boolean readOnly)
            implements CpuBufferArgument {}
    record Shorts(short[] carrier, long byteOffset, long byteSize, boolean readOnly)
            implements CpuBufferArgument {}
    record Ints(int[] carrier, long byteOffset, long byteSize, boolean readOnly)
            implements CpuBufferArgument {}
    record Longs(long[] carrier, long byteOffset, long byteSize, boolean readOnly)
            implements CpuBufferArgument {}
    record Bytes(byte[] carrier, long byteOffset, long byteSize, boolean readOnly)
            implements CpuBufferArgument {}
    record Segment(
            DataType dataType,
            MemorySegment segment,
            long byteSize,
            boolean readOnly) implements CpuBufferArgument {
        // byteOffset() is always zero because segment is the exact selected region.
    }
}

abstract class CpuPreparedExecutable extends PreparedExecutable {
    protected CpuPreparedExecutable(
            PreparedMemoryPlan memoryPlan,
            List<BufferSelection> bufferSelections,
            List<WorkspaceSelection> workspaceSelections,
            List<BufferAccess> bufferAccesses,
            List<DataType> bufferDataTypes);

    protected abstract BoundInvocation bindCpu(
            RunState runState,
            CpuBufferArgument[] bufferArguments,
            CpuNativeWorkspace[] workspaces);
}

@FunctionalInterface
interface CpuRangeBody {
    void execute(long startInclusive, long endExclusive, int rangeIndex);
}

final class CpuWorkerGroup implements AutoCloseable {
    CpuWorkerGroup(int workerCount);
    int workerCount();
    boolean isAccessibleByEveryWorker(MemorySegment segment);
    void execute(
            long startInclusive,
            long endExclusive,
            long minimumRangeSize,
            boolean deterministic,
            CpuRangeBody body);
    boolean isClosed();
    @Override public void close();
}

final class CpuParallelExecutionException extends RuntimeException {
    // Package-private construction only; exact construction follows the failure rules below.
}
```

Record [canonical](../../../../glossary.md#term-canonical-form) constructors and accessors may be declared explicitly for [validation](../../../../glossary.md#term-validation) and
[Javadoc](../../../../glossary.md#term-javadoc). `CpuBufferArgument.Segment` implements `byteOffset()` with [constant](../../../../glossary.md#term-constant) zero. Array variants
validate non-null [carrier](../../../../glossary.md#term-storage-carrier), non-negative offset/size, carrier-width [alignment](../../../../glossary.md#term-alignment), and in-carrier span.
The exact-segment variant validates non-null [data type](../../../../glossary.md#data-type--datatype)/segment, exact size, [liveness](../../../../glossary.md#term-liveness)/access, and
zero offset semantics. Its name does not assert native [provenance](../../../../glossary.md#provenance). No variant exposes `Object`, a
raw address, a storage-kind string/enum, or a mutation method.

`CpuPreparedExecutable` requires `bufferDataTypes.size()` to equal the [buffer](../../../../glossary.md#term-buffer)-selection count and
validates those entries after [Runtime](../../../../glossary.md#term-runtime)'s four constructor inputs. Its final Runtime hooks accept
only open `CpuBufferRepresentation` values and `CpuNativeWorkspace` values. For buffer selection
`i`, the retained data type must be `bufferDataTypes[i]`, byte size must equal the selected
`PreparedMemoryPlan.BufferEntry.byteSize()`, and the segment must satisfy that entry's alignment;
the aligned `BufferAccess` supplies writability rules. Workspaces must match their selected plan
entry's exact size and requested alignment. Only after every selection passes does the base copy
the already-classified direct `CpuBufferArgument` references and typed workspace references into
fresh cold arrays and invoke `bindCpu` once. The subclass must return a non-null Runtime
[`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation) for the exact state; Runtime's existing final association check remains
authoritative.

The implementation roles are:

- `CpuBufferRepresentation` — common [immutable](../../../../glossary.md#term-immutable) [metadata](../../../../glossary.md#term-metadata)/access base implementing Runtime
  `BufferRepresentation`; retains exact [`DataType`](../../../../glossary.md#data-type--datatype), byte size, and exact [`MemorySegment`](../../../../glossary.md#term-memorysegment).
- `CpuBorrowedBuffer` — non-owning wrapper over exact `HostTensorStorage`; close is a no-op and
  never closes, replaces, or prolongs the storage segment.
- `CpuNativeBuffer` — [run](../../../../glossary.md#term-run)-owned shared-[arena](../../../../glossary.md#term-arena) aligned native buffer with [idempotent](../../../../glossary.md#term-idempotent) close.
- `CpuNativeWorkspace` — run-owned shared-arena aligned native workspace implementing Runtime
  `WorkspaceRepresentation`, with the same allocation/close rules and no logical validity.
- `CpuBufferArgument` — sealed cold-bound argument family with one variant per six exact primitive
  array carriers plus one exact-segment variant. Array variants retain a direct array,
  carrier-relative byte offset, and exact byte length; the segment variant retains the exact
  selected segment or slice regardless of native or heap provenance. No variant owns memory.<span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed); [argument glossary](../../../../glossary.md#term-argument); [slice glossary](../../../../glossary.md#slice); [heap glossary](../../../../glossary.md#term-heap))</span>
- `CpuPreparedExecutable` — abstract [CPU](../../../../glossary.md#term-cpu) [specialization](../../../../glossary.md#term-specialization) of Runtime [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable); its
  final compatibility/binding hooks validate and convert selected CPU representations once, then
  call one package-private abstract route-specific binder.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup) — fixed package-private platform-worker owner with synchronous bounded-range
  dispatch and idempotent close.<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- `CpuRangeBody` — package-private functional contract receiving only primitive
  `startInclusive`, `endExclusive`, and deterministic `rangeIndex` values.
- `CpuParallelExecutionException` — package-private unchecked failure used only when interruption
  or worker-coordination failure cannot be propagated as the original unchecked worker failure.

Do not add a public representation/access type, a generic typed-buffer abstraction, a public
executor, an [operation](../../../../glossary.md#operation) [callback](../../../../glossary.md#term-callback), a route [registry](../../../../glossary.md#term-registry), a provider [facade](../../../../glossary.md#term-facade), or a second [backend](../../../../glossary.md#backend) ID
[owner](../../../../glossary.md#term-owner).

## Native representation [ownership](../../../../glossary.md#term-ownership-meanings), allocation, and failure semantics

- Internal [buffer](../../../../glossary.md#term-buffer) and workspace factories accept exact non-negative `byteSize` and positive
  power-of-two `byteAlignment`, matching [Runtime](../../../../glossary.md#term-runtime) [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) geometry rules.
- [Validation](../../../../glossary.md#term-validation) order is byte size, [alignment](../../../../glossary.md#term-alignment), then [arena](../../../../glossary.md#term-arena)/allocation work. Stable failures are
  `byteSize must be non-negative` and `byteAlignment must be a positive power of two`.
- Each factory creates a distinct `Arena.ofShared()` and allocates exactly one segment with the
  requested byte size and at least the requested alignment. The native address must be divisible
  by `byteAlignment`; a stronger platform/JDK alignment remains valid. The arena is shared because
  the representation may be accessed by the orchestrating thread, CPU workers, or an FFM call
  while its run remains open.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [CPU glossary](../../../../glossary.md#term-cpu); [FFM glossary](../../../../glossary.md#term-ffm); [run glossary](../../../../glossary.md#term-run))</span>
- Zero size follows the same path and yields a live zero-byte segment associated with its arena.
  It is not represented by `MemorySegment.NULL`, a global singleton, a heap array, or an absent
  optional.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [heap glossary](../../../../glossary.md#term-heap))</span>
- If arena creation or allocation fails, the original unchecked exception or error is preserved.
  An arena created before allocation failure is closed once; a distinct cleanup failure is
  suppressed, and the same exact primary object is not self-suppressed.
- Successful construction transfers physical cleanup to the representation. Runtime transfers
  run-owned cleanup responsibility only when complete [`RunState`](../../../../glossary.md#run-state--runstate) creation succeeds and already
  supplies reverse rollback/cleanup with first/suppressed failure rules.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>
- Representation `close` is idempotent and thread-safe. It marks the representation closed before
  closing its arena, calls the arena at most once, and propagates that close's unchecked failure
  unchanged. A failed close is not retried.<span data-glossary-navigation="true"> ([idempotent glossary](../../../../glossary.md#term-idempotent); [thread-safe glossary](../../../../glossary.md#term-thread-safe))</span>
- Access and cold [binding](../../../../glossary.md#term-binding) after close fail with `IllegalStateException("CPU representation is closed")`.
- Allocation neither initializes logical validity nor promises a numerical zero value. Runtime's
  `CreatedBuffer` starts invalid; an `InitializedBuffer` creator is responsible for materializing
  its logical value before return.
- No pooling, shared arena across representations, cleaner/finalizer fallback, global native
  allocator, address cache, manual free, or ownership transfer outside Runtime is introduced.<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [cache glossary](../../../../glossary.md#term-cache))</span>

## [Borrowed](../../../../glossary.md#ownership-borrowing-and-lease) storage and cold typed [binding](../../../../glossary.md#term-binding)

- `CpuBorrowedBuffer` accepts one non-null `HostTensorStorage`, checks its current liveness, and
  retains the exact storage and exact segment. It performs no copy, slice, reinterpretation,
  alignment upgrade, arena operation, or carrier allocation.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [alignment glossary](../../../../glossary.md#term-alignment); [arena glossary](../../../../glossary.md#term-arena); [operation glossary](../../../../glossary.md#operation))</span>
- Binding validates selections in Runtime's existing buffer-selection order. The executable's
  aligned <span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [buffer glossary](../../../../glossary.md#term-buffer))</span>`BufferAccess` decides whether a read-only segment is acceptable: `READ_ONLY` accepts
  it; `WRITE_ONLY` and `READ_WRITE` reject it.
- Every segment must be alive and accessible by the binding thread. A later parallel binder must
  additionally prove accessibility from every selected worker before dispatch; confined borrowed
  segments therefore cannot silently enter parallel execution.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- The segment byte size must equal the representation's retained byte size. Native internal
  segments must satisfy the retained declared alignment. Borrowed segments make no alignment
  promise beyond what a selected later route explicitly requires during its cold compatibility
  check.
- When `heapBase()` exposes a heap base, cold binding requires that object to be the exact
  primitive-array carrier mapped from [`DataType`](../../../../glossary.md#data-type--datatype). It retains the exact array plus the segment's
  carrier-relative byte offset and exact length. Misaligned offsets for that carrier fail binding.
- When `heapBase()` is unavailable, cold binding retains the exact selected segment or [slice](../../../../glossary.md#slice) in
  `CpuBufferArgument.Segment`. JDK 26 makes this path necessary for read-only heap segments as
  well as genuine native segments. <span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>`Segment.byteOffset()` is zero relative to the exact retained
  region. Binding does not extract/cache a raw address, copy bytes, or assert native provenance.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [provenance glossary](../../../../glossary.md#provenance))</span>
- A segment whose observable [heap](../../../../glossary.md#term-heap) [carrier](../../../../glossary.md#term-storage-carrier) conflicts with `DataType`, an unsupported observable
  heap base, a dead/inaccessible scope, a read-only write selection, an invalid offset/length, or
  a required alignment mismatch returns incompatibility through Runtime's checked binding hook so
  Runtime emits its existing stable indexed failure.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Mixed signatures are ordinary ordered selections. Each argument is specialized independently;
  no global heap/native mode, copying, materialization, or route choice occurs during binding.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [argument glossary](../../../../glossary.md#term-argument); [materialization glossary](../../../../glossary.md#materialization))</span>
- The route-specific binder may allocate the bound invocation and ordinary fixed-size binding
  state. It must copy required direct argument fields out of the fresh cold-binding arrays and
  must not retain those arrays, nominal Runtime representations, a <span data-glossary-navigation="true"> ([bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [nominal glossary](../../../../glossary.md#term-nominal))</span>[`RunState`](../../../../glossary.md#run-state--runstate) lookup path, or a
  generic object carrier for hot execution.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>
- Tests use non-computational probe invocations only to prove direct typed field retention. They
  must not add a scalar, Vector, native, or operation kernel.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [Vector glossary](../../../../glossary.md#term-vector); [kernel glossary](../../../../glossary.md#kernel))</span>

## Parallel worker, range, [cancellation](../../../../glossary.md#term-cancellation), and failure semantics

- [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup) is constructed with a positive worker count and owns exactly that many named
  daemon platform worker threads for its lifetime. It uses no common pool, virtual threads,<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime))</span>
  `parallelStream`, global executor, or externally supplied scheduler.
- One [synchronous](../../../../glossary.md#term-synchronous) [dispatch](../../../../glossary.md#term-dispatch) accepts a [half-open](../../../../glossary.md#term-half-open-interval) range `[startInclusive, endExclusive)`, a positive
  minimum elements per range, a determinism flag, and one non-null <span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element))</span>`CpuRangeBody`.
- Validation order is body, non-negative/ordered range, then positive minimum range size. Stable
  failures identify <span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation))</span>`body`, `startInclusive`, `endExclusive`, and
  `minimumRangeSize must be positive` respectively.
- An empty range returns without waking a worker or allocating per-range state.
- A non-empty range creates between one and `workerCount` contiguous non-empty ranges. The count
  is bounded by both worker count and the ceiling of element count over minimum range size.
  Boundaries use quotient/remainder partitioning, cover every element exactly once, and are
  deterministic from the four primitive inputs.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous))</span>
- Deterministic mode assigns and later exposes stable ascending `rangeIndex` values and does not
  authorize a nondeterministic reduction/combine order. Task 0001 implements no combine step;
  later reduction routes must consume completed range results in ascending index order.
- Non-deterministic mode may let workers claim the next range through one shared primitive cursor.
  It does not change boundaries or semantic eligibility.
- Dispatch allocates at most one call-level coordination object and arrays proportional to the
  bounded range/worker count. It allocates no object per element, Model node, loop iteration, or
  hot scalar/vector lane.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [node glossary](../../../../glossary.md#node); [scalar glossary](../../../../glossary.md#term-scalar); [vector lane glossary](../../../../glossary.md#simd))</span>
- Normal return occurs only after every dispatched range completes. At most one range executes at
  a time on one worker, while different ranges may run concurrently.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- The first observed unchecked worker failure or error becomes primary. It requests cancellation,
  prevents unclaimed ranges from starting, waits for already-running ranges, and receives every
  later distinct worker failure as suppressed in observation order. Repetition of the exact
  primary object is skipped to avoid self-suppression. The primary object is rethrown unchanged.
- Cancellation is cooperative at range boundaries. The worker group does not stop a Java thread,
  interrupt arbitrary route code, retry a range, or claim rollback of partially written output.
  Runtime's existing output-validity rules keep declared outputs invalid after backend failure.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- If the orchestrating thread is interrupted while waiting, dispatch records interruption,
  requests cancellation, waits until all already-running work quiesces, restores the thread's
  interrupt status, and throws `CpuParallelExecutionException("CPU parallel execution interrupted", cause)`.
  Distinct worker failures observed during quiescence are suppressed on that failure in
  observation order.<span data-glossary-navigation="true"> ([quiescence glossary](../../../../glossary.md#term-quiescence))</span>
- Concurrent dispatches are supported and isolated; worker capacity is shared without sharing
  call-level failure, cancellation, or range state. Reentrant dispatch from an owned worker is
  rejected with <span data-glossary-navigation="true"> ([Reentrant glossary](../../../../glossary.md#memory-management-atomic-counters-and-complexity))</span>`IllegalStateException("CPU worker must not submit parallel work")` to prevent
  pool starvation.
- `close` is [thread-safe](../../../../glossary.md#term-thread-safe) and [idempotent](../../../../glossary.md#term-idempotent). It rejects new dispatch with
  `IllegalStateException("CPU worker group is closed")`, requests cancellation of queued work,
  lets already-running range bodies reach their boundary, wakes every worker, joins the owned
  threads, and returns only after shutdown. Calling close from an owned worker is rejected with
  `IllegalStateException("CPU worker must not close its worker group")`.
- A dispatch cancelled by concurrent group close waits for its already-running ranges and fails
  with `CpuParallelExecutionException("CPU parallel execution cancelled by worker-group close")`
  unless an earlier worker failure is already primary. Later distinct failures are suppressed on
  the selected primary in observation order.
- An interruption while close waits follows the same quiesce-and-restore rule and throws one
  package-private unchecked shutdown failure. No worker survives a normally returning close.

## [Validation](../../../../glossary.md#term-validation) order and stable failures

Tests must lock at least these externally or cross-contract observable orders/messages:

1. `CpuCapabilityProvider.supports`: `query`, then unconditional `false`.
2. Native geometry: byte size, [alignment](../../../../glossary.md#term-alignment), [arena](../../../../glossary.md#term-arena) creation, allocation.
3. Cold executable binding: Runtime plan/open/range checks, CPU buffer selections in order,
   Runtime workspace selections in order, then route-specific bound-invocation construction.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [buffer glossary](../../../../glossary.md#term-buffer))</span>
4. CPU selection compatibility: representation kind, open/liveness/access, exact byte size,
   access writability, data-type carrier, offset/length, then route-specific alignment.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
5. Worker dispatch: body, range start/end, minimum range size, group-open/reentrancy state, then
   call-level allocation/dispatch.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [reentrancy glossary](../../../../glossary.md#term-reentrancy))</span>

Use [Runtime](../../../../glossary.md#term-runtime)'s existing indexed incompatibility failures rather than inventing a second public
exception vocabulary. Package-private [CPU](../../../../glossary.md#term-cpu) failures may be exact where tests need to lock cleanup,
interruption, shutdown, or invalid construction behavior.

## Affected files

Expected production paths:

- remove `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuBackendModule.java`
- add `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- add `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- add `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuBufferRepresentation.java`
- add `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuBorrowedBuffer.java`
- add `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuNativeBuffer.java`
- add `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuNativeWorkspace.java`
- add `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuBufferArgument.java`
- add `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPreparedExecutable.java`
- add `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuWorkerGroup.java`
- add `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuRangeBody.java`
- add `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuParallelExecutionException.java`
- add `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/package-info.java`

Expected test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderPublicShapeTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuNativeRepresentationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuBufferBindingTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuWorkerGroupTest.java`

Expected explanatory documentation:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`

Expected [planning](../../../../glossary.md#term-planning) paths:

- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless a concrete contradiction is found: authoritative/focused architecture, ADRs
0010/0011, documentation rules and profiles, [CPU](../../../../glossary.md#term-cpu) and prerequisite master plans/tasks, current
Planning/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)/[Backend](../../../../glossary.md#backend) Contract/[Trace](../../../../glossary.md#trace)/Config/[Model](../../../../glossary.md#term-model)/[OpenBLAS](../../../../glossary.md#term-openblas) production and test
contracts, Java 26 root/[module](../../../../glossary.md#term-module) [Gradle](../../../../glossary.md#term-gradle) configuration, architecture tests, backend-[conformance](../../../../glossary.md#term-conformance)
placeholder, and current [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26 [FFM](../../../../glossary.md#term-ffm)/threading [API](../../../../glossary.md#term-application-programming-interface-api) behavior.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 24 paths:

- 13 [CPU](../../../../glossary.md#term-cpu) production paths, counting removal of the marker;
- 6 CPU test paths;
- 2 explanatory documentation paths; and
- 3 [planning](../../../../glossary.md#term-planning) paths.

No `.gitkeep` needs removal. No other source, test, documentation, build, architecture, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), or
planning path may change. If implementation needs a public CPU type beyond
`CpuCapabilityProvider`, a CPU preparer/finalizer, a cross-package internal [API](../../../../glossary.md#term-application-programming-interface-api), a [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)
extension, another worker abstraction, an [operation](../../../../glossary.md#operation) semantic, or more than 24 paths, stop and
update planning or propose a bounded follow-up.

## Acceptance criteria

- [x] The marker is removed and the exact public task-0001 surface is only
      `CpuCapabilityProvider` plus inherited/record-independent [JDK](../../../../glossary.md#term-java-development-kit-jdk) members.
- [x] `CPU_BACKEND_ID.value()` is exactly `cpu`; every provider returns the exact constant and
      performs no discovery or mutable registration.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- [x] Every non-null capability query returns `false`, and null fails exactly as specified; tests
      prove that no operation kind, data type, shape, layout, zero-element case, or native-provider
      presence changes that answer.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [data type glossary](../../../../glossary.md#data-type--datatype); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [element glossary](../../../../glossary.md#term-element))</span>
- [x] Internal buffers and workspaces allocate exact-size native segments with at least the
      requested alignment from distinct shared arenas, including zero-size geometry, and expose
      no public storage API.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [alignment glossary](../../../../glossary.md#term-alignment); [arenas glossary](../../../../glossary.md#term-arena))</span>
- [x] Allocation failure closes a created arena once and preserves primary/suppressed failure;
      successful representation close is thread-safe, idempotent, closed-first, and never retried.<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe); [idempotent glossary](../../../../glossary.md#term-idempotent))</span>
- [x] Borrowed Model storage remains borrowed and exact; wrapper creation/binding performs no
      copy, allocation, arena operation, lifetime extension, or close of caller memory.<span data-glossary-navigation="true"> ([Borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [Model glossary](../../../../glossary.md#term-model); [binding glossary](../../../../glossary.md#term-binding); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- [x] All six exact heap-carrier mappings, heap slices/offsets, genuine native segments,
      exact-segment fallback for unavailable heap carriers, read-only access, dead/inaccessible
      scopes, and required exact-segment alignment are validated at cold bind.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- [x] Mixed array/exact-segment ordered selections bind independently into direct typed arguments
      with no global storage mode or materialization.<span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument); [materialization glossary](../../../../glossary.md#materialization))</span>
- [x] The CPU executable seam extends current Runtime contracts, retains exact plan/access
      declarations, uses checked compatibility once, and constructs a <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Runtime glossary](../../../../glossary.md#term-runtime))</span>[`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation) whose hot
      method needs no array lookup, nominal-representation access, type switch, cast, heap-base
      discovery, route choice, graph fact, or allocation.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [nominal glossary](../../../../glossary.md#term-nominal); [graph glossary](../../../../glossary.md#graph))</span>
- [x] The worker group owns a fixed bounded platform-worker set; ranges are contiguous,
      non-empty, exactly covering, and deterministic, with no allocation per element/node.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [node glossary](../../../../glossary.md#node))</span>
- [x] Empty, single-range, saturated-range, deterministic, concurrent, cancelled, failed,
      interrupted, reentrant, closing, and closed dispatch cases satisfy the specified synchronous
      lifecycle.<span data-glossary-navigation="true"> ([reentrant glossary](../../../../glossary.md#memory-management-atomic-counters-and-complexity); [dispatch glossary](../../../../glossary.md#term-dispatch); [synchronous glossary](../../../../glossary.md#term-synchronous); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [x] First/suppressed worker failure identity/order, interruption restoration, cooperative
      cancellation, idempotent shutdown, and no surviving worker after close are tested.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [cancellation glossary](../../../../glossary.md#term-cancellation))</span>
- [x] Production CPU code contains no operation/kernel switch, Vector API, Class-File API, ASM,
      OpenBLAS invocation, tuning/cache, reflection, <span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [Vector API glossary](../../../../glossary.md#term-vector-api); [Class-File API glossary](../../../../glossary.md#term-class-file-api); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [cache glossary](../../../../glossary.md#term-cache))</span>`ServiceLoader`, executor exposure, registry,
      service locator, <span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator))</span>`Map<String,Object>`, raw `Object` [API](../../../../glossary.md#term-application-programming-interface-api), or unchecked generic access.
- [x] Existing [module](../../../../glossary.md#term-module) dependencies and Java 26 build configuration remain unchanged.
- [x] Focused CPU tests and the final CPU module suite pass without an installed native library or
      platform-specific prerequisite.<span data-glossary-navigation="true"> ([native library glossary](../../../../glossary.md#term-native-library))</span>
- [x] Public and internal production declarations have meaningful Javadoc covering ownership,
      lifetime, threading, nullability, parameters, results, and failures.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [nullability glossary](../../../../glossary.md#term-nullability); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- [x] A separate clean documentation-focused agent pass finalizes affected Javadocs, CPU guidance,
      glossary impact, links, and planning evidence in the same overall change.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- [x] CPU Javadoc, targeted Markdown/link/anchor/fence checks, exact public/package-private shape,
      exact path scope, no-later-specification, status synchronization, and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>`git diff --check`
      pass.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation development may use focused classes. After executable Java stabilizes, [run](../../../../glossary.md#term-run) once:

```bash
./gradlew :backends:cpu:test
```

The implementation pass must also record focused evidence that the public surface and direct [hot
path](../../../../glossary.md#term-hot-path) are enforced by automated tests. Do not rely on recurring manual `javap` or [bytecode](../../../../glossary.md#term-bytecode) checks
when the [invariant](../../../../glossary.md#term-invariant) can be expressed in the listed tests.

Documentation-focused pass, after final Javadocs and explanatory text:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

Validate repository-local Markdown links and heading anchors for the task, [CPU](../../../../glossary.md#term-cpu) master plan,
roadmap, CPU guide, and glossary. Validate exact changed-path [scope](../../../../glossary.md#term-scope); exactly one CPU detailed task
specification; task 0001 as the sole [`Ready`](../../../../glossary.md#term-ready) CPU row during implementation; tasks 0002–0016
remaining `Draft` with no detailed specifications; unchanged [Gradle](../../../../glossary.md#term-gradle)/architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/architecture-
test/[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance)/integration paths; and final task/master/roadmap status [synchronization](../../../../glossary.md#term-synchronization).

Repository-wide Java validation is deferred to the portable CPU foundation/coverage [checkpoint](../../../../glossary.md#term-checkpoint) or
[continuous integration](../../../../glossary.md#term-continuous-integration-ci). If implementation changes a [dependency](../../../../glossary.md#term-dependency), build rule, shared [module](../../../../glossary.md#term-module),
architecture boundary, or another module despite this task's exclusions, stop rather than
silently expanding validation.

The documentation pass reuses the successful implementation test evidence and does not rerun
Java tests unless it changes executable behavior or records a concrete stale-evidence risk.

## Dependencies

- Complete Planning tasks 0001–0006: capability provider/query, ownership, partitions, and logical
  memory boundaries.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Complete Runtime tasks 0001–0014: prepared geometry, representations, run ownership/creation,
  checked executable binding, schedule, transfer/publication, runner, cleanup, and enforcement.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [run glossary](../../../../glossary.md#term-run); [binding glossary](../../../../glossary.md#term-binding); [publication glossary](../../../../glossary.md#term-publication))</span>
- Complete Prepare tasks 0001–0003: partition analysis/declarations, deterministic slot
  assignment/finalization, and complete schedule orchestration.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [partition glossary](../../../../glossary.md#partition))</span>
- Complete [Backend](../../../../glossary.md#backend) Contract tasks 0001–0004: [`BackendId`](../../../../glossary.md#backend-identity--backendid), device/availability facts, and hard
  requirements.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>
- Complete [Trace](../../../../glossary.md#trace) tasks 0001–0002 as a preserved [DTO](../../../../glossary.md#data-transfer-object--dto)-only [leaf](../../../../glossary.md#term-leaf); no current [CPU](../../../../glossary.md#term-cpu) [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) is consumed.
- Complete Config tasks 0001–0003 as preserved declarative inputs; no current CPU prepare-target
  input is stable or introduced here.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Complete [Model](../../../../glossary.md#term-model) [host](../../../../glossary.md#term-host)-storage foundation, including `HostTensorStorage`,
  `MemorySegmentStorage`, and the exact six primitive [carrier](../../../../glossary.md#term-storage-carrier) mappings.
- Complete OpenBLAS provider tasks 0001–0003 as a preserved low-level leaf; CPU task 0001 invokes
  none of its APIs.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26 [FFM](../../../../glossary.md#term-ffm) and platform-thread APIs already selected by the root build.

## Follow-up tasks

- CPU 0002 remains Draft and blocked from Ready status by the separately authorized architecture
  synchronization replacing the current ASM-specific wording before any generated-kernel work.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [synchronization glossary](../../../../glossary.md#term-synchronization); [kernel glossary](../../../../glossary.md#kernel))</span>
- CPU 0003–0009 remain ordered Draft work for generation, caching, typed portable preparation,
  semantic-family coverage, and the portable closure checkpoint.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- [CPU](../../../../glossary.md#term-cpu) 0010–0016 remain Draft optional native-provider and compatible tuning-[cache](../../../../glossary.md#term-cache) integration.
- Later CPU preparation must supply typed `BackendAnalysisInputs`, [opaque](../../../../glossary.md#term-opaque)
  `BackendPreparationPlan`, a [`BackendPartitionPreparer`](../../../../glossary.md#backend-partition-preparer--backendpartitionpreparer), a `BackendPartitionFinalizer`, concrete
  representation creators, and a schedule assembler only when at least one truthful operation
  route exists.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Later Engine composition supplies the provider, availability/target facts, CPU preparation
  collaborators, and caller-input wrapping. It does not broaden this task's public surface.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [target glossary](../../../../glossary.md#term-target))</span>

## Architecture impact

Expected impact: None.

This task implements existing concrete-[backend](../../../../glossary.md#backend) responsibilities and consumes existing [Planning](../../../../glossary.md#term-planning),
[Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), Backend Contract, [Model](../../../../glossary.md#term-model), and [Trace](../../../../glossary.md#trace) boundaries. It does not change the
authoritative [CPU](../../../../glossary.md#term-cpu) ASM wording or authorize generated [kernels](../../../../glossary.md#kernel). If implementation requires an
architecture change, stop and report the exact conflict.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/backends/cpu/master-plan.md, and
docs/planning/backends/cpu/tasks/0001-cpu-capability-representation-binding-and-parallel-foundation.md.
Read the directly referenced Planning, Prepare, Runtime, Backend Contract, Model storage, Trace,
Config, OpenBLAS, architecture, documentation, and Java 26 build contracts needed by the task.

Implement task CPU 0001 exactly as specified. Do not implement an operation, kernel, CPU
preparer/finalizer, generated-code path, OpenBLAS route, public buffer/executor API, architecture
change, dependency/build change, or later CPU task. Stop and report any architecture or scope
conflict instead of inventing a new boundary.

After code implementation and the one final CPU test run, hand the resulting diff and exact test
evidence to a separate documentation-focused agent/thread with clean context. That pass must
follow docs/developer-guide/documentation-rules.md, independently finalize affected Javadocs,
CPU guidance, glossary impact, planning evidence, and documentation validation in the same
overall change, and must not repeat successful Java tests unless executable behavior changes or
a concrete stale-evidence risk is recorded.

Update this task with local decisions, known limitations, validation evidence, implementation
notes, completion summary, and final synchronized status. Do not mark Complete before every
acceptance criterion and the documentation pass succeed.
```

## Local decisions

- Capability remains unconditionally fail-closed because this task supplies infrastructure but no
  executable semantic coverage.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- The only public CPU type is the capability provider. Physical memory, typed arguments,
  executable specialization, and parallel coordination remain package-private in one cohesive
  execution package.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [arguments glossary](../../../../glossary.md#term-argument); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Shared arenas are selected for run-owned native representations because later CPU worker and
  FFM access may cross threads during one run. One representation owns one arena, keeping cleanup
  and partial-failure rollback explicit.<span data-glossary-navigation="true"> ([arenas glossary](../../../../glossary.md#term-arena); [run glossary](../../../../glossary.md#term-run); [FFM glossary](../../../../glossary.md#term-ffm); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- A segment binds to an exact primitive carrier with a carrier-relative byte offset only when the
  JDK exposes that matching carrier. Otherwise <span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>`CpuBufferArgument.Segment` retains the exact
  selected segment or slice and reports offset zero relative to it. This access form deliberately
  covers genuine native segments and heap-backed segments with unavailable carriers; it does not
  claim native provenance. Storage access is resolved once per selection, not encoded as a global
  mode.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [heap glossary](../../../../glossary.md#term-heap); [provenance glossary](../../../../glossary.md#provenance))</span>
- The parallel foundation owns fixed platform workers and bounded ranges without selecting thread
  count from Config, tuning evidence, or workload semantics. Those later preparation decisions
  remain outside this task.

## Known limitations

- [CPU](../../../../glossary.md#term-cpu) advertises and executes no [Model](../../../../glossary.md#term-model) [operation](../../../../glossary.md#operation) after this task.
- No production CPU preparer, finalizer, schedule assembler, executable occurrence, transfer, or
  publication recipe exists yet.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [publication glossary](../../../../glossary.md#term-publication))</span>
- Borrowed confined segments can support only access from threads allowed by their JDK scope;
  parallel use must fail cold unless every selected worker is allowed.<span data-glossary-navigation="true"> ([Borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [scope glossary](../../../../glossary.md#term-scope))</span>
- Internal native memory is one arena/allocation per representation. Pooling, reuse, aliasing,
  shared arenas, and persistent prepared memory are intentionally absent.<span data-glossary-navigation="true"> ([native memory glossary](../../../../glossary.md#aliasing-and-arena); [arena glossary](../../../../glossary.md#term-arena); [Pooling glossary](../../../../glossary.md#term-pooling); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- Parallel cancellation is cooperative between bounded ranges and cannot undo writes made before
  a failure. Runtime validity remains the failure-safety boundary.<span data-glossary-navigation="true"> ([cancellation glossary](../../../../glossary.md#term-cancellation); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Deterministic range boundaries do not by themselves define a deterministic reduction. Later
  reduction routes must define stable partial-result and combine order.
- Trace has no current backend/run payload family, and Config has no stable CPU prepare-target
  contract. This task preserves both boundaries without inventing replacements.<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [backend glossary](../../../../glossary.md#backend); [run glossary](../../../../glossary.md#term-run); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [prepare glossary](../../../../glossary.md#prepare); [target glossary](../../../../glossary.md#term-target))</span>
- On [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26, `MemorySegment.asReadOnly()` over a heap segment produces a read-only heap-backed
  segment whose <span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap))</span>`heapBase()` is empty. Such a segment therefore binds through the exact
  `CpuBufferArgument.Segment` form; the implementation preserves its identity, byte size,
  read-only state, ownership, liveness, and current-thread access without copying.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [liveness glossary](../../../../glossary.md#term-liveness))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation development: `./gradlew :backends:cpu:compileJava` initially failed with seven
  record-constructor visibility errors and passed after their correction.
- Implementation development: `./gradlew :backends:cpu:compileTestJava` initially failed with one
  lambda-capture compiler error and passed after its correction.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Implementation development: the first filtered CPU suite ran 12 tests with 11 passing and one
  failure because a read-only heap segment's JDK 26 <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [heap glossary](../../../../glossary.md#term-heap); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>`heapBase()` was empty. The resolved contract
  retains that exact segment as `CpuBufferArgument.Segment`; the replacement filtered run passed
  12 of 12 tests.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- Implementation development: the strengthened filtered suite first ran 16 tests with 15 passing
  and one test-only concurrent-close scheduling race. Synchronizing the test on <span data-glossary-navigation="true"> ([race glossary](../../../../glossary.md#term-race))</span>`isClosed()`
  resolved the race; the replacement filtered run passed all 16 tests.
- Implementation final executable evidence: `./gradlew :backends:cpu:test` passed once after
  executable stabilization with `BUILD SUCCESSFUL`, 21 actionable tasks (2 executed and 19
  up-to-date), 16 tests, and no failures, errors, or skips. Production forbidden-feature scanning
  and <span data-glossary-navigation="true"> ([feature glossary](../../../../glossary.md#term-software-feature))</span>`git diff --check` also passed before documentation handoff. No executable Java changed
  after this run.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
- Separate clean documentation context `/root/cpu0001_docs` reviewed the final 13 production
  paths, six test paths, generated public Javadoc, CPU guide, glossary, task/master/roadmap, and
  directly relevant Planning, Prepare, Runtime, Backend Contract, Model storage, Trace, Config,
  OpenBLAS, architecture, ADR 0010/0011, and Java 26 build contracts. It changed Javadocs and the
  five authorized documentation/planning paths only; executable Java and tests remained unchanged.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Backend glossary](../../../../glossary.md#backend); [Model glossary](../../../../glossary.md#term-model); [Trace glossary](../../../../glossary.md#trace); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Documentation validation: `./gradlew :backends:cpu:javadoc` passed with 11 actionable tasks
  (2 executed and 9 up-to-date). After the last Javadoc wording correction, the same command was
  rerun and passed with the same task counts. Generated `CpuCapabilityProvider` and package pages
  were inspected.
- The first local Markdown-link checker invocation had a checker-only Ruby regular-expression
  interpolation syntax error and inspected no result. The corrected checker then exposed an
  anchor-normalization ambiguity around slash-separated headings; after matching the repository's
  accepted collapsed-hyphen form, the final five-file relative-link and heading-anchor check
  passed. Five-file fence/final-newline and production/documentation trailing-whitespace checks
  passed.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>
- Exact expected-versus-actual 24-path comparison, one-and-only-one CPU detailed specification,
  absence of CPU 0002–0016 specifications, and unchanged Gradle/architecture/ADR/architecture-
  test/backend-conformance/integration paths passed. A first status script assumed the value
  immediately followed the <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>`## Status` heading; the corrected nonblank-line check passed task,
  master-plan, and roadmap synchronization. Public/package-private shape, forbidden mechanism,
  stale-status, resolved Segment-content, and generated-page scans passed.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [shape glossary](../../../../glossary.md#shape))</span>
- Final coordinator audit found one stale current-state roadmap sentence that still called task
  0001 Ready. The clean documentation context corrected it to Complete while preserving the
  explicitly historical earlier planning-pass sentence, then reran the applicable five-file
  Markdown, status, exact-scope, whitespace, and <span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` gates. No executable Java,
  tests, or Javadocs changed, so Java tests and Javadoc generation were not rerun. The first
  combined status invocation treated ripgrep's no-match output as a literal zero; the corrected
  zero-match count passed without requiring a repository change.
- Final targeted Markdown/link/anchor/fence/newline/whitespace, public/package-private shape,
  forbidden-content, exact 24-path, no-later-specification, unchanged excluded-path, synchronized
  status, and `git diff --check` checks passed as recorded by `/root/cpu0001_docs`.

## Implementation notes

- Replaced the [CPU](../../../../glossary.md#term-cpu) marker with the sole public `CpuCapabilityProvider`; it keeps exact `cpu`
  identity and unconditional fail-closed support.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Added package-private run-owned native buffer/workspace allocation, non-owning borrowed storage,
  direct typed cold arguments, the Runtime executable binding seam, and fixed-worker range
  coordination within <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [buffer glossary](../../../../glossary.md#term-buffer); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [arguments glossary](../../../../glossary.md#term-argument); [Runtime glossary](../../../../glossary.md#term-runtime); [binding glossary](../../../../glossary.md#term-binding))</span>`io.github.pho001.synaptik.backend.cpu.execution`.
- The final binding implementation classifies an observable matching primitive carrier to its
  typed array argument. If <span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>`heapBase()` is unavailable, it retains the exact segment or [slice](../../../../glossary.md#slice) in
  `CpuBufferArgument.Segment`; `byteOffset()` is zero because the retained segment is already the
  selected region. No copy or native-provenance assertion is made.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Tests lock all six carrier mappings, heap-slice offsets, read-only heap exact-segment retention,
  native and mixed binding, lifetime/access/alignment rejection, direct hot fields, allocation,
  worker range, cancellation, interruption, failure, concurrency, and shutdown contracts.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [lifetime glossary](../../../../glossary.md#term-lifetime); [alignment glossary](../../../../glossary.md#term-alignment); [cancellation glossary](../../../../glossary.md#term-cancellation); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>

## Completion summary

- Completed changes: truthful CPU identity/capability provider; aligned shared-arena native
  buffer/workspace foundation; exact borrowed storage and typed cold binding; direct CPU prepared
  invocation seam; fixed bounded worker/range coordination; finalized Javadocs, CPU guide,
  glossary, and synchronized planning records.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [identity glossary](../../../../glossary.md#term-identity-meanings); [arena glossary](../../../../glossary.md#term-arena); [buffer glossary](../../../../glossary.md#term-buffer); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [binding glossary](../../../../glossary.md#term-binding); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: the 13 authorized CPU production paths (including marker removal), six
  authorized CPU test paths, `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task,
  the CPU master plan, and the roadmap.
- Tests and validation: focused development evidence and the one final 16-test CPU module suite
  passed; the clean documentation context reused that executable evidence and passed CPU Javadoc,
  generated-page inspection, targeted documentation/scope/surface/status/content checks, and
  final whitespace validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: `/root/cpu0001_docs`, using General, API/Javadoc, Planning, and
  Backend Guide profiles, finalized the combined change without changing executable behavior.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Backend glossary](../../../../glossary.md#backend))</span>
- Documentation impact: CPU guidance now distinguishes the implemented non-computational
  foundation from planned routes and explains exact array/segment cold binding.
- Javadoc review: every CPU production declaration and contract-relevant member was reviewed for
  ownership, lifetime, threading, nullability, inputs, results, and failures; final generation
  passed.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifetime glossary](../../../../glossary.md#term-lifetime); [nullability glossary](../../../../glossary.md#term-nullability))</span>
- Glossary impact: current implementation status and the reusable CPU buffer argument distinction
  are now documented, including the JDK 26 read-only heap behavior.<span data-glossary-navigation="true"> ([CPU buffer argument glossary](../../../../glossary.md#cpu-buffer-argument); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [heap glossary](../../../../glossary.md#term-heap))</span>
- Architecture, dependency, build, ADR, architecture-test, backend-conformance, and integration
  impact: none; the existing contracts and excluded paths remain unchanged.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0001. CPU 0002 remains Draft and requires the separately
  authorized architecture wording synchronization before it can become Ready.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [Ready glossary](../../../../glossary.md#term-ready))</span>

Status: Complete
