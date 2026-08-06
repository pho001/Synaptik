# Task 0001: CPU Capability, Representation, Binding, and Parallel Foundation

## Status

Superseded

## Goal

Replace the CPU marker with the smallest truthful concrete-backend foundation needed before any
CPU operation route is implemented:

- one explicitly constructible CPU capability provider with the stable backend identity `cpu`;
- a fail-closed capability answer that advertises no executable operation semantics yet;
- one run-owned aligned native off-heap `MemorySegment` representation for internal CPU buffers
  and workspaces;
- package-private cold binding of borrowed `MemorySegment` storage into direct,
  data-type-correct heap-carrier or exact-segment arguments;
- a narrow package-private CPU prepared-executable/bound-invocation seam for later generated and
  native routes; and
- one package-private shared worker foundation for later scalar-parallel and Vector API-parallel
  kernels.

The result establishes identity, memory, binding, lifetime, threading, and failure contracts. It
does not claim that CPU can execute any Model operation.

## Rationale and mental model

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
`supports` must return `false` for every non-null query. Runtime already owns representation
creation ordering, rollback, per-run ownership, validity, and cleanup orchestration. CPU owns only
its physical allocation/access/close mechanics and its checked specialization at the cold binding
boundary. The worker group is backend-private infrastructure, not a public executor or a route.

## Scope

- Remove `CpuBackendModule`.
- Add one public final `CpuCapabilityProvider` implementing Planning's existing
  `BackendCapabilityProvider`.
- Define one public immutable `BackendId` constant with exact value `cpu`, returned by exact
  reference from every provider instance.
- Validate `supports(query)` with `NullPointerException("query")`, then return `false` without
  inspecting, switching on, or retaining the operation occurrence.
- Add package documentation that distinguishes delivered foundation from later semantic coverage.
- Add package-private CPU buffer and workspace representation implementations under the CPU
  execution package.
- Allocate every run-owned internal buffer/workspace with `Arena.ofShared()` and
  `Arena.allocate(byteSize, byteAlignment)` so later CPU worker threads and Foreign Function and
  Memory (FFM) calls can use the same native segment with at least the requested alignment.
- Treat zero-byte geometry as a real run-owned representation with a live shared arena and a
  zero-byte segment allocated using the declared alignment.
- Close the owning arena exactly once through an idempotent representation close operation.
- Add package-private borrowed CPU buffer construction from `HostTensorStorage`, retaining the
  exact storage and exact `MemorySegment` while acquiring no ownership and making close a no-op.
- During cold binding, validate data type, exact byte size, liveness, current-thread
  accessibility, required writability, heap-carrier compatibility, and native alignment before
  constructing a direct typed argument.
- Recognize the six exact heap carrier mappings already used by Model:
  `FLOAT64 -> double[]`, `FLOAT32 -> float[]`, `BFLOAT16 -> short[]`, `INT32 -> int[]`,
  `INT64 -> long[]`, and `BOOL -> byte[]`.
- Preserve an observable heap carrier's carrier-relative byte offset and exact byte length so
  slices bind without copying. When the exact matching primitive carrier is not observable,
  retain the exact selected segment or slice without copying; this includes both genuine native
  segments and JDK 26 read-only heap segments whose `heapBase()` is empty.
- Permit each selection independently to use a typed array argument or an exact-segment argument,
  including mixed input/output signatures. Perform no materialization merely because argument
  forms or representation provenance differ.
- Add a package-private abstract CPU prepared-executable base over Runtime's exact
  `PreparedExecutable` contract. It performs only the CPU representation/type/access checks and
  hands fresh cold-binding arrays to a route-specific subclass, which must build a bound
  invocation retaining direct typed fields rather than those arrays.
- Add a package-private CPU workspace representation/access role for native shared-arena scratch.
- Add a package-private fixed worker group and range-body contract for later scalar and Vector
  routes.
- Split one non-empty half-open element range into at most the configured worker count, with
  deterministic contiguous boundaries and no empty range. Empty work completes without dispatch.
- Allocate coordination only once per submitted parallel call, never per element or graph node.
- Define synchronous completion, cancellation at range boundaries, interruption, first/suppressed
  failure, worker shutdown, and idempotent close semantics.
- Add focused tests for exact public shape, fail-closed capability, native allocation/alignment,
  zero size, lifetime and rollback compatibility, all six heap carriers, heap slices, native and
  mixed binding, access rejection, direct bound fields, range partitioning, cancellation,
  interruption, failure suppression, concurrent submissions, and close.
- Finalize affected Javadocs, CPU backend guidance, glossary impact, and planning records through
  the required separate clean documentation-focused pass.

## Out of scope

- support or execution of any Tensor operation, operation family, metadata-only occurrence, view,
  scalar kernel, Vector API kernel, generated kernel, native kernel, or fused partition
- changing `supports` to `true` for any query, including zero-element or view-like occurrences
- `java.lang.classfile.CodeBuilder`, ASM use or replacement, class generation, hidden classes,
  method-handle entry points, generator schemas, generated-artifact keys, or caches
- task 0002 or any other generated-kernel work before the separately authorized architecture
  synchronization recorded in the CPU master plan
- CPU partition analysis, lowering, route candidates, route selection, representation-plan
  reconciliation, schedule assembly, finalization of a real operation, or a production executable
- OpenBLAS invocation integration, library discovery, provider fallback, thread-count mutation, or
  any oneMKL, oneDNN, Accelerate, AOCL, ZenDNN, Metal, or CUDA integration
- planning orchestration, ownership scoring, compiler behavior, Engine composition, public Tensor
  API, publication/output access, or benchmark/model-autotuning behavior
- a public CPU buffer, workspace, executor, scheduler, range, route, config-map, target-discovery,
  registry, manager, service-locator, or generic native-memory API
- pooling, reuse, alias analysis, persistent prepared resources, mapped memory, pinning, NUMA
  policy, device memory, automatic materialization, packing, layout conversion, or coherence
- per-operation trace payloads or event emission; the current Trace payload families remain Draft
- Config, Planning, Prepare, Runtime, Backend Contract, Trace, Model, OpenBLAS provider, Engine,
  Gradle, dependency, architecture-contract, architecture-explanation, ADR, architecture-test,
  backend-conformance, or integration-test changes
- detailed specifications for CPU tasks 0002–0016

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core invariants
  - `modules/runtime`
  - `modules/prepare`
  - Concrete backend modules
  - CPU backend routes
  - Prepare lifecycle
  - Run lifecycle
  - Dependency rules
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Tracing](../../../../architecture/tracing.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)

## Architecture constraints

- Planning selects CPU ownership; CPU prepare will later select one implementation route. This
  task supplies only the Planning-owned capability-provider collaboration and cannot select a
  route.
- Capability is semantic ownership truth, not availability, registration, target discovery,
  preparation success, or execution readiness. Infrastructure alone justifies no `true` answer.
- The authoritative contract still names CPU ASM and separately gates bytecode-generated CPU
  kernels on an architecture update. This task does not edit or reinterpret that contract, use
  ASM, use `CodeBuilder`, or create generated code.
- Concrete backends own physical representation implementations and allocation, release,
  transfer, and access mechanics. Runtime owns prepared geometry, per-run logical state,
  ownership transitions, validity/residency, rollback, and cleanup orchestration.
- Caller inputs are borrowed for one run. CPU wrappers never close or extend a borrowed Model
  storage lifetime. Internal buffers and workspaces are run-owned and own their shared arena.
- `HostTensorStorage` and `MemorySegmentStorage` remain unchanged borrowed Model contracts. CPU
  adds no alignment, allocation, route, or Runtime ownership meaning to them.
- Prepare analysis declares exact byte size/alignment before slot assignment. CPU allocation
  consumes the final Runtime geometry only through backend creator callbacks; it does not infer
  geometry from a Tensor, graph value, or operation in the run path.
- Backend finalization may construct immutable executable recipes but cannot allocate closeable
  physical resources. The native CPU allocation remains per-run representation creation.
- Runtime's existing `PreparedExecutable.bind` is the only heterogeneous compatibility boundary.
  Later bound CPU invocations retain direct typed fields and perform no representation lookup,
  heap-base discovery, cast, data-type switch, route selection, or allocation in their hot call.
- One immutable executable may bind concurrently to distinct `RunState` instances. A bound
  invocation remains non-thread-safe and cannot race its `RunState` closure.
- The CPU worker group owns workers and coordination. Generated classes and later kernels do not
  create pools, submit one task per element/node, or own scheduler lifecycle.
- No dependency direction, public shared contract, module boundary, or architecture rule changes.
  If implementation requires one, stop and request a separate planning/architecture decision.

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.backend.cpu` — replaces its marker with the only task-0001 public
  surface, `CpuCapabilityProvider`, plus package documentation.

Package added:

- `io.github.pho001.synaptik.backend.cpu.execution` — package-private CPU physical
  representations, cold typed arguments, prepared/bound route seam, and shared worker/range
  foundation. The package is implementation-facing inside the CPU backend and exports no public
  type.

No `buffer`, `storage`, `executor`, `service`, `manager`, `registry`, `config`, `route`, `kernel`,
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
message `query` and otherwise returns `false`. The class has no singleton accessor, availability
snapshot, device identity, target probe, configuration, preparer, finalizer, executable factory,
route list, diagnostic-reason result, or mutable field.

All declarations in `io.github.pho001.synaptik.backend.cpu.execution` are package-private. The
implementation must contain the following narrow roles and source-level shape; a signature change
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

Record canonical constructors and accessors may be declared explicitly for validation and
Javadoc. `CpuBufferArgument.Segment` implements `byteOffset()` with constant zero. Array variants
validate non-null carrier, non-negative offset/size, carrier-width alignment, and in-carrier span.
The exact-segment variant validates non-null data type/segment, exact size, liveness/access, and
zero offset semantics. Its name does not assert native provenance. No variant exposes `Object`, a
raw address, a storage-kind string/enum, or a mutation method.

`CpuPreparedExecutable` requires `bufferDataTypes.size()` to equal the buffer-selection count and
validates those entries after Runtime's four constructor inputs. Its final Runtime hooks accept
only open `CpuBufferRepresentation` values and `CpuNativeWorkspace` values. For buffer selection
`i`, the retained data type must be `bufferDataTypes[i]`, byte size must equal the selected
`PreparedMemoryPlan.BufferEntry.byteSize()`, and the segment must satisfy that entry's alignment;
the aligned `BufferAccess` supplies writability rules. Workspaces must match their selected plan
entry's exact size and requested alignment. Only after every selection passes does the base copy
the already-classified direct `CpuBufferArgument` references and typed workspace references into
fresh cold arrays and invoke `bindCpu` once. The subclass must return a non-null Runtime
`BoundInvocation` for the exact state; Runtime's existing final association check remains
authoritative.

The implementation roles are:

- `CpuBufferRepresentation` — common immutable metadata/access base implementing Runtime
  `BufferRepresentation`; retains exact `DataType`, byte size, and exact `MemorySegment`.
- `CpuBorrowedBuffer` — non-owning wrapper over exact `HostTensorStorage`; close is a no-op and
  never closes, replaces, or prolongs the storage segment.
- `CpuNativeBuffer` — run-owned shared-arena aligned native buffer with idempotent close.
- `CpuNativeWorkspace` — run-owned shared-arena aligned native workspace implementing Runtime
  `WorkspaceRepresentation`, with the same allocation/close rules and no logical validity.
- `CpuBufferArgument` — sealed cold-bound argument family with one variant per six exact primitive
  array carriers plus one exact-segment variant. Array variants retain a direct array,
  carrier-relative byte offset, and exact byte length; the segment variant retains the exact
  selected segment or slice regardless of native or heap provenance. No variant owns memory.
- `CpuPreparedExecutable` — abstract CPU specialization of Runtime `PreparedExecutable`; its
  final compatibility/binding hooks validate and convert selected CPU representations once, then
  call one package-private abstract route-specific binder.
- `CpuWorkerGroup` — fixed package-private platform-worker owner with synchronous bounded-range
  dispatch and idempotent close.
- `CpuRangeBody` — package-private functional contract receiving only primitive
  `startInclusive`, `endExclusive`, and deterministic `rangeIndex` values.
- `CpuParallelExecutionException` — package-private unchecked failure used only when interruption
  or worker-coordination failure cannot be propagated as the original unchecked worker failure.

Do not add a public representation/access type, a generic typed-buffer abstraction, a public
executor, an operation callback, a route registry, a provider facade, or a second backend ID
owner.

## Native representation ownership, allocation, and failure semantics

- Internal buffer and workspace factories accept exact non-negative `byteSize` and positive
  power-of-two `byteAlignment`, matching Runtime `PreparedMemoryPlan` geometry rules.
- Validation order is byte size, alignment, then arena/allocation work. Stable failures are
  `byteSize must be non-negative` and `byteAlignment must be a positive power of two`.
- Each factory creates a distinct `Arena.ofShared()` and allocates exactly one segment with the
  requested byte size and at least the requested alignment. The native address must be divisible
  by `byteAlignment`; a stronger platform/JDK alignment remains valid. The arena is shared because
  the representation may be accessed by the orchestrating thread, CPU workers, or an FFM call
  while its run remains open.
- Zero size follows the same path and yields a live zero-byte segment associated with its arena.
  It is not represented by `MemorySegment.NULL`, a global singleton, a heap array, or an absent
  optional.
- If arena creation or allocation fails, the original unchecked exception or error is preserved.
  An arena created before allocation failure is closed once; a distinct cleanup failure is
  suppressed, and the same exact primary object is not self-suppressed.
- Successful construction transfers physical cleanup to the representation. Runtime transfers
  run-owned cleanup responsibility only when complete `RunState` creation succeeds and already
  supplies reverse rollback/cleanup with first/suppressed failure rules.
- Representation `close` is idempotent and thread-safe. It marks the representation closed before
  closing its arena, calls the arena at most once, and propagates that close's unchecked failure
  unchanged. A failed close is not retried.
- Access and cold binding after close fail with `IllegalStateException("CPU representation is closed")`.
- Allocation neither initializes logical validity nor promises a numerical zero value. Runtime's
  `CreatedBuffer` starts invalid; an `InitializedBuffer` creator is responsible for materializing
  its logical value before return.
- No pooling, shared arena across representations, cleaner/finalizer fallback, global native
  allocator, address cache, manual free, or ownership transfer outside Runtime is introduced.

## Borrowed storage and cold typed binding

- `CpuBorrowedBuffer` accepts one non-null `HostTensorStorage`, checks its current liveness, and
  retains the exact storage and exact segment. It performs no copy, slice, reinterpretation,
  alignment upgrade, arena operation, or carrier allocation.
- Binding validates selections in Runtime's existing buffer-selection order. The executable's
  aligned `BufferAccess` decides whether a read-only segment is acceptable: `READ_ONLY` accepts
  it; `WRITE_ONLY` and `READ_WRITE` reject it.
- Every segment must be alive and accessible by the binding thread. A later parallel binder must
  additionally prove accessibility from every selected worker before dispatch; confined borrowed
  segments therefore cannot silently enter parallel execution.
- The segment byte size must equal the representation's retained byte size. Native internal
  segments must satisfy the retained declared alignment. Borrowed segments make no alignment
  promise beyond what a selected later route explicitly requires during its cold compatibility
  check.
- When `heapBase()` exposes a heap base, cold binding requires that object to be the exact
  primitive-array carrier mapped from `DataType`. It retains the exact array plus the segment's
  carrier-relative byte offset and exact length. Misaligned offsets for that carrier fail binding.
- When `heapBase()` is unavailable, cold binding retains the exact selected segment or slice in
  `CpuBufferArgument.Segment`. JDK 26 makes this path necessary for read-only heap segments as
  well as genuine native segments. `Segment.byteOffset()` is zero relative to the exact retained
  region. Binding does not extract/cache a raw address, copy bytes, or assert native provenance.
- A segment whose observable heap carrier conflicts with `DataType`, an unsupported observable
  heap base, a dead/inaccessible scope, a read-only write selection, an invalid offset/length, or
  a required alignment mismatch returns incompatibility through Runtime's checked binding hook so
  Runtime emits its existing stable indexed failure.
- Mixed signatures are ordinary ordered selections. Each argument is specialized independently;
  no global heap/native mode, copying, materialization, or route choice occurs during binding.
- The route-specific binder may allocate the bound invocation and ordinary fixed-size binding
  state. It must copy required direct argument fields out of the fresh cold-binding arrays and
  must not retain those arrays, nominal Runtime representations, a `RunState` lookup path, or a
  generic object carrier for hot execution.
- Tests use non-computational probe invocations only to prove direct typed field retention. They
  must not add a scalar, Vector, native, or operation kernel.

## Parallel worker, range, cancellation, and failure semantics

- `CpuWorkerGroup` is constructed with a positive worker count and owns exactly that many named
  daemon platform worker threads for its lifetime. It uses no common pool, virtual threads,
  `parallelStream`, global executor, or externally supplied scheduler.
- One synchronous dispatch accepts a half-open range `[startInclusive, endExclusive)`, a positive
  minimum elements per range, a determinism flag, and one non-null `CpuRangeBody`.
- Validation order is body, non-negative/ordered range, then positive minimum range size. Stable
  failures identify `body`, `startInclusive`, `endExclusive`, and
  `minimumRangeSize must be positive` respectively.
- An empty range returns without waking a worker or allocating per-range state.
- A non-empty range creates between one and `workerCount` contiguous non-empty ranges. The count
  is bounded by both worker count and the ceiling of element count over minimum range size.
  Boundaries use quotient/remainder partitioning, cover every element exactly once, and are
  deterministic from the four primitive inputs.
- Deterministic mode assigns and later exposes stable ascending `rangeIndex` values and does not
  authorize a nondeterministic reduction/combine order. Task 0001 implements no combine step;
  later reduction routes must consume completed range results in ascending index order.
- Non-deterministic mode may let workers claim the next range through one shared primitive cursor.
  It does not change boundaries or semantic eligibility.
- Dispatch allocates at most one call-level coordination object and arrays proportional to the
  bounded range/worker count. It allocates no object per element, Model node, loop iteration, or
  hot scalar/vector lane.
- Normal return occurs only after every dispatched range completes. At most one range executes at
  a time on one worker, while different ranges may run concurrently.
- The first observed unchecked worker failure or error becomes primary. It requests cancellation,
  prevents unclaimed ranges from starting, waits for already-running ranges, and receives every
  later distinct worker failure as suppressed in observation order. Repetition of the exact
  primary object is skipped to avoid self-suppression. The primary object is rethrown unchanged.
- Cancellation is cooperative at range boundaries. The worker group does not stop a Java thread,
  interrupt arbitrary route code, retry a range, or claim rollback of partially written output.
  Runtime's existing output-validity rules keep declared outputs invalid after backend failure.
- If the orchestrating thread is interrupted while waiting, dispatch records interruption,
  requests cancellation, waits until all already-running work quiesces, restores the thread's
  interrupt status, and throws `CpuParallelExecutionException("CPU parallel execution interrupted", cause)`.
  Distinct worker failures observed during quiescence are suppressed on that failure in
  observation order.
- Concurrent dispatches are supported and isolated; worker capacity is shared without sharing
  call-level failure, cancellation, or range state. Reentrant dispatch from an owned worker is
  rejected with `IllegalStateException("CPU worker must not submit parallel work")` to prevent
  pool starvation.
- `close` is thread-safe and idempotent. It rejects new dispatch with
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

## Validation order and stable failures

Tests must lock at least these externally or cross-contract observable orders/messages:

1. `CpuCapabilityProvider.supports`: `query`, then unconditional `false`.
2. Native geometry: byte size, alignment, arena creation, allocation.
3. Cold executable binding: Runtime plan/open/range checks, CPU buffer selections in order,
   Runtime workspace selections in order, then route-specific bound-invocation construction.
4. CPU selection compatibility: representation kind, open/liveness/access, exact byte size,
   access writability, data-type carrier, offset/length, then route-specific alignment.
5. Worker dispatch: body, range start/end, minimum range size, group-open/reentrancy state, then
   call-level allocation/dispatch.

Use Runtime's existing indexed incompatibility failures rather than inventing a second public
exception vocabulary. Package-private CPU failures may be exact where tests need to lock cleanup,
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

Expected planning paths:

- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless a concrete contradiction is found: authoritative/focused architecture, ADRs
0010/0011, documentation rules and profiles, CPU and prerequisite master plans/tasks, current
Planning/Prepare/Runtime/Backend Contract/Trace/Config/Model/OpenBLAS production and test
contracts, Java 26 root/module Gradle configuration, architecture tests, backend-conformance
placeholder, and current JDK 26 FFM/threading API behavior.

## Maximum scope

At most 24 paths:

- 13 CPU production paths, counting removal of the marker;
- 6 CPU test paths;
- 2 explanatory documentation paths; and
- 3 planning paths.

No `.gitkeep` needs removal. No other source, test, documentation, build, architecture, ADR, or
planning path may change. If implementation needs a public CPU type beyond
`CpuCapabilityProvider`, a CPU preparer/finalizer, a cross-package internal API, a Runtime/Prepare
extension, another worker abstraction, an operation semantic, or more than 24 paths, stop and
update planning or propose a bounded follow-up.

## Acceptance criteria

- [x] The marker is removed and the exact public task-0001 surface is only
      `CpuCapabilityProvider` plus inherited/record-independent JDK members.
- [x] `CPU_BACKEND_ID.value()` is exactly `cpu`; every provider returns the exact constant and
      performs no discovery or mutable registration.
- [x] Every non-null capability query returns `false`, and null fails exactly as specified; tests
      prove that no operation kind, data type, shape, layout, zero-element case, or native-provider
      presence changes that answer.
- [x] Internal buffers and workspaces allocate exact-size native segments with at least the
      requested alignment from distinct shared arenas, including zero-size geometry, and expose
      no public storage API.
- [x] Allocation failure closes a created arena once and preserves primary/suppressed failure;
      successful representation close is thread-safe, idempotent, closed-first, and never retried.
- [x] Borrowed Model storage remains borrowed and exact; wrapper creation/binding performs no
      copy, allocation, arena operation, lifetime extension, or close of caller memory.
- [x] All six exact heap-carrier mappings, heap slices/offsets, genuine native segments,
      exact-segment fallback for unavailable heap carriers, read-only access, dead/inaccessible
      scopes, and required exact-segment alignment are validated at cold bind.
- [x] Mixed array/exact-segment ordered selections bind independently into direct typed arguments
      with no global storage mode or materialization.
- [x] The CPU executable seam extends current Runtime contracts, retains exact plan/access
      declarations, uses checked compatibility once, and constructs a `BoundInvocation` whose hot
      method needs no array lookup, nominal-representation access, type switch, cast, heap-base
      discovery, route choice, graph fact, or allocation.
- [x] The worker group owns a fixed bounded platform-worker set; ranges are contiguous,
      non-empty, exactly covering, and deterministic, with no allocation per element/node.
- [x] Empty, single-range, saturated-range, deterministic, concurrent, cancelled, failed,
      interrupted, reentrant, closing, and closed dispatch cases satisfy the specified synchronous
      lifecycle.
- [x] First/suppressed worker failure identity/order, interruption restoration, cooperative
      cancellation, idempotent shutdown, and no surviving worker after close are tested.
- [x] Production CPU code contains no operation/kernel switch, Vector API, Class-File API, ASM,
      OpenBLAS invocation, tuning/cache, reflection, `ServiceLoader`, executor exposure, registry,
      service locator, `Map<String,Object>`, raw `Object` API, or unchecked generic access.
- [x] Existing module dependencies and Java 26 build configuration remain unchanged.
- [x] Focused CPU tests and the final CPU module suite pass without an installed native library or
      platform-specific prerequisite.
- [x] Public and internal production declarations have meaningful Javadoc covering ownership,
      lifetime, threading, nullability, parameters, results, and failures.
- [x] A separate clean documentation-focused agent pass finalizes affected Javadocs, CPU guidance,
      glossary impact, links, and planning evidence in the same overall change.
- [x] CPU Javadoc, targeted Markdown/link/anchor/fence checks, exact public/package-private shape,
      exact path scope, no-later-specification, status synchronization, and `git diff --check`
      pass.

## Tests / validation

Implementation development may use focused classes. After executable Java stabilizes, run once:

```bash
./gradlew :backends:cpu:test
```

The implementation pass must also record focused evidence that the public surface and direct hot
path are enforced by automated tests. Do not rely on recurring manual `javap` or bytecode checks
when the invariant can be expressed in the listed tests.

Documentation-focused pass, after final Javadocs and explanatory text:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

Validate repository-local Markdown links and heading anchors for the task, CPU master plan,
roadmap, CPU guide, and glossary. Validate exact changed-path scope; exactly one CPU detailed task
specification; task 0001 as the sole `Ready` CPU row during implementation; tasks 0002–0016
remaining `Draft` with no detailed specifications; unchanged Gradle/architecture/ADR/architecture-
test/backend-conformance/integration paths; and final task/master/roadmap status synchronization.

Repository-wide Java validation is deferred to the portable CPU foundation/coverage checkpoint or
continuous integration. If implementation changes a dependency, build rule, shared module,
architecture boundary, or another module despite this task's exclusions, stop rather than
silently expanding validation.

The documentation pass reuses the successful implementation test evidence and does not rerun
Java tests unless it changes executable behavior or records a concrete stale-evidence risk.

## Dependencies

- Complete Planning tasks 0001–0006: capability provider/query, ownership, partitions, and logical
  memory boundaries.
- Complete Runtime tasks 0001–0014: prepared geometry, representations, run ownership/creation,
  checked executable binding, schedule, transfer/publication, runner, cleanup, and enforcement.
- Complete Prepare tasks 0001–0003: partition analysis/declarations, deterministic slot
  assignment/finalization, and complete schedule orchestration.
- Complete Backend Contract tasks 0001–0004: `BackendId`, device/availability facts, and hard
  requirements.
- Complete Trace tasks 0001–0002 as a preserved DTO-only leaf; no current CPU payload is consumed.
- Complete Config tasks 0001–0003 as preserved declarative inputs; no current CPU prepare-target
  input is stable or introduced here.
- Complete Model host-storage foundation, including `HostTensorStorage`,
  `MemorySegmentStorage`, and the exact six primitive carrier mappings.
- Complete OpenBLAS provider tasks 0001–0003 as a preserved low-level leaf; CPU task 0001 invokes
  none of its APIs.
- JDK 26 FFM and platform-thread APIs already selected by the root build.

## Follow-up tasks

- CPU 0002 remains Draft and blocked from Ready status by the separately authorized architecture
  synchronization replacing the current ASM-specific wording before any generated-kernel work.
- CPU 0003–0009 remain ordered Draft work for generation, caching, typed portable preparation,
  semantic-family coverage, and the portable closure checkpoint.
- CPU 0010–0016 remain Draft optional native-provider and compatible tuning-cache integration.
- Later CPU preparation must supply typed `BackendAnalysisInputs`, opaque
  `BackendPreparationPlan`, a `BackendPartitionPreparer`, a `BackendPartitionFinalizer`, concrete
  representation creators, and a schedule assembler only when at least one truthful operation
  route exists.
- Later Engine composition supplies the provider, availability/target facts, CPU preparation
  collaborators, and caller-input wrapping. It does not broaden this task's public surface.

## Architecture impact

Expected impact: None.

This task implements existing concrete-backend responsibilities and consumes existing Planning,
Prepare, Runtime, Backend Contract, Model, and Trace boundaries. It does not change the
authoritative CPU ASM wording or authorize generated kernels. If implementation requires an
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
  executable semantic coverage.
- The only public CPU type is the capability provider. Physical memory, typed arguments,
  executable specialization, and parallel coordination remain package-private in one cohesive
  execution package.
- Shared arenas are selected for run-owned native representations because later CPU worker and
  FFM access may cross threads during one run. One representation owns one arena, keeping cleanup
  and partial-failure rollback explicit.
- A segment binds to an exact primitive carrier with a carrier-relative byte offset only when the
  JDK exposes that matching carrier. Otherwise `CpuBufferArgument.Segment` retains the exact
  selected segment or slice and reports offset zero relative to it. This access form deliberately
  covers genuine native segments and heap-backed segments with unavailable carriers; it does not
  claim native provenance. Storage access is resolved once per selection, not encoded as a global
  mode.
- The parallel foundation owns fixed platform workers and bounded ranges without selecting thread
  count from Config, tuning evidence, or workload semantics. Those later preparation decisions
  remain outside this task.

## Known limitations

- CPU advertises and executes no Model operation after this task.
- No production CPU preparer, finalizer, schedule assembler, executable occurrence, transfer, or
  publication recipe exists yet.
- Borrowed confined segments can support only access from threads allowed by their JDK scope;
  parallel use must fail cold unless every selected worker is allowed.
- Internal native memory is one arena/allocation per representation. Pooling, reuse, aliasing,
  shared arenas, and persistent prepared memory are intentionally absent.
- Parallel cancellation is cooperative between bounded ranges and cannot undo writes made before
  a failure. Runtime validity remains the failure-safety boundary.
- Deterministic range boundaries do not by themselves define a deterministic reduction. Later
  reduction routes must define stable partial-result and combine order.
- Trace has no current backend/run payload family, and Config has no stable CPU prepare-target
  contract. This task preserves both boundaries without inventing replacements.
- On JDK 26, `MemorySegment.asReadOnly()` over a heap segment produces a read-only heap-backed
  segment whose `heapBase()` is empty. Such a segment therefore binds through the exact
  `CpuBufferArgument.Segment` form; the implementation preserves its identity, byte size,
  read-only state, ownership, liveness, and current-thread access without copying.

## Validation evidence

- Implementation development: `./gradlew :backends:cpu:compileJava` initially failed with seven
  record-constructor visibility errors and passed after their correction.
- Implementation development: `./gradlew :backends:cpu:compileTestJava` initially failed with one
  lambda-capture compiler error and passed after its correction.
- Implementation development: the first filtered CPU suite ran 12 tests with 11 passing and one
  failure because a read-only heap segment's JDK 26 `heapBase()` was empty. The resolved contract
  retains that exact segment as `CpuBufferArgument.Segment`; the replacement filtered run passed
  12 of 12 tests.
- Implementation development: the strengthened filtered suite first ran 16 tests with 15 passing
  and one test-only concurrent-close scheduling race. Synchronizing the test on `isClosed()`
  resolved the race; the replacement filtered run passed all 16 tests.
- Implementation final executable evidence: `./gradlew :backends:cpu:test` passed once after
  executable stabilization with `BUILD SUCCESSFUL`, 21 actionable tasks (2 executed and 19
  up-to-date), 16 tests, and no failures, errors, or skips. Production forbidden-feature scanning
  and `git diff --check` also passed before documentation handoff. No executable Java changed
  after this run.
- Separate clean documentation context `/root/cpu0001_docs` reviewed the final 13 production
  paths, six test paths, generated public Javadoc, CPU guide, glossary, task/master/roadmap, and
  directly relevant Planning, Prepare, Runtime, Backend Contract, Model storage, Trace, Config,
  OpenBLAS, architecture, ADR 0010/0011, and Java 26 build contracts. It changed Javadocs and the
  five authorized documentation/planning paths only; executable Java and tests remained unchanged.
- Documentation validation: `./gradlew :backends:cpu:javadoc` passed with 11 actionable tasks
  (2 executed and 9 up-to-date). After the last Javadoc wording correction, the same command was
  rerun and passed with the same task counts. Generated `CpuCapabilityProvider` and package pages
  were inspected.
- The first local Markdown-link checker invocation had a checker-only Ruby regular-expression
  interpolation syntax error and inspected no result. The corrected checker then exposed an
  anchor-normalization ambiguity around slash-separated headings; after matching the repository's
  accepted collapsed-hyphen form, the final five-file relative-link and heading-anchor check
  passed. Five-file fence/final-newline and production/documentation trailing-whitespace checks
  passed.
- Exact expected-versus-actual 24-path comparison, one-and-only-one CPU detailed specification,
  absence of CPU 0002–0016 specifications, and unchanged Gradle/architecture/ADR/architecture-
  test/backend-conformance/integration paths passed. A first status script assumed the value
  immediately followed the `## Status` heading; the corrected nonblank-line check passed task,
  master-plan, and roadmap synchronization. Public/package-private shape, forbidden mechanism,
  stale-status, resolved Segment-content, and generated-page scans passed.
- Final coordinator audit found one stale current-state roadmap sentence that still called task
  0001 Ready. The clean documentation context corrected it to Complete while preserving the
  explicitly historical earlier planning-pass sentence, then reran the applicable five-file
  Markdown, status, exact-scope, whitespace, and `git diff --check` gates. No executable Java,
  tests, or Javadocs changed, so Java tests and Javadoc generation were not rerun. The first
  combined status invocation treated ripgrep's no-match output as a literal zero; the corrected
  zero-match count passed without requiring a repository change.
- Final targeted Markdown/link/anchor/fence/newline/whitespace, public/package-private shape,
  forbidden-content, exact 24-path, no-later-specification, unchanged excluded-path, synchronized
  status, and `git diff --check` checks passed as recorded by `/root/cpu0001_docs`.

## Implementation notes

- Replaced the CPU marker with the sole public `CpuCapabilityProvider`; it keeps exact `cpu`
  identity and unconditional fail-closed support.
- Added package-private run-owned native buffer/workspace allocation, non-owning borrowed storage,
  direct typed cold arguments, the Runtime executable binding seam, and fixed-worker range
  coordination within `io.github.pho001.synaptik.backend.cpu.execution`.
- The final binding implementation classifies an observable matching primitive carrier to its
  typed array argument. If `heapBase()` is unavailable, it retains the exact segment or slice in
  `CpuBufferArgument.Segment`; `byteOffset()` is zero because the retained segment is already the
  selected region. No copy or native-provenance assertion is made.
- Tests lock all six carrier mappings, heap-slice offsets, read-only heap exact-segment retention,
  native and mixed binding, lifetime/access/alignment rejection, direct hot fields, allocation,
  worker range, cancellation, interruption, failure, concurrency, and shutdown contracts.

## Completion summary

- Completed changes: truthful CPU identity/capability provider; aligned shared-arena native
  buffer/workspace foundation; exact borrowed storage and typed cold binding; direct CPU prepared
  invocation seam; fixed bounded worker/range coordination; finalized Javadocs, CPU guide,
  glossary, and synchronized planning records.
- Files changed or created: the 13 authorized CPU production paths (including marker removal), six
  authorized CPU test paths, `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task,
  the CPU master plan, and the roadmap.
- Tests and validation: focused development evidence and the one final 16-test CPU module suite
  passed; the clean documentation context reused that executable evidence and passed CPU Javadoc,
  generated-page inspection, targeted documentation/scope/surface/status/content checks, and
  final whitespace validation.
- Documentation-agent review: `/root/cpu0001_docs`, using General, API/Javadoc, Planning, and
  Backend Guide profiles, finalized the combined change without changing executable behavior.
- Documentation impact: CPU guidance now distinguishes the implemented non-computational
  foundation from planned routes and explains exact array/segment cold binding.
- Javadoc review: every CPU production declaration and contract-relevant member was reviewed for
  ownership, lifetime, threading, nullability, inputs, results, and failures; final generation
  passed.
- Glossary impact: current implementation status and the reusable CPU buffer argument distinction
  are now documented, including the JDK 26 read-only heap behavior.
- Architecture, dependency, build, ADR, architecture-test, backend-conformance, and integration
  impact: none; the existing contracts and excluded paths remain unchanged.
- Unresolved issues: None.
- Follow-up required: None for task 0001. CPU 0002 remains Draft and requires the separately
  authorized architecture wording synchronization before it can become Ready.

Status: Complete
