# Task 0002: Portable Class-File API Generator Foundation

## Status

Complete

## Goal

Establish one backend-private, deterministic Java 26 Class-File API foundation that can generate,
verify, define, retain, and directly invoke a tiny synthetic CPU probe kernel across the four
portable execution modes. The task creates reusable code-generation machinery and typed extension
seams only. It does not lower or execute a Model operation, select a production CPU route, or make
CPU capability reporting truthful for any operation semantic.

The mental model is:

```text
later CPU analysis
  -> selects one complete typed specialization and family lowering
  -> shared Prepare assigns slots
later CPU finalization
  -> calls this generator foundation
  -> verified hidden class + exact MethodHandle entry point
later cold binding
  -> CpuBufferArgument direct fields -> exact generated entry arguments
run
  -> invoke only already-selected generated code
```

Task 0002 validates the middle generator step with test-only probe lowerings. It does not implement
either surrounding production CPU preparation step.

## Scope

- Add a package-private generator schema/version contract and a deterministic typed specialization
  descriptor plus content fingerprint.
- Describe exactly the facts that may change emitted bytecode: family-lowering fingerprint,
  ordered typed carrier/access signature, baked extents and element strides, exact portable mode,
  Vector API species and byte order when applicable, unroll/tile/tail structure, exact/default
  numerical mode, combine-order choice, and Java class-file target.
- Exclude facts that do not change generated code: graph/runtime identities, physical addresses,
  slot numbers, run objects, worker-group identity, thread count, chunk size, cache state, and
  mutable target observations.
- Add one family-owned emission collaboration with separate scalar and Vector API entry methods.
  Test source supplies the only implementation. Later family tasks add their own typed lowering
  implementations beside their semantics; this task adds no family registry or implementation.
- Add separate low-level scalar, Vector API, carrier, range/tile/tail, and reduction-structure
  emitters. They expose bytecode construction primitives, not Model operation meaning.
- Support exact direct entry signatures containing primitive heap carriers, exact
  `MemorySegment` arguments, or both. Heap byte offsets and every non-baked extent/range remain
  primitive entry arguments; generation and invocation perform no copy.
- Preserve task-0001 access meaning: read-only arguments may be read but never emitted as stores;
  write-only arguments may be written but never emitted as loads; read-write arguments permit
  both. Exact-segment writability remains a cold-binding obligation and is also locked into the
  specialization signature.
- Implement exactly four `CpuPortableExecutionMode` values: scalar single-thread, scalar
  parallel, Vector API single-thread, and Vector API parallel.
- Make `CpuPortableExecutionMode` the sole owner of structural scalar-versus-Vector emission
  dispatch. It constructs only the selected mode's scalar or Vector emitter plus the shared
  carrier, loop, and reduction emitters, then invokes exactly one matching family callback.
- Reuse `CpuRangeBody`, `CpuWorkerGroup`, and `CpuBufferArgument` in tests. Generated parallel
  entries accept an already-selected half-open range; `CpuWorkerGroup` owns dispatch,
  cancellation, synchronization, and failure propagation.
- Generate one static typed entry method with Java 26 `ClassFile`/`CodeBuilder`, verify the bytes,
  define it as a nestmate hidden class from a full-privilege CPU-package lookup, resolve its exact
  static `MethodHandle`, and retain the hidden lookup/class/handle/bytes in one immutable artifact.
- Prove the generated artifact can be invoked directly for bounded synthetic probes over heap,
  exact-segment, and mixed signatures in all four modes.
- Add the CPU-module-only `jdk.incubator.vector` compile, test-runtime, and Javadoc module flags
  required by Java 26. Keep the root toolchain/release at 26 and add no preview flag or dependency.
- Keep `CpuCapabilityProvider.supports(...)` unconditionally `false` and lock the unchanged public
  surface in tests.
- Finalize affected Javadocs, CPU guidance, glossary impact, and planning evidence through the
  mandatory separate clean documentation-focused pass after implementation.

## Out of scope

- any Model operation kind, operation-family implementation, semantic lowering, fused-partition
  implementation, numerical algorithm, Tensor value evaluation, or advertised CPU operation
  capability;
- public CPU APIs or a public code-generation, kernel, route, storage, vector, or execution API;
- CPU capability-provider changes, route selection, candidate generation, target discovery,
  tuning, numerical-policy selection, or determinism-policy selection;
- a generated-artifact cache, cache key/store/lookup/eviction/single-flight mechanism, persistent
  bytes, class serialization, or task 0003 behavior;
- a production `BackendPartitionPreparer`, `BackendPartitionFinalizer`, CPU analysis plan,
  finalization orchestrator, schedule assembler, representation plan, physical allocation change,
  materialization decision, or task 0004 behavior;
- worker-pool, scheduler, thread-count, chunk-size, cancellation, or failure-policy changes;
- OpenBLAS, oneMKL, oneDNN, Accelerate, AOCL, ZenDNN, Metal, CUDA, vendor, native-provider, or
  Foreign Function and Memory invocation routes;
- compiler, Planning, shared Prepare, Runtime, Engine, Config, Model, Backend Contract, Trace,
  OpenBLAS-provider, backend-conformance, integration-test, or architecture-test changes;
- architecture contract, focused architecture explanation, ADR, kernel-routes guide, CPU strategy
  note, or general coding-rule changes;
- root Gradle, settings, shared build logic, dependency coordinates, ASM, another bytecode library,
  `--enable-preview`, or another incubator module;
- broad execution algorithms, scalar/Vector performance claims, arbitrary vector-lane promises,
  benchmarks, or a real Tensor-operation result; and
- detailed task specifications for CPU 0003 or any later CPU task.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core invariants
  - `modules/runtime`
  - `modules/prepare`
  - Concrete backend modules
  - CPU backend routes
  - Prepare lifecycle
  - Run lifecycle
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0002: Backend-owned lowering](../../../../design/decisions/0002-backend-owned-lowering.md)
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [CPU kernel strategy](../../../../design/notes/cpu-kernel-strategy.md)
- [Task 0001 CPU capability, representation, binding, and parallel foundation](0001-cpu-capability-representation-binding-and-parallel-foundation.md)

## Architecture constraints

- Generated JVM bytecode is a CPU-internal computation route. It does not create a separate
  backend or shared lowering module.
- CPU analysis later owns semantic lowering, route choice, specialization, fusion, and exact
  shared-resource declaration. Task 0002 accepts an already-complete specialization and a
  family-owned emitter; it performs none of those choices.
- CPU finalization may generate and define a selected class only after shared slot assignment.
  The generator API must therefore be usable from a later CPU finalizer, but task 0002 must not
  invent that finalizer or call shared Prepare contracts.
- Runtime receives only a finalized immutable executable recipe. Generated hot code must not see
  `Operation`, `CompiledNode`, `Tensor`, `PrepareContext`, `BackendPartitionAnalysis`, Runtime
  slots, `RunState`, or nominal representation objects.
- Generated code must not discover storage, inspect `heapBase()`, choose routes, look up cache
  entries, select scalar versus Vector API, decide parallelism, own workers, or switch on Model
  semantics. Cold CPU binding and later finalization resolve those facts once.
- Hidden-class definition and direct entry-point construction are generation/finalization
  mechanics. Keeping the artifact reachable keeps its hidden lookup, class, and method handle
  reachable. Once neither a future cache nor a prepared executable retains the artifact, the
  implementation may become unloadable; this task must not promise or force a collection time.
- Task 0001's `CpuBufferArgument` variants remain the cold storage classification. Generation
  consumes only the corresponding direct carrier/segment signature and never copies or owns the
  carrier or segment.
- Task 0001's `CpuWorkerGroup` remains the sole CPU worker owner. A parallel generated entry is a
  range body, not a scheduler.
- `jdk.incubator.vector` is a task-local CPU implementation dependency supplied by JDK 26, not a
  new Gradle library or architecture edge. No Java Platform Module System descriptor is added.
- Capability remains fail-closed because a synthetic probe is not operation coverage.
- If implementation needs a new public type, shared contract, dependency edge, production CPU
  preparer/finalizer, operation semantic, cache, or architecture change, stop and report the
  conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — unchanged public capability provider used only by
  public-surface regression tests.
- `io.github.pho001.synaptik.backend.cpu.execution` — owns all new package-private generation,
  specialization, artifact, and emission types beside task 0001's package-private buffer and
  worker foundations.

Packages added or changed:

- No Java package is added. The existing CPU execution package is extended because its new types
  must consume package-private `CpuBufferArgument`, `CpuRangeBody`, and `CpuWorkerGroup` without
  widening them or manufacturing a public cross-package API.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.execution.CpuGeneratorSchema` — schema constants and
  canonical fingerprint encoding owner.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuPortableExecutionMode` — the exact closed
  four-mode vocabulary.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuLoweringFingerprint` — immutable typed
  family-lowering digest supplied by the family owner.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuKernelSpecialization` — immutable complete
  bytecode-relevant descriptor and derived specialization fingerprint.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuFamilyKernelEmitter` — family-specific
  scalar/Vector emission seam; test source supplies the only task-0002 implementation.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuScalarEmitter` — scalar instruction helper.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuVectorEmitter` — Vector API instruction and
  species helper.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuCarrierEmitter` — exact heap-carrier and
  `MemorySegment` load/store helper.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuLoopEmitter` — range, tile, unroll, and tail
  control-flow helper.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuReductionEmitter` — structural partial and
  combine-order helper with no mathematical reduction policy.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuClassFileKernelGenerator` — narrow
  build/verify/define/resolve orchestrator.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuGeneratedKernel` — immutable hidden-class
  lifetime and exact entry-point artifact.

Every new production declaration is package-private. Do not add `generated`, `codegen`, `kernel`,
`vector`, `util`, `registry`, `service`, or `manager` packages.

## Exact package-private surface

The source-level surface below is fixed for this task. Record canonical constructors and
accessors may be declared explicitly for validation and Javadoc. A signature change requires
updating this specification before implementation continues.

```java
final class CpuGeneratorSchema {
    static final int CURRENT_VERSION = 1;
    static final int CLASSFILE_MAJOR_VERSION = ClassFile.JAVA_26_VERSION;
    static final int FINGERPRINT_BYTE_COUNT = 32;
}

enum CpuPortableExecutionMode {
    SCALAR_SINGLE_THREAD,
    SCALAR_PARALLEL,
    VECTOR_API_SINGLE_THREAD,
    VECTOR_API_PARALLEL;

    boolean vectorized();
    boolean parallel();
    void emit(
            CodeBuilder code,
            CpuKernelSpecialization specialization,
            CpuFamilyKernelEmitter familyEmitter);
}

final class CpuLoweringFingerprint {
    static CpuLoweringFingerprint of(byte[] canonicalFamilyBytes);
    static CpuLoweringFingerprint fromDigest(byte[] digest);
    byte[] bytes();
    // content-based equals/hashCode; lowercase fixed-width hexadecimal toString
}

final class CpuKernelSpecialization {
    enum Carrier {
        DOUBLE_ARRAY, FLOAT_ARRAY, SHORT_ARRAY, INT_ARRAY, LONG_ARRAY, BYTE_ARRAY, MEMORY_SEGMENT
    }

    enum Tail {
        NONE, SCALAR, MASKED
    }

    enum NumericalMode {
        EXACT_DEFAULT
    }

    enum CombineOrder {
        FIXED, UNRESTRICTED
    }

    record Argument(
            DataType dataType,
            Carrier carrier,
            PreparedExecutable.BufferAccess access,
            boolean byteOffsetBaked,
            long bakedByteOffset,
            List<Long> bakedElementStrides) {}

    record VectorShape(DataType laneType, int vectorBitSize, int laneCount) {}

    CpuKernelSpecialization(
            int schemaVersion,
            CpuLoweringFingerprint loweringFingerprint,
            CpuPortableExecutionMode executionMode,
            List<Argument> arguments,
            List<Long> bakedExtents,
            int dynamicExtentCount,
            VectorShape vectorShape,
            ByteOrder byteOrder,
            int unrollFactor,
            long tileElementCount,
            Tail tail,
            NumericalMode numericalMode,
            CombineOrder combineOrder,
            int classFileMajorVersion);

    int schemaVersion();
    CpuLoweringFingerprint loweringFingerprint();
    CpuPortableExecutionMode executionMode();
    List<Argument> arguments();
    List<Long> bakedExtents();
    int dynamicExtentCount();
    Optional<VectorShape> vectorShape();
    ByteOrder byteOrder();
    int unrollFactor();
    long tileElementCount();
    Tail tail();
    NumericalMode numericalMode();
    CombineOrder combineOrder();
    int classFileMajorVersion();
    MethodType entryType();
    CpuLoweringFingerprint specializationFingerprint();
    // structural equals/hashCode over every component; diagnostic-only toString
}

interface CpuFamilyKernelEmitter {
    CpuLoweringFingerprint loweringFingerprint();
    void emitScalar(
            CpuScalarEmitter scalar,
            CpuCarrierEmitter carriers,
            CpuLoopEmitter loops,
            CpuReductionEmitter reductions);
    void emitVector(
            CpuVectorEmitter vector,
            CpuCarrierEmitter carriers,
            CpuLoopEmitter loops,
            CpuReductionEmitter reductions);
}

final class CpuScalarEmitter {
    CodeBuilder code();
    CpuKernelSpecialization specialization();
    TypeKind typeKind(DataType dataType);
    int allocateLocal(DataType dataType);
    void loadLocal(DataType dataType, int slot);
    void storeLocal(DataType dataType, int slot);
}

final class CpuVectorEmitter {
    CodeBuilder code();
    CpuKernelSpecialization specialization();
    CpuKernelSpecialization.VectorShape vectorShape();
    ClassDesc vectorClass();
    ClassDesc speciesClass();
    void loadSpecies();
}

final class CpuCarrierEmitter {
    CodeBuilder code();
    CpuKernelSpecialization specialization();
    void emitScalarLoad(int argumentIndex, int elementIndexLocal);
    void emitScalarStore(int argumentIndex, int elementIndexLocal);
    void emitVectorLoad(int argumentIndex, int elementIndexLocal);
    void emitVectorStore(int argumentIndex, int elementIndexLocal);
}

final class CpuLoopEmitter {
    @FunctionalInterface interface IndexedBody { void emit(int elementIndexLocal); }
    @FunctionalInterface interface TiledBody {
        void emit(int tileStartLocal, int tileEndLocal, int tileIndexLocal);
    }

    CodeBuilder code();
    CpuKernelSpecialization specialization();
    void emitRange(IndexedBody body);
    void emitTiles(TiledBody body);
    void emitTail(IndexedBody scalarBody, IndexedBody maskedVectorBody);
}

final class CpuReductionEmitter {
    @FunctionalInterface interface PartialBody {
        void emit(int rangeIndexLocal, int partialValueLocal);
    }
    @FunctionalInterface interface CombineBody {
        void emit(int leftValueLocal, int rightValueLocal, int resultValueLocal);
    }

    CodeBuilder code();
    CpuKernelSpecialization specialization();
    void emitPartials(DataType accumulatorType, PartialBody body);
    void emitCombine(DataType accumulatorType, CombineBody body);
}

final class CpuClassFileKernelGenerator {
    CpuGeneratedKernel generate(
            CpuKernelSpecialization specialization,
            CpuFamilyKernelEmitter familyEmitter);
}

final class CpuGeneratedKernel {
    CpuKernelSpecialization specialization();
    MethodHandles.Lookup hiddenLookup();
    Class<?> hiddenClass();
    MethodHandle entryPoint();
    MethodType entryType();
    byte[] classBytes();
    // identity equality; no close lifecycle
}
```

`CodeBuilder` exposure is deliberately confined to `CpuPortableExecutionMode`, the separate
low-level emitters, and the family-owned callback. `CpuPortableExecutionMode.emit(...)` owns only
structural scalar-versus-Vector dispatch: it constructs the mode-appropriate
`CpuScalarEmitter` or `CpuVectorEmitter`, constructs the shared `CpuCarrierEmitter`,
`CpuLoopEmitter`, and `CpuReductionEmitter`, and calls exactly one of `emitScalar` or
`emitVector`. It performs no Model-semantic, family, route, cache, storage, worker, target, or
run-time dispatch. `CpuClassFileKernelGenerator` delegates exactly once to that method and may
otherwise orchestrate only class name/version, one static `invoke` method, verification, hidden
definition, and exact handle resolution. It must not contain a scalar/Vector branch or any
carrier, loop, reduction, family, operation, or route switch.

## Schema, specialization, and fingerprint contract

- Schema version `1` identifies the canonical binary encoding and generator behavior established
  by this task. It is not the task number, project version, JDK version, or cache schema.
- `CpuLoweringFingerprint.of(...)` requires a non-null, non-empty canonical family byte sequence,
  copies it, hashes it with SHA-256 during cold construction, and retains exactly 32 digest bytes.
  Tests lock a fixed vector. A later family owns the canonical typed encoding of its semantic
  lowering; task 0002's test probe uses one fixed test-only encoding.
- `CpuLoweringFingerprint.fromDigest(...)` is the internal trusted-digest factory used only for
  the derived specialization fingerprint. It validates null first with
  `NullPointerException("digest")`, then requires exactly 32 bytes with
  `IllegalArgumentException("digest must contain exactly 32 bytes")`, makes exactly one defensive
  copy, and retains those bytes without hashing them again.
- `CpuKernelSpecialization.specializationFingerprint()` is SHA-256 over one versioned canonical
  big-endian encoding. Each enum has an explicit stable integer tag; enum ordinal and
  `toString()` are forbidden. Every list is length-prefixed and order-sensitive. Every long and
  int uses fixed-width signed big-endian representation. Boolean values use one byte `0` or `1`.
- The specialization encoding order is exactly: schema version, lowering digest, execution mode,
  ordered arguments with data type/carrier/access/offset-baked flag/baked offset/ordered strides,
  ordered baked extents, dynamic-extent count, optional vector shape, exact byte order, unroll
  factor, tile count, tail mode, numerical mode, combine order, and class-file major version.
- The deterministic generated class name is derived from the complete specialization fingerprint
  inside `io.github.pho001.synaptik.backend.cpu.execution`. It is diagnostic identity only; hidden
  classes still have distinct JVM identities and are not found by name.
- `Argument.bakedByteOffset` is non-negative. For primitive-array carriers it is the constant
  carrier byte offset when `byteOffsetBaked` is true and must be aligned to
  `dataType.byteWidth()`; misalignment fails with
  `IllegalArgumentException("bakedByteOffset must be aligned to dataType byte width")`. When
  `byteOffsetBaked` is false, `bakedByteOffset` must be zero and one exact primitive `long`
  byte-offset argument appears in `entryType()`; dynamic array offsets are primitive runtime
  arguments, and their runtime alignment and bounds are outside this foundation task.
  `MEMORY_SEGMENT` requires `byteOffsetBaked == true` and the exact baked offset zero because it
  always uses one exact selected segment.
- The exact `Argument` validation order is: non-null `dataType`, non-null `carrier`, non-null
  `access`, non-baked-offset consistency (`bakedByteOffset must be zero when byteOffsetBaked is
  false`), non-negative baked offset (`bakedByteOffset must be non-negative`), non-null strides
  list and ordered non-null/non-negative stride entries, the exact-segment baked-zero rule
  (`MEMORY_SEGMENT requires a baked zero byte offset`) when the carrier is `MEMORY_SEGMENT`,
  otherwise exact data-type/carrier match (`carrier does not match dataType`), and finally baked
  primitive-array width alignment (`bakedByteOffset must be aligned to dataType byte width`). The
  alignment check therefore never precedes an existing null, consistency, range, stride,
  exact-segment, or carrier-match failure.
- An argument carrier must match its `DataType` exactly as task 0001 does: FLOAT64/double[],
  FLOAT32/float[], BFLOAT16/short[], INT32/int[], INT64/long[], BOOL/byte[], while
  `MEMORY_SEGMENT` accepts every current type. No generic `Object`, carrier name, or class-name
  string is encoded.
- Baked extents and strides are non-negative. Non-baked extents are represented only by
  `dynamicExtentCount` and exact ordered primitive `long` entry parameters. Runtime IDs and
  symbolic dimension identities are never encoded.
- Vector modes require one `VectorShape`; scalar modes forbid it. The shape must resolve exactly
  to a Java 26 supported species for FLOAT64, FLOAT32, INT32, or INT64, with positive power-of-two
  bit size and matching lane count. BFLOAT16 and BOOL vector semantics are not claimed by this
  foundation. Unsupported species fail before class bytes are built.
- `byteOrder` is exactly `ByteOrder.LITTLE_ENDIAN` or `ByteOrder.BIG_ENDIAN` and uses a stable
  explicit fingerprint tag. Later CPU analysis normally supplies `ByteOrder.nativeOrder()`.
  Segment scalar/Vector access emits the selected order directly; generated hot code performs no
  byte-order switch or target discovery.
- `unrollFactor` and `tileElementCount` are positive. Scalar modes require `Tail.NONE`; after the
  non-null tail check, any other scalar tail fails with
  `IllegalArgumentException("scalar execution mode requires Tail.NONE")`. Vector modes permit
  `NONE`, `SCALAR`, and `MASKED`: `NONE` requires later analysis to prove exact divisibility,
  while `SCALAR` and `MASKED` represent the selected remainder structures under the same existing
  divisibility and tail contracts. Task-0002 scalar probes use only `NONE`.
- In the exact `CpuKernelSpecialization` constructor order, scalar-tail compatibility is checked
  after schema, lowering fingerprint, execution mode, argument snapshot, baked extents, dynamic
  extent count, vector-shape presence/absence, byte order, positive unroll factor, positive tile
  element count, and non-null tail, but before numerical mode, combine order, and class-file
  version. Thus a scalar `SCALAR` or `MASKED` tail fails at that position with exactly
  `scalar execution mode requires Tail.NONE`; no earlier or later invariant displaces it.
- Schema version 1 permits only `NumericalMode.EXACT_DEFAULT`; this records the current exact/default
  arithmetic contract without inventing relaxed or fast math. `CombineOrder` records whether
  emitted partial-combine control flow is fixed or unrestricted. Neither value is a public Config
  policy, and fixed control flow alone does not claim deterministic floating results.
- `classFileMajorVersion` must equal `ClassFile.JAVA_26_VERSION` in schema version 1. A later JDK or
  emission change requires a schema decision rather than silent reuse.
- The constructor validates and snapshots every list before computing `entryType()` or the
  fingerprint. Returned lists are unmodifiable; fingerprint and class-byte accessors return
  defensive copies. Equality and hashing never depend on array identity, lookup/class/handle
  identity, object addresses, or mutable external state.
- Thread count, worker count, minimum range size, chunk size, `BufferSlot`, `WorkspaceSlot`,
  `ValueId`, `NodeId`, `TensorId`, model/run identity, `RunState`, carrier object identity,
  `MemorySegment.address()`, lookup identity, hidden-class identity, cache state, and timing data
  are forbidden fingerprint inputs.

## Direct entry signature and access contract

- The generated class has no instance fields, static mutable fields, constructor use, class
  initializer, native method, bridge, reflection, service lookup, or generic entry point. It has
  exactly one package-private static `void invoke(...)` method plus JVM-required class metadata.
- Ordered buffer arguments come first. An array carrier contributes its exact primitive array
  class and, when the byte offset is not baked, one adjacent primitive `long` byte-offset
  parameter. An exact segment contributes `MemorySegment` only.
- Non-baked extents follow in descriptor order as primitive `long` parameters.
- Single-thread modes then accept one primitive `long elementCount`. Parallel modes instead
  accept primitive `long startInclusive`, `long endExclusive`, and `int rangeIndex`. The
  generated parallel method executes only that range and never submits work.
- The descriptor derives one exact `MethodType`; the class builder uses the matching
  `MethodTypeDesc`; hidden lookup resolves `findStatic(hiddenClass, "invoke", entryType)` without
  `asType`, reflection, proxy, lambda metafactory, spreader, collector, `invokeWithArguments`, or
  raw `Object` adaptation.
- `CpuCarrierEmitter` validates argument index, access direction, data type, carrier kind,
  element-width alignment, and supported vector species during generation. Invalid access fails
  closed before class definition. Generated code contains only the already-specialized primitive
  array or exact-segment access sequence.
- Heap carriers and exact segments are borrowed for invocation. The generated method neither
  copies nor closes them. Read-only exact-segment rejection remains task 0001's cold binding
  responsibility; an emitted write is additionally forbidden when descriptor access is
  `READ_ONLY`.
- Mixed signatures preserve exact argument order and require no global heap/native mode.

## Scalar, Vector API, loop, tile, reduction, and tail boundaries

- `CpuScalarEmitter` owns primitive local/type-kind mechanics only. It has no operation-kind or
  family switch and does not define arithmetic semantics.
- `CpuVectorEmitter` resolves the exact selected `VectorSpecies` constant and emits direct
  `FloatVector`, `DoubleVector`, `IntVector`, or `LongVector` calls. It does not choose a species,
  vectorize a scalar lowering automatically, or promise support for BFLOAT16/BOOL semantics.
- `CpuCarrierEmitter` is executable for exact scalar/vector load and store plumbing across all six
  heap carrier forms and exact `MemorySegment`; tests may exercise only the bounded types needed
  to prove each structural path. Later operation-family tasks own semantic type coverage.
- `CpuLoopEmitter.emitRange` emits one counted half-open loop. In single-thread mode its bounds
  are `[0, elementCount)`; in parallel mode they are the supplied `[startInclusive,
  endExclusive)`. It emits no dispatch or cancellation check inside an element loop.
- `emitTiles` derives deterministic non-empty tiles from the already-baked tile size inside the
  supplied range. It does not choose tile size, allocate workspace, or schedule a tile.
- `emitTail` is a Vector-mode structural helper and emits only the already-selected `NONE`,
  scalar, or masked-vector tail. Scalar modes never enter it with `SCALAR` or `MASKED`. It does
  not inspect workload facts or choose a policy at run time.
- `CpuReductionEmitter` provides structural local-slot partial and fixed/unfixed combine-order
  control flow only. The family callback supplies mathematical partial/combine instructions. No
  production task-0002 callback supplies a sum, mean, extrema, scan, softmax, or other Model
  semantic. The synthetic test may use tiny arithmetic solely to validate the seam and ordering.
- None of these helpers owns `CpuWorkerGroup`, an executor, a pool, a scheduler, a cache, storage
  discovery, route choice, or a semantic registry.

## Generation, definition, and failure contract

Validation order for `CpuClassFileKernelGenerator.generate(...)` is:

1. non-null specialization (`specialization`);
2. non-null family emitter (`familyEmitter`);
3. exact lowering fingerprint equality, else
   `familyEmitter lowering fingerprint does not match specialization`;
4. specialization/schema/species/signature invariants, already established by construction;
5. class-byte emission;
6. `ClassFile.of().verify(bytes)` with no verification errors, else
   `generated class verification failed` plus stable indexed verification details;
7. `MethodHandles.lookup().defineHiddenClass(bytes, false, NESTMATE)`;
8. exact static `invoke` resolution; and
9. immutable artifact construction.

Unchecked emitter failures propagate unchanged. Class-File API argument/verification failures and
checked hidden-definition/handle-resolution failures are wrapped once in
`IllegalArgumentException("generated kernel definition failed", cause)`, except for the explicit
verification message above. No partial result is installed in static state, registered, cached,
closed, or retried. Generation is stateless and safe for concurrent calls; equal specializations
and emitters produce byte-for-byte equal bytes before hidden definition, while their hidden
classes/artifacts remain distinct because caching is deferred.

`CpuGeneratedKernel` retains exact specialization, hidden lookup, hidden class, direct method
handle, exact method type, and a private copy of bytes. Construction verifies lookup-class,
handle-owner, and type consistency. It owns no `close()` lifecycle and overrides neither `equals`
nor `hashCode`. No static collection retains generated artifacts.

## Validation order and stable failures

Automated tests lock at least these failures in order:

1. `CpuLoweringFingerprint.of`: source null, source empty; `fromDigest`: digest null, exact
   32-byte length, then defensive retention without rehashing.
2. `Argument`: data type, carrier, access, offset-baked consistency, non-negative baked offset,
   ordered non-negative strides, exact data-type/carrier match or exact-segment baked-zero rule,
   then baked primitive-array width alignment. Dynamic primitive-array offsets receive no
   generation-time alignment validation.
3. `VectorShape`: lane type, bit size, lane count, then exact supported-species resolution.
4. `CpuKernelSpecialization`: schema, lowering fingerprint, mode, argument list and entries,
   baked extents, dynamic count, vector presence/absence, byte order, unroll, tile, non-null tail,
   scalar-mode `Tail.NONE` compatibility, numerical mode, combine order, and class-file version.
5. Generator inputs and lowering-fingerprint match before class-byte work.
6. Carrier emitter argument range, read/write access, scalar/vector compatibility, alignment, and
   only then instruction emission.

Stable messages use exact parameter/index names such as `arguments[1]`, `bakedExtents[0]`,
`bakedElementStrides[2]`, and `dynamicExtentCount must be non-negative`. Ordinary Class-File
API/JDK verification detail may remain a cause/detail list rather than a new public exception.

## Affected files

Expected CPU production/build paths:

- `backends/cpu/build.gradle.kts`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortableExecutionMode.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuLoweringFingerprint.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuFamilyKernelEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuScalarEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuVectorEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuCarrierEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuLoopEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuReductionEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuGeneratedKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/package-info.java`

Expected CPU test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuGeneratedKernelShapeTest.java`

Expected explanatory documentation:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`

Expected planning paths:

- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review only, with a reasoned no-change conclusion unless a concrete contradiction is found:
architecture/focused architecture/ADRs, kernel-routes guide, CPU strategy note, documentation
rules/profiles, Planning/Prepare/Runtime/Config/Model contracts, task-0001 source/tests/Javadocs,
root/settings/other Gradle files, architecture tests, backend conformance, integration tests, and
CPU task 0001.

## Maximum scope

At most 22 paths:

- 14 CPU production/build paths;
- 3 CPU test paths;
- 2 explanatory documentation paths; and
- 3 planning paths.

No existing CPU production or test path other than `execution/package-info.java` and the CPU
module build file may change. If another path, type, test split, build file, dependency, public
surface, or semantic implementation is needed, stop and update planning or propose a bounded
follow-up.

## Java 26 Class-File and Vector API build contract

- The root build continues to select Java toolchain 26 and `options.release = 26`.
- `java.lang.classfile` and `CodeBuilder` are stable Java 26 APIs and require no preview or
  incubator flag.
- Java 26 `jdk.incubator.vector` requires explicit module activation. The CPU build file must add
  `--add-modules jdk.incubator.vector` to every CPU `JavaCompile`, the same module to CPU `Test`
  JVM arguments, and the equivalent option to the CPU `Javadoc` task. The intended task-local
  Gradle shape is:

  ```kotlin
  tasks.withType<JavaCompile>().configureEach {
      options.compilerArgs.addAll(listOf("--add-modules", "jdk.incubator.vector"))
  }

  tasks.withType<Test>().configureEach {
      jvmArgs("--add-modules", "jdk.incubator.vector")
  }

  tasks.named<Javadoc>("javadoc") {
      (options as StandardJavadocDocletOptions).apply {
          memberLevel = JavadocMemberLevel.PACKAGE
          addStringOption("-add-modules", "jdk.incubator.vector")
      }
  }
  ```

  Gradle prefixes the Javadoc option key, so `-add-modules` produces the tool's
  `--add-modules` option. The task must confirm the effective invocation instead of weakening the
  check if the local Gradle/Javadoc API differs.
- Do not add `--enable-preview`, a `module-info.java`, a Maven dependency, a root/subproject-wide
  flag, or an option to another module. Final evidence must prove compile, test execution, and
  Javadoc all receive the CPU-local Vector module.
- Incubator warnings are expected and are not semantic failures. Any missing-module failure in
  compile, test, or Javadoc is a task failure, not a reason to omit Vector validation.
- Generated classes link only against Java 26/JDK classes already visible to the CPU module. No
  ASM or other bytecode dependency is permitted.

## Acceptance criteria

- [x] Exactly the listed package-private production types and no new public CPU type are present.
- [x] Schema version 1 and both typed fingerprints are immutable, content-based, deterministic,
      and locked by fixed canonical vectors plus equal/not-equal tests.
- [x] The specialization contains every and only bytecode-relevant fact listed above; deep
      immutability, structural equality/hash, entry-type derivation, validation order, and stable
      messages are tested.
- [x] Runtime identities, addresses, worker counts, chunk sizes, cache state, maps, strings used
      for dispatch, and mutable objects are absent from specialization equality/fingerprints.
- [x] The family seam is a direct typed collaboration with separate scalar/vector emission and no
      registry, service locator, reflection, string dispatch, `Map<String, ?>`, generic config bag,
      broad facade, or god generator.
- [x] `CpuPortableExecutionMode.emit(...)` constructs only the selected scalar or Vector emitter
      plus shared carrier/loop/reduction emitters and calls exactly one matching family callback;
      the generator delegates exactly once to it and contains no scalar/Vector switch.
- [x] Separate scalar, Vector, carrier, loop/tile/tail, and partial/combine helpers contain only
      their stated low-level responsibilities; no production family semantic is implemented.
- [x] Class bytes are deterministic, target Java 26, have exactly one static typed `invoke`
      method, verify through `ClassFile.of().verify(bytes)`, define as a hidden nestmate, and
      resolve to one
      exact direct static method handle without adaptation.
- [x] Equal generation inputs produce identical bytes but distinct hidden artifacts without
      task-0003 caching.
- [x] Heap primitive carriers, exact `MemorySegment` values, and a mixed signature are invoked
      without copying; array offsets, segment-relative zero offsets, access constraints, and
      direct typed argument order are proved.
- [x] Exactly scalar single-thread, scalar parallel, Vector API single-thread, and Vector API
      parallel modes exist. Synthetic probes execute in all four modes with equivalent bounded
      results.
- [x] Scalar modes reject `SCALAR` and `MASKED` tails with the exact specified message; Vector
      modes accept `NONE`, `SCALAR`, and `MASKED` under the stated divisibility/tail contracts.
- [x] Baked primitive-array offsets reject misalignment to `DataType.byteWidth()` in the exact
      stated validation position, exact segments retain baked zero offset, and dynamic array
      offsets receive no generation-time alignment check.
- [x] Parallel probes reuse `CpuWorkerGroup` and pass only assigned range primitives to generated
      entries; generated classes own no thread, pool, executor, scheduler, synchronization,
      cancellation, or failure mechanism.
- [x] Vector probes use one exact supported Java 26 species, full-lane and tail plumbing, heap and
      segment access, and no arbitrary lane promise.
- [x] Range, tile, scalar/masked tail, partial, and fixed-order combine structures are tested with
      synthetic callbacks and never described as Model operation semantics.
- [x] The artifact retains exact hidden lookup/class/handle while reachable; no static collection
      retains it; collection/unloading timing is not asserted.
- [x] Failures are closed before advertising or invocation, and malformed/unverified bytes never
      produce a returned artifact.
- [x] `CpuCapabilityProvider` remains byte-for-byte unchanged, returns `false` for every non-null
      query, and the public CPU surface advertises no operation capability.
- [x] Generated/hot paths contain no reflection, `ServiceLoader`, map lookup, string dispatch,
      storage discovery, route/cache choice, graph/slot lookup, generic `Object` argument,
      `invokeWithArguments`, or per-element allocation.
- [x] The CPU build alone activates `jdk.incubator.vector` for compile, test, and Javadoc; Java 26
      toolchain/release remain unchanged; no preview flag, module descriptor, dependency, or ASM
      is added.
- [x] Focused development tests and exactly one final CPU module test pass after executable Java
      stabilizes.
- [x] Every new/changed production declaration has meaningful complete Javadoc.
- [x] A separate clean documentation-focused pass finalizes affected Javadocs, CPU guidance,
      glossary impact, planning evidence, links, and status in the same overall change, reusing
      successful Java evidence unless it changes executable behavior.
- [x] CPU Javadoc, generated pages, Class-File parsing/verification, surface/mechanism/scope/status/
      Markdown/whitespace checks pass.

## Tests / validation

Implementation development may run focused classes:

```bash
./gradlew :backends:cpu:test --tests '*CpuKernelSpecializationTest'
./gradlew :backends:cpu:test --tests '*CpuClassFileKernelGeneratorTest' --tests '*CpuGeneratedKernelShapeTest'
```

After executable Java and CPU build flags stabilize, run exactly once:

```bash
./gradlew :backends:cpu:test
```

The suite must use Java 26 `ClassFile.of().parse(bytes)` and
`ClassFile.of().verify(bytes)` for class version, method/field shape, descriptors, references, and
deterministic bytes. It must use exact `MethodHandle.type()` and
`invokeExact` probes instead of recurring manual reflection or `javap` as the primary invariant.
If a bytecode risk cannot be expressed with the Class-File model, record one bounded `javap -v`
check against test-emitted bytes; otherwise record why manual `javap` was unnecessary.

Implementation pass:

```bash
git diff --check
```

Documentation-focused pass, after final Javadocs and explanatory text:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

The documentation pass reuses exact final CPU test evidence and does not rerun Java tests unless
it changes executable Java behavior or records a concrete stale-evidence risk.

Validate local Markdown links/anchors for the task, master plan, roadmap, CPU guide, and glossary;
fences, final newlines, and trailing whitespace; exact 22-path scope; exactly one detailed CPU
0002 specification; CPU 0001 `Complete`; CPU 0002 synchronized to its implementation status; CPU
0003–0016 `Draft` with no detailed specs; and unchanged excluded paths.

Repository-wide Java and backend-conformance validation are deferred to CPU task 0009 or CI.
Task 0002 changes only CPU-local incubator flags and backend-private machinery. If a dependency
edge, shared build rule, architecture boundary, or advertised behavior changes, stop.

## Dependencies

- Complete CPU 0001 exact heap/segment/mixed arguments, binding seam, and worker/range foundation.
- Synchronized architecture permission for generated JVM-bytecode CPU computation kernels in CPU
  finalization after shared slot assignment.
- Complete Prepare finalization and Runtime prepared-executable/bound-invocation contracts,
  preserved without a production CPU implementation here.
- Java 26 stable Class-File API and CPU-local `jdk.incubator.vector` activation.
- Current Model `DataType` and exact six task-0001 carrier mappings. Shape, layout, descriptors,
  operation signatures, and occurrences inform later family fingerprints but are not hot/generated
  inputs here.
- Current Config has no stable CPU target, determinism, tuning, or execution-mode contract. Task
  0002 introduces no substitute public config; its descriptor receives selected bytecode facts.

## Follow-up tasks

- CPU 0003 remains Draft and owns bounded single-flight generated-artifact caching and lifetime.
- CPU 0004 remains Draft and owns production portable analysis, candidate/specialization decisions,
  and finalization orchestration after slot assignment.
- CPU 0005–0008 remain Draft and own family semantic coverage.
- CPU 0009 remains Draft and owns advertised portable coverage closure and conformance.
- CPU 0010–0016 remain Draft for optional native routes and tuning-cache selection.
- No follow-up may treat task-0002 probes as operation capability or a production route.

## Architecture impact

Expected impact: None.

This task implements the authorized CPU-internal mechanism and one CPU-local Java 26 incubator
module option. It changes no ownership, dependency edge, shared lifecycle, public API, or
architecture rule. Stop if implementation reveals a required shared or architecture change.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/backends/cpu/master-plan.md, and
docs/planning/backends/cpu/tasks/0002-portable-class-file-api-generator-foundation.md. Read the
directly referenced CPU 0001, Runtime/Prepare, Model, build/toolchain, architecture, documentation,
and Java 26 Class-File/Vector API contracts needed by the task.

Implement CPU task 0002 exactly as specified. Do not implement a Model operation, advertise CPU
semantic capability, add a production CPU preparer/finalizer, add caching, change a shared/public
API, add ASM/dependencies, or implement a later CPU task. Stop and report any architecture,
toolchain, package, or maximum-scope conflict instead of inventing a new boundary.

After implementation and the one final CPU test run, hand the diff and exact evidence to a
separate clean documentation-focused context. It must follow the documentation rules, finalize
affected Javadocs, CPU guidance, glossary impact, planning evidence, and documentation validation
in the same overall change, and reuse Java evidence unless executable behavior changes or a
concrete stale-evidence risk is recorded.

Update this task with local decisions, known limitations, validation evidence, implementation
notes, canonical completion summary, and synchronized final status. Do not mark Complete before
every acceptance criterion and the documentation pass succeed.
```

## Local decisions

- Keep all generator types in the existing package-private CPU execution package to reuse task
  0001 internals without a cross-package facade.
- Use Java 26 `ClassFile`/`CodeBuilder` as current mechanism while keeping it non-authoritative.
- Select hidden nestmate classes with one static exact `invoke`; retain lookup/class/handle rather
  than a named loader or static registry.
- Use SHA-256 over explicit canonical typed encodings for cold compatibility identity, not
  security authentication or a persistent cache format.
- Use `CpuLoweringFingerprint.fromDigest(byte[])` only to retain the already-derived 32-byte
  specialization digest without rehashing.
- Derive exact direct method types; do not use `Object[]`, spreaders, collectors, or adapters.
- Make four modes structurally distinct: single-thread accepts count; parallel accepts assigned
  range/index. Scheduling remains outside generated code.
- Keep reduction support structural; later family tasks attach semantics.
- Include CPU Gradle because Java 26 Vector compile/test/Javadoc require local module activation.
- Retain class bytes for deterministic verification and later cache design, without adding cache.
- `CpuPortableExecutionMode` owns the structural scalar-versus-Vector emitter dispatch through
  the exact fixed-surface `emit(...)` method; `CpuClassFileKernelGenerator` delegates without a
  scalar/Vector switch.
- Scalar execution modes accept only `Tail.NONE`; all three tail values remain valid for Vector
  modes under the existing divisibility/tail contracts.
- Baked primitive-array byte offsets are aligned to `DataType.byteWidth()` at specialization
  construction. Exact segments retain baked zero offset, while dynamic primitive-array offsets
  remain cold invocation values.

## Known limitations

- CPU still advertises and executes no Model operation.
- Only bounded synthetic test probes execute; they have no Tensor, route, or conformance meaning.
- No production CPU analysis, finalizer, prepared executable, or generated bound invocation exists.
- Equal generation requests define distinct hidden classes until CPU 0003 adds bounded caching.
- Hidden-class unloading is collector-dependent; no collection deadline is asserted.
- Vector API incubates in Java 26; no later-JDK source/binary compatibility is promised.
- BFLOAT16/BOOL have carrier plumbing but no claimed Vector arithmetic semantics.
- Range/tile/tail/reduction helpers choose no algorithm, accuracy, accumulation, determinism, or
  performance threshold.
- Artifact persistence, corruption handling, broader target fingerprints, and JIT reuse remain
  deferred.
- No generated-artifact cache, production CPU analysis/finalizer, prepared executable, bound
  generated invocation, family semantic lowering, or advertised Model-operation coverage exists.
  These are deliberately deferred to CPU tasks 0003–0009.

## Validation evidence

This 2026-08-03 planning repair changed only the CPU 0002 task, CPU master plan, and roadmap. It
made the dispatch owner/signature, trusted-digest factory, tail matrix, and byte-offset rules
actionable without changing architecture, public API, package placement, or the 22-path maximum.

- `ruby /tmp/check_cpu_0002_repair.rb` — passed Ready/In-progress synchronization, the exact
  repaired surface and invariants, unchecked acceptance criteria, empty implementation completion
  summary, CPU 0001 `Complete`, CPU 0003–0016 `Draft` with no later detailed specifications,
  exactly the three authorized planning paths, and 20 total task paths under the 22-path maximum.
- `ruby /tmp/check_cpu_0002_markdown.rb
  docs/planning/backends/cpu/tasks/0002-portable-class-file-api-generator-foundation.md
  docs/planning/backends/cpu/master-plan.md docs/planning/roadmap.md` — passed three Markdown files,
  368 local links, balanced fences, final newlines, and trailing-whitespace checks.
- `git diff --check` — passed.
- The obsolete blocker-era `/tmp/check_cpu_0002_status.rb` was run once and rejected the repaired
  Ready state because it hard-codes the former status and completion record. It supplied no
  validation result and was replaced by the repair-specific checker above.
- No Java, Gradle, test, or Javadoc command ran because this repair changes planning only.

Final implementation and documentation evidence:

- Implementation context `019fc815-42aa-7de2-8970-a2fcab3a390e` recorded the final focused
  command passing 3 suites and 18 tests and the sole new final
  `./gradlew :backends:cpu:test` passing 9 suites and 34 tests, both with zero failures, errors,
  or skips. It also recorded Class-File parsing/verification, exact `MethodHandle` invocation,
  source/import/forbidden-mechanism scans, `javap` inspection, exact 20-path implementation
  scope, final newlines, and `git diff --check` as passing. No executable file changed after that
  final suite.
- Mandatory clean documentation context `019fc81f-f023-7ce2-b81e-9c0af980751f` independently
  reviewed the General, API/Javadoc, Backend Guide, Architecture, Planning, and Example profiles;
  the architecture, ADR, planning, task-0001, source, test, build, CPU guide, glossary, and
  generated-page contracts listed by this task; and the actual uncommitted diff. It changed only
  Javadocs and the five authorized explanatory/planning documents, so it did not rerun Java tests.
- `./gradlew :backends:cpu:javadoc` — final run passed all 11 tasks with CPU-local
  `jdk.incubator.vector` activation. The expected incubator warning and 28 pre-existing task-0001
  or implicit-default-constructor doclint warnings remained non-failing; every task-0002 page was
  generated. An earlier documentation run failed on invalid record-level `@throws` tags; moving
  those contracts to the explicit compact-constructor Javadocs resolved all task-0002 doclet
  errors.
- Generated-page assertions — passed for `CpuKernelSpecialization`, all four-mode and family/
  low-level emitter pages, `CpuClassFileKernelGenerator`, `CpuGeneratedKernel`, and the execution
  package limitation. Manual content checks found the complete specialization contract, fresh
  identity-artifact behavior, and synthetic-probe limitation in rendered HTML.
- `ruby /tmp/check_cpu_0002_markdown.rb
  docs/planning/backends/cpu/tasks/0002-portable-class-file-api-generator-foundation.md
  docs/planning/backends/cpu/master-plan.md docs/planning/roadmap.md
  docs/backend-guide/cpu-backend.md docs/glossary.md` — passed five Markdown files, 671 local
  links, 287 anchors, balanced fences, final newlines, and trailing whitespace. The first run
  exposed an older-Ruby checker incompatibility and two stale OpenBLAS glossary anchors in the CPU
  guide; the checker and the two authorized links were corrected before the passing run.
- `ruby /tmp/check_cpu_0002_complete.rb` — passed exact 22-path scope; CPU 0001/0002 `Complete`,
  CPU project `In progress`, CPU 0003–0016 `Draft`, and no later detailed task; package-private
  source surface; import and forbidden-mechanism exclusions; exact four modes; CPU-local Vector
  compile/test/Javadoc flags with no preview flag; and required generated Javadoc pages.
- Changed-file newline/whitespace validation — passed all 22 paths, including untracked Java and
  test files. Excluded-path comparison proved no changes to `ARCHITECTURE.md`, focused architecture
  pages, ADRs, architecture/conformance/integration tests, the public `CpuCapabilityProvider`,
  shared modules/extensions, root settings/build, or shared build logic.
- `git diff --check` — passed on the final combined change.

Reasoned no-change conclusions:

- `CpuCapabilityProvider` remains byte-for-byte unchanged and fail-closed; synthetic probes do not
  justify advertising any operation.
- Prepare, Runtime, Compiler, and Planning APIs remain accurate and unchanged because this task
  adds neither a preparer/finalizer nor a prepared/runtime route, compiler behavior, planning
  ownership input, or shared lifecycle handoff.
- `ARCHITECTURE.md`, focused architecture pages, ADRs, architecture tests, and the CPU strategy
  note remain accurate: the existing contract already permits implementation-neutral generated
  JVM bytecode inside CPU finalization and deliberately does not prescribe Class-File or Vector
  APIs. No ownership, lifecycle, or dependency rule changed.
- Backend-conformance and integration tests remain unchanged because no Model operation is
  advertised, lowered, prepared, or executed. The new tests correctly remain CPU-module
  synthetic foundation tests.
- The CPU build change is limited to the required Java 26 Vector module activation for CPU
  compile, test, and Javadoc. No dependency coordinate, root/shared Gradle rule, preview option,
  module descriptor, ASM dependency, or other module build changed.
- Model, Config, Backend Contract, Trace, OpenBLAS provider, Prepare, Runtime, Compiler, Planning,
  Engine, extensions, and other backends require no documentation or source change because the
  task exposes no public/shared contract and makes no route, execution, storage-ownership, or
  capability claim across those boundaries.

## Implementation notes

Implementation added the exact package-private schema, lowering and specialization fingerprints,
four-mode vocabulary, family-emission seam, separate scalar/Vector/carrier/loop/reduction helpers,
Class-File API generator, and retained hidden-class artifact specified above. The specialization
derives one exact direct `void` entry type from ordered heap-array and `MemorySegment` carriers,
dynamic offsets/extents, and single-thread or parallel range controls. Generation validates the
family fingerprint before emission, verifies deterministic Java 26 bytes, defines a fresh hidden
nestmate, and resolves the exact direct static handle without adaptation.

The tests supply the only family emitter and exercise bounded synthetic copy and structural
probes. They cover exact six-array carrier mapping, segment and mixed signatures, all four modes,
external `CpuWorkerGroup` range dispatch, access restrictions, tail/reduction seams, canonical
fingerprints, class shape, direct handle types, deterministic equal bytes, and distinct hidden
artifacts for equal requests. No production operation meaning or route was added.

The clean documentation pass independently reviewed source, tests, build flags, generated
Javadoc, CPU guidance, glossary terminology, and planning synchronization. It clarified the
current generator boundary, typed specialization and fingerprints, four modes, direct carrier
forms, family/emitter ownership, hidden-artifact lifetime, absence of caching, and the continued
fail-closed CPU capability boundary without changing executable Java.

## Completion summary

- Completed changes: added the package-private deterministic Java 26 generated-kernel foundation,
  exact typed specialization/fingerprint contracts, four structural execution modes, low-level
  emission seams, verified hidden-class artifact construction, and CPU-local Vector module flags.
- Files changed or created: 14 CPU production/build paths, three CPU test paths, the CPU guide,
  glossary, this task, CPU master plan, and roadmap; exactly 22 paths.
- Tests and validation: implementation context `019fc815-42aa-7de2-8970-a2fcab3a390e`
  recorded the focused 3-suite/18-test pass and the sole final CPU 9-suite/34-test pass, both with
  zero failures, errors, or skips. The documentation pass reused that evidence because it changed
  no executable Java behavior and completed CPU Javadoc and documentation/surface/scope/status/
  whitespace validation.
- Documentation-agent review: the mandatory clean documentation-focused pass independently
  finalized all affected production Javadocs, CPU guidance, glossary impact, planning evidence,
  and synchronized status.
- Documentation impact: `docs/backend-guide/cpu-backend.md` now explains the current generator
  foundation and its limits; architecture pages, ADRs, and the CPU strategy note remain accurate.
- Javadoc review: every new or changed production declaration was reviewed for purpose, behavior,
  invariants, validation, ownership, side effects, parameters, returns, and expected failures.
- Glossary impact: added current CPU specialization and generated-artifact definitions and
  clarified the generated-kernel entry without turning implementation mechanisms into invariants.
- Unresolved issues: None within task 0002 scope.
- Follow-up required: None for task 0002; CPU 0003–0016 remain Draft planned work.

Status: Complete
