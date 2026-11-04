# Task 0002: Portable [Class-File API](../../../../glossary.md#term-class-file-api) Generator Foundation

## Reader context

This generator checkpoint separated writing executable code from executing a Tensor operation. A
family emitter supplied generation-time instructions to the Java Class-File API; the generator
verified the resulting bytes, defined a hidden JVM class, and resolved its exact static
`MethodHandle`. The signature and build snippets describe that historical internal seam, not a
public Tensor execution API.

A specialization described the types and facts that changed generated code. Actual arrays,
segments, and per-run values were invocation arguments, not cached Tensor results. The synthetic
probes demonstrated this mechanism without establishing Model-operation support. Parallel probes
borrowed the worker infrastructure outside the generated class; the class itself did not own
threads or their cleanup. This record predates the later partition architecture reset.

Current reading alongside this historical record:

- [CPU generated-code ownership](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalcodegenemit)
  — separates generation-time emitters, emitted typed work, and JVM compilation.
- [CPU preparation/run handoffs](../../../../explanation/packages/cpu.md#read-the-pipeline-before-the-packages)
  — names the participants and separates descriptions, assigned positions, and live run memory.

## Status

Superseded

## Goal

Establish one [backend](../../../../glossary.md#backend)-private, deterministic Java 26 [Class-File API](../../../../glossary.md#term-class-file-api) foundation that can generate,
verify, define, retain, and directly invoke a tiny synthetic [CPU](../../../../glossary.md#term-cpu) probe [kernel](../../../../glossary.md#kernel) across the four
portable execution modes. The task creates reusable code-generation machinery and typed extension
seams only. It does not lower or execute a [Model](../../../../glossary.md#term-model) [operation](../../../../glossary.md#operation), select a production CPU route, or make
CPU capability reporting truthful for any operation semantic.

The [mental model](../../../../glossary.md#term-mental-model) is:

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

## [Scope](../../../../glossary.md#term-scope)

- Add a package-private generator schema/version contract and a deterministic typed specialization
  descriptor plus content fingerprint.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [specialization glossary](../../../../glossary.md#term-specialization); [descriptor glossary](../../../../glossary.md#term-descriptor); [fingerprint glossary](../../../../glossary.md#term-fingerprint))</span>
- Describe exactly the facts that may change emitted bytecode: family-lowering fingerprint,
  ordered typed carrier/access signature, baked extents and element strides, exact portable mode,
  Vector API species and byte order when applicable, unroll/tile/tail structure, exact/default
  numerical mode, combine-order choice, and Java class-file target.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [lowering glossary](../../../../glossary.md#lowering); [carrier glossary](../../../../glossary.md#term-storage-carrier); [element glossary](../../../../glossary.md#term-element); [strides glossary](../../../../glossary.md#term-stride); [Vector API glossary](../../../../glossary.md#term-vector-api); [byte order glossary](../../../../glossary.md#text-encoding-and-file-boundaries); [target glossary](../../../../glossary.md#term-target))</span>
- Exclude facts that do not change generated code: graph/runtime identities, physical addresses,
  slot numbers, run objects, worker-group identity, thread count, chunk size, cache state, and
  mutable target observations.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [runtime glossary](../../../../glossary.md#term-runtime); [run glossary](../../../../glossary.md#term-run); [identity glossary](../../../../glossary.md#term-identity-meanings); [cache glossary](../../../../glossary.md#term-cache); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Add one family-owned emission collaboration with separate scalar and Vector API entry methods.
  Test source supplies the only implementation. Later family tasks add their own typed lowering
  implementations beside their semantics; this task adds no family registry or implementation.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [registry glossary](../../../../glossary.md#term-registry))</span>
- Add separate low-level scalar, Vector API, carrier, range/tile/tail, and reduction-structure
  emitters. They expose bytecode construction primitives, not Model operation meaning.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation))</span>
- Support exact direct entry [signatures](../../../../glossary.md#term-signature) containing primitive [heap](../../../../glossary.md#term-heap) carriers, exact
  [`MemorySegment`](../../../../glossary.md#term-memorysegment) arguments, or both. Heap byte offsets and every non-baked extent/range remain
  primitive entry arguments; generation and invocation perform no copy.<span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument); [extent glossary](../../../../glossary.md#term-extent))</span>
- Preserve task-0001 access meaning: read-only arguments may be read but never emitted as stores;
  write-only arguments may be written but never emitted as loads; read-write arguments permit
  both. Exact-segment writability remains a cold-binding obligation and is also locked into the
  specialization signature.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Implement exactly four `CpuPortableExecutionMode` values: scalar single-thread, scalar
  parallel, Vector API single-thread, and Vector API parallel.
- Make `CpuPortableExecutionMode` the sole owner of structural scalar-versus-Vector emission
  dispatch. It constructs only the selected mode's scalar or Vector emitter plus the shared
  carrier, loop, and reduction emitters, then invokes exactly one matching family callback.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [Vector glossary](../../../../glossary.md#term-vector); [dispatch glossary](../../../../glossary.md#term-dispatch); [callback glossary](../../../../glossary.md#term-callback))</span>
- Reuse `CpuRangeBody`, [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup), and `CpuBufferArgument` in tests. Generated parallel
  entries accept an already-selected half-open range; <span data-glossary-navigation="true"> ([half-open glossary](../../../../glossary.md#term-half-open-interval))</span>`CpuWorkerGroup` owns dispatch,
  cancellation, synchronization, and failure propagation.<span data-glossary-navigation="true"> ([cancellation glossary](../../../../glossary.md#term-cancellation); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Generate one static typed entry method with Java 26 `ClassFile`/`CodeBuilder`, verify the bytes,
  define it as a nestmate hidden class from a full-privilege CPU-package lookup, resolve its exact
  static <span data-glossary-navigation="true"> ([nestmate glossary](../../../../glossary.md#term-nestmate); [hidden class glossary](../../../../glossary.md#term-hidden-class))</span>[`MethodHandle`](../../../../glossary.md#native-numerical-library-and-instruction-architecture), and retain the hidden [lookup](../../../../glossary.md#term-method-handle-lookup)/class/handle/bytes in one [immutable](../../../../glossary.md#term-immutable) [artifact](../../../../glossary.md#term-artifact).
- Prove the generated artifact can be invoked directly for bounded synthetic probes over heap,
  exact-segment, and mixed signatures in all four modes.
- Add the [CPU](../../../../glossary.md#term-cpu)-[module](../../../../glossary.md#term-module)-only `jdk.incubator.vector` compile, test-runtime, and Javadoc module flags
  required by Java 26. Keep the root toolchain/release at 26 and add no preview flag or dependency.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [Javadoc glossary](../../../../glossary.md#term-javadoc); [toolchain glossary](../../../../glossary.md#term-toolchain); [release glossary](../../../../glossary.md#term-release); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Keep `CpuCapabilityProvider.supports(...)` unconditionally `false` and lock the unchanged public
  surface in tests.
- Finalize affected Javadocs, CPU guidance, glossary impact, and planning evidence through the
  mandatory separate clean documentation-focused pass after implementation.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- any Model operation kind, operation-family implementation, semantic lowering, fused-partition
  implementation, numerical algorithm, Tensor value evaluation, or advertised CPU operation
  capability;<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation); [lowering glossary](../../../../glossary.md#lowering); [fused glossary](../../../../glossary.md#specialization-and-fusion); [partition glossary](../../../../glossary.md#partition); [Tensor glossary](../../../../glossary.md#tensor); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- public [CPU](../../../../glossary.md#term-cpu) APIs or a public code-generation, [kernel](../../../../glossary.md#kernel), route, storage, [vector](../../../../glossary.md#term-vector), or execution [API](../../../../glossary.md#term-application-programming-interface-api);
- CPU capability-provider changes, route selection, candidate generation, target discovery,
  tuning, numerical-policy selection, or determinism-policy selection;<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- a generated-artifact cache, cache key/store/lookup/eviction/single-flight mechanism, persistent
  bytes, class serialization, or task 0003 behavior;<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [cache glossary](../../../../glossary.md#term-cache); [cache key glossary](../../../../glossary.md#term-cache-key); [lookup glossary](../../../../glossary.md#term-lookup); [single-flight glossary](../../../../glossary.md#single-flight-and-least-recently-used-caching); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- a production [`BackendPartitionPreparer`](../../../../glossary.md#backend-partition-preparer--backendpartitionpreparer), `BackendPartitionFinalizer`, CPU analysis plan,
  finalization orchestrator, schedule assembler, representation plan, physical allocation change,
  materialization decision, or task 0004 behavior;<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- worker-pool, scheduler, thread-count, chunk-size, [cancellation](../../../../glossary.md#term-cancellation), or failure-policy changes;
- OpenBLAS, oneMKL, oneDNN, Accelerate, AOCL, ZenDNN, Metal, CUDA, vendor, native-provider, or
  Foreign Function and Memory invocation routes;<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [oneMKL glossary](../../../../glossary.md#term-onemkl); [oneDNN glossary](../../../../glossary.md#term-onednn); [AOCL glossary](../../../../glossary.md#term-aocl); [ZenDNN glossary](../../../../glossary.md#term-zendnn); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [Foreign Function and Memory glossary](../../../../glossary.md#application-binary-interface-and-native-bridge))</span>
- compiler, Planning, shared Prepare, Runtime, Engine, Config, Model, Backend Contract, Trace,
  OpenBLAS-provider, backend-conformance, integration-test, or architecture-test changes;<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [Backend glossary](../../../../glossary.md#backend); [Trace glossary](../../../../glossary.md#trace); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- architecture contract, focused architecture explanation, ADR, kernel-routes guide, CPU strategy
  note, or general coding-rule changes;<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- root [Gradle](../../../../glossary.md#term-gradle), settings, shared build logic, [dependency](../../../../glossary.md#term-dependency) [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), ASM, another [bytecode](../../../../glossary.md#term-bytecode) library,
  `--enable-preview`, or another [incubator module](../../../../glossary.md#term-incubator-module);
- broad execution algorithms, scalar/Vector performance claims, arbitrary vector-lane promises,
  benchmarks, or a real Tensor-operation result; and<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [lane glossary](../../../../glossary.md#term-lane))</span>
- detailed task specifications for CPU 0003 or any later CPU task.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core invariants
  - `modules/runtime`
  - `modules/prepare`
  - [Concrete backend](../../../../glossary.md#concrete-backend) [modules](../../../../glossary.md#term-module)
  - [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) routes
  - [Prepare](../../../../glossary.md#prepare) [lifecycle](../../../../glossary.md#lifecycle)
  - [Run](../../../../glossary.md#term-run) lifecycle
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [ADR 0002: Backend-owned lowering](../../../../design/decisions/0002-backend-owned-lowering.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Backend-owned lowering glossary](../../../../glossary.md#backend-owned-lowering))</span>
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [CPU kernel strategy](../../../../design/notes/cpu-kernel-strategy.md)<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel))</span>
- [Task 0001 CPU capability, representation, binding, and parallel foundation](0001-cpu-capability-representation-binding-and-parallel-foundation.md)<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>

## Architecture constraints

- Generated JVM bytecode is a CPU-internal computation route. It does not create a separate
  backend or shared lowering module.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [bytecode glossary](../../../../glossary.md#term-bytecode); [CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [module glossary](../../../../glossary.md#term-module))</span>
- CPU analysis later owns semantic lowering, route choice, specialization, fusion, and exact
  shared-resource declaration. Task 0002 accepts an already-complete specialization and a
  family-owned emitter; it performs none of those choices.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- CPU finalization may generate and define a selected class only after shared slot assignment.
  The generator API must therefore be usable from a later CPU finalizer, but task 0002 must not
  invent that finalizer or call shared Prepare contracts.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Prepare glossary](../../../../glossary.md#prepare))</span>
- [Runtime](../../../../glossary.md#term-runtime) receives only a finalized [immutable](../../../../glossary.md#term-immutable) executable recipe. Generated hot code must not see
  [`Operation`](../../../../glossary.md#operation), `CompiledNode`, [`Tensor`](../../../../glossary.md#tensor), [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext), [`BackendPartitionAnalysis`](../../../../glossary.md#backend-partition-analysis-backendpartitionanalysis), Runtime
  slots, [`RunState`](../../../../glossary.md#run-state--runstate), or [nominal](../../../../glossary.md#term-nominal) representation objects.
- Generated code must not discover storage, inspect `heapBase()`, choose routes, look up cache
  entries, select scalar versus Vector API, decide parallelism, own workers, or switch on Model
  semantics. Cold CPU binding and later finalization resolve those facts once.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [scalar glossary](../../../../glossary.md#term-scalar); [Vector API glossary](../../../../glossary.md#term-vector-api); [Model glossary](../../../../glossary.md#term-model); [binding glossary](../../../../glossary.md#term-binding))</span>
- Hidden-class definition and direct entry-point construction are generation/finalization
  mechanics. Keeping the artifact reachable keeps its hidden lookup, class, and method handle
  reachable. Once neither a future cache nor a prepared executable retains the artifact, the
  implementation may become unloadable; this task must not promise or force a collection time.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [lookup glossary](../../../../glossary.md#term-lookup); [method handle glossary](../../../../glossary.md#term-method-handle); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable))</span>
- Task 0001's `CpuBufferArgument` variants remain the cold storage classification. Generation
  consumes only the corresponding direct carrier/segment signature and never copies or owns the
  carrier or segment.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [signature glossary](../../../../glossary.md#term-signature))</span>
- Task 0001's [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup) remains the sole CPU worker owner. A parallel generated entry is a
  range body, not a scheduler.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- `jdk.incubator.vector` is a task-local CPU implementation dependency supplied by JDK 26, not a
  new Gradle library or architecture edge. No Java Platform Module System descriptor is added.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [Gradle glossary](../../../../glossary.md#term-gradle); [Java Platform Module System glossary](../../../../glossary.md#term-java-platform-module-system-jpms); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Capability remains [fail-closed](../../../../glossary.md#term-fail-closed) because a synthetic probe is not operation coverage.
- If implementation needs a new public type, shared contract, dependency edge, production CPU
  preparer/finalizer, operation semantic, cache, or architecture change, stop and report the
  conflict.<span data-glossary-navigation="true"> ([dependency edge glossary](../../../../glossary.md#term-dependency-edge))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — unchanged public capability provider used only by
  public-surface regression tests.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression))</span>
- `io.github.pho001.synaptik.backend.cpu.execution` — owns all new package-private generation,
  specialization, artifact, and emission types beside task 0001's package-private buffer and
  worker foundations.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [artifact glossary](../../../../glossary.md#term-artifact); [buffer glossary](../../../../glossary.md#term-buffer))</span>

Packages added or changed:

- No Java package is added. The existing CPU execution package is extended because its new types
  must consume package-private <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>`CpuBufferArgument`, `CpuRangeBody`, and [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup) without
  widening them or manufacturing a public cross-package API.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

Type placement:

- `io.github.pho001.synaptik.backend.cpu.execution.CpuGeneratorSchema` — schema constants and
  canonical fingerprint encoding owner.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [constants glossary](../../../../glossary.md#term-constant); [canonical glossary](../../../../glossary.md#term-canonical-form); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [owner glossary](../../../../glossary.md#term-owner))</span>
- `io.github.pho001.synaptik.backend.cpu.execution.CpuPortableExecutionMode` — the exact closed
  four-mode vocabulary.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuLoweringFingerprint` — immutable typed
  family-lowering digest supplied by the family owner.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [lowering glossary](../../../../glossary.md#lowering); [digest glossary](../../../../glossary.md#term-digest))</span>
- `io.github.pho001.synaptik.backend.cpu.execution.CpuKernelSpecialization` — immutable complete
  bytecode-relevant descriptor and derived specialization fingerprint.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- `io.github.pho001.synaptik.backend.cpu.execution.CpuFamilyKernelEmitter` — family-specific
  scalar/Vector emission seam; test source supplies the only task-0002 implementation.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuScalarEmitter` — [scalar](../../../../glossary.md#term-scalar) instruction helper.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuVectorEmitter` — Vector API instruction and
  species helper.<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api))</span>
- `io.github.pho001.synaptik.backend.cpu.execution.CpuCarrierEmitter` — exact [heap](../../../../glossary.md#term-heap)-[carrier](../../../../glossary.md#term-storage-carrier) and
  [`MemorySegment`](../../../../glossary.md#term-memorysegment) load/store helper.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuLoopEmitter` — range, tile, unroll, and tail
  control-flow helper.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuReductionEmitter` — structural partial and
  combine-order helper with no mathematical reduction policy.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuClassFileKernelGenerator` — narrow
  build/verify/define/resolve orchestrator.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuGeneratedKernel` — immutable hidden-class
  lifetime and exact entry-point artifact.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime))</span>

Every new production declaration is package-private. Do not add `generated`, `codegen`, [`kernel`](../../../../glossary.md#kernel),
[`vector`](../../../../glossary.md#term-vector), `util`, [`registry`](../../../../glossary.md#term-registry), `service`, or `manager` packages.

## Exact package-private surface

The source-level surface below is fixed for this task. Record [canonical](../../../../glossary.md#term-canonical-form) constructors and
accessors may be declared explicitly for [validation](../../../../glossary.md#term-validation) and [Javadoc](../../../../glossary.md#term-javadoc). A [signature](../../../../glossary.md#term-signature) change requires
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
low-level emitters, and the family-owned [callback](../../../../glossary.md#term-callback). `CpuPortableExecutionMode.emit(...)` owns only
structural [scalar](../../../../glossary.md#term-scalar)-versus-[Vector](../../../../glossary.md#term-vector) [dispatch](../../../../glossary.md#term-dispatch): it constructs the mode-appropriate
`CpuScalarEmitter` or `CpuVectorEmitter`, constructs the shared `CpuCarrierEmitter`,
`CpuLoopEmitter`, and `CpuReductionEmitter`, and calls exactly one of `emitScalar` or
`emitVector`. It performs no [Model](../../../../glossary.md#term-model)-semantic, family, route, [cache](../../../../glossary.md#term-cache), storage, worker, [target](../../../../glossary.md#term-target), or
[run](../../../../glossary.md#term-run)-time dispatch. `CpuClassFileKernelGenerator` delegates exactly once to that method and may
otherwise orchestrate only class name/version, one static `invoke` method, verification, hidden
definition, and exact handle resolution. It must not contain a scalar/Vector branch or any
[carrier](../../../../glossary.md#term-storage-carrier), loop, reduction, family, [operation](../../../../glossary.md#operation), or route switch.

## [Schema](../../../../glossary.md#term-schema), [specialization](../../../../glossary.md#term-specialization), and [fingerprint](../../../../glossary.md#term-fingerprint) contract

- Schema version `1` identifies the canonical binary encoding and generator behavior established
  by this task. It is not the task number, project version, JDK version, or cache schema.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [cache glossary](../../../../glossary.md#term-cache))</span>
- `CpuLoweringFingerprint.of(...)` requires a non-null, non-empty canonical family byte sequence,
  copies it, hashes it with SHA-256 during cold construction, and retains exactly 32 digest bytes.
  Tests lock a fixed vector. A later family owns the canonical typed encoding of its semantic
  lowering; task 0002's test probe uses one fixed test-only encoding.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([SHA-256 glossary](../../../../glossary.md#term-sha-256); [digest glossary](../../../../glossary.md#term-digest); [lowering glossary](../../../../glossary.md#lowering))</span>
- `CpuLoweringFingerprint.fromDigest(...)` is the internal trusted-digest factory used only for
  the derived specialization fingerprint. It validates null first with
  `NullPointerException("digest")`, then requires exactly 32 bytes with
  `IllegalArgumentException("digest must contain exactly 32 bytes")`, makes exactly one defensive
  copy, and retains those bytes without hashing them again.<span data-glossary-navigation="true"> ([defensive copy glossary](../../../../glossary.md#term-defensive-copy))</span>
- `CpuKernelSpecialization.specializationFingerprint()` is SHA-256 over one versioned canonical
  big-endian encoding. Each enum has an explicit stable integer tag; enum ordinal and<span data-glossary-navigation="true"> ([big-endian glossary](../../../../glossary.md#text-encoding-and-file-boundaries))</span>
  `toString()` are forbidden. Every list is length-prefixed and order-sensitive. Every long and
  int uses fixed-width signed big-endian representation. Boolean values use one byte <span data-glossary-navigation="true"> ([Boolean glossary](../../../../glossary.md#data-type--datatype))</span>`0` or `1`.
- The specialization encoding order is exactly: schema version, lowering digest, execution mode,
  ordered arguments with data type/carrier/access/offset-baked flag/baked offset/ordered strides,
  ordered baked extents, dynamic-extent count, optional vector shape, exact byte order, unroll
  factor, tile count, tail mode, numerical mode, combine order, and class-file major version.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [extent glossary](../../../../glossary.md#term-extent); [shape glossary](../../../../glossary.md#shape); [byte order glossary](../../../../glossary.md#text-encoding-and-file-boundaries))</span>
- The deterministic generated class name is derived from the complete specialization fingerprint
  inside `io.github.pho001.synaptik.backend.cpu.execution`. It is diagnostic identity only; hidden
  classes still have distinct JVM identities and are not found by name.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- `Argument.bakedByteOffset` is non-negative. For primitive-array carriers it is the constant
  carrier byte offset when <span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>`byteOffsetBaked` is true and must be aligned to
  `dataType.byteWidth()`; misalignment fails with
  `IllegalArgumentException("bakedByteOffset must be aligned to dataType byte width")`. When
  `byteOffsetBaked` is false, `bakedByteOffset` must be zero and one exact primitive `long`
  byte-offset argument appears in `entryType()`; dynamic array offsets are primitive runtime
  arguments, and their runtime alignment and bounds are outside this foundation task.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [alignment glossary](../../../../glossary.md#term-alignment))</span>
  `MEMORY_SEGMENT` requires `byteOffsetBaked == true` and the exact baked offset zero because it
  always uses one exact selected segment.
- The exact [`Argument`](../../../../glossary.md#term-argument) [validation](../../../../glossary.md#term-validation) order is: non-null `dataType`, non-null [`carrier`](../../../../glossary.md#term-storage-carrier), non-null
  `access`, non-baked-offset consistency (`bakedByteOffset must be zero when byteOffsetBaked is
  false`), non-negative baked offset (`bakedByteOffset must be non-negative`), non-null strides
  list and ordered non-null/non-negative stride entries, the exact-segment baked-zero rule
  (`MEMORY_SEGMENT requires a baked zero byte offset`) when the carrier is `MEMORY_SEGMENT`,
  otherwise exact data-type/carrier match (`carrier does not match dataType`), and finally baked
  primitive-array width alignment (`bakedByteOffset must be aligned to dataType byte width`). The
  alignment check therefore never precedes an existing null, consistency, range, stride,
  exact-segment, or carrier-match failure.
- An argument carrier must match its [`DataType`](../../../../glossary.md#data-type--datatype) exactly as task 0001 does: FLOAT64/double[],
  FLOAT32/float[], BFLOAT16/short[], INT32/int[], INT64/long[], BOOL/byte[], while<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
  `MEMORY_SEGMENT` accepts every current type. No generic `Object`, carrier name, or class-name
  string is encoded.
- Baked extents and [strides](../../../../glossary.md#term-stride) are non-negative. Non-baked extents are represented only by
  `dynamicExtentCount` and exact ordered primitive `long` entry parameters. Runtime IDs and
  symbolic dimension identities are never encoded.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [dimension glossary](../../../../glossary.md#dimension))</span>
- [Vector](../../../../glossary.md#term-vector) modes require one `VectorShape`; scalar modes forbid it. The shape must resolve exactly
  to a Java 26 supported species for FLOAT64, FLOAT32, INT32, or INT64, with positive power-of-two
  bit size and matching lane count. BFLOAT16 and BOOL vector semantics are not claimed by this
  foundation. Unsupported species fail before class bytes are built.<span data-glossary-navigation="true"> ([lane glossary](../../../../glossary.md#term-lane))</span>
- `byteOrder` is exactly `ByteOrder.LITTLE_ENDIAN` or `ByteOrder.BIG_ENDIAN` and uses a stable
  explicit fingerprint tag. Later CPU analysis normally supplies <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>`ByteOrder.nativeOrder()`.
  Segment scalar/Vector access emits the selected order directly; generated hot code performs no
  byte-order switch or target discovery.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- `unrollFactor` and `tileElementCount` are positive. [Scalar](../../../../glossary.md#term-scalar) modes require `Tail.NONE`; after the
  non-null tail check, any other scalar tail fails with<span data-glossary-navigation="true"> ([scalar tail glossary](../../../../glossary.md#term-scalar-tail))</span>
  `IllegalArgumentException("scalar execution mode requires Tail.NONE")`. Vector modes permit
  `NONE`, `SCALAR`, and `MASKED`: `NONE` requires later analysis to prove exact divisibility,
  while `SCALAR` and `MASKED` represent the selected remainder structures under the same existing
  divisibility and tail contracts. Task-0002 scalar probes use only `NONE`.
- In the exact `CpuKernelSpecialization` constructor order, scalar-tail compatibility is checked
  after schema, lowering fingerprint, execution mode, argument snapshot, baked extents, dynamic
  extent count, vector-shape presence/absence, byte order, positive unroll factor, positive tile
  element count, and non-null tail, but before numerical mode, combine order, and class-file
  version. Thus a scalar <span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [element glossary](../../../../glossary.md#term-element))</span>`SCALAR` or `MASKED` tail fails at that position with exactly
  `scalar execution mode requires Tail.NONE`; no earlier or later [invariant](../../../../glossary.md#term-invariant) displaces it.
- Schema version 1 permits only `NumericalMode.EXACT_DEFAULT`; this records the current exact/default
  arithmetic contract without inventing relaxed or fast math. `CombineOrder` records whether
  emitted partial-combine control flow is fixed or unrestricted. Neither value is a public Config
  policy, and fixed control flow alone does not claim deterministic floating results.
- `classFileMajorVersion` must equal `ClassFile.JAVA_26_VERSION` in schema version 1. A later JDK or
  emission change requires a schema decision rather than silent reuse.
- The constructor validates and snapshots every list before computing `entryType()` or the
  fingerprint. Returned lists are unmodifiable; fingerprint and class-byte accessors return
  defensive copies. Equality and hashing never depend on array identity, lookup/class/handle
  identity, object addresses, or mutable external state.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Thread count, worker count, minimum range size, chunk size, [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot), [`WorkspaceSlot`](../../../../glossary.md#workspace-slot--workspaceslot),
  [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), [`NodeId`](../../../../glossary.md#nodeid), [`TensorId`](../../../../glossary.md#tensorid), [model](../../../../glossary.md#term-model)/[run](../../../../glossary.md#term-run) [identity](../../../../glossary.md#term-identity-meanings), [`RunState`](../../../../glossary.md#run-state--runstate), carrier object [identity](../../../../glossary.md#term-reference-identity),
  `MemorySegment.address()`, lookup identity, hidden-class identity, cache state, and timing data
  are forbidden fingerprint inputs.

## Direct entry [signature](../../../../glossary.md#term-signature) and access contract

- The generated class has no instance fields, static mutable fields, constructor use, class
  initializer, native method, bridge, reflection, service lookup, or generic entry point. It has
  exactly one package-private static <span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable))</span>`void invoke(...)` method plus [JVM](../../../../glossary.md#term-java-virtual-machine-jvm)-required class [metadata](../../../../glossary.md#term-metadata).
- Ordered buffer arguments come first. An array carrier contributes its exact primitive array
  class and, when the byte offset is not baked, one adjacent primitive <span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>`long` byte-offset
  parameter. An exact segment contributes [`MemorySegment`](../../../../glossary.md#term-memorysegment) only.
- Non-baked extents follow in [descriptor](../../../../glossary.md#term-descriptor) order as primitive `long` [parameters](../../../../glossary.md#term-parameter).
- Single-thread modes then accept one primitive `long elementCount`. Parallel modes instead
  accept primitive `long startInclusive`, `long endExclusive`, and `int rangeIndex`. The
  generated parallel method executes only that range and never submits work.
- The descriptor derives one exact `MethodType`; the class builder uses the matching
  `MethodTypeDesc`; hidden [lookup](../../../../glossary.md#term-lookup) resolves `findStatic(hiddenClass, "invoke", entryType)` without
  `asType`, reflection, proxy, lambda metafactory, spreader, collector, `invokeWithArguments`, or
  raw `Object` adaptation.
- `CpuCarrierEmitter` validates argument index, access direction, data type, carrier kind,
  element-width alignment, and supported vector species during generation. Invalid access fails
  closed before class definition. Generated code contains only the already-specialized primitive
  array or exact-segment access sequence.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [element glossary](../../../../glossary.md#term-element); [alignment glossary](../../../../glossary.md#term-alignment); [vector species glossary](../../../../glossary.md#term-vector-species))</span>
- Heap carriers and exact segments are borrowed for invocation. The generated method neither
  copies nor closes them. Read-only exact-segment rejection remains task 0001's cold binding
  responsibility; an emitted write is additionally forbidden when descriptor access is<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [binding glossary](../../../../glossary.md#term-binding))</span>
  `READ_ONLY`.
- Mixed signatures preserve exact [argument](../../../../glossary.md#term-argument) order and require no global [heap](../../../../glossary.md#term-heap)/native mode.

## [Scalar](../../../../glossary.md#term-scalar), [Vector API](../../../../glossary.md#term-vector-api), loop, tile, reduction, and tail boundaries

- `CpuScalarEmitter` owns primitive local/type-kind mechanics only. It has no operation-kind or
  family switch and does not define arithmetic semantics.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- `CpuVectorEmitter` resolves the exact selected `VectorSpecies` [constant](../../../../glossary.md#term-constant) and emits direct
  `FloatVector`, `DoubleVector`, `IntVector`, or `LongVector` calls. It does not choose a species,
  vectorize a scalar lowering automatically, or promise support for BFLOAT16/BOOL semantics.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- `CpuCarrierEmitter` is executable for exact scalar/vector load and store plumbing across all six
  heap carrier forms and exact <span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment); tests may exercise only the bounded types needed
  to prove each structural path. Later operation-family tasks own semantic type coverage.
- `CpuLoopEmitter.emitRange` emits one counted half-open loop. In single-thread mode its bounds
  are <span data-glossary-navigation="true"> ([half-open glossary](../../../../glossary.md#term-half-open-interval))</span>`[0, elementCount)`; in parallel mode they are the supplied `[startInclusive,
  endExclusive)`. It emits no [dispatch](../../../../glossary.md#term-dispatch) or [cancellation](../../../../glossary.md#term-cancellation) check inside an [element](../../../../glossary.md#term-element) loop.
- `emitTiles` derives deterministic non-empty tiles from the already-baked tile size inside the
  supplied range. It does not choose tile size, allocate workspace, or schedule a tile.
- `emitTail` is a [Vector](../../../../glossary.md#term-vector)-mode structural helper and emits only the already-selected `NONE`,
  scalar, or masked-vector tail. Scalar modes never enter it with `SCALAR` or `MASKED`. It does
  not inspect workload facts or choose a policy at run time.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- `CpuReductionEmitter` provides structural local-slot partial and fixed/unfixed combine-order
  control flow only. The family callback supplies mathematical partial/combine instructions. No
  production task-0002 callback supplies a sum, mean, extrema, scan, softmax, or other Model
  semantic. The synthetic test may use tiny arithmetic solely to validate the seam and ordering.<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback); [mean glossary](../../../../glossary.md#term-mean); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [scan glossary](../../../../glossary.md#term-cumulative-scan); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [Model glossary](../../../../glossary.md#term-model))</span>
- None of these helpers owns [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup), an executor, a pool, a scheduler, a cache, storage
  discovery, route choice, or a semantic registry.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [registry glossary](../../../../glossary.md#term-registry))</span>

## Generation, definition, and failure contract

[Validation](../../../../glossary.md#term-validation) order for `CpuClassFileKernelGenerator.generate(...)` is:

1. non-null [specialization](../../../../glossary.md#term-specialization) (`specialization`);
2. non-null family emitter (`familyEmitter`);
3. exact [lowering](../../../../glossary.md#lowering) [fingerprint](../../../../glossary.md#term-fingerprint) equality, else
   `familyEmitter lowering fingerprint does not match specialization`;
4. specialization/[schema](../../../../glossary.md#term-schema)/species/[signature](../../../../glossary.md#term-signature) invariants, already established by construction;
5. class-byte emission;
6. `ClassFile.of().verify(bytes)` with no verification errors, else
   `generated class verification failed` plus stable indexed verification details;
7. `MethodHandles.lookup().defineHiddenClass(bytes, false, NESTMATE)`;
8. exact static `invoke` resolution; and
9. [immutable](../../../../glossary.md#term-immutable) [artifact](../../../../glossary.md#term-artifact) construction.

Unchecked emitter failures propagate unchanged. [Class-File API](../../../../glossary.md#term-class-file-api) [argument](../../../../glossary.md#term-argument)/verification failures and
checked hidden-definition/handle-resolution failures are wrapped once in
`IllegalArgumentException("generated kernel definition failed", cause)`, except for the explicit
verification message above. No partial result is installed in static state, registered, cached,
closed, or retried. Generation is stateless and safe for concurrent calls; equal specializations
and emitters produce byte-for-byte equal bytes before hidden definition, while their hidden
classes/artifacts remain distinct because caching is deferred.

`CpuGeneratedKernel` retains exact specialization, hidden [lookup](../../../../glossary.md#term-lookup), [hidden class](../../../../glossary.md#term-hidden-class), direct [method
handle](../../../../glossary.md#term-method-handle), exact method type, and a private copy of bytes. Construction verifies lookup-class,
handle-[owner](../../../../glossary.md#term-owner), and type consistency. It owns no `close()` [lifecycle](../../../../glossary.md#lifecycle) and overrides neither `equals`
nor [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests). No static collection retains generated artifacts.

## [Validation](../../../../glossary.md#term-validation) order and stable failures

Automated tests lock at least these failures in order:

1. `CpuLoweringFingerprint.of`: source null, source empty; `fromDigest`: digest null, exact
   32-byte length, then defensive retention without rehashing.<span data-glossary-navigation="true"> ([digest glossary](../../../../glossary.md#term-digest))</span>
2. [`Argument`](../../../../glossary.md#term-argument): data type, carrier, access, offset-baked consistency, non-negative baked offset,
   ordered non-negative strides, exact data-type/carrier match or exact-segment baked-zero rule,
   then baked primitive-array width alignment. Dynamic primitive-array offsets receive no
   generation-time alignment validation.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [carrier glossary](../../../../glossary.md#term-storage-carrier); [strides glossary](../../../../glossary.md#term-stride); [alignment glossary](../../../../glossary.md#term-alignment))</span>
3. `VectorShape`: [lane](../../../../glossary.md#term-lane) type, bit size, lane count, then exact supported-species resolution.
4. `CpuKernelSpecialization`: schema, lowering fingerprint, mode, argument list and entries,
   baked extents, dynamic count, vector presence/absence, byte order, unroll, tile, non-null tail,
   scalar-mode <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [vector glossary](../../../../glossary.md#term-vector); [byte order glossary](../../../../glossary.md#text-encoding-and-file-boundaries); [scalar glossary](../../../../glossary.md#term-scalar))</span>`Tail.NONE` compatibility, numerical mode, combine order, and class-file version.
5. Generator inputs and [lowering](../../../../glossary.md#lowering)-[fingerprint](../../../../glossary.md#term-fingerprint) match before class-byte work.
6. Carrier emitter argument range, read/write access, scalar/vector compatibility, alignment, and
   only then instruction emission.

Stable messages use exact [parameter](../../../../glossary.md#term-parameter)/index names such as `arguments[1]`, `bakedExtents[0]`,
`bakedElementStrides[2]`, and `dynamicExtentCount must be non-negative`. Ordinary [Class-File
API](../../../../glossary.md#term-class-file-api)/[JDK](../../../../glossary.md#term-java-development-kit-jdk) verification detail may remain a cause/detail list rather than a new public exception.

## Affected files

Expected [CPU](../../../../glossary.md#term-cpu) production/build paths:

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

Expected [planning](../../../../glossary.md#term-planning) paths:

- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review only, with a reasoned no-change conclusion unless a concrete contradiction is found:
architecture/focused architecture/ADRs, [kernel](../../../../glossary.md#kernel)-routes guide, CPU strategy note, documentation
rules/profiles, Planning/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)/Config/[Model](../../../../glossary.md#term-model) contracts, task-0001 source/tests/Javadocs,
root/settings/other [Gradle](../../../../glossary.md#term-gradle) files, architecture tests, [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), integration tests, and
CPU task 0001.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 22 paths:

- 14 [CPU](../../../../glossary.md#term-cpu) production/build paths;
- 3 CPU test paths;
- 2 explanatory documentation paths; and
- 3 [planning](../../../../glossary.md#term-planning) paths.

No existing CPU production or test path other than `execution/package-info.java` and the CPU
[module](../../../../glossary.md#term-module) build file may change. If another path, type, test split, build file, [dependency](../../../../glossary.md#term-dependency), public
surface, or semantic implementation is needed, stop and update planning or propose a bounded
follow-up.

## Java 26 Class-File and [Vector API](../../../../glossary.md#term-vector-api) build contract

- The root build continues to select Java [toolchain](../../../../glossary.md#term-toolchain) 26 and `options.release = 26`.
- `java.lang.classfile` and `CodeBuilder` are stable Java 26 APIs and require no preview or
  incubator flag.
- Java 26 `jdk.incubator.vector` requires explicit [module](../../../../glossary.md#term-module) [activation](../../../../glossary.md#term-activation). The [CPU](../../../../glossary.md#term-cpu) build file must add
  `--add-modules jdk.incubator.vector` to every CPU `JavaCompile`, the same module to CPU `Test`
  JVM arguments, and the equivalent option to the CPU <span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [arguments glossary](../../../../glossary.md#term-argument))</span>[`Javadoc`](../../../../glossary.md#term-javadoc) task. The intended task-local
  Gradle shape is:<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>

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

  [Gradle](../../../../glossary.md#term-gradle) prefixes the Javadoc option key, so `-add-modules` produces the tool's
  `--add-modules` option. The task must confirm the effective invocation instead of weakening the
  check if the local Gradle/Javadoc API differs.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Do not add `--enable-preview`, a `module-info.java`, a Maven dependency, a root/subproject-wide
  flag, or an option to another module. Final evidence must prove compile, test execution, and
  Javadoc all receive the CPU-local Vector module.<span data-glossary-navigation="true"> ([Maven glossary](../../../../glossary.md#term-maven); [dependency glossary](../../../../glossary.md#term-dependency); [compile glossary](../../../../glossary.md#compile); [Vector glossary](../../../../glossary.md#term-vector))</span>
- Incubator warnings are expected and are not semantic failures. Any missing-module failure in
  compile, test, or Javadoc is a task failure, not a reason to omit Vector validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Generated classes link only against Java 26/JDK classes already visible to the CPU module. No
  ASM or other bytecode dependency is permitted.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>

## Acceptance criteria

- [x] Exactly the listed package-private production types and no new public [CPU](../../../../glossary.md#term-cpu) type are present.
- [x] Schema version 1 and both typed fingerprints are immutable, content-based, deterministic,
      and locked by fixed canonical vectors plus equal/not-equal tests.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [immutable glossary](../../../../glossary.md#term-immutable); [canonical glossary](../../../../glossary.md#term-canonical-form); [vectors glossary](../../../../glossary.md#term-vector))</span>
- [x] The specialization contains every and only bytecode-relevant fact listed above; deep
      immutability, structural equality/hash, entry-type derivation, validation order, and stable
      messages are tested.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [bytecode glossary](../../../../glossary.md#term-bytecode); [structural equality glossary](../../../../glossary.md#term-structural-equality); [hash glossary](../../../../glossary.md#term-hash); [validation glossary](../../../../glossary.md#term-validation))</span>
- [x] Runtime identities, addresses, worker counts, chunk sizes, cache state, maps, strings used
      for dispatch, and mutable objects are absent from specialization equality/fingerprints.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [cache glossary](../../../../glossary.md#term-cache); [dispatch glossary](../../../../glossary.md#term-dispatch); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- [x] The family seam is a direct typed collaboration with separate scalar/vector emission and no
      registry, service locator, reflection, string dispatch, <span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator))</span>`Map<String, ?>`, generic config bag,
      broad facade, or god generator.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>
- [x] `CpuPortableExecutionMode.emit(...)` constructs only the selected scalar or Vector emitter
      plus shared carrier/loop/reduction emitters and calls exactly one matching family callback;
      the generator delegates exactly once to it and contains no scalar/Vector switch.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [callback glossary](../../../../glossary.md#term-callback))</span>
- [x] Separate scalar, Vector, carrier, loop/tile/tail, and partial/combine helpers contain only
      their stated low-level responsibilities; no production family semantic is implemented.
- [x] Class bytes are deterministic, [target](../../../../glossary.md#term-target) Java 26, have exactly one static typed `invoke`
      method, verify through `ClassFile.of().verify(bytes)`, define as a hidden nestmate, and
      resolve to one
      exact direct static method handle without adaptation.<span data-glossary-navigation="true"> ([nestmate glossary](../../../../glossary.md#term-nestmate); [method handle glossary](../../../../glossary.md#term-method-handle))</span>
- [x] Equal generation inputs produce identical bytes but distinct hidden artifacts without
      task-0003 caching.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact))</span>
- [x] [Heap](../../../../glossary.md#term-heap) primitive carriers, exact [`MemorySegment`](../../../../glossary.md#term-memorysegment) values, and a mixed signature are invoked
      without copying; array offsets, segment-relative zero offsets, access constraints, and
      direct typed argument order are proved.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- [x] Exactly scalar single-thread, scalar parallel, Vector API single-thread, and Vector API
      parallel modes exist. Synthetic probes execute in all four modes with equivalent bounded
      results.<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api))</span>
- [x] [Scalar](../../../../glossary.md#term-scalar) modes reject `SCALAR` and `MASKED` tails with the exact specified message; Vector
      modes accept `NONE`, `SCALAR`, and `MASKED` under the stated divisibility/tail contracts.
- [x] Baked primitive-array offsets reject misalignment to `DataType.byteWidth()` in the exact
      stated validation position, exact segments retain baked zero offset, and dynamic array
      offsets receive no generation-time alignment check.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span>
- [x] Parallel probes reuse [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup) and pass only assigned range primitives to generated
      entries; generated classes own no thread, pool, executor, scheduler, synchronization,
      cancellation, or failure mechanism.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [cancellation glossary](../../../../glossary.md#term-cancellation))</span>
- [x] Vector probes use one exact supported Java 26 species, full-lane and tail plumbing, heap and
      segment access, and no arbitrary lane promise.<span data-glossary-navigation="true"> ([lane glossary](../../../../glossary.md#term-lane))</span>
- [x] Range, tile, scalar/masked tail, partial, and fixed-order combine structures are tested with
      synthetic callbacks and never described as Model operation semantics.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation))</span>
- [x] The artifact retains exact hidden lookup/class/handle while reachable; no static collection
      retains it; collection/unloading timing is not asserted.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>
- [x] Failures are closed before advertising or invocation, and malformed/unverified bytes never
      produce a returned artifact.
- [x] `CpuCapabilityProvider` remains byte-for-byte unchanged, returns `false` for every non-null
      query, and the public CPU surface advertises no operation capability.
- [x] Generated/hot paths contain no reflection, `ServiceLoader`, map lookup, string dispatch,
      storage discovery, route/cache choice, graph/slot lookup, generic <span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>`Object` [argument](../../../../glossary.md#term-argument),
      `invokeWithArguments`, or per-[element](../../../../glossary.md#term-element) allocation.
- [x] The CPU build alone activates `jdk.incubator.vector` for compile, test, and Javadoc; Java 26
      toolchain/release remain unchanged; no preview flag, module descriptor, dependency, or ASM
      is added.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [toolchain glossary](../../../../glossary.md#term-toolchain); [release glossary](../../../../glossary.md#term-release); [module glossary](../../../../glossary.md#term-module); [descriptor glossary](../../../../glossary.md#term-descriptor); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- [x] Focused development tests and exactly one final CPU module test pass after executable Java
      stabilizes.
- [x] Every new/changed production declaration has meaningful complete [Javadoc](../../../../glossary.md#term-javadoc).
- [x] A separate clean documentation-focused pass finalizes affected Javadocs, CPU guidance,
      glossary impact, planning evidence, links, and status in the same overall change, reusing
      successful Java evidence unless it changes executable behavior.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- [x] CPU Javadoc, generated pages, Class-File parsing/verification, surface/mechanism/scope/status/
      Markdown/whitespace checks pass.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation development may [run](../../../../glossary.md#term-run) focused classes:

```bash
./gradlew :backends:cpu:test --tests '*CpuKernelSpecializationTest'
./gradlew :backends:cpu:test --tests '*CpuClassFileKernelGeneratorTest' --tests '*CpuGeneratedKernelShapeTest'
```

After executable Java and [CPU](../../../../glossary.md#term-cpu) build flags stabilize, run exactly once:

```bash
./gradlew :backends:cpu:test
```

The suite must use Java 26 `ClassFile.of().parse(bytes)` and
`ClassFile.of().verify(bytes)` for class version, method/field [shape](../../../../glossary.md#shape), [descriptors](../../../../glossary.md#term-descriptor), references, and
deterministic bytes. It must use exact `MethodHandle.type()` and
`invokeExact` probes instead of recurring manual reflection or `javap` as the primary [invariant](../../../../glossary.md#term-invariant).
If a [bytecode](../../../../glossary.md#term-bytecode) risk cannot be expressed with the Class-File [model](../../../../glossary.md#term-model), record one bounded `javap -v`
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
fences, final newlines, and trailing whitespace; exact 22-path [scope](../../../../glossary.md#term-scope); exactly one detailed CPU
0002 specification; CPU 0001 `Complete`; CPU 0002 synchronized to its implementation status; CPU
0003–0016 `Draft` with no detailed specs; and unchanged excluded paths.

Repository-wide Java and [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) validation are deferred to CPU task 0009 or [CI](../../../../glossary.md#term-continuous-integration-ci).
Task 0002 changes only CPU-local incubator flags and backend-private machinery. If a [dependency
edge](../../../../glossary.md#term-dependency-edge), shared build rule, architecture boundary, or advertised behavior changes, stop.

## Dependencies

- Complete [CPU](../../../../glossary.md#term-cpu) 0001 exact [heap](../../../../glossary.md#term-heap)/segment/mixed [arguments](../../../../glossary.md#term-argument), [binding](../../../../glossary.md#term-binding) seam, and worker/range foundation.
- Synchronized architecture permission for generated JVM-bytecode CPU computation kernels in CPU
  finalization after shared slot assignment.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [bytecode glossary](../../../../glossary.md#term-bytecode); [kernels glossary](../../../../glossary.md#kernel))</span>
- Complete Prepare finalization and Runtime prepared-executable/bound-invocation contracts,
  preserved without a production CPU implementation here.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Java 26 stable [Class-File API](../../../../glossary.md#term-class-file-api) and CPU-local `jdk.incubator.vector` [activation](../../../../glossary.md#term-activation).
- Current [Model](../../../../glossary.md#term-model) [`DataType`](../../../../glossary.md#data-type--datatype) and exact six task-0001 carrier mappings. Shape, layout, descriptors,
  operation signatures, and occurrences inform later family fingerprints but are not hot/generated
  inputs here.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [descriptors glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Current Config has no stable CPU target, determinism, tuning, or execution-mode contract. Task
  0002 introduces no substitute public config; its descriptor receives selected bytecode facts.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0003 remains Draft and owns bounded [single-flight](../../../../glossary.md#single-flight-and-least-recently-used-caching) generated-[artifact](../../../../glossary.md#term-artifact) caching and [lifetime](../../../../glossary.md#term-lifetime).
- CPU 0004 remains Draft and owns production portable analysis, candidate/specialization decisions,
  and finalization orchestration after slot assignment.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>
- CPU 0005–0008 remain Draft and own family semantic coverage.
- CPU 0009 remains Draft and owns advertised portable coverage closure and [conformance](../../../../glossary.md#term-conformance).
- CPU 0010–0016 remain Draft for optional native routes and tuning-[cache](../../../../glossary.md#term-cache) selection.
- No follow-up may treat task-0002 probes as [operation](../../../../glossary.md#operation) capability or a production route.

## Architecture impact

Expected impact: None.

This task implements the authorized [CPU](../../../../glossary.md#term-cpu)-internal mechanism and one CPU-local Java 26 [incubator
module](../../../../glossary.md#term-incubator-module) option. It changes no [ownership](../../../../glossary.md#term-ownership-meanings), [dependency edge](../../../../glossary.md#term-dependency-edge), shared [lifecycle](../../../../glossary.md#lifecycle), public [API](../../../../glossary.md#term-application-programming-interface-api), or
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
  0001 internals without a cross-package facade.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>
- Use Java 26 `ClassFile`/`CodeBuilder` as current mechanism while keeping it non-authoritative.
- Select hidden [nestmate](../../../../glossary.md#term-nestmate) classes with one static exact `invoke`; retain lookup/class/handle rather
  than a named loader or static registry.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [registry glossary](../../../../glossary.md#term-registry))</span>
- Use SHA-256 over explicit canonical typed encodings for cold compatibility identity, not
  security authentication or a persistent cache format.<span data-glossary-navigation="true"> ([compatibility identity glossary](../../../../glossary.md#identity-meanings-and-algebraic-identities))</span><span data-glossary-navigation="true"> ([SHA-256 glossary](../../../../glossary.md#term-sha-256); [canonical glossary](../../../../glossary.md#term-canonical-form); [identity glossary](../../../../glossary.md#term-identity-meanings); [authentication glossary](../../../../glossary.md#term-authentication))</span>
- Use `CpuLoweringFingerprint.fromDigest(byte[])` only to retain the already-derived 32-byte
  specialization digest without rehashing.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [digest glossary](../../../../glossary.md#term-digest))</span>
- Derive exact direct method types; do not use `Object[]`, spreaders, collectors, or adapters.
- Make four modes structurally distinct: single-thread accepts count; parallel accepts assigned
  range/index. Scheduling remains outside generated code.
- Keep reduction support structural; later family tasks attach semantics.
- Include [CPU](../../../../glossary.md#term-cpu) [Gradle](../../../../glossary.md#term-gradle) because Java 26 [Vector](../../../../glossary.md#term-vector) [compile](../../../../glossary.md#compile)/test/[Javadoc](../../../../glossary.md#term-javadoc) require local [module](../../../../glossary.md#term-module) [activation](../../../../glossary.md#term-activation).
- Retain class bytes for deterministic verification and later [cache](../../../../glossary.md#term-cache) design, without adding cache.
- `CpuPortableExecutionMode` owns the structural scalar-versus-Vector emitter dispatch through
  the exact fixed-surface <span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>`emit(...)` method; `CpuClassFileKernelGenerator` delegates without a
  scalar/Vector switch.
- [Scalar](../../../../glossary.md#term-scalar) execution modes accept only `Tail.NONE`; all three tail values remain valid for Vector
  modes under the existing divisibility/tail contracts.
- Baked primitive-array byte offsets are aligned to `DataType.byteWidth()` at specialization
  construction. Exact segments retain baked zero offset, while dynamic primitive-array offsets
  remain cold invocation values.

## Known limitations

- [CPU](../../../../glossary.md#term-cpu) still advertises and executes no [Model](../../../../glossary.md#term-model) [operation](../../../../glossary.md#operation).
- Only bounded synthetic test probes execute; they have no [Tensor](../../../../glossary.md#tensor), route, or [conformance](../../../../glossary.md#term-conformance) meaning.
- No production CPU analysis, finalizer, [prepared executable](../../../../glossary.md#prepared-executable--preparedexecutable), or generated [bound invocation](../../../../glossary.md#bound-invocation--boundinvocation) exists.
- Equal generation requests define distinct hidden classes until CPU 0003 adds bounded caching.
- Hidden-[class unloading](../../../../glossary.md#weak-reference-and-weak-interning) is collector-dependent; no collection deadline is asserted.
- [Vector API](../../../../glossary.md#term-vector-api) incubates in Java 26; no later-[JDK](../../../../glossary.md#term-java-development-kit-jdk) source/[binary](../../../../glossary.md#term-binary-descriptor-and-metadata) compatibility is promised.
- [BFLOAT16](../../../../glossary.md#term-bfloat16)/[BOOL](../../../../glossary.md#data-type--datatype) have [carrier](../../../../glossary.md#term-storage-carrier) plumbing but no claimed [Vector](../../../../glossary.md#term-vector) arithmetic semantics.
- Range/tile/tail/reduction helpers choose no algorithm, accuracy, accumulation, determinism, or
  performance threshold.
- Artifact persistence, corruption handling, broader target fingerprints, and JIT reuse remain
  deferred.<span data-glossary-navigation="true"> ([Artifact glossary](../../../../glossary.md#term-artifact); [persistence glossary](../../../../glossary.md#term-persistence); [target glossary](../../../../glossary.md#term-target); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit))</span>
- No generated-artifact cache, production CPU analysis/finalizer, prepared executable, bound
  generated invocation, family semantic lowering, or advertised Model-operation coverage exists.
  These are deliberately deferred to CPU tasks 0003–0009.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [lowering glossary](../../../../glossary.md#lowering))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

This 2026-08-03 [planning](../../../../glossary.md#term-planning) repair changed only the [CPU](../../../../glossary.md#term-cpu) 0002 task, CPU master plan, and roadmap. It
made the [dispatch](../../../../glossary.md#term-dispatch) [owner](../../../../glossary.md#term-owner)/[signature](../../../../glossary.md#term-signature), trusted-[digest](../../../../glossary.md#term-digest) factory, tail [matrix](../../../../glossary.md#term-matrix-meanings), and byte-offset rules
actionable without changing architecture, public [API](../../../../glossary.md#term-application-programming-interface-api), package placement, or the 22-path maximum.

- `ruby /tmp/check_cpu_0002_repair.rb` — passed Ready/In-progress synchronization, the exact
  repaired surface and invariants, unchecked acceptance criteria, empty implementation completion
  summary, CPU 0001 <span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>`Complete`, CPU 0003–0016 `Draft` with no later detailed specifications,
  exactly the three authorized planning paths, and 20 total task paths under the 22-path maximum.
- `ruby /tmp/check_cpu_0002_markdown.rb
  docs/planning/backends/cpu/tasks/0002-portable-class-file-api-generator-foundation.md
  docs/planning/backends/cpu/master-plan.md docs/planning/roadmap.md` — passed three Markdown files,
  368 local links, balanced fences, final newlines, and trailing-whitespace checks.
- `git diff --check` — passed.
- The obsolete blocker-era `/tmp/check_cpu_0002_status.rb` was run once and rejected the repaired
  Ready state because it hard-codes the former status and completion record. It supplied no
  validation result and was replaced by the repair-specific checker above.
- No Java, [Gradle](../../../../glossary.md#term-gradle), test, or [Javadoc](../../../../glossary.md#term-javadoc) command ran because this repair changes planning only.

Final implementation and documentation evidence:

- Implementation context `019fc815-42aa-7de2-8970-a2fcab3a390e` recorded the final focused
  command passing 3 suites and 18 tests and the sole new final
  `./gradlew :backends:cpu:test` passing 9 suites and 34 tests, both with zero failures, errors,
  or skips. It also recorded Class-File parsing/verification, exact [`MethodHandle`](../../../../glossary.md#native-numerical-library-and-instruction-architecture) invocation,
  source/import/forbidden-mechanism scans, `javap` inspection, exact 20-path implementation
  scope, final newlines, and `git diff --check` as passing. No executable file changed after that
  final suite.
- Mandatory clean documentation context `019fc81f-f023-7ce2-b81e-9c0af980751f` independently
  reviewed the General, API/Javadoc, Backend Guide, Architecture, Planning, and Example profiles;
  the architecture, ADR, planning, task-0001, source, test, build, CPU guide, glossary, and
  generated-page contracts listed by this task; and the actual uncommitted diff. It changed only
  Javadocs and the five authorized explanatory/planning documents, so it did not rerun Java tests.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- `./gradlew :backends:cpu:javadoc` — final [run](../../../../glossary.md#term-run) passed all 11 tasks with CPU-local
  `jdk.incubator.vector` activation. The expected incubator warning and 28 pre-existing task-0001
  or implicit-default-constructor doclint warnings remained non-failing; every task-0002 page was
  generated. An earlier documentation run failed on invalid record-level <span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation))</span>`@throws` tags; moving
  those contracts to the explicit compact-constructor Javadocs resolved all task-0002 doclet
  errors.
- Generated-page assertions — passed for `CpuKernelSpecialization`, all four-mode and family/
  low-level emitter pages, `CpuClassFileKernelGenerator`, `CpuGeneratedKernel`, and the execution
  package limitation. Manual content checks found the complete specialization contract, fresh
  identity-artifact behavior, and synthetic-probe limitation in rendered HTML.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [identity glossary](../../../../glossary.md#term-identity-meanings); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- `ruby /tmp/check_cpu_0002_markdown.rb
  docs/planning/backends/cpu/tasks/0002-portable-class-file-api-generator-foundation.md
  docs/planning/backends/cpu/master-plan.md docs/planning/roadmap.md
  docs/backend-guide/cpu-backend.md docs/glossary.md` — passed five Markdown files, 671 local
  links, 287 anchors, balanced fences, final newlines, and trailing whitespace. The first run
  exposed an older-Ruby checker incompatibility and two stale OpenBLAS glossary anchors in the CPU
  guide; the checker and the two authorized links were corrected before the passing run.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- `ruby /tmp/check_cpu_0002_complete.rb` — passed exact 22-path [scope](../../../../glossary.md#term-scope); CPU 0001/0002 `Complete`,
  CPU project `In progress`, CPU 0003–0016 `Draft`, and no later detailed task; package-private
  source surface; import and forbidden-mechanism exclusions; exact four modes; CPU-local Vector
  compile/test/Javadoc flags with no preview flag; and required generated Javadoc pages.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [compile glossary](../../../../glossary.md#compile))</span>
- Changed-file newline/whitespace validation — passed all 22 paths, including untracked Java and
  test files. Excluded-path comparison proved no changes to `ARCHITECTURE.md`, focused architecture
  pages, ADRs, architecture/conformance/integration tests, the public <span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>`CpuCapabilityProvider`,
  shared modules/extensions, root settings/build, or shared build logic.<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
- `git diff --check` — passed on the final combined change.

Reasoned no-change conclusions:

- `CpuCapabilityProvider` remains byte-for-byte unchanged and fail-closed; synthetic probes do not
  justify advertising any operation.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [operation glossary](../../../../glossary.md#operation))</span>
- Prepare, Runtime, Compiler, and Planning APIs remain accurate and unchanged because this task
  adds neither a preparer/finalizer nor a prepared/runtime route, compiler behavior, planning
  ownership input, or shared lifecycle handoff.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Compiler glossary](../../../../glossary.md#term-compiler); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- `ARCHITECTURE.md`, focused architecture pages, ADRs, architecture tests, and the CPU strategy
  note remain accurate: the existing contract already permits implementation-neutral generated
  JVM bytecode inside CPU finalization and deliberately does not prescribe Class-File or Vector
  APIs. No ownership, lifecycle, or dependency rule changed.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [bytecode glossary](../../../../glossary.md#term-bytecode); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Backend-conformance and integration tests remain unchanged because no Model operation is
  advertised, lowered, prepared, or executed. The new tests correctly remain CPU-module
  synthetic foundation tests.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- The CPU build change is limited to the required Java 26 Vector module activation for CPU
  compile, test, and Javadoc. No dependency coordinate, root/shared Gradle rule, preview option,
  module descriptor, ASM dependency, or other module build changed.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Model, Config, Backend Contract, Trace, OpenBLAS provider, Prepare, Runtime, Compiler, Planning,
  Engine, extensions, and other backends require no documentation or source change because the
  task exposes no public/shared contract and makes no route, execution, storage-ownership, or
  capability claim across those boundaries.<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine))</span>

## Implementation notes

Implementation added the exact package-private [schema](../../../../glossary.md#term-schema), [lowering](../../../../glossary.md#lowering) and [specialization](../../../../glossary.md#term-specialization) fingerprints,
four-mode vocabulary, family-emission seam, separate [scalar](../../../../glossary.md#term-scalar)/[Vector](../../../../glossary.md#term-vector)/[carrier](../../../../glossary.md#term-storage-carrier)/loop/reduction helpers,
[Class-File API](../../../../glossary.md#term-class-file-api) generator, and retained hidden-class [artifact](../../../../glossary.md#term-artifact) specified above. The specialization
derives one exact direct `void` entry type from ordered [heap](../../../../glossary.md#term-heap)-array and [`MemorySegment`](../../../../glossary.md#term-memorysegment) carriers,
dynamic offsets/extents, and single-thread or parallel range controls. Generation validates the
family [fingerprint](../../../../glossary.md#term-fingerprint) before emission, verifies deterministic Java 26 bytes, defines a fresh hidden
[nestmate](../../../../glossary.md#term-nestmate), and resolves the exact direct static handle without adaptation.

The tests supply the only family emitter and exercise bounded synthetic copy and structural
probes. They cover exact six-array carrier mapping, segment and mixed [signatures](../../../../glossary.md#term-signature), all four modes,
external [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup) range [dispatch](../../../../glossary.md#term-dispatch), access restrictions, tail/reduction seams, [canonical](../../../../glossary.md#term-canonical-form)
fingerprints, class [shape](../../../../glossary.md#shape), direct handle types, deterministic equal bytes, and distinct hidden
artifacts for equal requests. No production [operation](../../../../glossary.md#operation) meaning or route was added.

The clean documentation pass independently reviewed source, tests, build flags, generated
[Javadoc](../../../../glossary.md#term-javadoc), [CPU](../../../../glossary.md#term-cpu) guidance, glossary terminology, and [planning](../../../../glossary.md#term-planning) [synchronization](../../../../glossary.md#term-synchronization). It clarified the
current generator boundary, typed specialization and fingerprints, four modes, direct carrier
forms, family/emitter [ownership](../../../../glossary.md#term-ownership-meanings), hidden-artifact [lifetime](../../../../glossary.md#term-lifetime), absence of caching, and the continued
[fail-closed](../../../../glossary.md#term-fail-closed) CPU capability boundary without changing executable Java.

## Completion summary

- Completed changes: added the package-private deterministic Java 26 generated-kernel foundation,
  exact typed specialization/fingerprint contracts, four structural execution modes, low-level
  emission seams, verified hidden-class artifact construction, and CPU-local Vector module flags.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [specialization glossary](../../../../glossary.md#term-specialization); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [artifact glossary](../../../../glossary.md#term-artifact); [Vector glossary](../../../../glossary.md#term-vector); [module glossary](../../../../glossary.md#term-module))</span>
- Files changed or created: 14 CPU production/build paths, three CPU test paths, the CPU guide,
  glossary, this task, CPU master plan, and roadmap; exactly 22 paths.
- Tests and [validation](../../../../glossary.md#term-validation): implementation context `019fc815-42aa-7de2-8970-a2fcab3a390e`
  recorded the focused 3-suite/18-test pass and the sole final CPU 9-suite/34-test pass, both with
  zero failures, errors, or skips. The documentation pass reused that evidence because it changed
  no executable Java behavior and completed CPU Javadoc and documentation/surface/scope/status/
  whitespace validation.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Documentation-agent review: the mandatory clean documentation-focused pass independently
  finalized all affected production Javadocs, CPU guidance, glossary impact, planning evidence,
  and synchronized status.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: `docs/backend-guide/cpu-backend.md` now explains the current generator
  foundation and its limits; architecture pages, ADRs, and the CPU strategy note remain accurate.
- Javadoc review: every new or changed production declaration was reviewed for purpose, behavior,
  invariants, validation, ownership, side effects, parameters, returns, and expected failures.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- Glossary impact: added current CPU specialization and generated-artifact definitions and
  clarified the generated-kernel entry without turning implementation mechanisms into invariants.
- Unresolved issues: None within task 0002 [scope](../../../../glossary.md#term-scope).
- Follow-up required: None for task 0002; [CPU](../../../../glossary.md#term-cpu) 0003–0016 remain Draft planned work.

Status: Complete
