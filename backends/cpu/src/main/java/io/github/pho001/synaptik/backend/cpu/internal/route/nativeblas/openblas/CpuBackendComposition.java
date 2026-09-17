package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuConcurrencyBudget;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBorrowedBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuHostSnapshotExporter;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPreparedScheduleAssembler;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.prepare.PreparedScheduleAssembler;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Owns the CPU-private exact/default preparation collaborators and optional OpenBLAS lifetime.
 *
 * <p>One composition always describes the fixed {@code cpu/host} device and retains the portable
 * route. Bounded automatic discovery and qualification may add one eligible OpenBLAS candidate;
 * provider presence neither changes backend identity nor guarantees native selection. The owner
 * is safe for concurrent collaborator access and idempotent close, but recipes and runs must not
 * outlive it.</p>
 */
public final class CpuBackendComposition implements AutoCloseable {
    private static final BackendAvailabilitySnapshot AVAILABILITY =
            new BackendAvailabilitySnapshot(CpuCapabilityProvider.CPU_BACKEND_ID,
                    Map.of(new BackendDeviceId(CpuCapabilityProvider.CPU_BACKEND_ID, "host"),
                            DeviceClass.CPU));

    private final CpuConcurrencyBudget budget;
    private final Optional<CpuOpenBlasCoordinator> coordinator;
    private final Optional<CpuOpenBlasQualification> qualification;
    private final CpuPartitionPreparer preparer = new CpuPartitionPreparer();
    private final CpuPartitionFinalizer finalizer;
    private final CpuPreparedScheduleAssembler assembler;
    private final AtomicBoolean closed = new AtomicBoolean();

    /**
     * Creates one composition from already established portable and optional native state.
     *
     * @param budget the non-null shared CPU concurrency budget retained by collaborators
     * @param coordinator the non-null optional coordinator whose lifetime this owner assumes
     * @param qualification the non-null optional qualification associated with the coordinator
     * @throws NullPointerException if an argument is {@code null}
     */
    private CpuBackendComposition(CpuConcurrencyBudget budget,
            Optional<CpuOpenBlasCoordinator> coordinator,
            Optional<CpuOpenBlasQualification> qualification) {
        this.budget = Objects.requireNonNull(budget, "budget");
        this.coordinator = Objects.requireNonNull(coordinator, "coordinator");
        this.qualification = Objects.requireNonNull(qualification, "qualification");
        this.finalizer = new CpuPartitionFinalizer(Optional.empty(), Optional.empty(), budget,
                coordinator);
        this.assembler = new CpuPreparedScheduleAssembler(() -> !closed.get());
    }

    /**
     * Opens the production exact/default composition using bounded automatic discovery.
     *
     * @return a new non-null composition; optional discovery or setup failure yields a portable
     *     composition
     * @throws Error if automatic discovery reports a fatal error
     */
    public static CpuBackendComposition openAutomatic() {
        try {
            return open(CpuOpenBlasDiscovery.discover(CpuOpenBlasDiscoveryRequest.automatic()));
        } catch (RuntimeException optionalFailure) {
            return portable();
        }
    }

    /**
     * Completes ownership transfer and qualification for one caller-owned discovery session.
     *
     * @param session the non-null open discovery session supplying optional provider ownership
     * @return a new non-null composition with qualified native state when setup succeeds, or a
     *     portable composition when the session is unavailable or optional setup fails
     * @throws NullPointerException if {@code session} is {@code null}
     * @throws Error if setup or cleanup reports a fatal error
     */
    static CpuBackendComposition open(CpuOpenBlasDiscoverySession session) {
        Objects.requireNonNull(session, "session");
        CpuConcurrencyBudget budget = new CpuConcurrencyBudget(1);
        CpuOpenBlasCoordinator coordinator = null;
        try {
            if (session.invocation().isEmpty()) {
                session.close();
                return new CpuBackendComposition(budget, Optional.empty(), Optional.empty());
            }
            coordinator = session.transferToCoordinator(budget);
            CpuOpenBlasQualification qualification =
                    CpuOpenBlasQualifier.qualify(session.result(), coordinator);
            session.close();
            return new CpuBackendComposition(budget, Optional.of(coordinator),
                    Optional.of(qualification));
        } catch (RuntimeException optionalFailure) {
            closeAfterFailure(coordinator, session, optionalFailure);
            return new CpuBackendComposition(budget, Optional.empty(), Optional.empty());
        } catch (Error fatalFailure) {
            closeAfterFailure(coordinator, session, fatalFailure);
            throw fatalFailure;
        }
    }

    /**
     * Creates one independent portable-only exact/default composition.
     *
     * @return a new non-null open portable composition with capacity-one concurrency
     */
    private static CpuBackendComposition portable() {
        return new CpuBackendComposition(new CpuConcurrencyBudget(1), Optional.empty(),
                Optional.empty());
    }

    /**
     * Returns the fixed host CPU availability fact without provider discovery.
     *
     * @return the retained immutable snapshot containing exactly the {@code cpu/host} CPU device
     */
    public BackendAvailabilitySnapshot availabilitySnapshot() {
        return AVAILABILITY;
    }

    /**
     * Builds the single positional CPU preparation after validating complete partition coverage.
     *
     * @param artifacts non-null compile artifacts containing exactly one non-empty CPU-owned
     *     partition with complete resolved facts required by current CPU analysis
     * @return a non-null immutable singleton preparation hiding CPU-private types behind shared
     *     Prepare roles
     * @throws NullPointerException if {@code artifacts} is {@code null}
     * @throws IllegalStateException if this composition is closed
     * @throws IllegalArgumentException if partition count, membership, ownership, graph
     *     projection, or resolved CPU boundary facts are unsupported or incomplete
     */
    public List<PartitionPreparation<?, ?>> preparations(CompileArtifacts artifacts) {
        requireOpen();
        Objects.requireNonNull(artifacts, "artifacts");
        if (artifacts.partitions().size() != 1) {
            throw new IllegalArgumentException(
                    "CPU integration requires exactly one non-empty CPU partition");
        }
        var partition = artifacts.partitions().getFirst();
        if (partition.nodeIds().isEmpty()
                || !partition.owner().equals(CpuCapabilityProvider.CPU_BACKEND_ID)) {
            throw new IllegalArgumentException(
                    "CPU integration requires exactly one non-empty CPU partition");
        }

        CpuPartitionAnalysisInputs inputs = analysisInputs(artifacts);
        PartitionPreparation<CpuPartitionAnalysisInputs,
                io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan>
                preparation = new PartitionPreparation<>(inputs, preparer, finalizer);
        return List.of(preparation);
    }

    /**
     * Returns the retained immutable schedule assembler while this owner is open.
     *
     * @return the non-null retained assembler; ownership remains with this composition
     * @throws IllegalStateException if this composition is closed
     */
    public PreparedScheduleAssembler scheduleAssembler() {
        requireOpen();
        return assembler;
    }

    /**
     * Validates intrinsic host-storage facts and creates a non-owning CPU representation.
     *
     * <p>No expected logical binding descriptor, required span, or access role is available at
     * this boundary. Those comparisons remain the responsibility of later typed Engine binding.</p>
     *
     * @param storage non-null live host storage whose exact segment is accessible to the current
     *     thread and whose capacity, byte size, data type, and observable heap carrier agree
     * @return a new non-null borrowed representation; closing it does not close {@code storage}
     * @throws NullPointerException if {@code storage} or one of its required intrinsic facts is
     *     {@code null}
     * @throws ArithmeticException if element-capacity to byte-size conversion overflows
     * @throws IllegalArgumentException if intrinsic geometry, data type, or observable heap
     *     carrier is inconsistent or unsupported
     * @throws IllegalStateException if this composition or storage is closed, or the exact segment
     *     is inaccessible to the current thread
     */
    public BufferRepresentation borrow(HostTensorStorage storage) {
        requireOpen();
        Objects.requireNonNull(storage, "storage");
        var type = Objects.requireNonNull(storage.dataType(), "storage.dataType()");
        long expectedBytes = Math.multiplyExact(storage.elementCapacity(), type.byteWidth());
        if (storage.elementCapacity() < 0 || expectedBytes != storage.byteSize()) {
            throw new IllegalArgumentException("host storage element and byte geometry disagree");
        }
        var segment = Objects.requireNonNull(storage.segment(), "storage.segment()");
        if (!storage.isAlive() || !segment.scope().isAlive()) {
            throw new IllegalStateException("host storage scope is not alive");
        }
        if (!segment.isAccessibleBy(Thread.currentThread())) {
            throw new IllegalStateException("host storage is inaccessible to the current thread");
        }
        if (segment.byteSize() != storage.byteSize()) {
            throw new IllegalArgumentException("host storage and segment byte sizes disagree");
        }
        CpuBorrowedBuffer borrowed = CpuBorrowedBuffer.borrow(storage);
        borrowed.argument();
        return borrowed;
    }

    /**
     * Copies one supported CPU publication representation to detached canonical host bytes.
     *
     * <p>The delegate reads the representation without retaining, closing, mutating, transferring,
     * or changing its validity. The caller must keep this composition and the representation's
     * result lease open, preserve current-thread accessibility, and prevent source mutation or
     * closure for the synchronous call. Independent valid calls may execute concurrently, but a
     * racing source mutation has no atomic-snapshot guarantee.</p>
     *
     * @param representation non-null exact current CPU publication representation borrowed from
     *     its open result lease; ownership remains with that result
     * @param descriptor non-null fully static resolved logical descriptor paired with that exact
     *     publication occurrence; inspected but not retained or mutated
     * @param maximumBytes non-negative caller payload limit in bytes
     * @return fresh non-null caller-owned mutable bytes in canonical row-major logical order and
     *     fixed big-endian element encoding; never {@code null}
     * @throws NullPointerException if an object argument is {@code null}
     * @throws IllegalStateException if this composition or the representation is closed, or the
     *     representation is inaccessible to the current thread
     * @throws IllegalArgumentException if the byte limit is negative; the shape is not fully static
     *     or the layout is unresolved; the result exceeds the limit or JVM array ceiling; the
     *     representation class, data type, element geometry, carrier, or capacity is incompatible;
     *     or represented BOOL content is not exactly {@code 0} or {@code 1}
     * @throws ArithmeticException if checked element-count, byte-count, or address arithmetic
     *     overflows
     * @throws OutOfMemoryError if the JVM cannot allocate the otherwise valid result array
     */
    public byte[] copyToCanonicalHostBytes(BufferRepresentation representation,
            TensorDescriptor descriptor, long maximumBytes) {
        requireOpen();
        return CpuHostSnapshotExporter.copy(representation, descriptor, maximumBytes);
    }

    /**
     * Derives the immutable exact/default CPU-private facts for the accepted partition.
     *
     * @param artifacts the non-null already validated one-partition compile artifacts
     * @return non-null CPU analysis inputs with no tuning decision and, when qualification exists,
     *     only the bounded single-thread FLOAT32/FLOAT64 OpenBLAS candidate
     * @throws IllegalArgumentException if an eligible MATMUL boundary lacks a graph node or
     *     resolved descriptor
     */
    private CpuPartitionAnalysisInputs analysisInputs(CompileArtifacts artifacts) {
        List<CpuPartitionAnalysisInputs.BoundaryStorageFact> storageFacts = List.of();
        var partition = artifacts.partitions().getFirst();
        if (partition.nodeIds().size() == 1) {
            var node = artifacts.graph().nodes().stream()
                    .filter(candidate -> candidate.id().equals(partition.nodeIds().getFirst()))
                    .findFirst().orElseThrow(() -> new IllegalArgumentException(
                            "CPU partition node is absent from compile graph"));
            if (node.operation().kind() == MatmulKind.MATMUL
                    && node.inputs().size() == 2 && node.outputs().size() == 1) {
                var bindable = java.util.Set.copyOf(artifacts.constants().bindableInputs());
                var values = artifacts.graph().values().stream().collect(java.util.stream.Collectors
                        .toMap(value -> value.id(), value -> value.descriptor()));
                var facts = new java.util.ArrayList<CpuPartitionAnalysisInputs.BoundaryStorageFact>(3);
                for (var valueId : List.of(node.inputs().get(0), node.inputs().get(1),
                        node.outputs().getFirst())) {
                    var descriptor = values.get(valueId);
                    if (descriptor == null || descriptor.layout().isEmpty()) {
                        throw new IllegalArgumentException(
                                "CPU boundary requires a resolved descriptor");
                    }
                    facts.add(new CpuPartitionAnalysisInputs.BoundaryStorageFact(
                            !bindable.contains(valueId), descriptor.dataType().byteWidth()));
                }
                storageFacts = List.copyOf(facts);
            }
        }

        var route = qualification
                .<CpuPartitionAnalysisInputs.OpenBlasRouteConfig>map(value ->
                        CpuPartitionAnalysisInputs.OpenBlasRouteConfig.qualifiedSingleThread(
                                value, 1_000, 10, 100, 1, 1, 1, 1, 1))
                .orElse(CpuPartitionAnalysisInputs.OpenBlasRouteConfig.DISABLED);
        return new CpuPartitionAnalysisInputs(false, List.of(),
                CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT,
                CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED, false,
                CpuPartitionAnalysisInputs.PartialReductionEvidence.NONE, storageFacts, route);
    }

    /**
     * Rejects access after composition closure.
     *
     * @throws IllegalStateException if this composition is closed
     */
    private void requireOpen() {
        if (closed.get()) {
            throw new IllegalStateException("CPU backend integration is closed");
        }
    }

    /**
     * Attempts deterministic cleanup of partially transferred optional provider ownership.
     *
     * @param coordinator the possibly null transferred coordinator to close first
     * @param session the non-null discovery session to close after the coordinator
     * @param primary the non-null setup failure that receives distinct cleanup failures as
     *     suppressed exceptions
     */
    private static void closeAfterFailure(CpuOpenBlasCoordinator coordinator,
            CpuOpenBlasDiscoverySession session, Throwable primary) {
        if (coordinator != null) {
            try {
                coordinator.close();
            } catch (RuntimeException | Error cleanup) {
                if (cleanup != primary) primary.addSuppressed(cleanup);
            }
        }
        try {
            session.close();
        } catch (RuntimeException | Error cleanup) {
            if (cleanup != primary) primary.addSuppressed(cleanup);
        }
    }

    /**
     * Publishes closed state, then quiesces and closes the optional provider owner exactly once.
     * Repeated and concurrent calls are idempotent.
     *
     * @throws RuntimeException if provider thread restoration or cleanup reports an unchecked
     *     failure
     * @throws Error if provider thread restoration or cleanup reports an error
     */
    @Override
    public void close() {
        if (closed.compareAndSet(false, true) && coordinator.isPresent()) {
            coordinator.orElseThrow().close();
        }
    }
}
