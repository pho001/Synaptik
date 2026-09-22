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
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.prepare.PreparedScheduleAssembler;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
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
     * @return a non-null immutable preparation hiding CPU-private types behind shared
     *     Prepare roles
     * @throws IllegalStateException if this composition is closed
     */
    public PartitionPreparation<?, ?> preparation() {
        requireOpen();
        return new PartitionPreparation<>(analysisInputs(Optional.empty(),
                CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED), preparer, finalizer);
    }

    /**
     * Performs fresh CPU analysis and returns the current complete tunable batch, if any.
     *
     * @param context non-null exact stable partition projection
     * @return the freshly analyzed immutable batch, or empty when the valid workload is not
     *     currently tunable
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalStateException if this composition is closed
     * @throws IllegalArgumentException if the projection is outside the supported domain
     */
    public Optional<CpuOpenBlasTuningBatch> tuningBatch(PrepareContext<?> context) {
        requireOpen();
        return analyze(cpuContext(context, analysisInputs(Optional.empty(),
                CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED)))
                .plan().openBlasTuningBatch();
    }

    /**
     * Freshly enumerates the complete retained CPU plan identities under the fixed profile.
     *
     * <p>The method revalidates Phase-1 state and performs analysis only. It neither prepares nor
     * executes a recipe and does not time or rank alternatives.</p>
     *
     * @param context non-null exact stable one-partition projection; inspected but not mutated
     * @param phaseOneDecision non-null optional exact authenticated local selection, or empty for
     *     freshly proved absence; the decision is borrowed
     * @return a new immutable complete-plan identity snapshot and completeness outcome
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the projection or Phase-1 state is invalid
     * @throws ArithmeticException if exact candidate or resource arithmetic overflows
     * @throws IllegalStateException if this composition is closed
     */
    public CpuPartitionPreparer.CompletePlanCandidates completePlanCandidates(
            PrepareContext<?> context, Optional<CpuOpenBlasTuningDecision> phaseOneDecision) {
        requireOpen();
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(phaseOneDecision, "phaseOneDecision");
        validatePhaseOne(context, phaseOneDecision);
        CpuPartitionAnalysisInputs inputs = analysisInputs(phaseOneDecision,
                new CpuPartitionAnalysisInputs.MaterializationPolicy(true, 0, 1, 3, 1,
                        1, Long.MAX_VALUE, 0, 0));
        return preparer.completePlanCandidates(cpuContext(context, inputs));
    }

    /**
     * Freshly prepares one exact retained complete plan without heuristic substitution.
     *
     * <p>The returned collaboration fixes authoritative CPU analysis and finalization for the
     * selected plan. Shared Prepare still owns assignment, finalization invocation, schedule
     * assembly, validation, and construction of the complete execution. This operation creates
     * no {@code RunState}, binds no representative input, and executes or times no trial.</p>
     *
     * @param context non-null exact stable one-partition projection; inspected but not mutated
     * @param phaseOneDecision non-null optional exact authenticated local selection, or empty for
     *     freshly proved absence; the decision is borrowed
     * @param selectedPlan non-null immutable retained complete-plan identity and association
     *     fingerprint; inspected but not mutated
     * @return a fresh immutable CPU-owned partition preparation borrowing this composition's
     *     lifetime
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if any fresh compatibility or selection check fails
     * @throws ArithmeticException if exact preparation geometry overflows
     * @throws IllegalStateException if this composition is closed
     */
    public PartitionPreparation<?, ?> completePlanPreparation(PrepareContext<?> context,
            Optional<CpuOpenBlasTuningDecision> phaseOneDecision,
            CpuPartitionPreparer.SelectedCompletePlan selectedPlan) {
        requireOpen();
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(phaseOneDecision, "phaseOneDecision");
        Objects.requireNonNull(selectedPlan, "selectedPlan");
        if (selectedPlan.schemaVersion() != 1) {
            throw new IllegalArgumentException("CPU complete-plan schema is unsupported");
        }
        validatePhaseOne(context, phaseOneDecision);
        CpuPartitionAnalysisInputs inputs = analysisInputs(phaseOneDecision,
                new CpuPartitionAnalysisInputs.MaterializationPolicy(true, 0, 1, 3, 1,
                        1, Long.MAX_VALUE, 0, 0));
        var selectedPreparer = new io.github.pho001.synaptik.prepare.analysis.BackendPartitionPreparer<
                CpuPartitionAnalysisInputs,
                io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan>() {
            @Override public BackendPartitionAnalysis<io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan>
                    analyze(PrepareContext<CpuPartitionAnalysisInputs> context) {
                return preparer.analyzeSelected(context, selectedPlan);
            }
        };
        return new PartitionPreparation<>(inputs, selectedPreparer, finalizer);
    }

    private void validatePhaseOne(PrepareContext<?> context,
            Optional<CpuOpenBlasTuningDecision> decision) {
        Optional<CpuOpenBlasTuningBatch> fresh = analyze(cpuContext(context,
                analysisInputs(Optional.empty(),
                        CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED)))
                .plan().openBlasTuningBatch();
        if (fresh.isPresent() != decision.isPresent()) {
            throw new IllegalArgumentException("CPU Phase-1 eligibility changed");
        }
        if (decision.isPresent()) {
            CpuOpenBlasTuningBatch.Candidate matched = decision.orElseThrow()
                    .match(fresh.orElseThrow()).orElseThrow(() ->
                            new IllegalArgumentException(
                                    "CPU Phase-1 decision is stale or incompatible"));
            BackendPartitionAnalysis<io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan>
                    selected = analyze(cpuContext(context, analysisInputs(decision,
                            CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED)));
            if (!selected.plan().selectedOpenBlasTuningCandidate().orElseThrow()
                    .equals(matched.identity())) {
                throw new IllegalArgumentException("CPU Phase-1 decision was not selected");
            }
        }
    }

    /**
     * Re-analyzes one exact CPU-owned tuning decision without heuristic fallback.
     *
     * @param context non-null exact stable projection retained by the tuning association
     * @param decision non-null CPU decision to validate against fresh authoritative analysis
     * @return an immutable CPU-owned partition preparation for exactly the selected candidate
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalStateException if this composition is closed
     * @throws IllegalArgumentException if the decision is stale, ineligible, or not selected by
     *     the fresh batch
     */
    public PartitionPreparation<?, ?> selectedPreparation(PrepareContext<?> context,
            CpuOpenBlasTuningDecision decision) {
        requireOpen();
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(decision, "decision");
        BackendPartitionAnalysis<io.github.pho001.synaptik.backend.cpu.internal.prepare
                .CpuPartitionPreparationPlan> analysis = analyze(cpuContext(context,
                        analysisInputs(Optional.of(decision),
                                CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED)));
        var batch = analysis.plan().openBlasTuningBatch().orElseThrow(() ->
                new IllegalArgumentException("CPU tuning decision is no longer eligible"));
        var matched = decision.match(batch).orElseThrow(() ->
                new IllegalArgumentException("CPU tuning decision is stale or incompatible"));
        if (!analysis.plan().selectedOpenBlasTuningCandidate().orElseThrow()
                .equals(matched.identity())) {
            throw new IllegalArgumentException("CPU tuning decision was not selected");
        }
        return new PartitionPreparation<>(analysisInputs(Optional.of(decision),
                CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED), preparer, finalizer);
    }

    /**
     * Reconstructs shared Prepare's authoritative partition projection and runs CPU analysis.
     *
     * @param context non-null exact stable CPU context to analyze
     * @return the fresh non-null immutable CPU analysis
     */
    private BackendPartitionAnalysis<io.github.pho001.synaptik.backend.cpu.internal.prepare
            .CpuPartitionPreparationPlan> analyze(
                    PrepareContext<CpuPartitionAnalysisInputs> context) {
        return preparer.analyze(context);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> cpuContext(
            PrepareContext<?> context, CpuPartitionAnalysisInputs inputs) {
        Objects.requireNonNull(context, "context");
        if (context.partition().nodeIds().isEmpty()
                || !context.partition().owner().equals(CpuCapabilityProvider.CPU_BACKEND_ID)) {
            throw new IllegalArgumentException(
                    "CPU integration requires exactly one non-empty CPU partition");
        }
        return new PrepareContext<>(context.partitionDag(), context.values(),
                context.memoryRequirements(), context.constants(), inputs);
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
     * Derives immutable CPU analysis inputs for one explicit materialization policy.
     *
     * @param decision non-null optional exact Phase-1 decision retained in immutable inputs
     * @param materializationPolicy non-null immutable policy retained in the returned inputs
     * @return new non-null complete CPU analysis inputs
     */
    private CpuPartitionAnalysisInputs analysisInputs(
            Optional<CpuOpenBlasTuningDecision> decision,
            CpuPartitionAnalysisInputs.MaterializationPolicy materializationPolicy) {
        var route = qualification
                .<CpuPartitionAnalysisInputs.OpenBlasRouteConfig>map(value ->
                        CpuPartitionAnalysisInputs.OpenBlasRouteConfig.qualifiedSingleThread(
                                value, 1_000, 10, 100, 1, 1, 1, 1, 1))
                .orElse(CpuPartitionAnalysisInputs.OpenBlasRouteConfig.DISABLED);
        return new CpuPartitionAnalysisInputs(false, List.of(),
                CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT,
                materializationPolicy, false,
                CpuPartitionAnalysisInputs.PartialReductionEvidence.NONE, List.of(), route,
                CpuOpenBlasTuningBatch.HardwareIdentity.UNSPECIFIED,
                CpuOpenBlasTuningBatch.WorkloadCohort.DEFAULT, decision);
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
     * Rejects supported adapter work after this composition has closed.
     *
     * @throws IllegalStateException if this composition is closed
     */
    public void assertOpen() {
        requireOpen();
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
