package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.CpuTraceObserver;
import io.github.pho001.synaptik.backend.cpu.CpuPreparationDiagnostics;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuTraceProducer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.route.CpuLocalTuningBatch;
import io.github.pho001.synaptik.backend.cpu.internal.route.CpuLocalTuningDecision;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuConcurrencyBudget;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBorrowedBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuHostSnapshotExporter;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPreparedScheduleAssembler;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnLibrary;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnRoutePlan;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnSelector;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.prepare.PreparedScheduleAssembler;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Owns CPU preparation collaborators, optional OpenBLAS, and explicitly required oneDNN state.
 *
 * <p>One composition always describes the fixed {@code cpu/host} device and retains the portable
 * scalar, single-thread compatibility route for ordinary preparation. Explicit tuning can admit
 * declared canonical copies for caller-owned inputs without assuming that their storage is native.
 * Optional qualified OpenBLAS and explicitly configured oneDNN remain routes of this same owner.
 * The owner is safe for concurrent collaborator access and idempotent close. Leased native resources
 * keep already-admitted Runtime runs alive after this owner closes; new preparation still requires
 * this owner to remain open.</p>
 */
public final class CpuBackendComposition implements AutoCloseable {
    private static final BackendAvailabilitySnapshot AVAILABILITY =
            new BackendAvailabilitySnapshot(CpuCapabilityProvider.CPU_BACKEND_ID,
                    Map.of(new BackendDeviceId(CpuCapabilityProvider.CPU_BACKEND_ID, "host"),
                            DeviceClass.CPU));

    /** Copy admission for local trials; every copied byte remains a declared run-owned workspace. */
    private static final CpuPartitionAnalysisInputs.MaterializationPolicy LOCAL_TUNING_POLICY =
            new CpuPartitionAnalysisInputs.MaterializationPolicy(
                    true, 0, 0, 0, 0, 1, Long.MAX_VALUE, 0, 0);
    /** Topology/representation policy; local choices retain their separately authenticated policy. */
    private static final CpuPartitionAnalysisInputs.MaterializationPolicy COMPLETE_PLAN_POLICY =
            new CpuPartitionAnalysisInputs.MaterializationPolicy(
                    true, 0, 1, 3, 1, 1, Long.MAX_VALUE, 0, 0);

    private final CpuConcurrencyBudget budget;
    private final Optional<CpuOpenBlasCoordinator> coordinator;
    private final Optional<CpuOpenBlasQualification> qualification;
    private final Optional<CpuOneDnnLibrary> oneDnnLibrary;
    private final CpuOneDnnRoutePlan.Api oneDnnApi;
    private final String oneDnnPolicyFingerprint;
    private final CpuPartitionPreparer preparer;
    private final CpuPartitionFinalizer finalizer;
    private final CpuPreparedScheduleAssembler assembler;
    private final CpuTraceProducer trace;
    private final AtomicBoolean closed = new AtomicBoolean();

    /**
     * Creates one composition from already established portable and optional native state.
     *
     * @param budget the non-null shared CPU concurrency budget retained by collaborators
     * @param coordinator the non-null optional coordinator whose lifetime this owner assumes
     * @param qualification the non-null optional qualification associated with the coordinator
     * @param oneDnnLibrary non-null optional required native owner transferred to this composition
     * @param oneDnnApi non-null immutable API preference
     * @param observer caller-owned observer, or null for no diagnostic producer or overhead
     * @throws NullPointerException if an argument is {@code null}
     */
    private CpuBackendComposition(CpuConcurrencyBudget budget,
            Optional<CpuOpenBlasCoordinator> coordinator,
            Optional<CpuOpenBlasQualification> qualification,
            Optional<CpuOneDnnLibrary> oneDnnLibrary, CpuOneDnnRoutePlan.Api oneDnnApi,
            CpuTraceObserver observer) {
        this.budget = Objects.requireNonNull(budget, "budget");
        this.coordinator = Objects.requireNonNull(coordinator, "coordinator");
        this.qualification = Objects.requireNonNull(qualification, "qualification");
        this.oneDnnLibrary = Objects.requireNonNull(oneDnnLibrary, "oneDnnLibrary");
        this.oneDnnApi = Objects.requireNonNull(oneDnnApi, "oneDnnApi");
        this.trace = observer == null ? null : new CpuTraceProducer(observer,
                oneDnnLibrary.isEmpty() ? CpuPreparationDiagnostics.Api.NONE : switch (oneDnnApi) {
                    case AUTO -> CpuPreparationDiagnostics.Api.AUTO;
                    case PRIMITIVE -> CpuPreparationDiagnostics.Api.PRIMITIVES;
                    case GRAPH -> CpuPreparationDiagnostics.Api.GRAPH;
                });
        this.oneDnnPolicyFingerprint = oneDnnLibrary.map(library ->
                "onednn-policy-v2:typed-f32-literal-units:3.14.0-lp64-seq:"
                        + library.sessionIdentity() + ":" + oneDnnApi.name())
                .orElse("onednn-policy-v2:disabled");
        this.preparer = new CpuPartitionPreparer(new CpuPartitionLowering(),
                oneDnnLibrary.map(library -> new CpuOneDnnSelector(library, oneDnnApi, trace)));
        this.finalizer = new CpuPartitionFinalizer(Optional.empty(), Optional.empty(), budget,
                coordinator, oneDnnLibrary);
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
        return openAutomatic(Optional.empty(), CpuOneDnnRoutePlan.Api.AUTO, null);
    }

    /**
     * Opens the same automatic composition with passive cold CPU diagnostics.
     * @param observer non-null caller-owned thread-safe observer; never closed; callback failures
     *     disable diagnostics without affecting backend work, as specified by {@link CpuTraceObserver}
     * @return new open composition retaining the observer for its lifetime
     */
    public static CpuBackendComposition openAutomatic(CpuTraceObserver observer) {
        return openAutomatic(Optional.empty(), CpuOneDnnRoutePlan.Api.AUTO,
                Objects.requireNonNull(observer, "observer"));
    }

    /**
     * Opens an explicitly required exact oneDNN 3.14.0 LP64 SEQ library.
     * Optional OpenBLAS discovery is independent and cannot mask a required oneDNN failure.
     *
     * @param absolutePath non-null absolute library path; discovery is never used for oneDNN
     * @param api non-null frozen native API preference
     * @return new composition owning the exact library and optional qualified OpenBLAS provider
     * @throws NullPointerException if an argument is null
     * @throws IllegalArgumentException if the path, ABI, version, or threading runtime is invalid
     * @throws RuntimeException if required native loading or composition construction fails
     * @throws Error if native loading, construction, or cleanup reports an error
     */
    public static CpuBackendComposition openOneDnn(Path absolutePath, CpuOneDnnRoutePlan.Api api) {
        return openRequired(absolutePath, api, null);
    }

    /**
     * Opens required oneDNN with the same optional diagnostic semantics as automatic composition.
     * @param absolutePath non-null absolute exact-library path
     * @param api non-null immutable API preference, not forced native selection
     * @param observer non-null caller-owned thread-safe sink; never closed; observer failures
     *     disable diagnostics without replacing native or preparation failures
     * @return new composition owning required native state and borrowing the observer
     */
    public static CpuBackendComposition openOneDnn(Path absolutePath, CpuOneDnnRoutePlan.Api api,
            CpuTraceObserver observer) {
        return openRequired(absolutePath, api, Objects.requireNonNull(observer, "observer"));
    }

    private static CpuBackendComposition openRequired(Path absolutePath, CpuOneDnnRoutePlan.Api api,
            CpuTraceObserver observer) {
        Objects.requireNonNull(api, "api");
        CpuOneDnnLibrary library = CpuOneDnnLibrary.open(absolutePath);
        try {
            return openAutomatic(Optional.of(library), api, observer);
        } catch (RuntimeException | Error failure) {
            try { library.close(); } catch (RuntimeException | Error cleanup) {
                if (cleanup != failure) failure.addSuppressed(cleanup);
            }
            throw failure;
        }
    }

    private static CpuBackendComposition openAutomatic(Optional<CpuOneDnnLibrary> library,
            CpuOneDnnRoutePlan.Api api, CpuTraceObserver observer) {
        CpuOpenBlasDiscoverySession session;
        try {
            session = CpuOpenBlasDiscovery.discover(CpuOpenBlasDiscoveryRequest.automatic());
        } catch (RuntimeException optionalFailure) {
            return create(new CpuConcurrencyBudget(1), Optional.empty(), Optional.empty(),
                    library, api, observer);
        }
        return open(session, library, api, observer);
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
        return open(Objects.requireNonNull(session, "session"), Optional.empty(),
                CpuOneDnnRoutePlan.Api.AUTO, null);
    }

    private static CpuBackendComposition open(CpuOpenBlasDiscoverySession session,
            Optional<CpuOneDnnLibrary> library, CpuOneDnnRoutePlan.Api api, CpuTraceObserver observer) {
        CpuConcurrencyBudget budget;
        try {
            budget = new CpuConcurrencyBudget(1);
        } catch (RuntimeException | Error failure) {
            closeAfterFailure(null, session, failure);
            throw failure;
        }
        CpuOpenBlasCoordinator coordinator = null;
        CpuOpenBlasQualification qualification = null;
        try {
            if (session.invocation().isPresent()) {
                coordinator = session.transferToCoordinator(budget);
                qualification = CpuOpenBlasQualifier.qualify(session.result(), coordinator);
            }
            session.close();
        } catch (RuntimeException optionalFailure) {
            closeAfterFailure(coordinator, session, optionalFailure);
            return create(budget, Optional.empty(), Optional.empty(), library, api, observer);
        } catch (Error fatalFailure) {
            closeAfterFailure(coordinator, session, fatalFailure);
            throw fatalFailure;
        }
        return create(budget, Optional.ofNullable(coordinator), Optional.ofNullable(qualification),
                library, api, observer);
    }

    private static CpuBackendComposition create(CpuConcurrencyBudget budget,
            Optional<CpuOpenBlasCoordinator> coordinator,
            Optional<CpuOpenBlasQualification> qualification, Optional<CpuOneDnnLibrary> library,
            CpuOneDnnRoutePlan.Api api, CpuTraceObserver observer) {
        try {
            return new CpuBackendComposition(budget, coordinator, qualification, library, api, observer);
        } catch (RuntimeException | Error failure) {
            coordinator.ifPresent(value -> {
                try { value.close(); } catch (RuntimeException | Error cleanup) {
                    if (cleanup != failure) failure.addSuppressed(cleanup);
                }
            });
            throw failure;
        }
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
        return partitionPreparation(analysisInputs(Optional.empty(),
                CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED), preparer);
    }

    private PartitionPreparation<?, ?> partitionPreparation(CpuPartitionAnalysisInputs inputs,
            BackendPartitionPreparer<CpuPartitionAnalysisInputs, CpuPartitionPreparationPlan> analysis) {
        return trace == null ? new PartitionPreparation<>(inputs, analysis, finalizer)
                : trace.preparation(inputs, analysis, finalizer);
    }

    /**
     * Performs fresh CPU analysis and returns the current complete tunable batch, if any.
     * The explicit local policy admits canonical input copies with exact resource declarations;
     * it never upgrades borrowed caller storage to a native-storage fact.
     *
     * @param context non-null exact stable partition projection
     * @return the freshly analyzed immutable batch, or empty when the valid workload is not
     *     currently tunable
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalStateException if this composition is closed
     * @throws IllegalArgumentException if the projection is outside the supported domain
     */
    public Optional<CpuLocalTuningBatch> tuningBatch(PrepareContext<?> context) {
        requireOpen();
        return preparer.localTuningBatch(cpuContext(context,
                analysisInputs(Optional.empty(), LOCAL_TUNING_POLICY)));
    }

    /**
     * Freshly enumerates complete CPU plan identities under one profile-free route policy.
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
            PrepareContext<?> context, Optional<CpuLocalTuningDecision> phaseOneDecision) {
        requireOpen();
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(phaseOneDecision, "phaseOneDecision");
        validatePhaseOne(context, phaseOneDecision);
        CpuPartitionAnalysisInputs inputs = analysisInputs(phaseOneDecision, COMPLETE_PLAN_POLICY);
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
     * <p>The selected-plan association carries the absence proof established during complete
     * discovery. Repreparing it revalidates its exact source/configuration fingerprint, rather than
     * rerunning local discovery. A present local decision is revalidated only as its frozen
     * realization; its unselected route peers are never queried here.</p>
     *
     * @param context non-null exact stable one-partition projection; inspected but not mutated
     * @param phaseOneDecision non-null optional exact authenticated local selection, or empty for
     *     absence already authenticated by the selected-plan association; the decision is borrowed
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
            Optional<CpuLocalTuningDecision> phaseOneDecision,
            CpuPartitionPreparer.SelectedCompletePlan selectedPlan) {
        requireOpen();
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(phaseOneDecision, "phaseOneDecision");
        Objects.requireNonNull(selectedPlan, "selectedPlan");
        if (selectedPlan.schemaVersion() != 5) {
            throw new IllegalArgumentException("CPU complete-plan schema is unsupported");
        }
        if (phaseOneDecision.isPresent()) validatePhaseOne(context, phaseOneDecision);
        CpuPartitionAnalysisInputs inputs = analysisInputs(phaseOneDecision, COMPLETE_PLAN_POLICY);
        var selectedPreparer = new io.github.pho001.synaptik.prepare.analysis.BackendPartitionPreparer<
                CpuPartitionAnalysisInputs,
                io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan>() {
            @Override public BackendPartitionAnalysis<io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan>
                    analyze(PrepareContext<CpuPartitionAnalysisInputs> context) {
                return preparer.analyzeSelected(context, selectedPlan);
            }
        };
        return partitionPreparation(inputs, selectedPreparer);
    }

    /**
     * Revalidates an exact frozen realization without enumerating its local peers again.
     * Absence must still be established against the current bounded discovery domain.
     *
     * @param context non-null current stable partition projection, borrowed
     * @param decision non-null optional immutable local selection, borrowed
     * @throws IllegalArgumentException if eligibility or exact realization compatibility changed
     */
    private void validatePhaseOne(PrepareContext<?> context,
            Optional<CpuLocalTuningDecision> decision) {
        if (decision.isEmpty()) {
            if (preparer.localTuningBatch(cpuContext(context,
                    analysisInputs(Optional.empty(), LOCAL_TUNING_POLICY))).isPresent()) {
                throw new IllegalArgumentException("CPU Phase-1 eligibility changed");
            }
            return;
        }
        var selected = analyze(cpuContext(context, analysisInputs(decision, LOCAL_TUNING_POLICY)));
        var batch = selected.plan().localTuningBatch().orElseThrow(() ->
                new IllegalArgumentException("CPU Phase-1 decision is no longer eligible"));
        var matched = decision.orElseThrow().match(batch).orElseThrow(() ->
                new IllegalArgumentException("CPU Phase-1 decision is stale or incompatible"));
        if (!selected.plan().selectedLocalTuningCandidate().orElseThrow()
                .equals(matched.identity())) {
            throw new IllegalArgumentException("CPU Phase-1 decision was not selected");
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
            CpuLocalTuningDecision decision) {
        requireOpen();
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(decision, "decision");
        BackendPartitionAnalysis<io.github.pho001.synaptik.backend.cpu.internal.prepare
                .CpuPartitionPreparationPlan> analysis = analyze(cpuContext(context,
                        analysisInputs(Optional.of(decision), LOCAL_TUNING_POLICY)));
        var batch = analysis.plan().localTuningBatch().orElseThrow(() ->
                new IllegalArgumentException("CPU tuning decision is no longer eligible"));
        var matched = decision.match(batch).orElseThrow(() ->
                new IllegalArgumentException("CPU tuning decision is stale or incompatible"));
        if (!analysis.plan().selectedLocalTuningCandidate().orElseThrow()
                .equals(matched.identity())) {
            throw new IllegalArgumentException("CPU tuning decision was not selected");
        }
        return partitionPreparation(analysisInputs(Optional.of(decision),
                LOCAL_TUNING_POLICY), preparer);
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
        return new PrepareContext<>(context.partitionDag(), context.values(), context.memoryRequirements(),
                context.constants(), inputs, context.scanActivityPlans());
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
            Optional<CpuLocalTuningDecision> decision,
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
                CpuLocalTuningBatch.HardwareIdentity.UNSPECIFIED,
                CpuLocalTuningBatch.WorkloadCohort.DEFAULT, decision);
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
     * Returns the immutable complete-plan identity of this owner's oneDNN admission policy.
     *
     * @return stable disabled identity, or the exact library session, API and qualified policy
     * @throws IllegalStateException if this composition is closed
     */
    public String oneDnnPolicyFingerprint() {
        requireOpen();
        return oneDnnPolicyFingerprint;
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
     * Publishes closed state, then attempts every owned provider close once in reverse order.
     * Native owner close preserves already-acquired kernel and run-workspace leases.
     *
     * @throws RuntimeException if provider thread restoration or cleanup reports an unchecked
     *     failure
     * @throws Error if provider thread restoration or cleanup reports an error
     */
    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) return;
        Throwable failure = null;
        try {
            coordinator.ifPresent(CpuOpenBlasCoordinator::close);
        } catch (RuntimeException | Error cleanup) {
            failure = cleanup;
        }
        try {
            oneDnnLibrary.ifPresent(CpuOneDnnLibrary::close);
        } catch (RuntimeException | Error cleanup) {
            if (failure == null) failure = cleanup;
            else if (cleanup != failure) failure.addSuppressed(cleanup);
        }
        if (failure instanceof RuntimeException unchecked) throw unchecked;
        if (failure instanceof Error fatal) throw fatal;
    }
}
