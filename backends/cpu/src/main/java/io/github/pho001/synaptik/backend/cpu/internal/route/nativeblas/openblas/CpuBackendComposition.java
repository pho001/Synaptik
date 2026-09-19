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
import io.github.pho001.synaptik.compiler.CompileConstantPlan;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.prepare.ProducerlessPublishedConstantResource;
import io.github.pho001.synaptik.prepare.PreparedScheduleAssembler;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.prepare.analysis.PartitionDag;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
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
        validateSoleCpuPartition(artifacts);
        return preparationsForValidatedArtifacts(artifacts);
    }

    /**
     * Builds the complete reusable CPU execution recipe, including exact physical declarations
     * for canonical source-only published constants.
     *
     * <p>The returned recipe owns no physical constant representation. Each fresh Runtime run
     * state invokes its retained initializer recipe once and owns the resulting distinct
     * representation. Source-only constants add no executable schedule occurrence, and this
     * operation continues to reject pure zero-node graphs.</p>
     *
     * @param artifacts non-null compile artifacts containing exactly one non-empty CPU partition
     * @return non-null immutable reusable execution recipe with no run-owned physical resource
     * @throws NullPointerException if {@code artifacts} is {@code null}
     * @throws IllegalStateException if this composition is closed
     * @throws IllegalArgumentException if the CPU partition domain or a source-only constant's
     *     role, descriptor, scalar type, or geometry is unsupported or inconsistent
     * @throws ArithmeticException if canonical layout or byte-size arithmetic overflows
     */
    public PreparedExecution prepare(CompileArtifacts artifacts) {
        requireOpen();
        Objects.requireNonNull(artifacts, "artifacts");
        validateSoleCpuPartition(artifacts);
        List<PartitionPreparation<?, ?>> preparations =
                preparationsForValidatedArtifacts(artifacts);
        List<ProducerlessPublishedConstantResource> resources =
                producerlessPublishedConstantResources(artifacts);
        return GraphPreparation.prepare(artifacts, preparations, resources, assembler);
    }

    /**
     * Performs fresh CPU analysis and returns the current complete tunable batch, if any.
     *
     * @param artifacts non-null artifacts in the supported sole-CPU-partition domain
     * @return the freshly analyzed immutable batch, or empty when the valid workload is not
     *     currently tunable
     * @throws NullPointerException if {@code artifacts} is {@code null}
     * @throws IllegalStateException if this composition is closed
     * @throws IllegalArgumentException if the artifacts are outside the supported domain
     */
    public Optional<CpuOpenBlasTuningBatch> tuningBatch(CompileArtifacts artifacts) {
        requireOpen();
        Objects.requireNonNull(artifacts, "artifacts");
        validateSoleCpuPartition(artifacts);
        return analyze(artifacts, Optional.empty()).plan().openBlasTuningBatch();
    }

    /**
     * Freshly enumerates the complete retained CPU plan identities under the fixed profile.
     *
     * <p>The method revalidates Phase-1 state and performs analysis only. It neither prepares nor
     * executes a recipe and does not time or rank alternatives.</p>
     *
     * @param artifacts non-null exact supported one-partition artifacts; inspected but not
     *     mutated
     * @param phaseOneDecision non-null optional exact authenticated local selection, or empty for
     *     freshly proved absence; the decision is borrowed
     * @return a new immutable complete-plan identity snapshot and completeness outcome
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if artifacts or Phase-1 state are invalid
     * @throws ArithmeticException if exact candidate or resource arithmetic overflows
     * @throws IllegalStateException if this composition is closed
     */
    public CpuPartitionPreparer.CompletePlanCandidates completePlanCandidates(
            CompileArtifacts artifacts, Optional<CpuOpenBlasTuningDecision> phaseOneDecision) {
        requireOpen();
        Objects.requireNonNull(artifacts, "artifacts");
        Objects.requireNonNull(phaseOneDecision, "phaseOneDecision");
        validateSoleCpuPartition(artifacts);
        validatePhaseOne(artifacts, phaseOneDecision);
        CpuPartitionAnalysisInputs inputs = completePlanAnalysisInputs(artifacts,
                phaseOneDecision);
        return preparer.completePlanCandidates(context(artifacts, inputs));
    }

    /**
     * Freshly prepares one exact retained complete plan without heuristic substitution.
     *
     * <p>Preparation repeats authoritative analysis, shared assignment, CPU finalization, and
     * schedule validation. It creates no {@code RunState}, binds no representative input, and
     * executes or times no trial.</p>
     *
     * @param artifacts non-null exact supported one-partition artifacts; inspected but not
     *     mutated
     * @param phaseOneDecision non-null optional exact authenticated local selection, or empty for
     *     freshly proved absence; the decision is borrowed
     * @param selectedPlan non-null immutable retained complete-plan identity and association
     *     fingerprint; inspected but not mutated
     * @return a fresh complete immutable prepared recipe that borrows this composition's lifetime
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if any fresh compatibility or selection check fails
     * @throws ArithmeticException if exact preparation geometry overflows
     * @throws IllegalStateException if this composition is closed
     */
    public PreparedExecution prepareCompletePlan(CompileArtifacts artifacts,
            Optional<CpuOpenBlasTuningDecision> phaseOneDecision,
            CpuPartitionPreparer.SelectedCompletePlan selectedPlan) {
        requireOpen();
        Objects.requireNonNull(artifacts, "artifacts");
        Objects.requireNonNull(phaseOneDecision, "phaseOneDecision");
        Objects.requireNonNull(selectedPlan, "selectedPlan");
        if (selectedPlan.schemaVersion() != 1) {
            throw new IllegalArgumentException("CPU complete-plan schema is unsupported");
        }
        validateSoleCpuPartition(artifacts);
        validatePhaseOne(artifacts, phaseOneDecision);
        CpuPartitionAnalysisInputs inputs = completePlanAnalysisInputs(artifacts,
                phaseOneDecision);
        var selectedPreparer = new io.github.pho001.synaptik.prepare.analysis.BackendPartitionPreparer<
                CpuPartitionAnalysisInputs,
                io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan>() {
            @Override public BackendPartitionAnalysis<io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan>
                    analyze(PrepareContext<CpuPartitionAnalysisInputs> context) {
                return preparer.analyzeSelected(context, selectedPlan);
            }
        };
        var preparation = new PartitionPreparation<>(inputs, selectedPreparer, finalizer);
        return GraphPreparation.prepare(artifacts, List.of(preparation),
                producerlessPublishedConstantResources(artifacts), assembler);
    }

    private void validatePhaseOne(CompileArtifacts artifacts,
            Optional<CpuOpenBlasTuningDecision> decision) {
        Optional<CpuOpenBlasTuningBatch> fresh = analyze(artifacts, Optional.empty())
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
                    selected = analyze(artifacts, decision);
            if (!selected.plan().selectedOpenBlasTuningCandidate().orElseThrow()
                    .equals(matched.identity())) {
                throw new IllegalArgumentException("CPU Phase-1 decision was not selected");
            }
        }
    }

    /**
     * Re-analyzes and prepares one exact CPU-owned tuning decision without heuristic fallback.
     *
     * @param artifacts non-null exact artifacts retained by the supported tuning association
     * @param decision non-null CPU decision to validate against fresh authoritative analysis
     * @return a complete immutable prepared execution for exactly the selected candidate
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalStateException if this composition is closed
     * @throws IllegalArgumentException if the decision is stale, ineligible, or not selected by
     *     the fresh batch
     */
    public PreparedExecution prepareSelected(CompileArtifacts artifacts,
            CpuOpenBlasTuningDecision decision) {
        requireOpen();
        Objects.requireNonNull(artifacts, "artifacts");
        Objects.requireNonNull(decision, "decision");
        validateSoleCpuPartition(artifacts);
        BackendPartitionAnalysis<io.github.pho001.synaptik.backend.cpu.internal.prepare
                .CpuPartitionPreparationPlan> analysis = analyze(artifacts, Optional.of(decision));
        var batch = analysis.plan().openBlasTuningBatch().orElseThrow(() ->
                new IllegalArgumentException("CPU tuning decision is no longer eligible"));
        var matched = decision.match(batch).orElseThrow(() ->
                new IllegalArgumentException("CPU tuning decision is stale or incompatible"));
        if (!analysis.plan().selectedOpenBlasTuningCandidate().orElseThrow()
                .equals(matched.identity())) {
            throw new IllegalArgumentException("CPU tuning decision was not selected");
        }
        var preparation = new PartitionPreparation<>(analysisInputs(artifacts,
                Optional.of(decision)), preparer, finalizer);
        return GraphPreparation.prepare(artifacts, List.of(preparation),
                producerlessPublishedConstantResources(artifacts), assembler);
    }

    /**
     * Reconstructs shared Prepare's authoritative partition projection and runs CPU analysis.
     *
     * @param artifacts non-null already validated sole-partition artifacts
     * @param decision non-null optional exact decision to validate during analysis
     * @return the fresh non-null immutable CPU analysis
     */
    private BackendPartitionAnalysis<io.github.pho001.synaptik.backend.cpu.internal.prepare
            .CpuPartitionPreparationPlan> analyze(CompileArtifacts artifacts,
            Optional<CpuOpenBlasTuningDecision> decision) {
        CpuPartitionAnalysisInputs inputs = analysisInputs(artifacts, decision);
        return preparer.analyze(context(artifacts, inputs));
    }

    /**
     * Reconstructs the exact partition-local shared Prepare projection for supplied CPU inputs.
     *
     * @param artifacts non-null already validated sole-partition artifacts; inspected but not
     *     mutated
     * @param inputs non-null immutable CPU analysis inputs retained by the returned context
     * @return a new non-null immutable partition-local preparation context
     * @throws NullPointerException if a required projected fact is absent
     * @throws IllegalArgumentException if projected facts violate shared Prepare invariants
     */
    private PrepareContext<CpuPartitionAnalysisInputs> context(CompileArtifacts artifacts,
            CpuPartitionAnalysisInputs inputs) {
        var partition = artifacts.partitions().getFirst();
        var nodesById = new HashMap<NodeId, CompiledNode>();
        artifacts.graph().nodes().forEach(node -> nodesById.put(node.id(), node));
        var requirementsById = new HashMap<ValueId, LogicalMemoryRequirement>();
        artifacts.memory().requirements().forEach(value ->
                requirementsById.put(value.valueId(), value));
        var nodes = partition.nodeIds().stream().map(nodesById::get).toList();
        var projectedIds = new HashSet<ValueId>();
        nodes.forEach(node -> {
            projectedIds.addAll(node.inputs());
            projectedIds.addAll(node.outputs());
        });
        var values = artifacts.graph().values().stream()
                .filter(value -> projectedIds.contains(value.id())).toList();
        var requirements = values.stream().map(value -> requirementsById.get(value.id())).toList();
        var constants = new LinkedHashMap<ValueId, ScalarValue>();
        artifacts.constants().constantSources().stream()
                .filter(source -> projectedIds.contains(source.valueId()))
                .forEach(source -> constants.put(source.valueId(), source.value()));
        var context = new PrepareContext<>(new PartitionDag(partition, nodes), values,
                requirements, constants, inputs);
        return context;
    }

    /**
     * Validates the complete-schedule domain before any CPU fact derivation or backend analysis.
     *
     * @param artifacts non-null compile artifacts already admitted by the caller
     * @throws IllegalArgumentException if partition coverage is not exactly one non-empty
     *     CPU-owned maximal partition
     */
    private static void validateSoleCpuPartition(CompileArtifacts artifacts) {
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
    }

    /**
     * Constructs the existing positional CPU preparation after complete-domain validation.
     *
     * @param artifacts non-null artifacts already accepted by
     *     {@link #validateSoleCpuPartition(CompileArtifacts)}
     * @return a non-null immutable singleton preparation retaining current CPU analysis order
     * @throws IllegalArgumentException if CPU boundary analysis rejects incomplete facts
     */
    private List<PartitionPreparation<?, ?>> preparationsForValidatedArtifacts(
            CompileArtifacts artifacts) {
        CpuPartitionAnalysisInputs inputs = analysisInputs(artifacts);
        PartitionPreparation<CpuPartitionAnalysisInputs,
                io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan>
                preparation = new PartitionPreparation<>(inputs, preparer, finalizer);
        return List.of(preparation);
    }

    /**
     * Derives exact physical contributions for source-only published splats in graph-value order.
     *
     * @param artifacts non-null artifacts with an already validated executable CPU domain
     * @return a non-null immutable ordered list retaining exact graph-value and logical-requirement
     *     references; ordinary consumed or unpublished constants are excluded
     * @throws IllegalArgumentException if source, publication, node-use, logical-memory, Shape,
     *     layout, or scalar-type facts are contradictory or unsupported
     * @throws ArithmeticException if canonical layout or checked byte-size arithmetic overflows
     */
    private static List<ProducerlessPublishedConstantResource>
            producerlessPublishedConstantResources(CompileArtifacts artifacts) {
        var inputs = new HashSet<>(artifacts.graph().inputs());
        var publications = new HashSet<ValueId>();
        artifacts.publication().forwardBindings()
                .forEach(binding -> publications.add(binding.valueId()));
        artifacts.publication().gradientBindings()
                .forEach(binding -> publications.add(binding.valueId()));

        var produced = new HashSet<ValueId>();
        var consumed = new HashSet<ValueId>();
        artifacts.graph().nodes().forEach(node -> {
            produced.addAll(node.outputs());
            consumed.addAll(node.inputs());
        });

        var sources = new LinkedHashMap<ValueId, CompileConstantPlan.ConstantSource>();
        for (CompileConstantPlan.ConstantSource source : artifacts.constants().constantSources()) {
            if (sources.putIfAbsent(source.valueId(), source) != null) {
                throw new IllegalArgumentException(
                        "CPU constant sources duplicate " + source.valueId());
            }
        }
        var requirements = new HashMap<ValueId, LogicalMemoryRequirement>();
        for (LogicalMemoryRequirement requirement : artifacts.memory().requirements()) {
            if (requirements.putIfAbsent(requirement.valueId(), requirement) != null) {
                throw new IllegalArgumentException(
                        "CPU logical memory requirements duplicate " + requirement.valueId());
            }
        }

        var resources = new ArrayList<ProducerlessPublishedConstantResource>();
        for (var value : artifacts.graph().values()) {
            CompileConstantPlan.ConstantSource source = sources.get(value.id());
            if (source == null || !inputs.contains(value.id()) || !publications.contains(value.id())
                    || consumed.contains(value.id()) || produced.contains(value.id())) {
                continue;
            }
            LogicalMemoryRequirement requirement = requirements.get(value.id());
            if (requirement == null
                    || !requirement.descriptor().equals(value.descriptor())
                    || requirement.producerPartition().isPresent()
                    || !requirement.consumerPartitions().isEmpty()
                    || !requirement.graphOutput()) {
                throw new IllegalArgumentException(
                        "CPU source-only published constant has contradictory logical memory: "
                                + value.id());
            }
            var descriptor = value.descriptor();
            var shape = descriptor.shape();
            if (!shape.isFullyStatic()) {
                throw new IllegalArgumentException(
                        "CPU source-only published constant has dynamic shape: " + value.id());
            }
            if (descriptor.layout().isEmpty()) {
                throw new IllegalArgumentException(
                        "CPU source-only published constant has unresolved layout: " + value.id());
            }
            LayoutDescriptor layout = descriptor.layout().orElseThrow();
            if (!layout.equals(LayoutDescriptor.contiguous(shape))) {
                throw new IllegalArgumentException(
                        "CPU source-only published constant has non-canonical layout: " + value.id());
            }
            if (source.value().dataType() != descriptor.dataType()) {
                throw new IllegalArgumentException(
                        "CPU source-only published constant scalar type disagrees with descriptor: "
                                + value.id());
            }
            long byteSize = Math.multiplyExact(
                    layout.referencedElementSpan(), descriptor.dataType().byteWidth());
            resources.add(new ProducerlessPublishedConstantResource(
                    value, requirement, byteSize, descriptor.dataType().byteWidth()));
        }
        return List.copyOf(resources);
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
        return analysisInputs(artifacts, Optional.empty());
    }

    /**
     * Derives current immutable production analysis facts with an optional exact selection.
     *
     * @param artifacts non-null already validated artifacts; inspected but not mutated
     * @param decision non-null optional decision retained only in immutable analysis input
     * @return non-null complete CPU analysis inputs
     */
    private CpuPartitionAnalysisInputs analysisInputs(CompileArtifacts artifacts,
            Optional<CpuOpenBlasTuningDecision> decision) {
        return analysisInputs(artifacts, decision,
                CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED);
    }

    /**
     * Derives the fixed complete-plan enumeration profile without changing ordinary preparation.
     *
     * @param artifacts non-null already validated artifacts; inspected but not mutated
     * @param decision non-null optional exact Phase-1 decision retained in immutable inputs
     * @return new complete CPU analysis inputs with candidate-only representation enumeration
     */
    private CpuPartitionAnalysisInputs completePlanAnalysisInputs(CompileArtifacts artifacts,
            Optional<CpuOpenBlasTuningDecision> decision) {
        return analysisInputs(artifacts, decision,
                new CpuPartitionAnalysisInputs.MaterializationPolicy(true, 0, 1, 3, 1,
                        1, Long.MAX_VALUE, 0, 0));
    }

    /**
     * Derives immutable CPU analysis inputs for one explicit materialization policy.
     *
     * @param artifacts non-null already validated artifacts; inspected but not mutated
     * @param decision non-null optional exact Phase-1 decision retained in immutable inputs
     * @param materializationPolicy non-null immutable policy retained in the returned inputs
     * @return new non-null complete CPU analysis inputs
     * @throws IllegalArgumentException if an eligible MATMUL boundary lacks a graph node or
     *     resolved descriptor
     */
    private CpuPartitionAnalysisInputs analysisInputs(CompileArtifacts artifacts,
            Optional<CpuOpenBlasTuningDecision> decision,
            CpuPartitionAnalysisInputs.MaterializationPolicy materializationPolicy) {
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
                materializationPolicy, false,
                CpuPartitionAnalysisInputs.PartialReductionEvidence.NONE, storageFacts, route,
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
