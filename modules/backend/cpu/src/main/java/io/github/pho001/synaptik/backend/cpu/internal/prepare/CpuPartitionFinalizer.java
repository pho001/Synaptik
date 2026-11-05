package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStore;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGenerator;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutable;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedPartitionExecutable;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuConcurrencyBudget;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuWorkerGroup;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasInvocation;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasCoordinator;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasPreparedExecutable;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnCompiler;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnLibrary;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnPreparedExecutable;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnRoutePlan;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnOutputNarrowing;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalization;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalizationResult;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalizer;
import io.github.pho001.synaptik.prepare.PreparationResourceAssignment;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.resource.PreparedResource;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.LinkedHashMap;

/**
 * Post-assignment verification, artifact realization, and partition executable finalization.
 *
 * <p>Every deduplicated partition buffer assignment and every unit-local exact workspace
 * assignment is resolved and checked before the first artifact-store call. Portable finalization
 * then realizes each generated unit's selected artifact or constructs a direct recurrent child,
 * without an artifact-store call for that child. An OpenBLAS route requires
 * the exact live coordinator that issued its retained qualification, validates that session and
 * target plus the shared budget snapshot, and installs its selected count before realizing any
 * selected
 * affine-copy artifact. Finalization cannot change unit
 * topology, dependencies, materialization, generated carrier patterns, route, strategy,
 * specialization, or declaration geometry. A multi-unit result is wrapped in one CPU-private
 * atomic sequential composite. A single recurrent unit also uses that composite. A single
 * constant-write unit uses the same ValueId-based child selection: mandatory original constant
 * sources remain partition declarations but never become arguments to its output-only kernel.
 * Other established generated one-unit results without materialization remain direct child recipes.
 * When the exact
 * analyzed plan contains an explicitly chosen representation candidate, finalization also
 * realizes its one or two generated affine-copy artifacts before consumer artifacts. It does not
 * select among retained candidates. For a selected direct loss unit, finalization passes only its
 * already-proved cold loss geometry to the prepared recipe; it does not interpret targets or add
 * the separate future pre-write index-validation lifecycle.</p>
 *
 * <p>Selected oneDNN units reuse the same composite topology, even for one unit. After every
 * assignment is validated, finalization authenticates the retained library session and compiles
 * only the exact selected API and descriptor. Scratch size and implementation must match analysis.
 * Native computation never enters the generated artifact store. A low logical output instead
 * retains its separately declared F32 staging span and realizes one typed generated final CAST;
 * the logical buffer is checked at its own byte width, not the staging width. Unchanged F32 units
 * acquire no narrowing artifact. Acquired kernels transfer to Runtime in acquisition order; any
 * failure rolls back that call's resources in reverse order, with no route retry, reranking, or
 * portable fallback.</p>
 */
public final class CpuPartitionFinalizer implements BackendPartitionFinalizer<CpuPartitionPreparationPlan> {
    private final CpuGeneratedKernelArtifactStore artifactStore;
    private final CpuClassFileKernelGenerator partialReductionGenerator =
            new CpuClassFileKernelGenerator();
    private final Optional<CpuWorkerGroup> workerGroup;
    private final Optional<CpuOpenBlasInvocation> openBlasInvocation;
    private final Optional<CpuConcurrencyBudget> concurrencyBudget;
    private final Optional<CpuOpenBlasCoordinator> openBlasCoordinator;
    private final Optional<CpuOneDnnLibrary> oneDnnLibrary;

    /** Creates the default in-memory-only finalizer for single-thread plans. */
    public CpuPartitionFinalizer() { this(Optional.empty(), Optional.empty(), Optional.empty()); }
    /**
     * Creates a finalizer with optional trusted class-byte persistence.
     * @param trustedArtifactRoot non-null optional trusted local root; empty selects in-memory-only
     *     realization
     * @throws NullPointerException if {@code trustedArtifactRoot} is {@code null}
     */
    public CpuPartitionFinalizer(Optional<Path> trustedArtifactRoot) {
        this(trustedArtifactRoot, Optional.empty(), Optional.empty());
    }

    /**
     * Creates a finalizer with optional persistence and an explicitly borrowed worker group.
     * The finalizer verifies group openness and capacity for a parallel plan but never closes it.
     *
     * @param trustedArtifactRoot non-null optional trusted local root; empty selects
     *     in-memory-only realization
     * @param workerGroup non-null optional caller-owned group borrowed by parallel executables
     * @throws NullPointerException if either optional reference is {@code null}
     */
    public CpuPartitionFinalizer(Optional<Path> trustedArtifactRoot,
            Optional<CpuWorkerGroup> workerGroup) {
        this(trustedArtifactRoot, workerGroup, Optional.empty());
    }

    /**
     * Creates a finalizer with optional persistence and explicitly borrowed execution resources.
     * Neither resource is configured, restored, or closed. A borrowed invocation alone cannot
     * finalize a qualified OpenBLAS route; that route requires the coordinated constructor.
     *
     * @param trustedArtifactRoot non-null optional trusted local artifact root
     * @param workerGroup non-null optional caller-owned CPU worker group
     * @param openBlasInvocation non-null optional caller-owned OpenBLAS invocation retained only
     *     for compatibility tests; it cannot authorize qualified route finalization
     * @throws NullPointerException if an optional reference is {@code null}
     */
    public CpuPartitionFinalizer(Optional<Path> trustedArtifactRoot,
            Optional<CpuWorkerGroup> workerGroup,
            Optional<CpuOpenBlasInvocation> openBlasInvocation) {
        artifactStore = new CpuGeneratedKernelArtifactStore(trustedArtifactRoot);
        this.workerGroup = Objects.requireNonNull(workerGroup, "workerGroup");
        this.openBlasInvocation = Objects.requireNonNull(openBlasInvocation,
                "openBlasInvocation");
        concurrencyBudget = Optional.empty();
        openBlasCoordinator = Optional.empty();
        oneDnnLibrary = Optional.empty();
    }

    /**
     * Creates an explicitly coordinated finalizer borrowing one exact shared CPU budget.
     *
     * @param trustedArtifactRoot non-null optional trusted artifact root
     * @param workerGroup non-null optional worker group; when present it must borrow {@code budget}
     * @param budget non-null shared budget matching analysis capacity
     * @param coordinator non-null optional OpenBLAS coordinator borrowing {@code budget}
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if a supplied worker group or coordinator does not borrow
     *     the exact same budget object
     */
    public CpuPartitionFinalizer(Optional<Path> trustedArtifactRoot,
            Optional<CpuWorkerGroup> workerGroup, CpuConcurrencyBudget budget,
            Optional<CpuOpenBlasCoordinator> coordinator) {
        this(trustedArtifactRoot, workerGroup, budget, coordinator, Optional.empty());
    }

    /**
     * Creates a finalizer borrowing exact coordinated CPU and optional oneDNN owners.
     *
     * @param trustedArtifactRoot non-null optional trusted artifact root
     * @param workerGroup non-null optional group borrowing the exact {@code budget}
     * @param budget non-null shared budget matching analysis capacity
     * @param coordinator non-null optional OpenBLAS coordinator borrowing {@code budget}
     * @param oneDnnLibrary non-null optional exact oneDNN owner; not closed by this finalizer
     * @throws NullPointerException if a required reference is null
     * @throws IllegalArgumentException if execution resources do not share the exact budget
     */
    public CpuPartitionFinalizer(Optional<Path> trustedArtifactRoot,
            Optional<CpuWorkerGroup> workerGroup, CpuConcurrencyBudget budget,
            Optional<CpuOpenBlasCoordinator> coordinator,
            Optional<CpuOneDnnLibrary> oneDnnLibrary) {
        artifactStore = new CpuGeneratedKernelArtifactStore(trustedArtifactRoot);
        this.workerGroup = Objects.requireNonNull(workerGroup, "workerGroup");
        this.concurrencyBudget = Optional.of(Objects.requireNonNull(budget, "budget"));
        this.openBlasCoordinator = Objects.requireNonNull(coordinator, "coordinator");
        this.openBlasInvocation = Optional.empty();
        this.oneDnnLibrary = Objects.requireNonNull(oneDnnLibrary, "oneDnnLibrary");
        if (workerGroup.isPresent() && workerGroup.orElseThrow().concurrencyBudget() != budget) {
            throw new IllegalArgumentException("CPU worker group must share finalizer budget");
        }
        if (coordinator.isPresent() && coordinator.orElseThrow().budget() != budget) {
            throw new IllegalArgumentException("OpenBLAS coordinator must share finalizer budget");
        }
    }

    /** @return the stable non-null CPU ownership identity */
    @Override public BackendId backendId() { return CpuCapabilityProvider.CPU_BACKEND_ID; }

    /**
     * Verifies exact assignments and realizes generated units or direct recurrent children without
     * changing analysis.
     * @param finalization non-null complete shared post-assignment handoff
     * @return a non-null result retaining the exact direct or composite executable and the
     *     acquisition-ordered persistent native kernels whose ownership transfers to Runtime;
     *     portable and OpenBLAS results retain no newly owned persistent resources
     * @throws NullPointerException if {@code finalization} is {@code null}
     * @throws IllegalArgumentException if ownership, assignments, typed specialization or retained
     *     realization facts disagree
     * @throws IllegalStateException if a required native owner is not open, native compilation
     *     fails, or coordinated configuration, verification, or restoration fails
     * @throws CpuConcurrencyBudget.CpuCoordinationException if interrupted while waiting for a
     *     coordinated configuration transition; interrupt status is restored
     */
    @Override public BackendPartitionFinalizationResult finalizePartition(
            BackendPartitionFinalization<CpuPartitionPreparationPlan> finalization) {
        var resources = new ArrayList<PreparedResource>();
        try {
            PreparedExecutable executable = finalizeExecutable(finalization, resources);
            return new BackendPartitionFinalizationResult(executable, resources);
        } catch (RuntimeException | Error failure) {
            for (int index = resources.size() - 1; index >= 0; index--) {
                try { resources.get(index).close(); } catch (RuntimeException | Error cleanup) {
                    if (cleanup != failure) failure.addSuppressed(cleanup);
                }
            }
            throw failure;
        }
    }

    /**
     * Resolves every assigned resource before realizing the already-selected partition recipe.
     * Output-only count units use composite child selection so retained original sources do not
     * enter their generated argument list.
     * @param finalization non-null immutable analysis and exact assigned-resource handoff, borrowed
     * @return non-null immutable executable retaining the selected resource coordinates
     * @throws NullPointerException if {@code finalization} is {@code null}
     * @throws IllegalArgumentException if owner, assignments or selected executable facts disagree
     */
    private PreparedExecutable finalizeExecutable(
            BackendPartitionFinalization<CpuPartitionPreparationPlan> finalization,
            List<PreparedResource> resources) {
        Objects.requireNonNull(finalization, "finalization");
        var plan = finalization.analysis().plan();
        if (!finalization.analysis().partition().owner().equals(CpuCapabilityProvider.CPU_BACKEND_ID)) {
            throw new IllegalArgumentException("partition owner must be CPU");
        }
        var selections = new ArrayList<PreparedExecutable.BufferSelection>(4);
        for (var declaration : plan.bufferDeclarations()) {
            PreparationResourceAssignment.Buffer match = null;
            for (var assignment : finalization.assignments()) {
                if (assignment instanceof PreparationResourceAssignment.Buffer buffer
                        && buffer.requirement() == declaration) { match = buffer; break; }
            }
            if (match == null) throw new IllegalArgumentException(
                    "buffer declaration has no exact shared assignment: " + declaration.valueId());
            var entry = finalization.memoryPlan().buffers().get(match.planIndex());
            if (entry.slot() != match.slot() || (entry.size().fixedByteSize().isPresent() && declaration.size().fixedByteSize().isPresent()
                    ? entry.byteSize() < declaration.byteSize() : !entry.size().equals(declaration.size()))
                    || entry.byteAlignment() < declaration.byteAlignment()) {
                throw new IllegalArgumentException("buffer assignment geometry disagrees");
            }
            selections.add(new PreparedExecutable.BufferSelection(
                    match.planIndex(), match.representationIndex()));
        }
        long bufferAssignments = finalization.assignments().stream()
                .filter(PreparationResourceAssignment.Buffer.class::isInstance).count();
        if (bufferAssignments != plan.bufferDeclarations().size()) {
            throw new IllegalArgumentException("CPU finalization requires the exact buffer set");
        }
        var workspaceDeclarations = finalization.analysis().requirements().stream()
                .filter(io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement.Workspace.class::isInstance)
                .map(io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement.Workspace.class::cast)
                .toList();
        var workspaceSelections = new LinkedHashMap<io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement.Workspace,
                PreparedExecutable.WorkspaceSelection>();
        for (var declaration : workspaceDeclarations) {
            PreparationResourceAssignment.Workspace match = null;
            for (var assignment : finalization.assignments()) {
                if (assignment instanceof PreparationResourceAssignment.Workspace workspace
                        && workspace.requirement() == declaration) { match = workspace; break; }
            }
            if (match == null) throw new IllegalArgumentException(
                    "workspace declaration has no exact shared assignment");
            var entry = finalization.memoryPlan().workspaces().get(match.planIndex());
            if (entry.slot() != match.slot() || !entry.size().equals(declaration.size())
                    || entry.byteAlignment() != declaration.byteAlignment()) {
                throw new IllegalArgumentException("workspace assignment geometry disagrees");
            }
            workspaceSelections.put(declaration,
                    new PreparedExecutable.WorkspaceSelection(match.planIndex()));
        }
        long assignedWorkspaces = finalization.assignments().stream()
                .filter(PreparationResourceAssignment.Workspace.class::isInstance).count();
        if (assignedWorkspaces != workspaceDeclarations.size()) {
            throw new IllegalArgumentException("CPU finalization requires the exact workspace set");
        }
        if (plan.route() == CpuPartitionPreparationPlan.Route.ONEDNN) {
            var nativeRoute = plan.oneDnnPlan().orElseThrow();
            var library = oneDnnLibrary.orElseThrow(() -> new IllegalArgumentException(
                    "selected oneDNN route requires its exact library owner"));
            library.requireOpen();
            if (!library.sessionIdentity().equals(nativeRoute.sessionIdentity())) {
                throw new IllegalArgumentException("oneDNN plan belongs to another library session");
            }
            nativeRoute.validate(plan.units());
            for (var requirement : nativeRoute.workspaceRequirements()) {
                if (workspaceDeclarations.stream().noneMatch(value -> value == requirement)
                        || workspaceSelections.get(requirement) == null) {
                    throw new IllegalArgumentException("selected oneDNN workspace assignment is absent");
                }
            }
            concurrencyBudget.orElseThrow(() -> new IllegalArgumentException(
                    "selected oneDNN route requires its shared CPU budget"));
        }
        if (plan.route() == CpuPartitionPreparationPlan.Route.OPENBLAS) {
            return finalizeOpenBlas(finalization, selections, workspaceSelections);
        }
        if (plan.route() == CpuPartitionPreparationPlan.Route.ONEDNN
                || plan.units().size() > 1 || !plan.materializations().isEmpty()
                || plan.units().getFirst().portablePlan() != null && plan.units().getFirst().portablePlan().portableKernelIr()
                    instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConstantWriteIr
                || plan.units().getFirst().portablePlan() != null
                    && (plan.units().getFirst().portablePlan().specialization().ordinaryBatch()
                        || plan.units().getFirst().portablePlan().ordinaryBatchGeometry().isPresent())
                || plan.units().getFirst().runtimeFacts().recurrentScan().isPresent()
                || plan.units().getFirst().runtimeFacts().generalScan().isPresent()) {
            return finalizeComposite(finalization, selections, workspaceSelections, resources);
        }
        var unit = plan.units().getFirst();
        CpuWorkerGroup selectedWorkers = null;
        if (plan.selectedRangeCount() >= 2) {
            if (concurrencyBudget.isPresent()
                    && plan.selectedRangeCount() > concurrencyBudget.orElseThrow().capacity()) {
                throw new IllegalArgumentException("parallel CPU plan exceeds shared budget");
            }
            selectedWorkers = workerGroup.orElseThrow(() -> new IllegalArgumentException(
                    "parallel CPU plan requires a worker group"));
            if (!selectedWorkers.isOpen() || selectedWorkers.workerCount() < plan.selectedRangeCount()) {
                throw new IllegalArgumentException("CPU worker group is closed or undersized");
            }
        }
        var artifact = artifactStore.loadOrGenerate(unit.portablePlan().specialization(),
                unit.portablePlan().kernelIr());
        var partialRecipe = plan.partialReductionRecipe();
        Optional<io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel.PartialReductionArtifact>
                partialArtifact = Optional.empty();
        if (partialRecipe.isPresent()) {
            var recipe = partialRecipe.orElseThrow();
            if (!recipe.generatedArtifactIdentity().equals(
                    CpuPartitionPreparationPlan.PartialReductionRecipe.artifactIdentity(recipe.ir()))
                    || selectedWorkers == null || plan.workspaceDeclaration().isEmpty()
                    || plan.workspaceDeclaration().orElseThrow().byteSize()
                            != recipe.ir().workspaceBytes()
                    || plan.workspaceDeclaration().orElseThrow().byteAlignment() != Long.BYTES
                    || plan.selectedRangeCount() != recipe.ir().partialCount()) {
                throw new IllegalArgumentException("partial-reduction finalization facts disagree");
            }
            partialArtifact = Optional.of(partialReductionGenerator.generatePartialReduction(recipe.ir()));
        }
        return new CpuPreparedExecutable(finalization.memoryPlan(), selections, artifact,
                plan.accessBindings(), plan.carrierPattern(), plan.generatedCarrierPattern(),
                0, plan.elementCount(),
                plan.selectedRangeCount(), plan.minimumElementsPerWorker(), selectedWorkers,
                plan.workspaceDeclaration().map(workspaceSelections::get), List.of(), List.of(),
                plan.units().getFirst().portablePlan().portableKernelIr()
                        instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr
                        ? plan.affineAddressPairs() : null,
                plan.movementGeometry(), plan.indexingGeometry(), plan.scatterGeometry(),
                plan.foldGeometry(), plan.orderingGeometry(), plan.randomGeometry(),
                plan.scanGeometry(), plan.aggregateGeometry(), plan.argExtremaGeometry(),
                plan.maskedReductionGeometry(), plan.advancedReductionGeometry(),
                plan.softmaxGeometry(), plan.trailingNormalizationGeometry(),
                plan.batchNormInferenceGeometry(), plan.batchNormTrainingGeometry(),
                plan.conv2dGeometry(), unit.conv3dGeometry(), unit.matmulGeometry(),
                unit.pool2dGeometry(), unit.pool3dGeometry(), unit.attentionGeometry(),
                lossGeometry(unit),
                unit.outputCount(), partialArtifact,
                selectedWorkers == null ? concurrencyBudget.orElse(null) : null);
    }

    private PreparedExecutable finalizeOpenBlas(
            BackendPartitionFinalization<CpuPartitionPreparationPlan> finalization,
            List<PreparedExecutable.BufferSelection> selections,
            java.util.Map<io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement.Workspace,
                    PreparedExecutable.WorkspaceSelection> workspaceSelections) {
        var plan = finalization.analysis().plan();
        var route = plan.openBlasPlan().orElseThrow(() -> new IllegalArgumentException(
                "selected OpenBLAS route has no exact route plan"));
        CpuOpenBlasInvocation invocation = openBlasInvocation.orElse(null);
        CpuOpenBlasCoordinator coordinator = openBlasCoordinator.orElse(null);
        if (coordinator != null) {
            var qualification = route.qualification().orElseThrow(() ->
                    new IllegalArgumentException(
                            "selected OpenBLAS route has no qualification credential"));
            coordinator.validateQualification(qualification);
            CpuConcurrencyBudget budget = concurrencyBudget.orElseThrow();
            if (coordinator.budget() != budget || coordinator.capacity() != route.analysisCapacity()
                    || budget.capacity() != route.analysisCapacity()) {
                throw new IllegalArgumentException("OpenBLAS plan and live CPU budget disagree");
            }
            coordinator.configure(route.threadCount());
        } else throw new IllegalArgumentException(
                "qualified OpenBLAS finalization requires its exact coordinator");
        List<PreparedExecutable.WorkspaceSelection> routeWorkspaces =
                route.workspaceRequirements().stream().map(requirement -> {
                    PreparedExecutable.WorkspaceSelection selection =
                            workspaceSelections.get(requirement);
                    if (selection == null) throw new IllegalArgumentException(
                            "selected OpenBLAS workspace assignment is absent");
                    return selection;
                }).toList();
        var inputArtifacts = route.inputMaterializations().stream().map(copy ->
                artifactStore.loadOrGenerate(copy.copySpecialization(),
                        copy.copyIr().encodedKernelIr())).toList();
        Optional<io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel>
                outputArtifact = route.outputCopy().map(copy -> artifactStore.loadOrGenerate(
                        copy.copySpecialization(), copy.copyIr().encodedKernelIr()));
        return new CpuOpenBlasPreparedExecutable(finalization.memoryPlan(), selections,
                routeWorkspaces, route, invocation, coordinator, inputArtifacts, outputArtifact);
    }

    /**
     * Maps partition selections to exact unit ValueIds before constructing sequential children.
     * Partition-only original constant sources retain their read roles without becoming child
     * kernel arguments or additional outputs. Native substitutions retain demand one and do not
     * acquire the replaced ordinary unit's worker team or exact accumulator storage.
     * @param finalization non-null immutable analyzed topology and assigned resources, borrowed
     * @param outerSelections non-null partition buffer coordinates in declaration order, borrowed
     * @param workspaceSelections non-null exact declaration-to-workspace coordinate map, borrowed
     * @param resources non-null call-local acquisition-ordered persistent-resource accumulator
     * @return non-null immutable composite with stable child order and dependency metadata
     * @throws IllegalArgumentException if a unit boundary or declared workspace has no selection
     */
    private PreparedExecutable finalizeComposite(
            BackendPartitionFinalization<CpuPartitionPreparationPlan> finalization,
            List<PreparedExecutable.BufferSelection> outerSelections,
            java.util.Map<io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement.Workspace,
                    PreparedExecutable.WorkspaceSelection> workspaceSelections,
            List<PreparedResource> resources) {
        var plan = finalization.analysis().plan();
        var selectionByValue = new LinkedHashMap<io.github.pho001.synaptik.model.graph.ValueId,
                PreparedExecutable.BufferSelection>();
        for (int i = 0; i < plan.boundaryValues().size(); i++) {
            selectionByValue.put(plan.boundaryValues().get(i), outerSelections.get(i));
        }
        var localSelections = new ArrayList<List<PreparedExecutable.BufferSelection>>();
        var selectedWorkers = new ArrayList<CpuWorkerGroup>();
        for (var unit : plan.units()) {
            var local = unit.boundaryValues().stream().map(selectionByValue::get).toList();
            if (local.stream().anyMatch(Objects::isNull))
                throw new IllegalArgumentException("unit boundary has no assigned selection");
            localSelections.add(local);
            int position = localSelections.size() - 1;
            boolean nativeSelected = plan.oneDnnPlan().isPresent()
                    && plan.oneDnnPlan().orElseThrow().units().stream()
                        .anyMatch(selected -> selected.unitPosition() == position);
            selectedWorkers.add(nativeSelected ? null : workers(unit));
        }
        var nativeUnits = new LinkedHashMap<Integer, CpuOneDnnRoutePlan.Unit>();
        if (plan.oneDnnPlan().isPresent()) {
            for (var nativeUnit : plan.oneDnnPlan().orElseThrow().units()) {
                nativeUnits.put(nativeUnit.unitPosition(), nativeUnit);
                var buffers = localSelections.get(nativeUnit.unitPosition());
                var projection = nativeUnit.projection();
                for (int input = 0; input < projection.spec().inputs().size(); input++) {
                    validateNativeBuffer(finalization, buffers.get(
                            projection.inputBoundaryPositions().get(input)),
                            Math.addExact(projection.byteOffsets().get(input),
                                    projection.spec().inputs().get(input).byteSize()));
                }
                validateNativeBuffer(finalization, buffers.get(projection.outputBoundaryPosition()),
                        Math.addExact(projection.byteOffsets().getLast(),
                                projection.logicalOutputByteSize()));
            }
        }
        var children = new ArrayList<PreparedExecutable>();
        var copyUnits = new ArrayList<CpuPreparedPartitionExecutable.CopyUnit>();
        var orderedWorkspaces = List.copyOf(workspaceSelections.keySet());
        for (var copy : plan.materializations()) {
            var declaration = orderedWorkspaces.stream().filter(value ->
                    value.requirementId() == copy.workspaceRequirementId()).findFirst()
                    .orElseThrow(() -> new IllegalArgumentException(
                            "copy workspace declaration is absent"));
            var artifact = artifactStore.loadOrGenerate(copy.copySpecialization(),
                    copy.copyIr().encodedKernelIr());
            copyUnits.add(new CpuPreparedPartitionExecutable.CopyUnit(copy, artifact,
                    copy.sourceBoundaryIndex(), orderedWorkspaces.indexOf(declaration)));
        }
        for (int index = 0; index < plan.units().size(); index++) {
            int unitIndex = index;
            var unit = plan.units().get(index);
            var nativeUnit = nativeUnits.get(index);
            if (nativeUnit != null) {
                var kernel = CpuOneDnnCompiler.compile(oneDnnLibrary.orElseThrow(),
                        nativeUnit.projection().spec(), nativeUnit.api());
                try { resources.add(kernel); } catch (RuntimeException | Error failure) {
                    try { kernel.close(); } catch (RuntimeException | Error cleanup) {
                        if (cleanup != failure) failure.addSuppressed(cleanup);
                    }
                    throw failure;
                }
                if (!kernel.spec().equals(nativeUnit.projection().spec())
                        || kernel.scratchpadBytes() != nativeUnit.scratchpadBytes()
                        || !kernel.implementation().equals(nativeUnit.implementation())) {
                    throw new IllegalArgumentException(
                            "oneDNN compilation differs from retained analysis metadata: scratch "
                            + nativeUnit.scratchpadBytes() + " -> " + kernel.scratchpadBytes()
                            + ", implementation " + nativeUnit.implementation() + " -> "
                            + kernel.implementation() + ", same spec="
                            + kernel.spec().equals(nativeUnit.projection().spec()));
                }
                Optional<io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel>
                        outputNarrowing = Optional.empty();
                if (nativeUnit.projection().requiresOutputNarrowing()) {
                    var recipe = CpuOneDnnOutputNarrowing.from(nativeUnit.projection());
                    outputNarrowing = Optional.of(artifactStore.loadOrGenerate(
                            recipe.specialization(), recipe.ir()));
                }
                children.add(new CpuOneDnnPreparedExecutable(finalization.memoryPlan(),
                        localSelections.get(index), nativeUnit.workspaceRequirements().stream()
                                .map(workspaceSelections::get).toList(),
                        nativeUnit, kernel, concurrencyBudget.orElseThrow(), outputNarrowing));
                continue;
            }
            if (unit.runtimeFacts().recurrentScan().isPresent()) {
                children.add(new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanPreparedExecutable(
                        finalization.memoryPlan(), localSelections.get(index),
                        unit.runtimeFacts().workspaceDeclaration().map(workspaceSelections::get),
                        unit.runtimeFacts().recurrentScan().orElseThrow(), concurrencyBudget.orElse(null)));
                continue;
            }
            var represented = plan.representationUnits().isEmpty() ? null
                    : plan.representationUnits().get(index);
            var route = represented == null ? unit.portablePlan() : represented.portablePlan();
            var artifact = artifactStore.loadOrGenerate(route.specialization(), route.kernelIr());
            if (unit.runtimeFacts().generalScan().isPresent()) {
                children.add(new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuGeneralScanPreparedExecutable(
                        finalization.memoryPlan(), localSelections.get(index),
                        unit.runtimeFacts().workspaceDeclaration().map(workspaceSelections::get),
                        unit.runtimeFacts().generalScan().orElseThrow(), artifact, concurrencyBudget.orElse(null),
                        selectedWorkers.get(index),unit.minimumElementsPerWorker()));
                continue;
            }
            var consumedCopies = plan.materializations().stream().filter(copy -> copy.consumers()
                    .stream().anyMatch(value -> value.unitPosition() == unitIndex)).toList();
            children.add(unitExecutable(finalization, unit, represented,
                    localSelections.get(index), artifact, selectedWorkers.get(index),
                    unit.runtimeFacts().workspaceDeclaration().map(workspaceSelections::get),
                    consumedCopies, consumedCopies.stream().map(copy -> orderedWorkspaces.stream()
                            .filter(value -> value.requirementId()
                                    == copy.workspaceRequirementId()).findFirst().orElseThrow())
                            .map(workspaceSelections::get).toList()));
        }
        var writeValues = new java.util.HashSet<io.github.pho001.synaptik.model.graph.ValueId>();
        for (var unit : plan.units()) writeValues.addAll(unit.boundaryValues().subList(
                unit.boundaryValues().size() - unit.outputCount(), unit.boundaryValues().size()));
        var accesses = plan.boundaryValues().stream().map(value -> writeValues.contains(value)
                ? PreparedExecutable.BufferAccess.WRITE_ONLY
                : PreparedExecutable.BufferAccess.READ_ONLY).toList();
        return new CpuPreparedPartitionExecutable(finalization.memoryPlan(), outerSelections,
                List.copyOf(workspaceSelections.values()), accesses, copyUnits, children,
                plan.units().stream().map(CpuPartitionPreparationPlan.ExecutionUnitPlan::dependencies)
                    .toList());
    }

    private static void validateNativeBuffer(
            BackendPartitionFinalization<CpuPartitionPreparationPlan> finalization,
            PreparedExecutable.BufferSelection selection, long requiredBytes) {
        var fixed = finalization.memoryPlan().buffers().get(selection.bufferIndex())
                .size().fixedByteSize();
        if (fixed.isEmpty() || fixed.getAsLong() < requiredBytes) {
            throw new IllegalArgumentException(
                    "oneDNN boundary requires a sufficient fixed buffer declaration");
        }
    }

    private CpuWorkerGroup workers(CpuPartitionPreparationPlan.ExecutionUnitPlan unit) {
        if (unit.selectedRangeCount() < 2) return null;
        if (concurrencyBudget.isPresent()
                && unit.selectedRangeCount() > concurrencyBudget.orElseThrow().capacity()) {
            throw new IllegalArgumentException("parallel CPU split unit exceeds shared budget");
        }
        CpuWorkerGroup selected = workerGroup.orElseThrow(() -> new IllegalArgumentException(
                "parallel CPU split unit requires a worker group"));
        if (!selected.isOpen() || selected.workerCount() < unit.selectedRangeCount()) {
            throw new IllegalArgumentException("CPU worker group is closed or undersized");
        }
        return selected;
    }

    private CpuPreparedExecutable unitExecutable(
            BackendPartitionFinalization<CpuPartitionPreparationPlan> finalization,
            CpuPartitionPreparationPlan.ExecutionUnitPlan unit,
            CpuPartitionPreparationPlan.RepresentationUnitPlan represented,
            List<PreparedExecutable.BufferSelection> selections,
            io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel artifact,
            CpuWorkerGroup workers, Optional<PreparedExecutable.WorkspaceSelection> workspace,
            List<io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaterializationPlan> copies,
            List<PreparedExecutable.WorkspaceSelection> copyWorkspaces) {
        var facts = unit.runtimeFacts();
        if (unit.portablePlan().specialization().ordinaryBatch()
                || unit.portablePlan().ordinaryBatchGeometry().isPresent()) {
            return new CpuPreparedExecutable(finalization.memoryPlan(), selections, artifact,
                    unit.portablePlan(), workspace, concurrencyBudget.orElse(null),
                    unit.selectedRangeCount(),unit.minimumElementsPerWorker(),workers);
        }
        var bindings = represented == null ? unit.accessBindings() : represented.accessBindings();
        var generatedCarriers = represented == null ? unit.generatedCarrierPattern()
                : represented.carrierPattern();
        return new CpuPreparedExecutable(finalization.memoryPlan(), selections, artifact,
                bindings, unit.carrierPattern(), generatedCarriers,
                0, unit.elementCount(), unit.selectedRangeCount(),
                unit.minimumElementsPerWorker(), workers, workspace, copies, copyWorkspaces,
                unit.portablePlan().portableKernelIr()
                        instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr
                        ? facts.affineAddressPairs() : null,
                facts.movementGeometry(), facts.indexingGeometry(), facts.scatterGeometry(),
                facts.foldGeometry(), facts.orderingGeometry(), facts.randomGeometry(),
                facts.scanGeometry(), facts.aggregateGeometry(), facts.argExtremaGeometry(),
                facts.maskedReductionGeometry(), facts.advancedReductionGeometry(),
                facts.softmaxGeometry(), facts.trailingNormalizationGeometry(),
                facts.batchNormInferenceGeometry(), facts.batchNormTrainingGeometry(),
                unit.conv2dGeometry(), unit.conv3dGeometry(),
                representedMatmulGeometry(unit, represented),
                unit.pool2dGeometry(), unit.pool3dGeometry(), unit.attentionGeometry(),
                lossGeometry(unit),
                unit.outputCount(), Optional.empty(),
                workers == null ? concurrencyBudget.orElse(null) : null);
    }

    /**
     * Binds packed matrix offsets/strides to the represented consumer, never the copied source.
     * @param unit exact immutable logical unit
     * @param represented selected representation route, or null for direct execution
     * @return original geometry when unchanged, otherwise exact cold represented geometry
     */
    private static Optional<io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMatmulLowering.Geometry>
            representedMatmulGeometry(CpuPartitionPreparationPlan.ExecutionUnitPlan unit,
                    CpuPartitionPreparationPlan.RepresentationUnitPlan represented) {
        if (represented == null || unit.matmulGeometry().isEmpty()
                || represented.accessBindings().equals(unit.accessBindings())) return unit.matmulGeometry();
        return Optional.of(io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMatmulLowering
                .representedGeometry(unit.matmulGeometry().orElseThrow(), represented.accessBindings()));
    }
    private static Optional<io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuLossLowering.Geometry>
            lossGeometry(CpuPartitionPreparationPlan.ExecutionUnitPlan unit) {
        return unit.portablePlan().portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuLossIr loss
                ? Optional.of(loss.geometry()) : Optional.empty();
    }
}
