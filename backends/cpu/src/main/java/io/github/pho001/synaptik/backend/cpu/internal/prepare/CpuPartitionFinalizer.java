package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStore;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutable;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedPartitionExecutable;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuWorkerGroup;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalization;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalizer;
import io.github.pho001.synaptik.prepare.PreparationResourceAssignment;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
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
 * assignment is resolved and checked before the first artifact-store call. Finalization then
 * realizes one already-selected artifact per unit in stable order. It cannot change unit
 * topology, dependencies, materialization, generated carrier patterns, route, strategy,
 * specialization, or declaration geometry. A multi-unit result is wrapped in one CPU-private
 * atomic sequential composite; a one-unit result remains the direct child recipe. When the exact
 * analyzed plan contains an explicitly chosen representation candidate, finalization also
 * realizes its one or two generated affine-copy artifacts before consumer artifacts. It does not
 * select among retained candidates.</p>
 */
public final class CpuPartitionFinalizer implements BackendPartitionFinalizer<CpuPartitionPreparationPlan> {
    private final CpuGeneratedKernelArtifactStore artifactStore;
    private final Optional<CpuWorkerGroup> workerGroup;

    /** Creates the default in-memory-only finalizer for single-thread plans. */
    public CpuPartitionFinalizer() { this(Optional.empty(), Optional.empty()); }
    /**
     * Creates a finalizer with optional trusted class-byte persistence.
     * @param trustedArtifactRoot non-null optional trusted local root; empty selects in-memory-only
     *     realization
     * @throws NullPointerException if {@code trustedArtifactRoot} is {@code null}
     */
    public CpuPartitionFinalizer(Optional<Path> trustedArtifactRoot) {
        this(trustedArtifactRoot, Optional.empty());
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
        artifactStore = new CpuGeneratedKernelArtifactStore(trustedArtifactRoot);
        this.workerGroup = Objects.requireNonNull(workerGroup, "workerGroup");
    }

    /** @return the stable non-null CPU ownership identity */
    @Override public BackendId backendId() { return CpuCapabilityProvider.CPU_BACKEND_ID; }

    /**
     * Verifies the complete exact assignment set and realizes selected unit artifacts without
     * changing analysis.
     * @param finalization non-null complete shared post-assignment handoff
     * @return one immutable direct or composite partition-level executable that strongly retains
     *     every selected artifact and only immutable prepared geometry; never {@code null}
     * @throws NullPointerException if {@code finalization} is {@code null}
     * @throws IllegalArgumentException if ownership, assignments, specialization, or artifact
     *     realization is incompatible with the analyzed plan
     */
    @Override public PreparedExecutable finalizePartition(
            BackendPartitionFinalization<CpuPartitionPreparationPlan> finalization) {
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
            if (entry.slot() != match.slot() || entry.byteSize() < declaration.byteSize()
                    || entry.byteAlignment() < declaration.byteAlignment()) {
                throw new IllegalArgumentException("buffer assignment geometry disagrees");
            }
            selections.add(new PreparedExecutable.BufferSelection(match.planIndex(), 0));
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
            if (entry.slot() != match.slot() || entry.byteSize() != declaration.byteSize()
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
        if (plan.units().size() > 1 || !plan.materializations().isEmpty()) {
            return finalizeComposite(finalization, selections, workspaceSelections);
        }
        var unit = plan.units().getFirst();
        CpuWorkerGroup selectedWorkers = null;
        if (plan.selectedRangeCount() >= 2) {
            selectedWorkers = workerGroup.orElseThrow(() -> new IllegalArgumentException(
                    "parallel CPU plan requires a worker group"));
            if (!selectedWorkers.isOpen() || selectedWorkers.workerCount() < plan.selectedRangeCount()) {
                throw new IllegalArgumentException("CPU worker group is closed or undersized");
            }
        }
        var artifact = artifactStore.loadOrGenerate(unit.portablePlan().specialization(),
                unit.portablePlan().kernelIr());
        return new CpuPreparedExecutable(finalization.memoryPlan(), selections, artifact,
                plan.accessBindings(), plan.carrierPattern(), plan.generatedCarrierPattern(),
                0, plan.elementCount(),
                plan.selectedRangeCount(), plan.minimumElementsPerWorker(), selectedWorkers,
                plan.materialization(), plan.workspaceDeclaration().map(workspaceSelections::get),
                plan.units().getFirst().portablePlan().portableKernelIr()
                        instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr
                        ? plan.affineAddressPairs() : null,
                plan.movementGeometry(), plan.indexingGeometry(), plan.scatterGeometry(),
                plan.foldGeometry(), plan.orderingGeometry(), plan.randomGeometry(),
                plan.scanGeometry(), plan.aggregateGeometry(), plan.argExtremaGeometry(),
                plan.maskedReductionGeometry(), plan.advancedReductionGeometry(),
                plan.softmaxGeometry(), plan.trailingNormalizationGeometry(),
                plan.batchNormInferenceGeometry(), plan.batchNormTrainingGeometry(),
                plan.conv2dGeometry(), unit.conv3dGeometry(), unit.matmulGeometry(), unit.outputCount());
    }

    private PreparedExecutable finalizeComposite(
            BackendPartitionFinalization<CpuPartitionPreparationPlan> finalization,
            List<PreparedExecutable.BufferSelection> outerSelections,
            java.util.Map<io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement.Workspace,
                    PreparedExecutable.WorkspaceSelection> workspaceSelections) {
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
            selectedWorkers.add(workers(unit));
        }
        var children = new ArrayList<CpuPreparedExecutable>();
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
            var represented = plan.representationUnits().isEmpty() ? null
                    : plan.representationUnits().get(index);
            var route = represented == null ? unit.portablePlan() : represented.portablePlan();
            var artifact = artifactStore.loadOrGenerate(route.specialization(), route.kernelIr());
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

    private CpuWorkerGroup workers(CpuPartitionPreparationPlan.ExecutionUnitPlan unit) {
        if (unit.selectedRangeCount() < 2) return null;
        CpuWorkerGroup selected = workerGroup.orElseThrow(() -> new IllegalArgumentException(
                "parallel CPU split unit requires a worker group"));
        if (!selected.isOpen() || selected.workerCount() < unit.selectedRangeCount()) {
            throw new IllegalArgumentException("CPU worker group is closed or undersized");
        }
        return selected;
    }

    private static CpuPreparedExecutable unitExecutable(
            BackendPartitionFinalization<CpuPartitionPreparationPlan> finalization,
            CpuPartitionPreparationPlan.ExecutionUnitPlan unit,
            CpuPartitionPreparationPlan.RepresentationUnitPlan represented,
            List<PreparedExecutable.BufferSelection> selections,
            io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel artifact,
            CpuWorkerGroup workers, Optional<PreparedExecutable.WorkspaceSelection> workspace,
            List<io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaterializationPlan> copies,
            List<PreparedExecutable.WorkspaceSelection> copyWorkspaces) {
        var facts = unit.runtimeFacts();
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
                unit.conv2dGeometry(), unit.conv3dGeometry(), unit.matmulGeometry(), unit.outputCount());
    }
}
