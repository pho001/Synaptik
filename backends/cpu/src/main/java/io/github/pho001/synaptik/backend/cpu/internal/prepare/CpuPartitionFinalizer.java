package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStore;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutable;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableSequence;
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
 * <p>Every derived-boundary buffer assignment and optional exact workspace assignment is resolved and checked
 * before the single permitted artifact-store call. Finalization cannot change the selected copy,
 * generated carrier pattern, route, strategy, specialization, or declaration geometry. Fold,
 * cumulative-scan, and RMS-normalization plans retain no workspace assignment; Layer
 * normalization retains only its already-declared exact-state assignment.
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
     * Verifies exact assignments and realizes the selected artifact without changing analysis.
     * @param finalization non-null complete shared post-assignment handoff
     * @return one immutable partition-level executable that strongly owns its artifact and, for
     *     cumulative scans and trailing normalization, retains only immutable slice/layout
     *     geometry; never {@code null}
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
        PreparedExecutable.WorkspaceSelection workspaceSelection = null;
        if (plan.workspaceDeclaration().isPresent()) {
            var declaration = plan.workspaceDeclaration().orElseThrow();
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
            workspaceSelection = new PreparedExecutable.WorkspaceSelection(match.planIndex());
        } else if (finalization.assignments().stream()
                .anyMatch(PreparationResourceAssignment.Workspace.class::isInstance)) {
            throw new IllegalArgumentException("direct CPU plan rejects workspace assignments");
        }
        if (plan.form() == CpuPartitionPreparationPlan.PlanForm.CONV2D_MATERIALIZED_SUFFIX) {
            return finalizeConv2dSequence(finalization, selections);
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
                plan.materialization(), Optional.ofNullable(workspaceSelection),
                plan.units().getFirst().portablePlan().portableKernelIr()
                        instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr
                        ? plan.affineAddressPairs() : null,
                plan.movementGeometry(), plan.indexingGeometry(), plan.scatterGeometry(),
                plan.foldGeometry(), plan.orderingGeometry(), plan.randomGeometry(),
                plan.scanGeometry(), plan.aggregateGeometry(), plan.argExtremaGeometry(),
                plan.maskedReductionGeometry(), plan.advancedReductionGeometry(),
                plan.softmaxGeometry(), plan.trailingNormalizationGeometry(),
                plan.batchNormInferenceGeometry(), plan.batchNormTrainingGeometry(),
                plan.conv2dGeometry());
    }

    private PreparedExecutable finalizeConv2dSequence(
            BackendPartitionFinalization<CpuPartitionPreparationPlan> finalization,
            List<PreparedExecutable.BufferSelection> outerSelections) {
        var plan = finalization.analysis().plan();
        if (plan.units().size() != 2 || plan.workspaceDeclaration().isPresent()) {
            throw new IllegalArgumentException("malformed Conv2d sequence plan");
        }
        var selectionByValue = new LinkedHashMap<io.github.pho001.synaptik.model.graph.ValueId,
                PreparedExecutable.BufferSelection>();
        for (int i = 0; i < plan.boundaryValues().size(); i++) {
            selectionByValue.put(plan.boundaryValues().get(i), outerSelections.get(i));
        }
        var leadUnit = plan.units().get(0);
        var suffixUnit = plan.units().get(1);
        var leadSelections = leadUnit.boundaryValues().stream()
                .map(selectionByValue::get).toList();
        var suffixSelections = suffixUnit.boundaryValues().stream()
                .map(selectionByValue::get).toList();
        if (leadSelections.stream().anyMatch(Objects::isNull)
                || suffixSelections.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException("split unit boundary has no assigned selection");
        }
        CpuWorkerGroup leadWorkers = workers(leadUnit);
        CpuWorkerGroup suffixWorkers = workers(suffixUnit);
        var leadArtifact = artifactStore.loadOrGenerate(
                leadUnit.portablePlan().specialization(), leadUnit.portablePlan().kernelIr());
        var suffixArtifact = artifactStore.loadOrGenerate(
                suffixUnit.portablePlan().specialization(), suffixUnit.portablePlan().kernelIr());
        var lead = unitExecutable(finalization, leadUnit, leadSelections, leadArtifact, leadWorkers);
        var suffix = unitExecutable(finalization, suffixUnit, suffixSelections, suffixArtifact,
                suffixWorkers);
        var writeValues = new java.util.HashSet<io.github.pho001.synaptik.model.graph.ValueId>();
        writeValues.add(leadUnit.boundaryValues().getLast());
        writeValues.addAll(suffixUnit.boundaryValues().subList(
                suffixUnit.boundaryValues().size() - suffixUnit.outputCount(),
                suffixUnit.boundaryValues().size()));
        var accesses = plan.boundaryValues().stream().map(value -> writeValues.contains(value)
                ? PreparedExecutable.BufferAccess.WRITE_ONLY
                : PreparedExecutable.BufferAccess.READ_ONLY).toList();
        return new CpuPreparedExecutableSequence(finalization.memoryPlan(), outerSelections,
                accesses, lead, suffix);
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
            List<PreparedExecutable.BufferSelection> selections,
            io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel artifact,
            CpuWorkerGroup workers) {
        return new CpuPreparedExecutable(finalization.memoryPlan(), selections, artifact,
                unit.accessBindings(), unit.carrierPattern(), unit.generatedCarrierPattern(),
                0, unit.elementCount(), unit.selectedRangeCount(),
                unit.minimumElementsPerWorker(), workers, Optional.empty(), Optional.empty(), null,
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(),
                unit.conv2dGeometry(), unit.outputCount());
    }
}
