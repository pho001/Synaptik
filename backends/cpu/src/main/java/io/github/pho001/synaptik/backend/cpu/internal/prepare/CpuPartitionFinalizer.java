package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStore;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutable;
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

/** Post-assignment verification, artifact realization, and partition executable finalization. */
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
     * @return one immutable partition-level executable that strongly owns its artifact
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
            selections.add(new PreparedExecutable.BufferSelection(match.planIndex(), 0));
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
                plan.accessBindings(), plan.carrierPattern(), 0, plan.elementCount(),
                plan.selectedRangeCount(), plan.minimumElementsPerWorker(), selectedWorkers);
    }
}
