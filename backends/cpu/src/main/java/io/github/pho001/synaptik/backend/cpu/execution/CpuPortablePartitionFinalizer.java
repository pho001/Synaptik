package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalization;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalizer;
import io.github.pho001.synaptik.prepare.PreparationResourceAssignment;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;

/**
 * Finalizes an assigned portable CPU plan and performs its sole cold artifact-store request.
 *
 * <p>Every selected use is resolved against shared assigned slots and the borrowed worker group
 * is validated before artifact access. The result is an immutable executable recipe; this object
 * neither owns nor closes the worker group and finalization allocates no per-run representation
 * or other closeable prepared resource.</p>
 */
final class CpuPortablePartitionFinalizer
        implements BackendPartitionFinalizer<CpuPortablePreparationPlan> {
    private final CpuGeneratedKernelArtifactStore artifactStore;
    private final CpuWorkerGroup workerGroup;

    /**
     * Creates a finalizer for an explicitly supplied trust root and worker lifetime.
     *
     * @param artifactRoot non-null explicit trusted local artifact root
     * @param workerGroup non-null open already-owned worker group; never closed by this object
     * @throws NullPointerException if an argument is null, in declaration order
     * @throws IllegalStateException if {@code workerGroup} is closed
     */
    CpuPortablePartitionFinalizer(Path artifactRoot, CpuWorkerGroup workerGroup) {
        this.artifactStore = new CpuGeneratedKernelArtifactStore(
                Objects.requireNonNull(artifactRoot, "artifactRoot"));
        this.workerGroup = Objects.requireNonNull(workerGroup, "workerGroup");
        if (workerGroup.isClosed()) throw new IllegalStateException("CPU worker group is closed");
    }

    /** @return {@link CpuCapabilityProvider#CPU_BACKEND_ID} by exact reference; never null */
    @Override public BackendId backendId() { return CpuCapabilityProvider.CPU_BACKEND_ID; }

    /**
     * Resolves every assigned resource before loading or generating the selected artifact.
     *
     * @param finalization non-null validated shared finalization handoff
     * @return immutable portable executable retaining the exact plan, artifact, and worker group
     * @throws NullPointerException if {@code finalization} is null
     * @throws IllegalArgumentException if CPU ownership or selected assignment mapping is invalid
     * @throws IllegalStateException if the worker group is closed
     */
    @Override
    public PreparedExecutable finalizePartition(
            BackendPartitionFinalization<CpuPortablePreparationPlan> finalization) {
        Objects.requireNonNull(finalization, "finalization");
        if (!finalization.analysis().partition().owner()
                .equals(CpuCapabilityProvider.CPU_BACKEND_ID)) {
            throw new IllegalArgumentException("partition owner must be CPU");
        }
        CpuPortablePreparationPlan plan = finalization.analysis().plan();
        CpuPortableKernelCandidate candidate = plan.candidate();
        var assignments = new IdentityHashMap<
                PreparationResourceRequirement, PreparationResourceAssignment>();
        for (var assignment : finalization.assignments()) {
            switch (assignment) {
                case PreparationResourceAssignment.Buffer buffer ->
                        assignments.put(buffer.requirement(), buffer);
                case PreparationResourceAssignment.Workspace workspace ->
                        assignments.put(workspace.requirement(), workspace);
            }
        }
        var bufferSelections = new ArrayList<PreparedExecutable.BufferSelection>();
        var bufferAccesses = new ArrayList<PreparedExecutable.BufferAccess>();
        var bufferDataTypes = new ArrayList<DataType>();
        for (int index = 0; index < candidate.bufferUses().size(); index++) {
            var use = candidate.bufferUses().get(index);
            if (!(assignments.get(use.requirement())
                    instanceof PreparationResourceAssignment.Buffer assignment)) {
                throw new IllegalArgumentException(
                        "bufferUses[" + index + "] has no assigned buffer requirement");
            }
            var argument = candidate.specialization().arguments().get(index);
            bufferSelections.add(new PreparedExecutable.BufferSelection(
                    assignment.planIndex(), use.representationIndex()));
            bufferAccesses.add(argument.access());
            bufferDataTypes.add(argument.dataType());
        }
        var workspaceSelections = new ArrayList<PreparedExecutable.WorkspaceSelection>();
        for (int index = 0; index < candidate.workspaceUses().size(); index++) {
            var use = candidate.workspaceUses().get(index);
            if (!(assignments.get(use.requirement())
                    instanceof PreparationResourceAssignment.Workspace assignment)) {
                throw new IllegalArgumentException(
                        "workspaceUses[" + index + "] has no assigned workspace requirement");
            }
            workspaceSelections.add(
                    new PreparedExecutable.WorkspaceSelection(assignment.planIndex()));
        }
        if (workerGroup.workerCount() != plan.parallelConfiguration().workerCount()) {
            throw new IllegalArgumentException(
                    "worker group count does not match prepared parallel configuration");
        }
        if (workerGroup.isClosed()) throw new IllegalStateException("CPU worker group is closed");

        CpuGeneratedKernel generatedKernel = artifactStore.loadOrGenerate(
                candidate.specialization(), candidate.familyEmitter());
        return new CpuPortablePreparedExecutable(
                finalization.memoryPlan(), List.copyOf(bufferSelections),
                List.copyOf(workspaceSelections), List.copyOf(bufferAccesses),
                List.copyOf(bufferDataTypes), generatedKernel, plan.parallelConfiguration(),
                workerGroup, candidate.invocationBinder());
    }
}
