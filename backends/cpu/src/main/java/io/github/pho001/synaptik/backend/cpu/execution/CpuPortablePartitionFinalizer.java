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
 * Finalizes an assigned portable CPU partition into one ordered generated-kernel recipe.
 *
 * <p>The finalizer resolves every shared buffer and workspace declaration, computes the unioned
 * partition-level selections and access modes, and validates the borrowed worker configuration
 * before consulting the generated-artifact store. It then loads or generates one artifact per
 * node in partition order. The result owns no physical representation or close lifecycle and
 * borrows, but never closes, the worker group.</p>
 */
final class CpuPortablePartitionFinalizer
        implements BackendPartitionFinalizer<CpuPortablePreparationPlan> {
    private final CpuGeneratedKernelArtifactStore artifactStore;
    private final CpuWorkerGroup workerGroup;

    /**
     * @param artifactRoot non-null explicit trusted local artifact root
     * @param workerGroup non-null open borrowed worker group
     * @throws NullPointerException if an argument is null
     * @throws IllegalStateException if the worker group is closed
     */
    CpuPortablePartitionFinalizer(Path artifactRoot, CpuWorkerGroup workerGroup) {
        this.artifactStore = new CpuGeneratedKernelArtifactStore(
                Objects.requireNonNull(artifactRoot, "artifactRoot"));
        this.workerGroup = Objects.requireNonNull(workerGroup, "workerGroup");
        if (workerGroup.isClosed()) throw new IllegalStateException("CPU worker group is closed");
    }

    /**
     * Returns the stable CPU ownership identity.
     *
     * @return {@link CpuCapabilityProvider#CPU_BACKEND_ID} by exact reference; never
     *     {@code null}
     */
    @Override public BackendId backendId() { return CpuCapabilityProvider.CPU_BACKEND_ID; }

    /**
     * Resolves the complete partition recipe before consulting the artifact store in node order.
     *
     * @param finalization non-null validated shared finalization handoff
     * @return immutable ordered portable executable retaining all generated artifacts; never
     *     {@code null}
     * @throws NullPointerException if {@code finalization} is {@code null}
     * @throws IllegalArgumentException if the partition is not CPU-owned, an assigned shared
     *     requirement cannot be resolved, shared kernel facts disagree, or the worker count does
     *     not match the prepared configuration
     * @throws IllegalStateException if the borrowed worker group is closed or artifact loading,
     *     validation, publication, definition, or exact entry resolution fails
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
        CpuPortablePartitionCandidate candidate = plan.candidate();
        var assignments = new IdentityHashMap<PreparationResourceRequirement,
                PreparationResourceAssignment>();
        for (var assignment : finalization.assignments()) {
            switch (assignment) {
                case PreparationResourceAssignment.Buffer buffer ->
                        assignments.put(buffer.requirement(), buffer);
                case PreparationResourceAssignment.Workspace workspace ->
                        assignments.put(workspace.requirement(), workspace);
            }
        }

        var bufferPositions = new IdentityHashMap<PreparationResourceRequirement.Buffer,
                java.util.Map<Integer, Integer>>();
        var workspacePositions =
                new IdentityHashMap<PreparationResourceRequirement.Workspace, Integer>();
        var bufferSelections = new ArrayList<PreparedExecutable.BufferSelection>();
        var workspaceSelections = new ArrayList<PreparedExecutable.WorkspaceSelection>();
        for (int index = 0; index < candidate.requirements().size(); index++) {
            switch (candidate.requirements().get(index)) {
                case PreparationResourceRequirement.Buffer requirement -> {
                    if (!(assignments.get(requirement)
                            instanceof PreparationResourceAssignment.Buffer assignment)) {
                        throw new IllegalArgumentException(candidate.kernels().size() == 1
                                ? "bufferUses[0] has no assigned buffer requirement"
                                : "requirements[" + index + "] has no assigned buffer requirement");
                    }
                    bufferPositions.put(requirement, new java.util.LinkedHashMap<>());
                }
                case PreparationResourceRequirement.Workspace requirement -> {
                    if (!(assignments.get(requirement)
                            instanceof PreparationResourceAssignment.Workspace assignment)) {
                        throw new IllegalArgumentException(candidate.kernels().size() == 1
                                ? "workspaceUses[0] has no assigned workspace requirement"
                                : "requirements[" + index + "] has no assigned workspace requirement");
                    }
                    workspacePositions.put(requirement, workspaceSelections.size());
                    workspaceSelections.add(new PreparedExecutable.WorkspaceSelection(
                            assignment.planIndex()));
                }
            }
        }

        var accesses = new ArrayList<PreparedExecutable.BufferAccess>();
        var dataTypes = new ArrayList<DataType>();
        var resolved = new ArrayList<ResolvedKernel>(candidate.kernels().size());
        for (int kernelIndex = 0; kernelIndex < candidate.kernels().size(); kernelIndex++) {
            var kernel = candidate.kernels().get(kernelIndex);
            var bufferIndices = new int[kernel.bufferUses().size()];
            for (int index = 0; index < kernel.bufferUses().size(); index++) {
                var use = kernel.bufferUses().get(index);
                var representations = bufferPositions.get(use.requirement());
                if (representations == null) throw new IllegalArgumentException(
                        "kernels[" + kernelIndex + "].bufferUses[" + index
                                + "] has no shared assignment");
                Integer position = representations.get(use.representationIndex());
                if (position == null) {
                    var assignment = (PreparationResourceAssignment.Buffer)
                            assignments.get(use.requirement());
                    position = bufferSelections.size();
                    representations.put(use.representationIndex(), position);
                    bufferSelections.add(new PreparedExecutable.BufferSelection(
                            assignment.planIndex(), use.representationIndex()));
                    accesses.add(null);
                    dataTypes.add(null);
                }
                var argument = kernel.specialization().arguments().get(index);
                DataType priorType = dataTypes.get(position);
                if (priorType != null && priorType != argument.dataType()) throw new IllegalArgumentException(
                        "shared buffer data type differs between kernel uses");
                dataTypes.set(position, argument.dataType());
                accesses.set(position, union(accesses.get(position), argument.access()));
                bufferIndices[index] = position;
            }
            var workspaceIndices = new int[kernel.workspaceUses().size()];
            for (int index = 0; index < kernel.workspaceUses().size(); index++) {
                Integer position = workspacePositions.get(
                        kernel.workspaceUses().get(index).requirement());
                if (position == null) throw new IllegalArgumentException(
                        "kernels[" + kernelIndex + "].workspaceUses[" + index
                                + "] has no shared assignment");
                workspaceIndices[index] = position;
            }
            resolved.add(new ResolvedKernel(kernel, bufferIndices, workspaceIndices));
        }
        for (int index = 0; index < accesses.size(); index++) {
            if (accesses.get(index) == null || dataTypes.get(index) == null) throw new IllegalArgumentException(
                    "shared buffer requirement is unused at position " + index);
        }
        if (workerGroup.workerCount() != plan.parallelConfiguration().workerCount()) {
            throw new IllegalArgumentException(
                    "worker group count does not match prepared parallel configuration");
        }
        if (workerGroup.isClosed()) throw new IllegalStateException("CPU worker group is closed");

        var recipes = new ArrayList<CpuPortablePreparedExecutable.KernelRecipe>(resolved.size());
        for (ResolvedKernel item : resolved) {
            CpuGeneratedKernel generated = artifactStore.loadOrGenerate(
                    item.kernel.specialization(), item.kernel.familyEmitter());
            recipes.add(new CpuPortablePreparedExecutable.KernelRecipe(
                    generated, item.bufferIndices, item.workspaceIndices,
                    item.kernel.invocationBinder()));
        }
        return new CpuPortablePreparedExecutable(finalization.memoryPlan(), bufferSelections,
                workspaceSelections, accesses, dataTypes, recipes,
                plan.parallelConfiguration(), workerGroup);
    }

    private static PreparedExecutable.BufferAccess union(
            PreparedExecutable.BufferAccess left, PreparedExecutable.BufferAccess right) {
        if (left == null || left == right) return right;
        if (left == PreparedExecutable.BufferAccess.READ_WRITE
                || right == PreparedExecutable.BufferAccess.READ_WRITE) {
            return PreparedExecutable.BufferAccess.READ_WRITE;
        }
        return PreparedExecutable.BufferAccess.READ_WRITE;
    }

    private record ResolvedKernel(CpuPortableKernelCandidate kernel, int[] bufferIndices,
            int[] workspaceIndices) {}
}
