package io.github.pho001.synaptik.prepare;

import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import java.util.List;
import java.util.Objects;

/**
 * Supplies one backend with its exact analysis, shared memory plan, and source assignments.
 *
 * <p>The assignment snapshot follows the analysis requirement order exactly. Construction proves
 * that every assignment refers to the corresponding exact declaration and to sufficient geometry
 * in the supplied plan. The value contains no physical resource or per-run state.</p>
 *
 * @param <P> concrete backend-owned immutable selected-plan role
 * @param analysis exact non-null analysis being finalized
 * @param memoryPlan exact non-null shared prepared-memory-plan reference
 * @param assignments ordered non-null assignments, one per analysis requirement
 */
public record BackendPartitionFinalization<P extends BackendPreparationPlan>(
        BackendPartitionAnalysis<P> analysis,
        PreparedMemoryPlan memoryPlan,
        List<PreparationResourceAssignment> assignments) {
    /**
     * Validates and snapshots a typed backend finalization input.
     *
     * @param analysis exact non-null analysis to retain
     * @param memoryPlan exact non-null shared memory plan to retain
     * @param assignments non-null ordered assignment list to validate and snapshot
     * @throws NullPointerException if a component or assignment element is null
     * @throws IllegalArgumentException if assignment coverage, source identity, plan index, slot
     *     identity, or geometry does not match the analysis and plan
     */
    public BackendPartitionFinalization {
        Objects.requireNonNull(analysis, "analysis");
        Objects.requireNonNull(memoryPlan, "memoryPlan");
        Objects.requireNonNull(assignments, "assignments");
        int requirementCount = analysis.requirements().size();
        if (assignments.size() != requirementCount) {
            throw new IllegalArgumentException(
                    "assignments size must equal analysis requirement count " + requirementCount);
        }
        for (int index = 0; index < assignments.size(); index++) {
            PreparationResourceAssignment assignment =
                    Objects.requireNonNull(assignments.get(index), "assignments[" + index + "]");
            switch (assignment) {
                case PreparationResourceAssignment.Buffer buffer -> {
                    if (buffer.requirement() != analysis.requirements().get(index)) {
                        throw mismatch(index);
                    }
                    int planIndex = buffer.planIndex();
                    if (planIndex >= memoryPlan.buffers().size()) {
                        throw new IllegalArgumentException(
                                "assignments[" + index + "] buffer planIndex out of range: "
                                        + planIndex);
                    }
                    PreparedMemoryPlan.BufferEntry entry = memoryPlan.buffers().get(planIndex);
                    if (entry.slot() != buffer.slot()) {
                        throw new IllegalArgumentException(
                                "assignments[" + index + "] buffer slot does not match memoryPlan.buffers["
                                        + planIndex + "]");
                    }
                    if (entry.byteSize() < buffer.requirement().byteSize()
                            || entry.byteAlignment() < buffer.requirement().byteAlignment()) {
                        throw new IllegalArgumentException(
                                "assignments[" + index + "] buffer geometry does not satisfy requirement");
                    }
                }
                case PreparationResourceAssignment.Workspace workspace -> {
                    if (workspace.requirement() != analysis.requirements().get(index)) {
                        throw mismatch(index);
                    }
                    int planIndex = workspace.planIndex();
                    if (planIndex >= memoryPlan.workspaces().size()) {
                        throw new IllegalArgumentException(
                                "assignments[" + index + "] workspace planIndex out of range: "
                                        + planIndex);
                    }
                    PreparedMemoryPlan.WorkspaceEntry entry =
                            memoryPlan.workspaces().get(planIndex);
                    if (entry.slot() != workspace.slot()) {
                        throw new IllegalArgumentException(
                                "assignments[" + index + "] workspace slot does not match memoryPlan.workspaces["
                                        + planIndex + "]");
                    }
                    if (entry.byteSize() != workspace.requirement().byteSize()
                            || entry.byteAlignment() != workspace.requirement().byteAlignment()) {
                        throw new IllegalArgumentException(
                                "assignments[" + index + "] workspace geometry does not match requirement");
                    }
                }
            }
        }
        assignments = List.copyOf(assignments);
    }

    private static IllegalArgumentException mismatch(int index) {
        return new IllegalArgumentException(
                "assignments[" + index
                        + "].requirement does not match analysis.requirements[" + index + "]");
    }
}
