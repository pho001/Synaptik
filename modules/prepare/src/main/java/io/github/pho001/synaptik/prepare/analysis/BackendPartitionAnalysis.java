package io.github.pho001.synaptik.prepare.analysis;

import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

/**
 * Retains one backend's opaque partition plan and exact shared resource declarations.
 *
 * <p>The partition and plan references are retained exactly so later shared orchestration can
 * preserve the association established by {@link BackendPartitionPreparer#analyze(PrepareContext)}.
 * Requirements are an immutable ordered snapshot. Buffer value IDs are unique among buffer
 * declarations, and workspace requirement IDs are unique among workspace declarations.</p>
 *
 * <p>This analysis result precedes shared slot assignment and backend finalization. It contains no
 * Runtime slot, physical allocation, resource handle, executable, schedule, or per-run state.</p>
 *
 * @param <P> concrete backend-owned immutable selected-plan role
 * @param partition non-null exact planned partition reference analyzed by the backend
 * @param plan non-null opaque immutable backend plan retained by exact reference
 * @param requirements non-null ordered resource declarations to snapshot; elements must be
 *     non-null, with unique buffer value IDs and unique workspace requirement IDs
 */
public record BackendPartitionAnalysis<P extends BackendPreparationPlan>(
        PlannedPartition partition,
        P plan,
        List<PreparationResourceRequirement> requirements) {
    /**
     * Validates and snapshots one backend analysis result.
     *
     * @param partition non-null planned partition reference to retain exactly
     * @param plan non-null backend-owned immutable plan to retain opaquely and exactly
     * @param requirements non-null ordered resource declarations to validate and snapshot
     * @throws NullPointerException if a component or requirement element is {@code null}; the
     *     message identifies the component or indexed entry
     * @throws IllegalArgumentException if a later buffer repeats an earlier buffer value ID or a
     *     later workspace repeats an earlier workspace requirement ID
     */
    public BackendPartitionAnalysis {
        Objects.requireNonNull(partition, "partition");
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(requirements, "requirements");

        var bufferIds = new HashSet<ValueId>();
        var workspaceIds = new HashSet<Long>();
        for (int index = 0; index < requirements.size(); index++) {
            PreparationResourceRequirement requirement =
                    Objects.requireNonNull(requirements.get(index), "requirements[" + index + "]");
            switch (requirement) {
                case PreparationResourceRequirement.Buffer buffer -> {
                    if (!bufferIds.add(buffer.valueId())) {
                        throw new IllegalArgumentException(
                                "requirements["
                                        + index
                                        + "] duplicates buffer "
                                        + buffer.valueId());
                    }
                }
                case PreparationResourceRequirement.Workspace workspace -> {
                    if (!workspaceIds.add(workspace.requirementId())) {
                        throw new IllegalArgumentException(
                                "requirements["
                                        + index
                                        + "] duplicates workspace requirementId "
                                        + workspace.requirementId());
                    }
                }
            }
        }
        requirements = List.copyOf(requirements);
    }

    /**
     * Returns the analyzed planned partition.
     *
     * @return exact non-null immutable partition reference supplied at construction
     */
    @Override
    public PlannedPartition partition() {
        return partition;
    }

    /**
     * Returns the opaque backend-selected plan for later finalization.
     *
     * @return exact non-null immutable backend-owned plan reference supplied at construction
     */
    @Override
    public P plan() {
        return plan;
    }

    /**
     * Returns exact shared resource declarations in deterministic supplied order.
     *
     * @return non-null immutable ordered snapshot with unique buffer and workspace identities
     */
    @Override
    public List<PreparationResourceRequirement> requirements() {
        return requirements;
    }
}
