package io.github.pho001.synaptik.prepare;

import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.runtime.memory.WorkspaceSlot;
import java.util.Objects;

/**
 * Associates one prepared workspace with the exact partition that declared and owns it.
 *
 * <p>The immutable association is plan-ordered Prepare metadata. It identifies where a backend
 * must contribute the per-run workspace creator and owns no physical resource.</p>
 *
 * @param partition exact non-null declaring planned partition
 * @param slot exact non-null assigned Runtime workspace slot
 * @param planIndex non-negative dense position in the prepared memory plan's workspace list
 */
public record PreparedWorkspaceAssignment(
        PlannedPartition partition,
        WorkspaceSlot slot,
        int planIndex) {
    /**
     * Validates one partition-to-workspace association.
     *
     * @param partition exact non-null declaring partition to retain
     * @param slot exact non-null assigned slot to retain
     * @param planIndex non-negative dense prepared-workspace position
     * @throws NullPointerException if {@code partition} or {@code slot} is null
     * @throws IllegalArgumentException if {@code planIndex} is negative
     */
    public PreparedWorkspaceAssignment {
        Objects.requireNonNull(partition, "partition");
        Objects.requireNonNull(slot, "slot");
        if (planIndex < 0) {
            throw new IllegalArgumentException("planIndex must be non-negative");
        }
    }
}
