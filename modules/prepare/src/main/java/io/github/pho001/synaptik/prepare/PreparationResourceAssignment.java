package io.github.pho001.synaptik.prepare;

import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.WorkspaceSlot;
import java.util.Objects;

/**
 * Associates one exact analysis declaration with its assigned Runtime slot and plan position.
 *
 * <p>Assignments are immutable preparation results. They retain the exact declaration and slot
 * references used to construct the shared prepared memory plan; they do not allocate, bind, own,
 * or release physical resources.</p>
 */
public sealed interface PreparationResourceAssignment
        permits PreparationResourceAssignment.Buffer, PreparationResourceAssignment.Workspace {
    /**
     * Associates one buffer declaration with a shared buffer slot.
     *
     * @param requirement exact non-null declaration retained from the backend analysis
     * @param slot exact non-null assigned buffer-slot reference
     * @param planIndex non-negative dense index in the prepared plan's buffer list
     */
    record Buffer(
            PreparationResourceRequirement.Buffer requirement,
            BufferSlot slot,
            int planIndex) implements PreparationResourceAssignment {
        /**
         * Creates one immutable buffer assignment.
         *
         * @param requirement exact non-null declaration to retain
         * @param slot exact non-null assigned slot to retain
         * @param planIndex non-negative dense buffer-plan index
         * @throws NullPointerException if {@code requirement} or {@code slot} is null
         * @throws IllegalArgumentException if {@code planIndex} is negative
         */
        public Buffer {
            Objects.requireNonNull(requirement, "requirement");
            Objects.requireNonNull(slot, "slot");
            if (planIndex < 0) {
                throw new IllegalArgumentException("planIndex must be non-negative");
            }
        }
    }

    /**
     * Associates one analysis-local workspace declaration with its distinct workspace slot.
     *
     * @param requirement exact non-null declaration retained from the backend analysis
     * @param slot exact non-null assigned workspace-slot reference
     * @param planIndex non-negative dense index in the prepared plan's workspace list
     */
    record Workspace(
            PreparationResourceRequirement.Workspace requirement,
            WorkspaceSlot slot,
            int planIndex) implements PreparationResourceAssignment {
        /**
         * Creates one immutable workspace assignment.
         *
         * @param requirement exact non-null declaration to retain
         * @param slot exact non-null assigned slot to retain
         * @param planIndex non-negative dense workspace-plan index
         * @throws NullPointerException if {@code requirement} or {@code slot} is null
         * @throws IllegalArgumentException if {@code planIndex} is negative
         */
        public Workspace {
            Objects.requireNonNull(requirement, "requirement");
            Objects.requireNonNull(slot, "slot");
            if (planIndex < 0) {
                throw new IllegalArgumentException("planIndex must be non-negative");
            }
        }
    }
}
