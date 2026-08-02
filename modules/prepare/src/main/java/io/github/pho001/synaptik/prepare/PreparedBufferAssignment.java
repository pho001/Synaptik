package io.github.pho001.synaptik.prepare;

import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import java.util.Objects;

/**
 * Associates one logical graph value with its dense prepared-buffer position and exact slot.
 *
 * <p>This immutable Prepare-only translation fact is not a physical binding or Runtime state.</p>
 *
 * @param valueId exact non-null logical value identity
 * @param slot exact non-null assigned Runtime buffer slot
 * @param planIndex non-negative dense position in the prepared memory plan's buffer list
 */
public record PreparedBufferAssignment(ValueId valueId, BufferSlot slot, int planIndex) {
    /**
     * Validates one logical-to-prepared buffer association.
     *
     * @param valueId exact non-null value identity to retain
     * @param slot exact non-null slot reference to retain
     * @param planIndex non-negative dense prepared-buffer position
     * @throws NullPointerException if {@code valueId} or {@code slot} is null
     * @throws IllegalArgumentException if {@code planIndex} is negative
     */
    public PreparedBufferAssignment {
        Objects.requireNonNull(valueId, "valueId");
        Objects.requireNonNull(slot, "slot");
        if (planIndex < 0) {
            throw new IllegalArgumentException("planIndex must be non-negative");
        }
    }
}
