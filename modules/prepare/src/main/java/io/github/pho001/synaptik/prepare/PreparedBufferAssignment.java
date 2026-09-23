package io.github.pho001.synaptik.prepare;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

/**
 * Associates one logical graph value with its dense prepared-buffer position, exact slot, and
 * deterministic representation-owner order.
 *
 * <p>This immutable Prepare-only translation fact is not a physical binding or Runtime state.
 * Representation index zero belongs to the producing owner when one exists, otherwise the first
 * consuming owner. Remaining distinct owners follow logical consumer order.</p>
 *
 * @param valueId exact non-null logical value identity
 * @param slot exact non-null assigned Runtime buffer slot
 * @param planIndex non-negative dense position in the prepared memory plan's buffer list
 * @param representationOwners non-null non-empty ordered distinct backend owners, one per
 *     representation position
 */
public record PreparedBufferAssignment(
        ValueId valueId,
        BufferSlot slot,
        int planIndex,
        List<BackendId> representationOwners) {
    /**
     * Validates one logical-to-prepared buffer association and snapshots its owner order.
     *
     * @param valueId exact non-null value identity to retain
     * @param slot exact non-null slot reference to retain
     * @param planIndex non-negative dense prepared-buffer position
     * @param representationOwners non-null non-empty ordered owner list to snapshot
     * @throws NullPointerException if a reference or owner element is null
     * @throws IllegalArgumentException if {@code planIndex} is negative, the owner list is empty,
     *     or an owner repeats
     */
    public PreparedBufferAssignment {
        Objects.requireNonNull(valueId, "valueId");
        Objects.requireNonNull(slot, "slot");
        Objects.requireNonNull(representationOwners, "representationOwners");
        if (planIndex < 0) {
            throw new IllegalArgumentException("planIndex must be non-negative");
        }
        if (representationOwners.isEmpty()) {
            throw new IllegalArgumentException("representationOwners must not be empty");
        }
        var observed = new HashSet<BackendId>();
        for (int index = 0; index < representationOwners.size(); index++) {
            BackendId owner = Objects.requireNonNull(
                    representationOwners.get(index), "representationOwners[" + index + "]");
            if (!observed.add(owner)) {
                throw new IllegalArgumentException(
                        "representationOwners[" + index + "] duplicates " + owner);
            }
        }
        representationOwners = List.copyOf(representationOwners);
    }
}
