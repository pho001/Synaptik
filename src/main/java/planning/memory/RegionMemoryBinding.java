package planning.memory;

import planning.value.GraphValueRef;
import tensor.DataType;

import java.util.Objects;

/**
 * Runtime memory binding selected for a region value.
 *
 * @param valueRef graph value reference
 * @param kind binding kind
 * @param bindingId slot or continuation id, absent for {@link RegionMemoryBindingKind#NONE}
 * @param storageType dtype used for storage
 * @param transportType dtype used for region handoff
 * @param requiresAllocation whether runtime binding must allocate storage
 */
public record RegionMemoryBinding(
        GraphValueRef valueRef,
        RegionMemoryBindingKind kind,
        Integer bindingId,
        DataType storageType,
        DataType transportType,
        boolean requiresAllocation
) {
    public RegionMemoryBinding {
        valueRef = Objects.requireNonNull(valueRef, "valueRef cannot be null");
        kind = Objects.requireNonNull(kind, "kind cannot be null");
        storageType = Objects.requireNonNull(storageType, "storageType cannot be null");
        transportType = Objects.requireNonNull(transportType, "transportType cannot be null");
        if (kind == RegionMemoryBindingKind.NONE && bindingId != null) {
            throw new IllegalArgumentException("NONE binding kind must not carry a bindingId");
        }
        if (kind != RegionMemoryBindingKind.NONE && bindingId == null) {
            throw new IllegalArgumentException("Allocated binding kinds must carry a bindingId");
        }
    }

    /**
     * Returns whether this binding carries a binding id.
     *
     * @return {@code true} when {@code bindingId} is non-null
     */
    public boolean hasBindingId() {
        return bindingId != null;
    }
}
