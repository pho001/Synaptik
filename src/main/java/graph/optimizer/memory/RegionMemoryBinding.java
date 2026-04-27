package graph.optimizer.memory;

import graph.optimizer.region.RegionValueRef;
import tensor.DataType;

import java.util.Objects;

public record RegionMemoryBinding(
        RegionValueRef valueRef,
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

    public boolean hasBindingId() {
        return bindingId != null;
    }
}
