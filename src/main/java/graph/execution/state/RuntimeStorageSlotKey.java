package graph.execution.state;

import tensor.DataType;

import java.util.Objects;

/**
 * Stable run-scoped identity for reusable runtime storage.
 *
 * @param kind physical storage family
 * @param dataType tensor dtype stored in the slot
 * @param scope planner/runtime id namespace
 * @param storageId region slot id or node id, depending on {@code scope}
 * @param elements number of logical elements in the slot
 */
public record RuntimeStorageSlotKey(
        RuntimeStorageKind kind,
        DataType dataType,
        RuntimeStorageSlotScope scope,
        int storageId,
        int elements
) {
    public RuntimeStorageSlotKey {
        kind = Objects.requireNonNull(kind, "kind cannot be null");
        dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        scope = Objects.requireNonNull(scope, "scope cannot be null");
        if (storageId < 0) {
            throw new IllegalArgumentException("storageId cannot be negative: " + storageId);
        }
        if (elements < 0) {
            throw new IllegalArgumentException("elements cannot be negative: " + elements);
        }
    }

    public static RuntimeStorageSlotKey regionSlot(
            RuntimeStorageKind kind,
            DataType dataType,
            int slotId,
            int elements
    ) {
        return new RuntimeStorageSlotKey(kind, dataType, RuntimeStorageSlotScope.REGION_SLOT, slotId, elements);
    }

    public static RuntimeStorageSlotKey nodeOutput(
            RuntimeStorageKind kind,
            DataType dataType,
            int nodeId,
            int elements
    ) {
        return new RuntimeStorageSlotKey(kind, dataType, RuntimeStorageSlotScope.NODE_OUTPUT, nodeId, elements);
    }

    public RuntimeStorageSlotKey withKind(RuntimeStorageKind replacementKind) {
        return new RuntimeStorageSlotKey(replacementKind, dataType, scope, storageId, elements);
    }
}
