package backend.cpu1.kernels.dtype;

import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;

/**
 * Prepared cpu1 dtype kernel variants.
 */
public enum Cpu1DTypeKernelId {
    CAST_ARRAY_SCALAR(Operation.OpType.CAST, Cpu1StorageKind.JAVA_ARRAY),
    CAST_SEGMENT_SCALAR(Operation.OpType.CAST, Cpu1StorageKind.MEMORY_SEGMENT);

    private final Operation.OpType opType;
    private final Cpu1StorageKind storageKind;

    Cpu1DTypeKernelId(Operation.OpType opType, Cpu1StorageKind storageKind) {
        this.opType = opType;
        this.storageKind = storageKind;
    }

    public Operation.OpType opType() {
        return opType;
    }

    public Cpu1StorageKind storageKind() {
        return storageKind;
    }
}
