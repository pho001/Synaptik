package backend.lowering.partition;

public enum PartitionStorageContract {
    CPU_ARRAY,
    CPU_NATIVE,
    DEVICE_BUFFER,
    VIEW_ALIAS,
    MIXED_BOUNDARY,
    UNKNOWN
}
