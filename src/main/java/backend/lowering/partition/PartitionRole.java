package backend.lowering.partition;

public enum PartitionRole {
    PROVIDER,
    LOCAL_KERNEL,
    VIEW_ALIAS,
    BOUNDARY_OUTPUT,
    CONTROL,
    FALLBACK_ONLY,
    UNKNOWN
}
