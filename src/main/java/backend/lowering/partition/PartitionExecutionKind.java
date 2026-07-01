package backend.lowering.partition;

public enum PartitionExecutionKind {
    PROVIDER_CALL,
    FUSED_KERNEL,
    DIRECT_KERNEL,
    VIEW,
    COPY,
    FALLBACK,
    GRAPH_EXECUTABLE,
    UNKNOWN
}
