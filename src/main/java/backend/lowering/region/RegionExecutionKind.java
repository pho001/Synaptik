package backend.lowering.region;

public enum RegionExecutionKind {
    PROVIDER_CALL,
    FUSED_KERNEL,
    DIRECT_KERNEL,
    VIEW,
    COPY,
    FALLBACK,
    GRAPH_EXECUTABLE,
    UNKNOWN
}
