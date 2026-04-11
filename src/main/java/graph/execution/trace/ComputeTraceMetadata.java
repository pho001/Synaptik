package graph.execution.trace;

public record ComputeTraceMetadata(
        String mode,
        String storageType,
        String computeType,
        String backend,
        String accumulateType
) {
    public ComputeTraceMetadata {
        mode = mode == null ? "" : mode;
        storageType = storageType == null ? "" : storageType;
        computeType = computeType == null ? "" : computeType;
        backend = backend == null ? "" : backend;
        accumulateType = accumulateType == null ? "" : accumulateType;
    }
}
