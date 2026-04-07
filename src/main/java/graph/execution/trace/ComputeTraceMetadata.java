package graph.execution.trace;

public record ComputeTraceMetadata(
        String mode
) {
    public ComputeTraceMetadata {
        mode = mode == null ? "" : mode;
    }
}
