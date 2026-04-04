package graph.execution.trace;

public record LayoutTraceMetadata(
        int storageOffset,
        boolean contiguous,
        boolean stridedPath,
        String targetType
) {
}
