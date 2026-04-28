package graph.execution.trace;

/**
 * Tensor layout metadata for a step.
 *
 * @param storageOffset storage offset used by the tensor
 * @param contiguous whether the tensor layout is contiguous
 * @param stridedPath whether a strided kernel path was used
 * @param targetType target layout or storage type label
 */
public record LayoutTraceMetadata(
        int storageOffset,
        boolean contiguous,
        boolean stridedPath,
        String targetType
) {
}
