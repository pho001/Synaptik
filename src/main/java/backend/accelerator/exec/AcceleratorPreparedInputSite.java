package backend.accelerator.exec;

/**
 * Describes where an accelerator external input is consumed inside a prepared partition.
 *
 * @param externalInputNodeId semantic external input node id exposed by the native executable
 * @param consumerNodeId partition node that consumes the external input
 * @param consumerInputIndex input index on the consumer node
 * @param prepared whether the resolved input used a CPU prepared/remapped tensor
 */
public record AcceleratorPreparedInputSite(
        int externalInputNodeId,
        int consumerNodeId,
        int consumerInputIndex,
        boolean prepared
) {
}
