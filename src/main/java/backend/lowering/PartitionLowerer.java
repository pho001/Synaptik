package backend.lowering;

/**
 * Backend-specific lowering hook for executable partitions.
 *
 * <p>Lowerers translate executable partitions into backend execution units or accelerator DAG artifacts.
 * A lowerer returns {@code null} or a result with no lowered partition when it cannot handle the request,
 * allowing the lowering pipeline to try the next registered lowerer.</p>
 */
public interface PartitionLowerer {
    /**
     * Attempts to lower one executable partition.
     *
     * @param request lowering request containing the partition, memory plan, capabilities, and context
     * @return lowering result, or {@code null} when this lowerer does not support the partition
     */
    LoweringResult lowerPartition(LoweringRequest request);
}
