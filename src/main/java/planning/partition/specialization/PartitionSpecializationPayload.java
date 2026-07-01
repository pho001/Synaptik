package planning.partition.specialization;

/**
 * Structured backend-neutral payload for graph-level partition specialization candidates.
 */
public sealed interface PartitionSpecializationPayload
        permits EmptyPartitionSpecializationPayload, SdpaBackwardSpecializationPayload {
    /**
     * Returns the canonical empty payload used by specializations that do not need structured metadata.
     *
     * @return empty payload singleton
     */
    static PartitionSpecializationPayload empty() {
        return EmptyPartitionSpecializationPayload.INSTANCE;
    }
}
