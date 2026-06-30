package planning.region.specialization;

/**
 * Structured backend-neutral payload for graph-level region specialization candidates.
 */
public sealed interface RegionSpecializationPayload
        permits EmptyRegionSpecializationPayload, SdpaBackwardSpecializationPayload {
    /**
     * Returns the canonical empty payload used by specializations that do not need structured metadata.
     *
     * @return empty payload singleton
     */
    static RegionSpecializationPayload empty() {
        return EmptyRegionSpecializationPayload.INSTANCE;
    }
}
