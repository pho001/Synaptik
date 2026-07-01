package planning.partition;

/**
 * Shared safety policy for growing backend partitions across graph partitions.
 */
final class PartitionExpansionPolicy {
    private PartitionExpansionPolicy() {
    }

    static boolean allowsMixedTrainingPhases(PartitionPlanningRequest request) {
        return request != null
                && request.context().supportsBackward()
                && request.sourcePolicy() == PartitionSourcePolicy.CPU_OR_TARGET_BACKEND
                && (request.target() == PartitionTarget.GPU_METAL || request.target() == PartitionTarget.GPU_CUDA);
    }
}
