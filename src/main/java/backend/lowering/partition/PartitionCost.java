package backend.lowering.partition;

public record PartitionCost(
        long estimatedWork,
        long estimatedCopyInBytes,
        long estimatedCopyOutBytes,
        long estimatedTempBytes
) {
    public PartitionCost {
        estimatedWork = Math.max(0L, estimatedWork);
        estimatedCopyInBytes = Math.max(0L, estimatedCopyInBytes);
        estimatedCopyOutBytes = Math.max(0L, estimatedCopyOutBytes);
        estimatedTempBytes = Math.max(0L, estimatedTempBytes);
    }

    public static PartitionCost ofWork(long estimatedWork) {
        return new PartitionCost(estimatedWork, 0L, 0L, 0L);
    }
}
