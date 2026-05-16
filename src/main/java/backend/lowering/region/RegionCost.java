package backend.lowering.region;

public record RegionCost(
        long estimatedWork,
        long estimatedCopyInBytes,
        long estimatedCopyOutBytes,
        long estimatedTempBytes
) {
    public RegionCost {
        estimatedWork = Math.max(0L, estimatedWork);
        estimatedCopyInBytes = Math.max(0L, estimatedCopyInBytes);
        estimatedCopyOutBytes = Math.max(0L, estimatedCopyOutBytes);
        estimatedTempBytes = Math.max(0L, estimatedTempBytes);
    }

    public static RegionCost ofWork(long estimatedWork) {
        return new RegionCost(estimatedWork, 0L, 0L, 0L);
    }
}
