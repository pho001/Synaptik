package config.profile;

/**
 * Calibrated CPU scheduling and chunking policy.
 *
 * <p>The scheduler profile controls how much work is grouped per worker for low-, medium-, and
 * high-cost kernels, plus minimum chunk sizes for scalar, vector, and reduction work. These values are
 * measured per platform because too-small chunks can make parallel execution slower than scalar
 * execution.</p>
 *
 * @param lowCostTargetChunksPerWorker target chunks per worker for cheap kernels
 * @param mediumCostTargetChunksPerWorker target chunks per worker for medium-cost kernels
 * @param highCostTargetChunksPerWorker target chunks per worker for expensive kernels
 * @param minScalarChunkSize minimum scalar chunk size
 * @param minVectorChunkSize minimum vector chunk size
 * @param minReductionChunkSize minimum reduction chunk size
 * @param commonPoolLowCostMaxWorkPerWorker maximum low-cost work per common-pool worker
 */
public record SchedulerPlatformProfile(
        int lowCostTargetChunksPerWorker,
        int mediumCostTargetChunksPerWorker,
        int highCostTargetChunksPerWorker,
        int minScalarChunkSize,
        int minVectorChunkSize,
        int minReductionChunkSize,
        int commonPoolLowCostMaxWorkPerWorker
) {}
