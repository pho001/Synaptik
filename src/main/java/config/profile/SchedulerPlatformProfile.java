package config.profile;

public record SchedulerPlatformProfile(
        int lowCostTargetChunksPerWorker,
        int mediumCostTargetChunksPerWorker,
        int highCostTargetChunksPerWorker,
        int minScalarChunkSize,
        int minVectorChunkSize,
        int minReductionChunkSize,
        int commonPoolLowCostMaxWorkPerWorker
) {}
