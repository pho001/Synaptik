package graph.optimizer.memory;

public record MemoryPlanSummary(
        int reusableIntervalCount,
        int slotCount,
        int reuseCount,
        long peakLiveBytes,
        long peakReusableBytes,
        long peakSavedForwardBytes,
        long peakGradientTargetBytes,
        long peakForwardLiveBytes,
        long peakBackwardLiveBytes,
        int savedForwardCount,
        int gradientTargetCount,
        double averageSavedForwardHoldDistance
) {
}
