package graph.optimizer.memory;

public record MemoryPlanSummary(
        int reusableIntervalCount,
        int slotCount,
        int reuseCount,
        long peakLiveBytes,
        long peakForwardLiveBytes,
        long peakBackwardLiveBytes,
        int savedForwardCount,
        double averageSavedForwardHoldDistance
) {
}
