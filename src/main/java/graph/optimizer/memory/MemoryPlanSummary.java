package graph.optimizer.memory;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Collections;

/**
 * Aggregate memory planning metrics.
 *
 * @param reusableIntervalCount number of reusable intervals discovered
 * @param slotCount number of tensor-level slots allocated
 * @param reuseCount number of intervals placed into existing slots
 * @param reusableFreshAllocationCount number of reusable intervals that required fresh slots
 * @param reuseHitRate ratio of reused intervals to reusable intervals
 * @param allocatedSlotBytes total bytes represented by allocated slots
 * @param peakLiveBytes peak live bytes across the graph
 * @param peakReusableBytes peak bytes eligible for reuse
 * @param peakSavedForwardBytes peak bytes held for backward
 * @param peakGradientTargetBytes peak bytes held by gradient targets
 * @param peakForwardLiveBytes peak live bytes in forward phase
 * @param peakBackwardLiveBytes peak live bytes in backward phase
 * @param savedForwardCount number of saved forward values
 * @param gradientTargetCount number of gradient target values
 * @param averageSavedForwardHoldDistance average graph distance saved forward values are retained
 */
public record MemoryPlanSummary(
        int reusableIntervalCount,
        int slotCount,
        int reuseCount,
        int reusableFreshAllocationCount,
        double reuseHitRate,
        long allocatedSlotBytes,
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
    /**
     * Returns summary values as stable metric keys.
     *
     * @return immutable metric map
     */
    public Map<String, Object> toMetricMap() {
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("reusableIntervalCount", reusableIntervalCount);
        metrics.put("slotCount", slotCount);
        metrics.put("reuseCount", reuseCount);
        metrics.put("reusableFreshAllocationCount", reusableFreshAllocationCount);
        metrics.put("reuseHitRate", reuseHitRate);
        metrics.put("allocatedSlotBytes", allocatedSlotBytes);
        metrics.put("peakLiveBytes", peakLiveBytes);
        metrics.put("peakReusableBytes", peakReusableBytes);
        metrics.put("peakSavedForwardBytes", peakSavedForwardBytes);
        metrics.put("peakGradientTargetBytes", peakGradientTargetBytes);
        metrics.put("peakForwardLiveBytes", peakForwardLiveBytes);
        metrics.put("peakBackwardLiveBytes", peakBackwardLiveBytes);
        metrics.put("savedForwardCount", savedForwardCount);
        metrics.put("gradientTargetCount", gradientTargetCount);
        metrics.put("averageSavedForwardHoldDistance", averageSavedForwardHoldDistance);
        return Collections.unmodifiableMap(metrics);
    }
}
