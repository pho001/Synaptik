package graph.compile.planning.memory;

import tensor.DataType;
import tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

final class MemoryPlanSummaryBuilder {
    private MemoryPlanSummaryBuilder() {
    }

    static MemoryPlanSummary build(
            List<Tensor> sortedGraph,
            Map<Tensor, NodeLifetime> lifetimes,
            Map<Tensor, ReusableInterval> reusableIntervals,
            Map<Tensor, Integer> slotByOwner,
            Map<Integer, Integer> slotSizes,
            int forwardBoundaryIndex
    ) {
        int savedForwardCount = 0;
        int gradientTargetCount = 0;
        long savedForwardHoldDistanceSum = 0L;
        for (Map.Entry<Tensor, NodeLifetime> entry : lifetimes.entrySet()) {
            if (entry.getValue().storageOwner() != entry.getKey()) {
                continue;
            }
            if (entry.getValue().role() == MemoryRole.SAVED_FORWARD) {
                savedForwardCount++;
                savedForwardHoldDistanceSum += (long) entry.getValue().lastReadIndex() - entry.getValue().birthIndex();
            }
            if (entry.getValue().role() == MemoryRole.GRADIENT_TARGET) {
                gradientTargetCount++;
            }
        }

        long peakTotal = 0L;
        long peakReusable = 0L;
        long peakSavedForward = 0L;
        long peakGradientTarget = 0L;
        long peakForward = 0L;
        long peakBackward = 0L;
        for (int i = 0; i < sortedGraph.size(); i++) {
            long activeReusableBytes = 0L;
            long activeNonReusableSavedForwardBytes = 0L;
            long activeSavedForwardBytes = 0L;
            long activeGradientTargetBytes = 0L;
            long activeForwardBytes = 0L;
            long activeBackwardBytes = 0L;

            Set<Integer> countedSlots = new java.util.HashSet<>();
            Set<Tensor> reusableOwners = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
            for (Map.Entry<Tensor, ReusableInterval> entry : reusableIntervals.entrySet()) {
                ReusableInterval interval = entry.getValue();
                if (interval.birthIndex() <= i && i <= interval.lastReadIndex()) {
                    Integer slotId = slotByOwner.get(entry.getKey());
                    if (slotId != null && countedSlots.add(slotId)) {
                        long size = slotSizes.get(slotId);
                        activeReusableBytes += size;
                        reusableOwners.add(entry.getKey());
                        if (interval.birthIndex() <= forwardBoundaryIndex) {
                            activeForwardBytes += size;
                        } else {
                            activeBackwardBytes += size;
                        }
                    }
                }
            }

            Set<Tensor> countedOwners = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
            for (Map.Entry<Tensor, NodeLifetime> entry : lifetimes.entrySet()) {
                Tensor tensor = entry.getKey();
                NodeLifetime lifetime = entry.getValue();
                if (lifetime.storageOwner() != tensor) {
                    continue;
                }
                if (lifetime.birthIndex() > i || i > lifetime.lastReadIndex()) {
                    continue;
                }
                if (!countedOwners.add(tensor)) {
                    continue;
                }
                long bytes = bytesOf(tensor);
                if (lifetime.role() == MemoryRole.SAVED_FORWARD) {
                    activeSavedForwardBytes += bytes;
                    if (!reusableOwners.contains(tensor)) {
                        activeNonReusableSavedForwardBytes += bytes;
                        activeForwardBytes += bytes;
                    }
                } else if (lifetime.role() == MemoryRole.GRADIENT_TARGET) {
                    activeGradientTargetBytes += bytes;
                    activeBackwardBytes += bytes;
                }
            }

            long activeBytes = activeReusableBytes + activeNonReusableSavedForwardBytes + activeGradientTargetBytes;

            peakTotal = Math.max(peakTotal, activeBytes);
            peakReusable = Math.max(peakReusable, activeReusableBytes);
            peakSavedForward = Math.max(peakSavedForward, activeSavedForwardBytes);
            peakGradientTarget = Math.max(peakGradientTarget, activeGradientTargetBytes);
            peakForward = Math.max(peakForward, activeForwardBytes);
            peakBackward = Math.max(peakBackward, activeBackwardBytes);
        }

        int intervalCount = reusableIntervals.size();
        int slotCount = slotSizes.size();
        int reuseCount = Math.max(0, intervalCount - slotCount);
        int reusableFreshAllocationCount = slotCount;
        double reuseHitRate = intervalCount == 0 ? 0.0d : ((double) reuseCount / intervalCount);
        long allocatedSlotBytes = slotSizes.values().stream().mapToLong(Integer::longValue).sum();
        double averageSavedForwardHoldDistance = savedForwardCount == 0
                ? 0.0
                : ((double) savedForwardHoldDistanceSum / savedForwardCount);

        return new MemoryPlanSummary(
                intervalCount,
                slotCount,
                reuseCount,
                reusableFreshAllocationCount,
                reuseHitRate,
                allocatedSlotBytes,
                peakTotal,
                peakReusable,
                peakSavedForward,
                peakGradientTarget,
                peakForward,
                peakBackward,
                savedForwardCount,
                gradientTargetCount,
                averageSavedForwardHoldDistance
        );
    }

    private static long bytesOf(Tensor tensor) {
        return (long) tensor.getFlatDataSize() * bytesPerElement(tensor.getDataType());
    }

    private static int bytesPerElement(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> 8;
            case FLOAT32 -> 4;
            case BFLOAT16 -> 2;
            case INT32 -> 4;
            case INT64 -> 8;
            case BOOL -> 1;
        };
    }
}
