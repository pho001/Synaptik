package graph.compile.planning.memory;

import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

final class ReusableSlotAllocator {
    private ReusableSlotAllocator() {
    }

    static ReusableSlotAssignment allocate(
            List<ReusableInterval> intervals,
            int forwardBoundaryIndex,
            MemoryPlannerPolicy policy
    ) {
        if (intervals.isEmpty()) {
            return new ReusableSlotAssignment(Map.of(), Map.of());
        }

        List<ReusableInterval> sorted = new ArrayList<>(intervals);
        sorted.sort(Comparator
                .comparingInt(ReusableInterval::birthIndex)
                .thenComparingInt(ReusableInterval::lastReadIndex));

        Map<Tensor, Integer> slotByOwner = new IdentityHashMap<>();
        Map<Integer, Integer> slotSizes = new HashMap<>();
        List<SlotState> active = new ArrayList<>();
        List<SlotState> free = new ArrayList<>();
        int nextSlotId = 0;

        for (ReusableInterval interval : sorted) {
            releaseExpired(active, free, interval.birthIndex());

            SlotState chosen = chooseSlot(free, interval, forwardBoundaryIndex, policy);
            if (chosen == null) {
                chosen = new SlotState(nextSlotId++, interval.size(), interval.dataType(), phaseOf(interval, forwardBoundaryIndex), Integer.MIN_VALUE);
                slotSizes.put(chosen.slotId, chosen.size);
            } else {
                free.remove(chosen);
            }

            if (interval.role() == MemoryRole.SAVED_FORWARD) {
                chosen.phase = "shared";
            }
            chosen.lastReadIndex = interval.lastReadIndex();
            active.add(chosen);
            slotByOwner.put(interval.owner(), chosen.slotId);
        }

        return new ReusableSlotAssignment(Map.copyOf(slotByOwner), Map.copyOf(slotSizes));
    }

    private static void releaseExpired(List<SlotState> active, List<SlotState> free, int currentBirth) {
        List<SlotState> released = new ArrayList<>();
        for (SlotState state : active) {
            if (state.lastReadIndex < currentBirth) {
                released.add(state);
            }
        }
        active.removeAll(released);
        free.addAll(released);
    }

    private static SlotState chooseSlot(
            List<SlotState> free,
            ReusableInterval interval,
            int forwardBoundaryIndex,
            MemoryPlannerPolicy policy
    ) {
        SlotState best = null;
        String intervalPhase = phaseOf(interval, forwardBoundaryIndex);
        for (SlotState state : free) {
            if (state.dataType != interval.dataType()) {
                continue;
            }
            if (policy.separateForwardBackwardPools() && !isPhaseCompatible(state.phase, intervalPhase)) {
                continue;
            }
            if (!policy.allowLargerBufferReuse() && state.size != interval.size()) {
                continue;
            }
            if (policy.allowLargerBufferReuse() && state.size < interval.size()) {
                continue;
            }
            if (best == null || state.size < best.size) {
                best = state;
            }
        }
        return best;
    }

    private static String phaseOf(ReusableInterval interval, int forwardBoundaryIndex) {
        if (interval.role() == MemoryRole.SAVED_FORWARD) {
            return "shared";
        }
        return interval.birthIndex() <= forwardBoundaryIndex ? "forward" : "backward";
    }

    private static boolean isPhaseCompatible(String slotPhase, String intervalPhase) {
        if (slotPhase.equals(intervalPhase)) {
            return true;
        }
        return "shared".equals(slotPhase) || "shared".equals(intervalPhase);
    }

    private static final class SlotState {
        private final int slotId;
        private final int size;
        private final DataType dataType;
        private String phase;
        private int lastReadIndex;

        private SlotState(int slotId, int size, DataType dataType, String phase, int lastReadIndex) {
            this.slotId = slotId;
            this.size = size;
            this.dataType = dataType;
            this.phase = phase;
            this.lastReadIndex = lastReadIndex;
        }
    }
}
