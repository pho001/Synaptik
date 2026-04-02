package graph.optimizer.memory;

import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public final class MemoryPlanner {
    private MemoryPlanner() {
    }

    public static MemoryPlan plan(List<Tensor> sortedGraph) {
        return plan(sortedGraph, MemoryPlannerPolicy.defaults());
    }

    public static MemoryPlan plan(List<Tensor> sortedGraph, MemoryPlannerPolicy policy) {
        Objects.requireNonNull(policy, "policy cannot be null");
        if (sortedGraph == null || sortedGraph.isEmpty()) {
            return new MemoryPlan(Map.of(), Map.of(), Map.of(), Map.of(), policy,
                    new MemoryPlanSummary(0, 0, 0, 0L, 0L, 0L, 0L, 0L, 0L, 0, 0, 0.0));
        }

        Map<Tensor, Integer> indexByTensor = new IdentityHashMap<>();
        for (int i = 0; i < sortedGraph.size(); i++) {
            indexByTensor.put(sortedGraph.get(i), i);
        }
        int forwardBoundaryIndex = resolveForwardBoundaryIndex(sortedGraph);

        Map<Tensor, Tensor> storageOwnerByTensor = new IdentityHashMap<>();
        for (Tensor tensor : sortedGraph) {
            storageOwnerByTensor.put(tensor, resolveStorageOwner(tensor, storageOwnerByTensor));
        }

        Set<Tensor> gradientTargets = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Tensor tensor : sortedGraph) {
            if (tensor.getGradient() != null) {
                gradientTargets.add(tensor.getGradient());
            }
        }

        Map<Tensor, Integer> lastReadIndexByOwner = new IdentityHashMap<>();
        Map<Tensor, Integer> consumerCountsByOwner = new IdentityHashMap<>();
        Set<Tensor> savedForwardOwners = Collections.newSetFromMap(new IdentityHashMap<>());

        for (Tensor tensor : sortedGraph) {
            Tensor owner = storageOwnerByTensor.get(tensor);
            lastReadIndexByOwner.putIfAbsent(owner, -1);
            consumerCountsByOwner.putIfAbsent(owner, 0);
        }

        for (int i = 0; i < sortedGraph.size(); i++) {
            Tensor consumer = sortedGraph.get(i);
            List<Tensor> inputs = consumer.getPrevTensors();
            if (inputs == null || inputs.isEmpty()) {
                continue;
            }
            for (Tensor input : inputs) {
                Tensor owner = storageOwnerByTensor.get(input);
                if (owner == null) {
                    continue;
                }
                consumerCountsByOwner.merge(owner, 1, Integer::sum);
                lastReadIndexByOwner.merge(owner, i, Math::max);
                if (indexByTensor.get(owner) <= forwardBoundaryIndex && i > forwardBoundaryIndex) {
                    savedForwardOwners.add(owner);
                }
            }
        }

        for (Tensor owner : new ArrayList<>(lastReadIndexByOwner.keySet())) {
            if (consumerCountsByOwner.getOrDefault(owner, 0) == 0) {
                lastReadIndexByOwner.put(owner, Integer.MAX_VALUE);
            }
        }

        Map<Tensor, NodeLifetime> lifetimes = new IdentityHashMap<>();
        for (Tensor tensor : sortedGraph) {
            Tensor owner = storageOwnerByTensor.get(tensor);
            MemoryRole role = roleOf(tensor, owner, gradientTargets, savedForwardOwners);
            int birthIndex = indexByTensor.get(tensor);
            int lastReadIndex = lastReadIndexByOwner.getOrDefault(owner, Integer.MAX_VALUE);
            lifetimes.put(tensor, new NodeLifetime(birthIndex, lastReadIndex, role, owner));
        }

        Map<Tensor, ReusableInterval> reusableIntervals = buildReusableIntervals(sortedGraph, lifetimes, policy);
        SlotAssignment assignment = assignSlots(reusableIntervals.values().stream().toList(), forwardBoundaryIndex, policy);
        MemoryPlanSummary summary = buildSummary(sortedGraph, lifetimes, reusableIntervals, assignment.slotByOwner(), assignment.slotSizes(), forwardBoundaryIndex);

        return new MemoryPlan(lifetimes, reusableIntervals, assignment.slotByOwner(), assignment.slotSizes(), policy, summary);
    }

    private static Map<Tensor, ReusableInterval> buildReusableIntervals(
            List<Tensor> sortedGraph,
            Map<Tensor, NodeLifetime> lifetimes,
            MemoryPlannerPolicy policy
    ) {
        Map<Tensor, ReusableInterval> out = new IdentityHashMap<>();
        for (Tensor tensor : sortedGraph) {
            NodeLifetime lifetime = lifetimes.get(tensor);
            if (lifetime.storageOwner() != tensor) {
                continue;
            }
            if (lifetime.role() != MemoryRole.FORWARD_TEMP
                    && lifetime.role() != MemoryRole.BACKWARD_TEMP
                    && lifetime.role() != MemoryRole.SAVED_FORWARD) {
                continue;
            }
            int size = tensor.getFlatDataSize();
            if (size < policy.minReusableBufferSize()) {
                continue;
            }
            out.put(tensor, new ReusableInterval(
                    tensor,
                    lifetime.birthIndex(),
                    lifetime.lastReadIndex(),
                    size,
                    tensor.getDataType(),
                    lifetime.role()
            ));
        }
        return out;
    }

    private static SlotAssignment assignSlots(
            List<ReusableInterval> intervals,
            int forwardBoundaryIndex,
            MemoryPlannerPolicy policy
    ) {
        if (intervals.isEmpty()) {
            return new SlotAssignment(Map.of(), Map.of());
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

        return new SlotAssignment(Map.copyOf(slotByOwner), Map.copyOf(slotSizes));
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

    private static MemoryPlanSummary buildSummary(
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
            long activeSavedForwardBytes = 0L;
            long activeGradientTargetBytes = 0L;
            long activeForwardBytes = 0L;
            long activeBackwardBytes = 0L;

            Set<Integer> countedSlots = new java.util.HashSet<>();
            for (Map.Entry<Tensor, ReusableInterval> entry : reusableIntervals.entrySet()) {
                ReusableInterval interval = entry.getValue();
                if (interval.birthIndex() <= i && i <= interval.lastReadIndex()) {
                    Integer slotId = slotByOwner.get(entry.getKey());
                    if (slotId != null && countedSlots.add(slotId)) {
                        long size = slotSizes.get(slotId);
                        activeReusableBytes += size;
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
                    activeForwardBytes += bytes;
                } else if (lifetime.role() == MemoryRole.GRADIENT_TARGET) {
                    activeGradientTargetBytes += bytes;
                    activeBackwardBytes += bytes;
                }
            }

            long activeBytes = activeReusableBytes + activeSavedForwardBytes + activeGradientTargetBytes;

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
        double averageSavedForwardHoldDistance = savedForwardCount == 0
                ? 0.0
                : ((double) savedForwardHoldDistanceSum / savedForwardCount);

        return new MemoryPlanSummary(
                intervalCount,
                slotCount,
                reuseCount,
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
            case FLOAT16 -> 2;
        };
    }

    private static Tensor resolveStorageOwner(Tensor tensor, Map<Tensor, Tensor> storageOwnerByTensor) {
        if (!aliasesInput0AtRuntime(tensor)) {
            return tensor;
        }
        Tensor input0 = tensor.getPrevTensors().get(0);
        return storageOwnerByTensor.getOrDefault(input0, input0);
    }

    private static MemoryRole roleOf(
            Tensor tensor,
            Tensor owner,
            Set<Tensor> gradientTargets,
            Set<Tensor> savedForwardOwners
    ) {
        if (aliasesInput0AtRuntime(tensor)) {
            return MemoryRole.VIEW_ALIAS;
        }
        if (tensor.getOperation() == null) {
            return MemoryRole.LEAF;
        }
        if (gradientTargets.contains(tensor)) {
            return MemoryRole.GRADIENT_TARGET;
        }
        if (savedForwardOwners.contains(owner)) {
            return MemoryRole.SAVED_FORWARD;
        }
        if (tensor.isBackward()) {
            return MemoryRole.BACKWARD_TEMP;
        }
        return MemoryRole.FORWARD_TEMP;
    }

    private static int resolveForwardBoundaryIndex(List<Tensor> sortedGraph) {
        for (int i = sortedGraph.size() - 1; i >= 0; i--) {
            Tensor tensor = sortedGraph.get(i);
            if (tensor.getOperation() != null
                    && tensor.getOperation().opType() == Operation.OpType.NOOP
                    && Tensor.SYSTEM_FORWARD_OUTPUT_LABEL.equals(tensor.getLabel())) {
                return i;
            }
        }
        return sortedGraph.size() - 1;
    }

    private static boolean aliasesInput0AtRuntime(Tensor tensor) {
        if (tensor == null || tensor.getOperation() == null) {
            return false;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.isEmpty()) {
            return false;
        }
        return switch (tensor.getOperation().opType()) {
            case NOOP, PERMUTE -> true;
            case RESHAPE, EXPAND_DIMS, SQUEEZE -> inputs.get(0).isContiguous();
            default -> false;
        };
    }

    private record SlotAssignment(
            Map<Tensor, Integer> slotByOwner,
            Map<Integer, Integer> slotSizes
    ) {
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
