package graph.optimizer.memory;

import tensor.Tensor;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class MemoryPlan {
    private final Map<Tensor, NodeLifetime> lifetimes;
    private final Map<Tensor, ReusableInterval> reusableIntervals;
    private final Map<Tensor, Integer> slotByOwner;
    private final Map<Integer, Integer> slotSizes;
    private final MemoryPlannerPolicy policy;
    private final MemoryPlanSummary summary;

    public MemoryPlan(
            Map<Tensor, NodeLifetime> lifetimes,
            Map<Tensor, ReusableInterval> reusableIntervals,
            Map<Tensor, Integer> slotByOwner,
            Map<Integer, Integer> slotSizes,
            MemoryPlannerPolicy policy,
            MemoryPlanSummary summary
    ) {
        this.lifetimes = Map.copyOf(Objects.requireNonNull(lifetimes, "lifetimes cannot be null"));
        this.reusableIntervals = Map.copyOf(Objects.requireNonNull(reusableIntervals, "reusableIntervals cannot be null"));
        this.slotByOwner = Map.copyOf(Objects.requireNonNull(slotByOwner, "slotByOwner cannot be null"));
        this.slotSizes = Map.copyOf(Objects.requireNonNull(slotSizes, "slotSizes cannot be null"));
        this.policy = Objects.requireNonNull(policy, "policy cannot be null");
        this.summary = Objects.requireNonNull(summary, "summary cannot be null");
    }

    public NodeLifetime lifetimeOf(Tensor tensor) {
        NodeLifetime lifetime = lifetimes.get(tensor);
        if (lifetime == null) {
            throw new IllegalArgumentException("Missing lifetime for tensor: " + tensor.getLabel());
        }
        return lifetime;
    }

    public Tensor storageOwnerOf(Tensor tensor) {
        return lifetimeOf(tensor).storageOwner();
    }

    public int lastReadIndexOf(Tensor tensor) {
        return lifetimeOf(tensor).lastReadIndex();
    }

    public boolean isReusableOwner(Tensor tensor) {
        NodeLifetime lifetime = lifetimeOf(tensor);
        if (lifetime.storageOwner() != tensor) {
            return false;
        }
        return reusableIntervals.containsKey(tensor);
    }

    public Integer slotIdOf(Tensor tensor) {
        Tensor owner = storageOwnerOf(tensor);
        return slotByOwner.get(owner);
    }

    public int slotSize(int slotId) {
        Integer size = slotSizes.get(slotId);
        if (size == null) {
            throw new IllegalArgumentException("Missing slot size for slot " + slotId);
        }
        return size;
    }

    public MemoryPlannerPolicy policy() {
        return policy;
    }

    public MemoryPlanSummary summary() {
        return summary;
    }

    public Map<Tensor, Integer> slotByOwner() {
        return slotByOwner;
    }

    public Map<Integer, Integer> slotSizes() {
        return slotSizes;
    }

    public String explain() {
        StringBuilder sb = new StringBuilder();
        sb.append("MemoryPlanSummary{")
                .append("reusableIntervals=").append(summary.reusableIntervalCount())
                .append(", slots=").append(summary.slotCount())
                .append(", reuseCount=").append(summary.reuseCount())
                .append(", peakLiveBytes=").append(summary.peakLiveBytes())
                .append(", peakForwardLiveBytes=").append(summary.peakForwardLiveBytes())
                .append(", peakBackwardLiveBytes=").append(summary.peakBackwardLiveBytes())
                .append(", savedForwardCount=").append(summary.savedForwardCount())
                .append(", averageSavedForwardHoldDistance=").append(summary.averageSavedForwardHoldDistance())
                .append(", policy=").append(policy)
                .append("}\n");

        List<Map.Entry<Tensor, NodeLifetime>> entries = new ArrayList<>(lifetimes.entrySet());
        entries.sort(Comparator.comparingInt(e -> e.getValue().birthIndex()));
        for (Map.Entry<Tensor, NodeLifetime> entry : entries) {
            Tensor tensor = entry.getKey();
            NodeLifetime lifetime = entry.getValue();
            Integer slotId = slotByOwner.get(lifetime.storageOwner());
            sb.append("[")
                    .append(lifetime.birthIndex())
                    .append("] label=").append(tensor.getLabel())
                    .append(", op=").append(tensor.getOperation() == null ? "LEAF" : tensor.getOperation().opType())
                    .append(", role=").append(lifetime.role())
                    .append(", owner=").append(lifetime.storageOwner().getLabel())
                    .append(", lastRead=").append(lifetime.lastReadIndex())
                    .append(", reusable=").append(reusableIntervals.containsKey(lifetime.storageOwner()))
                    .append(", slot=").append(slotId == null ? "-" : slotId)
                    .append("\n");
        }
        return sb.toString();
    }
}
