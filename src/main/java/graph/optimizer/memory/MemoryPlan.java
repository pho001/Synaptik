package graph.optimizer.memory;

import tensor.Tensor;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

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
        appendSummary(sb);
        appendSlots(sb);
        appendNodes(sb);
        appendSavedForward(sb);
        return sb.toString();
    }

    private void appendSummary(StringBuilder sb) {
        sb.append("=== MemoryPlan Summary ===\n");
        sb.append("reusableIntervals=").append(summary.reusableIntervalCount()).append('\n');
        sb.append("slotCount=").append(summary.slotCount()).append('\n');
        sb.append("reuseCount=").append(summary.reuseCount()).append('\n');
        sb.append("peakLiveBytes=").append(summary.peakLiveBytes()).append('\n');
        sb.append("peakReusableBytes=").append(summary.peakReusableBytes()).append('\n');
        sb.append("peakSavedForwardBytes=").append(summary.peakSavedForwardBytes()).append('\n');
        sb.append("peakGradientTargetBytes=").append(summary.peakGradientTargetBytes()).append('\n');
        sb.append("peakForwardLiveBytes=").append(summary.peakForwardLiveBytes()).append('\n');
        sb.append("peakBackwardLiveBytes=").append(summary.peakBackwardLiveBytes()).append('\n');
        sb.append("savedForwardCount=").append(summary.savedForwardCount()).append('\n');
        sb.append("gradientTargetCount=").append(summary.gradientTargetCount()).append('\n');
        sb.append("averageSavedForwardHoldDistance=").append(summary.averageSavedForwardHoldDistance()).append('\n');
        sb.append("policy=").append(policy).append("\n\n");
    }

    private void appendSlots(StringBuilder sb) {
        sb.append("=== Slot Assignment ===\n");
        Map<Integer, List<ReusableInterval>> bySlot = new TreeMap<>();
        for (Map.Entry<Tensor, ReusableInterval> entry : reusableIntervals.entrySet()) {
            Integer slotId = slotByOwner.get(entry.getKey());
            if (slotId == null) {
                continue;
            }
            bySlot.computeIfAbsent(slotId, ignored -> new ArrayList<>()).add(entry.getValue());
        }
        for (Map.Entry<Integer, List<ReusableInterval>> entry : bySlot.entrySet()) {
            int slotId = entry.getKey();
            sb.append("slot ").append(slotId)
                    .append(" size=").append(slotSizes.get(slotId))
                    .append('\n');
            entry.getValue().stream()
                    .sorted(Comparator.comparingInt(ReusableInterval::birthIndex))
                    .forEach(interval -> sb.append("  - ")
                            .append(interval.owner().getLabel())
                            .append(" role=").append(interval.role())
                            .append(" [").append(interval.birthIndex()).append(", ").append(interval.lastReadIndex()).append("]")
                            .append('\n'));
        }
        if (bySlot.isEmpty()) {
            sb.append("(no reusable slots)\n");
        }
        sb.append('\n');
    }

    private void appendNodes(StringBuilder sb) {
        sb.append("=== Node Assignment ===\n");
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
        sb.append('\n');
    }

    private void appendSavedForward(StringBuilder sb) {
        sb.append("=== Saved Forward Values ===\n");
        List<Map.Entry<Tensor, NodeLifetime>> entries = new ArrayList<>(lifetimes.entrySet());
        entries.sort(Comparator.comparingInt(e -> e.getValue().birthIndex()));
        boolean any = false;
        for (Map.Entry<Tensor, NodeLifetime> entry : entries) {
            Tensor tensor = entry.getKey();
            NodeLifetime lifetime = entry.getValue();
            if (lifetime.storageOwner() != tensor || lifetime.role() != MemoryRole.SAVED_FORWARD) {
                continue;
            }
            any = true;
            sb.append("- ")
                    .append(tensor.getLabel())
                    .append(" [").append(lifetime.birthIndex()).append(", ").append(lifetime.lastReadIndex()).append("]")
                    .append(" slot=").append(slotByOwner.get(tensor))
                    .append('\n');
        }
        if (!any) {
            sb.append("(none)\n");
        }
    }
}
