package graph.optimizer.memory;

import graph.optimizer.region.RegionValueRef;
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
    private final StructuralMemoryView structuralView;
    private final Map<RegionValueRef, RegionValueLifetime> regionValueLifetimes;
    private final Map<RegionValueRef, MaterializationPlanEntry> materializationPlan;
    private final Map<RegionValueRef, RegionMemoryBinding> regionMemoryBindings;
    private final Map<RegionValueRef, Integer> regionSlotByValueRef;
    private final Map<Integer, Integer> regionSlotUseCounts;
    private final Map<Integer, Integer> regionSlotSizes;
    private final Map<Tensor, RegionValueRef> tensorToRegionValueRef;
    private final List<RegionHandoffRequirement> handoffRequirements;
    private final Map<Tensor, RuntimeMemoryBindingPolicy> runtimeBindingPolicies;

    public MemoryPlan(
            Map<Tensor, NodeLifetime> lifetimes,
            Map<Tensor, ReusableInterval> reusableIntervals,
            Map<Tensor, Integer> slotByOwner,
            Map<Integer, Integer> slotSizes,
            MemoryPlannerPolicy policy,
            MemoryPlanSummary summary,
            StructuralMemoryView structuralView,
            Map<RegionValueRef, RegionValueLifetime> regionValueLifetimes,
            Map<RegionValueRef, MaterializationPlanEntry> materializationPlan,
            Map<RegionValueRef, RegionMemoryBinding> regionMemoryBindings,
            Map<RegionValueRef, Integer> regionSlotByValueRef,
            Map<Integer, Integer> regionSlotSizes,
            Map<Tensor, RegionValueRef> tensorToRegionValueRef,
            List<RegionHandoffRequirement> handoffRequirements,
            Map<Tensor, RuntimeMemoryBindingPolicy> runtimeBindingPolicies
    ) {
        this.lifetimes = Map.copyOf(Objects.requireNonNull(lifetimes, "lifetimes cannot be null"));
        this.reusableIntervals = Map.copyOf(Objects.requireNonNull(reusableIntervals, "reusableIntervals cannot be null"));
        this.slotByOwner = Map.copyOf(Objects.requireNonNull(slotByOwner, "slotByOwner cannot be null"));
        this.slotSizes = Map.copyOf(Objects.requireNonNull(slotSizes, "slotSizes cannot be null"));
        this.policy = Objects.requireNonNull(policy, "policy cannot be null");
        this.summary = Objects.requireNonNull(summary, "summary cannot be null");
        this.structuralView = structuralView == null ? StructuralMemoryView.empty() : structuralView;
        this.regionValueLifetimes = Map.copyOf(Objects.requireNonNull(regionValueLifetimes, "regionValueLifetimes cannot be null"));
        this.materializationPlan = Map.copyOf(Objects.requireNonNull(materializationPlan, "materializationPlan cannot be null"));
        this.regionMemoryBindings = Map.copyOf(Objects.requireNonNull(regionMemoryBindings, "regionMemoryBindings cannot be null"));
        this.regionSlotByValueRef = Map.copyOf(Objects.requireNonNull(regionSlotByValueRef, "regionSlotByValueRef cannot be null"));
        this.regionSlotUseCounts = Map.copyOf(buildRegionSlotUseCounts(this.regionSlotByValueRef));
        this.regionSlotSizes = Map.copyOf(Objects.requireNonNull(regionSlotSizes, "regionSlotSizes cannot be null"));
        this.tensorToRegionValueRef = Map.copyOf(Objects.requireNonNull(tensorToRegionValueRef, "tensorToRegionValueRef cannot be null"));
        this.handoffRequirements = List.copyOf(Objects.requireNonNull(handoffRequirements, "handoffRequirements cannot be null"));
        this.runtimeBindingPolicies = Map.copyOf(Objects.requireNonNull(runtimeBindingPolicies, "runtimeBindingPolicies cannot be null"));
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

    public StructuralMemoryView structuralView() {
        return structuralView;
    }

    public RegionValueLifetime regionValueLifetimeOf(RegionValueRef valueRef) {
        RegionValueLifetime lifetime = regionValueLifetimes.get(valueRef);
        if (lifetime == null) {
            throw new IllegalArgumentException("Missing region value lifetime for: " + valueRef.valueId());
        }
        return lifetime;
    }

    public MaterializationPlanEntry materializationPlanOf(RegionValueRef valueRef) {
        MaterializationPlanEntry entry = materializationPlan.get(valueRef);
        if (entry == null) {
            throw new IllegalArgumentException("Missing materialization plan entry for: " + valueRef.valueId());
        }
        return entry;
    }

    public RegionMemoryBinding regionMemoryBindingOf(RegionValueRef valueRef) {
        RegionMemoryBinding binding = regionMemoryBindings.get(valueRef);
        if (binding == null) {
            throw new IllegalArgumentException("Missing region memory binding for: " + valueRef.valueId());
        }
        return binding;
    }

    public Map<RegionValueRef, RegionValueLifetime> regionValueLifetimes() {
        return regionValueLifetimes;
    }

    public Map<RegionValueRef, MaterializationPlanEntry> materializationPlan() {
        return materializationPlan;
    }

    public Map<RegionValueRef, RegionMemoryBinding> regionMemoryBindings() {
        return regionMemoryBindings;
    }

    public Integer regionSlotIdOf(RegionValueRef valueRef) {
        return regionSlotByValueRef.get(valueRef);
    }

    public int regionSlotUseCount(int slotId) {
        return regionSlotUseCounts.getOrDefault(slotId, 0);
    }

    public int regionSlotSize(int slotId) {
        Integer size = regionSlotSizes.get(slotId);
        if (size == null) {
            throw new IllegalArgumentException("Missing region slot size for slot " + slotId);
        }
        return size;
    }

    public Map<RegionValueRef, Integer> regionSlotByValueRef() {
        return regionSlotByValueRef;
    }

    public Map<Integer, Integer> regionSlotSizes() {
        return regionSlotSizes;
    }

    private static Map<Integer, Integer> buildRegionSlotUseCounts(Map<RegionValueRef, Integer> slotByValueRef) {
        TreeMap<Integer, Integer> counts = new TreeMap<>();
        for (Integer slotId : slotByValueRef.values()) {
            if (slotId != null) {
                counts.merge(slotId, 1, Integer::sum);
            }
        }
        return counts;
    }

    public RegionValueRef regionValueRefOf(Tensor tensor) {
        return tensorToRegionValueRef.get(tensor);
    }

    public Integer runtimeSlotIdOf(Tensor tensor) {
        RegionValueRef regionValueRef = regionValueRefOf(tensor);
        if (regionValueRef != null) {
            Integer regionSlotId = regionSlotIdOf(regionValueRef);
            if (regionSlotId != null) {
                return regionSlotId;
            }
        }
        return slotIdOf(tensor);
    }

    public int runtimeSlotSizeOf(Tensor tensor) {
        Integer runtimeSlotId = runtimeSlotIdOf(tensor);
        if (runtimeSlotId == null) {
            throw new IllegalArgumentException("Missing runtime slot for tensor: " + tensor.getLabel());
        }
        RegionValueRef regionValueRef = regionValueRefOf(tensor);
        if (regionValueRef != null && regionSlotIdOf(regionValueRef) != null && regionSlotIdOf(regionValueRef).equals(runtimeSlotId)) {
            return regionSlotSize(runtimeSlotId);
        }
        return slotSize(runtimeSlotId);
    }

    public List<RegionHandoffRequirement> handoffRequirements() {
        return handoffRequirements;
    }

    public RuntimeMemoryBindingPolicy runtimeBindingPolicyOf(Tensor tensor) {
        return runtimeBindingPolicies.getOrDefault(tensor, RuntimeMemoryBindingPolicy.REGION_BINDING_ALLOWED);
    }

    public Map<Tensor, RuntimeMemoryBindingPolicy> runtimeBindingPolicies() {
        return runtimeBindingPolicies;
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
        appendStructuralView(sb);
        appendRegionValuePlan(sb);
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
        sb.append("reusableFreshAllocationCount=").append(summary.reusableFreshAllocationCount()).append('\n');
        sb.append("reuseHitRate=").append(summary.reuseHitRate()).append('\n');
        sb.append("allocatedSlotBytes=").append(summary.allocatedSlotBytes()).append('\n');
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

    private void appendRegionValuePlan(StringBuilder sb) {
        sb.append("=== Region Value Plan ===\n");
        sb.append("regionValueCount=").append(regionValueLifetimes.size()).append('\n');
        sb.append("materializationEntries=").append(materializationPlan.size()).append('\n');
        sb.append("regionBindings=").append(regionMemoryBindings.size()).append('\n');
        sb.append("regionSlots=").append(regionSlotSizes.size()).append('\n');
        sb.append("handoffRequirements=").append(handoffRequirements.size()).append('\n');
        List<Map.Entry<RegionValueRef, RegionValueLifetime>> entries = new ArrayList<>(regionValueLifetimes.entrySet());
        entries.sort(Comparator.comparingInt(e -> e.getValue().birthStep()));
        for (Map.Entry<RegionValueRef, RegionValueLifetime> entry : entries) {
            RegionValueRef valueRef = entry.getKey();
            RegionValueLifetime lifetime = entry.getValue();
            MaterializationPlanEntry materializationEntry = materializationPlan.get(valueRef);
            RegionMemoryBinding binding = regionMemoryBindings.get(valueRef);
            sb.append("- ").append(valueRef.valueId())
                    .append(" [").append(lifetime.birthStep()).append(", ").append(lifetime.lastUseStep()).append("]")
                    .append(" decision=").append(lifetime.decision())
                    .append(" producerRegion=").append(lifetime.producerRegionId() == null ? "-" : lifetime.producerRegionId())
                    .append(" producerUnit=").append(lifetime.producerUnitId() == null ? "-" : lifetime.producerUnitId())
                    .append(" consumers=").append(lifetime.consumerUnitIds())
                    .append(" binding=").append(binding == null ? "-" : binding.kind())
                    .append(" slotId=").append(regionSlotByValueRef.get(valueRef) == null ? "-" : regionSlotByValueRef.get(valueRef))
                    .append(" slotSize=").append(regionSlotByValueRef.get(valueRef) == null ? "-" : regionSlotSizes.get(regionSlotByValueRef.get(valueRef)))
                    .append(" allocatesStorage=").append(materializationEntry != null && materializationEntry.allocatesStorage())
                    .append('\n');
        }
        if (entries.isEmpty()) {
            sb.append("(no region value plan)\n");
        }
        if (!handoffRequirements.isEmpty()) {
            sb.append("handoffs:\n");
            for (RegionHandoffRequirement handoff : handoffRequirements) {
                sb.append("  - ").append(handoff.valueRef().valueId())
                        .append(" ").append(handoff.producerRegionId()).append("/").append(handoff.producerUnitId())
                        .append(" -> ").append(handoff.consumerRegionId()).append("/").append(handoff.consumerUnitId())
                        .append(" transportType=").append(handoff.transportType())
                        .append(" decision=").append(handoff.decision())
                        .append('\n');
            }
        }
        sb.append('\n');
    }

    private void appendStructuralView(StringBuilder sb) {
        sb.append("=== Structural Memory View ===\n");
        sb.append("optimizedRegions=").append(structuralView.optimizedRegionIds().size()).append('\n');
        sb.append("materializedValues=").append(structuralView.materializedValues().size()).append('\n');
        sb.append("continuationValues=").append(structuralView.continuationValues().size()).append('\n');
        sb.append("virtualValues=").append(structuralView.virtualValues().size()).append('\n');
        sb.append("valueFlows=").append(structuralView.valueFlows().size()).append('\n');
        sb.append("crossRegionDependencies=").append(structuralView.crossRegionDependencyCount()).append('\n');
        if (!structuralView.optimizedRegionIds().isEmpty()) {
            sb.append("regionIds=").append(structuralView.optimizedRegionIds()).append('\n');
        }
        for (StructuralValueFlow flow : structuralView.valueFlows()) {
            sb.append("  - ")
                    .append(flow.valueRef().valueId())
                    .append(" decision=").append(flow.decision())
                    .append(" producerRegion=").append(flow.producerRegionId() == null ? "-" : flow.producerRegionId())
                    .append(" producerUnit=").append(flow.producerUnitId() == null ? "-" : flow.producerUnitId())
                    .append(" consumerRegions=").append(flow.consumerRegionIds())
                    .append(" consumerUnits=").append(flow.consumerUnitIds())
                    .append('\n');
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
