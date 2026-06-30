package planning.memory;

import planning.value.GraphValueRef;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * Immutable memory planning result consumed by runtime preparation and binding.
 *
 * <p>The plan has two layers:
 * <ul>
 *     <li>Tensor-level lifetimes and reusable slots for storage owners in the compiled graph.</li>
 *     <li>Region-level value lifetimes, materialization decisions, slot assignments, and handoff requirements derived
 *     from planned regions.</li>
 * </ul>
 *
 * <p>The plan is read-only. Runtime binding may allocate or reuse buffers according to this metadata, but should not
 * mutate the plan itself.
 */
public final class MemoryPlan {
    private final TensorMemoryPlan tensorPlan;
    private final RegionMemoryPlan regionPlan;
    private final RuntimeBindingPlan runtimeBindingPlan;
    private final MemoryPlannerPolicy policy;
    private final MemoryPlanSummary summary;

    /**
     * Creates a memory plan.
     *
     * @param tensor tensor-level memory planning
     * @param region region-level memory planning
     * @param runtimeBinding runtime binding policy
     * @param policy policy used to build the plan
     * @param summary summary metrics
     */
    public MemoryPlan(
            TensorMemoryPlan tensor,
            RegionMemoryPlan region,
            RuntimeBindingPlan runtimeBinding,
            MemoryPlannerPolicy policy,
            MemoryPlanSummary summary
    ) {
        this.tensorPlan = tensor == null ? TensorMemoryPlan.empty() : tensor;
        this.regionPlan = region == null ? RegionMemoryPlan.empty() : region;
        this.runtimeBindingPlan = runtimeBinding == null ? RuntimeBindingPlan.empty() : runtimeBinding;
        this.policy = Objects.requireNonNull(policy, "policy cannot be null");
        this.summary = Objects.requireNonNull(summary, "summary cannot be null");
    }

    public TensorMemoryPlan tensor() {
        return tensorPlan;
    }

    public RegionMemoryPlan region() {
        return regionPlan;
    }

    public RuntimeBindingPlan runtimeBinding() {
        return runtimeBindingPlan;
    }

    /**
     * Returns lifetime metadata for a tensor.
     *
     * @param tensor tensor to look up
     * @return node lifetime
     * @throws IllegalArgumentException if the tensor is absent from the plan
     */
    public NodeLifetime lifetimeOf(Tensor tensor) {
        NodeLifetime lifetime = tensorPlan.lifetimes().get(tensor);
        if (lifetime == null) {
            throw new IllegalArgumentException("Missing lifetime for tensor: " + tensor.getLabel());
        }
        return lifetime;
    }

    /**
     * Returns the tensor that owns storage for {@code tensor}.
     *
     * @param tensor tensor to inspect
     * @return storage owner tensor
     */
    public Tensor storageOwnerOf(Tensor tensor) {
        return lifetimeOf(tensor).storageOwner();
    }

    /**
     * Returns the last graph index that reads a tensor's storage owner.
     *
     * @param tensor tensor to inspect
     * @return last read index, or {@link Integer#MAX_VALUE} for externally observable values
     */
    public int lastReadIndexOf(Tensor tensor) {
        return lifetimeOf(tensor).lastReadIndex();
    }

    /**
     * Returns whether a tensor is a reusable storage owner.
     *
     * @param tensor tensor to inspect
     * @return {@code true} when the tensor owns storage and has a reusable interval
     */
    public boolean isReusableOwner(Tensor tensor) {
        NodeLifetime lifetime = lifetimeOf(tensor);
        if (lifetime.storageOwner() != tensor) {
            return false;
        }
        return tensorPlan.reusableIntervals().containsKey(tensor);
    }

    /**
     * Returns tensor-level slot id for a tensor's storage owner.
     *
     * @param tensor tensor to inspect
     * @return slot id, or {@code null} when no reusable slot was assigned
     */
    public Integer slotIdOf(Tensor tensor) {
        Tensor owner = storageOwnerOf(tensor);
        return tensorPlan.slotByOwner().get(owner);
    }

    /**
     * Returns tensor-level slot size.
     *
     * @param slotId slot id
     * @return slot size in elements
     */
    public int slotSize(int slotId) {
        Integer size = tensorPlan.slotSizes().get(slotId);
        if (size == null) {
            throw new IllegalArgumentException("Missing slot size for slot " + slotId);
        }
        return size;
    }

    /**
     * Returns the policy used to build this plan.
     *
     * @return memory planner policy
     */
    public MemoryPlannerPolicy policy() {
        return policy;
    }

    /**
     * Returns summary metrics for this plan.
     *
     * @return memory plan summary
     */
    public MemoryPlanSummary summary() {
        return summary;
    }

    /**
     * Returns structural region memory flow.
     *
     * @return structural memory view
     */
    public StructuralMemoryView structuralView() {
        return regionPlan.structuralView();
    }

    /**
     * Returns lifetime metadata for a region value.
     *
     * @param valueRef graph value reference
     * @return region value lifetime
     */
    public RegionValueLifetime regionValueLifetimeOf(GraphValueRef valueRef) {
        RegionValueLifetime lifetime = regionPlan.valueLifetimes().get(valueRef);
        if (lifetime == null) {
            throw new IllegalArgumentException("Missing region value lifetime for: " + valueRef.valueId());
        }
        return lifetime;
    }

    /**
     * Returns materialization metadata for a region value.
     *
     * @param valueRef graph value reference
     * @return materialization plan entry
     */
    public MaterializationPlanEntry materializationPlanOf(GraphValueRef valueRef) {
        MaterializationPlanEntry entry = regionPlan.materializationPlan().get(valueRef);
        if (entry == null) {
            throw new IllegalArgumentException("Missing materialization plan entry for: " + valueRef.valueId());
        }
        return entry;
    }

    /**
     * Returns memory binding metadata for a region value.
     *
     * @param valueRef graph value reference
     * @return region memory binding
     */
    public RegionMemoryBinding regionMemoryBindingOf(GraphValueRef valueRef) {
        RegionMemoryBinding binding = regionPlan.memoryBindings().get(valueRef);
        if (binding == null) {
            throw new IllegalArgumentException("Missing region memory binding for: " + valueRef.valueId());
        }
        return binding;
    }

    /**
     * Returns all region value lifetimes.
     *
     * @return immutable lifetime map
     */
    public Map<GraphValueRef, RegionValueLifetime> regionValueLifetimes() {
        return regionPlan.valueLifetimes();
    }

    /**
     * Returns all materialization plan entries.
     *
     * @return immutable materialization map
     */
    public Map<GraphValueRef, MaterializationPlanEntry> materializationPlan() {
        return regionPlan.materializationPlan();
    }

    /**
     * Returns all region memory bindings.
     *
     * @return immutable binding map
     */
    public Map<GraphValueRef, RegionMemoryBinding> regionMemoryBindings() {
        return regionPlan.memoryBindings();
    }

    /**
     * Returns the region slot id for a value.
     *
     * @param valueRef graph value reference
     * @return slot id, or {@code null} when the value has no region slot
     */
    public Integer regionSlotIdOf(GraphValueRef valueRef) {
        return regionPlan.slotByValueRef().get(valueRef);
    }

    /**
     * Returns how many region values use a slot.
     *
     * @param slotId region slot id
     * @return use count
     */
    public int regionSlotUseCount(int slotId) {
        return regionPlan.slotUseCounts().getOrDefault(slotId, 0);
    }

    /**
     * Returns region slot size.
     *
     * @param slotId region slot id
     * @return slot size in elements
     */
    public int regionSlotSize(int slotId) {
        Integer size = regionPlan.slotSizes().get(slotId);
        if (size == null) {
            throw new IllegalArgumentException("Missing region slot size for slot " + slotId);
        }
        return size;
    }

    /**
     * Returns region slot assignments by value.
     *
     * @return immutable region slot map
     */
    public Map<GraphValueRef, Integer> regionSlotByValueRef() {
        return regionPlan.slotByValueRef();
    }

    /**
     * Returns region slot sizes.
     *
     * @return immutable region slot size map
     */
    public Map<Integer, Integer> regionSlotSizes() {
        return regionPlan.slotSizes();
    }

    /**
     * Returns graph value reference associated with a tensor.
     *
     * @param tensor tensor to inspect
     * @return graph value ref, or {@code null} when the tensor is not region-owned
     */
    public GraphValueRef graphValueRefOf(Tensor tensor) {
        return regionPlan.tensorToGraphValueRef().get(tensor);
    }

    public GraphValueRef graphValueRefOfNodeId(int nodeId) {
        return regionPlan.nodeIdToGraphValueRef().get(nodeId);
    }

    /**
     * Returns the runtime slot id for a tensor.
     *
     * <p>Region slots take precedence when a tensor maps to a region value; otherwise the tensor-level slot is used.
     *
     * @param tensor tensor to inspect
     * @return runtime slot id, or {@code null} if no slot is assigned
     */
    public Integer runtimeSlotIdOf(Tensor tensor) {
        GraphValueRef graphValueRef = graphValueRefOf(tensor);
        if (graphValueRef != null) {
            Integer regionSlotId = regionSlotIdOf(graphValueRef);
            if (regionSlotId != null) {
                return regionSlotId;
            }
        }
        return slotIdOf(tensor);
    }

    public Integer runtimeSlotIdOfNodeId(int nodeId) {
        GraphValueRef graphValueRef = graphValueRefOfNodeId(nodeId);
        if (graphValueRef == null) {
            return null;
        }
        return regionSlotIdOf(graphValueRef);
    }

    /**
     * Returns the runtime slot size for a tensor.
     *
     * @param tensor tensor to inspect
     * @return slot size in elements
     */
    public int runtimeSlotSizeOf(Tensor tensor) {
        Integer runtimeSlotId = runtimeSlotIdOf(tensor);
        if (runtimeSlotId == null) {
            throw new IllegalArgumentException("Missing runtime slot for tensor: " + tensor.getLabel());
        }
        GraphValueRef graphValueRef = graphValueRefOf(tensor);
        if (graphValueRef != null && regionSlotIdOf(graphValueRef) != null && regionSlotIdOf(graphValueRef).equals(runtimeSlotId)) {
            return regionSlotSize(runtimeSlotId);
        }
        return slotSize(runtimeSlotId);
    }

    public int runtimeSlotSizeOfNodeId(int nodeId) {
        Integer runtimeSlotId = runtimeSlotIdOfNodeId(nodeId);
        if (runtimeSlotId == null) {
            throw new IllegalArgumentException("Missing runtime slot for nodeId: " + nodeId);
        }
        return regionSlotSize(runtimeSlotId);
    }

    /**
     * Returns region handoff requirements.
     *
     * @return immutable handoff list
     */
    public List<RegionHandoffRequirement> handoffRequirements() {
        return regionPlan.handoffRequirements();
    }

    /**
     * Returns runtime binding policy for a tensor.
     *
     * @param tensor tensor to inspect
     * @return binding policy, defaulting to region binding allowed
     */
    public RuntimeMemoryBindingPolicy runtimeBindingPolicyOf(Tensor tensor) {
        return runtimeBindingPlan.policiesByTensor().getOrDefault(tensor, RuntimeMemoryBindingPolicy.REGION_BINDING_ALLOWED);
    }

    public RuntimeMemoryBindingPolicy runtimeBindingPolicyOfNodeId(int nodeId) {
        return runtimeBindingPlan.policiesByNodeId().getOrDefault(nodeId, RuntimeMemoryBindingPolicy.REGION_BINDING_ALLOWED);
    }

    /**
     * Returns all runtime binding policies.
     *
     * @return immutable policy map
     */
    public Map<Tensor, RuntimeMemoryBindingPolicy> runtimeBindingPolicies() {
        return runtimeBindingPlan.policiesByTensor();
    }

    public Map<Integer, RuntimeMemoryBindingPolicy> runtimeBindingPoliciesByNodeId() {
        return runtimeBindingPlan.policiesByNodeId();
    }

    /**
     * Returns tensor-level slot assignments by storage owner.
     *
     * @return immutable slot map
     */
    public Map<Tensor, Integer> slotByOwner() {
        return tensorPlan.slotByOwner();
    }

    /**
     * Returns tensor-level slot sizes.
     *
     * @return immutable slot size map
     */
    public Map<Integer, Integer> slotSizes() {
        return tensorPlan.slotSizes();
    }

    /**
     * Returns a human-readable explanation of this plan.
     *
     * @return multi-section explanation string
     */
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
        for (Map.Entry<Tensor, ReusableInterval> entry : tensorPlan.reusableIntervals().entrySet()) {
            Integer slotId = tensorPlan.slotByOwner().get(entry.getKey());
            if (slotId == null) {
                continue;
            }
            bySlot.computeIfAbsent(slotId, ignored -> new ArrayList<>()).add(entry.getValue());
        }
        for (Map.Entry<Integer, List<ReusableInterval>> entry : bySlot.entrySet()) {
            int slotId = entry.getKey();
            sb.append("slot ").append(slotId)
                    .append(" size=").append(tensorPlan.slotSizes().get(slotId))
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
        sb.append("regionValueCount=").append(regionPlan.valueLifetimes().size()).append('\n');
        sb.append("materializationEntries=").append(regionPlan.materializationPlan().size()).append('\n');
        sb.append("regionBindings=").append(regionPlan.memoryBindings().size()).append('\n');
        sb.append("regionSlots=").append(regionPlan.slotSizes().size()).append('\n');
        sb.append("handoffRequirements=").append(regionPlan.handoffRequirements().size()).append('\n');
        List<Map.Entry<GraphValueRef, RegionValueLifetime>> entries = new ArrayList<>(regionPlan.valueLifetimes().entrySet());
        entries.sort(Comparator.comparingInt(e -> e.getValue().birthStep()));
        for (Map.Entry<GraphValueRef, RegionValueLifetime> entry : entries) {
            GraphValueRef valueRef = entry.getKey();
            RegionValueLifetime lifetime = entry.getValue();
            MaterializationPlanEntry materializationEntry = regionPlan.materializationPlan().get(valueRef);
            RegionMemoryBinding binding = regionPlan.memoryBindings().get(valueRef);
            sb.append("- ").append(valueRef.valueId())
                    .append(" [").append(lifetime.birthStep()).append(", ").append(lifetime.lastUseStep()).append("]")
                    .append(" decision=").append(lifetime.decision())
                    .append(" producerRegion=").append(lifetime.producerRegionId() == null ? "-" : lifetime.producerRegionId())
                    .append(" producerUnit=").append(lifetime.producerUnitId() == null ? "-" : lifetime.producerUnitId())
                    .append(" consumers=").append(lifetime.consumerUnitIds())
                    .append(" binding=").append(binding == null ? "-" : binding.kind())
                    .append(" slotId=").append(regionPlan.slotByValueRef().get(valueRef) == null ? "-" : regionPlan.slotByValueRef().get(valueRef))
                    .append(" slotSize=").append(regionPlan.slotByValueRef().get(valueRef) == null ? "-" : regionPlan.slotSizes().get(regionPlan.slotByValueRef().get(valueRef)))
                    .append(" allocatesStorage=").append(materializationEntry != null && materializationEntry.allocatesStorage())
                    .append('\n');
        }
        if (entries.isEmpty()) {
            sb.append("(no region value plan)\n");
        }
        if (!regionPlan.handoffRequirements().isEmpty()) {
            sb.append("handoffs:\n");
            for (RegionHandoffRequirement handoff : regionPlan.handoffRequirements()) {
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
        sb.append("plannedRegions=").append(regionPlan.structuralView().plannedRegionIds().size()).append('\n');
        sb.append("materializedValues=").append(regionPlan.structuralView().materializedValues().size()).append('\n');
        sb.append("continuationValues=").append(regionPlan.structuralView().continuationValues().size()).append('\n');
        sb.append("virtualValues=").append(regionPlan.structuralView().virtualValues().size()).append('\n');
        sb.append("valueFlows=").append(regionPlan.structuralView().valueFlows().size()).append('\n');
        sb.append("crossRegionDependencies=").append(regionPlan.structuralView().crossRegionDependencyCount()).append('\n');
        if (!regionPlan.structuralView().plannedRegionIds().isEmpty()) {
            sb.append("regionIds=").append(regionPlan.structuralView().plannedRegionIds()).append('\n');
        }
        for (StructuralValueFlow flow : regionPlan.structuralView().valueFlows()) {
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
        List<Map.Entry<Tensor, NodeLifetime>> entries = new ArrayList<>(tensorPlan.lifetimes().entrySet());
        entries.sort(Comparator.comparingInt(e -> e.getValue().birthIndex()));
        for (Map.Entry<Tensor, NodeLifetime> entry : entries) {
            Tensor tensor = entry.getKey();
            NodeLifetime lifetime = entry.getValue();
            Integer slotId = tensorPlan.slotByOwner().get(lifetime.storageOwner());
            sb.append("[")
                    .append(lifetime.birthIndex())
                    .append("] label=").append(tensor.getLabel())
                    .append(", op=").append(tensor.getOperation() == null ? "LEAF" : tensor.getOperation().opType())
                    .append(", role=").append(lifetime.role())
                    .append(", owner=").append(lifetime.storageOwner().getLabel())
                    .append(", lastRead=").append(lifetime.lastReadIndex())
                    .append(", reusable=").append(tensorPlan.reusableIntervals().containsKey(lifetime.storageOwner()))
                    .append(", slot=").append(slotId == null ? "-" : slotId)
                    .append("\n");
        }
        sb.append('\n');
    }

    private void appendSavedForward(StringBuilder sb) {
        sb.append("=== Saved Forward Values ===\n");
        List<Map.Entry<Tensor, NodeLifetime>> entries = new ArrayList<>(tensorPlan.lifetimes().entrySet());
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
                    .append(" slot=").append(tensorPlan.slotByOwner().get(tensor))
                    .append('\n');
        }
        if (!any) {
            sb.append("(none)\n");
        }
    }
}
