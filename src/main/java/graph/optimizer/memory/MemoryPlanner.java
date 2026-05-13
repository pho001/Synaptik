package graph.optimizer.memory;

import graph.optimizer.region.OptimizedRegion;
import graph.optimizer.region.ExecutionUnit;
import graph.optimizer.region.MaterializationDecision;
import graph.optimizer.region.RegionValue;
import graph.optimizer.region.RegionValueRef;
import graph.optimizer.region.ValueTypeContract;
import graph.optimizer.region.ValueTransportKind;
import graph.optimizer.state.OptimizerState;
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
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;

/**
 * Computes tensor and region memory reuse plans for optimized graphs.
 *
 * <p>The planner analyzes tensor storage ownership, last-read lifetimes, saved forward values, gradient targets, and
 * optimized region value flow. It assigns reusable storage owners to slots when policy allows reuse and also derives
 * region handoff requirements for materialized and continuation values.
 *
 * <p>This class is stateless and thread-safe as long as the input graph and optimizer state are not mutated
 * concurrently while planning.
 */
public final class MemoryPlanner {
    private MemoryPlanner() {
    }

    /**
     * Plans memory with the default policy.
     *
     * @param sortedGraph tensors in topological execution order
     * @return memory plan for tensor lifetimes and reusable slots
     */
    public static MemoryPlan plan(List<Tensor> sortedGraph) {
        return plan(sortedGraph, MemoryPlannerPolicy.defaults());
    }

    /**
     * Plans tensor memory without region optimization artifacts.
     *
     * @param sortedGraph tensors in topological execution order
     * @param policy memory reuse policy
     * @return memory plan for tensor lifetimes and reusable slots
     */
    public static MemoryPlan plan(List<Tensor> sortedGraph, MemoryPlannerPolicy policy) {
        return plan(sortedGraph, RegionValuePlanningArtifacts.empty(), policy);
    }

    /**
     * Plans memory using full optimizer state.
     *
     * <p>When optimized regions are present, the returned plan includes structural memory view, region value lifetimes,
     * materialization decisions, region slot assignment, and handoff requirements in addition to tensor slot reuse.
     *
     * @param state optimizer state after region optimization
     * @param policy memory reuse policy
     * @return memory plan for runtime binding
     */
    public static MemoryPlan plan(OptimizerState state, MemoryPlannerPolicy policy) {
        Objects.requireNonNull(state, "state cannot be null");
        RegionValuePlanningArtifacts artifacts = buildRegionValuePlanningArtifacts(state);
        return plan(state.graph(), artifacts, policy);
    }

    private static MemoryPlan plan(List<Tensor> sortedGraph, RegionValuePlanningArtifacts artifacts, MemoryPlannerPolicy policy) {
        Objects.requireNonNull(policy, "policy cannot be null");
        if (sortedGraph == null || sortedGraph.isEmpty()) {
            return new MemoryPlan(Map.of(), Map.of(), Map.of(), Map.of(), policy,
                    new MemoryPlanSummary(0, 0, 0, 0, 0.0d, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0, 0, 0.0),
                    artifacts.structuralView(),
                    artifacts.regionValueLifetimes(),
                    artifacts.materializationPlan(),
                    artifacts.regionMemoryBindings(),
                    artifacts.regionSlotByValueRef(),
                    artifacts.regionSlotSizes(),
                    artifacts.tensorToRegionValueRef(),
                    artifacts.handoffRequirements(),
                    Map.of());
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
        Map<Tensor, RuntimeMemoryBindingPolicy> runtimeBindingPolicies = buildRuntimeBindingPolicies(sortedGraph);

        return new MemoryPlan(
                lifetimes,
                reusableIntervals,
                assignment.slotByOwner(),
                assignment.slotSizes(),
                policy,
                summary,
                artifacts.structuralView(),
                artifacts.regionValueLifetimes(),
                artifacts.materializationPlan(),
                artifacts.regionMemoryBindings(),
                artifacts.regionSlotByValueRef(),
                artifacts.regionSlotSizes(),
                artifacts.tensorToRegionValueRef(),
                artifacts.handoffRequirements(),
                runtimeBindingPolicies
        );
    }

    private static Map<Tensor, RuntimeMemoryBindingPolicy> buildRuntimeBindingPolicies(List<Tensor> sortedGraph) {
        IdentityHashMap<Tensor, RuntimeMemoryBindingPolicy> policies = new IdentityHashMap<>();
        for (Tensor tensor : sortedGraph) {
            Operation operation = tensor.getOperation();
            if (operation == null || operation.opType() == null) {
                policies.put(tensor, RuntimeMemoryBindingPolicy.REGION_BINDING_ALLOWED);
                continue;
            }
            RuntimeMemoryBindingPolicy policy = switch (operation.opType()) {
                case MAX_POOL2D ->
                        RuntimeMemoryBindingPolicy.skip("workspace-sensitive-storage");
                default -> RuntimeMemoryBindingPolicy.REGION_BINDING_ALLOWED;
            };
            policies.put(tensor, policy);
        }
        return Map.copyOf(policies);
    }

    private static RegionValuePlanningArtifacts buildRegionValuePlanningArtifacts(OptimizerState state) {
        Objects.requireNonNull(state, "state cannot be null");
        List<OptimizedRegion> optimizedRegions = state.optimizedRegions();
        if (optimizedRegions == null || optimizedRegions.isEmpty()) {
            return RegionValuePlanningArtifacts.empty();
        }
        Map<Tensor, Integer> graphLastUseByTensor = buildGraphLastUseByTensor(state);
        List<String> regionIds = optimizedRegions.stream().map(OptimizedRegion::regionId).toList();
        LinkedHashSet<RegionValueRef> materialized = new LinkedHashSet<>();
        LinkedHashSet<RegionValueRef> continuation = new LinkedHashSet<>();
        LinkedHashSet<RegionValueRef> virtual = new LinkedHashSet<>();
        LinkedHashMap<RegionValueRef, StructuralValueFlowBuilder> flowBuilders = new LinkedHashMap<>();
        LinkedHashMap<RegionValueRef, RegionValueDescriptor> descriptors = new LinkedHashMap<>();
        LinkedHashMap<String, Integer> unitStepById = new LinkedHashMap<>();
        LinkedHashMap<RegionValueRef, Integer> producerStepByValue = new LinkedHashMap<>();
        int nextStep = 0;

        for (OptimizedRegion region : optimizedRegions) {
            for (RegionValue value : region.regionValues()) {
                MaterializationDecision decision = switch (value.transportKind()) {
                    case MATERIALIZED -> MaterializationDecision.MATERIALIZE;
                    case CONTINUATION -> MaterializationDecision.CONTINUE;
                    case VIRTUAL -> MaterializationDecision.VIRTUALIZE;
                };
                switch (value.transportKind()) {
                    case MATERIALIZED -> materialized.add(value.ref());
                    case CONTINUATION -> continuation.add(value.ref());
                    case VIRTUAL -> virtual.add(value.ref());
                }
                flowBuilders.computeIfAbsent(value.ref(), ignored -> new StructuralValueFlowBuilder(value.ref()))
                        .decision(decision)
                        .producerRegion(region.regionId());
                descriptors.put(value.ref(), new RegionValueDescriptor(
                        value.ref(),
                        decision,
                        value.typeContract(),
                        value.semanticTensor(),
                        value.elementCount(),
                        value.requiredMaterialized(),
                        region.regionId()
                ));
            }
            for (ExecutionUnit unit : region.executionUnits()) {
                unitStepById.put(unit.unitId(), nextStep++);
                for (RegionValueRef outputValueRef : unit.outputValueRefs()) {
                    flowBuilders.computeIfAbsent(outputValueRef, ignored -> new StructuralValueFlowBuilder(outputValueRef))
                            .producerRegion(region.regionId())
                            .producerUnit(unit.unitId());
                    producerStepByValue.put(outputValueRef, unitStepById.get(unit.unitId()));
                }
                for (RegionValueRef inputValueRef : unit.inputValueRefs()) {
                    StructuralValueFlowBuilder builder = flowBuilders.get(inputValueRef);
                    if (builder == null) {
                        continue;
                    }
                    builder.addConsumerRegion(region.regionId());
                    builder.addConsumerUnit(unit.unitId());
                }
            }
        }

        List<StructuralValueFlow> valueFlows = flowBuilders.values().stream()
                .map(StructuralValueFlowBuilder::build)
                .toList();
        StructuralMemoryView structuralView = new StructuralMemoryView(
                regionIds,
                List.copyOf(materialized),
                List.copyOf(continuation),
                List.copyOf(virtual),
                valueFlows
        );
        LinkedHashMap<RegionValueRef, RegionValueLifetime> regionValueLifetimes = new LinkedHashMap<>();
        LinkedHashMap<RegionValueRef, MaterializationPlanEntry> materializationPlan = new LinkedHashMap<>();
        LinkedHashMap<RegionValueRef, RegionMemoryBinding> regionMemoryBindings = new LinkedHashMap<>();
        LinkedHashMap<RegionValueRef, Integer> regionSlotByValueRef = new LinkedHashMap<>();
        LinkedHashMap<Integer, Integer> regionSlotSizes = new LinkedHashMap<>();
        IdentityHashMap<Tensor, RegionValueRef> tensorToRegionValueRef = new IdentityHashMap<>();
        ArrayList<RegionHandoffRequirement> handoffRequirements = new ArrayList<>();

        for (StructuralValueFlow flow : valueFlows) {
            RegionValueDescriptor descriptor = descriptors.get(flow.valueRef());
            if (descriptor == null) {
                continue;
            }
            tensorToRegionValueRef.put(descriptor.semanticTensor(), flow.valueRef());
            int birthStep = producerStepByValue.getOrDefault(flow.valueRef(), 0);
            int lastUseStep = birthStep;
            for (String consumerUnitId : flow.consumerUnitIds()) {
                Integer step = unitStepById.get(consumerUnitId);
                if (step != null) {
                    lastUseStep = Math.max(lastUseStep, step);
                }
            }
            Integer graphLastUseStep = graphLastUseByTensor.get(descriptor.semanticTensor());
            if (graphLastUseStep != null) {
                lastUseStep = Math.max(lastUseStep, graphLastUseStep);
            }
            RegionValueLifetime lifetime = new RegionValueLifetime(
                    flow.valueRef(),
                    birthStep,
                    lastUseStep,
                    descriptor.elementCount(),
                    descriptor.decision(),
                    descriptor.typeContract(),
                    flow.producerRegionId(),
                    flow.producerUnitId(),
                    flow.consumerRegionIds(),
                    flow.consumerUnitIds()
            );
            regionValueLifetimes.put(flow.valueRef(), lifetime);
            materializationPlan.put(flow.valueRef(), new MaterializationPlanEntry(
                    flow.valueRef(),
                    descriptor.decision(),
                    descriptor.requiredMaterialized(),
                    descriptor.decision() != MaterializationDecision.VIRTUALIZE
            ));
        }

        BindingAssignment bindingAssignment = assignRegionBindings(regionValueLifetimes.values().stream().toList());
        regionMemoryBindings.putAll(bindingAssignment.bindingsByValueRef());
        regionSlotByValueRef.putAll(bindingAssignment.slotByValueRef());
        regionSlotSizes.putAll(bindingAssignment.slotSizes());

        for (RegionValueLifetime lifetime : regionValueLifetimes.values()) {
            for (int i = 0; i < lifetime.consumerRegionIds().size(); i++) {
                String consumerRegionId = lifetime.consumerRegionIds().get(i);
                if (consumerRegionId == null
                        || consumerRegionId.isBlank()
                        || consumerRegionId.equals(lifetime.producerRegionId())) {
                    continue;
                }
                String consumerUnitId = i < lifetime.consumerUnitIds().size()
                        ? lifetime.consumerUnitIds().get(i)
                        : null;
                handoffRequirements.add(new RegionHandoffRequirement(
                        lifetime.valueRef(),
                        lifetime.producerRegionId(),
                        lifetime.producerUnitId(),
                        consumerRegionId,
                        consumerUnitId,
                        transportTypeFor(lifetime),
                        lifetime.decision()
                ));
            }
        }

        return new RegionValuePlanningArtifacts(
                structuralView,
                Map.copyOf(regionValueLifetimes),
                Map.copyOf(materializationPlan),
                Map.copyOf(regionMemoryBindings),
                Map.copyOf(regionSlotByValueRef),
                Map.copyOf(regionSlotSizes),
                Map.copyOf(tensorToRegionValueRef),
                List.copyOf(handoffRequirements)
        );
    }

    private static Map<Tensor, Integer> buildGraphLastUseByTensor(OptimizerState state) {
        List<Tensor> graph = state.graph();
        IdentityHashMap<Tensor, Integer> lastUseByTensor = new IdentityHashMap<>(graph.size());
        for (int i = 0; i < graph.size(); i++) {
            lastUseByTensor.put(graph.get(i), i);
        }
        for (int i = 0; i < graph.size(); i++) {
            Tensor consumer = graph.get(i);
            List<Tensor> inputs = consumer.getPrevTensors();
            if (inputs == null || inputs.isEmpty()) {
                continue;
            }
            for (Tensor input : inputs) {
                if (input != null) {
                    lastUseByTensor.merge(input, i, Math::max);
                }
            }
        }
        int terminalPublishStep = graph.size();
        extendPublishedLifetime(lastUseByTensor, state.forwardOutput(), terminalPublishStep);
        if (state.supportsBackward()) {
            for (Tensor tensor : graph) {
                Tensor gradient = tensor.getGradient();
                if (gradient != null) {
                    extendPublishedLifetime(lastUseByTensor, gradient, terminalPublishStep);
                }
            }
        }
        return Map.copyOf(lastUseByTensor);
    }

    private static void extendPublishedLifetime(
            Map<Tensor, Integer> lastUseByTensor,
            Tensor tensor,
            int terminalPublishStep
    ) {
        Tensor current = tensor;
        while (current != null) {
            lastUseByTensor.merge(current, terminalPublishStep, Math::max);
            if (!aliasesInput0AtRuntime(current)) {
                return;
            }
            List<Tensor> inputs = current.getPrevTensors();
            current = (inputs == null || inputs.isEmpty()) ? null : inputs.getFirst();
        }
    }

    private static BindingAssignment assignRegionBindings(List<RegionValueLifetime> lifetimes) {
        if (lifetimes.isEmpty()) {
            return new BindingAssignment(Map.of(), Map.of(), Map.of());
        }
        List<RegionValueLifetime> allocatable = lifetimes.stream()
                .filter(lifetime -> lifetime.decision() != MaterializationDecision.VIRTUALIZE)
                .sorted(Comparator.comparingInt(RegionValueLifetime::birthStep).thenComparingInt(RegionValueLifetime::lastUseStep))
                .toList();
        ArrayList<RegionBindingState> active = new ArrayList<>();
        ArrayList<RegionBindingState> free = new ArrayList<>();
        LinkedHashMap<RegionValueRef, RegionMemoryBinding> bindings = new LinkedHashMap<>();
        LinkedHashMap<RegionValueRef, Integer> slotByValueRef = new LinkedHashMap<>();
        LinkedHashMap<Integer, Integer> slotSizes = new LinkedHashMap<>();
        int nextBindingId = 0;
        for (RegionValueLifetime lifetime : allocatable) {
            releaseExpiredRegionBindings(active, free, lifetime.birthStep());
            RegionBindingState chosen = chooseRegionBinding(free, lifetime);
            if (chosen == null) {
                chosen = new RegionBindingState(
                        nextBindingId++,
                        bindingKindFor(lifetime.decision()),
                        storageTypeFor(lifetime),
                        transportTypeFor(lifetime),
                        slotSizeElementsFor(lifetime),
                        Integer.MIN_VALUE
                );
                slotSizes.put(chosen.bindingId, slotSizeElementsFor(lifetime));
            } else {
                free.remove(chosen);
            }
            chosen.lastUseStep = lifetime.lastUseStep();
            active.add(chosen);
            bindings.put(lifetime.valueRef(), new RegionMemoryBinding(
                    lifetime.valueRef(),
                    chosen.kind,
                    chosen.bindingId,
                    chosen.storageType,
                    chosen.transportType,
                    true
            ));
            slotByValueRef.put(lifetime.valueRef(), chosen.bindingId);
            slotSizes.merge(chosen.bindingId, slotSizeElementsFor(lifetime), Math::max);
        }
        for (RegionValueLifetime lifetime : lifetimes) {
            if (bindings.containsKey(lifetime.valueRef())) {
                continue;
            }
            bindings.put(lifetime.valueRef(), new RegionMemoryBinding(
                    lifetime.valueRef(),
                    RegionMemoryBindingKind.NONE,
                    null,
                    storageTypeFor(lifetime),
                    transportTypeFor(lifetime),
                    false
            ));
        }
        return new BindingAssignment(Map.copyOf(bindings), Map.copyOf(slotByValueRef), Map.copyOf(slotSizes));
    }

    private static void releaseExpiredRegionBindings(
            List<RegionBindingState> active,
            List<RegionBindingState> free,
            int currentBirthStep
    ) {
        List<RegionBindingState> released = new ArrayList<>();
        for (RegionBindingState state : active) {
            if (state.lastUseStep < currentBirthStep) {
                released.add(state);
            }
        }
        active.removeAll(released);
        free.addAll(released);
    }

    private static RegionBindingState chooseRegionBinding(List<RegionBindingState> free, RegionValueLifetime lifetime) {
        RegionMemoryBindingKind kind = bindingKindFor(lifetime.decision());
        DataType storageType = storageTypeFor(lifetime);
        DataType transportType = transportTypeFor(lifetime);
        int slotSize = slotSizeElementsFor(lifetime);
        for (RegionBindingState state : free) {
            if (state.kind == kind
                    && state.storageType == storageType
                    && state.transportType == transportType
                    && state.size == slotSize) {
                return state;
            }
        }
        return null;
    }

    private static RegionMemoryBindingKind bindingKindFor(MaterializationDecision decision) {
        return switch (decision) {
            case CONTINUE -> RegionMemoryBindingKind.CONTINUATION;
            case MATERIALIZE -> RegionMemoryBindingKind.MATERIALIZED;
            case VIRTUALIZE -> RegionMemoryBindingKind.NONE;
        };
    }

    private static DataType storageTypeFor(RegionValueLifetime lifetime) {
        return lifetime.typeContract().storageType();
    }

    private static DataType transportTypeFor(RegionValueLifetime lifetime) {
        return switch (lifetime.decision()) {
            case CONTINUE -> lifetime.typeContract().transportType();
            case MATERIALIZE, VIRTUALIZE -> lifetime.typeContract().storageType();
        };
    }

    private static int slotSizeElementsFor(RegionValueLifetime lifetime) {
        return Math.max(0, lifetime.elementCount());
    }

    private static final class StructuralValueFlowBuilder {
        private final RegionValueRef valueRef;
        private MaterializationDecision decision = MaterializationDecision.MATERIALIZE;
        private String producerRegionId;
        private String producerUnitId;
        private final LinkedHashSet<String> consumerRegionIds = new LinkedHashSet<>();
        private final LinkedHashSet<String> consumerUnitIds = new LinkedHashSet<>();

        private StructuralValueFlowBuilder(RegionValueRef valueRef) {
            this.valueRef = valueRef;
        }

        private StructuralValueFlowBuilder decision(MaterializationDecision decision) {
            this.decision = decision == null ? MaterializationDecision.MATERIALIZE : decision;
            return this;
        }

        private StructuralValueFlowBuilder producerRegion(String producerRegionId) {
            if (producerRegionId != null && !producerRegionId.isBlank()) {
                this.producerRegionId = producerRegionId;
            }
            return this;
        }

        private StructuralValueFlowBuilder producerUnit(String producerUnitId) {
            if (producerUnitId != null && !producerUnitId.isBlank()) {
                this.producerUnitId = producerUnitId;
            }
            return this;
        }

        private StructuralValueFlowBuilder addConsumerRegion(String consumerRegionId) {
            if (consumerRegionId != null && !consumerRegionId.isBlank()) {
                consumerRegionIds.add(consumerRegionId);
            }
            return this;
        }

        private StructuralValueFlowBuilder addConsumerUnit(String consumerUnitId) {
            if (consumerUnitId != null && !consumerUnitId.isBlank()) {
                consumerUnitIds.add(consumerUnitId);
            }
            return this;
        }

        private StructuralValueFlow build() {
            return new StructuralValueFlow(
                    valueRef,
                    decision,
                    producerRegionId,
                    producerUnitId,
                    List.copyOf(consumerRegionIds),
                    List.copyOf(consumerUnitIds)
            );
        }
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
            case NOOP, EXPAND, SELECT, SLICE, PERMUTE, EXPAND_DIMS, SQUEEZE -> true;
            case RESHAPE -> inputs.get(0).isContiguous();
            default -> false;
        };
    }

    private record SlotAssignment(
            Map<Tensor, Integer> slotByOwner,
            Map<Integer, Integer> slotSizes
    ) {
    }

    private record BindingAssignment(
            Map<RegionValueRef, RegionMemoryBinding> bindingsByValueRef,
            Map<RegionValueRef, Integer> slotByValueRef,
            Map<Integer, Integer> slotSizes
    ) {
    }

    private record RegionValueDescriptor(
            RegionValueRef valueRef,
            MaterializationDecision decision,
            ValueTypeContract typeContract,
            Tensor semanticTensor,
            int elementCount,
            boolean requiredMaterialized,
            String producerRegionId
    ) {
    }

    private record RegionValuePlanningArtifacts(
            StructuralMemoryView structuralView,
            Map<RegionValueRef, RegionValueLifetime> regionValueLifetimes,
            Map<RegionValueRef, MaterializationPlanEntry> materializationPlan,
            Map<RegionValueRef, RegionMemoryBinding> regionMemoryBindings,
            Map<RegionValueRef, Integer> regionSlotByValueRef,
            Map<Integer, Integer> regionSlotSizes,
            Map<Tensor, RegionValueRef> tensorToRegionValueRef,
            List<RegionHandoffRequirement> handoffRequirements
    ) {
        private static RegionValuePlanningArtifacts empty() {
            return new RegionValuePlanningArtifacts(
                    StructuralMemoryView.empty(),
                    Map.of(),
                    Map.of(),
                    Map.of(),
                    Map.of(),
                    Map.of(),
                    Map.of(),
                    List.of()
            );
        }
    }

    private static final class RegionBindingState {
        private final int bindingId;
        private final RegionMemoryBindingKind kind;
        private final DataType storageType;
        private final DataType transportType;
        private final int size;
        private int lastUseStep;

        private RegionBindingState(
                int bindingId,
                RegionMemoryBindingKind kind,
                DataType storageType,
                DataType transportType,
                int size,
                int lastUseStep
        ) {
            this.bindingId = bindingId;
            this.kind = kind;
            this.storageType = storageType;
            this.transportType = transportType;
            this.size = size;
            this.lastUseStep = lastUseStep;
        }
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
