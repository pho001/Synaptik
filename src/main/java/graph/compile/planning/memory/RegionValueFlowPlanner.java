package graph.compile.planning.memory;

import graph.CompiledNode;
import graph.compile.planning.region.ExecutionUnit;
import graph.compile.planning.region.MaterializationDecision;
import graph.compile.planning.region.OptimizedRegion;
import graph.compile.planning.region.RegionValue;
import graph.compile.planning.region.ValueTypeContract;
import graph.compile.planning.value.GraphValueRef;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;

final class RegionValueFlowPlanner {
    private RegionValueFlowPlanner() {
    }

    static RegionValueFlowPlan plan(MemoryPlanningInput input) {
        Objects.requireNonNull(input, "input cannot be null");
        List<OptimizedRegion> optimizedRegions = input.optimizedRegions();
        if (optimizedRegions == null || optimizedRegions.isEmpty()) {
            return RegionValueFlowPlan.empty();
        }

        Map<Integer, Integer> graphLastUseByNodeId = buildGraphLastUseByNodeId(input);
        List<String> regionIds = optimizedRegions.stream().map(OptimizedRegion::regionId).toList();
        LinkedHashSet<GraphValueRef> materialized = new LinkedHashSet<>();
        LinkedHashSet<GraphValueRef> continuation = new LinkedHashSet<>();
        LinkedHashSet<GraphValueRef> virtual = new LinkedHashSet<>();
        LinkedHashMap<GraphValueRef, StructuralValueFlowBuilder> flowBuilders = new LinkedHashMap<>();
        LinkedHashMap<GraphValueRef, RegionValueDescriptor> descriptors = new LinkedHashMap<>();
        LinkedHashMap<String, Integer> unitStepById = new LinkedHashMap<>();
        LinkedHashMap<GraphValueRef, Integer> producerStepByValue = new LinkedHashMap<>();
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
                        decision,
                        value.typeContract(),
                        value.producerNodeId(),
                        value.elementCount(),
                        value.requiredMaterialized()
                ));
            }
            for (ExecutionUnit unit : region.executionUnits()) {
                unitStepById.put(unit.unitId(), nextStep++);
                for (GraphValueRef outputValueRef : unit.outputValueRefs()) {
                    flowBuilders.computeIfAbsent(outputValueRef, ignored -> new StructuralValueFlowBuilder(outputValueRef))
                            .producerRegion(region.regionId())
                            .producerUnit(unit.unitId());
                    producerStepByValue.put(outputValueRef, unitStepById.get(unit.unitId()));
                }
                for (GraphValueRef inputValueRef : unit.inputValueRefs()) {
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
        LinkedHashMap<GraphValueRef, RegionValueLifetime> regionValueLifetimes = new LinkedHashMap<>();
        LinkedHashMap<GraphValueRef, MaterializationPlanEntry> materializationPlan = new LinkedHashMap<>();
        LinkedHashMap<Integer, GraphValueRef> nodeIdToGraphValueRef = new LinkedHashMap<>();

        for (StructuralValueFlow flow : valueFlows) {
            RegionValueDescriptor descriptor = descriptors.get(flow.valueRef());
            if (descriptor == null) {
                continue;
            }
            nodeIdToGraphValueRef.put(descriptor.producerNodeId(), flow.valueRef());
            int birthStep = producerStepByValue.getOrDefault(flow.valueRef(), 0);
            int lastUseStep = birthStep;
            for (String consumerUnitId : flow.consumerUnitIds()) {
                Integer step = unitStepById.get(consumerUnitId);
                if (step != null) {
                    lastUseStep = Math.max(lastUseStep, step);
                }
            }
            Integer graphLastUseStep = graphLastUseByNodeId.get(descriptor.producerNodeId());
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

        return new RegionValueFlowPlan(
                structuralView,
                Map.copyOf(regionValueLifetimes),
                Map.copyOf(materializationPlan),
                Map.of(),
                Map.copyOf(nodeIdToGraphValueRef)
        );
    }

    private static Map<Integer, Integer> buildGraphLastUseByNodeId(MemoryPlanningInput input) {
        List<CompiledNode> graph = input.compiledNodes();
        LinkedHashMap<Integer, Integer> lastUseByNodeId = new LinkedHashMap<>(graph.size());
        for (int i = 0; i < graph.size(); i++) {
            lastUseByNodeId.put(graph.get(i).id(), i);
        }
        for (int i = 0; i < graph.size(); i++) {
            CompiledNode consumer = graph.get(i);
            for (int inputNodeId : consumer.inputIds()) {
                lastUseByNodeId.merge(inputNodeId, i, Math::max);
            }
        }
        int terminalPublishStep = graph.size();
        extendPublishedLifetime(lastUseByNodeId, input.forwardBoundaryNodeId(), input.compiledNodes(), terminalPublishStep);
        if (input.supportsBackward()) {
            for (CompiledNode node : graph) {
                if (node.gradientTarget()) {
                    extendPublishedLifetime(lastUseByNodeId, node.id(), input.compiledNodes(), terminalPublishStep);
                }
            }
        }
        return Map.copyOf(lastUseByNodeId);
    }

    private static void extendPublishedLifetime(
            Map<Integer, Integer> lastUseByNodeId,
            int nodeId,
            List<CompiledNode> graph,
            int terminalPublishStep
    ) {
        int current = nodeId;
        while (current >= 0 && current < graph.size()) {
            CompiledNode currentNode = graph.get(current);
            lastUseByNodeId.merge(current, terminalPublishStep, Math::max);
            if (currentNode.storageOwnerId() == current) {
                return;
            }
            current = currentNode.storageOwnerId();
        }
    }

    private record RegionValueDescriptor(
            MaterializationDecision decision,
            ValueTypeContract typeContract,
            int producerNodeId,
            int elementCount,
            boolean requiredMaterialized
    ) {
    }

    private static final class StructuralValueFlowBuilder {
        private final GraphValueRef valueRef;
        private MaterializationDecision decision = MaterializationDecision.MATERIALIZE;
        private String producerRegionId;
        private String producerUnitId;
        private final LinkedHashSet<String> consumerRegionIds = new LinkedHashSet<>();
        private final LinkedHashSet<String> consumerUnitIds = new LinkedHashSet<>();

        private StructuralValueFlowBuilder(GraphValueRef valueRef) {
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
}
