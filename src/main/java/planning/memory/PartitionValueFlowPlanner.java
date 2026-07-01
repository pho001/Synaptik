package planning.memory;

import graph.model.CompiledNode;
import planning.partition.ExecutablePartitionPlan;
import planning.partition.execution.ExecutionUnit;
import planning.partition.execution.MaterializationDecision;
import planning.partition.execution.PartitionExecutionPlan;
import planning.partition.execution.PartitionExecutionValue;
import planning.partition.execution.ValueTypeContract;
import planning.value.GraphValueRef;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;

final class PartitionValueFlowPlanner {
    private PartitionValueFlowPlanner() {
    }

    static PartitionValueFlowPlan plan(MemoryPlanningInput input) {
        Objects.requireNonNull(input, "input cannot be null");
        List<ExecutablePartitionPlan> executablePartitions = input.executablePartitions();
        if (executablePartitions == null || executablePartitions.isEmpty()) {
            return PartitionValueFlowPlan.empty();
        }

        Map<Integer, Integer> graphLastUseByNodeId = buildGraphLastUseByNodeId(input);
        List<String> partitionIds = executablePartitions.stream()
                .map(executablePartition -> executablePartition.partition().partitionId())
                .toList();
        LinkedHashSet<GraphValueRef> materialized = new LinkedHashSet<>();
        LinkedHashSet<GraphValueRef> continuation = new LinkedHashSet<>();
        LinkedHashSet<GraphValueRef> virtual = new LinkedHashSet<>();
        LinkedHashMap<GraphValueRef, StructuralValueFlowBuilder> flowBuilders = new LinkedHashMap<>();
        LinkedHashMap<GraphValueRef, PartitionExecutionValueDescriptor> descriptors = new LinkedHashMap<>();
        LinkedHashMap<String, Integer> unitStepById = new LinkedHashMap<>();
        LinkedHashMap<GraphValueRef, Integer> producerStepByValue = new LinkedHashMap<>();
        int nextStep = 0;

        for (ExecutablePartitionPlan executablePartition : executablePartitions) {
            PartitionExecutionPlan executionPlan = executablePartition.executionPlan();
            String partitionId = executablePartition.partition().partitionId();
            for (PartitionExecutionValue value : executionPlan.executionValues()) {
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
                        .producerPartition(partitionId);
                descriptors.put(value.ref(), new PartitionExecutionValueDescriptor(
                        decision,
                        value.typeContract(),
                        value.producerNodeId(),
                        value.elementCount(),
                        value.requiredMaterialized()
                ));
            }
            for (ExecutionUnit unit : executionPlan.executionUnits()) {
                unitStepById.put(unit.unitId(), nextStep++);
                for (GraphValueRef outputValueRef : unit.outputValueRefs()) {
                    flowBuilders.computeIfAbsent(outputValueRef, ignored -> new StructuralValueFlowBuilder(outputValueRef))
                            .producerPartition(partitionId)
                            .producerUnit(unit.unitId());
                    producerStepByValue.put(outputValueRef, unitStepById.get(unit.unitId()));
                }
                for (GraphValueRef inputValueRef : unit.inputValueRefs()) {
                    StructuralValueFlowBuilder builder = flowBuilders.get(inputValueRef);
                    if (builder == null) {
                        continue;
                    }
                    builder.addConsumerPartition(partitionId);
                    builder.addConsumerUnit(unit.unitId());
                }
            }
        }

        List<StructuralValueFlow> valueFlows = flowBuilders.values().stream()
                .map(StructuralValueFlowBuilder::build)
                .toList();
        StructuralMemoryView structuralView = new StructuralMemoryView(
                partitionIds,
                List.copyOf(materialized),
                List.copyOf(continuation),
                List.copyOf(virtual),
                valueFlows
        );
        LinkedHashMap<GraphValueRef, PartitionValueLifetime> partitionValueLifetimes = new LinkedHashMap<>();
        LinkedHashMap<GraphValueRef, MaterializationPlanEntry> materializationPlan = new LinkedHashMap<>();
        LinkedHashMap<Integer, GraphValueRef> nodeIdToGraphValueRef = new LinkedHashMap<>();

        for (StructuralValueFlow flow : valueFlows) {
            PartitionExecutionValueDescriptor descriptor = descriptors.get(flow.valueRef());
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
            PartitionValueLifetime lifetime = new PartitionValueLifetime(
                    flow.valueRef(),
                    birthStep,
                    lastUseStep,
                    descriptor.elementCount(),
                    descriptor.decision(),
                    descriptor.typeContract(),
                    flow.producerPartitionId(),
                    flow.producerUnitId(),
                    flow.consumerPartitionIds(),
                    flow.consumerUnitIds()
            );
            partitionValueLifetimes.put(flow.valueRef(), lifetime);
            materializationPlan.put(flow.valueRef(), new MaterializationPlanEntry(
                    flow.valueRef(),
                    descriptor.decision(),
                    descriptor.requiredMaterialized(),
                    descriptor.decision() != MaterializationDecision.VIRTUALIZE
            ));
        }

        return new PartitionValueFlowPlan(
                structuralView,
                Map.copyOf(partitionValueLifetimes),
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

    private record PartitionExecutionValueDescriptor(
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
        private String producerPartitionId;
        private String producerUnitId;
        private final LinkedHashSet<String> consumerPartitionIds = new LinkedHashSet<>();
        private final LinkedHashSet<String> consumerUnitIds = new LinkedHashSet<>();

        private StructuralValueFlowBuilder(GraphValueRef valueRef) {
            this.valueRef = valueRef;
        }

        private StructuralValueFlowBuilder decision(MaterializationDecision decision) {
            this.decision = decision == null ? MaterializationDecision.MATERIALIZE : decision;
            return this;
        }

        private StructuralValueFlowBuilder producerPartition(String producerPartitionId) {
            if (producerPartitionId != null && !producerPartitionId.isBlank()) {
                this.producerPartitionId = producerPartitionId;
            }
            return this;
        }

        private StructuralValueFlowBuilder producerUnit(String producerUnitId) {
            if (producerUnitId != null && !producerUnitId.isBlank()) {
                this.producerUnitId = producerUnitId;
            }
            return this;
        }

        private StructuralValueFlowBuilder addConsumerPartition(String consumerPartitionId) {
            if (consumerPartitionId != null && !consumerPartitionId.isBlank()) {
                consumerPartitionIds.add(consumerPartitionId);
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
                    producerPartitionId,
                    producerUnitId,
                    List.copyOf(consumerPartitionIds),
                    List.copyOf(consumerUnitIds)
            );
        }
    }
}
