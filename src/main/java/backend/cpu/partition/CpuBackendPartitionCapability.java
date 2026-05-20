package backend.cpu.partition;

import backend.ComputeBackend;
import graph.CompiledNode;
import graph.compile.planning.partition.PartitionCandidate;
import graph.compile.planning.partition.PartitionPlan;
import graph.compile.planning.partition.PartitionPlanningContext;
import graph.compile.planning.partition.PartitionTarget;
import graph.compile.planning.value.GraphValueRef;
import graph.compile.planning.partition.BackendPartitionCapability;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public final class CpuBackendPartitionCapability implements BackendPartitionCapability {
    @Override
    public PartitionTarget target() {
        return PartitionTarget.CPU;
    }

    @Override
    public boolean canExecute(CompiledNode node, PartitionPlanningContext context) {
        return node != null
                && node.backend() == ComputeBackend.CPU
                && node.operation() != null
                && node.inputIds() != null
                && !node.inputIds().isEmpty()
                && isStructurallySupported(node)
                && compareSelectFamilyAllowed(node, context)
                && normalizationFamilyAllowed(node, context)
                && poolFamilyAllowed(node, context)
                && backwardSpecialFamilyAllowed(node, context)
                && inputsAreStructurallyCompatible(node, context);
    }

    @Override
    public boolean canSeed(CompiledNode node, PartitionPlanningContext context) {
        return canExecute(node, context);
    }

    @Override
    public boolean canUseExternalInput(
            CompiledNode producer,
            CompiledNode consumer,
            Set<Integer> selectedNodeIds,
            PartitionPlanningContext context
    ) {
        if (producer == null) {
            return false;
        }
        if (selectedNodeIds.contains(producer.id())) {
            return true;
        }
        return producer.operation() == null || !canExecute(producer, context);
    }

    @Override
    public PartitionCandidate createCandidate(
            Set<Integer> selectedNodeIds,
            PartitionPlanningContext context,
            Set<GraphValueRef> requiredMaterializedValueRefs
    ) {
        if (selectedNodeIds == null || selectedNodeIds.isEmpty()) {
            return null;
        }
        List<Integer> orderedNodeIds = selectedNodeIds.stream().sorted().toList();
        Integer computeNodeId = orderedNodeIds.getFirst();
        LinkedHashSet<Integer> outputNodeIds = determineOutputNodeIds(selectedNodeIds, orderedNodeIds, context, requiredMaterializedValueRefs);
        if (outputNodeIds.isEmpty()) {
            return null;
        }
        int anchorNodeId = outputNodeIds.stream().max(Integer::compareTo).orElseThrow();
        for (int nodeId : orderedNodeIds) {
            for (CompiledNode consumer : context.consumersFor(nodeId)) {
                if (consumer != null && !selectedNodeIds.contains(consumer.id()) && !outputNodeIds.contains(nodeId)) {
                    return null;
                }
            }
        }
        LinkedHashSet<Integer> externalInputIds = new LinkedHashSet<>();
        for (int nodeId : orderedNodeIds) {
            collectExternalInputs(context.compiledNode(nodeId), selectedNodeIds, externalInputIds);
        }
        long estimatedWork = orderedNodeIds.stream()
                .map(context::compiledNode)
                .filter(java.util.Objects::nonNull)
                .mapToLong(CompiledNode::flatDataSize)
                .sum();
        if (estimatedWork <= 0L) {
            estimatedWork = orderedNodeIds.size();
        }
        return new PartitionCandidate(
                computeNodeId,
                orderedNodeIds,
                List.copyOf(externalInputIds),
                List.copyOf(outputNodeIds),
                anchorNodeId
        );
    }

    @Override
    public PartitionPlan createPlan(PartitionCandidate candidate, PartitionPlanningContext context) {
        if (candidate == null) {
            return null;
        }
        long estimatedWork = candidate.orderedNodeIds().stream()
                .map(context::compiledNode)
                .filter(java.util.Objects::nonNull)
                .mapToLong(CompiledNode::flatDataSize)
                .sum();
        if (estimatedWork <= 0L) {
            estimatedWork = candidate.orderedNodeIds().size();
        }
        return new CpuPartitionPlan(
                candidate.anchorNodeId(),
                candidate.orderedNodeIds(),
                candidate.externalInputIds(),
                candidate.outputNodeIds(),
                estimatedWork
        );
    }

    private LinkedHashSet<Integer> determineOutputNodeIds(
            Set<Integer> selectedNodeIds,
            List<Integer> orderedNodeIds,
            PartitionPlanningContext context,
            Set<GraphValueRef> requiredMaterializedValueRefs
    ) {
        LinkedHashSet<Integer> outputs = new LinkedHashSet<>();
        for (int nodeId : selectedNodeIds) {
            boolean hasSelectedConsumer = false;
            boolean hasExternalConsumer = false;
            for (CompiledNode consumer : context.consumersFor(nodeId)) {
                if (consumer != null && selectedNodeIds.contains(consumer.id())) {
                    hasSelectedConsumer = true;
                } else if (consumer != null) {
                    hasExternalConsumer = true;
                }
            }
            if (!hasSelectedConsumer || hasExternalConsumer) {
                outputs.add(nodeId);
            }
        }
        if (requiredMaterializedValueRefs != null) {
            for (int nodeId : orderedNodeIds) {
                if (requiredMaterializedValueRefs.contains(GraphValueRef.node(nodeId))) {
                    outputs.add(nodeId);
                }
            }
        }
        return outputs;
    }

    private void collectExternalInputs(CompiledNode node, Set<Integer> candidateNodeIds, Set<Integer> externalInputIds) {
        if (node == null) {
            return;
        }
        for (int inputId : node.inputIds()) {
            if (!candidateNodeIds.contains(inputId)) {
                externalInputIds.add(inputId);
            }
        }
    }

    private boolean isStructurallySupported(CompiledNode node) {
        if (node == null || node.operation() == null || node.operation().opType() == null) {
            return false;
        }
        return !isStructurallyExcludedFamily(node.operation().opType());
    }

    private boolean inputsAreStructurallyCompatible(CompiledNode node, PartitionPlanningContext context) {
        if (node == null || node.operation() == null || node.operation().opType() == null) {
            return false;
        }
        return switch (node.operation().opType()) {
            case SUM, MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_ALL, REDUCE_ANY ->
                    node.inputIds().stream()
                            .map(context::compiledNode)
                            .filter(java.util.Objects::nonNull)
                            .allMatch(producer -> producer.operation() == null || isStructurallySupported(producer));
            default -> true;
        };
    }

    private boolean compareSelectFamilyAllowed(CompiledNode node, PartitionPlanningContext context) {
        if (node == null || node.operation() == null || node.operation().opType() == null) {
            return false;
        }
        return true;
    }

    private boolean normalizationFamilyAllowed(CompiledNode node, PartitionPlanningContext context) {
        if (node == null || node.operation() == null || node.operation().opType() == null) {
            return false;
        }
        return true;
    }

    private boolean poolFamilyAllowed(CompiledNode node, PartitionPlanningContext context) {
        if (node == null || node.operation() == null || node.operation().opType() == null) {
            return false;
        }
        return true;
    }

    private boolean backwardSpecialFamilyAllowed(CompiledNode node, PartitionPlanningContext context) {
        if (node == null || node.operation() == null || node.operation().opType() == null) {
            return false;
        }
        return true;
    }

    /**
     * CPU structural region coverage inventory.
     *
     * Families listed here are intentionally excluded from CPU structural region coverage because
     * they are graph/runtime boundary markers rather than meaningful region compute content.
     */
    private boolean isStructurallyExcludedFamily(operations.Operation.OpType opType) {
        if (opType == null) {
            return true;
        }
        return switch (opType) {
            // Not meaningful structural CPU region content.
            case NOOP, FUSED -> true;

            default -> false;
        };
    }
}
