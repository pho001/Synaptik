package backend.metal.lowering;

import backend.ComputeBackend;
import backend.accelerator.lowering.AcceleratorSubgraphLowerer;
import backend.accelerator.lowering.AcceleratorSubgraphLoweringResult;
import graph.optimizer.partition.PartitionPlanningContext;
import graph.CompiledNode;
import graph.optimizer.partition.PartitionPlan;
import graph.optimizer.partition.RegionLegalityAdapter;
import graph.optimizer.partition.PartitionCandidate;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionValueRef;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import backend.accelerator.dag.AcceleratorSubgraphSpec;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Partition legality adapter for Metal accelerator graph regions.
 */
public final class MetalRegionLegalityAdapter implements RegionLegalityAdapter {
    private final AcceleratorSubgraphLowerer lowerer = new AcceleratorSubgraphLowerer();

    /**
     * Returns the Metal partition target.
     */
    @Override
    public PartitionTarget target() {
        return PartitionTarget.GPU_METAL;
    }

    /**
     * Returns whether a compiled node can be represented in the Metal accelerator DAG.
     */
    @Override
    public boolean isNodeSupported(CompiledNode node, PartitionPlanningContext context) {
        return MetalPartitionSupport.isPlannerSupported(node, context);
    }

    /**
     * Returns whether the node can seed a Metal partition candidate.
     */
    @Override
    public boolean canSeed(CompiledNode node, PartitionPlanningContext context) {
        return MetalPartitionSupport.isPlannerSupported(node, context);
    }

    /**
     * Returns whether a producer outside the selected Metal candidate may be read as an external input.
     */
    @Override
    public boolean canUseAsExternalInput(
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
        if (sameTargetSupportedProducer(producer, consumer, context)) {
            return false;
        }
        return externalInputRolesAreSupported(producer, consumer);
    }

    private boolean sameTargetSupportedProducer(CompiledNode producer, CompiledNode consumer, PartitionPlanningContext context) {
        if (isForwardValueInputToBackwardRegion(producer, consumer)) {
            return false;
        }
        return producer != null
                && producer.operation() != null
                && producer.backend() == target().backend()
                && MetalPartitionSupport.isPlannerSupported(producer, context);
    }

    private boolean isForwardValueInputToBackwardRegion(CompiledNode producer, CompiledNode consumer) {
        return producer != null
                && consumer != null
                && !producer.backwardNode()
                && consumer.backwardNode();
    }

    private boolean externalInputRolesAreSupported(CompiledNode producer, CompiledNode consumer) {
        if (producer == null || consumer == null) {
            return false;
        }
        boolean matched = false;
        List<Integer> inputIds = consumer.inputIds();
        for (int i = 0; i < inputIds.size(); i++) {
            if (inputIds.get(i) == producer.id()) {
                matched = true;
                if (!MetalPartitionSupport.isExternalInputSupported(producer, consumer, i)) {
                    return false;
                }
            }
        }
        return matched;
    }

    /**
     * Builds a structurally valid Metal partition candidate from selected node ids.
     */
    @Override
    public PartitionCandidate tryCreateStructuralCandidate(
            Set<Integer> selectedNodeIds,
            PartitionPlanningContext context,
            Set<PartitionValueRef> requiredMaterializedValueRefs
    ) {
        if (selectedNodeIds == null || selectedNodeIds.isEmpty()) {
            return null;
        }
        List<Integer> orderedNodeIds = selectedNodeIds.stream().sorted().toList();
        if (mixesForwardAndBackward(orderedNodeIds, context)) {
            return null;
        }
        for (int nodeId : orderedNodeIds) {
            if (!MetalPartitionSupport.isPlannerSupported(context.compiledNode(nodeId), context)) {
                return null;
            }
        }
        LinkedHashSet<Integer> outputNodeIds = determineOutputNodeIds(selectedNodeIds, orderedNodeIds, context, requiredMaterializedValueRefs);
        if (outputNodeIds.isEmpty()) {
            return null;
        }
        Integer computeNodeId = orderedNodeIds.getFirst();
        for (int nodeId : orderedNodeIds) {
            if (MetalPartitionSupport.containsMatMulFamily(context.compiledNode(nodeId))) {
                computeNodeId = nodeId;
                break;
            }
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
            if (!collectExternalInputs(context.compiledNode(nodeId), selectedNodeIds, externalInputIds, context)) {
                return null;
            }
        }
        return new PartitionCandidate(
                computeNodeId,
                orderedNodeIds,
                List.copyOf(externalInputIds),
                List.copyOf(outputNodeIds),
                anchorNodeId
        );
    }

    private boolean mixesForwardAndBackward(List<Integer> orderedNodeIds, PartitionPlanningContext context) {
        boolean hasForward = false;
        boolean hasBackward = false;
        for (int nodeId : orderedNodeIds) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null) {
                continue;
            }
            if (node.backwardNode()) {
                hasBackward = true;
            } else {
                hasForward = true;
            }
            if (hasForward && hasBackward) {
                return true;
            }
        }
        return false;
    }

    /**
     * Lowers a Metal candidate into a concrete Metal partition plan.
     */
    @Override
    public PartitionPlan tryCreatePlan(
            PartitionCandidate candidate,
            PartitionPlanningContext context
    ) {
        if (candidate == null) {
            return null;
        }
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                candidate.computeNodeId(),
                candidate.orderedNodeIds(),
                toSubgraphOps(candidate.orderedNodeIds(), context),
                candidate.externalInputIds(),
                candidate.outputNodeIds()
        );
        AcceleratorSubgraphLoweringResult lowering = lowerer.tryLower(ComputeBackend.GPU_METAL, subgraph, context);
        if (lowering == null) {
            return null;
        }
        return new MetalPartitionPlan(candidate.anchorNodeId(), subgraph, lowering);
    }

    private LinkedHashSet<Integer> determineOutputNodeIds(
            Set<Integer> selectedNodeIds,
            List<Integer> orderedNodeIds,
            PartitionPlanningContext context,
            Set<PartitionValueRef> requiredMaterializedValueRefs
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
                if (requiredMaterializedValueRefs.contains(PartitionValueRef.ofNode(nodeId))) {
                    outputs.add(nodeId);
                }
            }
        }
        return outputs;
    }

    private boolean collectExternalInputs(
            CompiledNode node,
            Set<Integer> candidateNodeIds,
            Set<Integer> externalInputIds,
            PartitionPlanningContext context
    ) {
        if (node == null) {
            return false;
        }
        List<Integer> inputIds = node.inputIds();
        for (int i = 0; i < inputIds.size(); i++) {
            int inputId = inputIds.get(i);
            if (!candidateNodeIds.contains(inputId)) {
                CompiledNode producer = context.compiledNode(inputId);
                if (sameTargetSupportedProducer(producer, node, context)) {
                    return false;
                }
                if (!MetalPartitionSupport.isExternalInputSupported(producer, node, i)) {
                    return false;
                }
                externalInputIds.add(inputId);
            }
        }
        return true;
    }

    private List<AcceleratorSubgraphOp> toSubgraphOps(List<Integer> nodeIds, PartitionPlanningContext context) {
        List<AcceleratorSubgraphOp> out = new ArrayList<>(nodeIds.size());
        for (int nodeId : nodeIds) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null || node.operation() == null) {
                throw new IllegalStateException("Missing operation for Metal subgraph nodeId=" + nodeId);
            }
            out.add(new AcceleratorSubgraphOp(nodeId, node.operation().opType()));
        }
        return List.copyOf(out);
    }
}
