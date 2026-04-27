package backend.metal.lowering;

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
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public final class MetalRegionLegalityAdapter implements RegionLegalityAdapter {
    private final AcceleratorSubgraphLowerer lowerer = new AcceleratorSubgraphLowerer();

    @Override
    public PartitionTarget target() {
        return PartitionTarget.GPU_METAL;
    }

    @Override
    public boolean isNodeSupported(CompiledNode node, PartitionPlanningContext context) {
        return MetalPartitionSupport.isPlannerSupported(node);
    }

    @Override
    public boolean canSeed(CompiledNode node, PartitionPlanningContext context) {
        return MetalPartitionSupport.isPlannerSupported(node);
    }

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
        if (producer.operation() == null) {
            return true;
        }
        return !MetalPartitionSupport.isPlannerSupported(producer) || producer.backend() != target().backend();
    }

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
            collectExternalInputs(context.compiledNode(nodeId), selectedNodeIds, externalInputIds);
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
        AcceleratorSubgraphLoweringResult lowering = lowerer.tryLower(subgraph, context);
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
