package graph.optimizer.partition.apple;

import backend.prepare.BackendPrepareContext;
import graph.CompiledNode;
import graph.optimizer.partition.AcceleratorPartitionPlan;
import graph.optimizer.partition.AcceleratorRegionLegalityAdapter;
import graph.optimizer.partition.AcceleratorStructuralCandidate;
import graph.optimizer.partition.AcceleratorTarget;
import graph.optimizer.partition.model.AcceleratorSubgraphOp;
import graph.optimizer.partition.model.AcceleratorSubgraphSpec;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public final class AppleGpuRegionLegalityAdapter implements AcceleratorRegionLegalityAdapter {
    private final AppleGpuSubgraphLowerer lowerer = new AppleGpuSubgraphLowerer();

    @Override
    public AcceleratorTarget target() {
        return AcceleratorTarget.GPU_METAL;
    }

    @Override
    public boolean isNodeSupported(CompiledNode node, BackendPrepareContext context) {
        return AppleGpuPartitionSupport.isPlannerSupported(node);
    }

    @Override
    public boolean canSeed(CompiledNode node, BackendPrepareContext context) {
        return AppleGpuPartitionSupport.isPlannerSupported(node)
                && AppleGpuPartitionSupport.containsMatMulFamily(node);
    }

    @Override
    public boolean canUseAsExternalInput(
            CompiledNode producer,
            CompiledNode consumer,
            Set<Integer> selectedNodeIds,
            BackendPrepareContext context
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
        return !AppleGpuPartitionSupport.isPlannerSupported(producer) || producer.backend() != target().backend();
    }

    @Override
    public AcceleratorStructuralCandidate tryCreateStructuralCandidate(
            Set<Integer> selectedNodeIds,
            BackendPrepareContext context
    ) {
        if (selectedNodeIds == null || selectedNodeIds.isEmpty()) {
            return null;
        }
        List<Integer> orderedNodeIds = selectedNodeIds.stream().sorted().toList();
        if (!containsMatMulFamily(orderedNodeIds, context)) {
            return null;
        }
        List<Integer> sinks = findSinks(selectedNodeIds, context);
        if (sinks.size() != 1) {
            return null;
        }
        Integer computeNodeId = null;
        for (int nodeId : orderedNodeIds) {
            if (AppleGpuPartitionSupport.containsMatMulFamily(context.compiledNode(nodeId))) {
                computeNodeId = nodeId;
                break;
            }
        }
        if (computeNodeId == null) {
            return null;
        }
        int anchorNodeId = sinks.getFirst();
        for (int nodeId : orderedNodeIds) {
            if (nodeId == anchorNodeId) {
                continue;
            }
            for (CompiledNode consumer : context.consumersFor(nodeId)) {
                if (consumer != null && !selectedNodeIds.contains(consumer.id())) {
                    return null;
                }
            }
        }
        LinkedHashSet<Integer> externalInputIds = new LinkedHashSet<>();
        for (int nodeId : orderedNodeIds) {
            collectExternalInputs(context.compiledNode(nodeId), selectedNodeIds, externalInputIds);
        }
        return new AcceleratorStructuralCandidate(computeNodeId, orderedNodeIds, List.copyOf(externalInputIds), anchorNodeId);
    }

    @Override
    public AcceleratorPartitionPlan tryCreatePlan(
            AcceleratorStructuralCandidate candidate,
            BackendPrepareContext context
    ) {
        if (candidate == null) {
            return null;
        }
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                candidate.computeNodeId(),
                candidate.orderedNodeIds(),
                toSubgraphOps(candidate.orderedNodeIds(), context),
                candidate.externalInputIds(),
                List.of(candidate.anchorNodeId())
        );
        AppleGpuSubgraphLoweringResult lowering = lowerer.tryLower(subgraph, context);
        if (lowering == null) {
            return null;
        }
        return new AppleGpuPartitionPlan(candidate.anchorNodeId(), subgraph, lowering);
    }

    private boolean containsMatMulFamily(List<Integer> nodeIds, BackendPrepareContext context) {
        for (int nodeId : nodeIds) {
            if (AppleGpuPartitionSupport.containsMatMulFamily(context.compiledNode(nodeId))) {
                return true;
            }
        }
        return false;
    }

    private List<Integer> findSinks(Set<Integer> selectedNodeIds, BackendPrepareContext context) {
        List<Integer> sinks = new ArrayList<>();
        for (int nodeId : selectedNodeIds) {
            boolean hasSelectedConsumer = false;
            for (CompiledNode consumer : context.consumersFor(nodeId)) {
                if (consumer != null && selectedNodeIds.contains(consumer.id())) {
                    hasSelectedConsumer = true;
                    break;
                }
            }
            if (!hasSelectedConsumer) {
                sinks.add(nodeId);
            }
        }
        return List.copyOf(sinks);
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

    private List<AcceleratorSubgraphOp> toSubgraphOps(List<Integer> nodeIds, BackendPrepareContext context) {
        List<AcceleratorSubgraphOp> out = new ArrayList<>(nodeIds.size());
        for (int nodeId : nodeIds) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null || node.operation() == null) {
                throw new IllegalStateException("Missing operation for Apple subgraph nodeId=" + nodeId);
            }
            out.add(new AcceleratorSubgraphOp(nodeId, node.operation().opType()));
        }
        return List.copyOf(out);
    }
}
