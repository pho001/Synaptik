package backend.cuda.lowering;

import backend.ComputeBackend;
import backend.accelerator.lowering.GpuLoweringCoverageEntry;
import backend.accelerator.lowering.GpuLoweringCoverageMatrix;
import backend.accelerator.lowering.GpuLoweringCoverageStatus;
import backend.accelerator.lowering.AcceleratorSubgraphLowerer;
import graph.CompiledNode;
import graph.optimizer.partition.PartitionCandidate;
import graph.optimizer.partition.PartitionPlan;
import graph.optimizer.partition.PartitionPlanningContext;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionValueRef;
import graph.optimizer.partition.RegionLegalityAdapter;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import operations.Operation;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Partition legality adapter for CUDA accelerator graph regions.
 */
public final class CudaGpuRegionLegalityAdapter implements RegionLegalityAdapter {
    private final AcceleratorSubgraphLowerer lowerer = new AcceleratorSubgraphLowerer();

    /**
     * Returns the CUDA partition target.
     */
    @Override
    public PartitionTarget target() {
        return PartitionTarget.GPU_CUDA;
    }

    /**
     * Returns whether a compiled node can be represented in the CUDA accelerator DAG.
     */
    @Override
    public boolean isNodeSupported(CompiledNode node, PartitionPlanningContext context) {
        return plannerUnsupportedReason(node, context).isBlank();
    }

    /**
     * Returns a stable diagnostic reason when a node is not currently legal for CUDA planning.
     */
    public static String plannerUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        if (node == null) {
            return "node is null";
        }
        if (node.backend() != ComputeBackend.GPU_CUDA) {
            return "node is not assigned to GPU_CUDA";
        }
        if (node.operation() == null) {
            return "node has no operation";
        }
        if (node.inputIds().isEmpty()) {
            return "leaf nodes are external inputs, not CUDA compute nodes";
        }
        Operation.OpType opType = node.operation().opType();
        GpuLoweringCoverageEntry entry = GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_CUDA, opType);
        if (entry.status() != GpuLoweringCoverageStatus.SUPPORTED) {
            return compoundPatternPrefix(opType) + entry.reason().name() + ": operation " + opType + " is not supported by GPU_CUDA lowering";
        }
        if (hasDirectNonDenseInput(node, context)) {
            return "UNSUPPORTED_LAYOUT: direct non-dense CUDA compute remains conservative until metadata-only view propagation or dense materialization makes the consumer layout legal";
        }
        return "";
    }

    /**
     * Returns whether the node can seed a CUDA partition candidate.
     */
    @Override
    public boolean canSeed(CompiledNode node, PartitionPlanningContext context) {
        return isNodeSupported(node, context);
    }

    /**
     * Returns whether a producer outside the selected CUDA candidate may be read as an external input.
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
        if (producer.operation() == null) {
            return true;
        }
        return !isNodeSupported(producer, context);
    }

    /**
     * Builds a structurally valid CUDA partition candidate from selected node ids.
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
        LinkedHashSet<Integer> outputNodeIds = determineOutputNodeIds(selectedNodeIds, orderedNodeIds, context, requiredMaterializedValueRefs);
        if (outputNodeIds.isEmpty()) {
            return null;
        }
        int computeNodeId = orderedNodeIds.getFirst();
        for (int nodeId : orderedNodeIds) {
            if (containsMatMulFamily(context.compiledNode(nodeId))) {
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

    /**
     * Lowers a CUDA candidate into a concrete CUDA partition plan.
     */
    @Override
    public PartitionPlan tryCreatePlan(PartitionCandidate candidate, PartitionPlanningContext context) {
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
        var lowering = lowerer.tryLower(ComputeBackend.GPU_CUDA, subgraph, context);
        if (lowering == null) {
            return null;
        }
        return new CudaGpuPartitionPlan(
                candidate.anchorNodeId(),
                subgraph,
                lowering.dagSpec(),
                lowering.estimatedWork(),
                lowering.compoundSummary(),
                lowering.manifest()
        );
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
                throw new IllegalStateException("Missing operation for CUDA subgraph nodeId=" + nodeId);
            }
            out.add(new AcceleratorSubgraphOp(nodeId, node.operation().opType()));
        }
        return List.copyOf(out);
    }

    private boolean containsMatMulFamily(CompiledNode node) {
        if (node == null || node.operation() == null) {
            return false;
        }
        Operation.OpType opType = node.operation().opType();
        return opType == Operation.OpType.MATMUL || opType == Operation.OpType.LINEAR;
    }

    private static String compoundPatternPrefix(Operation.OpType opType) {
        return switch (opType) {
            case SUM, MEAN, REDUCE_MIN, REDUCE_MAX, LAYER_NORM, RMS_NORM -> "REDUCTION_ADJACENT: ";
            default -> "";
        };
    }

    private static boolean hasDirectNonDenseInput(CompiledNode node, PartitionPlanningContext context) {
        if (node == null || node.operation() == null || context == null || node.operation().opType().category() == Operation.OpArityClass.LAYOUT) {
            return false;
        }
        for (int inputId : node.inputIds()) {
            CompiledNode input = context.compiledNode(inputId);
            if (input != null && (!input.contiguous() || input.hasStorageOffset())) {
                return true;
            }
        }
        return false;
    }
}
