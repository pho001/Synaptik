package backend.cuda.lowering;

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
        if (node == null
                || node.backend() != backend.ComputeBackend.GPU_CUDA
                || node.operation() == null
                || node.inputIds().isEmpty()) {
            return false;
        }
        if (node.backwardNode()) {
            return switch (node.operation().opType()) {
                case MATMUL, LINEAR, SOFTMAX_GRAD, LOG_SOFTMAX_GRAD, REDUCE_MIN_GRAD, REDUCE_MAX_GRAD, MIN_GRAD, MAX_GRAD -> true;
                default -> false;
            };
        }
        return switch (node.operation().opType()) {
            case MATMUL, LINEAR, ADD, SUB, MUL, DIV, RELU, TANH, FAST_TANH, SIGMOID, ABS, EXP, FAST_EXP, LOG, NEG, SQRT, INV, MUL_SCALAR, WHERE, SOFTMAX, CLAMP_MIN, CLAMP_MAX, RESHAPE, CONTIGUOUS, NOOP, PERMUTE, EXPAND_DIMS, SQUEEZE -> true;
            default -> false;
        };
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
        var lowering = lowerer.tryLower(subgraph, context);
        if (lowering == null) {
            return null;
        }
        return new CudaGpuPartitionPlan(
                candidate.anchorNodeId(),
                subgraph,
                lowering.dagSpec(),
                lowering.estimatedWork()
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
}
