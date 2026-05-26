package backend.metal.lowering;

import backend.ComputeBackend;
import backend.accelerator.lowering.AcceleratorSubgraphLowerer;
import backend.accelerator.lowering.AcceleratorSubgraphLoweringResult;
import config.compile.BackendPlanningCostConfig;
import config.compile.TransferCostPreset;
import graph.compile.planning.partition.PartitionPlanningContext;
import graph.CompiledNode;
import graph.compile.planning.partition.PartitionPlan;
import graph.compile.planning.partition.BackendPartitionCapability;
import graph.compile.planning.partition.PartitionCandidate;
import graph.compile.planning.partition.PartitionTarget;
import graph.compile.planning.partition.cost.AcceleratorPartitionScoreModel;
import graph.compile.planning.value.GraphValueRef;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import operations.Operation;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Partition capability for Metal accelerator graph regions.
 */
public final class MetalBackendPartitionCapability implements BackendPartitionCapability {
    private final AcceleratorSubgraphLowerer lowerer = new AcceleratorSubgraphLowerer();

    /**
     * Returns the Metal partition target.
     */
    @Override
    public PartitionTarget target() {
        return PartitionTarget.GPU_METAL;
    }

    @Override
    public AcceleratorPartitionScoreModel.StaticCostPreset costPreset(BackendPlanningCostConfig costConfig) {
        TransferCostPreset model = costConfig == null
                ? TransferCostPreset.CONSERVATIVE
                : costConfig.planningCostProfile().transferCostPreset();
        return switch (model) {
            case CONSERVATIVE -> AcceleratorPartitionScoreModel.StaticCostPreset.conservative();
            case MEASURED -> AcceleratorPartitionScoreModel.StaticCostPreset.measured();
            case AGGRESSIVE -> AcceleratorPartitionScoreModel.StaticCostPreset.aggressive();
        };
    }

    /**
     * Returns whether a compiled node can be represented in the Metal accelerator DAG.
     */
    @Override
    public boolean canExecute(CompiledNode node, PartitionPlanningContext context) {
        return MetalPartitionSupport.isPlannerSupported(node, context);
    }

    /**
     * Returns whether the node can seed a Metal partition candidate.
     */
    @Override
    public boolean canSeed(CompiledNode node, PartitionPlanningContext context) {
        return MetalPartitionSupport.isPlannerSupported(node, context);
    }

    @Override
    public int partitionPriority(CompiledNode node, PartitionPlanningContext context) {
        if (node == null || node.operation() == null || node.operation().opType() == null) {
            return 0;
        }
        return operationPriority(node.operation().opType());
    }

    /**
     * Returns whether a producer outside the selected Metal candidate may be read as an external input.
     */
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
        if (sameTargetSupportedProducer(producer, consumer, context)) {
            return false;
        }
        return externalInputRolesAreSupported(producer, consumer);
    }

    private boolean sameTargetSupportedProducer(CompiledNode producer, CompiledNode consumer, PartitionPlanningContext context) {
        return producer != null
                && consumer != null
                && producer.backwardNode() == consumer.backwardNode()
                && producer.operation() != null
                && producer.backend() == target().backend()
                && MetalPartitionSupport.isPlannerSupported(producer, context);
    }

    private static int operationPriority(Operation.OpType opType) {
        return switch (opType) {
            case SCALED_DOT_PRODUCT_ATTENTION_BACKWARD -> 10_000;
            case SCALED_DOT_PRODUCT_ATTENTION -> 9_500;
            case CONV2D -> 9_000;
            case MATMUL, LINEAR -> 8_500;
            case CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES, CROSS_ENTROPY_LOSS_INDICES_GRAD,
                    NLL_LOSS -> 8_000;
            case LAYER_NORM, RMS_NORM -> 7_500;
            case SOFTMAX, LOG_SOFTMAX, SOFTMAX_GRAD, LOG_SOFTMAX_GRAD -> 7_000;
            case SUM, MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_MIN_GRAD, REDUCE_MAX_GRAD -> 6_500;
            case MAX_POOL2D, AVG_POOL2D -> 6_000;
            case ADD, SUB, MUL, DIV, MIN, MAX, RELU, TANH, FAST_TANH, SIGMOID, EXP, FAST_EXP,
                    ERF, LOG, SQRT, NEG, ABS, FLOOR, CEIL, SIGN, INV, POW, MUL_SCALAR -> 4_000;
            case RESHAPE, PERMUTE, CONTIGUOUS, EXPAND, EXPAND_DIMS, SQUEEZE, SELECT, SLICE, CONCAT,
                    UNFOLD_AXIS, UNFOLD2D, FOLD2D, NOOP -> 1_000;
            case SLICE_GRAD, SLICE_SCATTER_ADD, GATHER_AXIS, GATHER_AXIS_GRAD, GATHER_ND, GATHER_ND_GRAD,
                    SCATTER_AXIS_ADD -> 2_000;
            default -> 2_000;
        };
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
    public PartitionCandidate createCandidate(
            Set<Integer> selectedNodeIds,
            PartitionPlanningContext context,
            Set<GraphValueRef> requiredMaterializedValueRefs
    ) {
        if (selectedNodeIds == null || selectedNodeIds.isEmpty()) {
            return null;
        }
        List<Integer> orderedNodeIds = selectedNodeIds.stream().sorted().toList();
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
        if (hasExternalConsumerBeforeAnchor(outputNodeIds, selectedNodeIds, anchorNodeId, context)) {
            return null;
        }
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

    /**
     * Lowers a Metal candidate into a concrete Metal partition plan.
     */
    @Override
    public PartitionPlan createPlan(
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
                if (requiredMaterializedValueRefs.contains(GraphValueRef.node(nodeId))
                        && requiredMaterializationMustLeaveRegion(nodeId, selectedNodeIds, context)) {
                    outputs.add(nodeId);
                }
            }
        }
        addHiddenBackwardContextOutputs(outputs, selectedNodeIds, orderedNodeIds, context);
        return outputs;
    }

    private boolean requiredMaterializationMustLeaveRegion(
            int nodeId,
            Set<Integer> selectedNodeIds,
            PartitionPlanningContext context
    ) {
        boolean hasSelectedConsumer = false;
        for (CompiledNode consumer : context.consumersFor(nodeId)) {
            if (consumer == null) {
                continue;
            }
            if (selectedNodeIds.contains(consumer.id())) {
                hasSelectedConsumer = true;
            } else {
                return true;
            }
        }
        if (!hasSelectedConsumer) {
            return true;
        }
        CompiledNode node = context.compiledNode(nodeId);
        return node == null || !hasZeroStride(node.strides());
    }

    private boolean hasZeroStride(int[] strides) {
        if (strides == null) {
            return false;
        }
        for (int stride : strides) {
            if (stride == 0) {
                return true;
            }
        }
        return false;
    }

    private void addHiddenBackwardContextOutputs(
            LinkedHashSet<Integer> outputs,
            Set<Integer> selectedNodeIds,
            List<Integer> orderedNodeIds,
            PartitionPlanningContext context
    ) {
        for (int nodeId : orderedNodeIds) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null || node.operation() == null) {
                continue;
            }
            if (node.operation().opType() != operations.Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION) {
                continue;
            }
            for (int inputId : node.inputIds()) {
                if (selectedNodeIds.contains(inputId)) {
                    outputs.add(inputId);
                }
            }
        }
    }

    private boolean hasExternalConsumerBeforeAnchor(
            Set<Integer> outputNodeIds,
            Set<Integer> selectedNodeIds,
            int anchorNodeId,
            PartitionPlanningContext context
    ) {
        for (int outputNodeId : outputNodeIds) {
            for (CompiledNode consumer : context.consumersFor(outputNodeId)) {
                if (consumer != null && !selectedNodeIds.contains(consumer.id()) && consumer.id() < anchorNodeId) {
                    return true;
                }
            }
        }
        return false;
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
