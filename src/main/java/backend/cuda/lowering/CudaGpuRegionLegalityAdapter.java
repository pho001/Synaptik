package backend.cuda.lowering;

import backend.ComputeBackend;
import backend.accelerator.lowering.GpuLoweringCoverageEntry;
import backend.accelerator.lowering.GpuLoweringCoverageMatrix;
import backend.accelerator.lowering.GpuLoweringCoverageStatus;
import backend.accelerator.lowering.AcceleratorSubgraphLowerer;
import graph.CompiledNode;
import graph.compile.planning.partition.PartitionCandidate;
import graph.compile.planning.partition.PartitionPlan;
import graph.compile.planning.partition.PartitionPlanningContext;
import graph.compile.planning.partition.PartitionTarget;
import graph.compile.planning.value.GraphValueRef;
import graph.compile.planning.partition.RegionLegalityAdapter;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import operations.Operation;
import operations.normalization.layerNorm;
import operations.normalization.rmsNorm;
import graph.compile.descriptor.CompiledTensorDescriptor;
import tensor.DataType;

import java.util.ArrayList;
import java.util.Arrays;
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
        if (CudaNnSemantics.isHandled(opType)) {
            String nnReason = CudaNnSemantics.unsupportedReason(node, context);
            if (!nnReason.isBlank()) {
                return nnReason;
            }
        }
        if (CudaPartitionSupport.isForwardIndexOp(opType)) {
            String indexReason = CudaPartitionSupport.indexUnsupportedReason(node, context);
            if (!indexReason.isBlank()) {
                return indexReason;
            }
        }
        if (CudaIndexWriteSemantics.isHandled(opType)) {
            String indexWriteReason = CudaIndexWriteSemantics.unsupportedReason(node, context);
            if (!indexWriteReason.isBlank()) {
                return indexWriteReason;
            }
        }
        GpuLoweringCoverageEntry entry = GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_CUDA, opType);
        if (entry.status() != GpuLoweringCoverageStatus.SUPPORTED) {
            return compoundPatternPrefix(opType) + GpuLoweringCoverageMatrix.plannerUnsupportedDetail(ComputeBackend.GPU_CUDA, opType);
        }
        String normalizationReason = normalizationUnsupportedReason(node, context);
        if (!normalizationReason.isBlank()) {
            return normalizationReason;
        }
        if (hasDirectNonDenseInput(node, context) && isEpilogueAdd(node, context)) {
            return "UNSUPPORTED_LAYOUT: GPU_CUDA LINEAR_BIAS_ACTIVATION family=MATMUL_LINEAR epilogue input requires dense layout";
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
            Set<GraphValueRef> requiredMaterializedValueRefs
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
            case SUM, MEAN, REDUCE_MIN, REDUCE_MAX -> "REDUCTION_ADJACENT: ";
            default -> "";
        };
    }

    private static String normalizationUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        Operation.OpType opType = node.operation().opType();
        if (opType != Operation.OpType.LAYER_NORM && opType != Operation.OpType.RMS_NORM) {
            return "";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA normalization requires planning context";
        }
        int normalizedRank;
        if (opType == Operation.OpType.LAYER_NORM && node.operation() instanceof layerNorm op) {
            normalizedRank = op.getNormalizedRank();
            if (node.inputIds().size() != 3) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA LAYER_NORM requires input, gamma, and beta";
            }
        } else if (opType == Operation.OpType.RMS_NORM && node.operation() instanceof rmsNorm op) {
            normalizedRank = op.getNormalizedRank();
            if (node.inputIds().size() != 2) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA RMS_NORM requires input and gamma";
            }
        } else {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA normalization descriptor is unavailable";
        }
        CompiledNode input = context.compiledNode(node.inputIds().get(0));
        CompiledNode gamma = context.compiledNode(node.inputIds().get(1));
        CompiledNode beta = opType == Operation.OpType.LAYER_NORM ? context.compiledNode(node.inputIds().get(2)) : null;
        if (input == null || gamma == null || (opType == Operation.OpType.LAYER_NORM && beta == null)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA normalization inputs are unavailable";
        }
        if (dataType(context, input) != DataType.FLOAT32
                || dataType(context, gamma) != DataType.FLOAT32
                || (beta != null && dataType(context, beta) != DataType.FLOAT32)
                || dataType(context, node) != DataType.FLOAT32) {
            return "UNSUPPORTED_DTYPE: GPU_CUDA normalization supports only FLOAT32";
        }
        if (!dense(context, input) || !dense(context, gamma) || (beta != null && !dense(context, beta))) {
            return "UNSUPPORTED_LAYOUT: GPU_CUDA normalization inputs require dense layout";
        }
        int[] inputShape = shape(context, input);
        int[] gammaShape = shape(context, gamma);
        int[] betaShape = beta == null ? null : shape(context, beta);
        if (inputShape.length < 1 || inputShape.length > 4
                || normalizedRank < 1
                || normalizedRank > inputShape.length
                || gammaShape.length != normalizedRank
                || !Arrays.equals(inputShape, shape(context, node))
                || (betaShape != null && !Arrays.equals(gammaShape, betaShape))) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA normalization rank/shape contract is unsupported";
        }
        int tailStart = inputShape.length - normalizedRank;
        for (int i = 0; i < normalizedRank; i++) {
            if (gammaShape[i] != inputShape[tailStart + i]) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA normalization parameter shape must match input tail";
            }
        }
        return "";
    }

    private static boolean hasDirectNonDenseInput(CompiledNode node, PartitionPlanningContext context) {
        if (node == null || node.operation() == null || context == null || node.operation().opType().category() == Operation.OpArityClass.LAYOUT) {
            return false;
        }
        for (int inputId : node.inputIds()) {
            CompiledNode input = context.compiledNode(inputId);
            if (input != null && !dense(context, input)) {
                return true;
            }
        }
        return false;
    }

    private static DataType dataType(PartitionPlanningContext context, CompiledNode node) {
        return descriptor(context, node).dataType();
    }

    private static int[] shape(PartitionPlanningContext context, CompiledNode node) {
        return descriptor(context, node).shape();
    }

    private static boolean dense(PartitionPlanningContext context, CompiledNode node) {
        return descriptor(context, node).denseContiguousWithoutOffset();
    }

    private static CompiledTensorDescriptor descriptor(PartitionPlanningContext context, CompiledNode node) {
        if (context == null) {
            throw new IllegalArgumentException("descriptor lookup requires planning context");
        }
        if (node == null) {
            throw new IllegalArgumentException("descriptor lookup requires compiled node");
        }
        return context.descriptor(node.id());
    }

    private static boolean isEpilogueAdd(CompiledNode node, PartitionPlanningContext context) {
        if (node == null || node.operation() == null || node.operation().opType() != Operation.OpType.ADD || context == null) {
            return false;
        }
        for (int inputId : node.inputIds()) {
            CompiledNode input = context.compiledNode(inputId);
            if (input != null && input.operation() != null) {
                Operation.OpType opType = input.operation().opType();
                if (opType == Operation.OpType.MATMUL || opType == Operation.OpType.LINEAR) {
                    return true;
                }
            }
        }
        return false;
    }
}
