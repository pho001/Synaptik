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
import operations.linalg.scaledDotProductAttention;
import operations.normalization.layerNorm;
import operations.normalization.rmsNorm;
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
        String sdpaReason = sdpaUnsupportedReason(node, context);
        if (!sdpaReason.isBlank()) {
            return sdpaReason;
        }
        if (CudaPartitionSupport.isForwardIndexOp(opType)) {
            String indexReason = CudaPartitionSupport.indexUnsupportedReason(node, context);
            if (!indexReason.isBlank()) {
                return indexReason;
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

    private static String sdpaUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        if (node.operation().opType() != Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION) {
            return "";
        }
        if (!(node.operation() instanceof scaledDotProductAttention attention)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA SDPA descriptor is unavailable";
        }
        if (attention.hasMask()) {
            return "UNSUPPORTED_MASK_SEMANTICS: CUDA direct masked SDPA is not implemented; BOOL mask semantics require native evidence";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA SDPA requires planning context";
        }
        if (node.inputIds().size() != 3) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA unmasked SDPA requires query, key, and value";
        }
        CompiledNode query = context.compiledNode(node.inputIds().get(0));
        CompiledNode key = context.compiledNode(node.inputIds().get(1));
        CompiledNode value = context.compiledNode(node.inputIds().get(2));
        if (query == null || key == null || value == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA SDPA inputs are unavailable";
        }
        if (query.dataType() != DataType.FLOAT32
                || key.dataType() != DataType.FLOAT32
                || value.dataType() != DataType.FLOAT32
                || node.dataType() != DataType.FLOAT32) {
            return "UNSUPPORTED_DTYPE: GPU_CUDA SDPA supports only FLOAT32 query/key/value/output";
        }
        if (hasDirectNonDenseInput(node, context)) {
            return "UNSUPPORTED_LAYOUT: GPU_CUDA SDPA inputs require dense layout";
        }
        int[] queryShape = query.shape();
        int[] keyShape = key.shape();
        int[] valueShape = value.shape();
        int[] outputShape = node.shape();
        if (queryShape.length < 3 || queryShape.length > 4
                || keyShape.length != queryShape.length
                || valueShape.length != queryShape.length
                || outputShape.length != queryShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA SDPA supports rank 3 or 4 tensors";
        }
        if (queryShape[queryShape.length - 1] != keyShape[keyShape.length - 1]) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA SDPA query/key head dimension mismatch";
        }
        if (keyShape[keyShape.length - 2] != valueShape[valueShape.length - 2]) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA SDPA key/value sequence dimension mismatch";
        }
        if (outputShape[outputShape.length - 2] != queryShape[queryShape.length - 2]
                || outputShape[outputShape.length - 1] != valueShape[valueShape.length - 1]) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA SDPA output shape mismatch";
        }
        for (int i = 0; i < queryShape.length - 2; i++) {
            int q = queryShape[i];
            int k = keyShape[i];
            int v = valueShape[i];
            int o = outputShape[i];
            if (!broadcastCompatible(q, k) || !broadcastCompatible(q, v) || !broadcastCompatible(k, v)
                    || (o != Math.max(q, Math.max(k, v)))) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA SDPA batch dimensions are not broadcast-compatible";
            }
        }
        return "CAPABILITY_MISSING: CUDA direct forward SDPA native/lowered path is not implemented; target=transformer_block_hot_path";
    }

    private static boolean broadcastCompatible(int left, int right) {
        return left == right || left == 1 || right == 1;
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
        if (input.dataType() != DataType.FLOAT32
                || gamma.dataType() != DataType.FLOAT32
                || (beta != null && beta.dataType() != DataType.FLOAT32)
                || node.dataType() != DataType.FLOAT32) {
            return "UNSUPPORTED_DTYPE: GPU_CUDA normalization supports only FLOAT32";
        }
        if (!input.contiguous() || input.hasStorageOffset()
                || !gamma.contiguous() || gamma.hasStorageOffset()
                || (beta != null && (!beta.contiguous() || beta.hasStorageOffset()))) {
            return "UNSUPPORTED_LAYOUT: GPU_CUDA normalization inputs require dense layout";
        }
        int[] inputShape = input.shape();
        int[] gammaShape = gamma.shape();
        int[] betaShape = beta == null ? null : beta.shape();
        if (inputShape.length < 1 || inputShape.length > 4
                || normalizedRank < 1
                || normalizedRank > inputShape.length
                || gammaShape.length != normalizedRank
                || !Arrays.equals(inputShape, node.shape())
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
            if (input != null && (!input.contiguous() || input.hasStorageOffset())) {
                return true;
            }
        }
        return false;
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
