package backend.metal.lowering;

import backend.ComputeBackend;
import backend.accelerator.lowering.GpuLoweringCoverageEntry;
import backend.accelerator.lowering.GpuLoweringCoverageMatrix;
import backend.accelerator.lowering.GpuLoweringCoverageStatus;
import backend.metal.MetalMpsCapabilities;
import graph.CompiledNode;
import graph.optimizer.partition.PartitionPlanningContext;
import operations.Operation;
import operations.index.gather;
import operations.index.takeAlongAxis;
import operations.linalg.scaledDotProductAttention;
import operations.normalization.layerNorm;
import operations.normalization.rmsNorm;

import java.util.Arrays;

/**
 * Shared Metal partition planner predicates.
 */
public final class MetalPartitionSupport {
    private MetalPartitionSupport() {
    }

    /**
     * Returns whether a compiled node is currently supported by Metal graph lowering.
     *
     * @param node compiled node to test
     * @param context planning context; reserved for capability checks that need graph context
     * @return true when the node operation and output dtype can be represented by the Metal bridge
     */
    public static boolean isPlannerSupported(CompiledNode node, PartitionPlanningContext context) {
        return plannerUnsupportedReason(node, context).isBlank();
    }

    /**
     * Returns a stable diagnostic reason when a node is not currently legal for Metal planning.
     *
     * <p>This method is intentionally capability-oriented. It explains tested Metal planner coverage; it does not
     * estimate profitability and it does not inspect runtime tensor storage layout.</p>
     *
     * @param node compiled node to test
     * @param context planning context; reserved for capability checks that need graph context
     * @return empty string when supported, otherwise a readable rejection reason
     */
    public static String plannerUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        if (node == null) {
            return "node is null";
        }
        if (node.operation() == null) {
            return "node has no operation";
        }
        if (node.inputIds().isEmpty()) {
            return "leaf nodes are external inputs, not Metal compute nodes";
        }
        Operation.OpType opType = node.operation().opType();
        GpuLoweringCoverageEntry entry = GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_METAL, opType);
        if (entry.status() != GpuLoweringCoverageStatus.SUPPORTED
                && entry.reason() == backend.accelerator.lowering.GpuLoweringUnsupportedReason.UNSUPPORTED_DTYPE) {
            return compoundPatternPrefix(opType) + GpuLoweringCoverageMatrix.plannerUnsupportedDetail(ComputeBackend.GPU_METAL, opType);
        }
        if (!MetalMpsCapabilities.supportsComputeDType(node.dataType())
                || !MetalMpsCapabilities.supportsOutputDType(node.dataType())) {
            return MetalMpsCapabilities.unsupportedDTypeMessage(node.dataType());
        }
        var operationDTypeDecision = MetalMpsCapabilities.operationDecision(opType, node.dataType());
        if (!operationDTypeDecision.supported()) {
            return "UNSUPPORTED_DTYPE: " + operationDTypeDecision.detail();
        }
        if (isBoolCompare(opType)) {
            String boolCompareReason = boolCompareUnsupportedReason(node, context);
            if (!boolCompareReason.isBlank()) {
                return boolCompareReason;
            }
        }
        if (isBoolLogical(opType)) {
            String boolLogicalReason = boolLogicalUnsupportedReason(node, context);
            if (!boolLogicalReason.isBlank()) {
                return boolLogicalReason;
            }
        }
        if (isBoolReduction(opType)) {
            String boolReductionReason = boolReductionUnsupportedReason(node, context);
            if (!boolReductionReason.isBlank()) {
                return boolReductionReason;
            }
        }
        if (opType == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION) {
            String sdpaReason = sdpaUnsupportedReason(node, context);
            if (!sdpaReason.isBlank()) {
                return sdpaReason;
            }
        }
        if (entry.status() != GpuLoweringCoverageStatus.SUPPORTED) {
            return compoundPatternPrefix(opType) + GpuLoweringCoverageMatrix.plannerUnsupportedDetail(ComputeBackend.GPU_METAL, opType);
        }
        if (opType == Operation.OpType.GATHER || opType == Operation.OpType.TAKE_ALONG_AXIS) {
            String indexReason = indexGatherUnsupportedReason(node, context);
            if (!indexReason.isBlank()) {
                return indexReason;
            }
        }
        String normalizationReason = normalizationUnsupportedReason("GPU_METAL", node, context);
        if (!normalizationReason.isBlank()) {
            return normalizationReason;
        }
        if (hasDirectNonDenseInput(node, context) && isEpilogueAdd(node, context)) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL LINEAR_BIAS_ACTIVATION family=MATMUL_LINEAR epilogue input requires dense layout";
        }
        return "";
    }

    private static String sdpaUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        if (node.backwardNode()) {
            return "BACKWARD_CONTEXT_UNSUPPORTED: forward SDPA nodes are not legal inside Metal backward regions";
        }
        if (!(node.operation() instanceof scaledDotProductAttention attention)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SDPA descriptor is unavailable";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SDPA requires planning context";
        }
        if (!attention.hasMask() && node.inputIds().size() != 3) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL unmasked SDPA requires query, key, and value";
        }
        if (attention.hasMask() && node.inputIds().size() != 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL masked SDPA requires query, key, value, and BOOL mask";
        }
        CompiledNode query = context.compiledNode(node.inputIds().get(0));
        CompiledNode key = context.compiledNode(node.inputIds().get(1));
        CompiledNode value = context.compiledNode(node.inputIds().get(2));
        if (query == null || key == null || value == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SDPA inputs are unavailable";
        }
        if (query.dataType() != tensor.DataType.FLOAT32
                || key.dataType() != tensor.DataType.FLOAT32
                || value.dataType() != tensor.DataType.FLOAT32
                || node.dataType() != tensor.DataType.FLOAT32) {
            return "UNSUPPORTED_DTYPE: GPU_METAL SDPA supports only FLOAT32 query/key/value/output";
        }
        if (!query.contiguous() || query.hasStorageOffset()
                || !key.contiguous() || key.hasStorageOffset()
                || !value.contiguous() || value.hasStorageOffset()) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL SDPA inputs require dense layout";
        }
        int[] qShape = query.shape();
        int[] kShape = key.shape();
        int[] vShape = value.shape();
        int[] outShape = node.shape();
        if (!sdpaRankSupported(qShape) || !sdpaRankSupported(kShape) || !sdpaRankSupported(vShape) || !sdpaRankSupported(outShape)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SDPA supports rank 3 or 4 tensors";
        }
        if (qShape[qShape.length - 1] != kShape[kShape.length - 1]) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SDPA query/key head dimension mismatch";
        }
        if (kShape[kShape.length - 2] != vShape[vShape.length - 2]) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SDPA key/value sequence dimension mismatch";
        }
        if (outShape[outShape.length - 2] != qShape[qShape.length - 2]
                || outShape[outShape.length - 1] != vShape[vShape.length - 1]) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SDPA output shape mismatch";
        }
        int[] qBatch = Arrays.copyOf(qShape, qShape.length - 2);
        int[] kBatch = Arrays.copyOf(kShape, kShape.length - 2);
        int[] vBatch = Arrays.copyOf(vShape, vShape.length - 2);
        int[] outBatch = Arrays.copyOf(outShape, outShape.length - 2);
        if (!broadcastBatchMatches(outBatch, qBatch)
                || !broadcastBatchMatches(outBatch, kBatch)
                || !broadcastBatchMatches(outBatch, vBatch)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SDPA batch dimensions are not broadcast-compatible";
        }
        MetalSdpaMaskSemantics.Decision maskDecision = MetalSdpaMaskSemantics.classify(node, context);
        if (!maskDecision.supported()) {
            return maskDecision.unsupportedReason();
        }
        return "";
    }

    private static String indexGatherUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        Operation.OpType opType = node.operation().opType();
        if (node.backwardNode()) {
            return "BACKWARD_CONTEXT_UNSUPPORTED: forward " + opType + " nodes are not legal inside Metal backward regions";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " requires planning context";
        }
        if (node.inputIds().size() != 2) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " requires value and INT32 index inputs";
        }
        CompiledNode value = context.compiledNode(node.inputIds().get(0));
        CompiledNode indices = context.compiledNode(node.inputIds().get(1));
        if (value == null || indices == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " inputs are unavailable";
        }
        if (value.dataType() != tensor.DataType.FLOAT32 || node.dataType() != tensor.DataType.FLOAT32) {
            return "UNSUPPORTED_DTYPE: GPU_METAL " + opType + " currently supports only FLOAT32 value/output tensors";
        }
        if (indices.dataType() != tensor.DataType.INT32) {
            return "UNSUPPORTED_DTYPE: GPU_METAL " + opType + " index input requires INT32";
        }
        if (!value.contiguous() || value.hasStorageOffset()
                || !indices.contiguous() || indices.hasStorageOffset()) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL " + opType + " inputs require dense layout";
        }
        int[] valueShape = value.shape();
        int[] indexShape = indices.shape();
        int[] outputShape = node.shape();
        if (valueShape.length < 1 || valueShape.length > 4
                || indexShape.length < 1 || indexShape.length > 4
                || outputShape.length < 1 || outputShape.length > 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " supports rank 1..4 tensors";
        }
        int axis = indexAxis(node);
        if (axis < 0 || axis >= valueShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " axis is outside value rank";
        }
        String boundsReason = indexBoundsUnsupportedReason(indices, valueShape[axis], opType);
        if (!boundsReason.isBlank()) {
            return boundsReason;
        }
        if (opType == Operation.OpType.GATHER) {
            int[] expected = reduceShape(valueShape, axis);
            if (!Arrays.equals(indexShape, expected) || !Arrays.equals(outputShape, expected)) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER index/output shape must equal value shape without gathered axis";
            }
        } else {
            if (indexShape.length != valueShape.length || !Arrays.equals(outputShape, indexShape)) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL TAKE_ALONG_AXIS index/output rank and shape must match";
            }
            for (int i = 0; i < valueShape.length; i++) {
                if (i != axis && indexShape[i] != valueShape[i]) {
                    return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL TAKE_ALONG_AXIS non-axis dimensions must match value input";
                }
            }
        }
        return "";
    }

    private static String indexBoundsUnsupportedReason(CompiledNode indices, int axisSize, Operation.OpType opType) {
        if (axisSize <= 0) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opType + " axis size must be positive";
        }
        if (!indices.leaf()) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opType + " index bounds require a static INT32 leaf tensor";
        }
        int[] data;
        try {
            data = indices.semanticTensor().getInt32Data();
        } catch (RuntimeException ex) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opType + " index bounds require readable INT32 storage";
        }
        if (data == null) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opType + " index bounds require readable INT32 storage";
        }
        int logicalElements = indices.flatDataSize();
        if (logicalElements < 0 || logicalElements > data.length) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opType + " index bounds cannot be proven from storage";
        }
        for (int i = 0; i < logicalElements; i++) {
            int index = data[i];
            if (index < 0 || index >= axisSize) {
                return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opType + " index " + index + " is outside axis size " + axisSize;
            }
        }
        return "";
    }

    private static int indexAxis(CompiledNode node) {
        return switch (node.operation().opType()) {
            case GATHER -> node.operation() instanceof gather op ? op.getDimension() : -1;
            case TAKE_ALONG_AXIS -> node.operation() instanceof takeAlongAxis op ? op.getDimension() : -1;
            default -> -1;
        };
    }

    private static int[] reduceShape(int[] shape, int axis) {
        if (shape.length == 1) {
            return new int[]{1};
        }
        int[] reduced = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                reduced[j++] = shape[i];
            }
        }
        return reduced;
    }

    private static boolean sdpaRankSupported(int[] shape) {
        return shape != null && (shape.length == 3 || shape.length == 4);
    }

    private static boolean broadcastBatchMatches(int[] outBatch, int[] inBatch) {
        if (outBatch.length < inBatch.length) {
            return false;
        }
        int offset = outBatch.length - inBatch.length;
        for (int i = 0; i < inBatch.length; i++) {
            int in = inBatch[i];
            int out = outBatch[i + offset];
            if (in != 1 && in != out) {
                return false;
            }
        }
        return true;
    }

    /**
     * Returns whether an external producer can feed a Metal consumer at a specific input index.
     *
     * @param producer producer outside the selected candidate
     * @param consumer consumer inside the selected candidate
     * @param inputIndex input position on the consumer
     * @return true when the producer dtype is legal for that role
     */
    public static boolean isExternalInputSupported(CompiledNode producer, CompiledNode consumer, int inputIndex) {
        return MetalMpsCapabilities.supportsExternalInputRole(producer, consumer, inputIndex);
    }

    /**
     * Returns whether a node belongs to the matmul or linear operation family.
     */
    public static boolean containsMatMulFamily(CompiledNode node) {
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

    private static boolean isBoolCompare(Operation.OpType opType) {
        return switch (opType) {
            case GT, GE, LT, LE, EQ, NE -> true;
            default -> false;
        };
    }

    private static boolean isBoolLogical(Operation.OpType opType) {
        return switch (opType) {
            case LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT -> true;
            default -> false;
        };
    }

    private static boolean isBoolReduction(Operation.OpType opType) {
        return switch (opType) {
            case REDUCE_ALL, REDUCE_ANY -> true;
            default -> false;
        };
    }

    private static String boolCompareUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        if (node.dataType() != tensor.DataType.BOOL) {
            return "UNSUPPORTED_DTYPE: GPU_METAL BOOL compare output must be BOOL";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL BOOL compare requires planning context";
        }
        if (node.inputIds().size() != 2) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL BOOL compare requires two inputs";
        }
        for (int inputId : node.inputIds()) {
            CompiledNode input = context.compiledNode(inputId);
            if (input == null) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL BOOL compare inputs are unavailable";
            }
            if (input.dataType() != tensor.DataType.FLOAT32 && input.dataType() != tensor.DataType.BFLOAT16) {
                return "UNSUPPORTED_DTYPE: GPU_METAL BOOL compare inputs require FLOAT32/BFLOAT16 data";
            }
            if (!input.contiguous() || input.hasStorageOffset()) {
                return "UNSUPPORTED_LAYOUT: GPU_METAL BOOL compare inputs require dense layout";
            }
        }
        return "";
    }

    private static String boolLogicalUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        if (node.dataType() != tensor.DataType.BOOL) {
            return "UNSUPPORTED_DTYPE: GPU_METAL BOOL logical output must be BOOL";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL BOOL logical op requires planning context";
        }
        int expectedInputs = node.operation().opType() == Operation.OpType.LOGICAL_NOT ? 1 : 2;
        if (node.inputIds().size() != expectedInputs) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL BOOL logical op requires " + expectedInputs + " input(s)";
        }
        for (int inputId : node.inputIds()) {
            CompiledNode input = context.compiledNode(inputId);
            if (input == null) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL BOOL logical inputs are unavailable";
            }
            if (input.dataType() != tensor.DataType.BOOL) {
                return "UNSUPPORTED_DTYPE: GPU_METAL BOOL logical inputs require BOOL data";
            }
            if (!input.contiguous() || input.hasStorageOffset()) {
                return "UNSUPPORTED_LAYOUT: GPU_METAL BOOL logical inputs require dense layout";
            }
        }
        return "";
    }

    private static String boolReductionUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        if (node.dataType() != tensor.DataType.BOOL) {
            return "UNSUPPORTED_DTYPE: GPU_METAL BOOL reduction output must be BOOL";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL BOOL reduction requires planning context";
        }
        if (node.inputIds().size() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL BOOL reduction requires one input";
        }
        CompiledNode input = context.compiledNode(node.inputIds().getFirst());
        if (input == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL BOOL reduction input is unavailable";
        }
        if (input.dataType() != tensor.DataType.BOOL) {
            return "UNSUPPORTED_DTYPE: GPU_METAL BOOL reduction input requires BOOL data";
        }
        if (!input.contiguous() || input.hasStorageOffset()) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL BOOL reduction input requires dense layout";
        }
        int rank = input.shape().length;
        if (rank < 1 || rank > 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL BOOL reduction supports rank 1..4";
        }
        return "";
    }

    private static String normalizationUnsupportedReason(String backend, CompiledNode node, PartitionPlanningContext context) {
        Operation.OpType opType = node.operation().opType();
        if (opType != Operation.OpType.LAYER_NORM && opType != Operation.OpType.RMS_NORM) {
            return "";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: " + backend + " normalization requires planning context";
        }
        int normalizedRank;
        if (opType == Operation.OpType.LAYER_NORM && node.operation() instanceof layerNorm op) {
            normalizedRank = op.getNormalizedRank();
            if (node.inputIds().size() != 3) {
                return "UNSUPPORTED_RANK_OR_SHAPE: " + backend + " LAYER_NORM requires input, gamma, and beta";
            }
        } else if (opType == Operation.OpType.RMS_NORM && node.operation() instanceof rmsNorm op) {
            normalizedRank = op.getNormalizedRank();
            if (node.inputIds().size() != 2) {
                return "UNSUPPORTED_RANK_OR_SHAPE: " + backend + " RMS_NORM requires input and gamma";
            }
        } else {
            return "UNSUPPORTED_RANK_OR_SHAPE: " + backend + " normalization descriptor is unavailable";
        }
        CompiledNode input = context.compiledNode(node.inputIds().get(0));
        CompiledNode gamma = context.compiledNode(node.inputIds().get(1));
        CompiledNode beta = opType == Operation.OpType.LAYER_NORM ? context.compiledNode(node.inputIds().get(2)) : null;
        if (input == null || gamma == null || (opType == Operation.OpType.LAYER_NORM && beta == null)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: " + backend + " normalization inputs are unavailable";
        }
        tensor.DataType dtype = node.dataType();
        if (dtype != tensor.DataType.FLOAT32 && dtype != tensor.DataType.BFLOAT16) {
            return "UNSUPPORTED_DTYPE: " + backend + " normalization supports only FLOAT32/BFLOAT16";
        }
        if (input.dataType() != dtype
                || gamma.dataType() != dtype
                || (beta != null && beta.dataType() != dtype)) {
            return "UNSUPPORTED_DTYPE: " + backend + " normalization inputs and output must use the same dtype";
        }
        if (!input.contiguous() || input.hasStorageOffset()
                || !gamma.contiguous() || gamma.hasStorageOffset()
                || (beta != null && (!beta.contiguous() || beta.hasStorageOffset()))) {
            return "UNSUPPORTED_LAYOUT: " + backend + " normalization inputs require dense layout";
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
            return "UNSUPPORTED_RANK_OR_SHAPE: " + backend + " normalization rank/shape contract is unsupported";
        }
        int tailStart = inputShape.length - normalizedRank;
        for (int i = 0; i < normalizedRank; i++) {
            if (gammaShape[i] != inputShape[tailStart + i]) {
                return "UNSUPPORTED_RANK_OR_SHAPE: " + backend + " normalization parameter shape must match input tail";
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
            if (containsMatMulFamily(input)) {
                return true;
            }
        }
        return false;
    }
}
