package backend.metal.lowering;


import backend.contract.ComputeBackend;
import backend.accelerator.lowering.GpuLoweringCoverageEntry;
import backend.accelerator.lowering.GpuLoweringCoverageMatrix;
import backend.accelerator.lowering.GpuLoweringCoverageStatus;
import backend.metal.MetalCastPolicy;
import backend.metal.MetalMpsCapabilities;
import graph.model.CompiledNode;
import planning.descriptor.CompiledTensorDescriptor;
import planning.partition.PartitionPlanningContext;
import operations.Operation;
import operations.index.gather;
import operations.index.gatherAxis;
import operations.index.gatherNd;
import operations.index.takeAlongAxis;
import operations.linalg.scaledDotProductAttention;
import operations.layout.concat;
import operations.layout.fold2d;
import operations.layout.pad;
import operations.layout.slice;
import operations.layout.sliceBackward;
import operations.layout.tile;
import operations.layout.unfold2d;
import operations.layout.unfoldAxis;
import operations.normalization.layerNorm;
import operations.normalization.rmsNorm;
import operations.reduction.argMax;
import operations.reduction.cumSum;
import operations.reduction.reduceProd;
import tensor.options.Window2dOptions;

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
        if (opType == Operation.OpType.CAST) {
            String castReason = castUnsupportedReason(node, context);
            if (!castReason.isBlank()) {
                return castReason;
            }
            return "";
        }
        var operationDTypeDecision = MetalMpsCapabilities.operationDecision(opType, node.dataType());
        if (!MetalMpsCapabilities.supportsComputeDType(node.dataType())
                || !MetalMpsCapabilities.supportsOutputDType(node.dataType())) {
            if (node.dataType() == tensor.DataType.INT32 || node.dataType() == tensor.DataType.INT64) {
                return "UNSUPPORTED_DTYPE: " + operationDTypeDecision.detail();
            }
            return MetalMpsCapabilities.unsupportedDTypeMessage(node.dataType());
        }
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
        if (opType == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD) {
            String sdpaBackwardReason = sdpaBackwardUnsupportedReason(node, context);
            if (!sdpaBackwardReason.isBlank()) {
                return sdpaBackwardReason;
            }
        }
        if (opType == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS) {
            String weightsReason = sdpaWeightsUnsupportedReason(node, context);
            if (!weightsReason.isBlank()) {
                return weightsReason;
            }
        }
        if (MetalConvPoolSemantics.isForwardConvPool(opType)) {
            String convPoolReason = MetalConvPoolSemantics.unsupportedReason(node, context);
            if (!convPoolReason.isBlank()) {
                return convPoolReason;
            }
        }
        if (MetalIndexWriteSemantics.isIndexWriteOrGradient(opType)) {
            String indexWriteReason = MetalIndexWriteSemantics.unsupportedReason(node, context);
            if (!indexWriteReason.isBlank()) {
                return indexWriteReason;
            }
        }
        if (MetalLossSemantics.isDenseLoss(opType)) {
            String denseLossReason = MetalLossSemantics.unsupportedReason(node, context);
            if (!denseLossReason.isBlank()) {
                return denseLossReason;
            }
        }
        if (entry.status() != GpuLoweringCoverageStatus.SUPPORTED) {
            return compoundPatternPrefix(opType) + GpuLoweringCoverageMatrix.plannerUnsupportedDetail(ComputeBackend.GPU_METAL, opType);
        }
        if (isReductionScanParityOp(opType)) {
            String reductionReason = reductionScanUnsupportedReason(node, context);
            if (!reductionReason.isBlank()) {
                return reductionReason;
            }
        }
        if (isSupportedLayoutOp(opType)) {
            String layoutReason = layoutUnsupportedReason(node, context);
            if (!layoutReason.isBlank()) {
                return layoutReason;
            }
        }
        if (isUnaryMathParityOp(opType)) {
            String unaryReason = unaryMathUnsupportedReason(node, context);
            if (!unaryReason.isBlank()) {
                return unaryReason;
            }
        }
        if (opType == Operation.OpType.GATHER || opType == Operation.OpType.GATHER_AXIS || opType == Operation.OpType.TAKE_ALONG_AXIS) {
            String indexReason = indexGatherUnsupportedReason(node, context);
            if (!indexReason.isBlank()) {
                return indexReason;
            }
        }
        if (opType == Operation.OpType.GATHER_ND) {
            String indexReason = gatherNdUnsupportedReason(node, context);
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

    private static String sdpaWeightsUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SDPA weights publication requires planning context";
        }
        if (node.inputIds().size() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SDPA weights publication requires one SDPA output input";
        }
        CompiledNode attention = context.compiledNode(node.inputIds().getFirst());
        if (attention == null
                || attention.operation() == null
                || attention.operation().opType() != Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SDPA weights publication requires a direct SDPA producer";
        }
        String attentionReason = sdpaUnsupportedReason(attention, context);
        if (!attentionReason.isBlank()) {
            return attentionReason;
        }
        tensor.DataType attentionDType = dataType(context, attention);
        if (!isMetalFloatingDType(attentionDType)
                || dataType(context, node) != attentionDType) {
            return "UNSUPPORTED_DTYPE: GPU_METAL SDPA weights publication requires dtype-matched FLOAT32/BFLOAT16 attention and output";
        }
        int[] outputShape = shape(context, node);
        int[] queryShape = shape(context, context.compiledNode(attention.inputIds().get(0)));
        int[] keyShape = shape(context, context.compiledNode(attention.inputIds().get(1)));
        int[] expected = expectedScoresShape(queryShape, keyShape);
        if (expected.length == 0 || !Arrays.equals(outputShape, expected)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SDPA weights output shape must equal broadcasted score shape";
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
        tensor.DataType dtype = dataType(context, node);
        if (!isMetalFloatingDType(dtype)
                || dataType(context, query) != dtype
                || dataType(context, key) != dtype
                || dataType(context, value) != dtype) {
            return "UNSUPPORTED_DTYPE: GPU_METAL SDPA supports dtype-matched FLOAT32/BFLOAT16 query/key/value/output";
        }
        if (!sdpaInputLayoutSupported(context, query)
                || !sdpaInputLayoutSupported(context, key)
                || !sdpaInputLayoutSupported(context, value)) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL SDPA inputs require dense layout or GPU-side layout legalization";
        }
        int[] qShape = shape(context, query);
        int[] kShape = shape(context, key);
        int[] vShape = shape(context, value);
        int[] outShape = shape(context, node);
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

    private static String sdpaBackwardUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        if (!node.backwardNode()) {
            return "BACKWARD_CONTEXT_UNSUPPORTED: GPU_METAL SDPA backward nodes must live in a backward region";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SDPA backward requires planning context";
        }
        if (node.inputIds().size() != 2) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SDPA backward requires attention output and output gradient";
        }
        CompiledNode attentionOut = context.compiledNode(node.inputIds().get(0));
        CompiledNode outGrad = context.compiledNode(node.inputIds().get(1));
        if (attentionOut == null
                || attentionOut.operation() == null
                || attentionOut.operation().opType() != Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION
                || outGrad == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SDPA backward requires a direct SDPA producer and output gradient";
        }
        if (attentionOut.inputIds().size() != 3 && attentionOut.inputIds().size() != 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SDPA backward requires a 3-input or 4-input SDPA producer";
        }
        MetalSdpaMaskSemantics.Decision maskDecision = MetalSdpaMaskSemantics.classify(attentionOut, context);
        if (!maskDecision.supported()) {
            return maskDecision.unsupportedReason();
        }
        CompiledNode query = context.compiledNode(attentionOut.inputIds().get(0));
        CompiledNode key = context.compiledNode(attentionOut.inputIds().get(1));
        CompiledNode value = context.compiledNode(attentionOut.inputIds().get(2));
        if (query == null || key == null || value == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SDPA backward query/key/value inputs are unavailable";
        }
        tensor.DataType dtype = dataType(context, node);
        if (!isMetalFloatingDType(dtype)
                || dataType(context, query) != dtype
                || dataType(context, key) != dtype
                || dataType(context, value) != dtype
                || dataType(context, outGrad) != dtype) {
            return "UNSUPPORTED_DTYPE: GPU_METAL SDPA backward supports dtype-matched FLOAT32/BFLOAT16 query/key/value/outGrad/output";
        }
        if (!sdpaInputLayoutSupported(context, query)
                || !sdpaInputLayoutSupported(context, key)
                || !sdpaInputLayoutSupported(context, value)
                || !sdpaInputLayoutSupported(context, outGrad)) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL SDPA backward inputs require dense layout or GPU-side layout legalization";
        }
        int[] qShape = shape(context, query);
        int[] kShape = shape(context, key);
        int[] vShape = shape(context, value);
        int[] outGradShape = shape(context, outGrad);
        int[] outputShape = shape(context, node);
        if (!sdpaRankSupported(qShape)
                || !sdpaRankSupported(kShape)
                || !sdpaRankSupported(vShape)
                || !sdpaRankSupported(outGradShape)
                || !sdpaRankSupported(outputShape)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SDPA backward supports rank 3 or 4 tensors";
        }
        if (!Arrays.equals(outGradShape, shape(context, attentionOut))) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SDPA backward output gradient shape must match attention output";
        }
        return "";
    }

    private static boolean sdpaInputLayoutSupported(PartitionPlanningContext context, CompiledNode node) {
        if (dense(context, node)) {
            return true;
        }
        return isGpuSideLayoutLegalizationProducer(node);
    }

    private static boolean isGpuSideLayoutLegalizationProducer(CompiledNode node) {
        if (node == null || node.operation() == null) {
            return false;
        }
        return switch (node.operation().opType()) {
            case RESHAPE, PERMUTE, CONTIGUOUS, EXPAND, EXPAND_DIMS, SQUEEZE, SLICE, SLICE_BACKWARD, PAD, TILE,
                 UNFOLD_AXIS, UNFOLD2D, FOLD2D -> true;
            default -> false;
        };
    }

    private static boolean layoutInputSupported(PartitionPlanningContext context, CompiledNode node) {
        return dense(context, node) || isGpuSideLayoutLegalizationProducer(node);
    }

    private static int[] expectedScoresShape(int[] queryShape, int[] keyShape) {
        if (queryShape == null || keyShape == null || queryShape.length < 2 || keyShape.length < 2) {
            return new int[0];
        }
        int[] qBatch = Arrays.copyOf(queryShape, queryShape.length - 2);
        int[] kBatch = Arrays.copyOf(keyShape, keyShape.length - 2);
        int[] batch = broadcastBatchShape(qBatch, kBatch);
        if (batch == null) {
            return new int[0];
        }
        int[] out = Arrays.copyOf(batch, batch.length + 2);
        out[out.length - 2] = queryShape[queryShape.length - 2];
        out[out.length - 1] = keyShape[keyShape.length - 2];
        return out;
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
        tensor.DataType dtype = dataType(context, node);
        if (!isMetalFloatingDType(dtype) || dataType(context, value) != dtype) {
            return "UNSUPPORTED_DTYPE: GPU_METAL " + opType + " requires dtype-matched FLOAT32/BFLOAT16 value/output tensors";
        }
        if (dataType(context, indices) != tensor.DataType.INT32) {
            return "UNSUPPORTED_DTYPE: GPU_METAL " + opType + " index input requires INT32";
        }
        if (!dense(context, value) || !dense(context, indices)) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL " + opType + " inputs require dense layout";
        }
        int[] valueShape = shape(context, value);
        int[] indexShape = shape(context, indices);
        int[] outputShape = shape(context, node);
        if (valueShape.length < 1 || valueShape.length > 4
                || indexShape.length < 1 || indexShape.length > 4
                || outputShape.length < 1 || outputShape.length > 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " supports rank 1..4 tensors";
        }
        int axis = indexAxis(node);
        if (axis < 0 || axis >= valueShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " axis is outside value rank";
        }
        String boundsReason = indexBoundsUnsupportedReason(indices, context, valueShape[axis], opType);
        if (!boundsReason.isBlank()) {
            return boundsReason;
        }
        if (opType == Operation.OpType.GATHER) {
            int[] expected = reduceShape(valueShape, axis);
            if (!Arrays.equals(indexShape, expected) || !Arrays.equals(outputShape, expected)) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER index/output shape must equal value shape without gathered axis";
            }
        } else if (opType == Operation.OpType.GATHER_AXIS) {
            if (!(node.operation() instanceof gatherAxis) || indexShape.length != 1 || outputShape.length != valueShape.length) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_AXIS supports 1-D index tensors that preserve value rank";
            }
            for (int i = 0; i < valueShape.length; i++) {
                int expected = i == axis ? indexShape[0] : valueShape[i];
                if (outputShape[i] != expected) {
                    return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_AXIS output shape must equal value shape with gathered axis replaced by index length";
                }
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

    private static String indexBoundsUnsupportedReason(CompiledNode indices, PartitionPlanningContext context, int axisSize, Operation.OpType opType) {
        if (axisSize <= 0) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opType + " axis size must be positive";
        }
        if (!indices.leaf()) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opType + " index bounds require a static INT32 leaf tensor";
        }
        int[] data = indices.staticDataSnapshot().int32Values();
        if (data == null) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opType + " index bounds require readable INT32 storage";
        }
        long logicalElements = context.descriptor(indices.id()).logicalElementCount();
        if (logicalElements < 0 || logicalElements > data.length || logicalElements > Integer.MAX_VALUE) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opType + " index bounds cannot be proven from storage";
        }
        for (int i = 0; i < (int) logicalElements; i++) {
            int index = data[i];
            if (index < 0 || index >= axisSize) {
                return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opType + " index " + index + " is outside axis size " + axisSize;
            }
        }
        return "";
    }

    private static String gatherNdUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        Operation.OpType opType = Operation.OpType.GATHER_ND;
        if (node.backwardNode()) {
            return "BACKWARD_CONTEXT_UNSUPPORTED: forward GATHER_ND nodes are not legal inside Metal backward regions";
        }
        if (!(node.operation() instanceof gatherNd gatherOp)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_ND descriptor is unavailable";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_ND requires planning context";
        }
        if (node.inputIds().size() != 2) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_ND requires value and INT32 index inputs";
        }
        CompiledNode value = context.compiledNode(node.inputIds().get(0));
        CompiledNode indices = context.compiledNode(node.inputIds().get(1));
        if (value == null || indices == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_ND inputs are unavailable";
        }
        tensor.DataType dtype = dataType(context, node);
        if (!isMetalFloatingDType(dtype) || dataType(context, value) != dtype) {
            return "UNSUPPORTED_DTYPE: GPU_METAL GATHER_ND requires dtype-matched FLOAT32/BFLOAT16 value/output tensors";
        }
        if (dataType(context, indices) != tensor.DataType.INT32) {
            return "UNSUPPORTED_DTYPE: GPU_METAL GATHER_ND index input requires INT32";
        }
        if (!dense(context, value) || !dense(context, indices)) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL GATHER_ND inputs require dense layout";
        }
        int[] valueShape = shape(context, value);
        int[] indexShape = shape(context, indices);
        int[] outputShape = shape(context, node);
        if (valueShape.length < 1 || valueShape.length > 4
                || indexShape.length < 1 || indexShape.length > 4
                || outputShape.length < 1 || outputShape.length > 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_ND supports rank 1..4 tensors";
        }
        int batchDims = gatherOp.getBatchDims();
        String shapeReason = gatherNdShapeUnsupportedReason(valueShape, indexShape, outputShape, batchDims);
        if (!shapeReason.isBlank()) {
            return shapeReason;
        }
        String boundsReason = gatherNdBoundsUnsupportedReason(indices, context, valueShape, batchDims, indexShape[indexShape.length - 1], opType);
        if (!boundsReason.isBlank()) {
            return boundsReason;
        }
        return "";
    }

    private static String gatherNdShapeUnsupportedReason(int[] valueShape, int[] indexShape, int[] outputShape, int batchDims) {
        if (batchDims < 0 || batchDims >= indexShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_ND batchDims must be in [0, indices rank)";
        }
        if (batchDims > valueShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_ND batchDims cannot exceed value rank";
        }
        for (int i = 0; i < batchDims; i++) {
            if (indexShape[i] != valueShape[i]) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_ND batch dimensions must match value leading dimensions";
            }
        }
        int tupleRank = indexShape[indexShape.length - 1];
        if (tupleRank <= 0 || batchDims + tupleRank > valueShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_ND final index dimension must be in [1, value rank - batchDims]";
        }
        int expectedRank = indexShape.length - 1 + valueShape.length - batchDims - tupleRank;
        if (expectedRank == 0) {
            if (outputShape.length == 1 && outputShape[0] == 1) {
                return "";
            }
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_ND scalar output must use project scalar shape [1]";
        }
        if (outputShape.length != expectedRank) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_ND output rank must equal indices.rank - 1 + value.rank - batchDims - tupleRank";
        }
        int p = 0;
        for (int i = 0; i < indexShape.length - 1; i++) {
            if (outputShape[p++] != indexShape[i]) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_ND output prefix shape must match indices prefix shape";
            }
        }
        for (int i = batchDims + tupleRank; i < valueShape.length; i++) {
            if (outputShape[p++] != valueShape[i]) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_ND output suffix shape must match value slice suffix";
            }
        }
        return "";
    }

    private static String gatherNdBoundsUnsupportedReason(
            CompiledNode indices,
            PartitionPlanningContext context,
            int[] valueShape,
            int batchDims,
            int tupleRank,
            Operation.OpType opType
    ) {
        if (!indices.leaf()) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opType + " index bounds require a static INT32 leaf tensor";
        }
        int[] data = indices.staticDataSnapshot().int32Values();
        if (data == null) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opType + " index bounds require readable INT32 storage";
        }
        long logicalElements = context.descriptor(indices.id()).logicalElementCount();
        if (logicalElements < 0 || logicalElements > data.length || logicalElements > Integer.MAX_VALUE) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opType + " index bounds cannot be proven from storage";
        }
        if (logicalElements % tupleRank != 0) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opType + " index storage is not tuple-aligned";
        }
        for (int i = 0; i < (int) logicalElements; i++) {
            int tupleAxis = i % tupleRank;
            int valueAxis = batchDims + tupleAxis;
            int axisSize = valueShape[valueAxis];
            int index = data[i];
            if (index < 0 || index >= axisSize) {
                return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opType + " tuple index " + index
                        + " is outside axis " + valueAxis + " size " + axisSize;
            }
        }
        return "";
    }

    private static int indexAxis(CompiledNode node) {
        return switch (node.operation().opType()) {
            case GATHER -> node.operation() instanceof gather op ? op.getDimension() : -1;
            case GATHER_AXIS -> node.operation() instanceof gatherAxis op ? op.getAxis() : -1;
            case TAKE_ALONG_AXIS -> node.operation() instanceof takeAlongAxis op ? op.getDimension() : -1;
            default -> -1;
        };
    }

    private static boolean isSupportedLayoutOp(Operation.OpType opType) {
        return opType == Operation.OpType.SLICE
                || opType == Operation.OpType.SLICE_BACKWARD
                || opType == Operation.OpType.CONCAT
                || opType == Operation.OpType.PAD
                || opType == Operation.OpType.TILE
                || opType == Operation.OpType.UNFOLD_AXIS
                || opType == Operation.OpType.UNFOLD2D
                || opType == Operation.OpType.FOLD2D;
    }

    private static String layoutUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        Operation.OpType opType = node.operation().opType();
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " requires planning context";
        }
        tensor.DataType dtype = dataType(context, node);
        if (!isMetalFloatingDType(dtype)) {
            return "UNSUPPORTED_DTYPE: GPU_METAL " + opType + " supports FLOAT32/BFLOAT16 layout values";
        }
        int[] outputShape = shape(context, node);
        if (outputShape.length < 1 || outputShape.length > 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " supports rank 1..4 outputs";
        }
        return switch (opType) {
            case SLICE -> sliceUnsupportedReason(node, context, dtype, outputShape);
            case SLICE_BACKWARD -> sliceBackwardUnsupportedReason(node, context, dtype, outputShape);
            case CONCAT -> concatUnsupportedReason(node, context, dtype, outputShape);
            case PAD -> padUnsupportedReason(node, context, dtype, outputShape);
            case TILE -> tileUnsupportedReason(node, context, dtype, outputShape);
            case UNFOLD_AXIS -> unfoldAxisUnsupportedReason(node, context, dtype, outputShape);
            case UNFOLD2D -> unfold2dUnsupportedReason(node, context, dtype, outputShape);
            case FOLD2D -> fold2dUnsupportedReason(node, context, dtype, outputShape);
            default -> "";
        };
    }

    private static String sliceUnsupportedReason(CompiledNode node, PartitionPlanningContext context, tensor.DataType dtype, int[] outputShape) {
        if (!(node.operation() instanceof slice op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SLICE descriptor is unavailable";
        }
        if (node.inputIds().size() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SLICE requires one input";
        }
        CompiledNode input = context.compiledNode(node.inputIds().getFirst());
        if (input == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SLICE input is unavailable";
        }
        if (dataType(context, input) != dtype) {
            return "UNSUPPORTED_DTYPE: GPU_METAL SLICE input/output dtype must match";
        }
        if (!dense(context, input)) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL SLICE input requires dense layout";
        }
        int[] inputShape = shape(context, input);
        if (inputShape.length != outputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SLICE preserves rank";
        }
        int[] starts = op.getStarts();
        int[] ends = op.getEnds();
        int[] axes = op.getAxes();
        int[] steps = op.getSteps();
        if (starts.length != axes.length || ends.length != axes.length || steps.length != axes.length || axes.length > 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SLICE requires aligned starts/ends/axes/steps";
        }
        for (int i = 0; i < axes.length; i++) {
            int axis = axes[i];
            if (axis < 0 || axis >= inputShape.length) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SLICE axis is outside input rank";
            }
            if (steps[i] != 1) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SLICE supports step=1 only";
            }
            if (starts[i] < 0 || ends[i] < starts[i] || ends[i] > inputShape[axis]) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SLICE start/end must be statically in bounds";
            }
        }
        return "";
    }

    private static String sliceBackwardUnsupportedReason(CompiledNode node, PartitionPlanningContext context, tensor.DataType dtype, int[] outputShape) {
        if (!(node.operation() instanceof sliceBackward op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SLICE_BACKWARD descriptor is unavailable";
        }
        if (node.inputIds().size() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SLICE_BACKWARD requires one input gradient";
        }
        if (!dense(context, node)) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL SLICE_BACKWARD output requires dense layout";
        }
        CompiledNode input = context.compiledNode(node.inputIds().getFirst());
        if (input == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SLICE_BACKWARD input gradient is unavailable";
        }
        if (dataType(context, input) != dtype) {
            return "UNSUPPORTED_DTYPE: GPU_METAL SLICE_BACKWARD input/output dtype must match";
        }
        if (!layoutInputSupported(context, input)) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL SLICE_BACKWARD input requires dense layout or GPU-side layout producer";
        }
        int[] gradShape = shape(context, input);
        int[] inputShape = op.getInputShape();
        if (inputShape.length < 1 || inputShape.length > 4
                || outputShape.length != inputShape.length
                || gradShape.length != inputShape.length
                || !Arrays.equals(outputShape, inputShape)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SLICE_BACKWARD inputShape/output/gradient ranks must match rank 1..4";
        }
        int[] starts = op.getStarts();
        int[] axes = op.getAxes();
        int[] steps = op.getSteps();
        if (starts.length != axes.length || steps.length != axes.length || axes.length > 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SLICE_BACKWARD requires aligned starts/axes/steps";
        }
        boolean[] seenAxes = new boolean[inputShape.length];
        int[] before = new int[inputShape.length];
        for (int i = 0; i < axes.length; i++) {
            int axis = axes[i];
            if (axis < 0 || axis >= inputShape.length) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SLICE_BACKWARD axis is outside input rank";
            }
            if (seenAxes[axis]) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SLICE_BACKWARD duplicate axes are unsupported";
            }
            seenAxes[axis] = true;
            if (steps[i] != 1) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SLICE_BACKWARD supports step=1 only";
            }
            if (starts[i] < 0 || starts[i] > inputShape[axis]) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SLICE_BACKWARD starts must be statically in bounds";
            }
            before[axis] = starts[i];
        }
        for (int d = 0; d < inputShape.length; d++) {
            if (!seenAxes[d] && gradShape[d] != inputShape[d]) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SLICE_BACKWARD unsliced dimensions must match original input shape";
            }
            int after = inputShape[d] - before[d] - gradShape[d];
            if (after < 0) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SLICE_BACKWARD gradient shape must fit original input shape";
            }
        }
        return "";
    }

    private static String concatUnsupportedReason(CompiledNode node, PartitionPlanningContext context, tensor.DataType dtype, int[] outputShape) {
        if (!(node.operation() instanceof concat op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CONCAT descriptor is unavailable";
        }
        if (node.inputIds().size() < 2 || node.inputIds().size() > 5) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CONCAT supports 2..5 inputs in the current DAG ABI";
        }
        int axis = op.getAxis();
        if (axis < 0 || axis >= outputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CONCAT axis is outside output rank";
        }
        int concatSize = 0;
        for (int inputId : node.inputIds()) {
            CompiledNode input = context.compiledNode(inputId);
            if (input == null) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CONCAT input is unavailable";
            }
            if (dataType(context, input) != dtype) {
                return "UNSUPPORTED_DTYPE: GPU_METAL CONCAT inputs/output dtype must match";
            }
            if (!layoutInputSupported(context, input)) {
                return "UNSUPPORTED_LAYOUT: GPU_METAL CONCAT inputs require dense layout or GPU-side layout producer";
            }
            int[] inputShape = shape(context, input);
            if (inputShape.length != outputShape.length) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CONCAT inputs must match output rank";
            }
            for (int d = 0; d < outputShape.length; d++) {
                if (d == axis) {
                    concatSize += inputShape[d];
                } else if (inputShape[d] != outputShape[d]) {
                    return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CONCAT non-axis dimensions must match";
                }
            }
        }
        return concatSize == outputShape[axis] ? "" : "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CONCAT axis sizes must sum to output axis";
    }

    private static String padUnsupportedReason(CompiledNode node, PartitionPlanningContext context, tensor.DataType dtype, int[] outputShape) {
        if (!(node.operation() instanceof pad op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL PAD descriptor is unavailable";
        }
        if (node.inputIds().size() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL PAD requires one input";
        }
        CompiledNode input = context.compiledNode(node.inputIds().getFirst());
        if (input == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL PAD input is unavailable";
        }
        if (dataType(context, input) != dtype) {
            return "UNSUPPORTED_DTYPE: GPU_METAL PAD input/output dtype must match";
        }
        if (!dense(context, input)) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL PAD input requires dense layout";
        }
        int[] inputShape = shape(context, input);
        int[] before = op.getBefore();
        int[] after = op.getAfter();
        if (inputShape.length != outputShape.length || before.length != inputShape.length || after.length != inputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL PAD before/after must match rank";
        }
        for (int d = 0; d < inputShape.length; d++) {
            if (before[d] < 0 || after[d] < 0 || outputShape[d] != inputShape[d] + before[d] + after[d]) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL PAD requires non-negative static pads matching output shape";
            }
        }
        return "";
    }

    private static String tileUnsupportedReason(CompiledNode node, PartitionPlanningContext context, tensor.DataType dtype, int[] outputShape) {
        if (!(node.operation() instanceof tile op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL TILE descriptor is unavailable";
        }
        if (node.inputIds().size() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL TILE requires one input";
        }
        CompiledNode input = context.compiledNode(node.inputIds().getFirst());
        if (input == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL TILE input is unavailable";
        }
        if (dataType(context, input) != dtype) {
            return "UNSUPPORTED_DTYPE: GPU_METAL TILE input/output dtype must match";
        }
        if (!dense(context, input)) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL TILE input requires dense layout";
        }
        int[] inputShape = shape(context, input);
        int[] repeats = op.getRepeats();
        if (inputShape.length != outputShape.length || repeats.length != inputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL TILE repeats must match rank";
        }
        for (int d = 0; d < inputShape.length; d++) {
            if (repeats[d] <= 0 || outputShape[d] != inputShape[d] * repeats[d]) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL TILE requires positive static repeats matching output shape";
            }
        }
        return "";
    }

    private static String unfoldAxisUnsupportedReason(CompiledNode node, PartitionPlanningContext context, tensor.DataType dtype, int[] outputShape) {
        if (!(node.operation() instanceof unfoldAxis op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL UNFOLD_AXIS descriptor is unavailable";
        }
        if (node.inputIds().size() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL UNFOLD_AXIS requires one input";
        }
        CompiledNode input = context.compiledNode(node.inputIds().getFirst());
        if (input == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL UNFOLD_AXIS input is unavailable";
        }
        if (dataType(context, input) != dtype) {
            return "UNSUPPORTED_DTYPE: GPU_METAL UNFOLD_AXIS input/output dtype must match";
        }
        if (!dense(context, input)) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL UNFOLD_AXIS input requires dense layout";
        }
        int[] inputShape = shape(context, input);
        if (inputShape.length < 1 || inputShape.length > 3 || outputShape.length != inputShape.length + 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL UNFOLD_AXIS supports input rank 1..3 and output rank 2..4";
        }
        int axis = op.getAxis();
        if (axis < 0 || axis >= inputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL UNFOLD_AXIS axis is outside input rank";
        }
        if (op.getStep() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL UNFOLD_AXIS currently supports step=1 native lowering";
        }
        int windows = inputShape[axis] - op.getSize() + 1;
        if (op.getSize() <= 0 || windows <= 0 || outputShape[axis] != windows || outputShape[outputShape.length - 1] != op.getSize()) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL UNFOLD_AXIS output shape must match static sliding window geometry";
        }
        for (int d = 0; d < inputShape.length; d++) {
            if (d != axis && outputShape[d] != inputShape[d]) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL UNFOLD_AXIS non-window dimensions must match input shape";
            }
        }
        return "";
    }

    private static String unfold2dUnsupportedReason(CompiledNode node, PartitionPlanningContext context, tensor.DataType dtype, int[] outputShape) {
        if (!(node.operation() instanceof unfold2d op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL UNFOLD2D descriptor is unavailable";
        }
        if (node.inputIds().size() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL UNFOLD2D requires one input";
        }
        CompiledNode input = context.compiledNode(node.inputIds().getFirst());
        if (input == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL UNFOLD2D input is unavailable";
        }
        if (dataType(context, input) != dtype) {
            return "UNSUPPORTED_DTYPE: GPU_METAL UNFOLD2D input/output dtype must match";
        }
        if (!dense(context, input)) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL UNFOLD2D input requires dense NCHW layout";
        }
        int[] inputShape = shape(context, input);
        if (inputShape.length != 4 || outputShape.length != 3) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL UNFOLD2D requires rank-4 NCHW input and rank-3 column output";
        }
        return window2dUnsupportedReason("UNFOLD2D", op.getOptions(), inputShape, outputShape, false);
    }

    private static String fold2dUnsupportedReason(CompiledNode node, PartitionPlanningContext context, tensor.DataType dtype, int[] outputShape) {
        if (!(node.operation() instanceof fold2d op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL FOLD2D descriptor is unavailable";
        }
        if (node.inputIds().size() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL FOLD2D requires one column input";
        }
        CompiledNode input = context.compiledNode(node.inputIds().getFirst());
        if (input == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL FOLD2D input is unavailable";
        }
        if (dataType(context, input) != dtype) {
            return "UNSUPPORTED_DTYPE: GPU_METAL FOLD2D input/output dtype must match";
        }
        if (!layoutInputSupported(context, input)) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL FOLD2D input requires dense layout or GPU-side layout producer";
        }
        int[] inputShape = shape(context, input);
        if (inputShape.length != 3 || outputShape.length != 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL FOLD2D requires rank-3 columns and rank-4 NCHW output";
        }
        return window2dUnsupportedReason("FOLD2D", op.getOptions(), outputShape, inputShape, true);
    }

    private static String window2dUnsupportedReason(
            String opName,
            Window2dOptions options,
            int[] nchwShape,
            int[] columnShape,
            boolean fold
    ) {
        if (options.strideH() != 1 || options.strideW() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " currently supports stride=1 native lowering";
        }
        if (options.dilationH() != 1 || options.dilationW() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " currently supports dilation=1 native lowering";
        }
        if (options.kernelH() <= 0 || options.kernelW() <= 0 || options.padH() < 0 || options.padW() < 0) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " requires positive kernel and non-negative padding";
        }
        int outH = nchwShape[2] + 2 * options.padH() - options.kernelH() + 1;
        int outW = nchwShape[3] + 2 * options.padW() - options.kernelW() + 1;
        if (outH <= 0 || outW <= 0) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " output window geometry is empty";
        }
        int kernelArea = options.kernelH() * options.kernelW();
        int expectedColumnChannels = nchwShape[1] * kernelArea;
        int expectedWindows = outH * outW;
        if (columnShape[0] != nchwShape[0] || columnShape[1] != expectedColumnChannels || columnShape[2] != expectedWindows) {
            String shapeRole = fold ? "input column shape" : "output column shape";
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " " + shapeRole + " must match NCHW window geometry";
        }
        return "";
    }

    private static String castUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CAST requires planning context";
        }
        if (node.inputIds().size() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CAST requires one input";
        }
        CompiledNode input = context.compiledNode(node.inputIds().getFirst());
        if (input == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CAST input is unavailable";
        }
        int[] inputShape = shape(context, input);
        int[] outputShape = shape(context, node);
        if (inputShape.length < 1 || inputShape.length > 4 || outputShape.length < 1 || outputShape.length > 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CAST supports rank 1..4 tensors";
        }
        if (!Arrays.equals(inputShape, outputShape)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CAST must preserve input shape";
        }
        if (!dense(context, input)) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL CAST input requires dense layout";
        }
        MetalCastPolicy.Decision decision = MetalCastPolicy.decide(dataType(context, input), dataType(context, node));
        if (!decision.supported()) {
            return decision.reasonCode().name() + ": " + decision.detail();
        }
        return "";
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

    private static boolean isReductionScanParityOp(Operation.OpType opType) {
        return opType == Operation.OpType.REDUCE_PROD
                || opType == Operation.OpType.ARGMAX
                || opType == Operation.OpType.CUMSUM;
    }

    private static String reductionScanUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        Operation.OpType opType = node.operation().opType();
        if (node.backwardNode()) {
            return "BACKWARD_CONTEXT_UNSUPPORTED: forward " + opType + " nodes are not legal inside Metal backward regions";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " requires planning context";
        }
        if (node.inputIds().size() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " requires one input";
        }
        CompiledNode input = context.compiledNode(node.inputIds().getFirst());
        if (input == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " input is unavailable";
        }
        if (!dense(context, input)) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL " + opType + " input requires dense layout";
        }
        int[] inputShape = shape(context, input);
        int[] outputShape = shape(context, node);
        if (inputShape.length < 1 || inputShape.length > 4 || outputShape.length < 1 || outputShape.length > 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " supports rank 1..4 tensors";
        }
        return switch (opType) {
            case REDUCE_PROD -> reduceProdUnsupportedReason(node, input, inputShape, outputShape);
            case ARGMAX -> argMaxUnsupportedReason(node, input, inputShape, outputShape);
            case CUMSUM -> cumSumUnsupportedReason(node, input, inputShape, outputShape);
            default -> "";
        };
    }

    private static String reduceProdUnsupportedReason(CompiledNode node, CompiledNode input, int[] inputShape, int[] outputShape) {
        if (!(node.operation() instanceof reduceProd op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL REDUCE_PROD descriptor is unavailable";
        }
        tensor.DataType dtype = node.dataType();
        if (!isMetalFloatingDType(dtype) || input.dataType() != dtype) {
            return "UNSUPPORTED_DTYPE: GPU_METAL REDUCE_PROD requires dtype-matched FLOAT32/BFLOAT16 input/output";
        }
        return reductionOutputShapeReason("REDUCE_PROD", inputShape, outputShape, op.getDimension(), op.keepDims());
    }

    private static String argMaxUnsupportedReason(CompiledNode node, CompiledNode input, int[] inputShape, int[] outputShape) {
        if (!(node.operation() instanceof argMax op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL ARGMAX descriptor is unavailable";
        }
        if (!isMetalFloatingDType(input.dataType())) {
            return "UNSUPPORTED_DTYPE: GPU_METAL ARGMAX input requires FLOAT32/BFLOAT16 data";
        }
        if (node.dataType() != tensor.DataType.INT64) {
            return "UNSUPPORTED_DTYPE: GPU_METAL ARGMAX output must be INT64";
        }
        if (op.getDimension() < 0) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL ARGMAX requires one explicit axis";
        }
        return reductionOutputShapeReason("ARGMAX", inputShape, outputShape, op.getDimension(), op.keepDims());
    }

    private static String cumSumUnsupportedReason(CompiledNode node, CompiledNode input, int[] inputShape, int[] outputShape) {
        if (!(node.operation() instanceof cumSum op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CUMSUM descriptor is unavailable";
        }
        tensor.DataType dtype = node.dataType();
        if (!isMetalFloatingDType(dtype) || input.dataType() != dtype) {
            return "UNSUPPORTED_DTYPE: GPU_METAL CUMSUM requires dtype-matched FLOAT32/BFLOAT16 input/output";
        }
        int axis = op.getAxis();
        if (axis < 0 || axis >= inputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CUMSUM axis is outside input rank";
        }
        if (!Arrays.equals(inputShape, outputShape)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CUMSUM output shape must match input shape";
        }
        return "";
    }

    private static String reductionOutputShapeReason(String opName, int[] inputShape, int[] outputShape, int axis, boolean keepDims) {
        if (axis < -1 || axis >= inputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " axis is outside input rank";
        }
        int[] expected = axis == -1
                ? (keepDims ? allAxesKeepDimsShape(inputShape) : new int[]{1})
                : reduceShape(inputShape, axis, keepDims);
        return Arrays.equals(outputShape, expected)
                ? ""
                : "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " output shape must match reduced input shape";
    }

    private static int[] reduceShape(int[] shape, int axis, boolean keepDims) {
        if (keepDims) {
            int[] out = shape.clone();
            out[axis] = 1;
            return out;
        }
        return reduceShape(shape, axis);
    }

    private static int[] allAxesKeepDimsShape(int[] shape) {
        int[] out = new int[shape.length];
        Arrays.fill(out, 1);
        return out;
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

    private static int[] broadcastBatchShape(int[] left, int[] right) {
        int rank = Math.max(left.length, right.length);
        int[] out = new int[rank];
        for (int i = 0; i < rank; i++) {
            int leftIndex = left.length - 1 - i;
            int rightIndex = right.length - 1 - i;
            int l = leftIndex >= 0 ? left[leftIndex] : 1;
            int r = rightIndex >= 0 ? right[rightIndex] : 1;
            if (l != r && l != 1 && r != 1) {
                return null;
            }
            out[rank - 1 - i] = Math.max(l, r);
        }
        return out;
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
            case SUM, MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_PROD, ARGMAX, CUMSUM -> "REDUCTION_ADJACENT: ";
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

    private static boolean isUnaryMathParityOp(Operation.OpType opType) {
        return opType == Operation.OpType.ERF
                || opType == Operation.OpType.FLOOR
                || opType == Operation.OpType.CEIL
                || opType == Operation.OpType.SIGN;
    }

    private static String unaryMathUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        Operation.OpType opType = node.operation().opType();
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " requires planning context";
        }
        if (node.inputIds().size() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " requires one input";
        }
        CompiledNode input = context.compiledNode(node.inputIds().getFirst());
        if (input == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " input is unavailable";
        }
        tensor.DataType dtype = dataType(context, node);
        if (!isMetalFloatingDType(dtype) || dataType(context, input) != dtype) {
            return "UNSUPPORTED_DTYPE: GPU_METAL " + opType + " requires dtype-matched FLOAT32/BFLOAT16 input and output";
        }
        if (!layoutInputSupported(context, input)) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL " + opType + " input requires dense layout or GPU-side layout producer";
        }
        int[] inputShape = shape(context, input);
        int[] outputShape = shape(context, node);
        if (inputShape.length < 1 || inputShape.length > 4 || outputShape.length < 1 || outputShape.length > 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " supports rank 1..4 tensors";
        }
        if (!Arrays.equals(inputShape, outputShape)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " must preserve input shape";
        }
        return "";
    }

    private static String boolCompareUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL BOOL compare requires planning context";
        }
        if (dataType(context, node) != tensor.DataType.BOOL) {
            return "UNSUPPORTED_DTYPE: GPU_METAL BOOL compare output must be BOOL";
        }
        if (node.inputIds().size() != 2) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL BOOL compare requires two inputs";
        }
        for (int inputId : node.inputIds()) {
            CompiledNode input = context.compiledNode(inputId);
            if (input == null) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL BOOL compare inputs are unavailable";
            }
            tensor.DataType inputType = dataType(context, input);
            if (inputType != tensor.DataType.FLOAT32 && inputType != tensor.DataType.BFLOAT16) {
                return "UNSUPPORTED_DTYPE: GPU_METAL BOOL compare inputs require FLOAT32/BFLOAT16 data";
            }
            if (!dense(context, input)) {
                return "UNSUPPORTED_LAYOUT: GPU_METAL BOOL compare inputs require dense layout";
            }
        }
        return "";
    }

    private static String boolLogicalUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL BOOL logical op requires planning context";
        }
        if (dataType(context, node) != tensor.DataType.BOOL) {
            return "UNSUPPORTED_DTYPE: GPU_METAL BOOL logical output must be BOOL";
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
            if (dataType(context, input) != tensor.DataType.BOOL) {
                return "UNSUPPORTED_DTYPE: GPU_METAL BOOL logical inputs require BOOL data";
            }
            if (!dense(context, input)) {
                return "UNSUPPORTED_LAYOUT: GPU_METAL BOOL logical inputs require dense layout";
            }
        }
        return "";
    }

    private static String boolReductionUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL BOOL reduction requires planning context";
        }
        if (dataType(context, node) != tensor.DataType.BOOL) {
            return "UNSUPPORTED_DTYPE: GPU_METAL BOOL reduction output must be BOOL";
        }
        if (node.inputIds().size() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL BOOL reduction requires one input";
        }
        CompiledNode input = context.compiledNode(node.inputIds().getFirst());
        if (input == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL BOOL reduction input is unavailable";
        }
        if (dataType(context, input) != tensor.DataType.BOOL) {
            return "UNSUPPORTED_DTYPE: GPU_METAL BOOL reduction input requires BOOL data";
        }
        if (!dense(context, input)) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL BOOL reduction input requires dense layout";
        }
        int rank = shape(context, input).length;
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
        tensor.DataType dtype = dataType(context, node);
        if (dtype != tensor.DataType.FLOAT32 && dtype != tensor.DataType.BFLOAT16) {
            return "UNSUPPORTED_DTYPE: " + backend + " normalization supports only FLOAT32/BFLOAT16";
        }
        if (dataType(context, input) != dtype
                || dataType(context, gamma) != dtype
                || (beta != null && dataType(context, beta) != dtype)) {
            return "UNSUPPORTED_DTYPE: " + backend + " normalization inputs and output must use the same dtype";
        }
        if (!dense(context, input) || !dense(context, gamma) || (beta != null && !dense(context, beta))) {
            return "UNSUPPORTED_LAYOUT: " + backend + " normalization inputs require dense layout";
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
        if (node == null || node.operation() == null || context == null || node.operation().arityClass() == Operation.OpArityClass.LAYOUT) {
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

    private static tensor.DataType dataType(PartitionPlanningContext context, CompiledNode node) {
        return descriptor(context, node).dataType();
    }

    private static boolean isMetalFloatingDType(tensor.DataType dtype) {
        return dtype == tensor.DataType.FLOAT32 || dtype == tensor.DataType.BFLOAT16;
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
            if (containsMatMulFamily(input)) {
                return true;
            }
        }
        return false;
    }
}
