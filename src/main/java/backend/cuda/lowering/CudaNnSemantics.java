package backend.cuda.lowering;

import graph.CompiledNode;
import graph.compile.planning.partition.PartitionPlanningContext;
import operations.Operation;
import operations.linalg.scaledDotProductAttention;
import operations.loss.crossEntropyLoss;
import operations.loss.nllLoss;
import operations.nn.conv.conv2d;
import operations.nn.pool.avgPool2d;
import operations.nn.pool.maxPool2d;
import tensor.DataType;
import tensor.options.Conv2dOptions;
import tensor.options.Pool2dOptions;

import java.util.Arrays;

/**
 * CUDA NN operation semantic checks before support or stable capability rejection.
 */
final class CudaNnSemantics {
    private CudaNnSemantics() {
    }

    static boolean isHandled(Operation.OpType opType) {
        return opType == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION
                || isForwardConvPool(opType)
                || isDenseLoss(opType);
    }

    static String unsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        Operation.OpType opType = node.operation().opType();
        return switch (opType) {
            case SCALED_DOT_PRODUCT_ATTENTION -> sdpaUnsupportedReason(node, context);
            case CONV2D -> conv2dUnsupportedReason(node, context);
            case MAX_POOL2D, AVG_POOL2D -> poolUnsupportedReason(node, context);
            case NLL_LOSS, CROSS_ENTROPY_LOSS -> denseLossUnsupportedReason(node, context);
            default -> "";
        };
    }

    private static boolean isForwardConvPool(Operation.OpType opType) {
        return opType == Operation.OpType.CONV2D
                || opType == Operation.OpType.MAX_POOL2D
                || opType == Operation.OpType.AVG_POOL2D;
    }

    private static boolean isDenseLoss(Operation.OpType opType) {
        return opType == Operation.OpType.NLL_LOSS || opType == Operation.OpType.CROSS_ENTROPY_LOSS;
    }

    private static String sdpaUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        if (!(node.operation() instanceof scaledDotProductAttention attention)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA SDPA descriptor is unavailable";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA SDPA requires planning context";
        }
        int expectedInputs = attention.hasMask() ? 4 : 3;
        if (node.inputIds().size() != expectedInputs) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA "
                    + (attention.hasMask() ? "masked" : "unmasked")
                    + " SDPA requires " + expectedInputs + " inputs";
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
        if (!query.contiguous() || query.hasStorageOffset()
                || !key.contiguous() || key.hasStorageOffset()
                || !value.contiguous() || value.hasStorageOffset()) {
            return "UNSUPPORTED_LAYOUT: GPU_CUDA SDPA inputs require dense layout";
        }
        int[] queryShape = query.shape();
        int[] keyShape = key.shape();
        int[] valueShape = value.shape();
        int[] outputShape = node.shape();
        if (!sdpaRankSupported(queryShape) || keyShape.length != queryShape.length
                || valueShape.length != queryShape.length || outputShape.length != queryShape.length) {
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
        int[] scoreShape = expectedScoresShape(queryShape, keyShape);
        if (scoreShape.length == 0 || !broadcastBatchMatches(Arrays.copyOf(outputShape, outputShape.length - 2),
                Arrays.copyOf(scoreShape, scoreShape.length - 2))) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA SDPA batch dimensions are not broadcast-compatible";
        }
        String maskReason = attention.hasMask() ? maskUnsupportedReason(context.compiledNode(node.inputIds().get(3)), context, scoreShape) : "";
        if (!maskReason.isBlank()) {
            return maskReason;
        }
        String target = attention.hasMask() ? "target=masked_sdpa_small" : "target=transformer_block_hot_path";
        String maskMode = attention.hasMask() ? " maskMode=" + classifyMaskMode(context.compiledNode(node.inputIds().get(3)), context) : " maskMode=UNMASKED";
        return "CAPABILITY_MISSING: operation SCALED_DOT_PRODUCT_ATTENTION is not supported by GPU_CUDA lowering; "
                + "CUDA direct forward SDPA native/lowered path is not implemented; "
                + target + maskMode;
    }

    private static String maskUnsupportedReason(CompiledNode mask, PartitionPlanningContext context, int[] expectedScoreShape) {
        if (mask == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA SDPA mask input is unavailable";
        }
        if (mask.dataType() != DataType.BOOL) {
            return "UNSUPPORTED_DTYPE: GPU_CUDA SDPA public mask input requires BOOL dtype";
        }
        if (!mask.contiguous() || mask.hasStorageOffset()) {
            return "UNSUPPORTED_LAYOUT: GPU_CUDA SDPA mask input requires dense BOOL layout";
        }
        if (!Arrays.equals(mask.shape(), expectedScoreShape)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA SDPA mask shape must equal broadcasted score shape";
        }
        return "";
    }

    private static String conv2dUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        String common = commonForwardReason("CONV2D", node, context);
        if (!common.isBlank()) {
            return common;
        }
        if (!(node.operation() instanceof conv2d conv)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA CONV2D descriptor is unavailable";
        }
        int expectedInputs = conv.hasBias() ? 3 : 2;
        if (node.inputIds().size() != expectedInputs) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA CONV2D requires input, weight, and optional bias according to descriptor";
        }
        return conv2dLikeUnsupportedReason("CONV2D", node, context, conv.getOptions(), conv.hasBias());
    }

    private static String conv2dLikeUnsupportedReason(
            String opName,
            CompiledNode node,
            PartitionPlanningContext context,
            Conv2dOptions options,
            boolean hasBias
    ) {
        CompiledNode input = context.compiledNode(node.inputIds().get(0));
        CompiledNode weight = context.compiledNode(node.inputIds().get(1));
        CompiledNode bias = hasBias ? context.compiledNode(node.inputIds().get(2)) : null;
        if (input == null || weight == null || (hasBias && bias == null)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opName + " inputs are unavailable";
        }
        String dtypeReason = requireFloat32(opName, node, input, weight, bias);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense(opName, input, weight, bias);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        int[] inputShape = input.shape();
        int[] weightShape = weight.shape();
        int[] outShape = node.shape();
        if (inputShape.length != 4 || weightShape.length != 4 || outShape.length != 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opName + " requires rank-4 NCHW input/output and OIHW weight";
        }
        int n = inputShape[0];
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        if (bias != null && (bias.shape().length != 1 || bias.shape()[0] != outChannels)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opName + " bias must have shape [outChannels]";
        }
        if (inChannels % options.groups() != 0 || outChannels % options.groups() != 0
                || channelsPerGroup * options.groups() != inChannels) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opName + " groups/channel contract is invalid";
        }
        if (options.groups() != 1) {
            return "CAPABILITY_MISSING: operation " + opName
                    + " is not supported by GPU_CUDA lowering; grouped/depthwise native execution is not implemented; family=CONV_POOL";
        }
        if (options.dilationH() != 1 || options.dilationW() != 1) {
            return "CAPABILITY_MISSING: operation " + opName
                    + " is not supported by GPU_CUDA lowering; dilation native execution is not implemented; family=CONV_POOL";
        }
        int outH = inferConvOutput(inH, kernelH, options.padH(), options.strideH(), options.dilationH());
        int outW = inferConvOutput(inW, kernelW, options.padW(), options.strideW(), options.dilationW());
        if (outH < 0 || outW < 0) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opName + " effective kernel does not fit input";
        }
        if (!Arrays.equals(outShape, new int[]{n, outChannels, outH, outW})) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opName + " output shape does not match NCHW/OIHW contract";
        }
        return "CAPABILITY_MISSING: operation " + opName + " is not supported by GPU_CUDA lowering; GPU_CUDA " + opName
                + " native/routed forward execution is not implemented; family=CONV_POOL target=conv2d_resnet_3x3";
    }

    private static String poolUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        Operation.OpType opType = node.operation().opType();
        String opName = opType.name();
        String common = commonForwardReason(opName, node, context);
        if (!common.isBlank()) {
            return common;
        }
        Pool2dOptions options;
        if (node.operation() instanceof maxPool2d maxPool) {
            options = maxPool.getOptions();
        } else if (node.operation() instanceof avgPool2d avgPool) {
            options = avgPool.getOptions();
        } else {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opName + " descriptor is unavailable";
        }
        if (node.inputIds().size() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opName + " requires one input";
        }
        CompiledNode input = context.compiledNode(node.inputIds().getFirst());
        if (input == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opName + " input is unavailable";
        }
        String dtypeReason = requireFloat32(opName, node, input);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense(opName, input);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        if (input.shape().length != 4 || node.shape().length != 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opName + " requires rank-4 NCHW input/output";
        }
        if (opType == Operation.OpType.AVG_POOL2D && options.countIncludePad()) {
            return "CAPABILITY_MISSING: operation AVG_POOL2D is not supported by GPU_CUDA lowering; "
                    + "countIncludePad=true native divisor semantics are not implemented; family=CONV_POOL";
        }
        if (options.ceilMode()) {
            return "CAPABILITY_MISSING: GPU_CUDA " + opName + " ceilMode=true native output-shape semantics are not implemented; family=CONV_POOL";
        }
        int outH = inferPoolOutput(input.shape()[2], options.kernelH(), options.padH(), options.strideH());
        int outW = inferPoolOutput(input.shape()[3], options.kernelW(), options.padW(), options.strideW());
        if (outH < 0 || outW < 0) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opName + " kernel does not fit input";
        }
        if (!Arrays.equals(node.shape(), new int[]{input.shape()[0], input.shape()[1], outH, outW})) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opName + " output shape does not match NCHW pooling contract";
        }
        return "CAPABILITY_MISSING: operation " + opName + " is not supported by GPU_CUDA lowering; GPU_CUDA " + opName
                + " native/routed forward execution is not implemented; family=CONV_POOL target="
                + (opType == Operation.OpType.MAX_POOL2D ? "max_pool2d_small" : "avg_pool2d_small");
    }

    private static String denseLossUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        Operation.OpType opType = node.operation().opType();
        if (node.backwardNode()) {
            return "BACKWARD_CONTEXT_UNSUPPORTED: " + opType + " nodes are not legal inside nested CUDA backward regions";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opType + " requires planning context";
        }
        if (node.inputIds().size() != 2) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opType + " requires scores/log-probabilities and dense targets";
        }
        int classAxis = classAxis(node);
        if (classAxis < 0) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opType + " descriptor is unavailable";
        }
        CompiledNode scores = context.compiledNode(node.inputIds().get(0));
        CompiledNode targets = context.compiledNode(node.inputIds().get(1));
        if (scores == null || targets == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opType + " inputs are unavailable";
        }
        if (node.dataType() != DataType.FLOAT32 || scores.dataType() != DataType.FLOAT32 || targets.dataType() != DataType.FLOAT32) {
            return "UNSUPPORTED_DTYPE: GPU_CUDA dense " + opType
                    + " contract is scoped to FLOAT32 output, scores/log-probabilities, and dense targets";
        }
        if (!scores.contiguous() || scores.hasStorageOffset() || !targets.contiguous() || targets.hasStorageOffset()) {
            return "UNSUPPORTED_LAYOUT: GPU_CUDA dense " + opType + " inputs require dense zero-offset layout";
        }
        if (scores.shape().length < 1 || scores.shape().length > 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA dense " + opType + " supports rank 1..4 tensors";
        }
        if (classAxis >= scores.shape().length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA dense " + opType + " class axis is outside input rank";
        }
        if (!Arrays.equals(scores.shape(), targets.shape())) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA dense " + opType + " dense target shape must match input shape";
        }
        if (!Arrays.equals(node.shape(), new int[]{1})) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA dense " + opType + " currently locks mean-reduced scalar output shape [1]";
        }
        return "DAG_PRIMITIVE_UNSUPPORTED: operation " + opType + " is not supported by GPU_CUDA lowering; GPU_CUDA dense " + opType
                + " native/lowered loss primitive is not implemented; family=LOSS_ADJACENT target=dense_loss_small";
    }

    private static String commonForwardReason(String opName, CompiledNode node, PartitionPlanningContext context) {
        if (node.backwardNode()) {
            return "BACKWARD_CONTEXT_UNSUPPORTED: forward " + opName + " nodes are not legal inside CUDA backward regions";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opName + " requires planning context";
        }
        return "";
    }

    private static String requireFloat32(String opName, CompiledNode output, CompiledNode... inputs) {
        if (output.dataType() != DataType.FLOAT32) {
            return "UNSUPPORTED_DTYPE: GPU_CUDA " + opName + " currently supports only FLOAT32 output";
        }
        for (CompiledNode input : inputs) {
            if (input != null && input.dataType() != DataType.FLOAT32) {
                return "UNSUPPORTED_DTYPE: GPU_CUDA " + opName + " currently supports only FLOAT32 inputs";
            }
        }
        return "";
    }

    private static String requireDense(String opName, CompiledNode... inputs) {
        for (CompiledNode input : inputs) {
            if (input != null && (!input.contiguous() || input.hasStorageOffset())) {
                return "UNSUPPORTED_LAYOUT: GPU_CUDA " + opName + " inputs require dense layout";
            }
        }
        return "";
    }

    private static int inferConvOutput(int inputSize, int kernelSize, int pad, int stride, int dilation) {
        int effectiveKernel = dilation * (kernelSize - 1) + 1;
        int numerator = inputSize + 2 * pad - effectiveKernel;
        return numerator < 0 ? -1 : numerator / stride + 1;
    }

    private static int inferPoolOutput(int inputSize, int kernelSize, int pad, int stride) {
        int numerator = inputSize + 2 * pad - kernelSize;
        return numerator < 0 ? -1 : numerator / stride + 1;
    }

    private static int classAxis(CompiledNode node) {
        Operation operation = node.operation();
        return switch (operation.opType()) {
            case NLL_LOSS -> operation instanceof nllLoss op ? op.getClassDimension() : -1;
            case CROSS_ENTROPY_LOSS -> operation instanceof crossEntropyLoss op ? op.getClassDimension() : -1;
            default -> -1;
        };
    }

    private static boolean sdpaRankSupported(int[] shape) {
        return shape != null && (shape.length == 3 || shape.length == 4);
    }

    private static boolean broadcastBatchMatches(int[] left, int[] right) {
        if (left.length != right.length) {
            return false;
        }
        for (int i = 0; i < left.length; i++) {
            if (left[i] != right[i] && left[i] != 1 && right[i] != 1) {
                return false;
            }
        }
        return true;
    }

    private static int[] expectedScoresShape(int[] queryShape, int[] keyShape) {
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

    private static String classifyMaskMode(CompiledNode mask, PartitionPlanningContext context) {
        CompiledNode source = unwrapExpand(mask, context);
        if (isCausalMaskLeaf(source)) {
            return "CAUSAL_BOOL_MASK";
        }
        if (source != null && source.operation() != null && source.operation().opType() == Operation.OpType.LOGICAL_AND) {
            boolean hasCausal = false;
            boolean hasExternal = false;
            for (int inputId : source.inputIds()) {
                CompiledNode input = unwrapExpand(context.compiledNode(inputId), context);
                if (isCausalMaskLeaf(input)) {
                    hasCausal = true;
                } else if (input != null) {
                    hasExternal = true;
                }
            }
            if (hasCausal && hasExternal) {
                return "EXTERNAL_AND_CAUSAL_BOOL_MASK";
            }
        }
        return "EXTERNAL_BOOL_MASK";
    }

    private static CompiledNode unwrapExpand(CompiledNode node, PartitionPlanningContext context) {
        CompiledNode current = node;
        while (current != null
                && current.operation() != null
                && current.operation().opType() == Operation.OpType.EXPAND
                && current.inputIds().size() == 1) {
            current = context.compiledNode(current.inputIds().getFirst());
        }
        return current;
    }

    private static boolean isCausalMaskLeaf(CompiledNode node) {
        return node != null && node.leaf() && node.dataType() == DataType.BOOL && "causal_mask".equals(node.label());
    }
}
