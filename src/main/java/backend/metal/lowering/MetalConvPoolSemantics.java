package backend.metal.lowering;

import graph.CompiledNode;
import graph.optimizer.partition.PartitionPlanningContext;
import operations.Operation;
import operations.nn.conv.conv2d;
import operations.nn.conv.conv2dGemm;
import operations.nn.pool.avgPool2d;
import operations.nn.pool.maxPool2d;
import tensor.DataType;
import tensor.options.Conv2dOptions;
import tensor.options.Pool2dOptions;

import java.util.Arrays;

final class MetalConvPoolSemantics {
    private MetalConvPoolSemantics() {
    }

    static boolean isForwardConvPool(Operation.OpType opType) {
        return opType == Operation.OpType.CONV2D
                || opType == Operation.OpType.CONV2D_GEMM
                || opType == Operation.OpType.MAX_POOL2D
                || opType == Operation.OpType.AVG_POOL2D;
    }

    static String unsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        Operation.OpType opType = node.operation().opType();
        return switch (opType) {
            case CONV2D -> conv2dUnsupportedReason(node, context);
            case CONV2D_GEMM -> conv2dGemmUnsupportedReason(node, context);
            case MAX_POOL2D, AVG_POOL2D -> poolUnsupportedReason(node, context);
            default -> "";
        };
    }

    private static String conv2dUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        String common = commonForwardReason("CONV2D", node, context);
        if (!common.isBlank()) {
            return common;
        }
        if (!(node.operation() instanceof conv2d conv)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CONV2D descriptor is unavailable";
        }
        int expectedInputs = conv.hasBias() ? 3 : 2;
        if (node.inputIds().size() != expectedInputs) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CONV2D requires input, weight, and optional bias according to descriptor";
        }
        CompiledNode input = context.compiledNode(node.inputIds().get(0));
        CompiledNode weight = context.compiledNode(node.inputIds().get(1));
        CompiledNode bias = conv.hasBias() ? context.compiledNode(node.inputIds().get(2)) : null;
        if (input == null || weight == null || (conv.hasBias() && bias == null)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CONV2D inputs are unavailable";
        }
        String dtypeReason = requireFloat32("CONV2D", node, input, weight, bias);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense("CONV2D", input, weight, bias);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        int[] inputShape = input.shape();
        int[] weightShape = weight.shape();
        int[] outShape = node.shape();
        if (inputShape.length != 4 || weightShape.length != 4 || outShape.length != 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CONV2D requires rank-4 NCHW input/output and OIHW weight";
        }
        int n = inputShape[0];
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        if (bias != null) {
            int[] biasShape = bias.shape();
            if (biasShape.length != 1 || biasShape[0] != outChannels) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CONV2D bias must have shape [outChannels]";
            }
        }
        Conv2dOptions options = conv.getOptions();
        if (inChannels % options.groups() != 0 || outChannels % options.groups() != 0
                || channelsPerGroup * options.groups() != inChannels) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CONV2D groups/channel contract is invalid";
        }
        if (options.groups() != 1) {
            return "CAPABILITY_MISSING: GPU_METAL CONV2D grouped/depthwise native execution is not implemented; family=CONV_POOL";
        }
        if (options.dilationH() != 1 || options.dilationW() != 1) {
            return "CAPABILITY_MISSING: GPU_METAL CONV2D dilation native execution is not implemented; family=CONV_POOL";
        }
        int outH = inferConvOutput(inH, kernelH, options.padH(), options.strideH(), options.dilationH(), "height");
        if (outH < 0) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CONV2D effective kernel does not fit input height";
        }
        int outW = inferConvOutput(inW, kernelW, options.padW(), options.strideW(), options.dilationW(), "width");
        if (outW < 0) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CONV2D effective kernel does not fit input width";
        }
        if (!Arrays.equals(outShape, new int[]{n, outChannels, outH, outW})) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CONV2D output shape does not match NCHW/OIHW contract";
        }
        return "CAPABILITY_MISSING: GPU_METAL CONV2D forward semantic contract is legal but native execution is not implemented; family=CONV_POOL target=conv2d_resnet_3x3";
    }

    private static String conv2dGemmUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        String common = commonForwardReason("CONV2D_GEMM", node, context);
        if (!common.isBlank()) {
            return common;
        }
        if (!(node.operation() instanceof conv2dGemm conv)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CONV2D_GEMM descriptor is unavailable";
        }
        int expectedInputs = conv.hasBias() ? 3 : 2;
        if (node.inputIds().size() != expectedInputs) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CONV2D_GEMM requires input, weight, and optional bias according to descriptor";
        }
        return "CAPABILITY_MISSING: GPU_METAL CONV2D_GEMM remains CPU-owned until im2col/GEMM/output-layout semantics are represented in the accelerator DAG; family=CONV_POOL target=conv2d_resnet_3x3";
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
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " descriptor is unavailable";
        }
        if (node.inputIds().size() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " requires one input";
        }
        CompiledNode input = context.compiledNode(node.inputIds().getFirst());
        if (input == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " input is unavailable";
        }
        String dtypeReason = requireFloat32(opName, node, input);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense(opName, input);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        int[] inputShape = input.shape();
        int[] outShape = node.shape();
        if (inputShape.length != 4 || outShape.length != 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " requires rank-4 NCHW input/output";
        }
        if (opType == Operation.OpType.AVG_POOL2D && options.countIncludePad()) {
            return "CAPABILITY_MISSING: GPU_METAL AVG_POOL2D countIncludePad=true native divisor semantics are not implemented; family=CONV_POOL";
        }
        int outH = inferPoolOutput(inputShape[2], options.kernelH(), options.padH(), options.strideH(), "height");
        if (outH < 0) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " kernel does not fit input height";
        }
        int outW = inferPoolOutput(inputShape[3], options.kernelW(), options.padW(), options.strideW(), "width");
        if (outW < 0) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " kernel does not fit input width";
        }
        String coverageReason = validatePoolWindowCoverage(inputShape[2], options.kernelH(), options.padH(), options.strideH(), outH, opName, "height");
        if (!coverageReason.isBlank()) {
            return coverageReason;
        }
        coverageReason = validatePoolWindowCoverage(inputShape[3], options.kernelW(), options.padW(), options.strideW(), outW, opName, "width");
        if (!coverageReason.isBlank()) {
            return coverageReason;
        }
        if (!Arrays.equals(outShape, new int[]{inputShape[0], inputShape[1], outH, outW})) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " output shape does not match NCHW pooling contract";
        }
        String target = opType == Operation.OpType.MAX_POOL2D ? " target=max_pool2d_small" : "";
        return "CAPABILITY_MISSING: GPU_METAL " + opName + " forward semantic contract is legal but native execution is not implemented; family=CONV_POOL" + target;
    }

    private static String commonForwardReason(String opName, CompiledNode node, PartitionPlanningContext context) {
        if (node.backwardNode()) {
            return "BACKWARD_CONTEXT_UNSUPPORTED: forward " + opName + " nodes are not legal inside Metal backward regions";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " requires planning context";
        }
        return "";
    }

    private static String requireFloat32(String opName, CompiledNode output, CompiledNode... inputs) {
        if (output.dataType() != DataType.FLOAT32) {
            return "UNSUPPORTED_DTYPE: GPU_METAL " + opName + " currently supports only FLOAT32 output";
        }
        for (CompiledNode input : inputs) {
            if (input != null && input.dataType() != DataType.FLOAT32) {
                return "UNSUPPORTED_DTYPE: GPU_METAL " + opName + " currently supports only FLOAT32 inputs";
            }
        }
        return "";
    }

    private static String requireDense(String opName, CompiledNode... inputs) {
        for (CompiledNode input : inputs) {
            if (input != null && (!input.contiguous() || input.hasStorageOffset())) {
                return "UNSUPPORTED_LAYOUT: GPU_METAL " + opName + " inputs require dense layout";
            }
        }
        return "";
    }

    private static int inferConvOutput(int inputSize, int kernelSize, int pad, int stride, int dilation, String axisName) {
        int effectiveKernel = dilation * (kernelSize - 1) + 1;
        int numerator = inputSize + 2 * pad - effectiveKernel;
        if (numerator < 0) {
            return -1;
        }
        return numerator / stride + 1;
    }

    private static int inferPoolOutput(int inputSize, int kernelSize, int pad, int stride, String axisName) {
        int numerator = inputSize + 2 * pad - kernelSize;
        if (numerator < 0) {
            return -1;
        }
        return numerator / stride + 1;
    }

    private static String validatePoolWindowCoverage(
            int inputSize,
            int kernelSize,
            int pad,
            int stride,
            int outSize,
            String opName,
            String axisName
    ) {
        for (int outIndex = 0; outIndex < outSize; outIndex++) {
            int origin = outIndex * stride - pad;
            boolean hasValid = false;
            for (int k = 0; k < kernelSize; k++) {
                int inputIndex = origin + k;
                if (inputIndex >= 0 && inputIndex < inputSize) {
                    hasValid = true;
                    break;
                }
            }
            if (!hasValid) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " creates an all-padding window on " + axisName + " axis";
            }
        }
        return "";
    }
}
