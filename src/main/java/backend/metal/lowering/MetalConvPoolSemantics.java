package backend.metal.lowering;

import graph.CompiledNode;
import graph.compile.planning.partition.PartitionPlanningContext;
import operations.Operation;
import operations.nn.conv.conv2d;
import operations.nn.conv.conv2dBackwardInput;
import operations.nn.conv.conv2dBackwardInputGemm;
import operations.nn.conv.conv2dBackwardWeight;
import operations.nn.conv.conv2dBackwardWeightGemm;
import operations.nn.conv.conv2dGemm;
import operations.nn.pool.avgPool2d;
import operations.nn.pool.avgPool2dBackwardInput;
import operations.nn.pool.maxPool2d;
import operations.nn.pool.maxPool2dBackwardInput;
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
                || opType == Operation.OpType.AVG_POOL2D
                || opType == Operation.OpType.CONV2D_BACKWARD_INPUT
                || opType == Operation.OpType.CONV2D_BACKWARD_INPUT_GEMM
                || opType == Operation.OpType.CONV2D_BACKWARD_WEIGHT
                || opType == Operation.OpType.CONV2D_BACKWARD_WEIGHT_GEMM
                || opType == Operation.OpType.MAX_POOL2D_BACKWARD_INPUT
                || opType == Operation.OpType.AVG_POOL2D_BACKWARD_INPUT;
    }

    static String unsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        Operation.OpType opType = node.operation().opType();
        return switch (opType) {
            case CONV2D -> conv2dUnsupportedReason(node, context);
            case CONV2D_GEMM -> conv2dGemmUnsupportedReason(node, context);
            case CONV2D_BACKWARD_INPUT, CONV2D_BACKWARD_INPUT_GEMM -> conv2dBackwardInputUnsupportedReason(node, context);
            case CONV2D_BACKWARD_WEIGHT, CONV2D_BACKWARD_WEIGHT_GEMM -> conv2dBackwardWeightUnsupportedReason(node, context);
            case MAX_POOL2D, AVG_POOL2D -> poolUnsupportedReason(node, context);
            case MAX_POOL2D_BACKWARD_INPUT -> maxPoolBackwardInputUnsupportedReason(node, context);
            case AVG_POOL2D_BACKWARD_INPUT -> avgPoolBackwardInputUnsupportedReason(node, context);
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
        return conv2dLikeUnsupportedReason("CONV2D", node, context, conv.getOptions(), conv.hasBias());
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
        return conv2dLikeUnsupportedReason("CONV2D_GEMM", node, context, conv.getOptions(), conv.hasBias());
    }

    private static String conv2dBackwardInputUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        String opName = node.operation().opType().name();
        String common = commonBackwardReason(opName, node, context);
        if (!common.isBlank()) {
            return common;
        }
        Conv2dOptions options;
        int[] inputGradShape;
        if (node.operation() instanceof conv2dBackwardInput op) {
            options = op.getOptions();
            inputGradShape = op.getInputShape();
        } else if (node.operation() instanceof conv2dBackwardInputGemm op) {
            options = op.getOptions();
            inputGradShape = op.getInputShape();
        } else {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " descriptor is unavailable";
        }
        if (node.inputIds().size() != 2) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " requires weight and output-gradient inputs";
        }
        CompiledNode weight = context.compiledNode(node.inputIds().get(0));
        CompiledNode outGrad = context.compiledNode(node.inputIds().get(1));
        if (weight == null || outGrad == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " inputs are unavailable";
        }
        String dtypeReason = requireMatchedFloating(opName, node, weight, outGrad);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense(opName, weight, outGrad);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        if (weight.shape().length != 4 || outGrad.shape().length != 4 || node.shape().length != 4 || inputGradShape.length != 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " requires rank-4 NCHW/OIHW tensors";
        }
        if (!Arrays.equals(node.shape(), inputGradShape)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " output shape must match requested input gradient shape";
        }
        return validateConv2dBackwardGeometry(opName, node.shape(), weight.shape(), outGrad.shape(), options);
    }

    private static String conv2dBackwardWeightUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        String opName = node.operation().opType().name();
        String common = commonBackwardReason(opName, node, context);
        if (!common.isBlank()) {
            return common;
        }
        Conv2dOptions options;
        int[] weightGradShape;
        if (node.operation() instanceof conv2dBackwardWeight op) {
            options = op.getOptions();
            weightGradShape = op.getWeightShape();
        } else if (node.operation() instanceof conv2dBackwardWeightGemm op) {
            options = op.getOptions();
            weightGradShape = op.getWeightShape();
        } else {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " descriptor is unavailable";
        }
        if (node.inputIds().size() != 2) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " requires input and output-gradient inputs";
        }
        CompiledNode input = context.compiledNode(node.inputIds().get(0));
        CompiledNode outGrad = context.compiledNode(node.inputIds().get(1));
        if (input == null || outGrad == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " inputs are unavailable";
        }
        String dtypeReason = requireMatchedFloating(opName, node, input, outGrad);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense(opName, input, outGrad);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        if (input.shape().length != 4 || outGrad.shape().length != 4 || node.shape().length != 4 || weightGradShape.length != 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " requires rank-4 NCHW/OIHW tensors";
        }
        if (!Arrays.equals(node.shape(), weightGradShape)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " output shape must match requested weight gradient shape";
        }
        return validateConv2dBackwardGeometry(opName, input.shape(), node.shape(), outGrad.shape(), options);
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
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " inputs are unavailable";
        }
        String dtypeReason = requireMatchedFloating(opName, node, input, weight, bias);
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
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " requires rank-4 NCHW input/output and OIHW weight";
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
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " bias must have shape [outChannels]";
            }
        }
        if (inChannels % options.groups() != 0 || outChannels % options.groups() != 0
                || channelsPerGroup * options.groups() != inChannels) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " groups/channel contract is invalid";
        }
        if (options.groups() != 1) {
            return "CAPABILITY_MISSING: GPU_METAL " + opName + " grouped/depthwise native execution is not implemented; family=CONV_POOL";
        }
        if (options.dilationH() != 1 || options.dilationW() != 1) {
            return "CAPABILITY_MISSING: GPU_METAL " + opName + " dilation native execution is not implemented; family=CONV_POOL";
        }
        if (options.strideH() > 255 || options.strideW() > 255 || options.padH() > 255 || options.padW() > 255) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " stride/padding metadata exceeds native DAG encoding";
        }
        int outH = inferConvOutput(inH, kernelH, options.padH(), options.strideH(), options.dilationH(), "height");
        if (outH < 0) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " effective kernel does not fit input height";
        }
        int outW = inferConvOutput(inW, kernelW, options.padW(), options.strideW(), options.dilationW(), "width");
        if (outW < 0) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " effective kernel does not fit input width";
        }
        if (!Arrays.equals(outShape, new int[]{n, outChannels, outH, outW})) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " output shape does not match NCHW/OIHW contract";
        }
        return "";
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
        String dtypeReason = requireMatchedFloating(opName, node, input);
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
        if (options.ceilMode()) {
            return "CAPABILITY_MISSING: GPU_METAL " + opName + " ceilMode=true native output-shape semantics are not implemented; family=CONV_POOL";
        }
        String metadataReason = poolMetadataReason(opName, options);
        if (!metadataReason.isBlank()) {
            return metadataReason;
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
        return "";
    }

    private static String avgPoolBackwardInputUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        String opName = Operation.OpType.AVG_POOL2D_BACKWARD_INPUT.name();
        String common = commonBackwardReason(opName, node, context);
        if (!common.isBlank()) {
            return common;
        }
        if (!(node.operation() instanceof avgPool2dBackwardInput op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL AVG_POOL2D_BACKWARD_INPUT descriptor is unavailable";
        }
        if (node.inputIds().size() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL AVG_POOL2D_BACKWARD_INPUT requires one output-gradient input";
        }
        CompiledNode outGrad = context.compiledNode(node.inputIds().getFirst());
        if (outGrad == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL AVG_POOL2D_BACKWARD_INPUT input is unavailable";
        }
        String dtypeReason = requireMatchedFloating(opName, node, outGrad);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense(opName, outGrad);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        int[] inputShape = op.getInputShape();
        int[] outGradShape = outGrad.shape();
        if (inputShape.length != 4 || outGradShape.length != 4 || node.shape().length != 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL AVG_POOL2D_BACKWARD_INPUT requires rank-4 NCHW tensors";
        }
        if (!Arrays.equals(node.shape(), inputShape)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL AVG_POOL2D_BACKWARD_INPUT output shape must match original input shape";
        }
        Pool2dOptions options = op.getOptions();
        if (options.countIncludePad()) {
            return "CAPABILITY_MISSING: GPU_METAL AVG_POOL2D_BACKWARD_INPUT countIncludePad=true native divisor semantics are not implemented; family=CONV_POOL";
        }
        String metadataReason = poolMetadataReason(opName, options);
        if (!metadataReason.isBlank()) {
            return metadataReason;
        }
        int outH = inferPoolOutput(inputShape[2], options.kernelH(), options.padH(), options.strideH(), "height");
        int outW = inferPoolOutput(inputShape[3], options.kernelW(), options.padW(), options.strideW(), "width");
        if (outH < 0 || outW < 0) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL AVG_POOL2D_BACKWARD_INPUT kernel does not fit input shape";
        }
        if (!Arrays.equals(outGradShape, new int[]{inputShape[0], inputShape[1], outH, outW})) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL AVG_POOL2D_BACKWARD_INPUT gradient input shape must match forward output shape";
        }
        return "";
    }

    private static String maxPoolBackwardInputUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        String opName = Operation.OpType.MAX_POOL2D_BACKWARD_INPUT.name();
        String common = commonBackwardReason(opName, node, context);
        if (!common.isBlank()) {
            return common;
        }
        if (!(node.operation() instanceof maxPool2dBackwardInput op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL MAX_POOL2D_BACKWARD_INPUT descriptor is unavailable";
        }
        if (node.inputIds().size() != 2) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL MAX_POOL2D_BACKWARD_INPUT requires output-gradient and original source inputs";
        }
        CompiledNode outGrad = context.compiledNode(node.inputIds().get(0));
        CompiledNode source = context.compiledNode(node.inputIds().get(1));
        if (outGrad == null || source == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL MAX_POOL2D_BACKWARD_INPUT inputs are unavailable";
        }
        String dtypeReason = requireMatchedFloating(opName, node, outGrad, source);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense(opName, outGrad, source);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        int[] inputShape = op.getInputShape();
        int[] outGradShape = outGrad.shape();
        int[] sourceShape = source.shape();
        if (inputShape.length != 4 || outGradShape.length != 4 || sourceShape.length != 4 || node.shape().length != 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL MAX_POOL2D_BACKWARD_INPUT requires rank-4 NCHW tensors";
        }
        if (!Arrays.equals(node.shape(), inputShape) || !Arrays.equals(sourceShape, inputShape)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL MAX_POOL2D_BACKWARD_INPUT output and source shapes must match original input shape";
        }
        Pool2dOptions options = op.getOptions();
        String metadataReason = poolMetadataReason(opName, options);
        if (!metadataReason.isBlank()) {
            return metadataReason;
        }
        int outH = inferPoolOutput(inputShape[2], options.kernelH(), options.padH(), options.strideH(), "height");
        int outW = inferPoolOutput(inputShape[3], options.kernelW(), options.padW(), options.strideW(), "width");
        if (outH < 0 || outW < 0) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL MAX_POOL2D_BACKWARD_INPUT kernel does not fit input shape";
        }
        if (!Arrays.equals(outGradShape, new int[]{inputShape[0], inputShape[1], outH, outW})) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL MAX_POOL2D_BACKWARD_INPUT gradient input shape must match forward output shape";
        }
        return "";
    }

    private static String validateConv2dBackwardGeometry(
            String opName,
            int[] inputShape,
            int[] weightShape,
            int[] outGradShape,
            Conv2dOptions options
    ) {
        int inChannels = inputShape[1];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        if (inChannels % options.groups() != 0 || outChannels % options.groups() != 0
                || channelsPerGroup * options.groups() != inChannels) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " groups/channel contract is invalid";
        }
        if (options.groups() != 1) {
            return "CAPABILITY_MISSING: GPU_METAL " + opName + " grouped/depthwise native execution is not implemented; family=CONV_POOL";
        }
        if (options.dilationH() != 1 || options.dilationW() != 1) {
            return "CAPABILITY_MISSING: GPU_METAL " + opName + " dilation native execution is not implemented; family=CONV_POOL";
        }
        if (options.strideH() > 255 || options.strideW() > 255 || options.padH() > 255 || options.padW() > 255) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " stride/padding metadata exceeds native DAG encoding";
        }
        int outH = inferConvOutput(inputShape[2], kernelH, options.padH(), options.strideH(), options.dilationH(), "height");
        int outW = inferConvOutput(inputShape[3], kernelW, options.padW(), options.strideW(), options.dilationW(), "width");
        if (outH < 0 || outW < 0) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " effective kernel does not fit input shape";
        }
        if (!Arrays.equals(outGradShape, new int[]{inputShape[0], outChannels, outH, outW})) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " output-gradient shape does not match forward conv contract";
        }
        return "";
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

    private static String commonBackwardReason(String opName, CompiledNode node, PartitionPlanningContext context) {
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " requires planning context";
        }
        return "";
    }

    private static String poolMetadataReason(String opName, Pool2dOptions options) {
        if (options.kernelH() > 15 || options.kernelW() > 15
                || options.strideH() > 15 || options.strideW() > 15
                || options.padH() > 15 || options.padW() > 15) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " kernel/stride/padding metadata exceeds native DAG encoding";
        }
        return "";
    }

    private static String requireMatchedFloating(String opName, CompiledNode output, CompiledNode... inputs) {
        if (!isMetalFloatingDType(output.dataType())) {
            return "UNSUPPORTED_DTYPE: GPU_METAL " + opName + " output requires FLOAT32/BFLOAT16";
        }
        for (CompiledNode input : inputs) {
            if (input != null && input.dataType() != output.dataType()) {
                return "UNSUPPORTED_DTYPE: GPU_METAL " + opName + " inputs and output must use the same FLOAT32/BFLOAT16 dtype";
            }
        }
        return "";
    }

    private static boolean isMetalFloatingDType(DataType dtype) {
        return dtype == DataType.FLOAT32 || dtype == DataType.BFLOAT16;
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
