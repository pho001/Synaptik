package tensor.ops.layout;

import tensor.DataType;
import tensor.Tensor;
import tensor.options.Window2dOptions;

final class Window2dShapeRules {
    private Window2dShapeRules() {
    }

    static void requireFloatingRank4(Tensor input, String opName) {
        if (input == null) {
            throw new IllegalArgumentException(opName + " input cannot be null");
        }
        if (input.getShapeUnsafe().length != 4) {
            throw new IllegalArgumentException(opName + " requires rank-4 NCHW input.");
        }
        DataType dataType = input.getDataType();
        if (dataType == DataType.BOOL || dataType == DataType.INT32 || dataType == DataType.INT64) {
            throw new IllegalArgumentException(opName + " requires floating input.");
        }
    }

    static int inferOutputSize(int inputSize, int kernelSize, int pad, int stride, int dilation, boolean ceilMode, String axisName) {
        int effectiveKernel = Math.addExact(Math.multiplyExact(dilation, kernelSize - 1), 1);
        int numerator = Math.subtractExact(Math.addExact(inputSize, Math.multiplyExact(2, pad)), effectiveKernel);
        if (numerator < 0) {
            throw new IllegalArgumentException("Window2d effective kernel does not fit input " + axisName + ".");
        }
        return (ceilMode ? (numerator + stride - 1) / stride : numerator / stride) + 1;
    }

    static int[] unfoldOutputShape(int[] inputShape, Window2dOptions options) {
        if (inputShape == null || inputShape.length != 4) {
            throw new IllegalArgumentException("unfold2d requires rank-4 NCHW input shape.");
        }
        int outH = inferOutputSize(inputShape[2], options.kernelH(), options.padH(), options.strideH(), options.dilationH(), options.ceilMode(), "height");
        int outW = inferOutputSize(inputShape[3], options.kernelW(), options.padW(), options.strideW(), options.dilationW(), options.ceilMode(), "width");
        int windowChannels = Math.multiplyExact(inputShape[1], Math.multiplyExact(options.kernelH(), options.kernelW()));
        int windowCount = Math.multiplyExact(outH, outW);
        return new int[]{inputShape[0], windowChannels, windowCount};
    }

    static void validateFoldShapes(int[] inputShape, int[] outputShape, Window2dOptions options) {
        if (inputShape == null || inputShape.length != 3) {
            throw new IllegalArgumentException("fold2d requires rank-3 column input.");
        }
        if (outputShape == null || outputShape.length != 4) {
            throw new IllegalArgumentException("fold2d output shape must be rank-4 NCHW.");
        }
        int kernelArea = Math.multiplyExact(options.kernelH(), options.kernelW());
        if (inputShape[1] % kernelArea != 0) {
            throw new IllegalArgumentException("fold2d channel-window dimension is not divisible by kernel area.");
        }
        int channels = inputShape[1] / kernelArea;
        if (outputShape[0] != inputShape[0] || outputShape[1] != channels) {
            throw new IllegalArgumentException("fold2d output shape is incompatible with column input shape.");
        }
        int outH = inferOutputSize(outputShape[2], options.kernelH(), options.padH(), options.strideH(), options.dilationH(), options.ceilMode(), "height");
        int outW = inferOutputSize(outputShape[3], options.kernelW(), options.padW(), options.strideW(), options.dilationW(), options.ceilMode(), "width");
        if (inputShape[2] != Math.multiplyExact(outH, outW)) {
            throw new IllegalArgumentException("fold2d column count does not match output shape and window geometry.");
        }
    }
}
