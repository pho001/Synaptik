package graph.optimizer.rewrite.lowering;

import operations.nn.conv.conv2d;
import operations.nn.conv.conv2dBackwardInput;
import operations.nn.conv.conv2dBackwardWeight;
import tensor.options.Conv2dOptions;
import tensor.Tensor;

final class Conv2dGemmLoweringHeuristics {
    private Conv2dGemmLoweringHeuristics() {
    }

    static boolean shouldLower(Tensor tensor, conv2d conv) {
        if (tensor == null || conv == null) {
            return false;
        }
        if (tensor.getPrevTensors() == null || tensor.getPrevTensors().size() < 2) {
            return false;
        }

        Tensor input = tensor.getPrevTensors().get(0);
        Tensor weight = tensor.getPrevTensors().get(1);
        int[] inputShape = input.getShapeUnsafe();
        int[] weightShape = weight.getShapeUnsafe();
        int[] outShape = tensor.getShapeUnsafe();
        return shouldLower(inputShape, weightShape, outShape, conv.getOptions());
    }

    static boolean shouldLower(Tensor tensor, conv2dBackwardInput conv) {
        if (tensor == null || conv == null) {
            return false;
        }
        if (tensor.getPrevTensors() == null || tensor.getPrevTensors().size() < 2) {
            return false;
        }
        Tensor weight = tensor.getPrevTensors().get(0);
        Tensor outGrad = tensor.getPrevTensors().get(1);
        return shouldLower(conv.getInputShape(), weight.getShapeUnsafe(), outGrad.getShapeUnsafe(), conv.getOptions());
    }

    static boolean shouldLower(Tensor tensor, conv2dBackwardWeight conv) {
        if (tensor == null || conv == null) {
            return false;
        }
        if (tensor.getPrevTensors() == null || tensor.getPrevTensors().size() < 2) {
            return false;
        }
        Tensor input = tensor.getPrevTensors().get(0);
        Tensor outGrad = tensor.getPrevTensors().get(1);
        return shouldLower(input.getShapeUnsafe(), conv.getWeightShape(), outGrad.getShapeUnsafe(), conv.getOptions());
    }

    private static boolean shouldLower(int[] inputShape, int[] weightShape, int[] outShape, Conv2dOptions options) {
        if (inputShape.length != 4 || weightShape.length != 4 || outShape.length != 4) {
            return false;
        }

        if (options.groups() != 1) {
            return false;
        }
        if (options.dilationH() != 1 || options.dilationW() != 1) {
            return false;
        }

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int outChannels = weightShape[0];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outShape[2];
        int outW = outShape[3];
        int spatial = batch * outH * outW;

        boolean pointwise = kernelH == 1
                && kernelW == 1
                && options.strideH() == 1
                && options.strideW() == 1
                && options.padH() == 0
                && options.padW() == 0;
        if (pointwise) {
            return inChannels >= 128
                    && outChannels >= 64
                    && outChannels <= inChannels * 2
                    && spatial >= 256;
        }

        boolean standard3x3 = kernelH == 3
                && kernelW == 3
                && options.strideH() == 1
                && options.strideW() == 1
                && options.padH() == 1
                && options.padW() == 1;
        if (standard3x3) {
            return inChannels >= 64
                    && outChannels >= 64
                    && spatial >= 512;
        }

        return false;
    }
}
