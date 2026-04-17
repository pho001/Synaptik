package graph.optimizer.rewrite;

import operations.conv2d;
import tensor.options.Conv2dOptions;
import tensor.Tensor;

final class Conv2dLoweringHeuristics {
    private Conv2dLoweringHeuristics() {
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
        if (inputShape.length != 4 || weightShape.length != 4 || outShape.length != 4) {
            return false;
        }

        Conv2dOptions options = conv.getOptions();
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
