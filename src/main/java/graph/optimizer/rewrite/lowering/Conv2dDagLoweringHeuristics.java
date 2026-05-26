package graph.optimizer.rewrite.lowering;

import config.optimizer.Conv2dDagLoweringProfile;
import operations.nn.conv.conv2d;
import tensor.Tensor;
import tensor.options.Conv2dOptions;

final class Conv2dDagLoweringHeuristics {
    private Conv2dDagLoweringHeuristics() {
    }

    static boolean shouldLower(Tensor tensor, conv2d conv, Conv2dDagLoweringProfile profile) {
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
        return shouldLower(inputShape, weightShape, outShape, conv.getOptions(), profile);
    }

    private static boolean shouldLower(
            int[] inputShape,
            int[] weightShape,
            int[] outShape,
            Conv2dOptions options,
            Conv2dDagLoweringProfile profile
    ) {
        if (inputShape.length != 4 || weightShape.length != 4 || outShape.length != 4) {
            return false;
        }
        Conv2dDagLoweringProfile resolved = profile == null ? Conv2dDagLoweringProfile.defaults() : profile;

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
        long spatial = (long) batch * outH * outW;

        boolean pointwise = kernelH == 1
                && kernelW == 1
                && options.strideH() == 1
                && options.strideW() == 1
                && options.padH() == 0
                && options.padW() == 0;
        if (pointwise) {
            return inChannels >= resolved.pointwiseMinInChannels()
                    && outChannels >= resolved.pointwiseMinOutChannels()
                    && outChannels <= Math.floor(inChannels * resolved.pointwiseMaxOutOverIn())
                    && spatial >= resolved.pointwiseMinSpatial();
        }

        boolean standard3x3 = kernelH == 3
                && kernelW == 3
                && options.strideH() == 1
                && options.strideW() == 1
                && options.padH() == 1
                && options.padW() == 1;
        if (standard3x3) {
            return inChannels >= resolved.standard3x3MinInChannels()
                    && outChannels >= resolved.standard3x3MinOutChannels()
                    && spatial >= resolved.standard3x3MinSpatial();
        }

        return false;
    }
}
