package tensor.ops.normalization;

import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypes;

final class NormalizationSupport {
    private NormalizationSupport() {
    }

    static void validateMatchingTailParameters(Tensor input, Tensor gamma, Tensor beta, String opName) {
        if (gamma.getShapeUnsafe().length != beta.getShapeUnsafe().length) {
            throw new IllegalArgumentException(opName + " requires gamma and beta to have the same rank.");
        }
        int[] gammaShape = gamma.getShapeUnsafe();
        int[] betaShape = beta.getShapeUnsafe();
        if (!java.util.Arrays.equals(gammaShape, betaShape)) {
            throw new IllegalArgumentException(opName + " requires gamma and beta to have identical shapes.");
        }
        validateMatchingTailParameter(input, gamma, opName + " gamma");
    }

    static void validateMatchingTailParameter(Tensor input, Tensor parameter, String name) {
        int[] inputShape = input.getShapeUnsafe();
        int[] parameterShape = parameter.getShapeUnsafe();
        if (parameterShape.length == 0 || parameterShape.length > inputShape.length) {
            throw new IllegalArgumentException(name + " must match a non-empty trailing shape of the input.");
        }
        int offset = inputShape.length - parameterShape.length;
        for (int i = 0; i < parameterShape.length; i++) {
            if (inputShape[offset + i] != parameterShape[i]) {
                throw new IllegalArgumentException(name + " shape must match the trailing input dimensions.");
            }
        }
    }

    static void validateChannelParameter(Tensor parameter, int channels, String name) {
        int[] shape = parameter.getShapeUnsafe();
        if (shape.length != 1 || shape[0] != channels) {
            throw new IllegalArgumentException(name + " must have shape [channels].");
        }
    }

    static void requireFloating(Tensor tensor, String name) {
        if (tensor == null) {
            throw new IllegalArgumentException(name + " cannot be null");
        }
        if (tensor.getDataType() == DataType.BOOL || tensor.getDataType() == DataType.INT32 || tensor.getDataType() == DataType.INT64) {
            throw new IllegalArgumentException(name + " must use a floating dtype.");
        }
    }

    static void requirePositiveEpsilon(double epsilon, String opName) {
        if (!(epsilon > 0.0)) {
            throw new IllegalArgumentException(opName + " epsilon must be positive.");
        }
    }

    static Tensor normalizeWithStats(
            Tensor input,
            Tensor gamma,
            Tensor beta,
            Tensor mean,
            Tensor variance,
            int channelDimension,
            double epsilon
    ) {
        int[] broadcastShape = new int[input.getShapeUnsafe().length];
        java.util.Arrays.fill(broadcastShape, 1);
        broadcastShape[channelDimension] = input.getShapeUnsafe()[channelDimension];

        Tensor meanView = mean.reshape(broadcastShape);
        Tensor varianceView = variance.reshape(broadcastShape);
        Tensor gammaView = gamma.reshape(broadcastShape);
        Tensor betaView = beta.reshape(broadcastShape);
        Tensor epsilonTensor = Tensor.scalar(epsilon, TensorDTypes.promoteFloating(input.getDataType(), gamma.getDataType()));

        return input
                .sub(meanView)
                .div(varianceView.add(epsilonTensor).sqrt())
                .mul(gammaView)
                .add(betaView);
    }

    static Tensor reduceAllButOne(Tensor input, int preservedAxis) {
        Tensor reduced = input;
        for (int axis = 0; axis < input.getShapeUnsafe().length; axis++) {
            if (axis == preservedAxis) {
                continue;
            }
            reduced = reduced.mean(axis, true);
        }
        return reduced.reshape(new int[]{input.getShapeUnsafe()[preservedAxis]});
    }

    static Tensor reshapeChannelParameter(int[] inputShape, int channelDimension, Tensor parameter) {
        int[] broadcastShape = new int[inputShape.length];
        java.util.Arrays.fill(broadcastShape, 1);
        broadcastShape[channelDimension] = inputShape[channelDimension];
        return parameter.reshape(broadcastShape);
    }

    static Tensor reduceTrailingKeepDims(Tensor input, int normalizedRank) {
        int inputRank = input.getShapeUnsafe().length;
        int startAxis = inputRank - normalizedRank;
        Tensor reduced = input;
        for (int axis = startAxis; axis < inputRank; axis++) {
            reduced = reduced.mean(axis, true);
        }
        return reduced;
    }

    static Tensor reduceLeadingKeepDims(Tensor input, int trailingRank) {
        int reductions = input.getShapeUnsafe().length - trailingRank;
        Tensor reduced = input;
        for (int i = 0; i < reductions; i++) {
            reduced = reduced.sum(i, true);
        }
        return reduced;
    }
}
