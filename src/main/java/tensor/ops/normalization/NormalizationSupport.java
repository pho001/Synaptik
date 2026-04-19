package tensor.ops.normalization;

import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

final class NormalizationSupport {
    private NormalizationSupport() {
    }

    static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            TensorInternalAccess.setGradient(input, gradientDelta);
            return;
        }
        TensorInternalAccess.setGradient(input, input.getGradient().add(gradientDelta));
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
        if (tensor.getDataType() == DataType.BOOL || tensor.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException(name + " must use a floating dtype.");
        }
    }

    static void requirePositiveEpsilon(double epsilon, String opName) {
        if (!(epsilon > 0.0)) {
            throw new IllegalArgumentException(opName + " epsilon must be positive.");
        }
    }
}
