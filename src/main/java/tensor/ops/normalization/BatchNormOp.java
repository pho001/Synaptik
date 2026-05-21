package tensor.ops.normalization;

import tensor.Tensor;
import tensor.layout.TensorLayoutTransform;

/**
 * Graph-building definition for differentiable batch normalization.
 */
public final class BatchNormOp {
    private BatchNormOp() {
    }

    /**
     * Applies batch normalization using mean and variance computed from {@code input}.
     *
     * @param input floating input tensor with at least two axes
     * @param gamma rank-1 scale parameter shaped {@code [channels]}
     * @param beta rank-1 bias parameter shaped {@code [channels]}
     * @param channelDimension channel axis; negative axes are normalized
     * @param epsilon positive value added to variance for numerical stability
     * @return normalized tensor with the same shape as {@code input}
     * @throws IllegalArgumentException if inputs are null/non-floating, parameter
     *                                  shapes are invalid, axis is invalid, or epsilon is non-positive
     */
    public static Tensor build(
            Tensor input,
            Tensor gamma,
            Tensor beta,
            int channelDimension,
            double epsilon
    ) {
        NormalizationRules.requireFloating(input, "batchNorm input");
        NormalizationRules.requireFloating(gamma, "batchNorm gamma");
        NormalizationRules.requireFloating(beta, "batchNorm beta");
        NormalizationRules.requirePositiveEpsilon(epsilon, "batchNorm");

        int[] inputShape = input.getShapeUnsafe();
        if (inputShape.length < 2) {
            throw new IllegalArgumentException("batchNorm requires at least one non-channel axis.");
        }
        int normalizedChannel = TensorLayoutTransform.normalizeAxis(channelDimension, inputShape.length);
        NormalizationRules.validateChannelParameter(gamma, inputShape[normalizedChannel], "batchNorm gamma");
        NormalizationRules.validateChannelParameter(beta, inputShape[normalizedChannel], "batchNorm beta");

        Tensor mean = NormalizationRules.reduceAllButOne(input, normalizedChannel);
        Tensor meanView = NormalizationRules.reshapeChannelParameter(inputShape, normalizedChannel, mean);
        Tensor centered = input.sub(meanView);
        Tensor variance = NormalizationRules.reduceAllButOne(centered.pow(2.0), normalizedChannel);
        Tensor out = NormalizationRules.normalizeWithStats(input, gamma, beta, mean, variance, normalizedChannel, epsilon);
        out.setLabel("batchNorm");
        return out;
    }

    /**
     * Applies batch normalization using supplied running statistics.
     *
     * @param input floating input tensor
     * @param gamma rank-1 scale parameter shaped {@code [channels]}
     * @param beta rank-1 bias parameter shaped {@code [channels]}
     * @param mean rank-1 mean tensor shaped {@code [channels]}
     * @param variance rank-1 variance tensor shaped {@code [channels]}
     * @param channelDimension channel axis; negative axes are normalized
     * @param epsilon positive value added to variance for numerical stability
     * @return normalized tensor with the same shape as {@code input}
     * @throws IllegalArgumentException if inputs are null/non-floating, parameter
     *                                  shapes are invalid, axis is invalid, or epsilon is non-positive
     */
    public static Tensor build(
            Tensor input,
            Tensor gamma,
            Tensor beta,
            Tensor mean,
            Tensor variance,
            int channelDimension,
            double epsilon
    ) {
        NormalizationRules.requireFloating(input, "batchNorm input");
        NormalizationRules.requireFloating(gamma, "batchNorm gamma");
        NormalizationRules.requireFloating(beta, "batchNorm beta");
        NormalizationRules.requireFloating(mean, "batchNorm mean");
        NormalizationRules.requireFloating(variance, "batchNorm variance");
        NormalizationRules.requirePositiveEpsilon(epsilon, "batchNorm");

        int[] inputShape = input.getShapeUnsafe();
        int normalizedChannel = TensorLayoutTransform.normalizeAxis(channelDimension, inputShape.length);
        int channels = inputShape[normalizedChannel];
        NormalizationRules.validateChannelParameter(gamma, channels, "batchNorm gamma");
        NormalizationRules.validateChannelParameter(beta, channels, "batchNorm beta");
        NormalizationRules.validateChannelParameter(mean, channels, "batchNorm mean");
        NormalizationRules.validateChannelParameter(variance, channels, "batchNorm variance");

        Tensor out = NormalizationRules.normalizeWithStats(input, gamma, beta, mean, variance, normalizedChannel, epsilon);
        out.setLabel("batchNorm");
        return out;
    }
}
