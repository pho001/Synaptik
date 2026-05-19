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
        NormalizationSupport.requireFloating(input, "batchNorm input");
        NormalizationSupport.requireFloating(gamma, "batchNorm gamma");
        NormalizationSupport.requireFloating(beta, "batchNorm beta");
        NormalizationSupport.requirePositiveEpsilon(epsilon, "batchNorm");

        int[] inputShape = input.getShapeUnsafe();
        if (inputShape.length < 2) {
            throw new IllegalArgumentException("batchNorm requires at least one non-channel axis.");
        }
        int normalizedChannel = TensorLayoutTransform.normalizeAxis(channelDimension, inputShape.length);
        NormalizationSupport.validateChannelParameter(gamma, inputShape[normalizedChannel], "batchNorm gamma");
        NormalizationSupport.validateChannelParameter(beta, inputShape[normalizedChannel], "batchNorm beta");

        Tensor mean = NormalizationSupport.reduceAllButOne(input, normalizedChannel);
        Tensor meanView = NormalizationSupport.reshapeChannelParameter(inputShape, normalizedChannel, mean);
        Tensor centered = input.sub(meanView);
        Tensor variance = NormalizationSupport.reduceAllButOne(centered.pow(2.0), normalizedChannel);
        Tensor out = NormalizationSupport.normalizeWithStats(input, gamma, beta, mean, variance, normalizedChannel, epsilon);
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
        NormalizationSupport.requireFloating(input, "batchNorm input");
        NormalizationSupport.requireFloating(gamma, "batchNorm gamma");
        NormalizationSupport.requireFloating(beta, "batchNorm beta");
        NormalizationSupport.requireFloating(mean, "batchNorm mean");
        NormalizationSupport.requireFloating(variance, "batchNorm variance");
        NormalizationSupport.requirePositiveEpsilon(epsilon, "batchNorm");

        int[] inputShape = input.getShapeUnsafe();
        int normalizedChannel = TensorLayoutTransform.normalizeAxis(channelDimension, inputShape.length);
        int channels = inputShape[normalizedChannel];
        NormalizationSupport.validateChannelParameter(gamma, channels, "batchNorm gamma");
        NormalizationSupport.validateChannelParameter(beta, channels, "batchNorm beta");
        NormalizationSupport.validateChannelParameter(mean, channels, "batchNorm mean");
        NormalizationSupport.validateChannelParameter(variance, channels, "batchNorm variance");

        Tensor out = NormalizationSupport.normalizeWithStats(input, gamma, beta, mean, variance, normalizedChannel, epsilon);
        out.setLabel("batchNorm");
        return out;
    }
}
