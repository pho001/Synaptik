package tensor.ops.normalization;

import operations.normalization.rmsNorm;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypes;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for differentiable RMS normalization.
 */
public final class RmsNormOp {
    private RmsNormOp() {
    }

    /**
     * Applies RMS normalization over the trailing dimensions represented by {@code gamma}.
     *
     * @param input floating input tensor
     * @param gamma scale parameter whose shape must match the normalized trailing axes
     * @param epsilon positive value added for numerical stability
     * @return normalized tensor with the same shape as {@code input}
     * @throws IllegalArgumentException if inputs are null/non-floating, parameter
     *                                  shape does not match the input tail, or epsilon is non-positive
     */
    public static Tensor build(
            Tensor input,
            Tensor gamma,
            double epsilon
    ) {
        NormalizationRules.requireFloating(input, "rmsNorm input");
        NormalizationRules.requireFloating(gamma, "rmsNorm gamma");
        NormalizationRules.requirePositiveEpsilon(epsilon, "rmsNorm");
        NormalizationRules.validateMatchingTailParameter(input, gamma, "rmsNorm gamma");

        int normalizedRank = gamma.getShapeUnsafe().length;
        DataType outputType = TensorDTypes.promoteFloating(input.getDataType(), gamma.getDataType());
        Tensor out = TensorPrimitiveBuilder.binary(
                input,
                gamma,
                input.getShape().clone(),
                new rmsNorm(normalizedRank, epsilon),
                "rmsNorm",
                outputType
        );
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }

            Tensor epsilonTensor = Tensor.scalar(epsilon, outputType);
            Tensor meanSquares = NormalizationRules.reduceTrailingKeepDims(input.pow(2.0), normalizedRank);
            Tensor invRms = meanSquares.add(epsilonTensor).sqrt().inv();

            if (input.getRequiresGrad()) {
                Tensor weighted = outGrad.mul(gamma);
                Tensor dotMean = NormalizationRules.reduceTrailingKeepDims(weighted.mul(input), normalizedRank);
                Tensor invRmsCubed = invRms.mul(invRms).mul(invRms);
                Tensor inputGrad = weighted.mul(invRms).sub(input.mul(dotMean).mul(invRmsCubed));
                context.accumulate(input, inputGrad);
            }

            if (gamma.getRequiresGrad()) {
                Tensor gammaGrad = outGrad.mul(input).mul(invRms);
                gammaGrad = NormalizationRules.reduceLeadingKeepDims(gammaGrad, normalizedRank).reshape(gamma.getShape());
                context.accumulate(gamma, gammaGrad);
            }
        });
        return out;
    }
}
