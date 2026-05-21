package tensor.ops.normalization;

import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypes;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for differentiable layer normalization.
 */
public final class LayerNormOp {
    private LayerNormOp() {
    }

    /**
     * Applies layer normalization over the trailing dimensions represented by {@code gamma}.
     *
     * @param input floating input tensor
     * @param gamma scale parameter whose shape must match the normalized trailing axes
     * @param beta bias parameter with the same shape as {@code gamma}
     * @param epsilon positive value added to variance for numerical stability
     * @return normalized tensor with the same shape as {@code input}
     * @throws IllegalArgumentException if inputs are null/non-floating, parameter
     *                                  shapes do not match the input tail, or epsilon is non-positive
     */
    public static Tensor build(
            Tensor input,
            Tensor gamma,
            Tensor beta,
            double epsilon
    ) {
        NormalizationRules.requireFloating(input, "layerNorm input");
        NormalizationRules.requireFloating(gamma, "layerNorm gamma");
        NormalizationRules.requireFloating(beta, "layerNorm beta");
        NormalizationRules.requirePositiveEpsilon(epsilon, "layerNorm");
        NormalizationRules.validateMatchingTailParameters(input, gamma, beta, "layerNorm");

        int normalizedRank = gamma.getShapeUnsafe().length;
        DataType outputType = TensorDTypes.promoteFloating(TensorDTypes.promoteFloating(input.getDataType(), gamma.getDataType()), beta.getDataType());
        Tensor out = TensorPrimitiveBuilder.ternary(
                input,
                gamma,
                beta,
                input.getShape().clone(),
                new operations.normalization.layerNorm(normalizedRank, epsilon),
                "layerNorm",
                outputType
        );
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }

            Tensor epsilonTensor = Tensor.scalar(epsilon, outputType);
            Tensor mean = NormalizationRules.reduceTrailingKeepDims(input, normalizedRank);
            Tensor centered = input.sub(mean);
            Tensor variance = NormalizationRules.reduceTrailingKeepDims(centered.pow(2.0), normalizedRank);
            Tensor invStd = variance.add(epsilonTensor).sqrt().inv();
            Tensor xHat = centered.mul(invStd);

            if (input.getRequiresGrad()) {
                double normalizedSize = gamma.getFlatDataSize();
                Tensor dxHat = outGrad.mul(gamma);
                Tensor sumDxHat = NormalizationRules.reduceTrailingKeepDims(dxHat, normalizedRank);
                Tensor sumDxHatXHat = NormalizationRules.reduceTrailingKeepDims(dxHat.mul(xHat), normalizedRank);
                Tensor inputGrad = dxHat.mul(normalizedSize)
                        .sub(sumDxHat)
                        .sub(xHat.mul(sumDxHatXHat))
                        .mul(invStd)
                        .mul(1.0d / normalizedSize);
                context.accumulate(input, inputGrad);
            }

            if (gamma.getRequiresGrad()) {
                Tensor gammaGrad = outGrad.mul(xHat);
                gammaGrad = NormalizationRules.reduceLeadingKeepDims(gammaGrad, normalizedRank).reshape(gamma.getShape());
                context.accumulate(gamma, gammaGrad);
            }

            if (beta.getRequiresGrad()) {
                Tensor betaGrad = NormalizationRules.reduceLeadingKeepDims(outGrad, normalizedRank).reshape(beta.getShape());
                context.accumulate(beta, betaGrad);
            }
        });
        return out;
    }
}
