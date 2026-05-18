package tensor.ops.normalization;

import tensor.DataType;
import tensor.Tensor;
import tensor.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;

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
        NormalizationSupport.requireFloating(input, "layerNorm input");
        NormalizationSupport.requireFloating(gamma, "layerNorm gamma");
        NormalizationSupport.requireFloating(beta, "layerNorm beta");
        NormalizationSupport.requirePositiveEpsilon(epsilon, "layerNorm");
        NormalizationSupport.validateMatchingTailParameters(input, gamma, beta, "layerNorm");

        int normalizedRank = gamma.getShapeUnsafe().length;
        DataType outputType = TensorDataTypeUtil.promote(TensorDataTypeUtil.promote(input.getDataType(), gamma.getDataType()), beta.getDataType());
        Tensor out = TensorPrimitiveBuilder.ternary(
                input,
                gamma,
                beta,
                input.getShape().clone(),
                new operations.normalization.layerNorm(normalizedRank, epsilon),
                "layerNorm",
                outputType
        );
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }

            Tensor epsilonTensor = Tensor.scalar(epsilon, outputType);
            Tensor mean = NormalizationSupport.reduceTrailingKeepDims(input, normalizedRank);
            Tensor centered = input.sub(mean);
            Tensor variance = NormalizationSupport.reduceTrailingKeepDims(centered.pow(2.0), normalizedRank);
            Tensor invStd = variance.add(epsilonTensor).sqrt().inv();
            Tensor xHat = centered.mul(invStd);

            if (input.getRequiresGrad()) {
                double normalizedSize = gamma.getFlatDataSize();
                Tensor dxHat = outGrad.mul(gamma);
                Tensor sumDxHat = NormalizationSupport.reduceTrailingKeepDims(dxHat, normalizedRank);
                Tensor sumDxHatXHat = NormalizationSupport.reduceTrailingKeepDims(dxHat.mul(xHat), normalizedRank);
                Tensor inputGrad = dxHat.mul(normalizedSize)
                        .sub(sumDxHat)
                        .sub(xHat.mul(sumDxHatXHat))
                        .mul(invStd)
                        .mul(1.0d / normalizedSize);
                NormalizationSupport.accumulateGradient(input, inputGrad);
            }

            if (gamma.getRequiresGrad()) {
                Tensor gammaGrad = outGrad.mul(xHat);
                gammaGrad = NormalizationSupport.reduceLeadingKeepDims(gammaGrad, normalizedRank).reshape(gamma.getShape());
                NormalizationSupport.accumulateGradient(gamma, gammaGrad);
            }

            if (beta.getRequiresGrad()) {
                Tensor betaGrad = NormalizationSupport.reduceLeadingKeepDims(outGrad, normalizedRank).reshape(beta.getShape());
                NormalizationSupport.accumulateGradient(beta, betaGrad);
            }
        });
        return out;
    }
}
