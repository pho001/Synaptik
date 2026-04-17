package tensor.ops.normalization;

import operations.rmsNorm;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorDataTypeUtil;
import tensor.TensorLayoutTransform;
import tensor.TensorPrimitiveBuilder;

public final class TensorNormalizationOps {
    private TensorNormalizationOps() {
    }

    public static Tensor batchNorm(
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

        Tensor mean = reduceAllButOne(input, normalizedChannel);
        Tensor meanView = reshapeChannelParameter(inputShape, normalizedChannel, mean);
        Tensor centered = input.sub(meanView);
        Tensor variance = reduceAllButOne(centered.pow(2.0), normalizedChannel);
        Tensor out = normalizeWithStats(input, gamma, beta, mean, variance, normalizedChannel, epsilon);
        out.setLabel("batchNorm");
        return out;
    }

    public static Tensor batchNorm(
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

        Tensor out = normalizeWithStats(input, gamma, beta, mean, variance, normalizedChannel, epsilon);
        out.setLabel("batchNorm");
        return out;
    }

    public static Tensor layerNorm(
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
                new operations.layerNorm(normalizedRank, epsilon),
                "layerNorm",
                outputType
        );
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }

            Tensor epsilonTensor = Tensor.scalar(epsilon, outputType);
            Tensor mean = reduceTrailingKeepDims(input, normalizedRank);
            Tensor centered = input.sub(mean);
            Tensor variance = reduceTrailingKeepDims(centered.pow(2.0), normalizedRank);
            Tensor invStd = variance.add(epsilonTensor).sqrt().inv();
            Tensor xHat = centered.mul(invStd);

            if (input.getRequiresGrad()) {
                double normalizedSize = gamma.getFlatDataSize();
                Tensor dxHat = outGrad.mul(gamma);
                Tensor sumDxHat = reduceTrailingKeepDims(dxHat, normalizedRank);
                Tensor sumDxHatXHat = reduceTrailingKeepDims(dxHat.mul(xHat), normalizedRank);
                Tensor inputGrad = dxHat.mul(normalizedSize)
                        .sub(sumDxHat)
                        .sub(xHat.mul(sumDxHatXHat))
                        .mul(invStd)
                        .mul(1.0d / normalizedSize);
                NormalizationSupport.accumulateGradient(input, inputGrad);
            }

            if (gamma.getRequiresGrad()) {
                Tensor gammaGrad = outGrad.mul(xHat);
                gammaGrad = reduceLeadingKeepDims(gammaGrad, normalizedRank).reshape(gamma.getShape());
                NormalizationSupport.accumulateGradient(gamma, gammaGrad);
            }

            if (beta.getRequiresGrad()) {
                Tensor betaGrad = reduceLeadingKeepDims(outGrad, normalizedRank).reshape(beta.getShape());
                NormalizationSupport.accumulateGradient(beta, betaGrad);
            }
        });
        return out;
    }

    public static Tensor rmsNorm(
            Tensor input,
            Tensor gamma,
            double epsilon
    ) {
        NormalizationSupport.requireFloating(input, "rmsNorm input");
        NormalizationSupport.requireFloating(gamma, "rmsNorm gamma");
        NormalizationSupport.requirePositiveEpsilon(epsilon, "rmsNorm");
        NormalizationSupport.validateMatchingTailParameter(input, gamma, "rmsNorm gamma");

        int normalizedRank = gamma.getShapeUnsafe().length;
        DataType outputType = TensorDataTypeUtil.promote(input.getDataType(), gamma.getDataType());
        Tensor out = TensorPrimitiveBuilder.binary(
                input,
                gamma,
                input.getShape().clone(),
                new rmsNorm(normalizedRank, epsilon),
                "rmsNorm",
                outputType
        );
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }

            Tensor epsilonTensor = Tensor.scalar(epsilon, outputType);
            Tensor meanSquares = reduceTrailingKeepDims(input.pow(2.0), normalizedRank);
            Tensor invRms = meanSquares.add(epsilonTensor).sqrt().inv();

            if (input.getRequiresGrad()) {
                Tensor weighted = outGrad.mul(gamma);
                Tensor dotMean = reduceTrailingKeepDims(weighted.mul(input), normalizedRank);
                Tensor invRmsCubed = invRms.mul(invRms).mul(invRms);
                Tensor inputGrad = weighted.mul(invRms).sub(input.mul(dotMean).mul(invRmsCubed));
                NormalizationSupport.accumulateGradient(input, inputGrad);
            }

            if (gamma.getRequiresGrad()) {
                Tensor gammaGrad = outGrad.mul(input).mul(invRms);
                gammaGrad = reduceLeadingKeepDims(gammaGrad, normalizedRank).reshape(gamma.getShape());
                NormalizationSupport.accumulateGradient(gamma, gammaGrad);
            }
        });
        return out;
    }

    private static Tensor normalizeWithStats(
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
        Tensor epsilonTensor = Tensor.scalar(epsilon, TensorDataTypeUtil.promote(input.getDataType(), gamma.getDataType()));

        return input
                .sub(meanView)
                .div(varianceView.add(epsilonTensor).sqrt())
                .mul(gammaView)
                .add(betaView);
    }

    private static Tensor reduceAllButOne(Tensor input, int preservedAxis) {
        Tensor reduced = input;
        for (int axis = 0; axis < input.getShapeUnsafe().length; axis++) {
            if (axis == preservedAxis) {
                continue;
            }
            reduced = reduced.mean(axis, true);
        }
        return reduced.reshape(new int[]{input.getShapeUnsafe()[preservedAxis]});
    }

    private static Tensor reshapeChannelParameter(int[] inputShape, int channelDimension, Tensor parameter) {
        int[] broadcastShape = new int[inputShape.length];
        java.util.Arrays.fill(broadcastShape, 1);
        broadcastShape[channelDimension] = inputShape[channelDimension];
        return parameter.reshape(broadcastShape);
    }

    private static Tensor reduceTrailingKeepDims(Tensor input, int normalizedRank) {
        int inputRank = input.getShapeUnsafe().length;
        int startAxis = inputRank - normalizedRank;
        Tensor reduced = input;
        for (int axis = startAxis; axis < inputRank; axis++) {
            reduced = reduced.mean(axis, true);
        }
        return reduced;
    }

    private static Tensor reduceLeadingKeepDims(Tensor input, int trailingRank) {
        int reductions = input.getShapeUnsafe().length - trailingRank;
        Tensor reduced = input;
        for (int i = 0; i < reductions; i++) {
            reduced = reduced.sum(i, true);
        }
        return reduced;
    }
}
