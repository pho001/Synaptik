package tensor;

import operations.rmsNorm;

final class TensorNormalizationOps {
    private TensorNormalizationOps() {
    }

    static Tensor batchNorm(
            Tensor input,
            Tensor gamma,
            Tensor beta,
            int channelDimension,
            double epsilon
    ) {
        requireFloating(input, "batchNorm input");
        requireFloating(gamma, "batchNorm gamma");
        requireFloating(beta, "batchNorm beta");
        requirePositiveEpsilon(epsilon, "batchNorm");

        int[] inputShape = input.getShapeUnsafe();
        if (inputShape.length < 2) {
            throw new IllegalArgumentException("batchNorm requires at least one non-channel axis.");
        }
        int normalizedChannel = TensorLayoutTransform.normalizeAxis(channelDimension, inputShape.length);
        validateChannelParameter(gamma, inputShape[normalizedChannel], "batchNorm gamma");
        validateChannelParameter(beta, inputShape[normalizedChannel], "batchNorm beta");

        Tensor mean = reduceAllButOne(input, normalizedChannel);
        Tensor meanView = reshapeChannelParameter(inputShape, normalizedChannel, mean);
        Tensor centered = input.sub(meanView);
        Tensor variance = reduceAllButOne(centered.pow(2.0), normalizedChannel);
        Tensor out = normalizeWithStats(input, gamma, beta, mean, variance, normalizedChannel, epsilon);
        out.setLabel("batchNorm");
        return out;
    }

    static Tensor batchNorm(
            Tensor input,
            Tensor gamma,
            Tensor beta,
            Tensor mean,
            Tensor variance,
            int channelDimension,
            double epsilon
    ) {
        requireFloating(input, "batchNorm input");
        requireFloating(gamma, "batchNorm gamma");
        requireFloating(beta, "batchNorm beta");
        requireFloating(mean, "batchNorm mean");
        requireFloating(variance, "batchNorm variance");
        requirePositiveEpsilon(epsilon, "batchNorm");

        int[] inputShape = input.getShapeUnsafe();
        int normalizedChannel = TensorLayoutTransform.normalizeAxis(channelDimension, inputShape.length);
        int channels = inputShape[normalizedChannel];
        validateChannelParameter(gamma, channels, "batchNorm gamma");
        validateChannelParameter(beta, channels, "batchNorm beta");
        validateChannelParameter(mean, channels, "batchNorm mean");
        validateChannelParameter(variance, channels, "batchNorm variance");

        Tensor out = normalizeWithStats(input, gamma, beta, mean, variance, normalizedChannel, epsilon);
        out.setLabel("batchNorm");
        return out;
    }

    static Tensor layerNorm(
            Tensor input,
            Tensor gamma,
            Tensor beta,
            double epsilon
    ) {
        requireFloating(input, "layerNorm input");
        requireFloating(gamma, "layerNorm gamma");
        requireFloating(beta, "layerNorm beta");
        requirePositiveEpsilon(epsilon, "layerNorm");
        validateMatchingTailParameters(input, gamma, beta, "layerNorm");

        int normalizedRank = gamma.getShapeUnsafe().length;
        DataType outputType = TensorDataTypeUtil.promote(TensorDataTypeUtil.promote(input.getDataType(), gamma.getDataType()), beta.getDataType());
        Tensor out = new Tensor(
                input.getShape().clone(),
                java.util.List.of(input, gamma, beta),
                new operations.layerNorm(normalizedRank, epsilon),
                "layerNorm"
        );
        out.setDataType(outputType);
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
                accumulateGradient(input, inputGrad);
            }

            if (gamma.getRequiresGrad()) {
                Tensor gammaGrad = outGrad.mul(xHat);
                gammaGrad = reduceLeadingKeepDims(gammaGrad, normalizedRank).reshape(gamma.getShape());
                accumulateGradient(gamma, gammaGrad);
            }

            if (beta.getRequiresGrad()) {
                Tensor betaGrad = reduceLeadingKeepDims(outGrad, normalizedRank).reshape(beta.getShape());
                accumulateGradient(beta, betaGrad);
            }
        });
        out.setLabel("layerNorm");
        return out;
    }

    static Tensor rmsNorm(
            Tensor input,
            Tensor gamma,
            double epsilon
    ) {
        requireFloating(input, "rmsNorm input");
        requireFloating(gamma, "rmsNorm gamma");
        requirePositiveEpsilon(epsilon, "rmsNorm");
        validateMatchingTailParameter(input, gamma, "rmsNorm gamma");

        int normalizedRank = gamma.getShapeUnsafe().length;
        DataType outputType = TensorDataTypeUtil.promote(input.getDataType(), gamma.getDataType());
        Tensor out = new Tensor(
                input.getShape().clone(),
                java.util.List.of(input, gamma),
                new rmsNorm(normalizedRank, epsilon),
                "rmsNorm"
        );
        out.setDataType(outputType);
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
                accumulateGradient(input, inputGrad);
            }

            if (gamma.getRequiresGrad()) {
                Tensor gammaGrad = outGrad.mul(input).mul(invRms);
                gammaGrad = reduceLeadingKeepDims(gammaGrad, normalizedRank).reshape(gamma.getShape());
                accumulateGradient(gamma, gammaGrad);
            }
        });
        out.setLabel("rmsNorm");
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
            reduced = reduced.sum(0, true);
        }
        return reduced;
    }

    private static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            input.setGradient(gradientDelta);
            return;
        }
        input.setGradient(input.getGradient().add(gradientDelta));
    }

    private static void validateMatchingTailParameters(Tensor input, Tensor gamma, Tensor beta, String opName) {
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

    private static void validateMatchingTailParameter(Tensor input, Tensor parameter, String name) {
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

    private static void validateChannelParameter(Tensor parameter, int channels, String name) {
        int[] shape = parameter.getShapeUnsafe();
        if (shape.length != 1 || shape[0] != channels) {
            throw new IllegalArgumentException(name + " must have shape [channels].");
        }
    }

    private static void requireFloating(Tensor tensor, String name) {
        if (tensor == null) {
            throw new IllegalArgumentException(name + " cannot be null");
        }
        if (tensor.getDataType() == DataType.BOOL || tensor.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException(name + " must use a floating dtype.");
        }
    }

    private static void requirePositiveEpsilon(double epsilon, String opName) {
        if (!(epsilon > 0.0)) {
            throw new IllegalArgumentException(opName + " epsilon must be positive.");
        }
    }
}
