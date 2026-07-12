package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormKind;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormTrainingAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated pure batch-normalization training expressions.
 *
 * <p>One occurrence consumes ordered
 * {@code [input, scale, bias, runningMean, runningVariance]} and produces ordered
 * {@code [output, nextRunningMean, nextRunningVariance, savedBatchMean,
 * savedInverseStandardDeviation]}. It reduces
 * every non-channel axis, uses biased batch variance for normalization, correction-one variance
 * for the explicit running-variance transition, and treats momentum as the new-batch weight.
 * Epsilon appears only inside {@code 1 / sqrt(biasedVariance + epsilon)}.</p>
 *
 * <p>The helper reads no values and owns no running state. It creates five fresh indexed Tensor
 * wrappers under one producer, returns slots zero through two, and discards the local wrappers for
 * saved slots three and four while their descriptors remain producer metadata for later
 * compiler-owned capture and lifetime decisions.</p>
 */
final class TensorBatchNormTrainingExpressions {
    private TensorBatchNormTrainingExpressions() {
    }

    /**
     * Creates one explicit training normalization and running-statistic transition.
     *
     * @param input non-null rank-at-least-two floating input at producer position zero
     * @param channelAxis positive or negative logical channel axis normalized exactly once
     * @param scale non-null floating rank-one per-channel scale at position one
     * @param bias non-null floating rank-one per-channel bias at position two
     * @param runningMean non-null floating rank-one old running mean at position three
     * @param runningVariance non-null floating rank-one old running variance at position four
     * @param momentum non-null exact finite new-batch weight in {@code [0, 1]} matching result type
     * @param epsilon non-null exact finite positive stabilizer matching result type
     * @return three-component public result selecting producer slots zero through two; never
     *     {@code null}
     * @throws NullPointerException if an input or scalar is null, checked in declaration order
     * @throws IllegalArgumentException if a Tensor is non-floating, input rank is below two, a
     *     per-channel operand is not rank one or is statically channel-incompatible, a statically
     *     positive channel has reduction count below two, or a scalar is invalid or not exactly
     *     result-typed
     * @throws IndexOutOfBoundsException if {@code channelAxis} is invalid for the input Shape
     * @throws IllegalStateException if tensor identifier space is exhausted; identifiers already
     *     allocated for earlier output positions remain consumed
     */
    static BatchNormTrainingResult apply(
            Tensor input,
            int channelAxis,
            Tensor scale,
            Tensor bias,
            Tensor runningMean,
            Tensor runningVariance,
            ScalarValue momentum,
            ScalarValue epsilon) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(scale, "scale");
        Objects.requireNonNull(bias, "bias");
        Objects.requireNonNull(runningMean, "runningMean");
        Objects.requireNonNull(runningVariance, "runningVariance");
        Objects.requireNonNull(momentum, "momentum");
        Objects.requireNonNull(epsilon, "epsilon");

        DataType inputType = requireFloating(input, "input");
        DataType scaleType = requireFloating(scale, "scale");
        DataType biasType = requireFloating(bias, "bias");
        DataType meanType = requireFloating(runningMean, "runningMean");
        DataType varianceType = requireFloating(runningVariance, "runningVariance");

        Shape inputShape = input.descriptor().shape();
        if (inputShape.rank() < 2) {
            throw new IllegalArgumentException(
                    "batchNormTraining input rank must be at least 2, but was "
                            + inputShape.rank());
        }
        int normalizedAxis = inputShape.normalizeAxis(channelAxis);
        requireVector(scale, "scale");
        requireVector(bias, "bias");
        requireVector(runningMean, "runningMean");
        requireVector(runningVariance, "runningVariance");

        Dimension channelDimension = inputShape.dimension(normalizedAxis);
        requireCompatibleChannel(channelDimension, scale, "scale");
        requireCompatibleChannel(channelDimension, bias, "bias");
        requireCompatibleChannel(channelDimension, runningMean, "runningMean");
        requireCompatibleChannel(channelDimension, runningVariance, "runningVariance");
        validateStaticReductionDomain(inputShape, normalizedAxis, channelDimension);

        DataType resultType = DataTypePromotion.promoteFloating(inputType, scaleType);
        resultType = DataTypePromotion.promoteFloating(resultType, biasType);
        resultType = DataTypePromotion.promoteFloating(resultType, meanType);
        resultType = DataTypePromotion.promoteFloating(resultType, varianceType);
        if (momentum.dataType() != resultType) {
            throw new IllegalArgumentException(
                    "batchNormTraining momentum data type must match result data type: momentum="
                            + momentum.dataType() + ", result=" + resultType);
        }
        if (epsilon.dataType() != resultType) {
            throw new IllegalArgumentException(
                    "batchNormTraining epsilon data type must match result data type: epsilon="
                            + epsilon.dataType() + ", result=" + resultType);
        }

        BatchNormTrainingAttrs attrs = new BatchNormTrainingAttrs(
                normalizedAxis, momentum, epsilon);
        Shape statisticShape = Shape.ofDimensions(channelDimension);
        boolean inputGrad = input.descriptor().requiresGrad();
        TensorDescriptor outputDescriptor = descriptor(
                resultType,
                inputShape,
                inputGrad || scale.descriptor().requiresGrad() || bias.descriptor().requiresGrad());
        TensorDescriptor nextMeanDescriptor = descriptor(
                resultType,
                statisticShape,
                inputGrad || runningMean.descriptor().requiresGrad());
        TensorDescriptor nextVarianceDescriptor = descriptor(
                resultType,
                statisticShape,
                inputGrad || runningVariance.descriptor().requiresGrad());
        TensorDescriptor savedMeanDescriptor = descriptor(resultType, statisticShape, inputGrad);
        TensorDescriptor savedInvStdDescriptor = descriptor(resultType, statisticShape, inputGrad);
        Operation operation = new Operation(BatchNormKind.BATCH_NORM_TRAINING, attrs);
        List<Tensor> outputs = TensorFactory.createDerivedOutputs(
                operation,
                List.of(input, scale, bias, runningMean, runningVariance),
                List.of(
                        outputDescriptor,
                        nextMeanDescriptor,
                        nextVarianceDescriptor,
                        savedMeanDescriptor,
                        savedInvStdDescriptor));
        Tensor output = outputs.get(0);
        Tensor nextRunningMean = outputs.get(1);
        Tensor nextRunningVariance = outputs.get(2);
        Tensor savedBatchMean = outputs.get(3);
        Tensor savedInverseStandardDeviation = outputs.get(4);
        return new BatchNormTrainingResult(output, nextRunningMean, nextRunningVariance);
    }

    private static DataType requireFloating(Tensor tensor, String role) {
        DataType dataType = tensor.descriptor().dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "batchNormTraining " + role
                            + " must have a floating data type, but was " + dataType);
        }
        return dataType;
    }

    private static void requireVector(Tensor tensor, String role) {
        int rank = tensor.descriptor().shape().rank();
        if (rank != 1) {
            throw new IllegalArgumentException(
                    "batchNormTraining " + role + " rank must be one, but was " + rank);
        }
    }

    private static void requireCompatibleChannel(
            Dimension inputDimension, Tensor tensor, String role) {
        Dimension operandDimension = tensor.descriptor().shape().dimension(0);
        if (!inputDimension.equals(operandDimension)
                && inputDimension instanceof StaticDimension
                && operandDimension instanceof StaticDimension) {
            throw new IllegalArgumentException(
                    "batchNormTraining " + role + " channel dimension mismatch: input="
                            + inputDimension + ", " + role + "=" + operandDimension);
        }
    }

    private static void validateStaticReductionDomain(
            Shape inputShape, int channelAxis, Dimension channelDimension) {
        if (!(channelDimension instanceof StaticDimension staticChannel)
                || staticChannel.size() == 0) {
            return;
        }
        for (int axis = 0; axis < inputShape.rank(); axis++) {
            if (axis == channelAxis) {
                continue;
            }
            Dimension dimension = inputShape.dimensions().get(axis);
            if (dimension instanceof StaticDimension staticDimension
                    && staticDimension.size() == 0) {
                rejectReductionDomain(0);
            }
        }
        long count = 1;
        for (int axis = 0; axis < inputShape.rank(); axis++) {
            if (axis == channelAxis) {
                continue;
            }
            Dimension dimension = inputShape.dimensions().get(axis);
            if (!(dimension instanceof StaticDimension staticDimension)) {
                return;
            }
            try {
                count = Math.multiplyExact(count, staticDimension.size());
            } catch (ArithmeticException overflow) {
                return;
            }
        }
        if (count < 2) {
            rejectReductionDomain(count);
        }
    }

    private static void rejectReductionDomain(long count) {
        throw new IllegalArgumentException(
                "batchNormTraining reduction domain count " + count
                        + " must be at least 2 when channel extent is non-zero");
    }

    private static TensorDescriptor descriptor(
            DataType dataType, Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad);
    }
}
