package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormInferenceAttrs;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, stateless batch-normalization inference expressions.
 *
 * <p>For channel {@code c}, the semantic formula is
 * {@code ((input - runningMean[c]) / sqrt(runningVariance[c] + epsilon)) * scale[c] + bias[c]}.
 * The helper reads no values, so negative running variance is retained for ordinary floating
 * square-root behavior rather than rejected or repaired. BFLOAT16 and FLOAT32 results compute in
 * FLOAT32; FLOAT64 results compute in FLOAT64. Empty and special-value behavior, permitted
 * reassociation, and rounding are semantic constraints for later conforming execution.</p>
 *
 * <p>Every success delegates exactly once to the factory and creates one fresh, storage-free,
 * layout-unresolved output with the exact input Shape, combined gradient eligibility, ordered
 * five-input provenance, and output index zero. It creates no training state, saved statistic,
 * mutation, compiler behavior, backend choice, or runtime state.</p>
 */
final class TensorBatchNormInferenceExpressions {
    private TensorBatchNormInferenceExpressions() {
    }

    /**
     * Creates one explicit five-input batch-normalization inference expression.
     *
     * @param input non-null rank-at-least-two floating data input retained at producer position 0
     * @param channelAxis positive or negative logical channel axis normalized against input Shape
     * @param scale non-null floating rank-one per-channel scale retained at position 1
     * @param bias non-null floating rank-one per-channel bias retained at position 2
     * @param runningMean non-null floating rank-one estimated mean retained at position 3
     * @param runningVariance non-null floating rank-one estimated variance retained at position 4
     * @param epsilon non-null exact finite positive floating value matching promoted result type
     * @return fresh unlabeled, storage-free, layout-unresolved result with exact input Shape,
     *     promoted type, combined gradient eligibility, and output index zero; never {@code null}
     * @throws NullPointerException if a Tensor argument or epsilon is null, checked in logical
     *     input order and then epsilon
     * @throws IllegalArgumentException if an input is non-floating, input rank is less than two,
     *     a per-channel operand is not rank one or is statically incompatible with the input
     *     channel extent, or epsilon is invalid or not exactly result-typed
     * @throws IndexOutOfBoundsException if {@code channelAxis} is invalid for input Shape
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    static Tensor apply(
            Tensor input,
            int channelAxis,
            Tensor scale,
            Tensor bias,
            Tensor runningMean,
            Tensor runningVariance,
            ScalarValue epsilon) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(scale, "scale");
        Objects.requireNonNull(bias, "bias");
        Objects.requireNonNull(runningMean, "runningMean");
        Objects.requireNonNull(runningVariance, "runningVariance");
        Objects.requireNonNull(epsilon, "epsilon");

        DataType inputType = requireFloating(input, "input");
        DataType scaleType = requireFloating(scale, "scale");
        DataType biasType = requireFloating(bias, "bias");
        DataType meanType = requireFloating(runningMean, "runningMean");
        DataType varianceType = requireFloating(runningVariance, "runningVariance");

        Shape inputShape = input.descriptor().shape();
        if (inputShape.rank() < 2) {
            throw new IllegalArgumentException(
                    "batchNormInference input rank must be at least 2, but was "
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

        DataType resultType = DataTypePromotion.promoteFloating(inputType, scaleType);
        resultType = DataTypePromotion.promoteFloating(resultType, biasType);
        resultType = DataTypePromotion.promoteFloating(resultType, meanType);
        resultType = DataTypePromotion.promoteFloating(resultType, varianceType);
        if (epsilon.dataType() != resultType) {
            throw new IllegalArgumentException(
                    "batchNormInference epsilon data type must match result data type: epsilon="
                            + epsilon.dataType() + ", result=" + resultType);
        }

        BatchNormInferenceAttrs attrs = new BatchNormInferenceAttrs(normalizedAxis, epsilon);
        boolean requiresGrad = input.descriptor().requiresGrad()
                || scale.descriptor().requiresGrad()
                || bias.descriptor().requiresGrad()
                || runningMean.descriptor().requiresGrad()
                || runningVariance.descriptor().requiresGrad();
        TensorDescriptor descriptor = new TensorDescriptor(
                resultType, inputShape, Optional.empty(), requiresGrad);
        Operation operation = new Operation(BatchNormKind.BATCH_NORM_INFERENCE, attrs);
        return TensorFactory.createDerived(
                descriptor,
                Optional.empty(),
                operation,
                List.of(input, scale, bias, runningMean, runningVariance));
    }

    private static DataType requireFloating(Tensor tensor, String role) {
        DataType dataType = tensor.descriptor().dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "batchNormInference " + role
                            + " must have a floating data type, but was " + dataType);
        }
        return dataType;
    }

    private static void requireVector(Tensor tensor, String role) {
        int rank = tensor.descriptor().shape().rank();
        if (rank != 1) {
            throw new IllegalArgumentException(
                    "batchNormInference " + role + " rank must be one, but was " + rank);
        }
    }

    private static void requireCompatibleChannel(
            Dimension inputDimension, Tensor tensor, String role) {
        Dimension operandDimension = tensor.descriptor().shape().dimension(0);
        if (!inputDimension.equals(operandDimension)
                && inputDimension instanceof StaticDimension
                && operandDimension instanceof StaticDimension) {
            throw new IllegalArgumentException(
                    "batchNormInference " + role + " channel dimension mismatch: input="
                            + inputDimension + ", " + role + "=" + operandDimension);
        }
    }
}
