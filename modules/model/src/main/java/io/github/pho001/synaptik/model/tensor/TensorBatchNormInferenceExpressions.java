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
 * square-root behavior rather than rejected or repaired. BFLOAT16, FLOAT16, and FLOAT32 results
 * compute in FLOAT32; FLOAT64 results compute in FLOAT64. FLOAT16 result metadata uses an exact
 * FLOAT32 epsilon. Empty and special-value behavior, permitted reassociation, and rounding are
 * semantic constraints for later conforming execution.</p>
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
     * @param epsilon non-null finite positive scalar matching the result type, except FLOAT16 uses FLOAT32
     * @return fresh unlabeled, storage-free, layout-unresolved result with exact input Shape,
     *     promoted type, combined gradient eligibility, and output index zero; never {@code null}
     * @throws NullPointerException if a Tensor argument or epsilon is null, checked in logical
     *     input order and then epsilon
     * @throws IllegalArgumentException if an input is non-floating, input rank is less than two,
     *     a per-channel operand is not rank one or is statically incompatible with the input
     *     channel extent, or epsilon does not match the result arithmetic type
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
        return applyChecked(input, channelAxis, scale, bias, runningMean, runningVariance, epsilon, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input non-null rank-at-least-two floating data input retained at producer position 0
     * @param channelAxis positive or negative logical channel axis normalized against input Shape
     * @param scale non-null floating rank-one per-channel scale retained at position 1
     * @param bias non-null floating rank-one per-channel bias retained at position 2
     * @param runningMean non-null floating rank-one estimated mean retained at position 3
     * @param runningVariance non-null floating rank-one estimated variance retained at position 4
     * @param epsilon non-null finite positive scalar matching the result type, except FLOAT16 uses FLOAT32
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if a Tensor argument or epsilon is null, checked in logical
     *     input order and then epsilon
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if an input is non-floating, input rank is less than two,
     *     a per-channel operand is not rank one or is statically incompatible with the input
     *     channel extent, or epsilon does not match the result arithmetic type
     * @throws IndexOutOfBoundsException if {@code channelAxis} is invalid for input Shape
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    static Tensor applyChecked(
            Tensor input,
            int channelAxis,
            Tensor scale,
            Tensor bias,
            Tensor runningMean,
            Tensor runningVariance,
            ScalarValue epsilon,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(scale, "scale");
        Objects.requireNonNull(bias, "bias");
        Objects.requireNonNull(runningMean, "runningMean");
        Objects.requireNonNull(runningVariance, "runningVariance");
        Objects.requireNonNull(epsilon, "epsilon");
        return applyCore(input.descriptor(), channelAxis, scale.descriptor(), bias.descriptor(),
                runningMean.descriptor(), runningVariance.descriptor(), epsilon,
                input, scale, bias, runningMean, runningVariance, descriptorCheck);
    }

    private static Tensor applyCore(
            TensorDescriptor inputDescriptor, int channelAxis,
            TensorDescriptor scaleDescriptor, TensorDescriptor biasDescriptor,
            TensorDescriptor meanDescriptor, TensorDescriptor varianceDescriptor,
            ScalarValue epsilon, Tensor input, Tensor scale, Tensor bias,
            Tensor runningMean, Tensor runningVariance,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {

        DataType inputType = requireFloating(inputDescriptor, "input");
        DataType scaleType = requireFloating(scaleDescriptor, "scale");
        DataType biasType = requireFloating(biasDescriptor, "bias");
        DataType meanType = requireFloating(meanDescriptor, "runningMean");
        DataType varianceType = requireFloating(varianceDescriptor, "runningVariance");

        Shape inputShape = inputDescriptor.shape();
        if (inputShape.rank() < 2) {
            throw new IllegalArgumentException(
                    "batchNormInference input rank must be at least 2, but was "
                            + inputShape.rank());
        }
        int normalizedAxis = inputShape.normalizeAxis(channelAxis);
        requireVector(scaleDescriptor, "scale");
        requireVector(biasDescriptor, "bias");
        requireVector(meanDescriptor, "runningMean");
        requireVector(varianceDescriptor, "runningVariance");

        Dimension channelDimension = inputShape.dimension(normalizedAxis);
        requireCompatibleChannel(channelDimension, scaleDescriptor, "scale");
        requireCompatibleChannel(channelDimension, biasDescriptor, "bias");
        requireCompatibleChannel(channelDimension, meanDescriptor, "runningMean");
        requireCompatibleChannel(channelDimension, varianceDescriptor, "runningVariance");

        DataType resultType = DataTypePromotion.promoteFloating(inputType, scaleType);
        resultType = DataTypePromotion.promoteFloating(resultType, biasType);
        resultType = DataTypePromotion.promoteFloating(resultType, meanType);
        resultType = DataTypePromotion.promoteFloating(resultType, varianceType);
        if (epsilon.dataType() != resultType
                && (resultType != DataType.FLOAT16
                        || epsilon.dataType() != DataType.FLOAT32)) {
            throw new IllegalArgumentException(
                    "batchNormInference epsilon data type must match result data type: epsilon="
                            + epsilon.dataType() + ", result=" + resultType);
        }

        BatchNormInferenceAttrs attrs = new BatchNormInferenceAttrs(normalizedAxis, epsilon);
        boolean requiresGrad = inputDescriptor.requiresGrad()
                || scaleDescriptor.requiresGrad()
                || biasDescriptor.requiresGrad()
                || meanDescriptor.requiresGrad()
                || varianceDescriptor.requiresGrad();
        TensorDescriptor descriptor = new TensorDescriptor(
                resultType, inputShape, Optional.empty(), requiresGrad);
        Operation operation = new Operation(BatchNormKind.BATCH_NORM_INFERENCE, attrs);
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(
                descriptor,
                Optional.empty(),
                operation,
                List.of(input, scale, bias, runningMean, runningVariance));
    }

    /**
     * Describes canonical inference metadata through the ordinary construction guards.
     * The sink is invoked synchronously exactly once without creating Tensor identities.
     */
    static void describeChecked(
            Operation operation, List<TensorDescriptor> inputs, int outputCount,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> sink) {
        BatchNormInferenceAttrs attrs = (BatchNormInferenceAttrs) operation.attrs();
        applyCore(inputs.get(0), attrs.channelAxis(), inputs.get(1), inputs.get(2),
                inputs.get(3), inputs.get(4), attrs.epsilon(),
                null, null, null, null, null, sink);
    }

    private static DataType requireFloating(TensorDescriptor descriptor, String role) {
        DataType dataType = descriptor.dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "batchNormInference " + role
                            + " must have a floating data type, but was " + dataType);
        }
        return dataType;
    }

    private static void requireVector(TensorDescriptor descriptor, String role) {
        int rank = descriptor.shape().rank();
        if (rank != 1) {
            throw new IllegalArgumentException(
                    "batchNormInference " + role + " rank must be one, but was " + rank);
        }
    }

    private static void requireCompatibleChannel(
            Dimension inputDimension, TensorDescriptor descriptor, String role) {
        Dimension operandDimension = descriptor.shape().dimension(0);
        if (!inputDimension.equals(operandDimension)
                && inputDimension instanceof StaticDimension
                && operandDimension instanceof StaticDimension) {
            throw new IllegalArgumentException(
                    "batchNormInference " + role + " channel dimension mismatch: input="
                            + inputDimension + ", " + role + "=" + operandDimension);
        }
    }
}
