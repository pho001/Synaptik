package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;

/**
 * Builds fixed-count average-pool and exact first-winner max-pool input cotangents.
 *
 * <p>Both formulas use the forward window geometry and public overlap-accumulating fold. Max
 * pooling reconstructs eligibility from the exact original input and same-occurrence output,
 * distinguishes padding from real negative infinity, and selects the first logical kernel
 * candidate without introducing hidden indices.</p>
 */
final class PoolingGradientRules {
    /**
     * Prevents construction of this stateless formula owner.
     */
    private PoolingGradientRules() {}

    /**
     * Builds the sole selected pooling input cotangent.
     *
     * @param producer non-null exact original pooling occurrence
     * @param gradient non-null accumulated cotangent for its sole output
     * @param constants non-null request-local exact floating logical-splat owner
     * @return a new one-position array containing the normalized input cotangent
     */
    static Tensor[] apply(
            TensorProducer producer,
            Tensor gradient,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor input = producer.inputs().getFirst();
        Tensor result = producer.operation().kind() == Pool2dKind.AVERAGE_POOL2D
                ? average(producer, gradient, constants)
                : maximum(producer, gradient, constants);
        return new Tensor[] {normalize(result, input)};
    }

    /**
     * Routes an average-pool cotangent through every logical kernel position using the fixed
     * kernel-element divisor.
     *
     * @param producer non-null exact original average-pool occurrence
     * @param gradient non-null accumulated output cotangent
     * @param constants non-null request-local exact floating logical-splat owner
     * @return a new overlap-accumulating input-cotangent expression
     */
    private static Tensor average(
            TensorProducer producer,
            Tensor gradient,
            FirstOrderAutograd.DerivativeConstants constants) {
        AveragePool2dAttrs attrs =
                (AveragePool2dAttrs) producer.operation().attrs();
        Window2dAttrs window = window(attrs);
        Tensor input = producer.inputs().getFirst();
        Shape output = gradient.descriptor().shape();
        Dimension positions =
                DimensionExpressions.multiply(output.dimension(2), output.dimension(3));
        Dimension channels = input.descriptor().shape().dimension(1);
        long kernelElements = Math.multiplyExact(attrs.kernelHeight(), attrs.kernelWidth());

        Tensor divisor = constants.oneBase(input.descriptor().dataType())
                .expand(Shape.of(attrs.kernelHeight(), attrs.kernelWidth()))
                .sum();
        Tensor perPosition = gradient
                .reshape(Shape.ofDimensions(
                        output.dimension(0), channels, positions))
                .div(divisor);
        Tensor columns = perPosition
                .expandDims(2)
                .expand(Shape.ofDimensions(
                        output.dimension(0),
                        channels,
                        new StaticDimension(kernelElements),
                        positions))
                .reshape(Shape.ofDimensions(
                        output.dimension(0),
                        DimensionExpressions.multiply(channels, kernelElements),
                        positions));
        return columns.fold2d(input.descriptor().shape(), window);
    }

    /**
     * Reconstructs the exact first eligible maximum and routes each output cotangent through that
     * one logical kernel position.
     *
     * @param producer non-null exact original maximum-pool occurrence
     * @param gradient non-null accumulated output cotangent
     * @param constants non-null request-local exact floating logical-splat owner
     * @return a new overlap-accumulating input-cotangent expression
     */
    private static Tensor maximum(
            TensorProducer producer,
            Tensor gradient,
            FirstOrderAutograd.DerivativeConstants constants) {
        MaxPool2dAttrs attrs = (MaxPool2dAttrs) producer.operation().attrs();
        Window2dAttrs window = window(attrs);
        Tensor input = producer.inputs().getFirst();
        Tensor output = producer.output(0);
        Shape outputShape = output.descriptor().shape();
        Dimension batch = outputShape.dimension(0);
        Dimension channels = outputShape.dimension(1);
        Dimension positions =
                DimensionExpressions.multiply(outputShape.dimension(2), outputShape.dimension(3));
        long kernelElements = Math.multiplyExact(attrs.kernelHeight(), attrs.kernelWidth());
        Tensor candidates = input
                .unfold2d(window, negativeInfinity(input.descriptor().dataType()))
                .reshape(Shape.ofDimensions(
                        batch,
                        channels,
                        new StaticDimension(kernelElements),
                        positions))
                .permute(0, 1, 3, 2);
        Tensor inBounds = constants.oneLike(input)
                .unfold2d(window, positiveZero(input.descriptor().dataType()))
                .reshape(Shape.ofDimensions(
                        batch,
                        channels,
                        new StaticDimension(kernelElements),
                        positions))
                .permute(0, 1, 3, 2)
                .notEqualTo(constants.zeroLike(candidates));
        Tensor alignedOutput = output
                .reshape(Shape.ofDimensions(batch, channels, positions))
                .expandDims(3);

        Tensor bothNaN = candidates.isNaN().logicalAnd(alignedOutput.isNaN());
        Tensor numericallyEqual = candidates.equalTo(alignedOutput);
        Tensor bothZero = candidates.equalTo(constants.zeroLike(candidates))
                .logicalAnd(alignedOutput.equalTo(constants.zeroLike(alignedOutput)));
        Tensor reciprocalSignsEqual = constants.oneLike(candidates).div(candidates)
                .equalTo(constants.oneLike(alignedOutput).div(alignedOutput));
        Tensor signedEqual = numericallyEqual.logicalAnd(
                bothZero.logicalNot().logicalOr(reciprocalSignsEqual));
        Tensor eligible = inBounds.logicalAnd(bothNaN.logicalOr(signedEqual));

        Tensor candidateValues = Tensor.where(
                eligible, constants.oneLike(candidates), constants.zeroLike(candidates));
        Tensor first = candidateValues
                .argMax(-1, false, ArgExtremaTiePolicy.FIRST_INDEX)
                .oneHot(kernelElements);
        Tensor selected = first.logicalAnd(eligible);
        Tensor routed = Tensor.where(
                selected,
                gradient.reshape(Shape.ofDimensions(batch, channels, positions)).expandDims(3),
                constants.zeroLike(candidates));
        Tensor columns = routed
                .permute(0, 1, 3, 2)
                .reshape(Shape.ofDimensions(
                        batch,
                        DimensionExpressions.multiply(channels, kernelElements),
                        positions));
        return columns.fold2d(input.descriptor().shape(), window);
    }

    /**
     * Converts average-pool geometry to the shared public window attribute.
     *
     * @param attrs non-null average-pool geometry
     * @return a new immutable window attribute with the same geometry
     */
    private static Window2dAttrs window(AveragePool2dAttrs attrs) {
        return new Window2dAttrs(
                attrs.kernelHeight(), attrs.kernelWidth(),
                attrs.strideHeight(), attrs.strideWidth(),
                attrs.paddingHeight(), attrs.paddingWidth(),
                attrs.dilationHeight(), attrs.dilationWidth(),
                attrs.ceilMode());
    }

    /**
     * Converts maximum-pool geometry to the shared public window attribute.
     *
     * @param attrs non-null maximum-pool geometry
     * @return a new immutable window attribute with the same geometry
     */
    private static Window2dAttrs window(MaxPool2dAttrs attrs) {
        return new Window2dAttrs(
                attrs.kernelHeight(), attrs.kernelWidth(),
                attrs.strideHeight(), attrs.strideWidth(),
                attrs.paddingHeight(), attrs.paddingWidth(),
                attrs.dilationHeight(), attrs.dilationWidth(),
                attrs.ceilMode());
    }

    /**
     * Returns exact typed positive-zero scalar metadata for padding in the in-bounds probe.
     *
     * @param dataType non-null BFLOAT16, FLOAT32, or FLOAT64 type
     * @return exact positive-zero scalar metadata of the requested floating type
     * @throws IllegalArgumentException if {@code dataType} is not floating
     */
    private static ScalarValue positiveZero(DataType dataType) {
        return switch (dataType) {
            case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0x0000);
            case FLOAT32 -> ScalarValue.float32(0.0f);
            case FLOAT64 -> ScalarValue.float64(0.0d);
            case INT32, INT64, BOOL ->
                    throw new IllegalArgumentException("pooling requires floating data");
        };
    }

    /**
     * Returns exact typed negative-infinity scalar metadata for maximum-pool padding.
     *
     * @param dataType non-null BFLOAT16, FLOAT32, or FLOAT64 type
     * @return exact negative-infinity scalar metadata of the requested floating type
     * @throws IllegalArgumentException if {@code dataType} is not floating
     */
    private static ScalarValue negativeInfinity(DataType dataType) {
        return switch (dataType) {
            case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0xFF80);
            case FLOAT32 -> ScalarValue.float32(Float.NEGATIVE_INFINITY);
            case FLOAT64 -> ScalarValue.float64(Double.NEGATIVE_INFINITY);
            case INT32, INT64, BOOL ->
                    throw new IllegalArgumentException("pooling requires floating data");
        };
    }

    /**
     * Restores one pooling cotangent to the original input's exact Shape and data type.
     *
     * @param gradient non-null selected cotangent
     * @param input non-null original pooling input whose descriptor is restored
     * @return a non-null ordinary Tensor expression with the input's exact descriptor
     */
    private static Tensor normalize(Tensor gradient, Tensor input) {
        Tensor result = gradient.descriptor().shape().equals(input.descriptor().shape())
                ? gradient
                : gradient.sumToShape(input.descriptor().shape());
        return result.descriptor().dataType() == input.descriptor().dataType()
                ? result
                : result.cast(input.descriptor().dataType());
    }
}
