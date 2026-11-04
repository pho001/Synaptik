package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
import io.github.pho001.synaptik.model.operation.pooling.Pool3dKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;

import java.util.function.Function;

/**
 * Builds fixed-count average-pool and exact first-winner max-pool input cotangents.
 *
 * <p>The rank-specific two- and three-dimensional formulas use the forward window geometry and
 * public overlap-accumulating fold. Max
 * pooling reconstructs eligibility from the exact original input and same-occurrence output,
 * distinguishes padding from real negative infinity, and selects the first logical kernel
 * candidate without introducing hidden indices.</p>
 *
 * <p>Descriptor access preserves original source roles without resolving primal values.
 * Real construction uses Tensor references directly; checked metadata construction retains
 * the same eager operation sites, including temporaries not used by a returned cotangent.</p>
 *
 * @param <V> direct Tensor or checked metadata operand representation
 */
final class PoolingGradientRules<V> {
    private final OrdinaryConstruction<V> ops;

    /**
     * Binds this formula owner to one compile-local construction route.
     * @param ops non-null direct Tensor or checked metadata construction
     */
    PoolingGradientRules(OrdinaryConstruction<V> ops) {
        this.ops = java.util.Objects.requireNonNull(ops);
    }

    /**
     * Builds the sole selected pooling input cotangent.
     *
     * @param producer non-null exact original pooling occurrence
     * @param gradient non-null accumulated cotangent for its sole output
     * @param primal non-null resolver from original primal wrappers to lexical values
     * @return a new one-position array containing the normalized input cotangent
     */
    V[] apply(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            Function<V, V> primal) {
        V input = producer.inputs().getFirst();
        V result;
        if (producer.operation().kind() == Pool2dKind.AVERAGE_POOL2D) {
            result = average(producer, gradient);
        } else if (producer.operation().kind() == Pool2dKind.MAX_POOL2D) {
            result = maximum(producer, gradient, primal);
        } else if (producer.operation().kind() == Pool3dKind.AVERAGE_POOL3D) {
            result = average3d(producer, gradient);
        } else {
            result = maximum3d(producer, gradient, primal);
        }
        return ops.result(normalize(result, input));
    }

    /**
     * Routes an average-pool cotangent through every logical kernel position using the fixed
     * kernel-element divisor.
     *
     * @param producer non-null exact original average-pool occurrence
     * @param gradient non-null accumulated output cotangent
     * @return a new overlap-accumulating input-cotangent expression
     */
    private V average(
            OrdinaryConstruction.Source<V> producer,
            V gradient) {
        AveragePool2dAttrs attrs =
                (AveragePool2dAttrs) producer.operation().attrs();
        Window2dAttrs window = window(attrs);
        V input = producer.inputs().getFirst();
        Shape output = ops.descriptor(gradient).shape();
        Dimension positions =
                DimensionExpressions.multiply(output.dimension(2), output.dimension(3));
        Dimension channels = ops.descriptor(input).shape().dimension(1);
        long kernelElements = Math.multiplyExact(attrs.kernelHeight(), attrs.kernelWidth());

        V divisor = ops.sum(ops.contiguous(ops.expand(ops.oneBase(ops.descriptor(input).dataType()), Shape.of(attrs.kernelHeight(), attrs.kernelWidth()))));
        V perPosition = ops.div(ops.contiguous(ops.reshape(gradient, Shape.ofDimensions(
                        output.dimension(0), channels, positions))), divisor);
        V columns = ops.contiguous(ops.reshape(ops.contiguous(ops.expand(ops.expandDims(perPosition, 2), Shape.ofDimensions(
                        output.dimension(0),
                        channels,
                        new StaticDimension(kernelElements),
                        positions))), Shape.ofDimensions(
                        output.dimension(0),
                        DimensionExpressions.multiply(channels, kernelElements),
                        positions)));
        return ops.fold2d(columns, ops.descriptor(input).shape(), window);
    }

    /**
     * Reconstructs the exact first eligible maximum and routes each output cotangent through that
     * one logical kernel position.
     *
     * @param producer non-null exact original maximum-pool occurrence
     * @param gradient non-null accumulated output cotangent
     * @param primal non-null resolver from original primal wrappers to lexical values
     * @return a new overlap-accumulating input-cotangent expression
     */
    private V maximum(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            Function<V, V> primal) {
        MaxPool2dAttrs attrs = (MaxPool2dAttrs) producer.operation().attrs();
        Window2dAttrs window = window(attrs);
        V input = producer.inputs().getFirst();
        V output = producer.output(0);
        Shape outputShape = ops.descriptor(output).shape();
        Dimension batch = outputShape.dimension(0);
        Dimension channels = outputShape.dimension(1);
        Dimension positions =
                DimensionExpressions.multiply(outputShape.dimension(2), outputShape.dimension(3));
        long kernelElements = Math.multiplyExact(attrs.kernelHeight(), attrs.kernelWidth());
        V candidates = ops.permute(ops.reshape(ops.unfold2d(primal.apply(input), window, maximumPadding(ops.descriptor(input).dataType())), Shape.ofDimensions(
                        batch,
                        channels,
                        new StaticDimension(kernelElements),
                        positions)), 0, 1, 3, 2);
        V inBounds = ops.notEqualTo(ops.permute(ops.reshape(ops.unfold2d(ops.oneLike(input), window, positiveZero(ops.descriptor(input).dataType())), Shape.ofDimensions(
                        batch,
                        channels,
                        new StaticDimension(kernelElements),
                        positions)), 0, 1, 3, 2), ops.zeroLike(candidates));
        V alignedOutput = ops.expandDims(ops.reshape(primal.apply(output), Shape.ofDimensions(batch, channels, positions)), 3);

        V bothNaN = ops.logicalAnd(ops.isNaN(candidates), ops.isNaN(alignedOutput));
        V numericallyEqual = ops.equalTo(candidates, alignedOutput);
        V bothZero = ops.logicalAnd(ops.equalTo(candidates, ops.zeroLike(candidates)), ops.equalTo(alignedOutput, ops.zeroLike(alignedOutput)));
        V reciprocalSignsEqual = reciprocalSignsEqual(candidates, alignedOutput);
        V signedEqual = ops.logicalAnd(numericallyEqual, ops.logicalOr(ops.logicalNot(bothZero), reciprocalSignsEqual));
        V eligible = ops.logicalAnd(inBounds, ops.logicalOr(bothNaN, signedEqual));

        V candidateValues = ops.where(eligible, ops.oneLike(candidates), ops.zeroLike(candidates));
        V first = ops.oneHot(ops.argMax(candidateValues, 3, false, ArgExtremaTiePolicy.FIRST_INDEX), kernelElements);
        V selected = ops.logicalAnd(first, eligible);
        V routed = ops.where(selected, ops.expandDims(ops.reshape(gradient, Shape.ofDimensions(batch, channels, positions)), 3), ops.zeroLike(candidates));
        V columns = ops.reshape(ops.contiguous(ops.permute(routed, 0, 1, 3, 2)), Shape.ofDimensions(
                        batch,
                        DimensionExpressions.multiply(channels, kernelElements),
                        positions));
        return ops.fold2d(columns, ops.descriptor(input).shape(), window);
    }

    /**
     * Routes an average Pool3d cotangent through every logical depth-height-width kernel
     * position using a divisor constructed entirely from same-type Tensor operations.
     *
     * @param producer non-null exact original average Pool3d occurrence
     * @param gradient non-null accumulated output cotangent
     * @return a new volumetric overlap-accumulating input-cotangent expression
     */
    private V average3d(
            OrdinaryConstruction.Source<V> producer,
            V gradient) {
        AveragePool3dAttrs attrs = (AveragePool3dAttrs) producer.operation().attrs();
        Window3dAttrs window = window(attrs);
        V input = producer.inputs().getFirst();
        Shape output = ops.descriptor(gradient).shape();
        Dimension positions = DimensionExpressions.multiply(
                DimensionExpressions.multiply(output.dimension(2), output.dimension(3)),
                output.dimension(4));
        Dimension channels = ops.descriptor(input).shape().dimension(1);
        long kernelElements = kernelElements(
                attrs.kernelDepth(), attrs.kernelHeight(), attrs.kernelWidth());

        V divisor = ops.sum(ops.contiguous(ops.expand(ops.oneBase(ops.descriptor(input).dataType()), Shape.of(attrs.kernelDepth(), attrs.kernelHeight(), attrs.kernelWidth()))));
        V perPosition = ops.div(ops.contiguous(ops.reshape(gradient, Shape.ofDimensions(output.dimension(0), channels, positions))), divisor);
        V columns = ops.contiguous(ops.reshape(ops.contiguous(ops.expand(ops.expandDims(perPosition, 2), Shape.ofDimensions(
                        output.dimension(0),
                        channels,
                        new StaticDimension(kernelElements),
                        positions))), Shape.ofDimensions(
                        output.dimension(0),
                        DimensionExpressions.multiply(channels, kernelElements),
                        positions)));
        return ops.contiguous(ops.fold3d(columns, ops.descriptor(input).shape(), window));
    }

    /**
     * Reconstructs the first exact eligible Pool3d occurrence in depth-height-width order and
     * routes each output cotangent through only that position.
     *
     * @param producer non-null exact original maximum Pool3d occurrence
     * @param gradient non-null accumulated output cotangent
     * @param primal non-null resolver from original primal wrappers to lexical values
     * @return a new volumetric overlap-accumulating input-cotangent expression
     */
    private V maximum3d(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            Function<V, V> primal) {
        MaxPool3dAttrs attrs = (MaxPool3dAttrs) producer.operation().attrs();
        Window3dAttrs window = window(attrs);
        V input = producer.inputs().getFirst();
        V output = producer.output(0);
        Shape outputShape = ops.descriptor(output).shape();
        Dimension batch = outputShape.dimension(0);
        Dimension channels = outputShape.dimension(1);
        Dimension positions = DimensionExpressions.multiply(
                DimensionExpressions.multiply(
                        outputShape.dimension(2), outputShape.dimension(3)),
                outputShape.dimension(4));
        long kernelElements = kernelElements(
                attrs.kernelDepth(), attrs.kernelHeight(), attrs.kernelWidth());
        Shape candidatesShape = Shape.ofDimensions(
                batch, channels, new StaticDimension(kernelElements), positions);
        V candidates = ops.permute(ops.reshape(ops.unfold3d(primal.apply(input), window, maximumPadding(ops.descriptor(input).dataType())), candidatesShape), 0, 1, 3, 2);
        V inBounds = ops.notEqualTo(ops.permute(ops.reshape(ops.unfold3d(ops.oneLike(input), window), candidatesShape), 0, 1, 3, 2),
                ops.zeroLike(candidates));
        V alignedOutput = ops.expandDims(ops.reshape(primal.apply(output), Shape.ofDimensions(batch, channels, positions)), 3);

        V bothNaN = ops.logicalAnd(ops.isNaN(candidates), ops.isNaN(alignedOutput));
        V numericallyEqual = ops.equalTo(candidates, alignedOutput);
        V bothZero = ops.logicalAnd(ops.equalTo(candidates, ops.zeroLike(candidates)), ops.equalTo(alignedOutput, ops.zeroLike(alignedOutput)));
        V reciprocalSignsEqual = reciprocalSignsEqual(candidates, alignedOutput);
        V signedEqual = ops.logicalAnd(numericallyEqual, ops.logicalOr(ops.logicalNot(bothZero), reciprocalSignsEqual));
        V eligible = ops.logicalAnd(inBounds, ops.logicalOr(bothNaN, signedEqual));

        V candidateValues = ops.where(eligible, ops.oneLike(candidates), ops.zeroLike(candidates));
        V first = ops.oneHot(ops.argMax(candidateValues, 3, false, ArgExtremaTiePolicy.FIRST_INDEX), kernelElements);
        V selected = ops.logicalAnd(first, eligible);
        V routed = ops.where(selected, ops.expandDims(ops.reshape(gradient, Shape.ofDimensions(batch, channels, positions)), 3), ops.zeroLike(candidates));
        V columns = ops.reshape(ops.contiguous(ops.permute(routed, 0, 1, 3, 2)), Shape.ofDimensions(
                        batch,
                        DimensionExpressions.multiply(channels, kernelElements),
                        positions));
        return ops.contiguous(ops.fold3d(columns, ops.descriptor(input).shape(), window));
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
     * Converts average Pool3d geometry to the shared immutable volumetric window attribute.
     *
     * @param attrs non-null average Pool3d geometry
     * @return a new immutable window attribute with identical geometry
     */
    private static Window3dAttrs window(AveragePool3dAttrs attrs) {
        return new Window3dAttrs(
                attrs.kernelDepth(), attrs.kernelHeight(), attrs.kernelWidth(),
                attrs.strideDepth(), attrs.strideHeight(), attrs.strideWidth(),
                attrs.paddingDepth(), attrs.paddingHeight(), attrs.paddingWidth(),
                attrs.dilationDepth(), attrs.dilationHeight(), attrs.dilationWidth(),
                attrs.ceilMode());
    }

    /**
     * Converts maximum Pool3d geometry to the shared immutable volumetric window attribute.
     *
     * @param attrs non-null maximum Pool3d geometry
     * @return a new immutable window attribute with identical geometry
     */
    private static Window3dAttrs window(MaxPool3dAttrs attrs) {
        return new Window3dAttrs(
                attrs.kernelDepth(), attrs.kernelHeight(), attrs.kernelWidth(),
                attrs.strideDepth(), attrs.strideHeight(), attrs.strideWidth(),
                attrs.paddingDepth(), attrs.paddingHeight(), attrs.paddingWidth(),
                attrs.dilationDepth(), attrs.dilationHeight(), attrs.dilationWidth(),
                attrs.ceilMode());
    }

    /**
     * Computes the positive structural three-dimensional kernel volume for reshape and one-hot
     * metadata. Average-pool numerical division does not use this host value.
     *
     * @param depth positive logical depth positions
     * @param height positive logical height positions
     * @param width positive logical width positions
     * @return the exact positive checked product
     * @throws ArithmeticException if the product is not representable as a {@code long}
     */
    private static long kernelElements(long depth, long height, long width) {
        return Math.multiplyExact(Math.multiplyExact(depth, height), width);
    }

    /**
     * Distinguishes represented zero signs without narrowing E4M3's reciprocal infinities to NaN.
     * Other formats retain their existing ordinary reciprocal sites.
     * @param candidates exact unfolded input values
     * @param output aligned exact saved maximum values
     * @return exact Boolean equality of reciprocal signs at zero positions
     */
    private V reciprocalSignsEqual(V candidates, V output) {
        if (ops.descriptor(candidates).dataType() == DataType.FLOAT8_E4M3) {
            candidates = ops.cast(candidates, DataType.FLOAT32);
            output = ops.cast(output, DataType.FLOAT32);
        }
        return ops.equalTo(
                ops.div(ops.oneLike(candidates), candidates),
                ops.div(ops.oneLike(output), output));
    }

    /**
     * Returns exact typed positive-zero scalar metadata for padding in the in-bounds probe.
     *
     * @param dataType non-null floating type, including either OFP8 type
     * @return exact positive-zero scalar metadata of the requested floating type
     * @throws IllegalArgumentException if {@code dataType} is not floating
     */
    private static ScalarValue positiveZero(DataType dataType) {
        return switch (dataType) {
            case FLOAT8_E4M3 -> ScalarValue.float8E4m3Bits((byte) 0);
            case FLOAT8_E5M2 -> ScalarValue.float8E5m2Bits((byte) 0);
            case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0x0000);
            case FLOAT16 -> ScalarValue.float16Bits((short) 0x0000);
            case FLOAT32 -> ScalarValue.float32(0.0f);
            case FLOAT64 -> ScalarValue.float64(0.0d);
            case INT32, INT64, BOOL ->
                    throw new IllegalArgumentException("pooling requires floating data");
        };
    }

    /**
     * Returns padding for the masked maximum-pool candidate probe. E4M3 uses its most negative
     * finite value because infinity is not representable; the separate exact in-bounds mask
     * excludes every padding position, including ties with that finite value.
     *
     * @param dataType non-null floating type, including either OFP8 type
     * @return exact negative infinity, or E4M3's most negative finite value
     * @throws IllegalArgumentException if {@code dataType} is not floating
     */
    private static ScalarValue maximumPadding(DataType dataType) {
        return switch (dataType) {
            case FLOAT8_E4M3 -> ScalarValue.float8E4m3Bits((byte) 0xFE);
            case FLOAT8_E5M2 -> ScalarValue.float8E5m2Bits((byte) 0xFC);
            case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0xFF80);
            case FLOAT16 -> ScalarValue.float16Bits((short) 0xFC00);
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
    private V normalize(V gradient, V input) {
        V result = ops.descriptor(gradient).shape().equals(ops.descriptor(input).shape())
                ? gradient
                : ops.sumToShape(gradient, ops.descriptor(input).shape());
        return ops.descriptor(result).dataType() == ops.descriptor(input).dataType()
                ? result
                : ops.cast(result, ops.descriptor(input).dataType());
    }
}
