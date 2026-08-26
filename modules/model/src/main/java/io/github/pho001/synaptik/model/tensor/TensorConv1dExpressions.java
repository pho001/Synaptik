package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.convolution.Conv1dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.Objects;

/**
 * Constructs NCW grouped cross-correlation as an exact four-producer Tensor composition.
 *
 * <p>This package-private field-free owner validates the rank-one contract before allocating an
 * intermediate identity. It then expands input and weight at axis two, delegates convolution to
 * the existing NCHW Conv2d expression with a singleton height, and squeezes that height. It owns
 * no operation kind, graph capture, gradient rule, storage, lowering, or execution behavior.</p>
 */
final class TensorConv1dExpressions {
    /** Prevents instantiation because construction is stateless. */
    private TensorConv1dExpressions() {
    }

    /**
     * Creates an unbiased NCW composition.
     *
     * @param input non-null rank-three floating input retained through the composition
     * @param weight non-null rank-three floating kernel retained through the composition
     * @param attrs non-null rank-one geometry translated to fresh Conv2d attributes
     * @return non-null fresh canonical squeeze output of the four-producer composition
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if a floating, rank, kernel, grouping, or geometry rule
     *     fails before composition
     * @throws ArithmeticException if checked geometry overflows {@code long}
     * @throws IllegalStateException if Tensor identifier space is exhausted; an exhaustion after
     *     composition starts may leave earlier intermediate identifiers consumed
     */
    static Tensor apply(Tensor input, Tensor weight, Conv1dAttrs attrs) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(weight, "weight");
        Objects.requireNonNull(attrs, "attrs");
        Conv2dAttrs mappedAttrs = validate(input, weight, null, attrs);
        Tensor expandedInput = input.expandDims(2);
        Tensor expandedWeight = weight.expandDims(2);
        return expandedInput.conv2d(expandedWeight, mappedAttrs).squeeze(2);
    }

    /**
     * Creates a biased NCW composition while retaining bias directly as Conv2d input two.
     *
     * @param input non-null rank-three floating input retained through the composition
     * @param weight non-null rank-three floating kernel retained through the composition
     * @param bias non-null rank-one floating output-channel bias retained without expansion
     * @param attrs non-null rank-one geometry translated to fresh Conv2d attributes
     * @return non-null fresh canonical squeeze output of the biased four-producer composition
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if a floating, rank, kernel, grouping, bias, or geometry
     *     rule fails before composition
     * @throws ArithmeticException if checked geometry overflows {@code long}
     * @throws IllegalStateException if Tensor identifier space is exhausted; an exhaustion after
     *     composition starts may leave earlier intermediate identifiers consumed
     */
    static Tensor apply(Tensor input, Tensor weight, Tensor bias, Conv1dAttrs attrs) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(weight, "weight");
        Objects.requireNonNull(bias, "bias");
        Objects.requireNonNull(attrs, "attrs");
        Conv2dAttrs mappedAttrs = validate(input, weight, bias, attrs);
        Tensor expandedInput = input.expandDims(2);
        Tensor expandedWeight = weight.expandDims(2);
        return expandedInput.conv2d(expandedWeight, bias, mappedAttrs).squeeze(2);
    }

    /**
     * Validates the complete rank-one contract and creates the attributes retained by Conv2d.
     *
     * @param input non-null candidate NCW input
     * @param weight non-null candidate rank-three kernel
     * @param bias optional candidate output-channel bias; may be {@code null}
     * @param attrs non-null validated rank-one geometry
     * @return a new non-null Conv2d attributes value with singleton-height geometry
     * @throws IllegalArgumentException if a floating, rank, kernel, grouping, bias, or geometry
     *     rule fails
     * @throws ArithmeticException if checked geometry or grouped-channel arithmetic overflows
     *     {@code long}
     */
    private static Conv2dAttrs validate(
            Tensor input, Tensor weight, Tensor bias, Conv1dAttrs attrs) {
        DataType inputType = requireFloating(input, "input");
        DataType weightType = requireFloating(weight, "weight");
        DataType biasType = bias == null ? null : requireFloating(bias, "bias");
        DataType resultType = DataTypePromotion.promoteFloating(inputType, weightType);
        if (biasType != null) {
            DataTypePromotion.promoteFloating(resultType, biasType);
        }

        Shape inputShape = input.descriptor().shape();
        Shape weightShape = weight.descriptor().shape();
        requireRank(inputShape, 3, "input");
        requireRank(weightShape, 3, "weight");
        Shape biasShape = bias == null ? null : bias.descriptor().shape();
        if (biasShape != null) {
            requireRank(biasShape, 1, "bias");
        }

        long kernelWidth = requirePositiveStaticKernel(weightShape.dimension(2));
        Dimension inputChannels = inputShape.dimension(1);
        Dimension outputChannels = weightShape.dimension(0);
        Dimension weightChannelsPerGroup = weightShape.dimension(1);
        validateDivisible(inputChannels, attrs.groups(), "input");
        validateDivisible(outputChannels, attrs.groups(), "output");
        if (inputChannels instanceof StaticDimension inputStatic
                && weightChannelsPerGroup instanceof StaticDimension weightStatic
                && Math.multiplyExact(weightStatic.size(), attrs.groups()) != inputStatic.size()) {
            throw new IllegalArgumentException(
                    "conv1d weight channels per group do not match input channels: weight="
                            + weightChannelsPerGroup + ", groups=" + attrs.groups()
                            + ", input=" + inputChannels);
        }
        if (biasShape != null) {
            Dimension biasLength = biasShape.dimension(0);
            if (biasLength instanceof StaticDimension biasStatic
                    && outputChannels instanceof StaticDimension outputStatic
                    && biasStatic.size() != outputStatic.size()) {
                throw new IllegalArgumentException(
                        "conv1d bias length must match output channels: bias=" + biasLength
                                + ", output=" + outputChannels);
            }
        }

        outputWidth(inputShape.dimension(2), kernelWidth, attrs);
        return new Conv2dAttrs(
                1, attrs.stride(), 0, attrs.padding(), 1, attrs.dilation(), attrs.groups());
    }

    /**
     * Requires one Tensor role to use a floating data type.
     *
     * @param tensor non-null Tensor whose descriptor is inspected without reading values
     * @param role non-null diagnostic role name
     * @return the Tensor's non-null floating data type
     * @throws IllegalArgumentException if the data type is not floating
     */
    private static DataType requireFloating(Tensor tensor, String role) {
        DataType dataType = tensor.descriptor().dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "conv1d " + role + " must have a floating data type, but was " + dataType);
        }
        return dataType;
    }

    /**
     * Requires one Shape to have the role-specific rank.
     *
     * @param shape non-null Shape to inspect
     * @param expected required non-negative rank
     * @param role non-null diagnostic role name
     * @throws IllegalArgumentException if the Shape rank differs from {@code expected}
     */
    private static void requireRank(Shape shape, int expected, String role) {
        if (shape.rank() != expected) {
            throw new IllegalArgumentException(
                    "conv1d " + role + " rank must be " + expected + ": " + shape.rank());
        }
    }

    /**
     * Resolves the required positive static kernel width.
     *
     * @param dimension non-null kernel-width Dimension
     * @return the positive static width
     * @throws IllegalArgumentException if the width is unresolved or zero
     */
    private static long requirePositiveStaticKernel(Dimension dimension) {
        if (!(dimension instanceof StaticDimension staticDimension)) {
            throw new IllegalArgumentException("conv1d kernel width must be static: " + dimension);
        }
        if (staticDimension.size() == 0) {
            throw new IllegalArgumentException("conv1d kernel width must be positive: " + dimension);
        }
        return staticDimension.size();
    }

    /**
     * Checks a channel divisibility relation when its extent is statically known.
     *
     * @param channels non-null channel Dimension; unresolved values defer the relation
     * @param groups positive group count
     * @param role non-null diagnostic role name
     * @throws IllegalArgumentException if a static channel count is not divisible by
     *     {@code groups}
     */
    private static void validateDivisible(Dimension channels, long groups, String role) {
        if (channels instanceof StaticDimension staticChannels
                && staticChannels.size() % groups != 0) {
            throw new IllegalArgumentException(
                    "conv1d " + role + " channels must be divisible by groups: channels="
                            + channels + ", groups=" + groups);
        }
    }

    /**
     * Derives the static or canonical symbolic output-width Dimension for validation.
     *
     * @param input non-null input-width Dimension
     * @param kernel positive static kernel width
     * @param attrs non-null rank-one geometry
     * @return a non-null static or symbolic output-width Dimension
     * @throws IllegalArgumentException if static padded width cannot contain the effective kernel
     * @throws ArithmeticException if checked signed-{@code long} geometry overflows
     */
    private static Dimension outputWidth(Dimension input, long kernel, Conv1dAttrs attrs) {
        long effectiveKernel = Math.addExact(
                Math.multiplyExact(attrs.dilation(), Math.subtractExact(kernel, 1)), 1);
        long doublePadding = Math.multiplyExact(2, attrs.padding());
        if (input instanceof StaticDimension staticInput) {
            long paddedInput = Math.addExact(staticInput.size(), doublePadding);
            long numerator = Math.subtractExact(paddedInput, effectiveKernel);
            if (numerator < 0) {
                throw new IllegalArgumentException(
                        "conv1d effective kernel does not fit padded width: input=" + input
                                + ", effectiveKernel=" + effectiveKernel
                                + ", padding=" + attrs.padding());
            }
            return new StaticDimension(Math.addExact(numerator / attrs.stride(), 1));
        }
        long offset = Math.subtractExact(doublePadding, effectiveKernel);
        Dimension numerator = DimensionExpressions.addConstant(input, offset);
        return DimensionExpressions.addConstant(
                DimensionExpressions.floorDivide(numerator, attrs.stride()), 1);
    }
}
