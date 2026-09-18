package io.github.pho001.synaptik.nn.layers;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.initialization.ParameterInitializers;
import io.github.pho001.synaptik.nn.module.Parameter;
import io.github.pho001.synaptik.nn.module.UnaryTensorModule;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Owns one channels-first two-dimensional cross-correlation weight and optional bias.
 *
 * <p>The first compatible NCHW forward call infers only the positive static input-channel extent,
 * initializes and atomically publishes {@code weight[, bias]}, then delegates to the ordinary
 * Model {@link Tensor#conv2d(Tensor, Conv2dAttrs)} operation. Strict state loading may bind the
 * complete group without initialization. Later calls may vary batch and spatial Dimensions but
 * must retain the bound channel schema. Weight Shape is
 * {@code [outChannels, inChannels/groups, kernelHeight, kernelWidth]}; optional bias Shape is
 * {@code [outChannels]}.</p>
 *
 * <p>Random policies use one fresh seeded {@code L64X128MixRandom} source per binding attempt;
 * zero and one policies create no source, and bias is always exact typed zero. A failed attempt
 * publishes no parameter wrapper and may be retried. Publication is synchronized only for this
 * layer's first binding; other inherited module mutation and traversal still require caller
 * coordination. The layer constructs expressions and performs no execution.</p>
 */
public final class Conv2d extends UnaryTensorModule {
    private final long outChannels;
    private final long kernelHeight;
    private final long kernelWidth;
    private final long groups;
    private final boolean biasConfigured;
    private final DataType dataType;
    private final ParameterInitialization weightInitialization;
    private final long seed;
    private final Conv2dAttrs attrs;

    /**
     * Creates an NCHW layer whose complete parameter group is bound by first forward or strict load.
     *
     * @param outChannels positive output-channel count divisible by {@code groups}
     * @param kernelHeight positive kernel height
     * @param kernelWidth positive kernel width
     * @param strideHeight positive output-height stride
     * @param strideWidth positive output-width stride
     * @param paddingHeight non-negative symmetric height padding per side
     * @param paddingWidth non-negative symmetric width padding per side
     * @param dilationHeight positive kernel-height dilation
     * @param dilationWidth positive kernel-width dilation
     * @param groups positive channel-group count
     * @param bias whether the layer owns a typed-zero bias
     * @param dataType non-null exact floating input and parameter type
     * @param weightInitialization non-null closed policy for the complete weight
     * @param seed any seed for a fresh {@code L64X128MixRandom} source on each random-policy
     *     attempt; accepted but unused by zero and one policies
     * @throws NullPointerException if {@code dataType} or {@code weightInitialization} is null
     * @throws IllegalArgumentException if an explicit schema value is invalid
     * @throws ArithmeticException if checked kernel, fan, padding, or geometry arithmetic overflows
     */
    public Conv2d(
            long outChannels, long kernelHeight, long kernelWidth,
            long strideHeight, long strideWidth, long paddingHeight, long paddingWidth,
            long dilationHeight, long dilationWidth, long groups, boolean bias,
            DataType dataType, ParameterInitialization weightInitialization, long seed) {
        requirePositive(outChannels, "outChannels");
        requirePositive(kernelHeight, "kernelHeight");
        requirePositive(kernelWidth, "kernelWidth");
        requirePositive(strideHeight, "strideHeight");
        requirePositive(strideWidth, "strideWidth");
        requirePositive(dilationHeight, "dilationHeight");
        requirePositive(dilationWidth, "dilationWidth");
        requireNonNegative(paddingHeight, "paddingHeight");
        requireNonNegative(paddingWidth, "paddingWidth");
        requirePositive(groups, "groups");
        if (outChannels % groups != 0L) {
            throw new IllegalArgumentException("outChannels must be divisible by groups");
        }
        this.dataType = Objects.requireNonNull(dataType, "dataType");
        this.weightInitialization = Objects.requireNonNull(weightInitialization, "weightInitialization");
        if (!this.dataType.isFloating()) {
            throw new IllegalArgumentException("convolution data type must be floating: " + dataType);
        }
        long volume = ConvolutionParameterInitialization.kernelVolume(kernelHeight, kernelWidth);
        ConvolutionParameterInitialization.fanOut(outChannels, groups, volume);
        ConvolutionParameterInitialization.effectiveKernel(kernelHeight, dilationHeight);
        ConvolutionParameterInitialization.effectiveKernel(kernelWidth, dilationWidth);
        ConvolutionParameterInitialization.doubledPadding(paddingHeight);
        ConvolutionParameterInitialization.doubledPadding(paddingWidth);
        this.outChannels = outChannels;
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.groups = groups;
        this.biasConfigured = bias;
        this.seed = seed;
        this.attrs = new Conv2dAttrs(strideHeight, strideWidth, paddingHeight, paddingWidth,
                dilationHeight, dilationWidth, groups);
        reserveParameter("weight", this::validateReservedWeight);
        if (bias) reserveParameter("bias", this::validateReservedBias);
    }

    /**
     * Returns the stable bound weight wrapper.
     *
     * @return the non-null stable
     *     {@code [outChannels, inChannels/groups, kernelHeight, kernelWidth]} parameter wrapper
     *     owned by this layer
     * @throws IllegalStateException if neither forward nor strict load has bound the state group
     */
    public Parameter weight() {
        return boundParameter("weight");
    }

    /**
     * Returns the configured bias wrapper.
     *
     * @return empty when bias is disabled, otherwise the non-null stable bound
     *     {@code [outChannels]} parameter wrapper owned by this layer
     * @throws IllegalStateException if bias is enabled and the state group remains unbound
     */
    public Optional<Parameter> bias() {
        return biasConfigured ? Optional.of(boundParameter("bias")) : Optional.empty();
    }

    /**
     * Builds one first-class Model Conv2d expression after binding compatible NCHW state.
     *
     * @param input non-null rank-four NCHW Tensor with configured type and positive static
     *     channels; retained in result provenance and not mutated
     * @return the fresh non-null storage-free Tensor returned by the matching
     *     {@code Tensor.conv2d} overload
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if input schema or geometry is incompatible
     * @throws ArithmeticException if checked Shape or geometry arithmetic overflows
     * @throws RuntimeException if a configured random initializer fails while sampling
     * @throws IllegalStateException if Tensor identifier space is exhausted or publication fails
     * @throws OutOfMemoryError if parameter or expression allocation fails
     */
    @Override
    public Tensor forward(Tensor input) {
        Tensor suppliedInput = Objects.requireNonNull(input, "input");
        Shape weightShape = validateInput(suppliedInput);
        if (!parameterReservationsBound()) {
            synchronized (this) {
                if (!parameterReservationsBound()) {
                    weightShape = validateInput(suppliedInput);
                    initialize(weightShape);
                    if (!parameterReservationsBound()) {
                        throw new IllegalStateException(
                                "Conv2d automatic parameter initialization did not publish complete state");
                    }
                }
            }
        }
        Tensor currentWeight = weight().value();
        validateBoundChannels(inputChannels(weightShape), currentWeight);
        return biasConfigured
                ? suppliedInput.conv2d(currentWeight, bias().orElseThrow().value(), attrs)
                : suppliedInput.conv2d(currentWeight, attrs);
    }

    private Shape validateInput(Tensor input) {
        if (input.descriptor().dataType() != dataType) {
            throw new IllegalArgumentException("convolution input data type must equal configured data type");
        }
        Shape shape = input.descriptor().shape();
        if (shape.rank() != 4) throw new IllegalArgumentException("Conv2d input must have rank four: " + shape.rank());
        long inChannels = positiveStatic(shape.dimension(1), "Conv2d input channels");
        if (inChannels % groups != 0L) throw new IllegalArgumentException("input channels must be divisible by groups");
        Shape weightShape = Shape.of(outChannels, inChannels / groups, kernelHeight, kernelWidth);
        Shape biasShape = Shape.of(outChannels);
        ConvolutionParameterInitialization.validateJavaArrayCount(weightShape, "Conv2d weight");
        if (biasConfigured) ConvolutionParameterInitialization.validateJavaArrayCount(biasShape, "Conv2d bias");
        ConvolutionParameterInitialization.validateSpatialGeometry(shape.dimension(2), kernelHeight,
                attrs.strideHeight(), attrs.paddingHeight(), attrs.dilationHeight(), "height");
        ConvolutionParameterInitialization.validateSpatialGeometry(shape.dimension(3), kernelWidth,
                attrs.strideWidth(), attrs.paddingWidth(), attrs.dilationWidth(), "width");
        long volume = ConvolutionParameterInitialization.kernelVolume(kernelHeight, kernelWidth);
        ConvolutionParameterInitialization.fanIn(inChannels, groups, volume);
        ConvolutionParameterInitialization.fanOut(outChannels, groups, volume);
        return weightShape;
    }

    private void initialize(Shape weightShape) {
        long inChannels = inputChannels(weightShape);
        long volume = ConvolutionParameterInitialization.kernelVolume(kernelHeight, kernelWidth);
        Tensor initializedWeight = ConvolutionParameterInitialization.initialize(
                weightShape, dataType, weightInitialization, seed,
                ConvolutionParameterInitialization.fanIn(inChannels, groups, volume),
                ConvolutionParameterInitialization.fanOut(outChannels, groups, volume));
        if (biasConfigured) {
            bindReservedParameters(List.of(initializedWeight,
                    ParameterInitializers.zeros(Shape.of(outChannels), dataType)));
        } else bindReservedParameters(List.of(initializedWeight));
    }

    private void validateReservedWeight(Tensor weight) {
        validateParameterCommon(weight, "Conv2d weight");
        Shape shape = weight.descriptor().shape();
        if (shape.rank() != 4 || !shape.isFullyStatic()) throw new IllegalArgumentException("Conv2d weight must have fully static rank-four Shape");
        requireExtent(shape, 0, outChannels, "outChannels");
        requireExtent(shape, 2, kernelHeight, "kernelHeight");
        requireExtent(shape, 3, kernelWidth, "kernelWidth");
        long inChannels = Math.multiplyExact(positiveStatic(shape.dimension(1), "Conv2d weight channels per group"), groups);
        long volume = ConvolutionParameterInitialization.kernelVolume(kernelHeight, kernelWidth);
        ConvolutionParameterInitialization.fanIn(inChannels, groups, volume);
        ConvolutionParameterInitialization.fanOut(outChannels, groups, volume);
        ConvolutionParameterInitialization.validateJavaArrayCount(shape, "Conv2d weight");
    }

    private void validateReservedBias(Tensor bias) {
        validateParameterCommon(bias, "Conv2d bias");
        Shape shape = bias.descriptor().shape();
        if (!shape.equals(Shape.of(outChannels))) throw new IllegalArgumentException("Conv2d bias must have Shape [outChannels]");
        ConvolutionParameterInitialization.validateJavaArrayCount(shape, "Conv2d bias");
    }

    private void validateBoundChannels(long inChannels, Tensor weight) {
        long expected = Math.multiplyExact(positiveStatic(weight.descriptor().shape().dimension(1), "Conv2d weight channels per group"), groups);
        if (inChannels != expected) throw new IllegalArgumentException("Conv2d input channels must match bound inChannels: expected=" + expected + ", actual=" + inChannels);
    }

    private long inputChannels(Shape weightShape) {
        return Math.multiplyExact(positiveStatic(weightShape.dimension(1),
                "Conv2d weight channels per group"), groups);
    }

    private void validateParameterCommon(Tensor tensor, String role) {
        if (tensor.descriptor().dataType() != dataType) throw new IllegalArgumentException(role + " data type must equal configured data type");
        if (!tensor.descriptor().requiresGrad()) throw new IllegalArgumentException(role + " must have requiresGrad == true");
    }

    private static long positiveStatic(Dimension dimension, String role) {
        if (!(dimension instanceof StaticDimension value)) throw new IllegalArgumentException(role + " must be static: " + dimension);
        if (value.size() <= 0L) throw new IllegalArgumentException(role + " must be positive: " + value.size());
        return value.size();
    }

    private static void requireExtent(Shape shape, int axis, long expected, String role) {
        long actual = positiveStatic(shape.dimension(axis), role);
        if (actual != expected) throw new IllegalArgumentException(role + " mismatch: expected=" + expected + ", actual=" + actual);
    }

    private static void requirePositive(long value, String name) {
        if (value <= 0L) throw new IllegalArgumentException(name + " must be positive: " + value);
    }

    private static void requireNonNegative(long value, String name) {
        if (value < 0L) throw new IllegalArgumentException(name + " must be non-negative: " + value);
    }

}
