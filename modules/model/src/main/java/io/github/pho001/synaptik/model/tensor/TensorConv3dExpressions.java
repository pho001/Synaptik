package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free grouped NCDHW cross-correlation expressions.
 *
 * <p>This package-private stateless boundary owns convolution-specific floating promotion, rank,
 * kernel, grouped-channel, bias, spatial Shape, descriptor, and provenance construction. Exact
 * input descriptors and attributes retain relations that cannot be proved until later compiler
 * validation or symbolic binding. It reads no values, creates no general constraints or gradient
 * rules, captures no graph, chooses no algorithm or backend, allocates no storage, and does not
 * execute.</p>
 */
final class TensorConv3dExpressions {
    /** Prevents instantiation because convolution expression construction is stateless. */
    private TensorConv3dExpressions() {
    }

    /**
     * Creates an unbiased expression with ordered inputs {@code [input, weight]}.
     *
     * @param input non-null rank-five NCDHW floating input retained as input zero
     * @param weight non-null rank-five floating weight retained as input one
     * @param attrs non-null exact convolution geometry retained by the operation
     * @return a non-null fresh canonical output with promoted type, derived NCDHW Shape,
     *     unresolved layout, gradient-request OR, and output-index-zero provenance
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if floating, rank, kernel, channel, or geometry validation
     *     fails
     * @throws ArithmeticException if checked convolution geometry overflows {@code long}
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    static Tensor apply(Tensor input, Tensor weight, Conv3dAttrs attrs) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(weight, "weight");
        Objects.requireNonNull(attrs, "attrs");
        return build(input, weight, null, attrs, List.of(input, weight));
    }

    /**
     * Creates a biased expression with ordered inputs {@code [input, weight, bias]}.
     *
     * @param input non-null rank-five NCDHW floating input retained as input zero
     * @param weight non-null rank-five floating weight retained as input one
     * @param bias non-null rank-one floating output-channel bias retained as input two
     * @param attrs non-null exact convolution geometry retained by the operation
     * @return a non-null fresh canonical output with promoted type, derived NCDHW Shape,
     *     unresolved layout, three-way gradient-request OR, and output-index-zero provenance
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if floating, rank, kernel, channel, bias, or geometry
     *     validation fails
     * @throws ArithmeticException if checked convolution geometry overflows {@code long}
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    static Tensor apply(Tensor input, Tensor weight, Tensor bias, Conv3dAttrs attrs) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(weight, "weight");
        Objects.requireNonNull(bias, "bias");
        Objects.requireNonNull(attrs, "attrs");
        return build(input, weight, bias, attrs, List.of(input, weight, bias));
    }

    private static Tensor build(
            Tensor input,
            Tensor weight,
            Tensor bias,
            Conv3dAttrs attrs,
            List<Tensor> inputs) {
        DataType inputType = requireFloating(input, "input");
        DataType weightType = requireFloating(weight, "weight");
        DataType biasType = bias == null ? null : requireFloating(bias, "bias");
        DataType resultType = DataTypePromotion.promoteFloating(inputType, weightType);
        if (biasType != null) {
            resultType = DataTypePromotion.promoteFloating(resultType, biasType);
        }

        Shape inputShape = input.descriptor().shape();
        Shape weightShape = weight.descriptor().shape();
        requireRank(inputShape, 5, "input");
        requireRank(weightShape, 5, "weight");
        Shape biasShape = bias == null ? null : bias.descriptor().shape();
        if (biasShape != null) {
            requireRank(biasShape, 1, "bias");
        }

        long kernelDepth = requirePositiveStaticKernel(weightShape.dimension(2), "depth");
        long kernelHeight = requirePositiveStaticKernel(weightShape.dimension(3), "height");
        long kernelWidth = requirePositiveStaticKernel(weightShape.dimension(4), "width");
        Dimension inputChannels = inputShape.dimension(1);
        Dimension outputChannels = weightShape.dimension(0);
        Dimension weightChannelsPerGroup = weightShape.dimension(1);
        validateDivisible(inputChannels, attrs.groups(), "input");
        validateDivisible(outputChannels, attrs.groups(), "output");
        if (inputChannels instanceof StaticDimension inputStatic
                && weightChannelsPerGroup instanceof StaticDimension weightStatic
                && Math.multiplyExact(weightStatic.size(), attrs.groups()) != inputStatic.size()) {
            throw new IllegalArgumentException(
                    "conv3d weight channels per group do not match input channels: weight="
                            + weightChannelsPerGroup + ", groups=" + attrs.groups()
                            + ", input=" + inputChannels);
        }
        if (biasShape != null) {
            Dimension biasLength = biasShape.dimension(0);
            if (biasLength instanceof StaticDimension biasStatic
                    && outputChannels instanceof StaticDimension outputStatic
                    && biasStatic.size() != outputStatic.size()) {
                throw new IllegalArgumentException(
                        "conv3d bias length must match output channels: bias=" + biasLength
                                + ", output=" + outputChannels);
            }
        }

        Dimension outputDepth = outputDimension(
                inputShape.dimension(2), kernelDepth, attrs.paddingDepth(),
                attrs.dilationDepth(), attrs.strideDepth(), "depth");
        Dimension outputHeight = outputDimension(
                inputShape.dimension(3), kernelHeight, attrs.paddingHeight(),
                attrs.dilationHeight(), attrs.strideHeight(), "height");
        Dimension outputWidth = outputDimension(
                inputShape.dimension(4), kernelWidth, attrs.paddingWidth(),
                attrs.dilationWidth(), attrs.strideWidth(), "width");
        Shape outputShape = Shape.ofDimensions(
                inputShape.dimension(0), outputChannels,
                outputDepth, outputHeight, outputWidth);
        boolean requiresGrad = input.descriptor().requiresGrad()
                || weight.descriptor().requiresGrad()
                || bias != null && bias.descriptor().requiresGrad();
        TensorDescriptor descriptor = new TensorDescriptor(
                resultType, outputShape, Optional.empty(), requiresGrad);
        Operation operation = new Operation(Conv3dKind.CONV3D, attrs);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, inputs);
    }

    private static DataType requireFloating(Tensor tensor, String role) {
        DataType dataType = tensor.descriptor().dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "conv3d " + role + " must have a floating data type, but was " + dataType);
        }
        return dataType;
    }

    private static void requireRank(Shape shape, int expected, String role) {
        if (shape.rank() != expected) {
            throw new IllegalArgumentException(
                    "conv3d " + role + " rank must be " + expected + ": " + shape.rank());
        }
    }

    private static long requirePositiveStaticKernel(Dimension dimension, String axis) {
        if (!(dimension instanceof StaticDimension staticDimension)) {
            throw new IllegalArgumentException(
                    "conv3d kernel " + axis + " must be static: " + dimension);
        }
        if (staticDimension.size() == 0) {
            throw new IllegalArgumentException(
                    "conv3d kernel " + axis + " must be positive: " + dimension);
        }
        return staticDimension.size();
    }

    private static void validateDivisible(Dimension channels, long groups, String role) {
        if (channels instanceof StaticDimension staticChannels
                && staticChannels.size() % groups != 0) {
            throw new IllegalArgumentException(
                    "conv3d " + role + " channels must be divisible by groups: channels="
                            + channels + ", groups=" + groups);
        }
    }

    private static Dimension outputDimension(
            Dimension input,
            long kernel,
            long padding,
            long dilation,
            long stride,
            String axis) {
        long effectiveKernel = Math.addExact(
                Math.multiplyExact(dilation, Math.subtractExact(kernel, 1)), 1);
        long doublePadding = Math.multiplyExact(2, padding);
        if (input instanceof StaticDimension staticInput) {
            long paddedInput = Math.addExact(staticInput.size(), doublePadding);
            long numerator = Math.subtractExact(paddedInput, effectiveKernel);
            if (numerator < 0) {
                throw new IllegalArgumentException(
                        "conv3d effective kernel does not fit padded " + axis + ": input="
                                + input + ", effectiveKernel=" + effectiveKernel
                                + ", padding=" + padding);
            }
            return new StaticDimension(Math.addExact(numerator / stride, 1));
        }
        long offset = Math.subtractExact(doublePadding, effectiveKernel);
        Dimension numerator = DimensionExpressions.addConstant(input, offset);
        return DimensionExpressions.addConstant(
                DimensionExpressions.floorDivide(numerator, stride), 1);
    }
}
