package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free grouped NCHW convolution expressions.
 *
 * <p>This package-private stateless boundary owns convolution-specific floating promotion, rank,
 * kernel, grouped-channel, bias, spatial Shape, descriptor, and provenance construction. Exact
 * input descriptors and attributes retain relations that cannot be proved until later compiler
 * validation or symbolic binding. It reads no values, creates no general constraints or gradient
 * rules, captures no graph, chooses no algorithm or backend, allocates no storage, and does not
 * execute.</p>
 */
final class TensorConv2dExpressions {
    /** Prevents instantiation because convolution expression construction is stateless. */
    private TensorConv2dExpressions() {
    }

    /**
     * Creates an unbiased expression with ordered inputs {@code [input, weight]}.
     *
     * @param input non-null rank-four NCHW floating input retained as input zero
     * @param weight non-null rank-four floating weight retained as input one
     * @param attrs non-null exact convolution geometry retained by the operation
     * @return a non-null fresh output with promoted type, derived NCHW Shape, unresolved layout,
     *     gradient-request OR, and output-index-zero provenance
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if floating, rank, kernel, channel, or geometry validation
     *     fails
     * @throws ArithmeticException if checked convolution geometry overflows {@code long}
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    static Tensor apply(Tensor input, Tensor weight, Conv2dAttrs attrs) {
        return applyChecked(input, weight, attrs, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input non-null rank-four NCHW floating input retained as input zero
     * @param weight non-null rank-four floating weight retained as input one
     * @param attrs non-null exact convolution geometry retained by the operation
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if floating, rank, kernel, channel, or geometry validation
     *     fails
     * @throws ArithmeticException if checked convolution geometry overflows {@code long}
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    static Tensor applyChecked(Tensor input, Tensor weight, Conv2dAttrs attrs,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(weight, "weight");
        Objects.requireNonNull(attrs, "attrs");
        return buildChecked(input, weight, null, attrs, List.of(input, weight), descriptorCheck);
    }

    /**
     * Creates a biased expression with ordered inputs {@code [input, weight, bias]}.
     *
     * @param input non-null rank-four NCHW floating input retained as input zero
     * @param weight non-null rank-four floating weight retained as input one
     * @param bias non-null rank-one floating output-channel bias retained as input two
     * @param attrs non-null exact convolution geometry retained by the operation
     * @return a non-null fresh output with promoted type, derived NCHW Shape, unresolved layout,
     *     three-way gradient-request OR, and output-index-zero provenance
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if floating, rank, kernel, channel, bias, or geometry
     *     validation fails
     * @throws ArithmeticException if checked convolution geometry overflows {@code long}
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    static Tensor apply(Tensor input, Tensor weight, Tensor bias, Conv2dAttrs attrs) {
        return applyChecked(input, weight, bias, attrs, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input non-null rank-four NCHW floating input retained as input zero
     * @param weight non-null rank-four floating weight retained as input one
     * @param bias non-null rank-one floating output-channel bias retained as input two
     * @param attrs non-null exact convolution geometry retained by the operation
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if floating, rank, kernel, channel, bias, or geometry
     *     validation fails
     * @throws ArithmeticException if checked convolution geometry overflows {@code long}
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    static Tensor applyChecked(Tensor input, Tensor weight, Tensor bias, Conv2dAttrs attrs,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(weight, "weight");
        Objects.requireNonNull(bias, "bias");
        Objects.requireNonNull(attrs, "attrs");
        return buildChecked(input, weight, bias, attrs, List.of(input, weight, bias), descriptorCheck);
    }

    /**
     * Delivers the canonical checked convolution descriptor once without constructing a Tensor.
     * @param operation convolution operation whose exact attributes are retained
     * @param inputs ordered input, weight, and optional bias descriptors
     * @param outputCount output cardinality validated by the metadata facade
     * @param sink synchronous, unretained receiver of canonical metadata
     */
    static void describeChecked(Operation operation, List<TensorDescriptor> inputs, int outputCount,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> sink) {
        Conv2dAttrs attrs = (Conv2dAttrs) operation.attrs();
        TensorDescriptor descriptor = describeInputs(inputs.get(0), inputs.get(1),
                inputs.size() == 3 ? inputs.get(2) : null, attrs);
        sink.accept(new Operation(Conv2dKind.CONV2D, attrs), List.of(descriptor));
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input non-null floating NCHW/NCDHW input; channel count must match grouped weights
     * @param weight non-null floating OIHW/OIDHW weights with valid grouped channel geometry
     * @param bias nullable floating rank-one output-channel bias; null selects the bias-free form
     * @param attrs non-null stride, padding, dilation and group geometry validated by ordinary convolution
     * @param inputs exact ordered input/weight and optional bias references used only for ordinary construction
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if ordinary semantic validation or the supplied check fails
     */
    private static Tensor buildChecked(
            Tensor input,
            Tensor weight,
            Tensor bias,
            Conv2dAttrs attrs,
            List<Tensor> inputs,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        TensorDescriptor descriptor = describeInputs(input.descriptor(), weight.descriptor(),
                bias == null ? null : bias.descriptor(), attrs);
        Operation operation = new Operation(Conv2dKind.CONV2D, attrs);
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, inputs);
    }

    private static TensorDescriptor describeInputs(
            TensorDescriptor input, TensorDescriptor weight, TensorDescriptor bias, Conv2dAttrs attrs) {
        DataType inputType = requireFloating(input, "input");
        DataType weightType = requireFloating(weight, "weight");
        DataType biasType = bias == null ? null : requireFloating(bias, "bias");
        DataType resultType = DataTypePromotion.promoteFloating(inputType, weightType);
        if (biasType != null) {
            resultType = DataTypePromotion.promoteFloating(resultType, biasType);
        }

        Shape inputShape = input.shape();
        Shape weightShape = weight.shape();
        requireRank(inputShape, 4, "input");
        requireRank(weightShape, 4, "weight");
        Shape biasShape = bias == null ? null : bias.shape();
        if (biasShape != null) {
            requireRank(biasShape, 1, "bias");
        }

        long kernelHeight = requirePositiveStaticKernel(weightShape.dimension(2), "height");
        long kernelWidth = requirePositiveStaticKernel(weightShape.dimension(3), "width");
        Dimension inputChannels = inputShape.dimension(1);
        Dimension outputChannels = weightShape.dimension(0);
        Dimension weightChannelsPerGroup = weightShape.dimension(1);
        validateDivisible(inputChannels, attrs.groups(), "input");
        validateDivisible(outputChannels, attrs.groups(), "output");
        if (inputChannels instanceof StaticDimension inputStatic
                && weightChannelsPerGroup instanceof StaticDimension weightStatic
                && Math.multiplyExact(weightStatic.size(), attrs.groups()) != inputStatic.size()) {
            throw new IllegalArgumentException(
                    "conv2d weight channels per group do not match input channels: weight="
                            + weightChannelsPerGroup + ", groups=" + attrs.groups()
                            + ", input=" + inputChannels);
        }
        if (biasShape != null) {
            Dimension biasLength = biasShape.dimension(0);
            if (biasLength instanceof StaticDimension biasStatic
                    && outputChannels instanceof StaticDimension outputStatic
                    && biasStatic.size() != outputStatic.size()) {
                throw new IllegalArgumentException(
                        "conv2d bias length must match output channels: bias=" + biasLength
                                + ", output=" + outputChannels);
            }
        }

        Dimension outputHeight = outputDimension(
                inputShape.dimension(2), kernelHeight, attrs.paddingHeight(),
                attrs.dilationHeight(), attrs.strideHeight(), "height");
        Dimension outputWidth = outputDimension(
                inputShape.dimension(3), kernelWidth, attrs.paddingWidth(),
                attrs.dilationWidth(), attrs.strideWidth(), "width");
        Shape outputShape = Shape.ofDimensions(
                inputShape.dimension(0), outputChannels, outputHeight, outputWidth);
        boolean requiresGrad = input.requiresGrad()
                || weight.requiresGrad()
                || bias != null && bias.requiresGrad();
        return new TensorDescriptor(
                resultType, outputShape, Optional.empty(), requiresGrad);
    }

    private static DataType requireFloating(TensorDescriptor tensor, String role) {
        DataType dataType = tensor.dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "conv2d " + role + " must have a floating data type, but was " + dataType);
        }
        return dataType;
    }

    private static void requireRank(Shape shape, int expected, String role) {
        if (shape.rank() != expected) {
            throw new IllegalArgumentException(
                    "conv2d " + role + " rank must be " + expected + ": " + shape.rank());
        }
    }

    private static long requirePositiveStaticKernel(Dimension dimension, String axis) {
        if (!(dimension instanceof StaticDimension staticDimension)) {
            throw new IllegalArgumentException(
                    "conv2d kernel " + axis + " must be static: " + dimension);
        }
        if (staticDimension.size() == 0) {
            throw new IllegalArgumentException(
                    "conv2d kernel " + axis + " must be positive: " + dimension);
        }
        return staticDimension.size();
    }

    private static void validateDivisible(Dimension channels, long groups, String role) {
        if (channels instanceof StaticDimension staticChannels
                && staticChannels.size() % groups != 0) {
            throw new IllegalArgumentException(
                    "conv2d " + role + " channels must be divisible by groups: channels="
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
                        "conv2d effective kernel does not fit padded " + axis + ": input="
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
