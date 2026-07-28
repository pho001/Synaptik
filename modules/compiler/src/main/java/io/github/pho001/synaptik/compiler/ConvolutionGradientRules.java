package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;

/**
 * Builds grouped NCHW convolution cotangents through public window and matrix operations.
 *
 * <p>Every reshape retains the occurrence's exact symbolic batch, channel, and spatial
 * dimensions. The leading group axis prevents cross-group contraction. Folding accumulates
 * overlap into the original input Shape; no transposed-convolution or backward-only kind is
 * introduced.</p>
 */
final class ConvolutionGradientRules {
    /**
     * Prevents construction of this stateless formula owner.
     */
    private ConvolutionGradientRules() {}

    /**
     * Builds selected input, weight, and optional bias cotangents.
     *
     * @param producer non-null exact original convolution occurrence
     * @param gradient non-null accumulated cotangent for its sole output
     * @param selectedInputs non-null input-position-aligned selected-route flags; observed but
     *     not mutated
     * @return a new input-position-aligned array of cotangents, with {@code null} for unselected
     *     roles
     */
    static Tensor[] apply(
            TensorProducer producer, Tensor gradient, boolean[] selectedInputs) {
        Tensor originalInput = producer.inputs().get(0);
        Tensor originalWeight = producer.inputs().get(1);
        DataType resultType = producer.output(0).descriptor().dataType();
        Tensor input = cast(originalInput, resultType);
        Tensor weight = cast(originalWeight, resultType);
        Conv2dAttrs attrs = (Conv2dAttrs) producer.operation().attrs();
        Window2dAttrs window = window(attrs, originalWeight);

        Shape inputShape = input.descriptor().shape();
        Shape weightShape = weight.descriptor().shape();
        Shape outputShape = gradient.descriptor().shape();
        Dimension batch = inputShape.dimension(0);
        Dimension outputChannels = weightShape.dimension(0);
        Dimension outputPerGroup =
                DimensionExpressions.floorDivide(outputChannels, attrs.groups());
        long kernelElements = Math.multiplyExact(
                ((StaticDimension) weightShape.dimension(2)).size(),
                ((StaticDimension) weightShape.dimension(3)).size());
        Dimension kernelPerGroup =
                DimensionExpressions.multiply(weightShape.dimension(1), kernelElements);
        Dimension outputPositions = DimensionExpressions.multiply(
                outputShape.dimension(2), outputShape.dimension(3));
        Dimension groupedChannels =
                DimensionExpressions.multiply(kernelPerGroup, attrs.groups());
        StaticDimension groups = new StaticDimension(attrs.groups());

        Tensor[] result = new Tensor[producer.inputs().size()];
        Tensor groupedGradient = gradient
                .reshape(Shape.ofDimensions(
                        batch, groups, outputPerGroup, outputPositions))
                .permute(1, 0, 2, 3);

        if (selectedInputs[0]) {
            Tensor groupedWeight = weight.reshape(
                    Shape.ofDimensions(groups, outputPerGroup, kernelPerGroup));
            Tensor groupedColumns = swapLastTwo(groupedWeight)
                    .expandDims(1)
                    .matmul(groupedGradient);
            Tensor columns = groupedColumns
                    .permute(1, 0, 2, 3)
                    .reshape(Shape.ofDimensions(batch, groupedChannels, outputPositions));
            result[0] = normalize(
                    columns.fold2d(originalInput.descriptor().shape(), window),
                    originalInput);
        }

        if (selectedInputs[1]) {
            Tensor inputColumns = input.unfold2d(window)
                    .reshape(Shape.ofDimensions(
                            batch, groups, kernelPerGroup, outputPositions))
                    .permute(1, 0, 2, 3);
            Tensor groupedWeightGradient = groupedGradient
                    .matmul(swapLastTwo(inputColumns))
                    .sum(1, false);
            result[1] = normalize(
                    groupedWeightGradient.reshape(originalWeight.descriptor().shape()),
                    originalWeight);
        }

        if (producer.inputs().size() == 3 && selectedInputs[2]) {
            result[2] = normalize(
                    gradient.sum(new int[] {0, 2, 3}, false),
                    producer.inputs().get(2));
        }
        return result;
    }

    /**
     * Reconstructs the exact public window attributes represented by one convolution occurrence.
     *
     * @param attrs non-null intrinsic convolution geometry
     * @param weight non-null weight Tensor whose static final axes define the kernel
     * @return a new immutable window attribute value for unfold and fold expressions
     */
    private static Window2dAttrs window(Conv2dAttrs attrs, Tensor weight) {
        Shape shape = weight.descriptor().shape();
        return new Window2dAttrs(
                ((StaticDimension) shape.dimension(2)).size(),
                ((StaticDimension) shape.dimension(3)).size(),
                attrs.strideHeight(),
                attrs.strideWidth(),
                attrs.paddingHeight(),
                attrs.paddingWidth(),
                attrs.dilationHeight(),
                attrs.dilationWidth(),
                false);
    }

    /**
     * Converts an operand to the requested promoted result type only when required.
     *
     * @param input non-null floating operand
     * @param dataType non-null floating result type
     * @return {@code input} when its type already matches; otherwise a new cast expression
     */
    private static Tensor cast(Tensor input, DataType dataType) {
        return input.descriptor().dataType() == dataType ? input : input.cast(dataType);
    }

    /**
     * Restores one selected cotangent to an original input's exact Shape and data type.
     *
     * @param gradient non-null selected cotangent
     * @param input non-null original input whose descriptor is restored
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

    /**
     * Builds a public permutation expression that exchanges only the final two axes.
     *
     * @param tensor non-null preflight-approved Tensor with rank at least two
     * @return a new Tensor expression with the final two axes exchanged
     */
    private static Tensor swapLastTwo(Tensor tensor) {
        int rank = tensor.descriptor().shape().rank();
        int[] axes = new int[rank];
        for (int axis = 0; axis < rank; axis++) {
            axes[axis] = axis;
        }
        axes[rank - 2] = rank - 1;
        axes[rank - 1] = rank - 2;
        return tensor.permute(axes);
    }
}
