package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;

import java.util.function.Function;

/**
 * Builds grouped NCHW convolution cotangents through public window and matrix operations.
 *
 * <p>Every reshape retains the occurrence's exact symbolic batch, channel, and spatial
 * dimensions. The leading group axis prevents cross-group contraction. Folding accumulates
 * overlap into the original input Shape; no transposed-convolution or backward-only kind is
 * introduced.</p>
 *
 * <p>Descriptor access preserves original source roles without resolving primal values.
 * Real construction uses Tensor references directly; checked metadata construction retains
 * the same eager operation sites, including temporaries not used by a returned cotangent.</p>
 *
 * @param <V> direct Tensor or checked metadata operand representation
 */
final class ConvolutionGradientRules<V> {
    private final OrdinaryConstruction<V> ops;

    /**
     * Binds this formula owner to one compile-local construction route.
     * @param ops non-null direct Tensor or checked metadata construction
     */
    ConvolutionGradientRules(OrdinaryConstruction<V> ops) {
        this.ops = java.util.Objects.requireNonNull(ops);
    }

    /**
     * Builds selected input, weight, and optional bias cotangents.
     *
     * @param producer non-null exact original convolution occurrence
     * @param gradient non-null accumulated cotangent for its sole output
     * @param selectedInputs non-null input-position-aligned selected-route flags; observed but
     *     not mutated
     * @param primal non-null resolver from original primal wrappers to lexical values
     * @return a new input-position-aligned array of cotangents, with {@code null} for unselected
     *     roles
     */
    V[] apply(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            boolean[] selectedInputs,
            Function<V, V> primal) {
        V originalInput = producer.inputs().get(0);
        V originalWeight = producer.inputs().get(1);
        DataType resultType = ops.descriptor(producer.output(0)).dataType();
        V input = cast(primal.apply(originalInput), resultType);
        V weight = cast(primal.apply(originalWeight), resultType);
        Conv2dAttrs attrs = (Conv2dAttrs) producer.operation().attrs();
        Window2dAttrs window = window(attrs, originalWeight);
        
        Shape inputShape = ops.descriptor(originalInput).shape();
        Shape weightShape = ops.descriptor(originalWeight).shape();
        Shape outputShape = ops.descriptor(gradient).shape();
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
        
        V[] result = ops.array(producer.inputs().size());
        V groupedGradient = ops.contiguous(ops.permute(ops.reshape(gradient, Shape.ofDimensions(
                        batch, groups, outputPerGroup, outputPositions)), 1, 0, 2, 3));
        
        if (selectedInputs[0]) {
            V groupedWeight = ops.contiguous(ops.reshape(weight, Shape.ofDimensions(groups, outputPerGroup, kernelPerGroup)));
            V groupedColumns = ops.matmul(ops.contiguous(ops.expandDims(swapLastTwo(groupedWeight), 1)), groupedGradient);
            V columns = ops.contiguous(ops.reshape(ops.contiguous(ops.permute(groupedColumns, 1, 0, 2, 3)), Shape.ofDimensions(batch, groupedChannels, outputPositions)));
            result[0] = normalize(
                    ops.fold2d(columns, ops.descriptor(originalInput).shape(), window),
                    originalInput);
        }
        
        if (selectedInputs[1]) {
            V inputColumns = ops.contiguous(ops.permute(ops.reshape(ops.unfold2d(input, window), Shape.ofDimensions(
                            batch, groups, kernelPerGroup, outputPositions)), 1, 0, 2, 3));
            V groupedWeightGradient = ops.sum(ops.matmul(groupedGradient, swapLastTwo(inputColumns)), 1, false);
            result[1] = normalize(
                    ops.contiguous(ops.reshape(groupedWeightGradient, ops.descriptor(originalWeight).shape())),
                    originalWeight);
        }
        
        if (producer.inputs().size() == 3 && selectedInputs[2]) {
            result[2] = normalize(
                    ops.sum(gradient, new int[] {0, 2, 3}, false),
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
    private Window2dAttrs window(Conv2dAttrs attrs, V weight) {
        Shape shape = ops.descriptor(weight).shape();
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
    private V cast(V input, DataType dataType) {
        return ops.descriptor(input).dataType() == dataType ? input : ops.cast(input, dataType);
    }

    /**
     * Restores one selected cotangent to an original input's exact Shape and data type.
     *
     * @param gradient non-null selected cotangent
     * @param input non-null original input whose descriptor is restored
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

    /**
     * Builds a public permutation expression that exchanges only the final two axes.
     *
     * @param tensor non-null preflight-approved Tensor with rank at least two
     * @return a new Tensor expression with the final two axes exchanged
     */
    private V swapLastTwo(V tensor) {
        int rank = ops.descriptor(tensor).shape().rank();
        int[] axes = new int[rank];
        for (int axis = 0; axis < rank; axis++) {
            axes[axis] = axis;
        }
        axes[rank - 2] = rank - 1;
        axes[rank - 1] = rank - 2;
        return ops.permute(tensor, axes);
    }
}
