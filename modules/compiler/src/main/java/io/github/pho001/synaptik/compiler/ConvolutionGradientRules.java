package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window3dAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;

import java.util.function.Function;

/**
 * Builds grouped NCHW and NCDHW convolution cotangents through public window and matrix operations.
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
        OperationAttrs attrs = producer.operation().attrs();
        long groupCount = attrs instanceof Conv2dAttrs two ? two.groups()
                : ((Conv3dAttrs) attrs).groups();
        
        Shape inputShape = ops.descriptor(originalInput).shape();
        Shape weightShape = ops.descriptor(originalWeight).shape();
        Shape outputShape = ops.descriptor(gradient).shape();
        Dimension batch = inputShape.dimension(0);
        Dimension outputChannels = weightShape.dimension(0);
        Dimension outputPerGroup =
                DimensionExpressions.floorDivide(outputChannels, groupCount);
        Dimension kernelPerGroup = DimensionExpressions.multiply(
                weightShape.dimension(1), spatialProduct(weightShape));
        Dimension outputPositions = spatialProduct(outputShape);
        Dimension groupedChannels =
                DimensionExpressions.multiply(kernelPerGroup, groupCount);
        StaticDimension groups = new StaticDimension(groupCount);
        OperationAttrs window = window(attrs, weightShape);
        
        V[] result = ops.array(producer.inputs().size());
        V groupedGradient = ops.contiguous(ops.permute(ops.reshape(gradient, Shape.ofDimensions(
                        batch, groups, outputPerGroup, outputPositions)), 1, 0, 2, 3));
        
        if (selectedInputs[0]) {
            V groupedWeight = ops.contiguous(ops.reshape(weight, Shape.ofDimensions(groups, outputPerGroup, kernelPerGroup)));
            V groupedColumns = ops.matmul(ops.contiguous(ops.expandDims(swapLastTwo(groupedWeight), 1)), groupedGradient);
            V columns = ops.contiguous(ops.reshape(ops.contiguous(ops.permute(groupedColumns, 1, 0, 2, 3)), Shape.ofDimensions(batch, groupedChannels, outputPositions)));
            result[0] = normalize(
                    window instanceof Window2dAttrs two
                            ? ops.fold2d(columns, inputShape, two)
                            : ops.fold3d(columns, inputShape, (Window3dAttrs) window),
                    originalInput);
        }
        
        if (selectedInputs[1]) {
            V unfolded = window instanceof Window2dAttrs two
                    ? ops.unfold2d(input, two) : ops.unfold3d(input, (Window3dAttrs) window);
            V inputColumns = ops.contiguous(ops.permute(ops.reshape(unfolded, Shape.ofDimensions(
                            batch, groups, kernelPerGroup, outputPositions)), 1, 0, 2, 3));
            V groupedWeightGradient = ops.sum(ops.matmul(groupedGradient, swapLastTwo(inputColumns)), 1, false);
            result[1] = normalize(
                    ops.contiguous(ops.reshape(groupedWeightGradient, ops.descriptor(originalWeight).shape())),
                    originalWeight);
        }
        
        if (producer.inputs().size() == 3 && selectedInputs[2]) {
            int[] reductionAxes = new int[outputShape.rank() - 1];
            for (int axis = 1; axis < reductionAxes.length; axis++) {
                reductionAxes[axis] = axis + 1;
            }
            result[2] = normalize(
                    ops.sum(gradient, reductionAxes, false),
                    producer.inputs().get(2));
        }
        return result;
    }

    /**
     * Reconstructs the exact public window attributes represented by one convolution occurrence.
     *
     * @param attrs non-null intrinsic convolution geometry
     * @param shape non-null weight Shape whose static final axes define the kernel
     * @return immutable matching 2D or 3D window attributes for unfold and fold expressions
     */
    static OperationAttrs window(OperationAttrs attrs, Shape shape) {
        long height = ((StaticDimension) shape.dimension(shape.rank() - 2)).size();
        long width = ((StaticDimension) shape.dimension(shape.rank() - 1)).size();
        if (attrs instanceof Conv2dAttrs two) {
            return new Window2dAttrs(
                    height, width, two.strideHeight(), two.strideWidth(),
                    two.paddingHeight(), two.paddingWidth(),
                    two.dilationHeight(), two.dilationWidth(), false);
        }
        Conv3dAttrs three = (Conv3dAttrs) attrs;
        return new Window3dAttrs(
                ((StaticDimension) shape.dimension(2)).size(), height, width,
                three.strideDepth(), three.strideHeight(), three.strideWidth(),
                three.paddingDepth(), three.paddingHeight(), three.paddingWidth(),
                three.dilationDepth(), three.dilationHeight(), three.dilationWidth(), false);
    }

    /**
     * Multiplies only spatial axes without resolving any symbolic dimension.
     *
     * @param shape non-null NCHW or NCDHW Shape
     * @return exact symbolic product of axes two through the final axis
     */
    private static Dimension spatialProduct(Shape shape) {
        Dimension product = shape.dimension(2);
        for (int axis = 3; axis < shape.rank(); axis++) {
            product = DimensionExpressions.multiply(product, shape.dimension(axis));
        }
        return product;
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
