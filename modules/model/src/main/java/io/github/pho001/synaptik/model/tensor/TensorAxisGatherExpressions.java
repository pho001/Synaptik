package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free axis-gather expressions for {@link Tensor}.
 *
 * <p>Every operation consumes ordered {@code [data, indices]} inputs and requires exact
 * {@link DataType#INT32} or {@link DataType#INT64} indices. {@link AxisGatherKind#GATHER} removes
 * one data axis and requires indices of that reduced Shape. {@link AxisGatherKind#GATHER_AXIS}
 * replaces one data axis with the complete indices Shape, and tensor-index {@code take} is its
 * exact alias. {@link AxisGatherKind#TAKE_ALONG_AXIS} requires same-rank non-axis alignment and
 * retains the exact indices Shape.</p>
 *
 * <p>Every fresh result preserves the data tensor's type and gradient eligibility, leaves layout
 * unresolved, and records exact ordered provenance without a label or storage. This field-free
 * helper never reads values, checks index bounds, defines gradient or scatter behavior, captures
 * a graph, chooses materialization or backend support, or executes work.</p>
 */
final class TensorAxisGatherExpressions {
    /** Prevents instantiation because axis-gather expression construction owns no state. */
    private TensorAxisGatherExpressions() {
    }

    /**
     * Validates and creates one shape-reducing GATHER expression.
     *
     * <p>For data {@code [2, 3, 4]}, axis {@code 1}, and indices {@code [2, 4]}, the result is
     * {@code [2, 4]}. Removing the only data axis requires scalar indices and produces the
     * canonical scalar Shape. Existing structural Dimension equality governs dynamic symbols.</p>
     *
     * @param data non-null value tensor retained as provenance input zero and never mutated
     * @param indices non-null INT32 or INT64 coordinate tensor retained as provenance input one;
     *     its Shape must equal the data Shape with the selected axis removed
     * @param axis raw positive or negative data axis normalized once against the data Shape
     * @return a non-null fresh GATHER tensor with reduced Shape and unresolved layout
     * @throws NullPointerException if {@code data} or {@code indices} is null, checked in order
     * @throws IllegalArgumentException if index type or reduced Shape compatibility is invalid
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the data Shape
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor gather(Tensor data, Tensor indices, int axis) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(indices, "indices");
        TensorDescriptor dataDescriptor = data.descriptor();
        TensorDescriptor indicesDescriptor = indices.descriptor();
        validateIndexType("gather", indicesDescriptor);
        Shape dataShape = dataDescriptor.shape();
        int normalizedAxis = dataShape.normalizeAxis(axis);
        Shape resultShape = removeAxis(dataShape, normalizedAxis);
        Shape indicesShape = indicesDescriptor.shape();
        if (!indicesShape.equals(resultShape)) {
            throw new IllegalArgumentException(
                    "gather indices shape must equal data shape without gathered axis: expected="
                            + resultShape + ", actual=" + indicesShape);
        }
        IndexAxisAttrs attrs = new IndexAxisAttrs(normalizedAxis);
        return create(
                data, indices, dataDescriptor, resultShape, AxisGatherKind.GATHER, attrs);
    }

    /**
     * Validates and creates one ONNX-style GATHER_AXIS expression.
     *
     * <p>The complete indices Shape replaces the selected data Dimension. Thus data
     * {@code [2, 3, 4]}, axis {@code 1}, and indices {@code [5, 6]} produce
     * {@code [2, 5, 6, 4]}; scalar indices produce {@code [2, 4]}. Every inserted indices
     * Dimension and unaffected data Dimension is retained exactly.</p>
     *
     * @param data non-null value tensor retained as provenance input zero and never mutated
     * @param indices non-null INT32 or INT64 coordinate tensor retained as provenance input one
     * @param axis raw positive or negative data axis normalized once against the data Shape
     * @return a non-null fresh GATHER_AXIS tensor with inserted indices Shape and unresolved layout
     * @throws NullPointerException if {@code data} or {@code indices} is null, checked in order
     * @throws IllegalArgumentException if the indices data type is not INT32 or INT64
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the data Shape
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor gatherAxis(Tensor data, Tensor indices, int axis) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(indices, "indices");
        TensorDescriptor dataDescriptor = data.descriptor();
        TensorDescriptor indicesDescriptor = indices.descriptor();
        validateIndexType("gatherAxis", indicesDescriptor);
        Shape dataShape = dataDescriptor.shape();
        int normalizedAxis = dataShape.normalizeAxis(axis);
        Shape resultShape = gatherAxisShape(
                dataShape, indicesDescriptor.shape(), normalizedAxis);
        IndexAxisAttrs attrs = new IndexAxisAttrs(normalizedAxis);
        return create(
                data, indices, dataDescriptor, resultShape, AxisGatherKind.GATHER_AXIS, attrs);
    }

    /**
     * Delegates tensor-index take exactly to {@link #gatherAxis(Tensor, Tensor, int)}.
     *
     * @param data data tensor passed unchanged to the shared GATHER_AXIS path
     * @param axis raw positive or negative data axis passed unchanged to the shared path
     * @param indices index tensor passed unchanged to the shared GATHER_AXIS path
     * @return the non-null fresh GATHER_AXIS tensor created by the shared path
     * @throws NullPointerException if {@code data} or {@code indices} is null, checked in order
     * @throws IllegalArgumentException if the indices data type is not INT32 or INT64
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the data Shape
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor take(Tensor data, int axis, Tensor indices) {
        return gatherAxis(data, indices, axis);
    }

    /**
     * Validates and creates one TAKE_ALONG_AXIS expression with aligned non-axis coordinates.
     *
     * <p>Data and indices ranks must match, and every non-selected Dimension must be structurally
     * equal. Selected extents may differ. Data {@code [2, 3, 4]}, indices {@code [2, 7, 4]}, and
     * axis {@code 1} retain the exact indices Shape {@code [2, 7, 4]} as the result.</p>
     *
     * @param data non-null value tensor retained as provenance input zero and never mutated
     * @param indices non-null same-rank INT32 or INT64 tensor retained as provenance input one;
     *     its non-axis Dimensions must equal data and its exact Shape becomes the result
     * @param axis raw positive or negative data axis normalized once against the data Shape
     * @return a non-null fresh TAKE_ALONG_AXIS tensor retaining exact indices Shape and unresolved
     *     layout
     * @throws NullPointerException if {@code data} or {@code indices} is null, checked in order
     * @throws IllegalArgumentException if index type, rank, or non-axis alignment is invalid
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the data Shape
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor takeAlongAxis(Tensor data, Tensor indices, int axis) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(indices, "indices");
        TensorDescriptor dataDescriptor = data.descriptor();
        TensorDescriptor indicesDescriptor = indices.descriptor();
        validateIndexType("takeAlongAxis", indicesDescriptor);
        Shape dataShape = dataDescriptor.shape();
        int normalizedAxis = dataShape.normalizeAxis(axis);
        Shape indicesShape = indicesDescriptor.shape();
        validateTakeAlongAxis(dataShape, indicesShape, normalizedAxis);
        IndexAxisAttrs attrs = new IndexAxisAttrs(normalizedAxis);
        return create(
                data,
                indices,
                dataDescriptor,
                indicesShape,
                AxisGatherKind.TAKE_ALONG_AXIS,
                attrs);
    }

    /**
     * Requires the exact supported signed-integral index representation.
     *
     * @param operation non-null stable public operation name used in exact failure text
     * @param indicesDescriptor non-null exact descriptor supplying the index data type
     * @throws IllegalArgumentException if the type is not INT32 or INT64
     */
    private static void validateIndexType(
            String operation, TensorDescriptor indicesDescriptor) {
        DataType dataType = indicesDescriptor.dataType();
        if (dataType != DataType.INT32 && dataType != DataType.INT64) {
            throw new IllegalArgumentException(
                    operation + " indices data type must be INT32 or INT64: " + dataType);
        }
    }

    /**
     * Removes one data axis while preserving every unaffected exact Dimension reference.
     *
     * @param dataShape non-null source Shape of rank at least one
     * @param normalizedAxis normalized existing data axis to remove
     * @return a non-null rank-minus-one Shape; rank-one input produces canonical scalar Shape
     */
    private static Shape removeAxis(Shape dataShape, int normalizedAxis) {
        Dimension[] resultDimensions = new Dimension[dataShape.rank() - 1];
        for (int dataAxis = 0, resultAxis = 0;
                dataAxis < dataShape.rank(); dataAxis++) {
            if (dataAxis != normalizedAxis) {
                resultDimensions[resultAxis++] = dataShape.dimensions().get(dataAxis);
            }
        }
        return Shape.ofDimensions(resultDimensions);
    }

    /**
     * Replaces one data axis with every indices Dimension in original order.
     *
     * @param dataShape non-null source data Shape of rank at least one
     * @param indicesShape non-null index Shape whose complete Dimensions are inserted
     * @param normalizedAxis normalized existing data axis to replace
     * @return a non-null Shape ordered as data-before, all indices, then data-after
     */
    private static Shape gatherAxisShape(
            Shape dataShape, Shape indicesShape, int normalizedAxis) {
        Dimension[] resultDimensions =
                new Dimension[dataShape.rank() - 1 + indicesShape.rank()];
        int resultAxis = 0;
        for (int dataAxis = 0; dataAxis < normalizedAxis; dataAxis++) {
            resultDimensions[resultAxis++] = dataShape.dimensions().get(dataAxis);
        }
        for (Dimension indicesDimension : indicesShape.dimensions()) {
            resultDimensions[resultAxis++] = indicesDimension;
        }
        for (int dataAxis = normalizedAxis + 1; dataAxis < dataShape.rank(); dataAxis++) {
            resultDimensions[resultAxis++] = dataShape.dimensions().get(dataAxis);
        }
        return Shape.ofDimensions(resultDimensions);
    }

    /**
     * Validates equal ranks and structural equality of every non-selected Dimension.
     *
     * @param dataShape non-null source data Shape
     * @param indicesShape non-null candidate indices Shape
     * @param normalizedAxis normalized data axis whose extents are deliberately not compared
     * @throws IllegalArgumentException if ranks differ or the first increasing non-axis Dimension
     *     differs structurally
     */
    private static void validateTakeAlongAxis(
            Shape dataShape, Shape indicesShape, int normalizedAxis) {
        if (indicesShape.rank() != dataShape.rank()) {
            throw new IllegalArgumentException(
                    "takeAlongAxis indices rank must match data rank: expected="
                            + dataShape.rank() + ", actual=" + indicesShape.rank());
        }
        for (int axis = 0; axis < dataShape.rank(); axis++) {
            if (axis == normalizedAxis) {
                continue;
            }
            Dimension expected = dataShape.dimensions().get(axis);
            Dimension actual = indicesShape.dimensions().get(axis);
            if (!actual.equals(expected)) {
                throw new IllegalArgumentException(
                        "takeAlongAxis indices dimension at axis " + axis
                                + " must match data: expected=" + expected + ", actual=" + actual);
            }
        }
    }

    /**
     * Creates exact descriptor, semantics, ordered provenance, and one fresh derived Tensor.
     *
     * @param data non-null exact provenance input zero
     * @param indices non-null exact provenance input one
     * @param dataDescriptor non-null exact descriptor supplying retained type and eligibility
     * @param resultShape non-null operation-specific result Shape
     * @param kind non-null exact axis-gather semantic identity
     * @param attrs non-null normalized axis attributes retained by the operation
     * @return a non-null fresh unlabeled, storage-free Tensor with unresolved layout
     * @throws IllegalArgumentException if descriptor invariants reject the supplied metadata
     * @throws IllegalStateException if tensor identifier space is exhausted at final delegation
     */
    private static Tensor create(
            Tensor data,
            Tensor indices,
            TensorDescriptor dataDescriptor,
            Shape resultShape,
            AxisGatherKind kind,
            IndexAxisAttrs attrs) {
        TensorDescriptor descriptor = new TensorDescriptor(
                dataDescriptor.dataType(),
                resultShape,
                Optional.empty(),
                dataDescriptor.requiresGrad());
        Operation operation = new Operation(kind, attrs);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(data, indices));
    }
}
