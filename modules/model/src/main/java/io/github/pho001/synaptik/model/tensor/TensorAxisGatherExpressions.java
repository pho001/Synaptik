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
 * Constructs locally validated, storage-free axis-gather expressions and embedding composition
 * for {@link Tensor}.
 *
 * <p>Every operation consumes ordered {@code [data, indices]} inputs and requires exact
 * {@link DataType#INT32} or {@link DataType#INT64} indices. {@link AxisGatherKind#GATHER}
 * replaces one data axis with the complete indices Shape. {@link AxisGatherKind#GATHER_ELEMENTS}
 * requires same-rank non-axis alignment and retains the exact indices Shape.</p>
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
     * Validates and creates one canonical GATHER expression.
     *
     * <p>The complete indices Shape replaces the selected data Dimension. Thus data
     * {@code [2, 3, 4]}, axis {@code 1}, and indices {@code [5, 6]} produce
     * {@code [2, 5, 6, 4]}; scalar indices produce {@code [2, 4]}. Every inserted indices
     * Dimension and unaffected data Dimension is retained exactly.</p>
     *
     * @param data non-null value tensor retained as provenance input zero and never mutated
     * @param indices non-null INT32 or INT64 coordinate tensor retained as provenance input one
     * @param axis raw positive or negative data axis normalized once against the data Shape
     * @return a non-null fresh GATHER tensor with inserted indices Shape and unresolved layout
     * @throws NullPointerException if {@code data} or {@code indices} is null, checked in order
     * @throws IllegalArgumentException if the indices data type is not INT32 or INT64
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
        Shape resultShape = gatherShape(
                dataShape, indicesDescriptor.shape(), normalizedAxis);
        IndexAxisAttrs attrs = new IndexAxisAttrs(normalizedAxis);
        return create(
                data, indices, dataDescriptor, resultShape, AxisGatherKind.GATHER, attrs);
    }

    /**
     * Validates an embedding-weight table and creates one canonical axis-zero GATHER expression.
     *
     * <p>{@code weights} must have rank two and exact BFLOAT16, FLOAT32, or FLOAT64 type. Its
     * axis zero is the vocabulary axis and axis one is the embedding Dimension. Indices may have
     * any rank, including scalar, but must have exact INT32 or INT64 type. The result Shape is the
     * complete indices Shape followed by the exact weight axis-one Dimension.</p>
     *
     * <p>After embedding-specific validation, this method delegates directly to
     * {@link #gather(Tensor, Tensor, int)} with axis zero. The sole resulting producer therefore
     * retains the ordinary {@link AxisGatherKind#GATHER} operation, {@code IndexAxisAttrs(0)},
     * exact ordered inputs {@code [weights, indices]}, and one output at provenance index zero.
     * It preserves the weight type and gradient eligibility and creates a fresh ID only during
     * final derived-Tensor construction. No index value is read: negative and out-of-range values
     * remain invalid for future ordinary Gather execution, where bounds must be enforced safely
     * without wrapping, clamping, padding, or selecting a default row. No padding-index,
     * sparse-gradient, maximum-norm, or frequency-scaling option changes that ordinary Gather
     * occurrence.</p>
     *
     * @param weights non-null rank-two BFLOAT16, FLOAT32, or FLOAT64 table retained as provenance
     *     input zero and never mutated
     * @param indices non-null INT32 or INT64 coordinates of any rank retained as provenance input
     *     one and never mutated
     * @return a non-null fresh storage-free GATHER tensor whose Shape is the indices Shape plus
     *     the exact weight embedding Dimension, with weight metadata and unresolved layout
     * @throws NullPointerException if {@code weights} or {@code indices} is null, checked in order
     * @throws IllegalArgumentException if weights are not rank two, weights are not floating, or
     *     indices are not INT32 or INT64, checked in that order before ID allocation
     * @throws ArithmeticException if checked Gather result-Shape metadata construction overflows
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor embedding(Tensor weights, Tensor indices) {
        Objects.requireNonNull(weights, "weights");
        Objects.requireNonNull(indices, "indices");
        TensorDescriptor weightsDescriptor = weights.descriptor();
        int weightsRank = weightsDescriptor.shape().rank();
        if (weightsRank != 2) {
            throw new IllegalArgumentException(
                    "embedding weights rank must be 2: actual=" + weightsRank);
        }
        DataType weightsDataType = weightsDescriptor.dataType();
        if (!weightsDataType.isFloating()) {
            throw new IllegalArgumentException(
                    "embedding weights data type must be BFLOAT16, FLOAT32, or FLOAT64: "
                            + weightsDataType);
        }
        validateIndexType("embedding", indices.descriptor());
        return gather(weights, indices, 0);
    }

    /**
     * Validates and creates one GATHER_ELEMENTS expression with aligned non-axis coordinates.
     *
     * <p>Data and indices ranks must match, and every non-selected Dimension must be structurally
     * equal. Selected extents may differ. Data {@code [2, 3, 4]}, indices {@code [2, 7, 4]}, and
     * axis {@code 1} retain the exact indices Shape {@code [2, 7, 4]} as the result.</p>
     *
     * @param data non-null value tensor retained as provenance input zero and never mutated
     * @param indices non-null same-rank INT32 or INT64 tensor retained as provenance input one;
     *     its non-axis Dimensions must equal data and its exact Shape becomes the result
     * @param axis raw positive or negative data axis normalized once against the data Shape
     * @return a non-null fresh GATHER_ELEMENTS tensor retaining exact indices Shape and unresolved
     *     layout
     * @throws NullPointerException if {@code data} or {@code indices} is null, checked in order
     * @throws IllegalArgumentException if index type, rank, or non-axis alignment is invalid
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the data Shape
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor gatherElements(Tensor data, Tensor indices, int axis) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(indices, "indices");
        TensorDescriptor dataDescriptor = data.descriptor();
        TensorDescriptor indicesDescriptor = indices.descriptor();
        validateIndexType("gatherElements", indicesDescriptor);
        Shape dataShape = dataDescriptor.shape();
        int normalizedAxis = dataShape.normalizeAxis(axis);
        Shape indicesShape = indicesDescriptor.shape();
        validateGatherElements(dataShape, indicesShape, normalizedAxis);
        IndexAxisAttrs attrs = new IndexAxisAttrs(normalizedAxis);
        return create(
                data,
                indices,
                dataDescriptor,
                indicesShape,
                AxisGatherKind.GATHER_ELEMENTS,
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
     * Replaces one data axis with every indices Dimension in original order.
     *
     * @param dataShape non-null source data Shape of rank at least one
     * @param indicesShape non-null index Shape whose complete Dimensions are inserted
     * @param normalizedAxis normalized existing data axis to replace
     * @return a non-null Shape ordered as data-before, all indices, then data-after
     */
    private static Shape gatherShape(
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
    private static void validateGatherElements(
            Shape dataShape, Shape indicesShape, int normalizedAxis) {
        if (indicesShape.rank() != dataShape.rank()) {
            throw new IllegalArgumentException(
                    "gatherElements indices rank must match data rank: expected="
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
                        "gatherElements indices dimension at axis " + axis
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
