package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free functional axis-scatter expressions for
 * {@link Tensor}.
 *
 * <p>Every operation consumes ordered {@code [data, indices, updates]} inputs. {@code data} is the
 * unchanged base value and supplies the exact result Shape and type. {@code indices} identifies a
 * target coordinate along one normalized axis, and {@code updates} supplies the value associated
 * with each index position. A reduction defines how the base and updates for one target are
 * combined; a duplicate target occurs when multiple index positions identify that same result
 * coordinate.</p>
 *
 * <p>Indices must use exact {@link DataType#INT32} or {@link DataType#INT64}; updates must use the
 * exact data type. Scatter-elements permits replacement for every current type and arithmetic
 * reductions for floating or integral data.</p>
 *
 * <p>Results preserve data/update gradient eligibility by logical OR, leave layout unresolved,
 * and record exact ordered provenance without a label or storage. This field-free helper never
 * mutates data, reads index or update values, checks bounds or duplicate targets, applies writes
 * or reductions, defines gradients, captures a graph, selects a backend, or executes work.</p>
 */
final class TensorAxisScatterExpressions {
    /** Prevents instantiation because axis-scatter expression construction owns no state. */
    private TensorAxisScatterExpressions() {
    }

    /**
     * Delegates scatter-elements replacement exactly to the explicit reduction path.
     *
     * @param data data tensor passed unchanged to the explicit path
     * @param indices indices tensor passed unchanged to the explicit path
     * @param updates updates tensor passed unchanged to the explicit path
     * @param axis raw data axis passed unchanged to the explicit path
     * @return the non-null fresh SCATTER_ELEMENTS tensor created with {@link ScatterReduction#NONE}
     * @throws NullPointerException if {@code data}, {@code indices}, or {@code updates} is null
     * @throws IllegalArgumentException if type, rank, or Shape validation fails
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the data Shape
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor scatterElements(Tensor data, Tensor indices, Tensor updates, int axis) {
        return scatterElements(data, indices, updates, axis, ScatterReduction.NONE);
    }

    /**
     * Validates and creates one same-rank configurable scatter-elements expression.
     *
     * <p>Indices and updates must have equal ranks and equal Dimensions, while their non-selected
     * Dimensions must equal data. The selected extent may differ from data. For data
     * {@code [2, 3, 4]}, axis {@code 1}, and indices/updates {@code [2, 5, 4]}, the result retains
     * data Shape {@code [2, 3, 4]}. {@code NONE} permits all current types; arithmetic reductions
     * reject BOOL.</p>
     *
     * @param data non-null base tensor retained as provenance input zero and never mutated
     * @param indices non-null INT32 or INT64 tensor retained as provenance input one
     * @param updates non-null exact-data-type tensor retained as provenance input two
     * @param axis raw positive or negative data axis normalized once against the data Shape
     * @param reduction non-null replacement or arithmetic reduction retained in exact attributes
     * @return a non-null fresh SCATTER_ELEMENTS tensor with data Shape/type and unresolved layout
     * @throws NullPointerException if {@code data}, {@code indices}, {@code updates}, or
     *     {@code reduction} is null, checked in order
     * @throws IllegalArgumentException if index type, update type, BOOL reduction, rank, or Shape
     *     compatibility is invalid
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the data Shape
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor scatterElements(
            Tensor data,
            Tensor indices,
            Tensor updates,
            int axis,
            ScatterReduction reduction) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(indices, "indices");
        Objects.requireNonNull(updates, "updates");
        Objects.requireNonNull(reduction, "reduction");
        TensorDescriptor dataDescriptor = data.descriptor();
        TensorDescriptor indicesDescriptor = indices.descriptor();
        TensorDescriptor updatesDescriptor = updates.descriptor();
        validateIndexType("scatterElements", indicesDescriptor);
        validateMatchingDataType("scatterElements", dataDescriptor, updatesDescriptor);
        if (dataDescriptor.dataType() == DataType.BOOL && reduction != ScatterReduction.NONE) {
            throw new IllegalArgumentException(
                    "scatterElements BOOL data supports only NONE reduction: " + reduction);
        }
        Shape dataShape = dataDescriptor.shape();
        int normalizedAxis = dataShape.normalizeAxis(axis);
        Shape indicesShape = indicesDescriptor.shape();
        Shape updatesShape = updatesDescriptor.shape();
        validateScatterElementsShape(
                dataShape, indicesShape, updatesShape, normalizedAxis);
        ScatterElementsAttrs attrs = new ScatterElementsAttrs(normalizedAxis, reduction);
        return create(
                data,
                indices,
                updates,
                dataDescriptor,
                updatesDescriptor,
                AxisScatterKind.SCATTER_ELEMENTS,
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
     * Requires updates to use the exact data element type without promotion or conversion.
     *
     * @param operation non-null stable public operation name used in exact failure text
     * @param dataDescriptor non-null descriptor supplying the expected data type
     * @param updatesDescriptor non-null descriptor supplying the actual updates type
     * @throws IllegalArgumentException if the two data types differ
     */
    private static void validateMatchingDataType(
            String operation,
            TensorDescriptor dataDescriptor,
            TensorDescriptor updatesDescriptor) {
        DataType expected = dataDescriptor.dataType();
        DataType actual = updatesDescriptor.dataType();
        if (actual != expected) {
            throw new IllegalArgumentException(
                    operation + " updates data type must match data: expected="
                            + expected + ", actual=" + actual);
        }
    }

    /**
     * Validates same-rank updates/indices equality and non-selected data alignment in exact order.
     *
     * @param dataShape non-null source data Shape
     * @param indicesShape non-null candidate indices Shape
     * @param updatesShape non-null candidate updates Shape
     * @param normalizedAxis normalized data axis whose extent may differ from indices/updates
     * @throws IllegalArgumentException if a rank or first increasing-axis Dimension check fails
     */
    private static void validateScatterElementsShape(
            Shape dataShape,
            Shape indicesShape,
            Shape updatesShape,
            int normalizedAxis) {
        if (indicesShape.rank() != dataShape.rank()) {
            throw new IllegalArgumentException(
                    "scatterElements indices rank must match data rank: expected="
                            + dataShape.rank() + ", actual=" + indicesShape.rank());
        }
        if (updatesShape.rank() != indicesShape.rank()) {
            throw new IllegalArgumentException(
                    "scatterElements updates rank must match indices rank: expected="
                            + indicesShape.rank() + ", actual=" + updatesShape.rank());
        }
        for (int axis = 0; axis < indicesShape.rank(); axis++) {
            Dimension expected = indicesShape.dimensions().get(axis);
            Dimension actual = updatesShape.dimensions().get(axis);
            if (!actual.equals(expected)) {
                throw new IllegalArgumentException(
                        "scatterElements updates dimension at axis " + axis
                                + " must match indices: expected=" + expected
                                + ", actual=" + actual);
            }
        }
        for (int axis = 0; axis < dataShape.rank(); axis++) {
            if (axis == normalizedAxis) {
                continue;
            }
            Dimension expected = dataShape.dimensions().get(axis);
            Dimension actual = indicesShape.dimensions().get(axis);
            if (!actual.equals(expected)) {
                throw new IllegalArgumentException(
                        "scatterElements indices dimension at axis " + axis
                                + " must match data: expected=" + expected
                                + ", actual=" + actual);
            }
        }
    }

    /**
     * Creates exact descriptor, semantics, ordered provenance, and one fresh derived Tensor.
     *
     * @param data non-null exact provenance input zero and source of result Shape/type
     * @param indices non-null exact provenance input one
     * @param updates non-null exact provenance input two
     * @param dataDescriptor non-null exact descriptor supplying retained Shape/type/eligibility
     * @param updatesDescriptor non-null exact descriptor supplying eligibility to combine
     * @param kind non-null exact axis-scatter semantic identity
     * @param attrs non-null normalized axis/reduction attributes retained by the operation
     * @return a non-null fresh unlabeled, storage-free Tensor with unresolved layout
     * @throws IllegalArgumentException if descriptor invariants reject supplied metadata
     * @throws IllegalStateException if tensor identifier space is exhausted at final delegation
     */
    private static Tensor create(
            Tensor data,
            Tensor indices,
            Tensor updates,
            TensorDescriptor dataDescriptor,
            TensorDescriptor updatesDescriptor,
            AxisScatterKind kind,
            ScatterElementsAttrs attrs) {
        TensorDescriptor descriptor = new TensorDescriptor(
                dataDescriptor.dataType(),
                dataDescriptor.shape(),
                Optional.empty(),
                dataDescriptor.requiresGrad() || updatesDescriptor.requiresGrad());
        Operation operation = new Operation(kind, attrs);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(data, indices, updates));
    }
}
