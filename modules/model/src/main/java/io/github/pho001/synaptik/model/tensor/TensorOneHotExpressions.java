package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated storage-free one-hot Tensor expressions.
 *
 * <p>This field-free helper validates index type before positive depth, appends one fresh static
 * trailing Dimension while preserving every input Dimension reference, and creates one BOOL
 * result. Scalar input becomes rank one; a zero-element input and {@link Long#MAX_VALUE} depth
 * remain structurally valid without element-count materialization. It does not inspect index
 * values, allocate storage, enforce execution-time bounds, define gradients, capture a graph,
 * lower an operation, or execute.</p>
 */
final class TensorOneHotExpressions {
    /** Prevents instantiation because one-hot expression construction is stateless. */
    private TensorOneHotExpressions() {
    }

    /**
     * Creates one fresh one-hot result from logical indices.
     *
     * <p>Validation checks {@code indices}, exact INT32/INT64 type, then positive depth. The
     * result is BOOL with false gradient eligibility, unresolved layout, no label or storage,
     * and exact sole-input provenance. One successful call delegates once to derived creation and
     * therefore creates one fresh single-output producer, provenance output index zero, and one
     * fresh Tensor identifier. Local validation failures consume no identifier.</p>
     *
     * <p>For an eventual index value {@code i}, trailing position {@code j} is true exactly when
     * {@code i == j}. Execution requires {@code 0 <= i < depth}; construction does not inspect
     * stored values, and invalid values do not wrap, clamp, select a default, or produce an
     * all-false row.</p>
     *
     * @param indices non-null INT32 or INT64 logical indices retained as the sole producer input
     * @param depth positive extent of the new trailing one-hot axis
     * @return a non-null fresh storage-free BOOL Tensor whose Shape retains every input Dimension
     *     reference and appends one fresh {@link StaticDimension}
     * @throws NullPointerException if {@code indices} is null, with message {@code indices}
     * @throws IllegalArgumentException if the input type is not INT32 or INT64, with message
     *     {@code oneHot indices data type must be INT32 or INT64: <type>}, or if depth is not
     *     positive, with message {@code depth must be positive: <depth>}; type is checked first
     * @throws IllegalStateException if Tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}
     */
    static Tensor apply(Tensor indices, long depth) {
        Objects.requireNonNull(indices, "indices");
        TensorDescriptor indicesDescriptor = indices.descriptor();
        DataType dataType = indicesDescriptor.dataType();
        if (dataType != DataType.INT32 && dataType != DataType.INT64) {
            throw new IllegalArgumentException(
                    "oneHot indices data type must be INT32 or INT64: " + dataType);
        }
        OneHotAttrs attrs = new OneHotAttrs(depth);
        Shape indicesShape = indicesDescriptor.shape();
        Dimension[] resultDimensions = new Dimension[indicesShape.rank() + 1];
        for (int axis = 0; axis < indicesShape.rank(); axis++) {
            resultDimensions[axis] = indicesShape.dimension(axis);
        }
        resultDimensions[indicesShape.rank()] = new StaticDimension(depth);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.BOOL,
                Shape.ofDimensions(resultDimensions),
                Optional.empty(),
                false);
        Operation operation = new Operation(OneHotKind.ONE_HOT, attrs);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(indices));
    }
}
