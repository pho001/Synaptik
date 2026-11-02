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
        return applyChecked(indices, depth, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param indices non-null INT32 or INT64 logical indices retained as the sole producer input
     * @param depth positive extent of the new trailing one-hot axis
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code indices} is null, with message {@code indices}
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if the input type is not INT32 or INT64, with message
     *     {@code oneHot indices data type must be INT32 or INT64: <type>}, or if depth is not
     *     positive, with message {@code depth must be positive: <depth>}; type is checked first
     * @throws IllegalStateException if Tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}
     */
    static Tensor applyChecked(Tensor indices, long depth,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        return applyDescriptors(
                indices == null ? null : indices.descriptor(),
                depth,
                descriptorCheck,
                indices);
    }

    /**
     * Completes ordinary checked semantics synchronously without creating Tensor identities.
     * The same descriptor guard core serves real construction; the sink is never retained.
     * @param operation non-null ordinary operation selecting this family and its attributes
     * @param inputs non-null ordered operand descriptors, including repeated occurrences
     * @param outputCount exact canonical sibling count
     * @param sink non-null consumer called exactly once with checked semantics and all outputs
     * @throws NullPointerException if a required reference is null
     * @throws IllegalArgumentException if cardinality or ordinary semantic guards reject the request
     */
    static void describeChecked(Operation operation, List<TensorDescriptor> inputs, int outputCount,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> sink) {
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(sink, "sink");
        if (inputs.size() != 1 || outputCount != 1) {
            throw new IllegalArgumentException("invalid ordinary descriptor cardinality");
        }
        for (TensorDescriptor input : inputs) {
            Objects.requireNonNull(input, "input descriptor");
        }
        applyDescriptors(inputs.get(0), ((OneHotAttrs) operation.attrs()).depth(), sink, null);
    }

    /**
     * Runs the shared descriptor guards; provenance references are used only for real construction.
     * @param indices descriptor supplying ordinary operand metadata
     * @param depth ordinary checked construction parameter
     * @param descriptorCheck synchronous, unretained sink; null selects real construction
     * @param originalIndices exact real operand, or null for descriptor completion
     * @return the real result, or null after synchronous descriptor completion
     */
    private static Tensor applyDescriptors(
            TensorDescriptor indices,
            long depth,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor originalIndices) {
        Objects.requireNonNull(indices, "indices");
        TensorDescriptor indicesDescriptor = indices;
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
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(originalIndices));
    }
}
