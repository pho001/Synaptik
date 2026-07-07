package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Objects;
import java.util.Optional;

/**
 * Adapts one primitive index sequence to the existing tensor-index take expression path.
 *
 * <p>The field-free helper validates and snapshots caller input, creates one independent dense
 * rank-one INT32 leaf tensor through {@link TensorFactory#fromFlatArray}, and delegates exactly
 * once to {@link TensorAxisGatherExpressions#take(Tensor, int, Tensor)}. It preserves every
 * signed value unchanged and never inspects index bounds.</p>
 *
 * <p>Null or empty input creates no storage and consumes no identifier. Index-tensor creation
 * occurs before axis validation, so an invalid axis leaves its storage allocated and identifier
 * consumed without a result identifier. If final result identity allocation is exhausted, the
 * generated index tensor already exists. No allocation or identifier is rolled back.</p>
 */
final class TensorPrimitiveTakeExpressions {
    /** Prevents instantiation because primitive take adaptation owns no state. */
    private TensorPrimitiveTakeExpressions() {
    }

    /**
     * Snapshots primitive indices, creates their eager index tensor, and delegates to GATHER_AXIS.
     *
     * <p>Validation order is data nullity, indices nullity, and non-empty indices. The caller array
     * is then cloned exactly once. Factory import copies that snapshot into one new JVM-managed
     * heap array, so neither source array is retained. The generated tensor precedes axis
     * validation and becomes exact provenance input one of a successful result; {@code data}
     * remains exact input zero.</p>
     *
     * @param data non-null exact value tensor retained as final provenance input zero
     * @param axis raw positive or negative data axis passed unchanged to tensor-index take after
     *     index-tensor creation
     * @param indices non-null, non-empty caller-owned values cloned once and copied unchanged;
     *     negative and extreme entries are not interpreted or bounds-checked
     * @return a non-null fresh GATHER_AXIS result with ordered
     *     {@code [data, generatedIndices]} provenance and existing tensor-index take metadata
     * @throws NullPointerException if {@code data} or {@code indices} is null, checked in that
     *     order with the corresponding parameter name as message, before allocation
     * @throws IllegalArgumentException if {@code indices} is empty, with message
     *     {@code take indices must not be empty}, before allocation
     * @throws IndexOutOfBoundsException if {@code axis} is invalid after index-tensor allocation
     * @throws IllegalStateException if identifier space is exhausted during eager index creation
     *     or final result creation; prior allocation and identity consumption are not rolled back
     * @throws OutOfMemoryError if snapshot or eager index-storage allocation fails
     */
    static Tensor take(Tensor data, int axis, int[] indices) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(indices, "indices");
        if (indices.length == 0) {
            throw new IllegalArgumentException("take indices must not be empty");
        }
        int[] snapshot = indices.clone();
        Tensor generatedIndices = createIndices(snapshot);
        return TensorAxisGatherExpressions.take(data, axis, generatedIndices);
    }

    /**
     * Creates one copied dense rank-one INT32 leaf tensor from the validated snapshot.
     *
     * <p>The tensor has Shape {@code [snapshot.length]}, canonical zero-offset row-major layout,
     * false gradient eligibility, absent label and provenance, and independent JVM-managed heap
     * storage. The factory copies every value unchanged and retains neither {@code snapshot} nor
     * its array-backed temporary segment. Storage allocation precedes identifier allocation, and
     * an exhaustion failure rolls back neither allocation nor any prior identity.</p>
     *
     * @param snapshot non-null, non-empty private snapshot whose signed values are copied exactly;
     *     ownership remains with this helper and the factory does not mutate or retain it
     * @return a non-null fresh dense INT32 leaf tensor with one factory-assigned identifier
     * @throws IllegalStateException if tensor identifier space is exhausted after destination
     *     storage allocation
     * @throws OutOfMemoryError if destination heap allocation fails before identifier allocation
     */
    private static Tensor createIndices(int[] snapshot) {
        Shape shape = Shape.of(snapshot.length);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.INT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
        return TensorFactory.fromFlatArray(descriptor, Optional.empty(), snapshot);
    }
}
