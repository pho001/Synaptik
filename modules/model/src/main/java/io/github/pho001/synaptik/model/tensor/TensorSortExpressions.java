package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs stable, storage-free sort and argsort expressions.
 *
 * <p>Ordering is independent for each logical slice along one normalized axis. Equal keys retain
 * increasing logical input-index order. Floating NaNs remain last in both directions, negative
 * zero precedes positive zero in ascending order, and infinities use numerical order. Empty,
 * singleton, dynamic, and expression extents are accepted because full ordering preserves the
 * exact input Shape. This boundary records metadata only and performs no comparison, permutation,
 * graph capture, gradient construction, algorithm selection, backend work, or execution.</p>
 */
final class TensorSortExpressions {
    /** Prevents instantiation because sort-expression construction owns no state. */
    private TensorSortExpressions() {
    }

    /**
     * Validates local metadata and creates one independent single-output ordering occurrence.
     *
     * <p>Validation checks {@code input}, then {@code kind}, then normalizes {@code axis}. Every
     * current data type is eligible. SORT preserves exact input type and gradient eligibility;
     * ARGSORT uses INT64 and false gradient eligibility. Both retain the exact Shape reference,
     * leave layout unresolved, and delegate exactly once to the central one-output factory seam.
     * Failures before factory delegation consume no Tensor identity.</p>
     *
     * @param input non-null Tensor retained as the exact sole producer input and never mutated
     * @param kind non-null exact SORT or ARGSORT identity retained by the operation
     * @param axis positive or negative input axis accepted by {@link Shape#normalizeAxis(int)}
     * @param descending whether non-NaN values use descending rather than ascending order
     * @return a non-null fresh, unlabeled, storage-free Tensor with unresolved layout, exact input
     *     Shape, selected result metadata, one-input provenance, and output index zero
     * @throws NullPointerException if {@code input} or {@code kind} is null, checked in that order
     *     with the corresponding parameter name as the message
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input Shape, including
     *     every axis for scalar input
     * @throws IllegalStateException if Tensor identifier space is exhausted after local metadata
     *     construction
     */
    static Tensor apply(Tensor input, OrderingKind kind, int axis, boolean descending) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");
        Shape shape = input.descriptor().shape();
        int normalizedAxis = shape.normalizeAxis(axis);
        SortAttrs attrs = new SortAttrs(normalizedAxis, descending);
        boolean argsort = kind == OrderingKind.ARGSORT;
        TensorDescriptor descriptor = new TensorDescriptor(
                argsort ? DataType.INT64 : input.descriptor().dataType(),
                shape,
                Optional.empty(),
                !argsort && input.descriptor().requiresGrad());
        Operation operation = new Operation(kind, attrs);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(input));
    }
}
