package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated storage-free floating-classification Tensor expressions.
 *
 * <p>This package-private boundary accepts only floating input data types and produces fixed BOOL
 * descriptors with the exact input shape, unresolved layout, and disabled gradient eligibility.
 * It records one parameterless classification producer without inspecting input storage or
 * values, calculating classifications, inserting casts, capturing a graph, defining gradients,
 * or executing work. Every successful result is fresh, unlabeled, storage-free, and has
 * provenance output index zero over the exact input reference.</p>
 */
final class TensorFloatingClassifications {
    /** Prevents instantiation because expression construction is stateless and package-local. */
    private TensorFloatingClassifications() {
    }

    /**
     * Creates one fresh derived Tensor for a floating classification request.
     *
     * <p>The result describes future elementwise classification values; this method neither reads
     * an attached host-storage value nor makes the classification result available eagerly.</p>
     *
     * @param input non-null floating Tensor retained by exact reference in result provenance
     * @param kind non-null parameterless classification kind retained in the result operation
     * @return the non-null exact fresh BOOL Tensor returned by the central factory
     * @throws NullPointerException if {@code input} or {@code kind} is null, checked in that order
     *     with the parameter name as the message
     * @throws IllegalArgumentException if the input data type is not floating, with the exact
     *     rejected data type in the message
     * @throws IllegalStateException if Tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    static Tensor apply(Tensor input, FloatingClassificationKind kind) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");

        DataType dataType = input.descriptor().dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "input must be a floating data type, but was " + dataType);
        }

        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.BOOL,
                input.descriptor().shape(),
                Optional.empty(),
                false);
        Operation operation = new Operation(kind, NoOperationAttrs.INSTANCE);
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(input));
    }
}
