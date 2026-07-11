package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated storage-free unary elementwise tensor expressions.
 *
 * <p>This package-private boundary owns the common deterministic construction order for all
 * nineteen parameterless unary elementwise kinds. It validates input and kind presence, accepts
 * only a floating input data type, retains the exact data type and shape, records unresolved
 * result layout and unchanged gradient eligibility, and delegates the exact sole producer input
 * once to the central Tensor factory. It does not inspect values or storage, evaluate
 * mathematics, validate numerical domains, decompose first-class transforms, choose an accuracy
 * policy, canonicalize expressions, insert casts, create gradient rules, or capture a graph.</p>
 */
final class TensorUnaryExpressions {
    /** Prevents instantiation because expression construction is stateless and package-local. */
    private TensorUnaryExpressions() {
    }

    /**
     * Creates one fresh derived tensor for a unary elementwise request.
     *
     * <p>Validation and construction occur in this exact order: null-check {@code input} and
     * {@code kind}; read and validate the input descriptor's floating data type; create one
     * unresolved-layout descriptor with the exact input data type, shape reference, and gradient
     * eligibility; create one operation from the exact supplied kind and
     * {@code NoOperationAttrs.INSTANCE}; and delegate the exact sole producer input once to
     * {@link TensorFactory#createDerived(TensorDescriptor, Optional, Operation, List)} with no
     * label. Failures before the final delegation allocate no Tensor identity. A successful call
     * returns the factory's exact fresh, unlabeled, storage-free result; the input and all of its
     * metadata and storage remain unchanged.</p>
     *
     * @param input non-null floating tensor retained by exact reference in result provenance
     * @param kind non-null parameterless unary semantic kind retained in the result operation
     * @return the non-null exact fresh derived tensor returned by the central factory
     * @throws NullPointerException if {@code input} or {@code kind} is null, checked in that order
     *     with the parameter name as the message
     * @throws IllegalArgumentException if the input data type is not floating, with the exact
     *     rejected data type in the message
     * @throws IllegalStateException if tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    static Tensor apply(Tensor input, UnaryElementwiseKind kind) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");

        DataType dataType = input.descriptor().dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "input must be a floating data type, but was " + dataType);
        }

        TensorDescriptor descriptor = new TensorDescriptor(
                dataType,
                input.descriptor().shape(),
                Optional.empty(),
                input.descriptor().requiresGrad());
        Operation operation = new Operation(kind, NoOperationAttrs.INSTANCE);
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(input));
    }
}
