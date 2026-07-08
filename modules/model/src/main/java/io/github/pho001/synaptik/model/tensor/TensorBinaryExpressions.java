package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated storage-free binary arithmetic tensor expressions.
 *
 * <p>This package-private boundary owns the common deterministic construction order for all seven
 * parameterless binary arithmetic kinds. It validates operand and kind presence, promotes only
 * floating data types, derives only locally provable right-aligned broadcast shape, records an
 * unresolved result layout and gradient-eligibility OR, and delegates exact ordered producer
 * inputs once to the central Tensor factory. It does not inspect or mutate storage,
 * execute arithmetic, canonicalize expressions, insert casts, create gradient rules, or capture a
 * graph.</p>
 */
final class TensorBinaryExpressions {
    /** Prevents instantiation because expression construction is stateless and package-local. */
    private TensorBinaryExpressions() {
    }

    /**
     * Creates one fresh derived tensor for an ordered binary arithmetic request.
     *
     * <p>Validation and construction occur in this exact order: null-check {@code left},
     * {@code right}, and {@code kind}; promote the two descriptor data types; broadcast the two
     * descriptor shapes; create one unresolved-layout descriptor with input gradient eligibility
     * combined by logical OR; create one operation from the exact supplied kind and
     * {@code NoOperationAttrs.INSTANCE}; and delegate ordered exact producer inputs
     * {@code [left, right]} once to
     * {@link TensorFactory#createDerived(TensorDescriptor, Optional, Operation, List)} with no
     * label. Failures before the final delegation allocate no Tensor identity. A successful call
     * returns the factory's exact fresh, unlabeled, storage-free result; neither input nor its
     * metadata or storage is mutated.</p>
     *
     * @param left non-null ordered left operand retained by exact reference in provenance
     * @param right non-null ordered right operand retained by exact reference in provenance
     * @param kind non-null parameterless binary arithmetic semantic kind retained in the result
     *     operation
     * @return the non-null exact fresh derived tensor returned by the central factory
     * @throws NullPointerException if {@code left}, {@code right}, or {@code kind} is null,
     *     checked in that order with the parameter name as the message
     * @throws IllegalArgumentException if either operand data type is not floating or if their
     *     shapes cannot be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    static Tensor apply(Tensor left, Tensor right, BinaryArithmeticKind kind) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        Objects.requireNonNull(kind, "kind");

        DataType dataType = DataTypePromotion.promoteFloating(
                left.descriptor().dataType(), right.descriptor().dataType());
        Shape shape = ShapeBroadcast.broadcast(
                left.descriptor().shape(), right.descriptor().shape());
        TensorDescriptor descriptor = new TensorDescriptor(
                dataType,
                shape,
                Optional.empty(),
                left.descriptor().requiresGrad() || right.descriptor().requiresGrad());
        Operation operation = new Operation(kind, NoOperationAttrs.INSTANCE);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(left, right));
    }
}
