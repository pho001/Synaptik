package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated storage-free binary comparison tensor expressions.
 *
 * <p>This package-private boundary owns the common deterministic construction order for all six
 * parameterless binary comparison kinds. It validates operand and kind presence, validates only
 * floating input pairs through the shared promotion contract, derives only locally provable
 * right-aligned broadcast shape, and records an unresolved non-differentiable {@code BOOL} result
 * before creating exact ordered provenance and delegating once to the central Tensor factory. It
 * does not inspect values or storage, execute comparisons, define numerical edge behavior,
 * canonicalize operands, insert casts, create gradient rules, or capture a graph.</p>
 */
final class TensorComparisonExpressions {
    /** Prevents instantiation because comparison construction is stateless and package-local. */
    private TensorComparisonExpressions() {
    }

    /**
     * Creates one fresh derived tensor for an ordered binary comparison request.
     *
     * <p>Validation and construction occur in this exact order: null-check {@code left},
     * {@code right}, and {@code kind}; invoke floating promotion exactly once to validate their
     * common comparison domain without retaining the promoted type; broadcast the two descriptor
     * shapes exactly once; create one unresolved-layout {@code BOOL} descriptor with false
     * gradient eligibility; create one operation from the exact supplied kind and
     * {@code NoOperationAttrs.INSTANCE}; and delegate ordered exact producer inputs
     * {@code [left, right]} once to
     * {@link TensorFactory#createDerived(TensorDescriptor, Optional, Operation, List)} with no
     * label. Failures before the final delegation allocate no Tensor identity. A successful call
     * returns the factory's exact fresh, unlabeled, storage-free result; neither input nor its
     * metadata, gradient eligibility, provenance, or storage is mutated.</p>
     *
     * @param left non-null ordered left floating operand retained by exact reference in provenance
     * @param right non-null ordered right floating operand retained by exact reference in provenance
     * @param kind non-null parameterless comparison semantic kind retained in the result operation
     * @return the non-null exact fresh derived {@code BOOL} tensor returned by the central factory
     * @throws NullPointerException if {@code left}, {@code right}, or {@code kind} is null,
     *     checked in that order with the parameter name as the message
     * @throws IllegalArgumentException if either operand data type is not floating or if their
     *     shapes cannot be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    static Tensor apply(Tensor left, Tensor right, BinaryComparisonKind kind) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        Objects.requireNonNull(kind, "kind");

        DataTypePromotion.promoteFloating(
                left.descriptor().dataType(), right.descriptor().dataType());
        Shape shape = ShapeBroadcast.broadcast(
                left.descriptor().shape(), right.descriptor().shape());
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.BOOL, shape, Optional.empty(), false);
        Operation operation = new Operation(kind, NoOperationAttrs.INSTANCE);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(left, right));
    }
}
