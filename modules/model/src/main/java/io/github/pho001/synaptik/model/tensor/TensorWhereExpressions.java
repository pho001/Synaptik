package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs the locally validated storage-free conditional-selection tensor expression.
 *
 * <p>This package-private boundary owns the deterministic validation and construction order for
 * the sole {@link WhereSelectionKind#WHERE} expression. It requires one exact {@code BOOL}
 * condition, promotes two floating branches, composes two pairwise local broadcasts, creates an
 * unresolved result descriptor, records exact ordered provenance, and delegates identity
 * allocation exactly once to the central Tensor factory.</p>
 *
 * <p>Construction is eager only for expression metadata. This helper does not inspect values or
 * storage, choose or evaluate a branch, define evaluation order or gradient routing, insert casts,
 * create a ternary broadcast plan or resolved layout, implement scalar-index selection, capture a
 * graph, or provide compiler, runtime, or backend behavior.</p>
 */
final class TensorWhereExpressions {
    /** Prevents instantiation because conditional expression construction is stateless. */
    private TensorWhereExpressions() {
    }

    /**
     * Creates one fresh derived tensor for ordered conditional selection.
     *
     * <p>Validation and construction occur in this exact order: null-check {@code condition},
     * {@code ifTrue}, and {@code ifFalse}; validate the condition's exact {@code BOOL} data type;
     * invoke floating promotion exactly once with the true branch as left operand and false branch
     * as right operand; broadcast the two branch shapes exactly once; broadcast the condition
     * shape with that common branch shape exactly once; create one unresolved descriptor from the
     * promoted type, final shape, and branch-only gradient-eligibility OR; create one
     * {@link Operation} from {@code WHERE} and {@link NoOperationAttrs#INSTANCE}; create one
     * provenance value with exact ordered inputs {@code [condition, ifTrue, ifFalse]}; and delegate
     * once to {@link TensorFactory#createDerived(TensorDescriptor, Optional, TensorProvenance)}
     * with no label.</p>
     *
     * <p>Failures before the final factory delegation allocate no Tensor identity. A successful
     * call returns the factory's exact fresh, unlabeled, storage-free result. No supplied Tensor,
     * descriptor, provenance, label, layout, storage association, or storage content is mutated or
     * copied to the result.</p>
     *
     * @param condition non-null exact {@code BOOL} condition retained by exact reference as the
     *     first provenance input; it is not mutated
     * @param ifTrue non-null floating true branch retained by exact reference as the second
     *     provenance input; it is not mutated
     * @param ifFalse non-null floating false branch retained by exact reference as the third
     *     provenance input; it is not mutated
     * @return the non-null exact fresh derived Tensor returned by the central factory
     * @throws NullPointerException if {@code condition}, {@code ifTrue}, or {@code ifFalse} is
     *     null, checked in that order with the parameter name as the message
     * @throws IllegalArgumentException if the condition is not {@code BOOL}, either branch is not
     *     floating, the branch shapes cannot be broadcast, or the condition cannot be broadcast
     *     with the common branch shape
     * @throws IllegalStateException if tensor identifier space is exhausted after all local model
     *     values have been constructed
     */
    static Tensor apply(Tensor condition, Tensor ifTrue, Tensor ifFalse) {
        Objects.requireNonNull(condition, "condition");
        Objects.requireNonNull(ifTrue, "ifTrue");
        Objects.requireNonNull(ifFalse, "ifFalse");

        DataType conditionDataType = condition.descriptor().dataType();
        if (conditionDataType != DataType.BOOL) {
            throw new IllegalArgumentException(
                    "condition must have BOOL data type, but was " + conditionDataType);
        }

        DataType dataType = DataTypePromotion.promoteFloating(
                ifTrue.descriptor().dataType(), ifFalse.descriptor().dataType());
        Shape branchShape = ShapeBroadcast.broadcast(
                ifTrue.descriptor().shape(), ifFalse.descriptor().shape());
        Shape shape = ShapeBroadcast.broadcast(condition.descriptor().shape(), branchShape);
        boolean requiresGrad =
                ifTrue.descriptor().requiresGrad() || ifFalse.descriptor().requiresGrad();
        TensorDescriptor descriptor =
                new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad);
        Operation operation =
                new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE);
        TensorProvenance provenance =
                new TensorProvenance(operation, List.of(condition, ifTrue, ifFalse));
        return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
    }
}
