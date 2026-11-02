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
     * {@link Operation} from {@code WHERE} and {@link NoOperationAttrs#INSTANCE}; and delegate
     * exact ordered producer inputs {@code [condition, ifTrue, ifFalse]} once to
     * {@link TensorFactory#createDerived(TensorDescriptor, Optional, Operation, List)} with no
     * label. The factory creates the producer and index-zero provenance.</p>
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
        return applyChecked(condition, ifTrue, ifFalse, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param condition non-null exact {@code BOOL} condition retained by exact reference as the
     *     first provenance input; it is not mutated
     * @param ifTrue non-null floating true branch retained by exact reference as the second
     *     provenance input; it is not mutated
     * @param ifFalse non-null floating false branch retained by exact reference as the third
     *     provenance input; it is not mutated
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code condition}, {@code ifTrue}, or {@code ifFalse} is
     *     null, checked in that order with the parameter name as the message
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if the condition is not {@code BOOL}, either branch is not
     *     floating, the branch shapes cannot be broadcast, or the condition cannot be broadcast
     *     with the common branch shape
     * @throws IllegalStateException if tensor identifier space is exhausted after all local model
     *     values have been constructed
     */
    static Tensor applyChecked(Tensor condition, Tensor ifTrue, Tensor ifFalse,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        return applyDescriptors(
                condition == null ? null : condition.descriptor(),
                ifTrue == null ? null : ifTrue.descriptor(),
                ifFalse == null ? null : ifFalse.descriptor(),
                descriptorCheck,
                condition,
                ifTrue,
                ifFalse);
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
        if (inputs.size() != 3 || outputCount != 1) {
            throw new IllegalArgumentException("invalid ordinary descriptor cardinality");
        }
        for (TensorDescriptor input : inputs) {
            Objects.requireNonNull(input, "input descriptor");
        }
        applyDescriptors(inputs.get(0), inputs.get(1), inputs.get(2), sink, null, null, null);
    }

    /**
     * Runs the shared descriptor guards; provenance references are used only for real construction.
     * @param condition descriptor supplying ordinary operand metadata
     * @param ifTrue descriptor supplying ordinary operand metadata
     * @param ifFalse descriptor supplying ordinary operand metadata
     * @param descriptorCheck synchronous, unretained sink; null selects real construction
     * @param originalCondition exact real operand, or null for descriptor completion
     * @param originalIfTrue exact real operand, or null for descriptor completion
     * @param originalIfFalse exact real operand, or null for descriptor completion
     * @return the real result, or null after synchronous descriptor completion
     */
    private static Tensor applyDescriptors(
            TensorDescriptor condition,
            TensorDescriptor ifTrue,
            TensorDescriptor ifFalse,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor originalCondition,
            Tensor originalIfTrue,
            Tensor originalIfFalse) {
        Objects.requireNonNull(condition, "condition");
        Objects.requireNonNull(ifTrue, "ifTrue");
        Objects.requireNonNull(ifFalse, "ifFalse");

        DataType conditionDataType = condition.dataType();
        if (conditionDataType != DataType.BOOL) {
            throw new IllegalArgumentException(
                    "condition must have BOOL data type, but was " + conditionDataType);
        }

        DataType dataType = DataTypePromotion.promoteFloating(
                ifTrue.dataType(), ifFalse.dataType());
        Shape branchShape = ShapeBroadcast.broadcast(
                ifTrue.shape(), ifFalse.shape());
        Shape shape = ShapeBroadcast.broadcast(condition.shape(), branchShape);
        boolean requiresGrad =
                ifTrue.requiresGrad() || ifFalse.requiresGrad();
        TensorDescriptor descriptor =
                new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad);
        Operation operation =
                new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE);
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(
                descriptor,
                Optional.empty(),
                operation,
                List.of(originalCondition, originalIfTrue, originalIfFalse));
    }
}
