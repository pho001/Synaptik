package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.tensor.TensorId;
import java.util.Objects;

/**
 * Associates one ordered functional-gradient target with its final graph value for publication.
 *
 * <p>The binding carries stable identities and stage-local ordering only. Its value may be shared
 * by several target bindings. Owning-graph membership and complete output-boundary order are
 * validated by {@link PublicationPlan}; no Tensor gradient state, storage, publication policy,
 * prepared state, or runtime behavior is retained.</p>
 *
 * @param derivativeOrder exact derivative order, either one or two
 * @param targetIndex zero-based target position within that derivative stage
 * @param target non-null stable identity projected from the exact validated target Tensor
 * @param valueId non-null final graph-local gradient value, which may be shared by several
 *     bindings
 */
public record GradientPublicationBinding(
        int derivativeOrder,
        int targetIndex,
        TensorId target,
        ValueId valueId) {
    /**
     * Validates one immutable gradient publication binding.
     *
     * @param derivativeOrder exact derivative order, either one or two
     * @param targetIndex zero-based target position within that derivative stage
     * @param target non-null stable exact-target identity projection
     * @param valueId non-null final graph-local gradient value
     * @throws NullPointerException if {@code target} is {@code null}, with message {@code target},
     *     or {@code valueId} is {@code null}, with message {@code valueId}
     * @throws IllegalArgumentException if the order is not one or two or the target index is
     *     negative
     */
    public GradientPublicationBinding {
        if (derivativeOrder != 1 && derivativeOrder != 2) {
            throw new IllegalArgumentException("derivativeOrder must be 1 or 2");
        }
        if (targetIndex < 0) {
            throw new IllegalArgumentException("targetIndex must not be negative");
        }
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(valueId, "valueId");
    }

    /**
     * Returns the derivative stage represented by this binding.
     *
     * @return one for a first-stage result or two for a second-stage result
     */
    public int derivativeOrder() {
        return derivativeOrder;
    }

    /**
     * Returns the target position within the represented derivative stage.
     *
     * @return the non-negative zero-based stage-local target index
     */
    public int targetIndex() {
        return targetIndex;
    }

    /**
     * Returns the stable identity projected from the exact validated target Tensor.
     *
     * @return the exact non-null immutable {@link TensorId} reference supplied at construction
     */
    public TensorId target() {
        return target;
    }

    /**
     * Returns the final graph-local gradient value associated with the target.
     *
     * @return the exact non-null immutable {@link ValueId} reference supplied at construction;
     *     the result does not establish membership in a particular graph
     */
    public ValueId valueId() {
        return valueId;
    }
}
