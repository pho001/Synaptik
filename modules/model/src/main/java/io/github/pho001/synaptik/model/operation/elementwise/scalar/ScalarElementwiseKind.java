package io.github.pho001.synaptik.model.operation.elementwise.scalar;

import io.github.pho001.synaptik.model.operation.OperationKind;

/**
 * Identifies backend-independent, parameterized, one-input scalar elementwise semantics.
 *
 * <p>Each kind describes how one scalar parameter, or one ordered pair of scalar bounds, refines
 * a computation over one logical Tensor input. The scalar attributes are semantic parameters;
 * they are not additional Tensor inputs. One-input arity is family context rather than stored
 * metadata, and a kind retains no input, graph occurrence, result fact, executable behavior, or
 * backend information.</p>
 *
 * <p>The valid kind-to-attributes pairings are {@link #MUL}, {@link #POW}, {@link #CLAMP_MIN},
 * and {@link #CLAMP_MAX} with {@link ScalarValueAttrs}, and {@link #CLAMP} with {@link
 * ClampRangeAttrs}. The generic {@link io.github.pho001.synaptik.model.operation.Operation
 * Operation} descriptor does not enforce these family pairings; consumers that understand this
 * family use the typed kind and attribute value directly.</p>
 *
 * <p>Enum identity supplies typed equality and hashing, so equally named constants in another
 * operation family remain different semantic values. The inherited {@link #name()} and {@link
 * #toString()} text is diagnostic vocabulary only, not a serialization token, registry key,
 * reflective identifier, or backend-dispatch contract.</p>
 */
public enum ScalarElementwiseKind implements OperationKind {
    /**
     * Multiplies each input value by the scalar multiplier in {@link ScalarValueAttrs}.
     *
     * <p>The multiplier is a semantic parameter rather than a second Tensor input. Input
     * eligibility, scalar conversion, result type and shape, numerical edge behavior, gradients,
     * execution, and backend support belong to later owning contracts.</p>
     */
    MUL,

    /**
     * Raises each input value, as the base, to the scalar exponent in {@link ScalarValueAttrs}.
     *
     * <p>The base-and-exponent roles are semantic. Input eligibility, scalar conversion, result
     * type and shape, domain and numerical edge behavior, gradients, execution, and backend
     * support belong to later owning contracts.</p>
     */
    POW,

    /**
     * Constrains each input value to the inclusive lower and upper bounds in {@link
     * ClampRangeAttrs}.
     *
     * <p>The bounds are semantic parameters rather than Tensor inputs. Input eligibility, scalar
     * conversion, result type and shape, special-value and boundary behavior, gradients,
     * execution, and backend support belong to later owning contracts.</p>
     */
    CLAMP,

    /**
     * Constrains each input value to be no lower than the minimum in {@link ScalarValueAttrs}.
     *
     * <p>The minimum is a semantic parameter rather than a Tensor input. Input eligibility,
     * scalar conversion, result type and shape, special-value and boundary behavior, gradients,
     * execution, and backend support belong to later owning contracts.</p>
     */
    CLAMP_MIN,

    /**
     * Constrains each input value to be no greater than the maximum in {@link ScalarValueAttrs}.
     *
     * <p>The maximum is a semantic parameter rather than a Tensor input. Input eligibility,
     * scalar conversion, result type and shape, special-value and boundary behavior, gradients,
     * execution, and backend support belong to later owning contracts.</p>
     */
    CLAMP_MAX
}
