package io.github.pho001.synaptik.model.operation.reduction;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;

/**
 * Identifies backend-independent aggregate-reduction semantics.
 *
 * <p>An aggregate reduction combines values from either every input axis or one selected input
 * axis. Full forms of {@link #SUM}, {@link #MEAN}, {@link #PROD}, {@link #MIN}, {@link #MAX},
 * {@link #ALL}, and {@link #ANY} pair with {@link NoOperationAttrs#INSTANCE}. Their single-axis
 * forms pair with {@link AxisReductionAttrs}, whose axis is already normalized and non-negative.
 * {@link #ARG_MAX} pairs only with {@link ArgMaxAttrs} because choosing among equal maxima is part
 * of that operation's semantics. The generic operation descriptor does not enforce these typed
 * family pairings.</p>
 *
 * <p>Each constant identifies requested mathematics only. A kind stores no axis, input, output
 * descriptor, graph occurrence, numerical or empty-domain policy, gradient rule, executable
 * behavior, or backend support. Enum identity supplies typed equality and hashing, so equal names
 * in another operation family remain distinct. Inherited enum text is diagnostic only and is not
 * a serialization, registry, dispatch, or kernel contract.</p>
 */
public enum AggregateReductionKind implements OperationKind {
    /**
     * Requests addition of values in the selected reduction domain.
     *
     * <p>The full form pairs with {@link NoOperationAttrs#INSTANCE}; a single-axis form pairs with
     * {@link AxisReductionAttrs}. Input and output types, accumulation order and precision,
     * empty-domain behavior, gradients, execution, and backend support are deliberately deferred.</p>
     */
    SUM,

    /**
     * Requests the arithmetic mean of values in the selected reduction domain.
     *
     * <p>The full form pairs with {@link NoOperationAttrs#INSTANCE}; a single-axis form pairs with
     * {@link AxisReductionAttrs}. Input and output types, denominator and accumulation policy,
     * empty-domain behavior, gradients, execution, and backend support are deliberately deferred.</p>
     */
    MEAN,

    /**
     * Requests multiplication of values in the selected reduction domain.
     *
     * <p>The full form pairs with {@link NoOperationAttrs#INSTANCE}; a single-axis form pairs with
     * {@link AxisReductionAttrs}. Input and output types, multiplication order and overflow,
     * empty-domain behavior, gradients, execution, and backend support are deliberately deferred.</p>
     */
    PROD,

    /**
     * Requests the minimum value in the selected reduction domain.
     *
     * <p>The full form pairs with {@link NoOperationAttrs#INSTANCE}; a single-axis form pairs with
     * {@link AxisReductionAttrs}. Input and output types, comparison and NaN policy, ties,
     * empty-domain behavior, gradients, execution, and backend support are deliberately deferred.</p>
     */
    MIN,

    /**
     * Requests the maximum value in the selected reduction domain.
     *
     * <p>The full form pairs with {@link NoOperationAttrs#INSTANCE}; a single-axis form pairs with
     * {@link AxisReductionAttrs}. Input and output types, comparison and NaN policy, ties,
     * empty-domain behavior, gradients, execution, and backend support are deliberately deferred.</p>
     */
    MAX,

    /**
     * Requests boolean conjunction of values in the selected reduction domain.
     *
     * <p>The full form pairs with {@link NoOperationAttrs#INSTANCE}; a single-axis form pairs with
     * {@link AxisReductionAttrs}. Input and output eligibility, the empty-domain identity,
     * gradients, execution, and backend support are deliberately deferred.</p>
     */
    ALL,

    /**
     * Requests boolean disjunction of values in the selected reduction domain.
     *
     * <p>The full form pairs with {@link NoOperationAttrs#INSTANCE}; a single-axis form pairs with
     * {@link AxisReductionAttrs}. Input and output eligibility, the empty-domain identity,
     * gradients, execution, and backend support are deliberately deferred.</p>
     */
    ANY,

    /**
     * Requests a logical index of a maximum value along one selected axis.
     *
     * <p>This kind pairs only with {@link ArgMaxAttrs}; unlike the ordinary aggregate kinds, it
     * has no full-reduction form in this contract. Input eligibility, index result type, comparison
     * and NaN policy, gradients, execution, and backend support are deliberately deferred.</p>
     */
    ARG_MAX
}
