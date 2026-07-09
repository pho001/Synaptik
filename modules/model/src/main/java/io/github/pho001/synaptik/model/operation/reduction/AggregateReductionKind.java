package io.github.pho001.synaptik.model.operation.reduction;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent aggregate-reduction semantics.
 *
 * <p>An aggregate reduction combines values from either every input axis or one selected input
 * axis. Full forms of {@link #SUM}, {@link #MEAN}, {@link #PROD}, {@link #MIN}, {@link #MAX},
 * {@link #ALL}, and {@link #ANY} pair with {@link NoOperationAttrs#INSTANCE}. Their single-axis
 * forms pair with {@link AxisReductionAttrs}, whose axis is already normalized and non-negative.
 * Masked, axis-removing {@link #SUM} and {@link #MEAN} forms pair with
 * {@link MaskedReductionAttrs}; their ordered provenance is {@code [input, mask]}, and the mask
 * must use ordinary right-aligned broadcasting to produce exactly the input Shape.
 * {@link #ARG_MAX} pairs only with {@link ArgMaxAttrs} because choosing among equal maxima is part
 * of that operation's semantics. Family-owned signatures enforce these exact pairings and declare
 * one input for ordinary forms or two ordered inputs for masked forms.</p>
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
     * {@link AxisReductionAttrs}. A masked axis-removing form pairs with
     * {@link MaskedReductionAttrs}: false broadcast mask positions exclude their input values
     * before aggregation, including NaN and infinity, and selecting no values produces floating
     * zero. Callers express non-right-aligned intent through visible Shape transformations. Input
     * and output types,
     * accumulation order and precision, gradients, execution, and backend support are deliberately
     * deferred.</p>
     */
    SUM,

    /**
     * Requests the arithmetic mean of values in the selected reduction domain.
     *
     * <p>The full form pairs with {@link NoOperationAttrs#INSTANCE}; a single-axis form pairs with
     * {@link AxisReductionAttrs}. A masked axis-removing form pairs with
     * {@link MaskedReductionAttrs}: false broadcast mask positions are excluded before
     * aggregation, including positions whose input is NaN or infinity; the denominator is the
     * selected true-count for each output, and a zero selected-count produces NaN in the result
     * floating type. Callers express non-right-aligned intent through visible Shape
     * transformations. NaN payload, input and output validation, accumulation policy, gradients,
     * execution, and backend support are deliberately deferred.</p>
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
    ARG_MAX;

    private static final List<OperationSignature> SUM_MEAN_SIGNATURES = List.of(
            OperationSignature.fixed(NoOperationAttrs.class, 1, 1),
            OperationSignature.fixed(AxisReductionAttrs.class, 1, 1),
            OperationSignature.fixed(MaskedReductionAttrs.class, 2, 1));
    private static final List<OperationSignature> ORDINARY_SIGNATURES = List.of(
            OperationSignature.fixed(NoOperationAttrs.class, 1, 1),
            OperationSignature.fixed(AxisReductionAttrs.class, 1, 1));
    private static final List<OperationSignature> ARG_MAX_SIGNATURES =
            List.of(OperationSignature.fixed(ArgMaxAttrs.class, 1, 1));

    /**
     * Returns the exact structural variants accepted by this aggregate reduction kind.
     *
     * @return the stable masked-capable variants for {@link #SUM} and {@link #MEAN}, the stable
     *     arg-max variant for {@link #ARG_MAX}, or the stable ordinary full/axis variants
     */
    @Override
    public List<OperationSignature> signatures() {
        return switch (this) {
            case SUM, MEAN -> SUM_MEAN_SIGNATURES;
            case ARG_MAX -> ARG_MAX_SIGNATURES;
            case PROD, MIN, MAX, ALL, ANY -> ORDINARY_SIGNATURES;
        };
    }
}
