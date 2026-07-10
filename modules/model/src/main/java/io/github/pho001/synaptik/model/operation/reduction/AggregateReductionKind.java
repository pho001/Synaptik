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
 * {@link #ARG_MIN} and {@link #ARG_MAX} pair only with {@link ArgExtremaAttrs} because choosing
 * among equal extrema is part of those operations' semantics. Family-owned signatures enforce
 * one input for ordinary forms or two ordered inputs for masked forms.</p>
 *
 * <p>Each constant identifies requested mathematics only. The ordinary Tensor-construction
 * contract accepts floating input for all five numeric kinds and signed-integral input for
 * {@code SUM}, {@code PROD}, {@code MIN}, and {@code MAX}. Integral results retain their input
 * type: sum and product use fixed-width modular arithmetic, while minimum and maximum use signed
 * order and the bounded type extrema as their empty-domain identities. The kind itself stores no
 * axis, input, output descriptor, graph occurrence, gradient rule, executable behavior, or
 * backend support. Enum identity supplies typed equality and hashing, so equal names in another
 * operation family remain distinct. Inherited enum text is diagnostic only and is not a
 * serialization, registry, dispatch, or kernel contract.</p>
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
     * and output types are selected by Tensor construction. For INT32 and INT64 ordinary forms,
     * the result retains the input type, addition is modulo {@code 2^32} or {@code 2^64},
     * reassociation is permitted, and an empty reduction domain has result zero. Floating
     * accumulation and empty-domain policy, gradients, execution, and backend support remain
     * outside this semantic kind.</p>
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
     * {@link AxisReductionAttrs}. For INT32 and INT64 ordinary forms, the result retains the input
     * type, multiplication is modulo {@code 2^32} or {@code 2^64}, reassociation is permitted,
     * and an empty reduction domain has result one. Floating multiplication order and
     * empty-domain policy, gradients, execution, and backend support remain outside this semantic
     * kind.</p>
     */
    PROD,

    /**
     * Requests the minimum value in the selected reduction domain.
     *
     * <p>The full form pairs with {@link NoOperationAttrs#INSTANCE}; a single-axis form pairs with
     * {@link AxisReductionAttrs}. For INT32 and INT64 ordinary forms, the result retains the input
     * type, values use signed order, and an empty domain produces {@link Integer#MAX_VALUE} or
     * {@link Long#MAX_VALUE}, respectively. Floating comparison and empty-domain policy,
     * gradients, execution, and backend support remain outside this semantic kind.</p>
     */
    MIN,

    /**
     * Requests the maximum value in the selected reduction domain.
     *
     * <p>The full form pairs with {@link NoOperationAttrs#INSTANCE}; a single-axis form pairs with
     * {@link AxisReductionAttrs}. For INT32 and INT64 ordinary forms, the result retains the input
     * type, values use signed order, and an empty domain produces {@link Integer#MIN_VALUE} or
     * {@link Long#MIN_VALUE}, respectively. Floating comparison and empty-domain policy,
     * gradients, execution, and backend support remain outside this semantic kind.</p>
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
     * <p>This kind pairs only with {@link ArgExtremaAttrs}; unlike the ordinary aggregate kinds,
     * it has no full-reduction form. The shared arg-extrema contract fixes numeric eligibility,
     * INT64 index results, ordering, ties, and empty-axis validity. Gradient rules, execution, and
     * backend support remain outside this semantic kind.</p>
     */
    ARG_MAX,

    /**
     * Requests a logical index of a minimum value along one selected axis.
     *
     * <p>This kind pairs only with {@link ArgExtremaAttrs}; unlike the ordinary aggregate kinds,
     * it has no full-reduction form. Numeric ordering, ties, and empty-axis validity are fixed by
     * the shared arg-extrema contract while execution and backend support remain deferred.</p>
     */
    ARG_MIN;

    private static final List<OperationSignature> SUM_MEAN_SIGNATURES = List.of(
            OperationSignature.fixed(NoOperationAttrs.class, 1, 1),
            OperationSignature.fixed(AxisReductionAttrs.class, 1, 1),
            OperationSignature.fixed(MaskedReductionAttrs.class, 2, 1));
    private static final List<OperationSignature> ORDINARY_SIGNATURES = List.of(
            OperationSignature.fixed(NoOperationAttrs.class, 1, 1),
            OperationSignature.fixed(AxisReductionAttrs.class, 1, 1));
    private static final List<OperationSignature> ARG_EXTREMA_SIGNATURES =
            List.of(OperationSignature.fixed(ArgExtremaAttrs.class, 1, 1));

    /**
     * Returns the exact structural variants accepted by this aggregate reduction kind.
     *
     * @return the stable masked-capable variants for {@link #SUM} and {@link #MEAN}, the stable
     *     arg-extrema variant for {@link #ARG_MIN} and {@link #ARG_MAX}, or the stable ordinary
     *     full/axis variants
     */
    @Override
    public List<OperationSignature> signatures() {
        return switch (this) {
            case SUM, MEAN -> SUM_MEAN_SIGNATURES;
            case ARG_MIN, ARG_MAX -> ARG_EXTREMA_SIGNATURES;
            case PROD, MIN, MAX, ALL, ANY -> ORDINARY_SIGNATURES;
        };
    }
}
