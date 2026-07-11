package io.github.pho001.synaptik.model.operation.reduction;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent aggregate-reduction semantics.
 *
 * <p>An aggregate reduction combines values from every input axis or from one ordered set of
 * distinct selected axes. Full forms of {@link #SUM}, {@link #MEAN}, {@link #PROD}, {@link #MIN}, {@link #MAX},
 * {@link #ALL}, and {@link #ANY} pair with {@link NoOperationAttrs#INSTANCE}. Their single-axis
 * forms pair with {@link AxisReductionAttrs}, whose axis is already normalized and non-negative;
 * their multi-axis forms pair with {@link MultiAxisReductionAttrs}.
 * Masked, axis-removing {@link #SUM} and {@link #MEAN} forms pair with
 * {@link MaskedReductionAttrs}; their ordered provenance is {@code [input, mask]}, and the mask
 * must use ordinary right-aligned broadcasting to produce exactly the input Shape.
 * {@link #ARG_MIN} and {@link #ARG_MAX} pair only with {@link ArgExtremaAttrs} because choosing
 * among equal extrema is part of those operations' semantics. Family-owned signatures enforce
 * The five advanced floating kinds pair only with ordered multi-axis attributes, using
 * {@link StatisticalReductionAttrs} when correction is part of the meaning. Family-owned
 * signatures enforce one input for ordinary and advanced forms or two ordered inputs for masked
 * forms.</p>
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
     * Floating semantics use exact real addition followed by result-format rounding: NaN
     * propagates, opposite infinities produce NaN, a sole infinity sign is preserved, an empty
     * domain is positive zero, and exact non-empty zero is negative only when every selected value
     * is negative zero. Gradients, execution algorithms, and backend support remain outside this
     * semantic kind.</p>
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
     * transformations. Ordinary floating mean is exact sum divided by positive count: NaN and
     * opposite infinities produce NaN, a sole infinity sign is preserved, empty is NaN, and zero
     * sign follows SUM. NaN payload, execution algorithm, gradients, and backend support remain
     * deliberately unspecified or separately owned.</p>
     */
    MEAN,

    /**
     * Requests multiplication of values in the selected reduction domain.
     *
     * <p>The full form pairs with {@link NoOperationAttrs#INSTANCE}; a single-axis form pairs with
     * {@link AxisReductionAttrs}. For INT32 and INT64 ordinary forms, the result retains the input
     * type, multiplication is modulo {@code 2^32} or {@code 2^64}, reassociation is permitted,
     * and an empty reduction domain has result one. Floating product propagates NaN, makes zero
     * times infinity NaN, follows multiplication parity for zero/infinity sign, and returns
     * positive one for an empty domain. Gradients, execution algorithms, and backend support
     * remain outside this semantic kind.</p>
     */
    PROD,

    /**
     * Requests the minimum value in the selected reduction domain.
     *
     * <p>The full form pairs with {@link NoOperationAttrs#INSTANCE}; a single-axis form pairs with
     * {@link AxisReductionAttrs}. For INT32 and INT64 ordinary forms, the result retains the input
     * type, values use signed order, and an empty domain produces {@link Integer#MAX_VALUE} or
     * {@link Long#MAX_VALUE}, respectively. Floating minimum propagates NaN, orders infinities
     * normally, selects negative zero, and returns positive infinity for an empty domain.
     * Gradients, execution algorithms, and backend support remain outside this semantic kind.</p>
     */
    MIN,

    /**
     * Requests the maximum value in the selected reduction domain.
     *
     * <p>The full form pairs with {@link NoOperationAttrs#INSTANCE}; a single-axis form pairs with
     * {@link AxisReductionAttrs}. For INT32 and INT64 ordinary forms, the result retains the input
     * type, values use signed order, and an empty domain produces {@link Integer#MIN_VALUE} or
     * {@link Long#MIN_VALUE}, respectively. Floating maximum propagates NaN, orders infinities
     * normally, selects positive zero, and returns negative infinity for an empty domain.
     * Gradients, execution algorithms, and backend support remain outside this semantic kind.</p>
     */
    MAX,

    /**
     * Requests boolean conjunction of values in the selected reduction domain.
     *
     * <p>The full form pairs with {@link NoOperationAttrs#INSTANCE}; a single-axis form pairs with
     * {@link AxisReductionAttrs}. Exact BOOL input/output eligibility is selected by Tensor
     * construction, and an empty domain produces true. Gradients, execution, and backend support
     * remain separately owned.</p>
     */
    ALL,

    /**
     * Requests boolean disjunction of values in the selected reduction domain.
     *
     * <p>The full form pairs with {@link NoOperationAttrs#INSTANCE}; a single-axis form pairs with
     * {@link AxisReductionAttrs}. Exact BOOL input/output eligibility is selected by Tensor
     * construction, and an empty domain produces false. Gradients, execution, and backend support
     * remain separately owned.</p>
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
    ARG_MIN,

    /**
     * Requests the logarithm of the sum of exponentials over selected axes.
     *
     * <p>This floating-only kind pairs with {@link MultiAxisReductionAttrs}. Its abstract target
     * is {@code log(sum(exp(x_i)))} without prescribing max subtraction or another algorithm.
     * Empty and all-negative-infinity domains produce negative infinity; NaN produces NaN; and
     * positive infinity wins unless NaN is present. A singleton finite point domain returns its
     * value and preserves signed zero.</p>
     */
    LOG_SUM_EXP,

    /**
     * Requests the corrected second central moment over selected axes.
     *
     * <p>This floating-only kind pairs with {@link StatisticalReductionAttrs} and means
     * {@code sum((x_i - mean)^2) / (N - correction)} with required positive denominator. NaN or
     * infinity produces NaN; a valid constant finite domain produces positive zero. Construction
     * records the formula and correction without choosing an evaluation algorithm.</p>
     */
    VARIANCE,

    /**
     * Requests the non-negative principal square root of corrected variance.
     *
     * <p>This floating-only kind has the same axes, correction, denominator validity, and
     * special-value policy as {@link #VARIANCE}. A valid zero result is positive zero.</p>
     */
    STANDARD_DEVIATION,

    /**
     * Requests the sum of absolute values over selected axes.
     *
     * <p>This floating-only kind pairs with {@link MultiAxisReductionAttrs}. Empty domains produce
     * positive zero, point domains produce absolute value, NaN produces NaN, and infinity produces
     * positive infinity unless NaN is present. The finite target is non-negative.</p>
     */
    L1_NORM,

    /**
     * Requests the non-negative square root of the sum of squares over selected axes.
     *
     * <p>This floating-only kind pairs with {@link MultiAxisReductionAttrs} and shares L1 norm's
     * empty, point, NaN, infinity, and positive-zero contract. It identifies first-class L2 norm
     * semantics rather than a stored square/sum/square-root decomposition.</p>
     */
    L2_NORM;

    private static final List<OperationSignature> SUM_MEAN_SIGNATURES = List.of(
            OperationSignature.fixed(NoOperationAttrs.class, 1, 1),
            OperationSignature.fixed(AxisReductionAttrs.class, 1, 1),
            OperationSignature.fixed(MultiAxisReductionAttrs.class, 1, 1),
            OperationSignature.fixed(MaskedReductionAttrs.class, 2, 1));
    private static final List<OperationSignature> ORDINARY_SIGNATURES = List.of(
            OperationSignature.fixed(NoOperationAttrs.class, 1, 1),
            OperationSignature.fixed(AxisReductionAttrs.class, 1, 1),
            OperationSignature.fixed(MultiAxisReductionAttrs.class, 1, 1));
    private static final List<OperationSignature> ARG_EXTREMA_SIGNATURES =
            List.of(OperationSignature.fixed(ArgExtremaAttrs.class, 1, 1));
    private static final List<OperationSignature> ADVANCED_SIGNATURES =
            List.of(OperationSignature.fixed(MultiAxisReductionAttrs.class, 1, 1));
    private static final List<OperationSignature> STATISTICAL_SIGNATURES =
            List.of(OperationSignature.fixed(StatisticalReductionAttrs.class, 1, 1));

    /**
     * Returns the exact structural variants accepted by this aggregate reduction kind.
     *
     * @return the stable masked-capable variants for {@link #SUM} and {@link #MEAN}, the stable
     *     arg-extrema variant for {@link #ARG_MIN} and {@link #ARG_MAX}, the appropriate advanced
     *     multi-axis or statistical variant, or the stable ordinary full/single-/multi-axis
     *     variants; never null or mutable
     */
    @Override
    public List<OperationSignature> signatures() {
        return switch (this) {
            case SUM, MEAN -> SUM_MEAN_SIGNATURES;
            case ARG_MIN, ARG_MAX -> ARG_EXTREMA_SIGNATURES;
            case PROD, MIN, MAX, ALL, ANY -> ORDINARY_SIGNATURES;
            case LOG_SUM_EXP, L1_NORM, L2_NORM -> ADVANCED_SIGNATURES;
            case VARIANCE, STANDARD_DEVIATION -> STATISTICAL_SIGNATURES;
        };
    }
}
