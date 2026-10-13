package io.github.pho001.synaptik.model.operation.normalization;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent softmax normalization semantics.
 *
 * <p>Both kinds have one logical input and preserve every logical input position. A normalization
 * slice contains the positions that differ only along the selected axis while every other logical
 * coordinate remains fixed. The already normalized, non-negative axis is carried by
 * {@link SoftmaxAttrs}; this enum stores no input, axis, shape, result, or graph-occurrence state.</p>
 *
 * <p>The valid family compositions pair either {@link #SOFTMAX} or {@link #LOG_SOFTMAX} with
 * {@link SoftmaxAttrs}. Their shared family-owned signature enforces that exact pairing and
 * declares one input and one output.</p>
 *
 * <p>All-finite nonempty slices use a maximum shift before exponentiation, so finite inputs do
 * not acquire NaN merely because an unshifted exponential would overflow. Any NaN input makes
 * all results NaN. With positive infinity and no NaN, those positions are NaN, and all other
 * positions are positive zero for {@code SOFTMAX} or negative infinity for
 * {@code LOG_SOFTMAX}. A slice containing negative infinity but no positive infinity or NaN
 * retains its literal unshifted formula and its overflow/underflow result classes; in
 * particular, all-negative-infinity slices produce NaN. Empty slices publish nothing.</p>
 *
 * <p>Axis, slice membership, positions, and contributor sets remain exact. On an all-finite
 * slice, select the exact represented maximum, form each working-dtype shift, include each
 * shifted EXP once in a rounded sum, and then divide for SOFTMAX or subtract LOG(sum) from each
 * shift for LOG_SOFTMAX. FLOAT64 inputs use FLOAT64 working arithmetic; FLOAT32 inputs and both
 * low types use FLOAT32. Compensated accumulation may track rounding residuals without adding
 * contributors. Shifted EXP may be recomputed when producing outputs. A working shift
 * may overflow to negative infinity; this gives zero probability or negative-infinity
 * log-probability without using unshifted EXP. Only named floating
 * arithmetic primitive inputs/results may use dtype-specific DAZ/FTZ for FLOAT32, BFLOAT16, and
 * FLOAT16, never FLOAT64. Homogeneous low arithmetic works and accumulates in FLOAT32, with one
 * final low narrowing per output. Existing NaN, infinity, and empty-slice rules remain; neither
 * composite receives a blanket output tolerance. See the <a
 * href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#profile-free-numerical-semantics">Model
 * numerical-semantics contract</a>.</p>
 *
 * <p>These kinds define Model meaning only. They do not define result descriptors, provenance,
 * gradients, compiler decomposition, storage, execution, or backend availability. Enum identity
 * supplies typed equality and hashing. Inherited {@link #name()} and {@link #toString()} text is
 * diagnostic only, not a serialization, parsing, registry, dispatch, or kernel contract.</p>
 */
public enum SoftmaxKind implements OperationKind {
    /**
     * Requests normalized probabilities along the axis in {@link SoftmaxAttrs}.
     *
     * <p>For finite slice values {@code x_i}, select {@code m = max_j(x_j)} and compute
     * {@code exp(x_i-m) / sum_j(exp(x_j-m))}. Exact arithmetic agrees with the familiar
     * unshifted formula, while finite working arithmetic avoids incidental EXP overflow.
     * Rounded probabilities may be zero after underflow and sum to approximately one. For
     * {@code [1, 2, 3]}, the ideal result is approximately
     * {@code [0.09003057, 0.24472847, 0.66524096]}.</p>
     *
     * <p>The operation preserves slice positions and axis order; it does not reduce rank or retain
     * a singleton reduction axis. The class contract above owns its numerical formula and named
     * arithmetic-site permissions. Input eligibility, result construction, gradients, execution, and backend
     * support belong to later owning contracts.</p>
     */
    SOFTMAX,

    /**
     * Requests natural-log probabilities along the axis in {@link SoftmaxAttrs}.
     *
     * <p>For finite slice values {@code x_i}, select {@code m = max_j(x_j)} and compute
     * {@code (x_i-m) - log(sum_j(exp(x_j-m)))}. In exact arithmetic this is the natural logarithm
     * of the corresponding {@link #SOFTMAX} output, but it is not computed by taking the
     * logarithm of a rounded probability. For {@code [1, 2, 3]}, the ideal result is
     * approximately {@code [-2.40760596, -1.40760596, -0.40760596]}; exponentiating those values
     * yields the corresponding softmax probabilities.</p>
     *
     * <p>This is a distinct first-class semantic kind rather than an implicit softmax-plus-log
     * graph fragment. It preserves slice positions and axis order. The class contract above owns
     * its numerical formula and named arithmetic-site permissions. Input eligibility, result construction,
     * gradients, compiler decomposition, execution, and backend support belong to later owning
     * contracts.</p>
     */
    LOG_SOFTMAX;

    private static final List<OperationSignature> SIGNATURES =
            List.of(OperationSignature.fixed(SoftmaxAttrs.class, 1, 1));

    /**
     * Returns the shared one-input, one-output softmax-attributes signature.
     *
     * @return the stable immutable singleton signature list
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
