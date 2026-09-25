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
 * <p>These kinds use the literal formulas documented below. Consequently, NaN in a nonempty slice
 * makes its formula results NaN; positive-infinity terms produce NaN at those terms and positive
 * zero at finite terms for {@code SOFTMAX}; and an all-negative-infinity nonempty slice produces
 * NaN by zero division or infinity subtraction. Empty slices have no positions to evaluate.</p>
 *
 * <p>Under the Model-owned numerical-profile contract, {@code STRICT_IEEE} retains those formulas
 * and current freedoms. For {@code ACCELERATOR FLOAT32}, exact axis, slice membership, position,
 * and contributor sets are unchanged. Exponential and logarithmic sites use DAZ/FTZ and the
 * five-ULP elementary-site ceiling; sums use the all-contributors-once aggregate floor and the
 * final division/subtraction uses one FLOAT32 operation. Stable algorithms are valid only when
 * their complete results remain inside that recursive set. There is no four-ULP softmax-output
 * oracle or other final-output envelope, and non-FLOAT32 behavior stays strict. See the
 * <a href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#numerical-profiles">normative
 * numerical-profile contract</a>.</p>
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
     * <p>For slice values {@code x_i}, the ideal output at position {@code i} is
     * {@code exp(x_i) / sum_j(exp(x_j))}. For an ordinary finite nonempty slice the outputs are
     * positive and sum to one. For {@code [1, 2, 3]}, the ideal result is approximately
     * {@code [0.09003057, 0.24472847, 0.66524096]}.</p>
     *
     * <p>The operation preserves slice positions and axis order; it does not reduce rank or retain
     * a singleton reduction axis. The class contract above owns its numerical formula and profile
     * inheritance. Input eligibility, result construction, gradients, execution, and backend
     * support belong to later owning contracts.</p>
     */
    SOFTMAX,

    /**
     * Requests natural-log probabilities along the axis in {@link SoftmaxAttrs}.
     *
     * <p>For slice values {@code x_i}, the ideal output at position {@code i} is
     * {@code x_i - log(sum_j(exp(x_j)))}, which is mathematically the natural logarithm of the
     * corresponding {@link #SOFTMAX} output. For {@code [1, 2, 3]}, the ideal result is
     * approximately {@code [-2.40760596, -1.40760596, -0.40760596]}; exponentiating those values
     * yields the corresponding softmax probabilities.</p>
     *
     * <p>This is a distinct first-class semantic kind rather than an implicit softmax-plus-log
     * graph fragment. It preserves slice positions and axis order. The class contract above owns
     * its numerical formula and profile inheritance. Input eligibility, result construction,
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
