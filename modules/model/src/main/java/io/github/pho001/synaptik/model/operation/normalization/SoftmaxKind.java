package io.github.pho001.synaptik.model.operation.normalization;

import io.github.pho001.synaptik.model.operation.OperationKind;

/**
 * Identifies backend-independent softmax normalization semantics.
 *
 * <p>Both kinds have one logical input and preserve every logical input position. A normalization
 * slice contains the positions that differ only along the selected axis while every other logical
 * coordinate remains fixed. The already normalized, non-negative axis is carried by
 * {@link SoftmaxAttrs}; this enum stores no input, axis, shape, result, or graph-occurrence state.</p>
 *
 * <p>The valid family compositions pair either {@link #SOFTMAX} or {@link #LOG_SOFTMAX} with
 * {@link SoftmaxAttrs}. The generic
 * {@link io.github.pho001.synaptik.model.operation.Operation Operation} descriptor checks only
 * that its kind and attributes are non-null and does not enforce either family-specific pairing.</p>
 *
 * <p>These kinds define ideal mathematical meaning only. They do not define eligible data types,
 * result descriptors or provenance, a finite-precision algorithm, numerical edge-case policy,
 * gradients, compiler decomposition, storage, execution, or backend availability. Enum identity
 * supplies typed equality and hashing. Inherited {@link #name()} and {@link #toString()} text is
 * diagnostic only, not a serialization, parsing, registry, dispatch, or kernel contract.</p>
 */
public enum SoftmaxKind implements OperationKind {
    /**
     * Requests normalized probabilities along the axis in {@link SoftmaxAttrs}.
     *
     * <p>For slice values {@code x_i}, the ideal output at position {@code i} is
     * {@code exp(x_i) / sum_j(exp(x_j))}. The outputs are positive and sum to one across the
     * slice. For {@code [1, 2, 3]}, the ideal result is approximately
     * {@code [0.09003057, 0.24472847, 0.66524096]}.</p>
     *
     * <p>The operation preserves slice positions and axis order; it does not reduce rank or retain
     * a singleton reduction axis. Input eligibility, shape and result construction, numerical
     * policy, gradients, execution, and backend support belong to later owning contracts.</p>
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
     * graph fragment. It preserves slice positions and axis order. Input eligibility, shape and
     * result construction, stable finite-precision evaluation, gradients, compiler decomposition,
     * execution, and backend support belong to later owning contracts.</p>
     */
    LOG_SOFTMAX
}
