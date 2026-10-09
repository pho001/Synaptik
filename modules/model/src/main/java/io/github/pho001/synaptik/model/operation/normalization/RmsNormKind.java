package io.github.pho001.synaptik.model.operation.normalization;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent root-mean-square normalization over an exact trailing
 * {@code Shape}.
 *
 * <p>The operation records an uncentered mean-square denominator with epsilon added inside the
 * square root. Its one safe signature accepts either the input alone or ordered
 * {@code [input, scale]} operands and produces exactly one output. This semantic identity does
 * not evaluate values, select an algorithm, create saved statistics or gradients, or claim
 * compiler, backend, runtime, or execution support.</p>
 *
 * <p>The exact formula is
 * {@code normalized = x / sqrt(sum(x * x) / N + epsilon)}, followed when present by
 * {@code normalized * scale}. NaN, infinity, signed-zero, overflow, and empty-result behavior
 * follow those named sites; an empty result evaluates no divisor.</p>
 *
 * <p>Exact slice membership, count, epsilon, guards, and scale mapping remain fixed. The formula
 * recurses through x*x, sum/count, epsilon addition, SQRT, division, and optional scale
 * multiplication. Every contributor participates once; only named floating arithmetic primitive
 * inputs/results may use dtype-specific DAZ/FTZ for FLOAT32, BFLOAT16, and FLOAT16, never FLOAT64.
 * Homogeneous low arithmetic uses FLOAT32 working and accumulator values and one final low
 * narrowing per output. The composite has no blanket result tolerance. See the <a
 * href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#profile-free-numerical-semantics">Model
 * numerical-semantics contract</a>.</p>
 */
public enum RmsNormKind implements OperationKind {
    /**
     * Requests normalization by the root of the uncentered population mean square plus epsilon,
     * optionally followed by explicit elementwise scale.
     */
    RMS_NORM;

    private static final List<OperationSignature> SIGNATURES = List.of(
            OperationSignature.inputRange(RmsNormAttrs.class, 1, 2, 1));

    /**
     * Returns the stable one-or-two-input RMS-normalization signature.
     *
     * @return immutable singleton signature accepting ordered {@code [input]} or
     *     {@code [input, scale]} and exactly one output; never {@code null}
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
