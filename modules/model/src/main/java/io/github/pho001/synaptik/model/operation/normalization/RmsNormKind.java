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
