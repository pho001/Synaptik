package io.github.pho001.synaptik.model.operation.normalization;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent layer normalization over an exact trailing {@code Shape}.
 *
 * <p>The operation records population-variance standardization with epsilon added inside the
 * square root. It has one no-affine signature and one signature with explicit scale and bias; both
 * produce exactly one output. This semantic identity does not calculate statistics, select a
 * numerical algorithm, create saved statistics or gradients, or claim executable backend
 * support.</p>
 *
 * <p>The exact formula is {@code mean = sum(x) / N},
 * {@code variance = sum((x - mean)^2) / N}, and
 * {@code standardized = (x - mean) / sqrt(variance + epsilon)}, followed when present by
 * {@code standardized * scale + bias}. NaN or infinity anywhere in a nonempty slice makes every
 * standardized value NaN; a finite constant slice produces exact positive-zero standardized
 * values; an empty result evaluates no slice.</p>
 *
 * <p>Exact slice membership, count, epsilon, guards, affine mapping, and special classes remain
 * fixed. The formula recurses through sum/count mean, subtraction, x*x, sum/count variance, epsilon
 * addition, SQRT, division, and optional scale multiplication and bias addition. Every contributor
 * participates once; only named floating arithmetic primitive inputs/results may use dtype-specific
 * DAZ/FTZ for FLOAT32, BFLOAT16, and FLOAT16, never FLOAT64. Homogeneous low arithmetic uses
 * FLOAT32 working and accumulator values and one final low narrowing per output. The composite has
 * no blanket result tolerance. See the <a
 * href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#profile-free-numerical-semantics">Model
 * numerical-semantics contract</a>.</p>
 */
public enum LayerNormKind implements OperationKind {
    /**
     * Requests population-variance normalization of non-empty trailing slices, optionally
     * followed by an explicit elementwise scale and bias affine transform.
     */
    LAYER_NORM;

    private static final List<OperationSignature> SIGNATURES = List.of(
            OperationSignature.fixed(LayerNormAttrs.class, 1, 1),
            OperationSignature.fixed(AffineLayerNormAttrs.class, 3, 1));

    /**
     * Returns the ordered no-affine and affine layer-normalization signatures.
     *
     * @return the stable immutable two-element signature list, first no-affine with one input and
     *     then affine with ordered inputs {@code [input, scale, bias]}; never {@code null}
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
