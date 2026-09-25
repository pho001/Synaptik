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
 * <p>Under the Model-owned numerical-profile contract, exact slice membership, {@code N},
 * epsilon, guards, affine mapping, and special classes remain unchanged. For
 * {@code ACCELERATOR FLOAT32}, the formula recurses through sum/count mean, subtraction,
 * {@code x*x}, sum/count variance, epsilon addition, square root, division, and optional scale
 * multiplication/bias addition. Every contributor participates once under the aggregate floor,
 * and the irreducible square-root site uses the inclusive ordered-binary32 distance ceiling of
 * five. The composite gains no final-output envelope; non-FLOAT32 behavior stays strict. See the
 * <a href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#numerical-profiles">normative
 * numerical-profile contract</a>.</p>
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
