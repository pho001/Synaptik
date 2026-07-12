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
