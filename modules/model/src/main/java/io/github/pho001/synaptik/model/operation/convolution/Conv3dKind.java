package io.github.pho001.synaptik.model.operation.convolution;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Model identity for grouped three-dimensional cross-correlation in batch, channel, depth,
 * height, width (NCDHW) axis order.
 *
 * <p>One occurrence consumes ordered {@code [input, weight]} or
 * {@code [input, weight, bias]} tensors and produces exactly one output. The kind defines the
 * selected numerical meaning, not a reversed mathematical kernel, algorithm, decomposition,
 * gradient rule, compiler support, backend capability, lowering, storage, or execution route.</p>
 */
public enum Conv3dKind implements OperationKind {
    /**
     * Grouped NCDHW cross-correlation with optional per-output-channel bias.
     *
     * <p>FLOAT64 output accumulates in FLOAT64. FLOAT32 and BFLOAT16 output accumulate in
     * FLOAT32, with final BFLOAT16 conversion when selected. Reassociation and fused multiply-add
     * are permitted. Conceptual padding is positive zero and participates in ordinary IEEE-754
     * multiplication, including multiplication by infinity. An empty channel contraction begins
     * from positive zero before optional bias.</p>
     */
    CONV3D;

    private static final List<OperationSignature> SIGNATURES = List.of(
            OperationSignature.inputRange(Conv3dAttrs.class, 2, 3, 1));

    /**
     * Returns the exact two-to-three-input and one-output convolution signature.
     *
     * @return stable immutable singleton list accepting only {@link Conv3dAttrs}
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
