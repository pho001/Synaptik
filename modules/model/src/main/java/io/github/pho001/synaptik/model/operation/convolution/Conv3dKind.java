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
 *
 * <p>Under the Model-owned graph numerical-profile contract, {@code STRICT_IEEE} retains this
 * exact convolution formula. For {@code ACCELERATOR FLOAT32}, groups, stride, dilation, padding,
 * layout mapping, padding values, contributor membership, and output placement remain exact.
 * Every declared term participates exactly once; each cell may use any binary tree, per-step
 * FLOAT32 rounding, DAZ/FTZ, and corresponding product/add fusion. It may not drop, duplicate,
 * invent, pretruncate, or replace a term, and convolution gains no MATMUL-specific final-zero
 * freedom. This is recursive construction freedom, not a final-output tolerance. Every current
 * non-FLOAT32 occurrence remains strict; the inactive low-precision reservation changes none of
 * them. See the
 * <a href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#numerical-profiles">normative
 * numerical-profile contract</a>.</p>
 */
public enum Conv3dKind implements OperationKind {
    /**
     * Grouped NCDHW cross-correlation with optional per-output-channel bias.
     *
     * <p>FLOAT64 output accumulates in FLOAT64. FLOAT32, FLOAT16, and BFLOAT16 output accumulate
     * in FLOAT32, with one final low-type conversion when selected. Reassociation and fused
     * multiply-add are permitted. Conceptual padding is positive zero and participates in ordinary
     * IEEE-754
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
