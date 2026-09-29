package io.github.pho001.synaptik.model.operation.pooling;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Backend-independent identities for three-dimensional NCDHW pooling operations.
 *
 * <p>Each occurrence consumes exactly one floating tensor and produces exactly one tensor. Max
 * and average pooling retain distinct attribute types and numerical policies. These kinds define
 * Model semantics only; they do not define gradients, compiler adoption, algorithms, backend
 * capabilities, storage, lowering, or execution.</p>
 *
 * <p>Under the Model-owned numerical-profile contract, max pooling remains exact/discrete in both
 * profiles: window geometry, eligibility, traversal, NaN dominance, signed-zero order, first-tie
 * winner, empty identity, and selected original payload are unchanged. For
 * {@code ACCELERATOR FLOAT32}, average pooling preserves every kernel position and its exact
 * divisor while allowing any binary tree, per-step FLOAT32 rounding, DAZ/FTZ, and only
 * corresponding multiply/add fusion. It may not drop padding or another contributor and gains no
 * final-output envelope. Every current non-FLOAT32 occurrence remains strict; the inactive
 * low-precision reservation changes none of them. See the
 * <a href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#numerical-profiles">normative
 * numerical-profile contract</a>.</p>
 */
public enum Pool3dKind implements OperationKind {
    /**
     * NCDHW maximum pooling with excluded padding and literal floor or ceiling window grids.
     *
     * <p>Logical samples are ordered by increasing depth, then height, then width. NaN dominates,
     * positive zero orders above negative zero, and equal candidates retain the first eligible
     * sample. A window with no in-bounds sample produces negative infinity in the input type.</p>
     */
    MAX_POOL3D,

    /**
     * NCDHW average pooling with a fixed kernel-position divisor and literal floor or ceiling
     * grids.
     *
     * <p>Every logical kernel position counts in the divisor, and out-of-bounds positions
     * contribute positive zero. FLOAT16, BFLOAT16, and FLOAT32 accumulate and divide in FLOAT32,
     * while FLOAT64 uses FLOAT64; low results narrow once after division. Finite accumulation may
     * reassociated. NaN propagates, opposing infinities produce NaN, and an exact-zero result is
     * negative only when every divisor position is an in-bounds negative zero.</p>
     */
    AVERAGE_POOL3D;

    private static final List<OperationSignature> MAX_SIGNATURES = List.of(
            OperationSignature.fixed(MaxPool3dAttrs.class, 1, 1));
    private static final List<OperationSignature> AVERAGE_SIGNATURES = List.of(
            OperationSignature.fixed(AveragePool3dAttrs.class, 1, 1));

    /**
     * Returns the exact one-input and one-output signature for this pooling kind.
     *
     * @return stable immutable singleton list accepting only this kind's attribute class
     */
    @Override
    public List<OperationSignature> signatures() {
        return switch (this) {
            case MAX_POOL3D -> MAX_SIGNATURES;
            case AVERAGE_POOL3D -> AVERAGE_SIGNATURES;
        };
    }
}
