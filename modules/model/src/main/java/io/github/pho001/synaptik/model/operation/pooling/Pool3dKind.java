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
     * contribute positive zero. BFLOAT16 and FLOAT32 accumulate and divide in FLOAT32, while
     * FLOAT64 uses FLOAT64; BFLOAT16 narrows once after division. Finite accumulation may be
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
