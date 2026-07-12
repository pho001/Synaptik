package io.github.pho001.synaptik.model.operation.pooling;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Backend-independent identities for two-dimensional NCHW pooling operations.
 *
 * <p>Each occurrence consumes exactly one floating tensor and produces exactly one tensor. Max
 * and average pooling retain distinct attribute types and numerical policies. These kinds do not
 * define gradients, algorithms, compiler support, backend capabilities, storage, or execution.</p>
 */
public enum Pool2dKind implements OperationKind {
    /**
     * NCHW maximum pooling with excluded padding and a literal floor or ceiling window grid.
     *
     * <p>NaN propagates, positive zero orders above negative zero, and equal candidates select the
     * first logical kernel sample. A window with no in-bounds sample produces negative infinity
     * in the input type.</p>
     */
    MAX_POOL2D,

    /**
     * NCHW average pooling with a fixed kernel-position divisor and literal floor or ceiling grid.
     *
     * <p>Every logical kernel position counts in the divisor. Out-of-bounds positions contribute
     * positive zero, so an all-padding window produces positive zero. BFLOAT16 and FLOAT32 use a
     * FLOAT32 accumulator and one final division; FLOAT64 uses FLOAT64. NaN propagates, opposing
     * infinities produce NaN, and one infinity sign is retained. Finite reassociation is
     * permitted, and exact zero is negative only when every contribution is an in-bounds negative
     * zero.</p>
     */
    AVERAGE_POOL2D;

    private static final List<OperationSignature> MAX_SIGNATURES = List.of(
            OperationSignature.fixed(MaxPool2dAttrs.class, 1, 1));
    private static final List<OperationSignature> AVERAGE_SIGNATURES = List.of(
            OperationSignature.fixed(AveragePool2dAttrs.class, 1, 1));

    /**
     * Returns the exact one-input and one-output signature for this pooling kind.
     *
     * @return stable immutable singleton list accepting only this kind's attribute class
     */
    @Override
    public List<OperationSignature> signatures() {
        return switch (this) {
            case MAX_POOL2D -> MAX_SIGNATURES;
            case AVERAGE_POOL2D -> AVERAGE_SIGNATURES;
        };
    }
}
