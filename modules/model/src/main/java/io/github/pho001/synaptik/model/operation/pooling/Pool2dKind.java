package io.github.pho001.synaptik.model.operation.pooling;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Backend-independent identities for two-dimensional NCHW pooling operations.
 *
 * <p>One {@link #MAX_POOL2D} occurrence consumes exactly one floating tensor and produces exactly
 * one tensor. The kind records the selected max-window meaning; it does not define indices,
 * gradients, algorithms, compiler support, backend capabilities, storage, or execution.</p>
 */
public enum Pool2dKind implements OperationKind {
    /**
     * NCHW maximum pooling with excluded padding and a literal floor or ceiling window grid.
     *
     * <p>NaN propagates, positive zero orders above negative zero, and equal candidates select the
     * first logical kernel sample. A window with no in-bounds sample produces negative infinity
     * in the input type.</p>
     */
    MAX_POOL2D;

    private static final List<OperationSignature> SIGNATURES = List.of(
            OperationSignature.fixed(MaxPool2dAttrs.class, 1, 1));

    /**
     * Returns the exact one-input and one-output max-pooling signature.
     *
     * @return stable immutable singleton list accepting only {@link MaxPool2dAttrs}
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
