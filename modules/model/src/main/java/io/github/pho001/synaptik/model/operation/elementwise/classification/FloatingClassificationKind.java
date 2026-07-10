package io.github.pho001.synaptik.model.operation.elementwise.classification;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies parameterless classifications of individual floating values.
 *
 * <p>Each kind consumes one floating Tensor value and produces one BOOL value at every input
 * position. The structural signature does not retain operand descriptors, infer result metadata,
 * inspect storage, evaluate classification, define gradients, or report backend support. Public
 * Tensor expression construction owns floating-input validation and the fixed non-gradient BOOL
 * result descriptor.</p>
 *
 * <p>These graph-visible classifications are distinct from trace diagnostics and from numeric
 * unary transforms whose result preserves the input data type. An {@link
 * io.github.pho001.synaptik.model.operation.Operation Operation} represents each kind with {@link
 * NoOperationAttrs#INSTANCE}.</p>
 */
public enum FloatingClassificationKind implements OperationKind {
    /**
     * Is true for finite normal, subnormal, and either signed-zero value, and false for both
     * infinities and every NaN.
     */
    IS_FINITE,

    /**
     * Is true only for NaN, independent of sign, quiet or signaling encoding, and payload.
     */
    IS_NAN,

    /** Is true only for positive or negative infinity and false for finite values and NaN. */
    IS_INF;

    private static final List<OperationSignature> SIGNATURES =
            List.of(OperationSignature.fixed(NoOperationAttrs.class, 1, 1));

    /**
     * Returns the parameterless one-input, one-output structural variant shared by this family.
     *
     * @return the stable immutable singleton signature list
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
