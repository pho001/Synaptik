package io.github.pho001.synaptik.model.operation.ordering;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies deterministic axis-wise top-K values-and-indices selection.
 *
 * <p>The kind records backend-independent selection meaning only. Its exact signature consumes
 * one Tensor and describes two ordered outputs: selected values at slot zero and their original
 * logical-axis indices at slot one. It selects no algorithm and provides no evaluation, gradient,
 * compiler, backend, runtime, or execution behavior.</p>
 */
public enum TopKKind implements OperationKind {
    /** Selects values and their logical input indices from one shared occurrence. */
    TOP_K;

    private static final List<OperationSignature> SIGNATURES =
            List.of(OperationSignature.fixed(TopKAttrs.class, 1, 2));

    /**
     * Returns the exact one-input, two-output top-K occurrence signature.
     *
     * @return the stable immutable signature list
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
