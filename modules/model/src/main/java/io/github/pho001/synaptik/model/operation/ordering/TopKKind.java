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
 *
 * <p>Top-K compares stored represented values, including subnormals, and retains exact
 * {@code k}, axis mapping, NaN-last class, infinity and signed-zero order, largest/smallest
 * direction, stability, ties, sorted/unsorted placement, selected original payloads, and logical
 * indices. Selection applies no arithmetic DAZ, FTZ, epsilon, or finite tolerance. See the
 *  <a
 * href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#profile-free-numerical-semantics">Model
 * numerical-semantics contract</a>.</p>
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
