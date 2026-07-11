package io.github.pho001.synaptik.model.operation.random;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Backend-independent semantic kind for training dropout with explicit graph RNG state.
 *
 * <p>One occurrence consumes ordered {@code [input, state]} Tensor positions and describes ordered
 * {@code [output, keep mask, next state]} positions. The keep mask is an auxiliary compiler-facing
 * result rather than a public dropout result. This kind performs no sampling or execution and
 * contains no backend-support or gradient metadata.</p>
 */
public enum DropoutKind implements OperationKind {
    /** Training-only inverted dropout with one non-public auxiliary BOOL keep-mask output. */
    DROPOUT;

    private static final List<OperationSignature> SIGNATURES =
            List.of(OperationSignature.fixed(DropoutAttrs.class, 2, 3));

    /**
     * Returns the exact two-input, three-output dropout occurrence signature.
     *
     * @return stable immutable singleton signature list accepting only {@link DropoutAttrs}
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
