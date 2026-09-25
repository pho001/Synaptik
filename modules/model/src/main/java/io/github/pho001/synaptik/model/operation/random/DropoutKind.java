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
 *
 * <p>The exact formula performs one abstract draw per logical element, keeps the exact BOOL mask,
 * advances the unsigned counter by the exact logical count modulo 2^64, publishes positive zero
 * for a dropped element, and computes a kept element as
 * {@code input * (1 / (1 - probability))}. Probability zero still advances state; an empty shape
 * draws zero values.</p>
 *
 * <p>Under the Model-owned numerical-profile contract, sampling membership, mask, state
 * transition, probability, guards, traversal, and dropped value remain exact in both profiles.
 * For {@code ACCELERATOR FLOAT32}, only a kept value's typed-one-minus-probability, typed-one
 * division, and input multiplication sites use DAZ/FTZ and one-round FLOAT32 operations. The mask
 * and next state cannot inherit arithmetic tolerance, and the composite gains no final-output
 * envelope. Non-FLOAT32 behavior stays strict. See the
 * <a href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#numerical-profiles">normative
 * numerical-profile contract</a>.</p>
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
