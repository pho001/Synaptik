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
 * <p>Sampling membership, mask, graph-random-number-generator state transition, probability,
 * guards, traversal, and dropped positive zero remain exact. A kept value uses
 * typed-one-minus-probability, typed-one division, then input multiplication. Only these named
 * floating arithmetic primitive inputs/results may use dtype-specific DAZ/FTZ for FLOAT32,
 * BFLOAT16, and FLOAT16, never FLOAT64. Homogeneous low arithmetic works in FLOAT32 with one final
 * low narrowing per value; the mask and next state gain no tolerance. See the <a
 * href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#profile-free-numerical-semantics">Model
 * numerical-semantics contract</a>.</p>
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
