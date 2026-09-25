package io.github.pho001.synaptik.model.operation.random;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent explicit graph RNG state semantics.
 *
 * <p>This family records state in Tensor-expression provenance. It does not select a random
 * algorithm, allocate state or storage, sample values, find a service, initialize runtime state,
 * identify a kernel, or promise a stable serialized token.</p>
 *
 * <p>The Model-owned numerical-profile contract leaves graph RNG state exact under
 * {@code STRICT_IEEE} and {@code ACCELERATOR}: raw words, key/counter meaning, occurrence
 * identity, and every explicit state transition remain unchanged. Numerical approximation cannot
 * alter state, draws, masks, or traversal. See the
 * <a href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#numerical-profiles">normative
 * numerical-profile contract</a>.</p>
 */
public enum GraphRngKind implements OperationKind {
    /**
     * Records one explicit key/counter state as a zero-input, one-output producer.
     *
     * <p>The associated {@link GraphRngStateAttrs} contains the exact two raw state words. The
     * output descriptor and opaque public wrapper are established by the Tensor-expression
     * construction boundary.</p>
     */
    INITIAL_STATE;

    private static final List<OperationSignature> SIGNATURES =
            List.of(OperationSignature.fixed(GraphRngStateAttrs.class, 0, 1));

    /**
     * Returns the fixed zero-input, one-output state-initializer signature.
     *
     * @return the stable immutable singleton signature list
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
