package io.github.pho001.synaptik.model.operation.attention;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Backend-independent semantic identity for scaled dot-product attention.
 *
 * <p>One occurrence consumes ordered {@code [query, key, value]} or
 * {@code [query, key, value, mask]} inputs. A conventional occurrence produces only the attention
 * output at slot zero; an explicitly requested two-output occurrence additionally produces the
 * normalized attention weights at slot one. Its attributes preserve scale and causal eligibility.
 * The kind expresses mathematical meaning and occurrence structure, not decomposition, gradients,
 * backend support, lowering, or execution.</p>
 *
 * <p>The exact formula forms every eligible score from a query/key dot product and exact scale,
 * applies the exact Boolean/causal eligibility guard, uses the documented guarded final-axis
 * softmax, and weights value rows with every eligible contributor exactly once. No eligible
 * position produces positive-zero output; all eligible negative infinities do likewise; eligible
 * NaN propagates; positive-infinity ties split unit weight equally; excluded score/value special
 * values never enter arithmetic.</p>
 *
 * <p>Mask and causal guards, scale derivation, contributor membership, and output placement remain
 * exact. Scores use query/key multiply-contractions and scale multiplication; an absent scale uses
 * exact embedding extent conversion, SQRT, and typed-one division. Eligible scores use literal EXP,
 * an all-contributors-once sum, and division; output rows use value-weight multiply-contractions.
 * Guarded no-eligible, all-negative-infinity, and positive-infinity-tie cases retain their distinct
 * exact results. Only named arithmetic primitive inputs/results may use dtype-specific DAZ/FTZ for
 * FLOAT32, BFLOAT16, and FLOAT16, never FLOAT64. Homogeneous low arithmetic works and accumulates
 * in FLOAT32 with one final low narrowing per output; no blanket attention tolerance applies. See
 * the <a
 * href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#profile-free-numerical-semantics">Model
 * numerical-semantics contract</a>.</p>
 */
public enum ScaledDotProductAttentionKind implements OperationKind {
    /** Scaled query/key scores, masked final-axis softmax, and weighted value aggregation. */
    SCALED_DOT_PRODUCT_ATTENTION;

    private static final List<OperationSignature> SIGNATURES = List.of(new OperationSignature(
            ScaledDotProductAttentionAttrs.class, 3, 4, 1, 2));

    /**
     * Returns the exact three-to-four-input and one-through-two-output occurrence signature.
     *
     * @return stable immutable singleton list accepting only attention attributes
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
