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
 * <p>Under the Model-owned numerical-profile contract, those guards, masks, scale, contributor
 * sets, special classes, output/weight slots, and saved weights remain exact. For
 * {@code ACCELERATOR FLOAT32}, scores use query/key multiply-contractions then scale
 * multiplication; an absent scale adds exact embedding-extent conversion, square root, and
 * typed-one division. Exact guards precede exponential/sum/division softmax sites, and outputs use
 * value/weight multiply-contractions. Each site uses its recursive floor and the composite gains
 * no final-output envelope. Every current non-FLOAT32 occurrence remains strict; the inactive
 * low-precision reservation changes none of them. See the
 * <a href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#numerical-profiles">normative
 * numerical-profile contract</a>.</p>
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
