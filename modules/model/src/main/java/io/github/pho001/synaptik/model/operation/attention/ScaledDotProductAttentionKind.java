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
