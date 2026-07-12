package io.github.pho001.synaptik.model.operation.attention;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Backend-independent semantic identity for one-output scaled dot-product attention.
 *
 * <p>One occurrence consumes ordered {@code [query, key, value]} or
 * {@code [query, key, value, mask]} inputs and produces only the attention output. Its attributes
 * preserve scale and causal eligibility. The kind expresses mathematical meaning and occurrence
 * structure, not decomposition, gradients, backend support, lowering, or execution.</p>
 */
public enum ScaledDotProductAttentionKind implements OperationKind {
    /** Scaled query/key scores, masked final-axis softmax, and weighted value aggregation. */
    SCALED_DOT_PRODUCT_ATTENTION;

    private static final List<OperationSignature> SIGNATURES = List.of(
            OperationSignature.inputRange(ScaledDotProductAttentionAttrs.class, 3, 4, 1));

    /**
     * Returns the exact three-to-four-input and one-output occurrence signature.
     *
     * @return stable immutable singleton list accepting only attention attributes
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
