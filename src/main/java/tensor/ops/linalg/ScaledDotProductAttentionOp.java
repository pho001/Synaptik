package tensor.ops.linalg;

import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.options.AttentionOptions;

/**
 * Graph-building definition for scaled dot-product attention.
 */
public final class ScaledDotProductAttentionOp {
    private ScaledDotProductAttentionOp() {
    }

    /**
     * Computes scaled dot-product attention without an explicit mask.
     *
     * @param query query tensor; must be non-null and floating numeric
     * @param key key tensor; must be non-null and floating numeric
     * @param value value tensor; must be non-null and floating numeric
     * @param options attention options; must be non-null
     * @return attention output tensor
     * @throws IllegalArgumentException if ranks, dimensions, dtypes, or options are invalid
     */
    public static Tensor build(
            Tensor query,
            Tensor key,
            Tensor value,
            AttentionOptions options
    ) {
        return build(query, key, value, null, options);
    }

    /**
     * Computes scaled dot-product attention with an optional boolean mask.
     *
     * <p>When {@code options.causal()} is true, a causal mask is combined with
     * the supplied mask using logical AND before attention scores are evaluated.</p>
     *
     * @param query query tensor; must be non-null and floating numeric
     * @param key key tensor; must be non-null and floating numeric
     * @param value value tensor; must be non-null and floating numeric
     * @param mask optional BOOL mask broadcastable to score shape; null means unmasked
     * @param options attention options; must be non-null
     * @return attention output tensor
     * @throws IllegalArgumentException if ranks, dimensions, dtypes, mask shape, or options are invalid
     */
    public static Tensor build(
            Tensor query,
            Tensor key,
            Tensor value,
            Tensor mask,
            AttentionOptions options
    ) {
        AttentionSpec spec = AttentionSpec.resolve(query, key, value, mask, options);

        Tensor effectiveMask = mask;
        if (options.causal()) {
            Tensor causalMask = AttentionSupport.createCausalMask(spec.scoresShape(), spec.queryLength(), spec.keyLength());
            effectiveMask = effectiveMask == null ? causalMask : effectiveMask.logicalAnd(causalMask);
        }
        if (effectiveMask != null) {
            effectiveMask = effectiveMask.expand(spec.scoresShape());
        }

        Tensor keyTransposed = LinalgSupport.transposeLastTwoAxes(key);
        Tensor scores = query.matmul(keyTransposed);
        Tensor scaled = Math.abs(spec.scale() - 1.0d) > 1e-12d
                ? scores.mul(spec.scale())
                : scores;
        Tensor logits = effectiveMask == null
                ? scaled
                : Tensor.where(
                        effectiveMask,
                        scaled,
                        Tensor.scalar(AttentionSupport.maskFillValue(spec.outputType()), spec.outputType())
                );
        Tensor weights = logits.softmax(logits.getShapeUnsafe().length - 1);
        Tensor out = weights.matmul(value);
        out.setLabel("scaledDotProductAttention");
        Tensor backwardMask = effectiveMask;
        TensorInternalAccess.setBackwardFunction(
                out,
                () -> AttentionSupport.backwardScaledDotProductAttention(out, query, key, value, backwardMask, weights, spec)
        );
        return out;
    }
}
