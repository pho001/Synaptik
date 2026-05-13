package tensor.ops.linalg;

import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorMetadata;
import tensor.options.AttentionOptions;

/**
 * Scaled dot-product attention operations.
 *
 * <p>Query, key, and value tensors must be floating numeric tensors whose last
 * dimensions satisfy the attention contract resolved by {@link AttentionSpec}.
 * Optional masks are BOOL tensors broadcastable to score shape. Methods build
 * graph tensors and do not mutate inputs.</p>
 */
public final class TensorAttentionOps {
    private TensorAttentionOps() {
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
    public static Tensor scaledDotProductAttention(
            Tensor query,
            Tensor key,
            Tensor value,
            AttentionOptions options
    ) {
        return scaledDotProductAttention(query, key, value, null, options);
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
    public static Tensor scaledDotProductAttention(
            Tensor query,
            Tensor key,
            Tensor value,
            Tensor mask,
            AttentionOptions options
    ) {
        AttentionSpec spec = AttentionSpec.resolve(query, key, value, mask, options);

        Tensor effectiveMask = mask;
        if (options.causal()) {
            Tensor causalMask = createCausalMask(spec.scoresShape(), spec.queryLength(), spec.keyLength());
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
                        Tensor.scalar(maskFillValue(spec.outputType()), spec.outputType())
                );
        Tensor weights = logits.softmax(logits.getShapeUnsafe().length - 1);
        Tensor out = weights.matmul(value);
        out.setLabel("scaledDotProductAttention");
        Tensor backwardMask = effectiveMask;
        TensorInternalAccess.setBackwardFunction(out, () -> backwardScaledDotProductAttention(out, query, key, value, backwardMask, weights, spec));
        return out;
    }

    private static void backwardScaledDotProductAttention(
            Tensor out,
            Tensor query,
            Tensor key,
            Tensor value,
            Tensor effectiveMask,
            Tensor weights,
            AttentionSpec spec
    ) {
        Tensor outGrad = out.getGradient();
        if (outGrad == null) {
            return;
        }

        int axis = spec.scoresShape().length - 1;

        if (value.getRequiresGrad()) {
            Tensor gradRaw = LinalgSupport.transposeLastTwoAxes(weights).matmul(outGrad);
            LinalgSupport.accumulateGradient(value, LinalgSupport.sumToShape(gradRaw, value.getShapeUnsafe()));
        }

        if (!query.getRequiresGrad() && !key.getRequiresGrad()) {
            return;
        }

        Tensor dWeights = outGrad.matmul(LinalgSupport.transposeLastTwoAxes(value));
        Tensor dot = dWeights.mul(weights).sum(axis, true);
        Tensor dScores = weights.mul(dWeights.sub(dot));
        if (effectiveMask != null) {
            dScores = Tensor.where(effectiveMask, dScores, Tensor.zerosLike(dScores));
        }
        if (Math.abs(spec.scale() - 1.0d) > 1e-12d) {
            dScores = dScores.mul(spec.scale());
        }

        if (query.getRequiresGrad()) {
            Tensor gradRaw = dScores.matmul(key);
            LinalgSupport.accumulateGradient(query, LinalgSupport.sumToShape(gradRaw, query.getShapeUnsafe()));
        }
        if (key.getRequiresGrad()) {
            Tensor gradRaw = LinalgSupport.transposeLastTwoAxes(dScores).matmul(query);
            LinalgSupport.accumulateGradient(key, LinalgSupport.sumToShape(gradRaw, key.getShapeUnsafe()));
        }
    }

    private static Tensor createCausalMask(int[] scoresShape, int queryLen, int keyLen) {
        int flatSize = 1;
        for (int dim : scoresShape) {
            flatSize *= dim;
        }
        byte[] mask = new byte[flatSize];
        int rank = scoresShape.length;
        int[] denseStrides = TensorMetadata.computeStrides(scoresShape);
        int prefixSize = 1;
        for (int i = 0; i < rank - 2; i++) {
            prefixSize *= scoresShape[i];
        }
        for (int prefix = 0; prefix < prefixSize; prefix++) {
            int prefixOffset = prefix * queryLen * keyLen;
            for (int q = 0; q < queryLen; q++) {
                for (int k = 0; k < keyLen; k++) {
                    mask[prefixOffset + q * keyLen + k] = (byte) (k <= q ? 1 : 0);
                }
            }
        }
        return new Tensor(mask, scoresShape.clone(), denseStrides, null, "causal_mask", DataType.BOOL);
    }

    private static double maskFillValue(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> -1.0e30d;
            case FLOAT32 -> -1.0e9d;
            case BFLOAT16 -> -1.0e30d;
            case INT32, INT64, BOOL -> throw new IllegalArgumentException("attention mask fill requires floating dtype.");
        };
    }
}
