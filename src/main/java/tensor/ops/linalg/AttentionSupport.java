package tensor.ops.linalg;

import tensor.DataType;
import tensor.Tensor;
import tensor.TensorMetadata;
import tensor.autograd.GradientContext;

final class AttentionSupport {
    private AttentionSupport() {
    }

    static void backwardScaledDotProductAttention(
            Tensor out,
            Tensor query,
            Tensor key,
            Tensor value,
            Tensor effectiveMask,
            Tensor weights,
            AttentionSpec spec,
            GradientContext context
    ) {
        Tensor outGrad = out.getGradient();
        if (outGrad == null) {
            return;
        }

        int axis = spec.scoresShape().length - 1;

        if (value.getRequiresGrad()) {
            Tensor gradRaw = LinalgSupport.transposeLastTwoAxes(weights).matmul(outGrad);
            context.accumulate(value, LinalgSupport.sumToShape(gradRaw, value.getShapeUnsafe()));
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
            context.accumulate(query, LinalgSupport.sumToShape(gradRaw, query.getShapeUnsafe()));
        }
        if (key.getRequiresGrad()) {
            Tensor gradRaw = LinalgSupport.transposeLastTwoAxes(dScores).matmul(query);
            context.accumulate(key, LinalgSupport.sumToShape(gradRaw, key.getShapeUnsafe()));
        }
    }

    static Tensor createCausalMask(int[] scoresShape, int queryLen, int keyLen) {
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

    static double maskFillValue(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> -1.0e30d;
            case FLOAT32 -> -1.0e9d;
            case BFLOAT16 -> -1.0e30d;
            case INT32, INT64, BOOL -> throw new IllegalArgumentException("attention mask fill requires floating dtype.");
        };
    }
}
