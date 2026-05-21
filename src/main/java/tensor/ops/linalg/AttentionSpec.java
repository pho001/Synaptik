package tensor.ops.linalg;

import tensor.DataType;
import tensor.Tensor;
import tensor.options.AttentionOptions;

record AttentionSpec(
        int[] scoresShape,
        int[] outShape,
        DataType outputType,
        double scale,
        int queryLength,
        int keyLength,
        boolean hasEffectiveMask
) {
    static AttentionSpec resolve(
            Tensor query,
            Tensor key,
            Tensor value,
            Tensor mask,
            AttentionOptions options
    ) {
        LinalgTensorRules.requireFloating(query, "attention query");
        LinalgTensorRules.requireFloating(key, "attention key");
        LinalgTensorRules.requireFloating(value, "attention value");
        if (options == null) {
            throw new IllegalArgumentException("attention options cannot be null");
        }
        if (mask != null && mask.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("attention mask must have BOOL dtype.");
        }

        int[] qShape = query.getShapeUnsafe();
        int[] kShape = key.getShapeUnsafe();
        int[] vShape = value.getShapeUnsafe();
        if (qShape.length < 2 || kShape.length < 2 || vShape.length < 2) {
            throw new IllegalArgumentException("scaledDotProductAttention requires rank >= 2 tensors.");
        }
        if (qShape[qShape.length - 1] != kShape[kShape.length - 1]) {
            throw new IllegalArgumentException("attention query/key head dimension mismatch.");
        }
        if (kShape[kShape.length - 2] != vShape[vShape.length - 2]) {
            throw new IllegalArgumentException("attention key/value sequence dimension mismatch.");
        }

        int[] scoresShape = resolveScoresShape(qShape, kShape);
        int[] outShape = resolveOutputShape(scoresShape, qShape, vShape);
        DataType outputType = LinalgTensorRules.promote(query.getDataType(), LinalgTensorRules.promote(key.getDataType(), value.getDataType()));
        double scale = options.resolveScale(qShape[qShape.length - 1]);
        return new AttentionSpec(
                scoresShape,
                outShape,
                outputType,
                scale,
                qShape[qShape.length - 2],
                kShape[kShape.length - 2],
                mask != null || options.causal()
        );
    }

    private static int[] resolveScoresShape(int[] qShape, int[] kShape) {
        int[] qBatch = java.util.Arrays.copyOf(qShape, qShape.length - 2);
        int[] kBatch = java.util.Arrays.copyOf(kShape, kShape.length - 2);
        int[] outBatch = LinalgTensorRules.broadcastLeadingShape(qBatch, kBatch, "attention batch dimensions are not broadcast-compatible.");
        int[] outShape = java.util.Arrays.copyOf(outBatch, outBatch.length + 2);
        outShape[outBatch.length] = qShape[qShape.length - 2];
        outShape[outBatch.length + 1] = kShape[kShape.length - 2];
        return outShape;
    }

    private static int[] resolveOutputShape(int[] scoresShape, int[] qShape, int[] vShape) {
        int[] scoresBatch = java.util.Arrays.copyOf(scoresShape, scoresShape.length - 2);
        int[] valueBatch = java.util.Arrays.copyOf(vShape, vShape.length - 2);
        int[] outBatch = LinalgTensorRules.broadcastLeadingShape(scoresBatch, valueBatch, "attention batch dimensions are not broadcast-compatible.");
        int[] outShape = java.util.Arrays.copyOf(outBatch, outBatch.length + 2);
        outShape[outBatch.length] = qShape[qShape.length - 2];
        outShape[outBatch.length + 1] = vShape[vShape.length - 1];
        return outShape;
    }
}
