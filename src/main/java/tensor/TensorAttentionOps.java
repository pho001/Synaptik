package tensor;

import operations.softmaxGrad;
import operations.scaledDotProductAttention;
import operations.scaledDotProductAttentionBackward;
import operations.scaledDotProductAttentionWeights;

import java.util.List;

final class TensorAttentionOps {
    private TensorAttentionOps() {
    }

    static Tensor scaledDotProductAttention(
            Tensor query,
            Tensor key,
            Tensor value,
            AttentionOptions options
    ) {
        return scaledDotProductAttention(query, key, value, null, options);
    }

    static Tensor scaledDotProductAttention(
            Tensor query,
            Tensor key,
            Tensor value,
            Tensor mask,
            AttentionOptions options
    ) {
        requireFloating(query, "attention query");
        requireFloating(key, "attention key");
        requireFloating(value, "attention value");
        if (options == null) {
            throw new IllegalArgumentException("attention options cannot be null");
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

        Tensor keyT = swapLastTwoAxes(key);
        Tensor scores = query.matmul(keyT);
        double scale = options.resolveScale(qShape[qShape.length - 1]);
        if (Math.abs(scale - 1.0d) > 1e-12d) {
            scores = scores.mul(scale);
        }

        Tensor effectiveMask = null;
        if (mask != null) {
            if (mask.getDataType() != DataType.BOOL) {
                throw new IllegalArgumentException("attention mask must have BOOL dtype.");
            }
            effectiveMask = mask;
        }
        if (options.causal()) {
            Tensor causalMask = createCausalMask(scores.getShapeUnsafe(), qShape[qShape.length - 2], kShape[kShape.length - 2]);
            effectiveMask = effectiveMask == null ? causalMask : effectiveMask.logicalAnd(causalMask);
        }
        if (effectiveMask != null) {
            effectiveMask = effectiveMask.expand(scores.getShapeUnsafe());
        }

        DataType outputType = promote(query.getDataType(), promote(key.getDataType(), value.getDataType()));
        int[] outShape = resolveAttentionOutputShape(qShape, kShape, vShape);
        List<Tensor> inputs = effectiveMask == null
                ? List.of(query, key, value)
                : List.of(query, key, value, effectiveMask);
        Tensor out = new Tensor(
                outShape,
                inputs,
                new scaledDotProductAttention(scale, effectiveMask != null),
                "scaledDotProductAttention",
                outputType
        );
        Tensor backwardMask = effectiveMask;
        out.setBackwardFunction(() -> backwardScaledDotProductAttention(out, query, key, value, backwardMask, scale));
        out.setLabel("scaledDotProductAttention");
        return out;
    }

    private static void backwardScaledDotProductAttention(
            Tensor out,
            Tensor query,
            Tensor key,
            Tensor value,
            Tensor effectiveMask,
            double scale
    ) {
        Tensor outGrad = out.getGradient();
        if (outGrad == null) {
            return;
        }

        int[] qShape = query.getShapeUnsafe();
        int[] kShape = key.getShapeUnsafe();
        int[] outShape = out.getShapeUnsafe();

        if (supportsLoweredBackward(out.getDataType())) {
            if (query.getRequiresGrad()) {
                Tensor gradRaw = attentionBackward(out, outGrad, rawQueryGradShape(outShape, qShape), scaledDotProductAttentionBackward.OutputKind.QUERY);
                accumulateGradient(query, TensorBroadcastOps.sumToShape(gradRaw, query.getShapeUnsafe()));
            }
            if (key.getRequiresGrad()) {
                Tensor gradRaw = attentionBackward(out, outGrad, rawKeyGradShape(outShape, key.getShapeUnsafe()), scaledDotProductAttentionBackward.OutputKind.KEY);
                accumulateGradient(key, TensorBroadcastOps.sumToShape(gradRaw, key.getShapeUnsafe()));
            }
            if (value.getRequiresGrad()) {
                Tensor gradRaw = attentionBackward(out, outGrad, rawValueGradShape(outShape, value.getShapeUnsafe()), scaledDotProductAttentionBackward.OutputKind.VALUE);
                accumulateGradient(value, TensorBroadcastOps.sumToShape(gradRaw, value.getShapeUnsafe()));
            }
            return;
        }

        int[] scoresShape = resolveAttentionScoresShape(qShape, kShape);
        int axis = scoresShape.length - 1;
        Tensor weights = attentionWeights(out, scoresShape);

        if (value.getRequiresGrad()) {
            Tensor gradRaw = swapLastTwoAxes(weights).matmul(outGrad);
            accumulateGradient(value, TensorBroadcastOps.sumToShape(gradRaw, value.getShapeUnsafe()));
        }

        if (!query.getRequiresGrad() && !key.getRequiresGrad()) {
            return;
        }

        Tensor dWeights = outGrad.matmul(swapLastTwoAxes(value));
        Tensor dScores = softmaxGrad(weights, dWeights, axis);
        if (effectiveMask != null) {
            dScores = Tensor.where(effectiveMask, dScores, Tensor.zerosLike(dScores));
        }
        if (Math.abs(scale - 1.0d) > 1e-12d) {
            dScores = dScores.mul(scale);
        }

        if (query.getRequiresGrad()) {
            Tensor gradRaw = dScores.matmul(key);
            accumulateGradient(query, TensorBroadcastOps.sumToShape(gradRaw, query.getShapeUnsafe()));
        }
        if (key.getRequiresGrad()) {
            Tensor gradRaw = swapLastTwoAxes(dScores).matmul(query);
            accumulateGradient(key, TensorBroadcastOps.sumToShape(gradRaw, key.getShapeUnsafe()));
        }
    }

    private static Tensor attentionWeights(Tensor attentionOut, int[] scoresShape) {
        Tensor weights = new Tensor(
                scoresShape.clone(),
                List.of(attentionOut),
                new scaledDotProductAttentionWeights(),
                "attentionWeights",
                attentionOut.getDataType()
        );
        weights.setRequiresGrad(false);
        return weights;
    }

    private static Tensor attentionBackward(
            Tensor attentionOut,
            Tensor outGrad,
            int[] rawShape,
            scaledDotProductAttentionBackward.OutputKind outputKind
    ) {
        Tensor grad = new Tensor(
                rawShape.clone(),
                List.of(attentionOut, outGrad),
                new scaledDotProductAttentionBackward(outputKind),
                "scaledDotProductAttentionBackward",
                attentionOut.getDataType()
        );
        grad.setRequiresGrad(false);
        return grad;
    }

    private static Tensor softmaxGrad(Tensor softmaxOut, Tensor outGrad, int dimension) {
        if (softmaxOut == null || outGrad == null) {
            throw new IllegalArgumentException("softmaxGrad inputs cannot be null");
        }
        Tensor out = new Tensor(
                softmaxOut.getShapeUnsafe().clone(),
                List.of(softmaxOut, outGrad),
                new softmaxGrad(dimension),
                "softmaxGrad",
                outGrad.getDataType()
        );
        out.setRequiresGrad(softmaxOut.getRequiresGrad() || outGrad.getRequiresGrad());
        return out;
    }

    private static boolean supportsLoweredBackward(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64;
    }

    private static Tensor swapLastTwoAxes(Tensor tensor) {
        int rank = tensor.getShapeUnsafe().length;
        if (rank == 2) {
            return tensor.transpose();
        }
        int[] axes = new int[rank];
        for (int i = 0; i < rank; i++) {
            axes[i] = i;
        }
        int tmp = axes[rank - 1];
        axes[rank - 1] = axes[rank - 2];
        axes[rank - 2] = tmp;
        return tensor.permute(axes);
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
            case INT32, BOOL -> throw new IllegalArgumentException("attention mask fill requires floating dtype.");
        };
    }

    private static void requireFloating(Tensor tensor, String name) {
        if (tensor == null) {
            throw new IllegalArgumentException(name + " cannot be null");
        }
        if (tensor.getDataType() == DataType.BOOL || tensor.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException(name + " must use a floating dtype.");
        }
    }

    private static int[] resolveAttentionScoresShape(int[] qShape, int[] kShape) {
        int[] qBatch = java.util.Arrays.copyOf(qShape, qShape.length - 2);
        int[] kBatch = java.util.Arrays.copyOf(kShape, kShape.length - 2);
        int[] outBatch = broadcastLeadingShape(qBatch, kBatch);
        int[] outShape = java.util.Arrays.copyOf(outBatch, outBatch.length + 2);
        outShape[outBatch.length] = qShape[qShape.length - 2];
        outShape[outBatch.length + 1] = kShape[kShape.length - 2];
        return outShape;
    }

    private static int[] resolveAttentionOutputShape(int[] qShape, int[] kShape, int[] vShape) {
        int[] scoresShape = resolveAttentionScoresShape(qShape, kShape);
        int[] scoresBatch = java.util.Arrays.copyOf(scoresShape, scoresShape.length - 2);
        int[] valueBatch = java.util.Arrays.copyOf(vShape, vShape.length - 2);
        int[] outBatch = broadcastLeadingShape(scoresBatch, valueBatch);
        int[] outShape = java.util.Arrays.copyOf(outBatch, outBatch.length + 2);
        outShape[outBatch.length] = qShape[qShape.length - 2];
        outShape[outBatch.length + 1] = vShape[vShape.length - 1];
        return outShape;
    }

    private static int[] rawQueryGradShape(int[] outShape, int[] queryShape) {
        int[] raw = outShape.clone();
        raw[raw.length - 2] = queryShape[queryShape.length - 2];
        raw[raw.length - 1] = queryShape[queryShape.length - 1];
        return raw;
    }

    private static int[] rawKeyGradShape(int[] outShape, int[] keyShape) {
        int[] raw = outShape.clone();
        raw[raw.length - 2] = keyShape[keyShape.length - 2];
        raw[raw.length - 1] = keyShape[keyShape.length - 1];
        return raw;
    }

    private static int[] rawValueGradShape(int[] outShape, int[] valueShape) {
        int[] raw = outShape.clone();
        raw[raw.length - 2] = valueShape[valueShape.length - 2];
        raw[raw.length - 1] = valueShape[valueShape.length - 1];
        return raw;
    }

    private static int[] broadcastLeadingShape(int[] first, int[] second) {
        int rank = Math.max(first.length, second.length);
        int[] out = new int[rank];
        for (int i = 0; i < rank; i++) {
            int a = i < rank - first.length ? 1 : first[i - (rank - first.length)];
            int b = i < rank - second.length ? 1 : second[i - (rank - second.length)];
            if (a != b && a != 1 && b != 1) {
                throw new IllegalArgumentException("attention batch dimensions are not broadcast-compatible.");
            }
            out[i] = Math.max(a, b);
        }
        return out;
    }

    private static DataType promote(DataType left, DataType right) {
        if (left == DataType.FLOAT64 || right == DataType.FLOAT64) {
            return DataType.FLOAT64;
        }
        if (left == DataType.FLOAT32 || right == DataType.FLOAT32) {
            return DataType.FLOAT32;
        }
        return DataType.BFLOAT16;
    }

    private static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            input.setGradient(gradientDelta);
        } else {
            input.setGradient(input.getGradient().add(gradientDelta));
        }
    }
}
