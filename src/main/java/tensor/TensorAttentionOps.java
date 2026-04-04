package tensor;

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
            scores = Tensor.where(
                    effectiveMask,
                    scores,
                    Tensor.scalar(maskFillValue(scores.getDataType()), scores.getDataType())
            );
        }

        Tensor weights = scores.softmax(scores.getShapeUnsafe().length - 1);
        Tensor out = weights.matmul(value);
        out.setLabel("scaledDotProductAttention");
        return out;
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
            case FLOAT16 -> -65504.0d;
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
}
