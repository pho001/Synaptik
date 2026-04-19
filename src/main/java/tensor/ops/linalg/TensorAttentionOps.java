package tensor.ops.linalg;

import operations.Operation;
import operations.linalg.scaledDotProductAttention;
import operations.linalg.scaledDotProductAttentionBackward;
import operations.linalg.scaledDotProductAttentionWeights;
import operations.reduction.softmaxGrad;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorMetadata;
import tensor.TensorPrimitiveBuilder;
import tensor.options.AttentionOptions;

import java.util.List;

public final class TensorAttentionOps {
    private TensorAttentionOps() {
    }

    public static Tensor scaledDotProductAttention(
            Tensor query,
            Tensor key,
            Tensor value,
            AttentionOptions options
    ) {
        return scaledDotProductAttention(query, key, value, null, options);
    }

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

        List<Tensor> inputs = effectiveMask == null
                ? List.of(query, key, value)
                : List.of(query, key, value, effectiveMask);
        Operation op = new scaledDotProductAttention(spec.scale(), effectiveMask != null);
        Tensor out = TensorPrimitiveBuilder.nary(spec.outShape(), inputs, op, "scaledDotProductAttention", spec.outputType());
        Tensor backwardMask = effectiveMask;
        TensorInternalAccess.setBackwardFunction(out, () -> backwardScaledDotProductAttention(out, query, key, value, backwardMask, spec));
        return out;
    }

    private static void backwardScaledDotProductAttention(
            Tensor out,
            Tensor query,
            Tensor key,
            Tensor value,
            Tensor effectiveMask,
            AttentionSpec spec
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
                LinalgSupport.accumulateGradient(query, LinalgSupport.sumToShape(gradRaw, query.getShapeUnsafe()));
            }
            if (key.getRequiresGrad()) {
                Tensor gradRaw = attentionBackward(out, outGrad, rawKeyGradShape(outShape, key.getShapeUnsafe()), scaledDotProductAttentionBackward.OutputKind.KEY);
                LinalgSupport.accumulateGradient(key, LinalgSupport.sumToShape(gradRaw, key.getShapeUnsafe()));
            }
            if (value.getRequiresGrad()) {
                Tensor gradRaw = attentionBackward(out, outGrad, rawValueGradShape(outShape, value.getShapeUnsafe()), scaledDotProductAttentionBackward.OutputKind.VALUE);
                LinalgSupport.accumulateGradient(value, LinalgSupport.sumToShape(gradRaw, value.getShapeUnsafe()));
            }
            return;
        }

        int axis = spec.scoresShape().length - 1;
        Tensor weights = attentionWeights(out, spec.scoresShape());

        if (value.getRequiresGrad()) {
            Tensor gradRaw = LinalgSupport.transposeLastTwoAxes(weights).matmul(outGrad);
            LinalgSupport.accumulateGradient(value, LinalgSupport.sumToShape(gradRaw, value.getShapeUnsafe()));
        }

        if (!query.getRequiresGrad() && !key.getRequiresGrad()) {
            return;
        }

        Tensor dWeights = outGrad.matmul(LinalgSupport.transposeLastTwoAxes(value));
        Tensor dScores = softmaxGrad(weights, dWeights, axis);
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

    private static Tensor attentionWeights(Tensor attentionOut, int[] scoresShape) {
        Tensor weights = TensorPrimitiveBuilder.unaryNoGrad(
                attentionOut,
                scoresShape.clone(),
                new scaledDotProductAttentionWeights(),
                "attentionWeights",
                attentionOut.getDataType()
        );
        return weights;
    }

    private static Tensor attentionBackward(
            Tensor attentionOut,
            Tensor outGrad,
            int[] rawShape,
            scaledDotProductAttentionBackward.OutputKind outputKind
    ) {
        Tensor grad = TensorPrimitiveBuilder.binaryNoGrad(
                attentionOut,
                outGrad,
                rawShape.clone(),
                new scaledDotProductAttentionBackward(outputKind),
                "scaledDotProductAttentionBackward",
                attentionOut.getDataType()
        );
        return grad;
    }

    private static Tensor softmaxGrad(Tensor softmaxOut, Tensor outGrad, int dimension) {
        Tensor grad = TensorPrimitiveBuilder.binary(
                softmaxOut,
                outGrad,
                softmaxOut.getShapeUnsafe().clone(),
                new softmaxGrad(dimension),
                "softmaxGrad",
                outGrad.getDataType()
        );
        return grad;
    }

    private static boolean supportsLoweredBackward(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64;
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
}
