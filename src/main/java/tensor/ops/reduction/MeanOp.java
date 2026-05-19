package tensor.ops.reduction;

import operations.reduction.mean;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.layout.TensorLayoutTransform;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for {@code mean} reductions.
 */
public final class MeanOp {
    private MeanOp() {
    }

    public static Tensor build(Tensor input, int dimension) {
        return build(input, dimension, false);
    }

    public static Tensor build(Tensor input, int dimension, boolean keepDims) {
        ReductionSupport.requireFloatingInput(input, "mean");
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, input.getShape().length);
        Tensor out = TensorPrimitiveBuilder.unary(
                input,
                ReductionSupport.reduceShape(input.getShape(), normalizedDimension, keepDims),
                new mean(normalizedDimension, keepDims),
                "mean",
                input.getDataType()
        );
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor aligned = keepDims ? outGrad : outGrad.expandDims(normalizedDimension);
            Tensor grad = aligned.expand(input.getShape()).mul(1.0 / input.getShape()[normalizedDimension]);
            ReductionSupport.accumulateGradient(input, grad);
        });
        return out;
    }

    public static Tensor buildMasked(Tensor input, int dimension, Tensor mask) {
        ReductionSupport.requireFloatingInput(input, "masked mean");
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, input.getShapeUnsafe().length);
        Tensor alignedMask = ReductionSupport.alignMaskToShape(mask, input.getShapeUnsafe(), normalizedDimension, "masked mean");
        Tensor maskedInput = Tensor.where(alignedMask, input, Tensor.zerosLike(input));
        Tensor valid = Tensor.where(alignedMask, Tensor.onesLike(input), Tensor.zerosLike(input));
        return maskedInput.sum(normalizedDimension).div(valid.sum(normalizedDimension).clampMin(1.0d));
    }

    public static Tensor buildAll(Tensor input) {
        ReductionSupport.requireFloatingInput(input, "mean");
        Tensor out = TensorPrimitiveBuilder.unary(input, new int[]{1}, new mean(-1), "mean", input.getDataType());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor grad = outGrad.expand(input.getShape()).mul(1.0 / input.getFlatDataSize());
            ReductionSupport.accumulateGradient(input, grad);
        });
        return out;
    }
}
