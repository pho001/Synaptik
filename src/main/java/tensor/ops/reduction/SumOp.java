package tensor.ops.reduction;

import operations.Operation;
import operations.reduction.sum;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorLayoutTransform;
import tensor.TensorPrimitiveBuilder;

/**
 * Graph-building definition for {@code sum} reductions.
 */
public final class SumOp {
    private SumOp() {
    }

    public static Tensor build(Tensor input, int dimension) {
        return build(input, dimension, false);
    }

    public static Tensor build(Tensor input, int dimension, boolean keepDims) {
        ReductionSupport.requireFloatingInput(input, "sum");
        int[] shape = input.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, shape.length);
        Operation op = new sum(normalizedDimension, keepDims);
        int[] newShape = ReductionSupport.reduceShape(shape, normalizedDimension, keepDims);
        Tensor out = TensorPrimitiveBuilder.unary(input, newShape, op, "sum", input.getDataType());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor aligned = keepDims ? outGrad : outGrad.expandDims(normalizedDimension);
            ReductionSupport.accumulateGradient(input, aligned.expand(input.getShape()));
        });
        return out;
    }

    public static Tensor buildMasked(Tensor input, int dimension, Tensor mask) {
        ReductionSupport.requireFloatingInput(input, "masked sum");
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, input.getShapeUnsafe().length);
        Tensor alignedMask = ReductionSupport.alignMaskToShape(mask, input.getShapeUnsafe(), normalizedDimension, "masked sum");
        Tensor maskedInput = Tensor.where(alignedMask, input, Tensor.zerosLike(input));
        return maskedInput.sum(normalizedDimension);
    }

    public static Tensor buildAll(Tensor input) {
        ReductionSupport.requireFloatingInput(input, "sum");
        Tensor out = TensorPrimitiveBuilder.unary(input, new int[]{1}, new sum(-1), "sum", input.getDataType());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            ReductionSupport.accumulateGradient(input, outGrad.expand(input.getShape()));
        });
        return out;
    }
}
