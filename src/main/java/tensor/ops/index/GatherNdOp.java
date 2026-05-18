package tensor.ops.index;

import operations.index.ScatterReduction;
import operations.index.gatherNd;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;

/**
 * Graph-building definition for ONNX-style {@code gatherNd}.
 */
public final class GatherNdOp {
    private GatherNdOp() {
    }

    public static Tensor build(Tensor input, Tensor indices) {
        return build(input, indices, 0);
    }

    public static Tensor build(Tensor input, Tensor indices, int batchDims) {
        if (input == null || indices == null) {
            throw new IllegalArgumentException("gatherNd inputs cannot be null");
        }
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("gatherNd indices must be numeric integral values.");
        }
        int[] outputShape = IndexSupport.gatherNdOutputShape(input.getShapeUnsafe(), indices.getShapeUnsafe(), batchDims);
        Tensor out = TensorPrimitiveBuilder.binary(
                input,
                indices,
                outputShape,
                new gatherNd(batchDims),
                "gatherNd",
                input.getDataType()
        );
        out.setRequiresGrad(input.getRequiresGrad() && IndexSupport.isFloating(input.getDataType()));
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad() || !IndexSupport.isFloating(input.getDataType())) {
                return;
            }
            Tensor grad = ScatterNdOp.build(Tensor.zerosLike(input), indices, outGrad, ScatterReduction.ADD, batchDims);
            IndexSupport.accumulateGradient(input, grad);
        });
        return out;
    }
}
