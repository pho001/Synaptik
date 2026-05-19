package tensor.ops.index;

import operations.index.scatterAdd;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.layout.TensorLayoutTransform;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for gather-shaped {@code scatterAdd}.
 */
public final class ScatterAddOp {
    private ScatterAddOp() {
    }

    public static Tensor build(Tensor base, Tensor indices, Tensor src, int dimension) {
        if (base == null || indices == null || src == null) {
            throw new IllegalArgumentException("scatterAdd inputs cannot be null");
        }
        if (base.getDataType() == DataType.BOOL || src.getDataType() == DataType.BOOL
                || base.getDataType() == DataType.INT32 || src.getDataType() == DataType.INT32
                || base.getDataType() == DataType.INT64 || src.getDataType() == DataType.INT64) {
            throw new IllegalArgumentException("scatterAdd requires floating numeric base and source tensors.");
        }
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("scatterAdd indices must be numeric integral values.");
        }
        if (base.getDataType() != src.getDataType()) {
            throw new IllegalArgumentException("scatterAdd requires base and source tensors to have matching dtypes.");
        }
        int[] baseShape = base.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, baseShape.length);
        int[] expectedSrcShape = IndexSupport.reduceShape(baseShape, normalizedDimension);
        IndexSupport.validateGatherIndicesShape(indices.getShape(), expectedSrcShape);
        IndexSupport.validateGatherIndicesShape(src.getShape(), expectedSrcShape);

        Tensor out = TensorPrimitiveBuilder.ternary(
                base,
                indices,
                src,
                base.getShape().clone(),
                new scatterAdd(normalizedDimension),
                "scatterAdd",
                base.getDataType()
        );
        out.setRequiresGrad(base.getRequiresGrad() || src.getRequiresGrad());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }
            if (base.getRequiresGrad()) {
                IndexSupport.accumulateGradient(base, outGrad);
            }
            if (src.getRequiresGrad()) {
                IndexSupport.accumulateGradient(src, outGrad.gather(indices, normalizedDimension));
            }
        });
        return out;
    }
}
