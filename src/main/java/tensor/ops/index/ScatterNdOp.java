package tensor.ops.index;

import operations.index.ScatterReduction;
import operations.index.scatterNd;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;

/**
 * Graph-building definition for tuple-index {@code scatterNd}.
 */
public final class ScatterNdOp {
    private ScatterNdOp() {
    }

    public static Tensor build(Tensor data, Tensor indices, Tensor updates, ScatterReduction reduction) {
        return build(data, indices, updates, reduction, 0);
    }

    public static Tensor build(Tensor data, Tensor indices, Tensor updates, ScatterReduction reduction, int batchDims) {
        if (data == null || indices == null || updates == null) {
            throw new IllegalArgumentException("scatterNd inputs cannot be null");
        }
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("scatterNd indices must be numeric integral values.");
        }
        if (data.getDataType() != updates.getDataType()) {
            throw new IllegalArgumentException("scatterNd requires data and updates to have matching dtypes.");
        }
        ScatterReduction effectiveReduction = reduction == null ? ScatterReduction.NONE : reduction;
        if (data.getDataType() == DataType.BOOL && effectiveReduction != ScatterReduction.NONE) {
            throw new IllegalArgumentException("scatterNd BOOL tensors support only NONE reduction.");
        }
        IndexSupport.validateScatterNdShape(data.getShapeUnsafe(), indices.getShapeUnsafe(), updates.getShapeUnsafe(), batchDims);
        boolean differentiable = IndexSupport.isFloating(data.getDataType())
                && (data.getRequiresGrad() || updates.getRequiresGrad());
        if (differentiable && effectiveReduction != ScatterReduction.NONE && effectiveReduction != ScatterReduction.ADD) {
            throw new UnsupportedOperationException("scatterNd backward supports only NONE and ADD reductions.");
        }

        Tensor out = TensorPrimitiveBuilder.ternary(
                data,
                indices,
                updates,
                data.getShape().clone(),
                new scatterNd(effectiveReduction, batchDims),
                "scatterNd",
                data.getDataType()
        );
        out.setRequiresGrad(differentiable);
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !IndexSupport.isFloating(data.getDataType())) {
                return;
            }
            if (data.getRequiresGrad()) {
                Tensor dataGrad = switch (effectiveReduction) {
                    case NONE -> build(outGrad, indices, Tensor.zerosLike(updates), ScatterReduction.NONE, batchDims);
                    case ADD -> outGrad;
                    case MUL, MAX, MIN -> throw new UnsupportedOperationException("scatterNd backward supports only NONE and ADD reductions.");
                };
                IndexSupport.accumulateGradient(data, dataGrad);
            }
            if (updates.getRequiresGrad()) {
                Tensor updatesGrad = outGrad.gatherNd(indices, batchDims);
                IndexSupport.accumulateGradient(updates, updatesGrad);
            }
        });
        return out;
    }
}
