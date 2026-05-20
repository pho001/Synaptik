package tensor.ops.index;

import operations.index.scatterAxisAdd;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.layout.TensorLayoutTransform;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for rank-changing axis {@code scatterAxisAdd}.
 */
public final class ScatterAxisAddOp {
    private ScatterAxisAddOp() {
    }

    public static Tensor build(Tensor data, Tensor indices, Tensor updates, int axis) {
        if (data == null || indices == null || updates == null) {
            throw new IllegalArgumentException("scatterAxisAdd inputs cannot be null");
        }
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("scatterAxisAdd indices must be numeric integral values.");
        }
        if (data.getDataType() != updates.getDataType()) {
            throw new IllegalArgumentException("scatterAxisAdd requires data and updates to have matching dtypes.");
        }
        if (!IndexSupport.isFloating(data.getDataType())) {
            throw new IllegalArgumentException("scatterAxisAdd requires floating numeric data and updates.");
        }
        int normalizedAxis = TensorLayoutTransform.normalizeAxis(axis, data.getShapeUnsafe().length);
        IndexSupport.validateScatterAxisAddShape(data.getShapeUnsafe(), indices.getShapeUnsafe(), updates.getShapeUnsafe(), normalizedAxis);
        Tensor out = TensorPrimitiveBuilder.ternary(
                data,
                indices,
                updates,
                data.getShape().clone(),
                new scatterAxisAdd(normalizedAxis),
                "scatterAxisAdd",
                data.getDataType()
        );
        out.setRequiresGrad(data.getRequiresGrad() || updates.getRequiresGrad());
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !IndexSupport.isFloating(data.getDataType())) {
                return;
            }
            if (data.getRequiresGrad()) {
                context.accumulate(data, outGrad);
            }
            if (updates.getRequiresGrad()) {
                context.accumulate(updates, outGrad.gatherAxis(indices, normalizedAxis));
            }
        });
        return out;
    }
}
