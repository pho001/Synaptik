package tensor.ops.index;

import operations.index.ScatterReduction;
import operations.index.scatterElements;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.layout.TensorLayoutTransform;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for rank-preserving {@code scatterElements}.
 */
public final class ScatterElementsOp {
    private ScatterElementsOp() {
    }

    public static Tensor build(Tensor data, Tensor indices, Tensor updates, int axis, ScatterReduction reduction) {
        if (data == null || indices == null || updates == null) {
            throw new IllegalArgumentException("scatterElements inputs cannot be null");
        }
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("scatterElements indices must be numeric integral values.");
        }
        if (data.getDataType() != updates.getDataType()) {
            throw new IllegalArgumentException("scatterElements requires data and updates to have matching dtypes.");
        }
        ScatterReduction effectiveReduction = reduction == null ? ScatterReduction.NONE : reduction;
        if (data.getDataType() == DataType.BOOL && effectiveReduction != ScatterReduction.NONE) {
            throw new IllegalArgumentException("scatterElements BOOL tensors support only NONE reduction.");
        }
        int[] dataShape = data.getShape();
        int normalizedAxis = TensorLayoutTransform.normalizeAxis(axis, dataShape.length);
        IndexSupport.validateScatterElementsShape(dataShape, indices.getShapeUnsafe(), updates.getShapeUnsafe(), normalizedAxis);
        boolean differentiable = IndexSupport.isFloating(data.getDataType())
                && (data.getRequiresGrad() || updates.getRequiresGrad());
        if (differentiable && effectiveReduction != ScatterReduction.NONE && effectiveReduction != ScatterReduction.ADD) {
            throw new UnsupportedOperationException("scatterElements backward supports only NONE and ADD reductions.");
        }

        Tensor out = TensorPrimitiveBuilder.ternary(
                data,
                indices,
                updates,
                dataShape.clone(),
                new scatterElements(normalizedAxis, effectiveReduction),
                "scatterElements",
                data.getDataType()
        );
        out.setRequiresGrad(differentiable);
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !IndexSupport.isFloating(data.getDataType())) {
                return;
            }
            if (data.getRequiresGrad()) {
                Tensor dataGrad = switch (effectiveReduction) {
                    case NONE -> outGrad.scatterElements(indices, Tensor.zerosLike(updates), normalizedAxis, ScatterReduction.NONE);
                    case ADD -> outGrad;
                    case MUL, MAX, MIN -> throw new UnsupportedOperationException("scatterElements backward supports only NONE and ADD reductions.");
                };
                context.accumulate(data, dataGrad);
            }
            if (updates.getRequiresGrad()) {
                Tensor updatesGrad = outGrad.takeAlongAxis(indices, normalizedAxis);
                context.accumulate(updates, updatesGrad);
            }
        });
        return out;
    }
}
