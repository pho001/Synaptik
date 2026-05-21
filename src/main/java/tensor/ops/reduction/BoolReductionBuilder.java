package tensor.ops.reduction;

import operations.Operation;
import operations.reduction.reduceAll;
import operations.reduction.reduceAny;
import tensor.DataType;
import tensor.Tensor;
import tensor.internal.TensorPrimitiveBuilder;
import tensor.layout.TensorLayoutTransform;

final class BoolReductionBuilder {
    private BoolReductionBuilder() {
    }

    static Tensor reduce(Tensor input, int dimension, boolean keepDims, boolean isAll) {
        if (input.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException((isAll ? "all" : "any") + " requires BOOL input.");
        }
        int[] shape = input.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, shape.length);
        int[] newShape = ReductionShapeRules.reduceShape(shape, normalizedDimension, keepDims);
        Operation op = isAll ? new reduceAll(normalizedDimension, keepDims) : new reduceAny(normalizedDimension, keepDims);
        return TensorPrimitiveBuilder.unaryNoGrad(input, newShape, op, isAll ? "all_reduce" : "any_reduce", DataType.BOOL);
    }

    static Tensor reduceAll(Tensor input, boolean isAll) {
        if (input.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException((isAll ? "all" : "any") + " requires BOOL input.");
        }
        return TensorPrimitiveBuilder.unaryNoGrad(
                input,
                new int[]{1},
                isAll ? new reduceAll(-1) : new reduceAny(-1),
                isAll ? "all_reduce" : "any_reduce",
                DataType.BOOL
        );
    }
}
