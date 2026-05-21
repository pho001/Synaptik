package tensor.ops.reduction;

import operations.reduction.reduceProd;
import tensor.Tensor;
import tensor.layout.TensorLayoutTransform;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for {@code prod} reductions.
 */
public final class ProdOp {
    private ProdOp() {
    }

    public static Tensor build(Tensor input, int dimension) {
        return build(input, dimension, false);
    }

    public static Tensor build(Tensor input, int dimension, boolean keepDims) {
        ReductionShapeRules.requireFloatingInput(input, "prod");
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, input.getShape().length);
        return TensorPrimitiveBuilder.unaryNoGrad(
                input,
                ReductionShapeRules.reduceShape(input.getShape(), normalizedDimension, keepDims),
                new reduceProd(normalizedDimension, keepDims),
                "prod_reduce",
                input.getDataType()
        );
    }

    public static Tensor buildAll(Tensor input) {
        ReductionShapeRules.requireFloatingInput(input, "prod");
        return TensorPrimitiveBuilder.unaryNoGrad(
                input,
                new int[]{1},
                new reduceProd(-1),
                "prod_reduce",
                input.getDataType()
        );
    }
}
