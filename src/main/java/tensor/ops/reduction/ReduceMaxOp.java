package tensor.ops.reduction;

import tensor.Tensor;

/**
 * Graph-building definition for reduce {@code max}.
 */
public final class ReduceMaxOp {
    private ReduceMaxOp() {
    }

    public static Tensor build(Tensor input, int dimension) {
        ReductionShapeRules.requireFloatingInput(input, "max");
        return build(input, dimension, false);
    }

    public static Tensor build(Tensor input, int dimension, boolean keepDims) {
        return MinMaxReductionBuilder.reduce(input, dimension, keepDims, true);
    }

    public static Tensor buildAll(Tensor input) {
        ReductionShapeRules.requireFloatingInput(input, "max");
        return MinMaxReductionBuilder.reduceAll(input, true);
    }
}
