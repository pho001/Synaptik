package tensor.ops.reduction;

import tensor.Tensor;

/**
 * Graph-building definition for reduce {@code min}.
 */
public final class ReduceMinOp {
    private ReduceMinOp() {
    }

    public static Tensor build(Tensor input, int dimension) {
        ReductionSupport.requireFloatingInput(input, "min");
        return build(input, dimension, false);
    }

    public static Tensor build(Tensor input, int dimension, boolean keepDims) {
        return ReductionSupport.reduceMinMax(input, dimension, keepDims, false);
    }

    public static Tensor buildAll(Tensor input) {
        ReductionSupport.requireFloatingInput(input, "min");
        return ReductionSupport.reduceMinMaxAll(input, false);
    }
}
