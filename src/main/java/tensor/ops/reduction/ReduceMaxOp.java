package tensor.ops.reduction;

import tensor.Tensor;

/**
 * Graph-building definition for reduce {@code max}.
 */
public final class ReduceMaxOp {
    private ReduceMaxOp() {
    }

    public static Tensor build(Tensor input, int dimension) {
        ReductionSupport.requireFloatingInput(input, "max");
        return build(input, dimension, false);
    }

    public static Tensor build(Tensor input, int dimension, boolean keepDims) {
        return ReductionSupport.reduceMinMax(input, dimension, keepDims, true);
    }

    public static Tensor buildAll(Tensor input) {
        ReductionSupport.requireFloatingInput(input, "max");
        return ReductionSupport.reduceMinMaxAll(input, true);
    }
}
