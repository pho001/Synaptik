package tensor.ops.reduction;

import tensor.Tensor;

/**
 * Graph-building definition for boolean {@code all}.
 */
public final class AllOp {
    private AllOp() {
    }

    public static Tensor build(Tensor input, int dimension) {
        return build(input, dimension, false);
    }

    public static Tensor build(Tensor input, int dimension, boolean keepDims) {
        return ReductionSupport.reduceBool(input, dimension, keepDims, true);
    }

    public static Tensor buildAll(Tensor input) {
        return ReductionSupport.reduceBoolAll(input, true);
    }
}
