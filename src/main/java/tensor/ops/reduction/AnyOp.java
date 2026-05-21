package tensor.ops.reduction;

import tensor.Tensor;

/**
 * Graph-building definition for boolean {@code any}.
 */
public final class AnyOp {
    private AnyOp() {
    }

    public static Tensor build(Tensor input, int dimension) {
        return build(input, dimension, false);
    }

    public static Tensor build(Tensor input, int dimension, boolean keepDims) {
        return BoolReductionBuilder.reduce(input, dimension, keepDims, false);
    }

    public static Tensor buildAll(Tensor input) {
        return BoolReductionBuilder.reduceAll(input, false);
    }
}
