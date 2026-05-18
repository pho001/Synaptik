package tensor.ops.unary;

import tensor.Tensor;

/**
 * Graph-building definition for range clamp composition.
 */
public final class ClampOp {
    private ClampOp() {
    }

    public static Tensor build(Tensor input, double minValue, double maxValue) {
        UnarySupport.requireNumeric(input, "clamp");
        if (minValue > maxValue) {
            throw new IllegalArgumentException("clamp requires minValue <= maxValue.");
        }
        return input.clampMax(maxValue).clampMin(minValue);
    }
}
