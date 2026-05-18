package tensor.ops.bool;

import operations.elementwise.logical.logicalAnd;
import tensor.BroadcastPlan;
import tensor.Tensor;

/**
 * Graph-building definition for elementwise boolean {@code logical_and}.
 */
public final class LogicalAndOp {
    private LogicalAndOp() {
    }

    /**
     * Computes elementwise logical AND with broadcasting.
     *
     * @param first left boolean tensor; must be non-null
     * @param second right boolean tensor; must be non-null
     * @return broadcasted boolean result
     * @throws IllegalArgumentException if an input is null, non-boolean, or not broadcast-compatible
     */
    public static Tensor build(Tensor first, Tensor second) {
        BroadcastPlan plan = BoolSupport.validateBinary(first, second);
        return BoolSupport.binary(first, second, plan, new logicalAnd(plan), "logical_and");
    }
}
