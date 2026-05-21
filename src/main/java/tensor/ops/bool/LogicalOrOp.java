package tensor.ops.bool;

import operations.elementwise.logical.logicalOr;
import tensor.layout.BroadcastPlan;
import tensor.Tensor;

/**
 * Graph-building definition for elementwise boolean {@code logical_or}.
 */
public final class LogicalOrOp {
    private LogicalOrOp() {
    }

    /**
     * Computes elementwise logical OR with broadcasting.
     *
     * @param first left boolean tensor; must be non-null
     * @param second right boolean tensor; must be non-null
     * @return broadcasted boolean result
     * @throws IllegalArgumentException if an input is null, non-boolean, or not broadcast-compatible
     */
    public static Tensor build(Tensor first, Tensor second) {
        BroadcastPlan plan = BooleanBroadcastRules.validateBinary(first, second);
        return BooleanBroadcastRules.binary(first, second, plan, new logicalOr(plan), "logical_or");
    }
}
