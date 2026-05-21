package tensor.ops.compare;

import operations.elementwise.compare.lessThan;
import tensor.layout.BroadcastPlan;
import tensor.Tensor;
import tensor.TensorBroadcastOps;

/**
 * Graph-building definition for elementwise {@code lessThan}.
 */
public final class LessThanOp {
    private LessThanOp() {
    }

    /**
     * Compares whether each element of {@code first} is less than {@code second}.
     *
     * @param first left operand; must be non-null and floating numeric
     * @param second right operand; must be non-null and floating numeric
     * @return broadcasted boolean tensor
     */
    public static Tensor build(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return ComparisonBroadcastRules.compare(first, second, plan, new lessThan(plan), "lt");
    }
}
