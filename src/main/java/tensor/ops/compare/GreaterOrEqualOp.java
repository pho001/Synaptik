package tensor.ops.compare;

import operations.elementwise.compare.greaterOrEqual;
import tensor.layout.BroadcastPlan;
import tensor.Tensor;
import tensor.TensorBroadcastOps;

/**
 * Graph-building definition for elementwise {@code greaterOrEqual}.
 */
public final class GreaterOrEqualOp {
    private GreaterOrEqualOp() {
    }

    /**
     * Compares whether each element of {@code first} is greater than or equal to {@code second}.
     *
     * @param first left operand; must be non-null and floating numeric
     * @param second right operand; must be non-null and floating numeric
     * @return broadcasted boolean tensor
     */
    public static Tensor build(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return ComparisonBroadcastRules.compare(first, second, plan, new greaterOrEqual(plan), "ge");
    }
}
