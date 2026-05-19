package tensor.ops.compare;

import operations.elementwise.compare.greaterThan;
import tensor.layout.BroadcastPlan;
import tensor.Tensor;
import tensor.TensorBroadcastOps;

/**
 * Graph-building definition for elementwise {@code greaterThan}.
 */
public final class GreaterThanOp {
    private GreaterThanOp() {
    }

    /**
     * Compares whether each element of {@code first} is greater than {@code second}.
     *
     * @param first left operand; must be non-null and floating numeric
     * @param second right operand; must be non-null and floating numeric
     * @return broadcasted boolean tensor
     */
    public static Tensor build(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return CompareSupport.compare(first, second, plan, new greaterThan(plan), "gt");
    }
}
