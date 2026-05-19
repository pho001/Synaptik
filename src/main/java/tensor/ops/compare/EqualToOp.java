package tensor.ops.compare;

import operations.elementwise.compare.equalTo;
import tensor.layout.BroadcastPlan;
import tensor.Tensor;
import tensor.TensorBroadcastOps;

/**
 * Graph-building definition for elementwise {@code equalTo}.
 */
public final class EqualToOp {
    private EqualToOp() {
    }

    /**
     * Compares whether corresponding broadcasted elements are equal.
     *
     * @param first left operand; must be non-null and floating numeric
     * @param second right operand; must be non-null and floating numeric
     * @return broadcasted boolean tensor
     */
    public static Tensor build(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return CompareSupport.compare(first, second, plan, new equalTo(plan), "eq");
    }
}
