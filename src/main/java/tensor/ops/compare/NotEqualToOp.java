package tensor.ops.compare;

import operations.elementwise.compare.notEqualTo;
import tensor.BroadcastPlan;
import tensor.Tensor;
import tensor.TensorBroadcastOps;

/**
 * Graph-building definition for elementwise {@code notEqualTo}.
 */
public final class NotEqualToOp {
    private NotEqualToOp() {
    }

    /**
     * Compares whether corresponding broadcasted elements are unequal.
     *
     * @param first left operand; must be non-null and floating numeric
     * @param second right operand; must be non-null and floating numeric
     * @return broadcasted boolean tensor
     */
    public static Tensor build(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return CompareSupport.compare(first, second, plan, new notEqualTo(plan), "ne");
    }
}
