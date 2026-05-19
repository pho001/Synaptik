package tensor.ops.compare;

import operations.elementwise.compare.lessOrEqual;
import tensor.layout.BroadcastPlan;
import tensor.Tensor;
import tensor.TensorBroadcastOps;

/**
 * Graph-building definition for elementwise {@code lessOrEqual}.
 */
public final class LessOrEqualOp {
    private LessOrEqualOp() {
    }

    /**
     * Compares whether each element of {@code first} is less than or equal to {@code second}.
     *
     * @param first left operand; must be non-null and floating numeric
     * @param second right operand; must be non-null and floating numeric
     * @return broadcasted boolean tensor
     */
    public static Tensor build(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return CompareSupport.compare(first, second, plan, new lessOrEqual(plan), "le");
    }
}
