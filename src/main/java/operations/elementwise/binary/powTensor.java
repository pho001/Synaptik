package operations.elementwise.binary;

import operations.Operation;
import tensor.layout.BroadcastPlan;

/**
 * Raises each element of the left operand to the elementwise exponent from the
 * right operand.
 */
public final class powTensor implements Operation {
    private final BroadcastPlan broadcastPlan;

    public powTensor() {
        this(null);
    }

    public powTensor(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    @Override
    public OpType opType() {
        return OpType.POW_TENSOR;
    }

    @Override
    public String getExpression() {
        return "pow";
    }
}
