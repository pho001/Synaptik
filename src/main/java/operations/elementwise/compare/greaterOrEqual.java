package operations.elementwise.compare;
import operations.Operation;

import tensor.BroadcastPlan;

public final class greaterOrEqual implements Operation {
    private final BroadcastPlan broadcastPlan;

    public greaterOrEqual(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    @Override
    public OpType opType() {
        return OpType.GE;
    }

    @Override
    public String getExpression() {
        return "greaterOrEqual";
    }
}
