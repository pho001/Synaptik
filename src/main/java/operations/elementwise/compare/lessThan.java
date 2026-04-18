package operations.elementwise.compare;
import operations.Operation;

import tensor.BroadcastPlan;

public final class lessThan implements Operation {
    private final BroadcastPlan broadcastPlan;

    public lessThan(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    @Override
    public OpType opType() {
        return OpType.LT;
    }

    @Override
    public String getExpression() {
        return "lessThan";
    }
}
