package operations;

import tensor.BroadcastPlan;

public final class greaterThan implements Operation {
    private final BroadcastPlan broadcastPlan;

    public greaterThan(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    @Override
    public OpType opType() {
        return OpType.GT;
    }

    @Override
    public String getExpression() {
        return "greaterThan";
    }
}
