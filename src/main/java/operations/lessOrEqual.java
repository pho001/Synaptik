package operations;

import tensor.BroadcastPlan;

public final class lessOrEqual implements Operation {
    private final BroadcastPlan broadcastPlan;

    public lessOrEqual(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    @Override
    public OpType opType() {
        return OpType.LE;
    }

    @Override
    public String getExpression() {
        return "lessOrEqual";
    }
}
