package operations;

import tensor.BroadcastPlan;

public final class logicalAnd implements Operation {
    private final BroadcastPlan broadcastPlan;

    public logicalAnd(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    @Override
    public OpType opType() {
        return OpType.LOGICAL_AND;
    }

    @Override
    public String getExpression() {
        return "logicalAnd";
    }
}
