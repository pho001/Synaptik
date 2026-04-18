package operations.elementwise.logical;
import operations.Operation;

import tensor.BroadcastPlan;

public final class logicalOr implements Operation {
    private final BroadcastPlan broadcastPlan;

    public logicalOr(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    @Override
    public OpType opType() {
        return OpType.LOGICAL_OR;
    }

    @Override
    public String getExpression() {
        return "logicalOr";
    }
}
