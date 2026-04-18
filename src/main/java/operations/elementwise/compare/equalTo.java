package operations.elementwise.compare;
import operations.Operation;

import tensor.BroadcastPlan;

public final class equalTo implements Operation {
    private final BroadcastPlan broadcastPlan;

    public equalTo(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    @Override
    public OpType opType() {
        return OpType.EQ;
    }

    @Override
    public String getExpression() {
        return "equalTo";
    }
}
