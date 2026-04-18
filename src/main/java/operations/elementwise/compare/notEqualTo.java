package operations.elementwise.compare;
import operations.Operation;

import tensor.BroadcastPlan;

public final class notEqualTo implements Operation {
    private final BroadcastPlan broadcastPlan;

    public notEqualTo(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    @Override
    public OpType opType() {
        return OpType.NE;
    }

    @Override
    public String getExpression() {
        return "notEqualTo";
    }
}
