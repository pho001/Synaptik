package operations.elementwise.binary;

import operations.Operation;
import tensor.BroadcastPlan;

public final class min implements Operation {
    private final BroadcastPlan broadcastPlan;

    public min() {
        this(null);
    }

    public min(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    @Override
    public OpType opType() {
        return OpType.MIN;
    }

    @Override
    public String getExpression() {
        return "min";
    }
}
