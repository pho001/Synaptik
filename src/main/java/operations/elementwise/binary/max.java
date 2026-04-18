package operations.elementwise.binary;

import operations.Operation;
import tensor.BroadcastPlan;

public final class max implements Operation {
    private final BroadcastPlan broadcastPlan;

    public max() {
        this(null);
    }

    public max(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    @Override
    public OpType opType() {
        return OpType.MAX;
    }

    @Override
    public String getExpression() {
        return "max";
    }
}
