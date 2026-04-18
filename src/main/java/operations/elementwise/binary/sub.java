package operations.elementwise.binary;

import operations.Operation;
import tensor.BroadcastPlan;

public final class sub implements Operation {
    private final BroadcastPlan broadcastPlan;

    public sub() {
        this(null);
    }

    public sub(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    @Override
    public OpType opType() {
        return OpType.SUB;
    }

    @Override
    public String getExpression() {
        return "-";
    }
}
