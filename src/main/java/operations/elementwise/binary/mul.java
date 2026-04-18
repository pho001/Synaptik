package operations.elementwise.binary;

import operations.Operation;
import tensor.BroadcastPlan;

public final class mul implements Operation {
    private final BroadcastPlan broadcastPlan;

    public mul() {
        this(null);
    }

    public mul(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    @Override
    public OpType opType() {
        return OpType.MUL;
    }

    @Override
    public String getExpression() {
        return "*";
    }
}
