package operations.elementwise.binary;

import operations.Operation;
import tensor.BroadcastPlan;

public final class div implements Operation {
    private final BroadcastPlan broadcastPlan;

    public div() {
        this(null);
    }

    public div(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    @Override
    public OpType opType() {
        return OpType.DIV;
    }

    @Override
    public String getExpression() {
        return "/";
    }
}
