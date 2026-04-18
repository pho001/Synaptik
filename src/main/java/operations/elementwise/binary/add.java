package operations.elementwise.binary;

import operations.Operation;
import tensor.BroadcastPlan;

public final class add implements Operation {
    private final BroadcastPlan broadcastPlan;

    public add() {
        this(null);
    }

    public add(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    @Override
    public OpType opType() {
        return OpType.ADD;
    }

    @Override
    public String getExpression() {
        return "+";
    }
}
