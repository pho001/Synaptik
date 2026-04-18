package operations.elementwise.binary;
import operations.Operation;

import tensor.BroadcastPlan;

public final class maxGrad implements Operation {
    private final BroadcastPlan broadcastPlan;
    private final boolean forFirstInput;

    public maxGrad(BroadcastPlan broadcastPlan, boolean forFirstInput) {
        this.broadcastPlan = broadcastPlan;
        this.forFirstInput = forFirstInput;
    }

    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    public boolean isForFirstInput() {
        return forFirstInput;
    }

    @Override
    public OpType opType() {
        return OpType.MAX_GRAD;
    }

    @Override
    public String getExpression() {
        return forFirstInput ? "max_grad_a" : "max_grad_b";
    }
}
