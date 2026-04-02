package operations;

import tensor.BroadcastPlan;

public final class minGrad implements Operation {
    private final BroadcastPlan broadcastPlan;
    private final boolean forFirstInput;

    public minGrad(BroadcastPlan broadcastPlan, boolean forFirstInput) {
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
        return OpType.MIN_GRAD;
    }

    @Override
    public String getExpression() {
        return forFirstInput ? "min_grad_a" : "min_grad_b";
    }
}
