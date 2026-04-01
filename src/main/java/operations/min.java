package operations;

import backend.ComputeBackend;
import tensor.BroadcastPlan;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;

public class min implements Operation {
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
