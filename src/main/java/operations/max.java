package operations;

import backend.ComputeBackend;
import tensor.BroadcastPlan;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;

public class max implements Operation {
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
