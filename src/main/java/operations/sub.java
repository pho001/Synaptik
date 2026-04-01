package operations;
import java.util.Arrays;
import java.util.List;

import backend.ComputeBackend;
import tensor.BroadcastPlan;
import tensor.Tensor;


public class sub implements Operation {
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

    //default implementation - CPU
        @Override
    public OpType opType() {
        return OpType.SUB;
    }


    @Override
    public String getExpression() {
        return "-";
    }



}
