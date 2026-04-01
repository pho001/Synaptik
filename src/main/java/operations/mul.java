package operations;

import backend.ComputeBackend;
import tensor.BroadcastPlan;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;

public class mul implements Operation {
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



    //default implementation - CPU
    @Override
    public OpType opType() {
        return OpType.MUL;
    }


    @Override
    public String getExpression() {
        return "*";
    }


}
