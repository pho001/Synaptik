package operations;

import java.util.Arrays;
import java.util.List;


import backend.ComputeBackend;
import tensor.BroadcastPlan;
import tensor.Tensor;

public class div implements Operation {
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
