package operations;

import backend.ComputeBackend;
import tensor.BroadcastPlan;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;

public class add implements Operation {
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


    //default implementation - CPU
    @Override
    public OpType opType() {
        return OpType.ADD;
    }


    @Override
    public String getExpression() {
        return "+";
    }



}
