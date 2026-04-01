package operations;
import backend.ComputeBackend;

import tensor.Tensor;

import java.util.List;

public class relu implements Operation {



    //default implementation - CPU
    @Override
    public OpType opType() {
        return OpType.RELU;
    }


    @Override
    public String getExpression() {
        return "relu";
    }


}
