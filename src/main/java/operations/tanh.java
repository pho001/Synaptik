package operations;
import backend.ComputeBackend;

import tensor.Tensor;

import java.util.List;

public class tanh implements Operation {



    //default implementation - CPU
    @Override
    public OpType opType() {
        return OpType.TANH;
    }

    @Override
    public String getExpression() {
        return "tanh";
    }



}
