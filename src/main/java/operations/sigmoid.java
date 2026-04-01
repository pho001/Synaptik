package operations;
import backend.ComputeBackend;

import tensor.Tensor;

import java.util.List;

public class sigmoid implements Operation {



    //default implementation - CPU
    @Override
    public OpType opType() {
        return OpType.SIGMOID;
    }


    @Override
    public String getExpression() {
        return "sigmoid";
    }


}
