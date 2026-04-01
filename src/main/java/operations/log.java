package operations;

import java.util.List;

import backend.ComputeBackend;
import tensor.Tensor;


public class log implements Operation {



    //default implementation - CPU
    @Override
    public OpType opType() {
        return OpType.LOG;
    }


    @Override
    public String getExpression() {
        return "log";
    }



}
