package operations;

import java.util.List;

import backend.ComputeBackend;
import tensor.Tensor;

public class exp implements Operation {



    //default implementation - CPU
    @Override
    public OpType opType() {
        return OpType.EXP;
    }


    @Override
    public String getExpression() {
        return "exp";
    }





}
