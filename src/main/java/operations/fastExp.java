package operations;

import backend.ComputeBackend;
import tensor.Tensor;
import utils.FastExp;

import java.util.List;

public class fastExp implements Operation {
    @Override
    public OpType opType() {
        return OpType.FAST_EXP;
    }


    @Override
    public String getExpression() {
        return "fastExp";
    }


}

