package operations;

import backend.ComputeBackend;
import tensor.Tensor;
import utils.FastExp;

import java.util.List;

public class fastTanh implements Operation {
    @Override
    public OpType opType() {
        return OpType.FAST_TANH;
    }


    @Override
    public String getExpression() {
        return "fastTanh";
    }


}

