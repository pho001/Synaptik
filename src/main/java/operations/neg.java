package operations;

import backend.ComputeBackend;
import tensor.Tensor;

import java.util.List;


public class neg implements Operation{
    @Override
    public OpType opType() {
        return OpType.NEG;
    }

    @Override
    public String getExpression() {
        return "*";
    }


    @Override
    public boolean isCheap() { return true;}
}
