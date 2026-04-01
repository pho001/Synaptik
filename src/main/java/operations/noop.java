package operations;

import backend.ComputeBackend;
import tensor.Tensor;

import java.util.List;

public class noop implements Operation
{

    @Override
    public OpType opType() {
        return OpType.NOOP;
    }


    @Override
    public String getExpression() {
        return "yield";
    }

    @Override
    public boolean isCheap() {
        return false;
    }
}
