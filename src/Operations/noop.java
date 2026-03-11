package Operations;

import Backend.ComputeBackend;
import Tensor.Tensor;

import java.util.List;

public class noop implements Operation
{

    @Override
    public void apply(List<Tensor> inputs, Tensor node) {

    }

    @Override
    public OpType opType() {
        return OpType.NOOP;
    }

    @Override
    public boolean isElementWise() {
        return false;
    }

    @Override
    public ComputeBackend getPreferredBackend() {
        return null;
    }

    @Override
    public boolean supportsBackend(ComputeBackend backend) {
        return false;
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
