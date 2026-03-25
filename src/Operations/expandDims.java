package Operations;

import Backend.ComputeBackend;
import Tensor.Tensor;

import java.util.List;

public class expandDims implements Operation {
    private final int axis;

    public expandDims(int axis) {
        this.axis = axis;
    }

    public int getAxis() {
        return axis;
    }

    @Override
    public OpType opType() {
        return OpType.EXPAND_DIMS;
    }

    @Override
    public boolean isElementWise() {
        return false;
    }

    @Override
    public void apply(List<Tensor> inputs, Tensor out) {}

    @Override
    public ComputeBackend getPreferredBackend() {
        return ComputeBackend.CPU;
    }

    @Override
    public boolean supportsBackend(ComputeBackend backend) {
        return backend == ComputeBackend.CPU;
    }

    @Override
    public String getExpression() {
        return "expandDims(" + axis + ")";
    }
}

