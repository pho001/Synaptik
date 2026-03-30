package operations;

import backend.ComputeBackend;
import tensor.Tensor;

import java.util.List;

public class squeeze implements Operation {
    private final int axis;

    public squeeze(int axis) {
        this.axis = axis;
    }

    public int getAxis() {
        return axis;
    }

    @Override
    public OpType opType() {
        return OpType.SQUEEZE;
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
        return "squeeze(" + axis + ")";
    }
}

