package Operations;

import Backend.ComputeBackend;
import Tensor.Tensor;

import java.util.Arrays;
import java.util.List;

public class permute implements Operation {
    private final int[] axes;

    public permute(int[] axes) {
        this.axes = axes == null ? new int[0] : axes.clone();
    }

    public int[] getAxes() {
        return axes.clone();
    }

    @Override
    public OpType opType() {
        return OpType.PERMUTE;
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
        return "permute" + Arrays.toString(axes);
    }
}

