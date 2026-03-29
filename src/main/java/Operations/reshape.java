package Operations;

import Backend.ComputeBackend;
import Tensor.Tensor;

import java.util.Arrays;
import java.util.List;

public class reshape implements Operation {
    private final int[] targetShape;

    public reshape(int[] targetShape) {
        this.targetShape = targetShape == null ? new int[0] : targetShape.clone();
    }

    public int[] getTargetShape() {
        return targetShape.clone();
    }

    @Override
    public OpType opType() {
        return OpType.RESHAPE;
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
        return "reshape" + Arrays.toString(targetShape);
    }
}

