package Operations;

import Backend.ComputeBackend;
import Tensor.Tensor;

import java.util.Arrays;
import java.util.List;

public class max implements Operation {
    @Override
    public void apply(List<Tensor> inputs, Tensor node) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException("The input array must contain exactly 2 elements");
        }
        if (!Arrays.equals(inputs.getFirst().getShape(), inputs.getLast().getShape())) {
            throw new IllegalArgumentException("Input shapes must match");
        }
        double[] a = inputs.getFirst().getData();
        double[] b = inputs.getLast().getData();
        double[] out = node.getData();
        for (int i = 0; i < out.length; i++) {
            out[i] = Math.max(a[i], b[i]);
        }
        node.setData(out);
    }

    @Override
    public OpType opType() {
        return OpType.MAX;
    }

    @Override
    public ComputeBackend getPreferredBackend() {
        return ComputeBackend.CPU;
    }

    @Override
    public boolean supportsBackend(ComputeBackend backend) {
        return backend == ComputeBackend.GPU_CUDA || backend == ComputeBackend.CPU;
    }

    @Override
    public String getExpression() {
        return "max";
    }

    @Override
    public boolean isElementWise() {
        return true;
    }
}
