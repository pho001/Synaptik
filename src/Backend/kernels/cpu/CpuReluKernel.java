package Backend.kernels.cpu;

import Operations.Operation;
import Tensor.Tensor;

import java.util.List;

public class CpuReluKernel implements CpuKernel {
    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node) {
        double[] in = inputs.get(0).getData();
        double[] out = node.getData();
        for (int i = 0; i < out.length; i++) out[i] = Math.max(0, in[i]);
    }
}
