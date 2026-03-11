package Backend.kernels.cpu;

import Operations.Operation;
import Tensor.Tensor;

import java.util.List;

public class CpuSubKernel implements CpuKernel {
    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node) {
        double[] a = inputs.get(0).getData();
        double[] b = inputs.get(1).getData();
        double[] out = node.getData();
        for (int i = 0; i < out.length; i++) out[i] = a[i] - b[i];
    }
}
