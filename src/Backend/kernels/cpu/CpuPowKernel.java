package Backend.kernels.cpu;

import Operations.Operation;
import Operations.pow;
import Tensor.Tensor;

import java.util.List;

public class CpuPowKernel implements CpuKernel {
    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node) {
        double exponent = ((pow) op).getExponent();
        double[] in = inputs.get(0).getData();
        double[] out = node.getData();
        for (int i = 0; i < out.length; i++) {
            if (exponent == 0.0) out[i] = 1.0;
            else if (exponent == 1.0) out[i] = in[i];
            else if (exponent == 2.0) out[i] = in[i] * in[i];
            else out[i] = Math.pow(in[i], exponent);
        }
    }
}
