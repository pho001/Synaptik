package Backend.kernels.cpu;

import Operations.Operation;
import Tensor.Tensor;
import Utils.remap;

import java.util.List;

public class CpuContiguousKernel implements CpuKernel {
    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node) {
        remap.apply(inputs.getFirst(), node, 10000);
    }
}
