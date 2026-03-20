package Backend.kernels.cpu;

import Operations.Operation;
import Tensor.Tensor;
import Tensor.TensorRemap;

import java.util.List;

public class CpuContiguousKernel implements CpuKernel {
    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node) {
        TensorRemap.apply(inputs.getFirst(), node, 10000);
    }
}
