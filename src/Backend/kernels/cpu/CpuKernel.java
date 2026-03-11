package Backend.kernels.cpu;

import Operations.Operation;
import Tensor.Tensor;

import java.util.List;

@FunctionalInterface
public interface CpuKernel {
    void forward(Operation op, List<Tensor> inputs, Tensor node);
}
