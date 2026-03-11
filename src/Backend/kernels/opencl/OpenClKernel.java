package Backend.kernels.opencl;

import Operations.Operation;
import Tensor.Tensor;

import java.util.List;

@FunctionalInterface
public interface OpenClKernel {
    void forward(Operation op, List<Tensor> inputs, Tensor node);
}
