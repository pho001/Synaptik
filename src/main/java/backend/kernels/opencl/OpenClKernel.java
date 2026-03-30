package backend.kernels.opencl;

import operations.Operation;
import tensor.Tensor;

import java.util.List;

@FunctionalInterface
public interface OpenClKernel {
    void forward(Operation op, List<Tensor> inputs, Tensor node);
}
