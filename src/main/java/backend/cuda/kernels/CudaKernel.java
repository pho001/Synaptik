package backend.cuda.kernels;

import operations.Operation;
import tensor.Tensor;

import java.util.List;

@FunctionalInterface
public interface CudaKernel {
    void forward(Operation op, List<Tensor> inputs, Tensor node);
}
