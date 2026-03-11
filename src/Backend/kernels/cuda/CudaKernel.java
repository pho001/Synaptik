package Backend.kernels.cuda;

import Operations.Operation;
import Tensor.Tensor;

import java.util.List;

@FunctionalInterface
public interface CudaKernel {
    void forward(Operation op, List<Tensor> inputs, Tensor node);
}
