package Backend.kernels.cpu;

import Operations.Operation;
import Tensor.Tensor;

import java.util.List;

public interface CpuKernel {
    void forward(Operation op, List<Tensor> inputs, Tensor node);

    default void forward(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        forward(op, inputs, node);
    }
}
