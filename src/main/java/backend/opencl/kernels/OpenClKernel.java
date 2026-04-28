package backend.opencl.kernels;

import operations.Operation;
import tensor.Tensor;

import java.util.List;

/**
 * Internal SPI for one OpenCL per-node kernel implementation.
 */
@FunctionalInterface
public interface OpenClKernel {
    /**
     * Executes the operation using resolved runtime inputs and output tensor.
     */
    void forward(Operation op, List<Tensor> inputs, Tensor node);
}
