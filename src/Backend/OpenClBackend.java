package Backend;

import Backend.kernels.opencl.OpenClKernel;
import Backend.registry.OpenClKernelRegistry;
import Tensor.Tensor;
import Operations.Operation;

import java.util.List;

public class OpenClBackend implements BackendExecutor {
    @Override
    public void execute(Operation op, List<Tensor> inputs, Tensor node) {
        if (op == null) {
            return;
        }
        OpenClKernel kernel = OpenClKernelRegistry.resolve(op.opType());
        if (kernel == null) {
            throw new UnsupportedOperationException(
                    "Missing OpenCL kernel for opType=" + op.opType() +
                            " (operation class: " + op.getClass().getName() + ")"
            );
        }
        kernel.forward(op, inputs, node);
    }

    @Override
    public void backward(Operation op, List<Tensor> inputs, Tensor node) {

    }
}
