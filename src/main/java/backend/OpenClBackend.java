package backend;

import backend.kernels.opencl.OpenClKernel;
import backend.registry.OpenClKernelRegistry;
import tensor.Tensor;
import operations.Operation;

import java.util.List;

public class OpenClBackend {
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

}
