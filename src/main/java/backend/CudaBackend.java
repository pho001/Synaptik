package backend;

import backend.kernels.cuda.CudaKernel;
import backend.registry.CudaKernelRegistry;
import tensor.Tensor;
import operations.Operation;

import java.util.List;

public class CudaBackend {


    public void execute(Operation op, List<Tensor> inputs, Tensor node) {
        if (op == null) {
            return;
        }
        CudaKernel kernel = CudaKernelRegistry.resolve(op.opType());
        if (kernel == null) {
            throw new UnsupportedOperationException(
                    "Missing CUDA kernel for opType=" + op.opType() +
                            " (operation class: " + op.getClass().getName() + ")"
            );
        }
        kernel.forward(op, inputs, node);
    }

}
