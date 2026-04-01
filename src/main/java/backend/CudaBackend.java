package backend;

import backend.runtime.ExecutionContext;
import graph.execution.CompiledNodeExecutionMetadata;
import operations.Operation;
import tensor.Tensor;
import backend.kernels.cuda.CudaKernel;
import backend.registry.CudaKernelRegistry;

import java.util.List;

public final class CudaBackend {
    public void execute(
            Tensor node,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext context
    ) {
        Operation op = node.getOperation();
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
        kernel.forward(op, node.getPrevTensors(), node);
    }
}