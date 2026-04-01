package backend;

import backend.runtime.ExecutionContext;
import backend.kernels.opencl.OpenClKernel;
import backend.registry.OpenClKernelRegistry;
import graph.execution.CompiledNodeExecutionMetadata;
import tensor.Tensor;
import operations.Operation;

public class OpenClBackend {
    public void execute(
            Tensor node,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext context
    ) {
        Operation op = node.getOperation();
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
        kernel.forward(op, node.getPrevTensors(), node);
    }

}
