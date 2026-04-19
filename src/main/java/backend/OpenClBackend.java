package backend;

import backend.runtime.ExecutionContext;
import backend.kernels.opencl.OpenClKernel;
import backend.registry.OpenClKernelRegistry;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;
import tensor.Tensor;
import operations.Operation;

public class OpenClBackend {
    public void execute(
            CompiledNode node,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext context
    ) {
        Operation op = node.operation();
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
        Tensor runtimeTensor = context.runtimeTensorForNodeId(node.id());
        java.util.List<Tensor> runtimeInputs = new java.util.ArrayList<>(node.inputIds().size());
        for (int inputNodeId : node.inputIds()) {
            runtimeInputs.add(context.runtimeTensorForNodeId(inputNodeId));
        }
        kernel.forward(op, runtimeInputs, runtimeTensor);
    }

}
