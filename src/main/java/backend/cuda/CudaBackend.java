package backend.cuda;

import backend.runtime.ExecutionContext;
import graph.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import operations.Operation;
import tensor.Tensor;
import backend.cuda.kernels.CudaKernel;
import backend.cuda.registry.CudaKernelRegistry;

import java.util.List;

/**
 * Legacy CUDA kernel dispatcher for per-node execution.
 *
 * <p>Partitioned CUDA graph execution uses {@link CudaGpuBackend}; this class is
 * retained for direct kernel registry based execution.</p>
 */
public final class CudaBackend {
    /**
     * Executes a single compiled node through the CUDA kernel registry.
     */
    public void execute(
            CompiledNode node,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext context
    ) {
        Operation op = node.operation();
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
        Tensor runtimeTensor = context.runtimeTensorForNodeId(node.id());
        List<Tensor> runtimeInputs = new java.util.ArrayList<>(node.inputIds().size());
        for (int inputNodeId : node.inputIds()) {
            runtimeInputs.add(context.runtimeTensorForNodeId(inputNodeId));
        }
        kernel.forward(op, runtimeInputs, runtimeTensor);
    }
}
