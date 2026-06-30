package backend.cuda.exec;

import backend.cuda.kernels.CudaKernel;
import backend.cuda.registry.CudaKernelRegistry;
import graph.model.CompiledNode;
import operations.Operation;
import runtime.execution.ExecutionContext;
import runtime.execution.PreparedStepExecutable;
import runtime.execution.PreparedStepMetadata;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Prepared direct CUDA execution with registry resolution completed during prepare. */
public record CudaDirectPreparedExecutable(Operation operation, CudaKernel kernel) implements PreparedStepExecutable {
    public CudaDirectPreparedExecutable {
        Objects.requireNonNull(operation, "operation cannot be null");
        Objects.requireNonNull(kernel, "kernel cannot be null");
    }

    public static CudaDirectPreparedExecutable prepare(CompiledNode node) {
        Operation operation = Objects.requireNonNull(node.operation(), "CUDA direct node operation cannot be null");
        CudaKernel kernel = CudaKernelRegistry.resolve(operation.opType());
        if (kernel == null) {
            throw new UnsupportedOperationException("Missing CUDA kernel for opType=" + operation.opType()
                    + " (operation class: " + operation.getClass().getName() + ")");
        }
        return new CudaDirectPreparedExecutable(operation, kernel);
    }

    @Override
    public void execute(CompiledNode node, PreparedStepMetadata metadata, ExecutionContext context) {
        Tensor output = context.runtimeTensorForNodeId(node.id());
        List<Tensor> inputs = new ArrayList<>(node.inputIds().size());
        for (int inputNodeId : node.inputIds()) {
            inputs.add(context.runtimeTensorForNodeId(inputNodeId));
        }
        kernel.forward(operation, inputs, output);
    }
}
