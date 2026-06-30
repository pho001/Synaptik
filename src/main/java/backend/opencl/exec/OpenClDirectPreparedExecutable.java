package backend.opencl.exec;

import backend.opencl.kernels.OpenClKernel;
import backend.opencl.registry.OpenClKernelRegistry;
import graph.model.CompiledNode;
import operations.Operation;
import runtime.execution.ExecutionContext;
import runtime.execution.PreparedStepExecutable;
import runtime.execution.PreparedStepMetadata;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Prepared direct OpenCL execution with registry resolution completed during prepare. */
public record OpenClDirectPreparedExecutable(Operation operation, OpenClKernel kernel)
        implements PreparedStepExecutable {
    public OpenClDirectPreparedExecutable {
        Objects.requireNonNull(operation, "operation cannot be null");
        Objects.requireNonNull(kernel, "kernel cannot be null");
    }

    public static OpenClDirectPreparedExecutable prepare(CompiledNode node) {
        Operation operation = Objects.requireNonNull(node.operation(), "OpenCL direct node operation cannot be null");
        OpenClKernel kernel = OpenClKernelRegistry.resolve(operation.opType());
        if (kernel == null) {
            throw new UnsupportedOperationException("Missing OpenCL kernel for opType=" + operation.opType()
                    + " (operation class: " + operation.getClass().getName() + ")");
        }
        return new OpenClDirectPreparedExecutable(operation, kernel);
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
