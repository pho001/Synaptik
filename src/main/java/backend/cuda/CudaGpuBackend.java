package backend.cuda;

import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.runtime.ExecutionContext;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;

/**
 * CUDA partition backend that delegates to prepared accelerator executables.
 */
public final class CudaGpuBackend {
    /**
     * Executes the prepared CUDA partition attached to the node metadata.
     */
    public void execute(
            CompiledNode node,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext context
    ) {
        PreparedAcceleratorExecutable executable = metadata.acceleratorExecutable();
        if (executable == null) {
            throw new UnsupportedOperationException(
                    "Missing CUDA accelerator executable for node " + node.label() + " (id=" + node.id() + ")"
            );
        }
        executable.execute(context);
    }
}
