package backend.cuda;

import backend.accelerator.exec.AcceleratorExecutionArtifact;
import backend.runtime.ExecutionContext;
import graph.model.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;

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
        if (!(metadata.artifact() instanceof AcceleratorExecutionArtifact artifact) || artifact.executable() == null) {
            throw new UnsupportedOperationException(
                    "Missing CUDA accelerator executable for node " + node.label() + " (id=" + node.id() + ")"
            );
        }
        artifact.executable().execute(context);
    }
}
