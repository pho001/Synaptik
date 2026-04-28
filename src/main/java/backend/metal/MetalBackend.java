package backend.metal;

import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.runtime.ExecutionContext;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;

/**
 * Metal partition backend that delegates to prepared accelerator executables.
 */
public final class MetalBackend {
    /**
     * Executes the prepared Metal partition attached to the node metadata.
     */
    public void execute(
            CompiledNode node,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext context
    ) {
        PreparedAcceleratorExecutable executable = metadata.acceleratorExecutable();
        if (executable == null) {
            throw new UnsupportedOperationException(
                    "Missing Metal executable for node " + node.label() + " (id=" + node.id() + ")"
            );
        }
        executable.execute(context);
    }
}
