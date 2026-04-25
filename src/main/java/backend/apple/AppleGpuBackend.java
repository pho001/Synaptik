package backend.apple;

import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.runtime.ExecutionContext;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;

public final class AppleGpuBackend {
    public void execute(
            CompiledNode node,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext context
    ) {
        PreparedAcceleratorExecutable executable = metadata.acceleratorExecutable();
        if (executable == null) {
            throw new UnsupportedOperationException(
                    "Missing Apple GPU executable for node " + node.label() + " (id=" + node.id() + ")"
            );
        }
        executable.execute(context);
    }
}
