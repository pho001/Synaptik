package backend.cpu1;

import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.runtime.ExecutionContext;
import graph.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;

/**
 * Experimental CPU backend entry point for prepared cpu1 artifacts.
 */
public final class Cpu1Backend {
    /**
     * Executes a node prepared with a {@link Cpu1PreparedArtifact}.
     *
     * @param node compiled node represented by the prepared step
     * @param metadata prepared node metadata
     * @param context run-scoped execution context
     */
    public void execute(CompiledNode node, CompiledNodeExecutionMetadata metadata, ExecutionContext context) {
        if (node == null) {
            throw new IllegalArgumentException("node cannot be null");
        }
        if (metadata == null) {
            throw new IllegalArgumentException("metadata cannot be null");
        }
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        if (!(metadata.artifact() instanceof Cpu1PreparedArtifact artifact)) {
            throw new IllegalArgumentException("cpu1 execution requires Cpu1PreparedArtifact for nodeId=" + node.id());
        }
        artifact.execute(context);
    }
}
