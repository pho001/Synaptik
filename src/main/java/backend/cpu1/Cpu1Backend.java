package backend.cpu1;

import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.runtime.ExecutionContext;
import graph.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;

import java.util.Objects;

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
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(metadata, "metadata cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        if (!(metadata.artifact() instanceof Cpu1PreparedArtifact artifact)) {
            throw new IllegalArgumentException("cpu1 execution requires Cpu1PreparedArtifact for nodeId=" + node.id());
        }
        artifact.execute(context);
    }
}
