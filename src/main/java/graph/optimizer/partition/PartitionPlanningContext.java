package graph.optimizer.partition;

import config.runtime.RuntimeConfig;
import graph.CompiledNode;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Read-only graph context used by partition planners and legality adapters.
 *
 * @param runtimeConfig runtime defaults used for backend support decisions
 * @param supportsBackward whether the compiled graph includes backward execution
 * @param compiledNodes compiled node snapshots in graph order
 * @param consumers consumer map keyed by producer node id
 */
public record PartitionPlanningContext(
        RuntimeConfig runtimeConfig,
        boolean supportsBackward,
        List<CompiledNode> compiledNodes,
        Map<Integer, List<CompiledNode>> consumers
) {
    public PartitionPlanningContext {
        runtimeConfig = Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null");
        compiledNodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        consumers = Map.copyOf(consumers == null ? Map.of() : consumers);
    }

    /**
     * Returns a compiled node by id.
     *
     * @param nodeId compiled node id
     * @return node, or {@code null} when the id is outside the graph
     */
    public CompiledNode compiledNode(int nodeId) {
        if (nodeId < 0 || nodeId >= compiledNodes.size()) {
            return null;
        }
        return compiledNodes.get(nodeId);
    }

    /**
     * Returns consumers for a producer node id.
     *
     * @param nodeId producer node id
     * @return immutable consumer list, empty when the producer has no consumers
     */
    public List<CompiledNode> consumersFor(int nodeId) {
        return consumers.getOrDefault(nodeId, List.of());
    }
}
