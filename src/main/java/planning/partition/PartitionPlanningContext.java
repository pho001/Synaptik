package planning.partition;

import graph.model.CompiledNode;
import planning.descriptor.CompiledTensorDescriptor;
import planning.descriptor.CompiledTensorDescriptorIndex;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Read-only graph context used by partition planners and backend partition capabilities.
 *
 * @param supportsBackward whether the compiled graph includes backward execution
 * @param compiledNodes compiled node snapshots in graph order
 * @param descriptorIndex immutable tensor descriptor facts for {@code compiledNodes}
 * @param consumers consumer map keyed by producer node id
 */
public record PartitionPlanningContext(
        boolean supportsBackward,
        List<CompiledNode> compiledNodes,
        CompiledTensorDescriptorIndex descriptorIndex,
        Map<Integer, List<CompiledNode>> consumers
) {
    public PartitionPlanningContext {
        compiledNodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        descriptorIndex = Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
        consumers = copyConsumers(consumers);
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

    public CompiledTensorDescriptor descriptor(int nodeId) {
        return descriptorIndex.byNodeId(nodeId);
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

    private static Map<Integer, List<CompiledNode>> copyConsumers(Map<Integer, List<CompiledNode>> input) {
        if (input == null || input.isEmpty()) {
            return Map.of();
        }
        Map<Integer, List<CompiledNode>> copy = new HashMap<>();
        input.forEach((nodeId, nodeConsumers) ->
                copy.put(nodeId, List.copyOf(nodeConsumers == null ? List.of() : nodeConsumers)));
        return Map.copyOf(copy);
    }
}
