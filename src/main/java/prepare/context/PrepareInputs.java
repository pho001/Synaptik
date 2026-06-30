package prepare.context;

import config.runtime.RuntimeConfig;
import graph.model.CompiledNode;
import planning.descriptor.CompiledTensorDescriptor;
import planning.descriptor.CompiledTensorDescriptorIndex;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable inputs shared by backend preparers during prepared execution construction.
 *
 * @param runtimeConfig runtime policy used for prepare-time decisions
 * @param supportsBackward whether compiled artifacts include backward execution
 * @param compiledNodes compiled nodes indexed by id
 * @param descriptorIndex immutable tensor descriptor facts for {@code compiledNodes}
 * @param consumers consumer map keyed by producer node id
 */
record PrepareInputs(
        RuntimeConfig runtimeConfig,
        boolean supportsBackward,
        List<CompiledNode> compiledNodes,
        CompiledTensorDescriptorIndex descriptorIndex,
        Map<Integer, List<CompiledNode>> consumers
) {
    public PrepareInputs {
        runtimeConfig = Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null");
        compiledNodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        descriptorIndex = Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
        consumers = Map.copyOf(consumers == null ? Map.of() : consumers);
    }

    /**
     * Returns the compiled node for an id.
     *
     * @param nodeId compiled node id
     * @return compiled node, or {@code null} when the id is outside the compiled node list
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
     * Returns consumers of a compiled node.
     *
     * @param nodeId producer node id
     * @return immutable consumer list, or an empty list when no consumers are known
     */
    public List<CompiledNode> consumersFor(int nodeId) {
        return consumers.getOrDefault(nodeId, List.of());
    }
}
