package backend.prepare;

import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class BackendPrepareContext {
    private final RuntimeConfig runtimeConfig;
    private final boolean supportsBackward;
    private final List<CompiledNode> compiledNodes;
    private final Map<Integer, List<CompiledNode>> consumers;
    private final Map<Integer, CompiledNodeExecutionMetadata> preparedMetadata;

    public BackendPrepareContext(
            RuntimeConfig runtimeConfig,
            boolean supportsBackward,
            List<CompiledNode> compiledNodes,
            Map<Integer, List<CompiledNode>> consumers
    ) {
        this.runtimeConfig = Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null");
        this.supportsBackward = supportsBackward;
        this.compiledNodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        this.consumers = Map.copyOf(consumers == null ? Map.of() : consumers);
        this.preparedMetadata = new HashMap<>();
    }

    public RuntimeConfig runtimeConfig() {
        return runtimeConfig;
    }

    public boolean supportsBackward() {
        return supportsBackward;
    }

    public List<CompiledNode> compiledNodes() {
        return compiledNodes;
    }

    public CompiledNode compiledNode(int nodeId) {
        if (nodeId < 0 || nodeId >= compiledNodes.size()) {
            return null;
        }
        return compiledNodes.get(nodeId);
    }

    public List<CompiledNode> consumersFor(int nodeId) {
        return consumers.getOrDefault(nodeId, List.of());
    }

    public CompiledNodeExecutionMetadata preparedMetadataFor(int nodeId) {
        return preparedMetadata.get(nodeId);
    }

    public void publishPreparedMetadata(int nodeId, CompiledNodeExecutionMetadata metadata) {
        if (metadata == null) {
            preparedMetadata.remove(nodeId);
            return;
        }
        preparedMetadata.put(nodeId, metadata);
    }
}
