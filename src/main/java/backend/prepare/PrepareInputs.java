package backend.prepare;

import config.runtime.RuntimeConfig;
import graph.CompiledNode;

import java.util.List;
import java.util.Map;
import java.util.Objects;

public record PrepareInputs(
        RuntimeConfig runtimeConfig,
        boolean supportsBackward,
        List<CompiledNode> compiledNodes,
        Map<Integer, List<CompiledNode>> consumers
) {
    public PrepareInputs {
        runtimeConfig = Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null");
        compiledNodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        consumers = Map.copyOf(consumers == null ? Map.of() : consumers);
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
}
