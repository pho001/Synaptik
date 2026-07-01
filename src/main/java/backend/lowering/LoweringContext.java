package backend.lowering;

import graph.model.CompiledNode;
import config.runtime.RuntimeConfig;
import planning.descriptor.CompiledTensorDescriptor;
import planning.descriptor.CompiledTensorDescriptorIndex;
import java.util.List;
import java.util.Objects;

public record LoweringContext(
        RuntimeConfig runtimeConfig,
        List<CompiledNode> compiledNodes,
        CompiledTensorDescriptorIndex descriptorIndex
) {
    public LoweringContext {
        compiledNodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        descriptorIndex = Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
    }

    public CompiledNode compiledNode(int nodeId) {
        if (nodeId < 0 || nodeId >= compiledNodes.size()) {
            return null;
        }
        return compiledNodes.get(nodeId);
    }

    public CompiledTensorDescriptor descriptor(int nodeId) {
        return descriptorIndex.byNodeId(nodeId);
    }

}
