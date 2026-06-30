package backend.lowering;

import graph.model.CompiledNode;
import config.runtime.RuntimeConfig;
import planning.descriptor.CompiledTensorDescriptor;
import planning.descriptor.CompiledTensorDescriptorIndex;
import planning.partition.PartitionPlan;

import java.util.List;
import java.util.Map;
import java.util.Objects;

public record LoweringContext(
        RuntimeConfig runtimeConfig,
        List<CompiledNode> compiledNodes,
        CompiledTensorDescriptorIndex descriptorIndex,
        Map<String, PartitionPlan> partitionPlansById
) {
    public LoweringContext {
        compiledNodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        descriptorIndex = Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
        partitionPlansById = Map.copyOf(partitionPlansById == null ? Map.of() : partitionPlansById);
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

    public PartitionPlan partitionPlanFor(String partitionId) {
        return partitionId == null ? null : partitionPlansById.get(partitionId);
    }

    public LoweringContext withPartitionPlans(Map<String, PartitionPlan> partitionPlansById) {
        return new LoweringContext(runtimeConfig, compiledNodes, descriptorIndex, partitionPlansById);
    }
}
