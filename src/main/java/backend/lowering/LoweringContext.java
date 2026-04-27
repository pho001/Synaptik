package backend.lowering;

import graph.CompiledNode;
import config.runtime.RuntimeConfig;
import graph.optimizer.partition.PartitionPlan;

import java.util.List;
import java.util.Map;

public record LoweringContext(
        RuntimeConfig runtimeConfig,
        List<CompiledNode> compiledNodes,
        Map<String, PartitionPlan> partitionPlansById
) {
    public LoweringContext {
        compiledNodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        partitionPlansById = Map.copyOf(partitionPlansById == null ? Map.of() : partitionPlansById);
    }

    public LoweringContext(
            RuntimeConfig runtimeConfig,
            List<CompiledNode> compiledNodes
    ) {
        this(runtimeConfig, compiledNodes, Map.of());
    }

    public CompiledNode compiledNode(int nodeId) {
        if (nodeId < 0 || nodeId >= compiledNodes.size()) {
            return null;
        }
        return compiledNodes.get(nodeId);
    }

    public PartitionPlan partitionPlanFor(String partitionId) {
        return partitionId == null ? null : partitionPlansById.get(partitionId);
    }

    public LoweringContext withPartitionPlans(Map<String, PartitionPlan> partitionPlansById) {
        return new LoweringContext(runtimeConfig, compiledNodes, partitionPlansById);
    }
}
