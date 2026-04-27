package graph.optimizer.region;

import config.optimizer.FuseConfig;
import graph.CompiledNode;

import java.util.List;

public record RegionOptimizationContext(
        List<CompiledNode> compiledNodes,
        FuseConfig fuseConfig
) {
    public RegionOptimizationContext {
        compiledNodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        fuseConfig = fuseConfig == null ? FuseConfig.trainingDefaults() : fuseConfig;
    }

    public CompiledNode compiledNode(int nodeId) {
        if (nodeId < 0 || nodeId >= compiledNodes.size()) {
            return null;
        }
        return compiledNodes.get(nodeId);
    }
}
