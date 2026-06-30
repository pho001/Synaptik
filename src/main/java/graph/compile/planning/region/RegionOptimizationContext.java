package graph.compile.planning.region;

import config.optimizer.CpuFusionConfig;
import config.optimizer.FuseConfig;
import graph.model.CompiledNode;

import java.util.List;

/**
 * Context available while optimizing regions.
 *
 * @param compiledNodes compiled node snapshots in graph order
 * @param fuseConfig fusion configuration
 * @param cpuFusionConfig CPU fused-loop configuration
 */
public record RegionOptimizationContext(
        List<CompiledNode> compiledNodes,
        FuseConfig fuseConfig,
        CpuFusionConfig cpuFusionConfig
) {
    public RegionOptimizationContext {
        compiledNodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        fuseConfig = fuseConfig == null ? FuseConfig.trainingDefaults() : fuseConfig;
        cpuFusionConfig = cpuFusionConfig == null ? CpuFusionConfig.defaults() : cpuFusionConfig;
    }

    /**
     * Creates a context with default CPU fusion policy.
     *
     * @param compiledNodes compiled node snapshots in graph order
     * @param fuseConfig fusion configuration
     */
    public RegionOptimizationContext(List<CompiledNode> compiledNodes, FuseConfig fuseConfig) {
        this(compiledNodes, fuseConfig, CpuFusionConfig.defaults());
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
}
