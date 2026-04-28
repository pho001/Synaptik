package graph.optimizer.region;

import config.optimizer.CpuFusionConfig;
import config.optimizer.FuseConfig;
import graph.CompiledNode;
import graph.optimizer.OptimizationRule;
import graph.optimizer.state.OptimizerState;

import java.util.List;
import java.util.Objects;

/**
 * Optimizer stage that turns accepted partitions into optimized regions.
 *
 * <p>The region stage is the fusion boundary between partition planning and memory planning. It takes backend
 * partitions, chooses execution units such as fused elementwise subchains or single-operation units, classifies region
 * values as materialized, virtual, or continuation values, and stores those optimized regions on the optimizer state.
 */
public class RegionOptimizationRule implements OptimizationRule {
    private final FuseConfig config;
    private final CpuFusionConfig cpuFusionConfig;
    private final RegionOptimizer regionOptimizer;

    /**
     * Creates a region optimization rule with training fusion defaults.
     */
    public RegionOptimizationRule() {
        this(FuseConfig.trainingDefaults());
    }

    /**
     * Creates a rule with training defaults and explicit shared-expensive-node behavior.
     *
     * @param preserveSharedExpensiveNodes whether shared expensive nodes should avoid fusion that would duplicate work
     */
    public RegionOptimizationRule(boolean preserveSharedExpensiveNodes) {
        this(FuseConfig.trainingDefaults().withPreserveSharedExpensiveNodes(preserveSharedExpensiveNodes));
    }

    /**
     * Creates a region optimization rule.
     *
     * @param config fusion configuration
     */
    public RegionOptimizationRule(FuseConfig config) {
        this(config, CpuFusionConfig.defaults());
    }

    /**
     * Creates a region optimization rule.
     *
     * @param config fusion configuration
     * @param cpuFusionConfig CPU fused-loop configuration
     */
    public RegionOptimizationRule(FuseConfig config, CpuFusionConfig cpuFusionConfig) {
        this.config = Objects.requireNonNull(config, "config cannot be null");
        this.cpuFusionConfig = cpuFusionConfig == null ? CpuFusionConfig.defaults() : cpuFusionConfig;
        this.regionOptimizer = new DefaultRegionOptimizer();
    }

    /**
     * Returns fusion configuration used by this rule.
     *
     * @return fusion configuration
     */
    public FuseConfig config() {
        return config;
    }

    /**
     * Builds optimized regions from the state's partitions.
     *
     * @param state optimizer state with partition metadata
     * @return state with optimized regions attached
     */
    @Override
    public OptimizerState apply(OptimizerState state) {
        Objects.requireNonNull(state, "state cannot be null");
        if (state.partitions().isEmpty()) {
            return state.withOptimizedRegions(List.of());
        }

        List<CompiledNode> compiledNodes = CompiledNode.snapshot(state.graph());
        RegionOptimizationContext context = new RegionOptimizationContext(compiledNodes, config, cpuFusionConfig);
        List<OptimizedRegion> regions = state.partitions().stream()
                .map(partition -> regionOptimizer.optimize(partition, context))
                .toList();
        return state.withOptimizedRegions(regions);
    }
}
