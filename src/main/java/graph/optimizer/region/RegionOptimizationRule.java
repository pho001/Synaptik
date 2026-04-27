package graph.optimizer.region;

import config.optimizer.FuseConfig;
import graph.CompiledNode;
import graph.optimizer.OptimizationRule;
import graph.optimizer.state.OptimizerState;

import java.util.List;
import java.util.Objects;

public class RegionOptimizationRule implements OptimizationRule {
    private final FuseConfig config;
    private final RegionOptimizer regionOptimizer;

    public RegionOptimizationRule() {
        this(FuseConfig.trainingDefaults());
    }

    public RegionOptimizationRule(boolean preserveSharedExpensiveNodes) {
        this(FuseConfig.trainingDefaults().withPreserveSharedExpensiveNodes(preserveSharedExpensiveNodes));
    }

    public RegionOptimizationRule(FuseConfig config) {
        this.config = Objects.requireNonNull(config, "config cannot be null");
        this.regionOptimizer = new DefaultRegionOptimizer();
    }

    public FuseConfig config() {
        return config;
    }

    @Override
    public OptimizerState apply(OptimizerState state) {
        Objects.requireNonNull(state, "state cannot be null");
        if (state.partitions().isEmpty()) {
            return state.withOptimizedRegions(List.of());
        }

        List<CompiledNode> compiledNodes = CompiledNode.snapshot(state.graph());
        RegionOptimizationContext context = new RegionOptimizationContext(compiledNodes, config);
        List<OptimizedRegion> regions = state.partitions().stream()
                .map(partition -> regionOptimizer.optimize(partition, context))
                .toList();
        return state.withOptimizedRegions(regions);
    }
}
