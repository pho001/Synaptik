package graph.optimizer.memory;

import graph.optimizer.OptimizationRule;
import graph.optimizer.state.OptimizerState;
import tensor.DataType;
import tensor.Tensor;
import java.util.List;

public class MemoryOptimizerRule implements OptimizationRule {
    private static final boolean ENABLE_MEMORY_REUSE =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.enableMemoryReuse", "true"));
    private static volatile MemoryPlan lastPlan;
    private final MemoryPlannerPolicy policy;

    public MemoryOptimizerRule() {
        this(MemoryPlannerPolicy.defaults());
    }

    public MemoryOptimizerRule(MemoryPlannerPolicy policy) {
        this.policy = policy == null ? MemoryPlannerPolicy.defaults() : policy;
    }

    @Override
    public OptimizerState apply(OptimizerState state) {
        List<Tensor> sortedGraph = state.graph();
        if (!ENABLE_MEMORY_REUSE || sortedGraph == null || sortedGraph.isEmpty()) {
            return state.withMemoryPlan(null);
        }

        DataType graphType = validateUniformGraphType(sortedGraph);
        MemoryPlan plan = MemoryPlanner.plan(state, policy);
        lastPlan = plan;
        if (graphType == null) {
            return state.withMemoryPlan(plan);
        }
        return state.withMemoryPlan(plan);
    }

    private DataType validateUniformGraphType(List<Tensor> sortedGraph) {
        DataType graphType = sortedGraph.get(0).getDataType();
        for (Tensor tensor : sortedGraph) {
            if (tensor.getDataType() != graphType) {
                return null;
            }
        }
        return graphType;
    }

    public static MemoryPlan lastPlan() {
        return lastPlan;
    }

    public static String lastExplain() {
        return lastPlan == null ? "MemoryPlan{unavailable}" : lastPlan.explain();
    }

    public static graph.optimizer.memory.MemoryPlanSummary lastSummary() {
        return lastPlan == null ? null : lastPlan.summary();
    }

    public MemoryPlannerPolicy policy() {
        return policy;
    }
}
