package graph.optimizer.memory;

import graph.optimizer.OptimizationRule;
import graph.optimizer.state.OptimizerState;
import tensor.DataType;
import tensor.Tensor;
import java.util.List;

/**
 * Optimizer stage that attaches a {@link MemoryPlan} to optimizer state.
 *
 * <p>The rule is controlled by the {@code cg.optimizer.enableMemoryReuse} system property. When disabled, it clears the
 * memory plan. When enabled, it runs {@link MemoryPlanner} and stores the most recent plan in a static volatile
 * diagnostic slot exposed by {@link #lastPlan()}, {@link #lastExplain()}, and {@link #lastSummary()}.
 *
 * <p>The optimizer state transformation is deterministic for stable inputs. The static last-plan diagnostics are a
 * process-wide side effect and should not be used for correctness in concurrent compiles.
 */
public class MemoryOptimizerRule implements OptimizationRule {
    private static final boolean ENABLE_MEMORY_REUSE =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.enableMemoryReuse", "true"));
    private static volatile MemoryPlan lastPlan;
    private final MemoryPlannerPolicy policy;

    /**
     * Creates a memory optimizer with the default policy.
     */
    public MemoryOptimizerRule() {
        this(MemoryPlannerPolicy.defaults());
    }

    /**
     * Creates a memory optimizer with an explicit policy.
     *
     * @param policy memory reuse policy, or {@code null} for defaults
     */
    public MemoryOptimizerRule(MemoryPlannerPolicy policy) {
        this.policy = policy == null ? MemoryPlannerPolicy.defaults() : policy;
    }

    /**
     * Attaches a memory plan to optimizer state.
     *
     * @param state optimizer state after region optimization
     * @return state with memory plan attached, or cleared when memory reuse is disabled
     */
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

    /**
     * Returns the last plan produced by any {@code MemoryOptimizerRule} in this JVM.
     *
     * @return last memory plan, or {@code null} if none has run or planning was disabled
     */
    public static MemoryPlan lastPlan() {
        return lastPlan;
    }

    /**
     * Returns a textual explanation of the last memory plan.
     *
     * @return explanation, or an unavailable marker when no plan exists
     */
    public static String lastExplain() {
        return lastPlan == null ? "MemoryPlan{unavailable}" : lastPlan.explain();
    }

    /**
     * Returns summary metrics for the last memory plan.
     *
     * @return summary, or {@code null} when no plan exists
     */
    public static graph.optimizer.memory.MemoryPlanSummary lastSummary() {
        return lastPlan == null ? null : lastPlan.summary();
    }

    /**
     * Returns the memory planning policy used by this rule.
     *
     * @return memory planner policy
     */
    public MemoryPlannerPolicy policy() {
        return policy;
    }
}
