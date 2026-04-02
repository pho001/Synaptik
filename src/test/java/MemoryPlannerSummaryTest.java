import config.optimizer.OptimizerConfig;
import graph.CompiledGraph;
import graph.optimizer.memory.MemoryPlan;
import graph.optimizer.memory.MemoryPlanner;
import graph.optimizer.memory.MemoryPlannerPolicy;
import graph.optimizer.rules.MemoryOptimizerRule;
import org.junit.jupiter.api.Test;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class MemoryPlannerSummaryTest {
    @Test
    void explainAndSummaryExposeReusablePlannerState() {
        Tensor a = Tensor.scalar(10.0);
        Tensor b = Tensor.scalar(2.0);
        Tensor c = Tensor.scalar(5.0);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);
        c.setRequiresGrad(true);

        Tensor graph = a.div(b).div(a.sub(c)).add(b.add(c).mul(a.div(b).div(a.sub(c)))).pow(2);
        CompiledGraph compiled = CompiledGraph.compile(graph, OptimizerConfig.trainingDefaults());

        MemoryPlan plan = MemoryPlanner.plan(compiled.getCompiledGraphAsList(), MemoryPlannerPolicy.defaults());
        String explain = plan.explain();

        assertTrue(plan.summary().reusableIntervalCount() > 0);
        assertTrue(plan.summary().slotCount() > 0);
        assertTrue(explain.contains("MemoryPlanSummary"));
        assertTrue(explain.contains("role="));
        assertTrue(explain.contains("slot="));
    }

    @Test
    void memoryRuleExposesLastPlanSummaryAndExplainHooks() {
        Tensor a = Tensor.scalar(4.0);
        Tensor b = Tensor.scalar(2.0);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor out = a.mul(b).add(a).pow(2.0);
        new MemoryOptimizerRule().apply(CompiledGraph.compile(out, OptimizerConfig.trainingDefaults()).getCompiledGraphAsList());

        assertNotNull(MemoryOptimizerRule.lastPlan());
        assertNotNull(MemoryOptimizerRule.lastSummary());
        assertTrue(MemoryOptimizerRule.lastExplain().contains("MemoryPlanSummary"));
    }
}
