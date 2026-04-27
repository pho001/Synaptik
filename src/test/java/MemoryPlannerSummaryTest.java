import config.optimizer.OptimizerConfig;
import graph.CompiledGraph;
import graph.optimizer.OptimizationRule;
import graph.optimizer.OptimizerFactory;
import graph.optimizer.memory.MemoryPlan;
import graph.optimizer.memory.MemoryPlanner;
import graph.optimizer.memory.MemoryPlannerPolicy;
import graph.optimizer.memory.MemoryOptimizerRule;
import org.junit.jupiter.api.Test;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertEquals;
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
        assertTrue(plan.summary().peakReusableBytes() > 0);
        assertTrue(plan.summary().allocatedSlotBytes() > 0);
        assertTrue(plan.summary().savedForwardCount() > 0);
        assertTrue(plan.summary().peakBackwardLiveBytes() > 0);
        assertTrue(plan.summary().reusableFreshAllocationCount() > 0);
        assertTrue(plan.summary().reuseHitRate() >= 0.0d);
        assertTrue(plan.summary().toMetricMap().containsKey("allocatedSlotBytes"));
        assertTrue(explain.contains("=== MemoryPlan Summary ==="));
        assertTrue(explain.contains("allocatedSlotBytes="));
        assertTrue(explain.contains("peakReusableBytes="));
        assertTrue(explain.contains("peakSavedForwardBytes="));
        assertTrue(explain.contains("peakGradientTargetBytes="));
        assertTrue(explain.contains("=== Slot Assignment ==="));
        assertTrue(explain.contains("=== Node Assignment ==="));
        assertTrue(explain.contains("=== Saved Forward Values ==="));
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
        assertTrue(MemoryOptimizerRule.lastExplain().contains("=== MemoryPlan Summary ==="));
    }

    @Test
    void largerBufferReusePolicyCanReduceSlotCount() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "a");
        Tensor b = new Tensor(new double[]{5, 6, 7, 8}, new int[]{2, 2}, null, "b");
        Tensor c = new Tensor(new double[]{9, 10}, new int[]{2}, null, "c");

        Tensor t1 = a.add(b);          // size 4
        Tensor t2 = t1.sum(1, true);   // size 2
        Tensor out = t2.add(c.reshape(2, 1)); // size 2

        var graph = CompiledGraph.compile(out, OptimizerConfig.noOptimization()).getCompiledGraphAsList();
        MemoryPlan strict = MemoryPlanner.plan(graph, new MemoryPlannerPolicy(true, false, false, 1));
        MemoryPlan flexible = MemoryPlanner.plan(graph, new MemoryPlannerPolicy(true, false, true, 1));

        assertTrue(strict.summary().slotCount() >= flexible.summary().slotCount());
        assertTrue(strict.summary().reuseCount() <= flexible.summary().reuseCount());
    }

    @Test
    void minReusableBufferSizePolicyCanExcludeSmallTemporaries() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "a");
        Tensor b = new Tensor(new double[]{5, 6, 7, 8}, new int[]{2, 2}, null, "b");

        Tensor t1 = a.add(b);          // size 4
        Tensor out = t1.sum(1, true);  // size 2

        var graph = CompiledGraph.compile(out, OptimizerConfig.noOptimization()).getCompiledGraphAsList();
        MemoryPlan smallAllowed = MemoryPlanner.plan(graph, new MemoryPlannerPolicy(true, false, false, 1));
        MemoryPlan smallExcluded = MemoryPlanner.plan(graph, new MemoryPlannerPolicy(true, false, false, 3));

        assertTrue(smallAllowed.summary().reusableIntervalCount() >= smallExcluded.summary().reusableIntervalCount());
        assertTrue(smallAllowed.summary().slotCount() >= smallExcluded.summary().slotCount());
    }

    @Test
    void memoryRuleUsesInjectedPolicy() {
        MemoryPlannerPolicy policy = new MemoryPlannerPolicy(true, false, true, 8);
        MemoryOptimizerRule rule = new MemoryOptimizerRule(policy);
        assertEquals(policy, rule.policy());
    }

    @Test
    void optimizerFactoryPassesConfiguredMemoryPolicyToMemStage() {
        var optimizerConfig = OptimizerConfig.trainingDefaults().withMemory(
                new config.optimizer.MemoryConfig(false, false, true, 16)
        );

        MemoryOptimizerRule memRule = null;
        for (OptimizationRule rule : OptimizerFactory.createRules(optimizerConfig)) {
            if (rule instanceof MemoryOptimizerRule found) {
                memRule = found;
                break;
            }
        }

        assertNotNull(memRule);
        assertEquals(
                MemoryPlannerPolicy.fromConfig(optimizerConfig.memory()),
                memRule.policy()
        );
    }

    @Test
    void defaultPolicyRemainsConservative() {
        MemoryPlannerPolicy defaults = MemoryPlannerPolicy.defaults();
        assertTrue(defaults.separateForwardBackwardPools());
        assertTrue(!defaults.allowCrossPhaseReuse());
        assertTrue(!defaults.allowLargerBufferReuse());
        assertEquals(1, defaults.minReusableBufferSize());
    }

    @Test
    void largerBufferReuseCanReduceSlotCountForIrregularIntervals() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "a");
        Tensor b = new Tensor(new double[]{5, 6, 7, 8}, new int[]{2, 2}, null, "b");
        Tensor c = new Tensor(new double[]{9, 10}, new int[]{2}, null, "c");

        Tensor t1 = a.add(b);                 // size 4
        Tensor t2 = t1.sum(1, true);         // size 2
        Tensor out = t2.add(c.reshape(2, 1)); // size 2

        var graph = CompiledGraph.compile(out, OptimizerConfig.noOptimization()).getCompiledGraphAsList();
        MemoryPlan strict = MemoryPlanner.plan(graph, new MemoryPlannerPolicy(true, false, false, 1));
        MemoryPlan flexible = MemoryPlanner.plan(graph, new MemoryPlannerPolicy(true, false, true, 1));

        assertTrue(strict.summary().slotCount() >= flexible.summary().slotCount());
        assertTrue(strict.summary().reuseCount() <= flexible.summary().reuseCount());
    }

    @Test
    void minReusableBufferSizeCanExcludeSmallIntervalsFromReusePlanning() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "a");
        Tensor b = new Tensor(new double[]{5, 6, 7, 8}, new int[]{2, 2}, null, "b");

        Tensor t1 = a.add(b);         // size 4
        Tensor out = t1.sum(1, true); // size 2

        var graph = CompiledGraph.compile(out, OptimizerConfig.noOptimization()).getCompiledGraphAsList();
        MemoryPlan allIntervals = MemoryPlanner.plan(graph, new MemoryPlannerPolicy(true, false, false, 1));
        MemoryPlan largerOnly = MemoryPlanner.plan(graph, new MemoryPlannerPolicy(true, false, false, 3));

        assertTrue(allIntervals.summary().reusableIntervalCount() >= largerOnly.summary().reusableIntervalCount());
        assertTrue(allIntervals.summary().slotCount() >= largerOnly.summary().slotCount());
    }
}
