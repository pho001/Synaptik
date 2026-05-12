package graph.optimizer;

import config.compile.CompileConfig;
import graph.CompiledGraph;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.state.OptimizerState;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class GraphOptimizerSinglePassTest {
    @Test
    void appliesConfiguredRuleSequenceExactlyOnce() {
        AtomicInteger firstCount = new AtomicInteger();
        AtomicInteger secondCount = new AtomicInteger();
        OptimizationRule first = state -> {
            firstCount.incrementAndGet();
            return state;
        };
        OptimizationRule second = state -> {
            secondCount.incrementAndGet();
            return state;
        };

        GraphOptimizer optimizer = new GraphOptimizer(List.of(first, second));
        optimizer.optimize(OptimizerState.ofGraph(buildGraph()));

        assertEquals(1, firstCount.get());
        assertEquals(1, secondCount.get());
    }

    @Test
    void compilerPreservesBackendPlanningRegionAndMemoryArtifacts() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor out = a.add(b).relu();

        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference());
        OptimizerState optimized = compiled.compileArtifacts().requireLoweringReadyOptimizerState();

        assertEquals(1, compiled.compileArtifacts().partitions().size());
        assertEquals(1, optimized.optimizedRegions().size());
        assertEquals(PartitionTarget.CPU, optimized.partitions().getFirst().target());
        assertEquals(1, optimized.memoryPlan().structuralView().optimizedRegionIds().size());
    }

    private static List<Tensor> buildGraph() {
        Tensor a = new Tensor(new double[]{1.0}, new int[]{1}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{2.0}, new int[]{1}, null, "b", DataType.FLOAT64);
        return a.add(b).topologicalSort();
    }
}
