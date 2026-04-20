package graph.optimizer;

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
        OptimizationRule first = graph -> {
            firstCount.incrementAndGet();
            return graph;
        };
        OptimizationRule second = graph -> {
            secondCount.incrementAndGet();
            return graph;
        };

        GraphOptimizer optimizer = new GraphOptimizer(List.of(first, second));
        optimizer.optimize(buildGraph());

        assertEquals(1, firstCount.get());
        assertEquals(1, secondCount.get());
    }

    private static List<Tensor> buildGraph() {
        Tensor a = new Tensor(new double[]{1.0}, new int[]{1}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{2.0}, new int[]{1}, null, "b", DataType.FLOAT64);
        return a.add(b).topologicalSort();
    }
}
