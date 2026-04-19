package graph.optimizer;

import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;

public class GraphOptimizerFixpointTest {
    @Test
    void fixpointStopsOnStructuralEqualityEvenWhenRuleReturnsFreshGraphObjects() {
        AtomicInteger applyCount = new AtomicInteger();
        OptimizationRule cloneRule = graph -> {
            applyCount.incrementAndGet();
            return cloneGraph(graph);
        };

        GraphOptimizer optimizer = new GraphOptimizer(List.of(cloneRule), 1, 8);
        List<Tensor> original = buildGraph();
        List<Tensor> optimized = optimizer.optimize(original);

        assertEquals(1, applyCount.get());
        assertEquals(
                OptimizerFingerprint.of(original),
                OptimizerFingerprint.of(optimized)
        );
        assertNotSame(original.getLast(), optimized.getLast());
    }

    private static List<Tensor> buildGraph() {
        Tensor a = new Tensor(new double[]{1.0}, new int[]{1}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{2.0}, new int[]{1}, null, "b", DataType.FLOAT64);
        Tensor c = new Tensor(new double[]{3.0}, new int[]{1}, null, "c", DataType.FLOAT64);
        return a.add(b).mul(c).topologicalSort();
    }

    private static List<Tensor> cloneGraph(List<Tensor> graph) {
        java.util.IdentityHashMap<Tensor, Tensor> clones = new java.util.IdentityHashMap<>();
        List<Tensor> out = new ArrayList<>(graph.size());
        for (Tensor tensor : graph) {
            out.add(cloneTensor(tensor, clones));
        }
        return out;
    }

    private static Tensor cloneTensor(Tensor tensor, java.util.IdentityHashMap<Tensor, Tensor> clones) {
        Tensor existing = clones.get(tensor);
        if (existing != null) {
            return existing;
        }
        List<Tensor> prev = tensor.getPrevTensors();
        List<Tensor> clonedPrev = null;
        if (prev != null) {
            clonedPrev = new ArrayList<>(prev.size());
            for (Tensor input : prev) {
                clonedPrev.add(cloneTensor(input, clones));
            }
        }
        Tensor clone = new Tensor(
                tensor.getShapeUnsafe().clone(),
                tensor.getStridesUnsafe().clone(),
                tensor.getStorageOffsetUnsafe(),
                clonedPrev,
                tensor.getOperation(),
                tensor.getLabel(),
                tensor.getDataType()
        );
        if (tensor.getOperation() == null) {
            clone.copyDataFrom(tensor);
        }
        clones.put(tensor, clone);
        return clone;
    }
}
