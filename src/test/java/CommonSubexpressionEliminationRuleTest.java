import config.optimizer.CseConfig;
import graph.optimizer.rules.CommonSubexpressionEliminationRule;
import operations.Operation;
import operations.noop;
import org.junit.jupiter.api.Test;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class CommonSubexpressionEliminationRuleTest {

    @Test
    void mergesCommutativeAdds() {
        Tensor a = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "a");
        Tensor b = new Tensor(new double[]{3.0, 4.0}, new int[]{2}, null, "b");

        Tensor left = a.add(b);
        Tensor right = b.add(a);
        Tensor root = new Tensor(new int[]{1}, List.of(left, right), new noop(), "root");

        List<Tensor> optimized = new CommonSubexpressionEliminationRule(CseConfig.strictDefaults())
                .apply(root.topologicalSort());

        assertEquals(1, countOps(optimized, Operation.OpType.ADD));
    }

    @Test
    void keepsDistinctSumKeepDimsVariants() {
        Tensor x = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{2, 2}, null, "x");

        Tensor noKeepDims = x.sum(1, false);
        Tensor keepDims = x.sum(1, true);
        Tensor root = new Tensor(new int[]{1}, List.of(noKeepDims, keepDims), new noop(), "root");

        List<Tensor> optimized = new CommonSubexpressionEliminationRule(CseConfig.aggressiveDefaults())
                .apply(root.topologicalSort());

        assertEquals(2, countOps(optimized, Operation.OpType.SUM));
    }

    @Test
    void keepsDistinctPermuteLayoutsInAggressiveMode() {
        Tensor x = new Tensor(
                new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
                new int[]{2, 2, 3},
                null,
                "x"
        );

        Tensor identity = x.permute(0, 1, 2);
        Tensor swapped = x.permute(1, 0, 2);
        Tensor root = new Tensor(new int[]{1}, List.of(identity, swapped), new noop(), "root");

        List<Tensor> optimized = new CommonSubexpressionEliminationRule(CseConfig.aggressiveDefaults())
                .apply(root.topologicalSort());

        assertEquals(2, countOps(optimized, Operation.OpType.PERMUTE));
    }

    private static long countOps(List<Tensor> graph, Operation.OpType opType) {
        return graph.stream()
                .map(Tensor::getOperation)
                .filter(operation -> operation != null && operation.opType() == opType)
                .count();
    }
}
