import config.optimizer.CseConfig;
import graph.optimizer.cse.CommonSubexpressionEliminationRule;
import operations.Operation;
import operations.layout.noop;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
    void keepsDistinctReduceAnyAxesWithSameOutputShape() {
        Tensor x = new Tensor(new byte[]{1, 0, 0, 1}, new int[]{2, 2}, null, "x");

        Tensor axis0 = x.any(0, false);
        Tensor axis1 = x.any(1, false);
        Tensor root = new Tensor(new int[]{1}, List.of(axis0, axis1), new noop(), "root");

        List<Tensor> optimized = new CommonSubexpressionEliminationRule(CseConfig.aggressiveDefaults())
                .apply(root.topologicalSort());

        assertEquals(2, countOps(optimized, Operation.OpType.REDUCE_ANY));
    }

    @Test
    void keepsDistinctReduceAllKeepDimsVariants() {
        Tensor x = new Tensor(new byte[]{1, 0, 1, 1}, new int[]{2, 2}, null, "x");

        Tensor noKeepDims = x.all(1, false);
        Tensor keepDims = x.all(1, true);
        Tensor root = new Tensor(new int[]{1}, List.of(noKeepDims, keepDims), new noop(), "root");

        List<Tensor> optimized = new CommonSubexpressionEliminationRule(CseConfig.aggressiveDefaults())
                .apply(root.topologicalSort());

        assertEquals(2, countOps(optimized, Operation.OpType.REDUCE_ALL));
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

    @Test
    void handlesBoolScalarLeafInWhereGraph() {
        Tensor mask = new Tensor(new byte[]{1}, new int[]{1}, null, "mask");
        Tensor x = new Tensor(new double[]{2.0}, new int[]{1}, null, "x");
        Tensor masked = Tensor.where(mask, x, Tensor.zerosLike(x));
        Tensor root = new Tensor(new int[]{1}, List.of(masked), new noop(), "root");

        List<Tensor> optimized = new CommonSubexpressionEliminationRule(CseConfig.aggressiveDefaults())
                .apply(root.topologicalSort());

        assertEquals(1, countOps(optimized, Operation.OpType.WHERE));
    }

    @Test
    void keepsScalarLeafSignaturesDtypeAware() {
        Tensor x = new Tensor(new double[]{2.0}, new int[]{1}, null, "x");
        Tensor floatOne = new Tensor(new float[]{1.0f}, new int[]{1}, null, "floatOne", DataType.FLOAT32);
        Tensor doubleOne = Tensor.scalar(1.0d, DataType.FLOAT64);

        Tensor floatBranch = x.add(floatOne);
        Tensor doubleBranch = x.add(doubleOne);
        Tensor root = new Tensor(new int[]{1}, List.of(floatBranch, doubleBranch), new noop(), "root");

        List<Tensor> optimized = new CommonSubexpressionEliminationRule(CseConfig.aggressiveDefaults())
                .apply(root.topologicalSort());

        assertEquals(2, countOps(optimized, Operation.OpType.ADD));
    }

    @Test
    void parameterKeyExplicitlyClassifiesEveryOperationType() throws Exception {
        String source = Files.readString(Path.of(
                "src/main/java/graph/optimizer/cse/CommonSubexpressionEliminationRule.java"
        ));
        String parameterKey = source.substring(
                source.indexOf("private SignatureComponent parameterKey"),
                source.indexOf("private SignatureComponent conv2dSignature")
        );

        assertTrue(!parameterKey.contains("default ->"));
        for (Operation.OpType opType : Operation.OpType.values()) {
            assertTrue(parameterKey.contains("case " + opType.name())
                            || parameterKey.contains(", " + opType.name())
                            || parameterKey.contains(opType.name() + ",")
                            || parameterKey.contains(opType.name() + " ->"),
                    "CSE parameterKey must explicitly classify " + opType);
        }
    }

    private static long countOps(List<Tensor> graph, Operation.OpType opType) {
        return graph.stream()
                .map(Tensor::getOperation)
                .filter(operation -> operation != null && operation.opType() == opType)
                .count();
    }
}
