package backend.cpu.partition;

import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;

import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.compile.planning.partition.PartitionPlanningContext;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.options.Pool2dOptions;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class CpuBackendPartitionCapabilityTest {
    private final CpuBackendPartitionCapability adapter = new CpuBackendPartitionCapability();

    @Test
    void keepsRepresentativeDeferredFamiliesEnabled() {
        Tensor a = Tensor.scalar(1.0);
        Tensor b = Tensor.scalar(2.0);
        Tensor where = Tensor.where(a.lessThan(b), a, b);
        Tensor reshape = where.reshape(1);
        Tensor select = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "selectInput", DataType.FLOAT64).select(0, 1);

        Tensor poolInput = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        }, new int[]{1, 1, 4, 4}, null, "poolInput", DataType.FLOAT64);
        Tensor maxPool = poolInput.maxPool2d(Pool2dOptions.square(2));

        Tensor x = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "x", DataType.FLOAT64);
        Tensor gamma = new Tensor(new double[]{1, 1}, new int[]{2}, null, "gamma", DataType.FLOAT64);
        Tensor beta = new Tensor(new double[]{0, 0}, new int[]{2}, null, "beta", DataType.FLOAT64);
        Tensor layerNorm = x.layerNorm(gamma, beta, 1e-5);

        PartitionPlanningContext context = contextOf(List.of(where, reshape, select, maxPool, layerNorm));

        assertTrue(adapter.canExecute(nodeFor(where, context), context));
        assertTrue(adapter.canExecute(nodeFor(reshape, context), context));
        assertTrue(adapter.canExecute(nodeFor(select, context), context));
        assertTrue(adapter.canExecute(nodeFor(maxPool, context), context));
        assertTrue(adapter.canExecute(nodeFor(layerNorm, context), context));
    }

    @Test
    void allowsReductionWhenItsProducerFamilyIsAllowed() {
        Tensor poolInput = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        }, new int[]{1, 1, 4, 4}, null, "poolInput", DataType.FLOAT64);
        Tensor reduced = poolInput.maxPool2d(Pool2dOptions.square(2)).sum();

        PartitionPlanningContext context = contextOf(List.of(reduced));
        assertTrue(adapter.canExecute(nodeFor(reduced, context), context));
    }

    @Test
    void keepsRepresentativeAllowedFamiliesEnabled() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5, 6, 7, 8}, new int[]{4}, null, "b", DataType.FLOAT64);
        Tensor add = a.add(b);
        Tensor sum = add.sum();

        Tensor m1 = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "m1", DataType.FLOAT64);
        Tensor m2 = new Tensor(new double[]{5, 6, 7, 8}, new int[]{2, 2}, null, "m2", DataType.FLOAT64);
        Tensor matmul = m1.matmul(m2);

        PartitionPlanningContext context = contextOf(List.of(sum, matmul));

        assertTrue(adapter.canExecute(nodeFor(add, context), context));
        assertTrue(adapter.canExecute(nodeFor(sum, context), context));
        assertTrue(adapter.canExecute(nodeFor(matmul, context), context));
    }

    private static PartitionPlanningContext contextOf(List<Tensor> roots) {
        List<Tensor> graph = new ArrayList<>();
        for (Tensor root : roots) {
            for (Tensor tensor : root.topologicalSort()) {
                if (!graph.contains(tensor)) {
                    graph.add(tensor);
                }
            }
        }
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        return new PartitionPlanningContext(
                false,
                compiledNodes,
                CompiledTensorDescriptorBuilder.build(compiledNodes),
                consumers(compiledNodes)
        );
    }

    private static CompiledNode nodeFor(Tensor tensor, PartitionPlanningContext context) {
        for (CompiledNode node : context.compiledNodes()) {
            if (node.label().equals(tensor.getLabel())
                    && ((node.operation() == null && tensor.getOperation() == null)
                    || (node.operation() != null
                    && tensor.getOperation() != null
                    && node.operation().opType() == tensor.getOperation().opType()))) {
                return node;
            }
        }
        throw new IllegalStateException("Missing compiled node for tensor " + tensor.getLabel());
    }

    private static Map<Integer, List<CompiledNode>> consumers(List<CompiledNode> graph) {
        Map<Integer, List<CompiledNode>> consumers = new HashMap<>();
        for (CompiledNode node : graph) {
            consumers.computeIfAbsent(node.id(), ignored -> new ArrayList<>());
        }
        for (CompiledNode node : graph) {
            for (int inputId : node.inputIds()) {
                consumers.computeIfAbsent(inputId, ignored -> new ArrayList<>()).add(node);
            }
        }
        return consumers;
    }
}
