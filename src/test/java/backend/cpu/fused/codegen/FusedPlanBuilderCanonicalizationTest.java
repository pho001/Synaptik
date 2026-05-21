package backend.cpu.fused.codegen;

import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import operations.elementwise.unary.mulScalar;
import operations.elementwise.unary.pow;
import operations.elementwise.binary.powTensor;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import graph.compile.intent.BackendIntentPlan;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FusedPlanBuilderCanonicalizationTest {

    @Test
    void canonicalizesPow2ToMulInsideFusedPlan() {
        Tensor input = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor root = new Tensor(new int[]{2}, List.of(input), new pow(2.0), "pow2", DataType.FLOAT64);

        FusedExpressionPlan plan = plan(root, List.of(input));

        FusedNodePlan node = plan.nodes().getFirst();
        assertEquals(operations.Operation.OpType.MUL, node.opType());
        assertEquals(List.of(0, 0), node.inputRefs());
        assertTrue(node.attributes() instanceof NoAttributes);
    }

    @Test
    void canonicalizesPow1ToNoopInsideFusedPlan() {
        Tensor input = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor root = new Tensor(new int[]{2}, List.of(input), new pow(1.0), "pow1", DataType.FLOAT64);

        FusedExpressionPlan plan = plan(root, List.of(input));

        FusedNodePlan node = plan.nodes().getFirst();
        assertEquals(operations.Operation.OpType.NOOP, node.opType());
        assertEquals(List.of(0), node.inputRefs());
        assertTrue(node.attributes() instanceof NoAttributes);
    }

    @Test
    void canonicalizesPowZeroToConstScalarInsideFusedPlan() {
        Tensor input = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor root = new Tensor(new int[]{2}, List.of(input), new pow(0.0), "pow0", DataType.FLOAT64);

        FusedExpressionPlan plan = plan(root, List.of(input));

        FusedNodePlan node = plan.nodes().getFirst();
        assertEquals(operations.Operation.OpType.CONST_SCALAR, node.opType());
        assertEquals(List.of(), node.inputRefs());
        assertEquals(1.0d, ((ScalarDoubleAttribute) node.attributes()).value());
    }

    @Test
    void canonicalizesPowNegativeOneToInvInsideFusedPlan() {
        Tensor input = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor root = new Tensor(new int[]{2}, List.of(input), new pow(-1.0), "powNeg1", DataType.FLOAT64);

        FusedExpressionPlan plan = plan(root, List.of(input));

        FusedNodePlan node = plan.nodes().getFirst();
        assertEquals(operations.Operation.OpType.INV, node.opType());
        assertEquals(List.of(0), node.inputRefs());
        assertTrue(node.attributes() instanceof NoAttributes);
    }

    @Test
    void tensorPowIsKeptAsBinaryPowTensorInsideFusedPlan() {
        Tensor base = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "base", DataType.FLOAT64);
        Tensor exponent = new Tensor(new double[]{3.0, 4.0}, new int[]{2}, null, "exponent", DataType.FLOAT64);
        Tensor root = new Tensor(new int[]{2}, List.of(base, exponent), new powTensor(), "powTensor", DataType.FLOAT64);

        FusedExpressionPlan plan = plan(root, List.of(base, exponent));

        FusedNodePlan node = plan.nodes().getFirst();
        assertEquals(operations.Operation.OpType.POW_TENSOR, node.opType());
        assertEquals(List.of(0, 1), node.inputRefs());
        assertTrue(node.attributes() instanceof NoAttributes);
    }

    @Test
    void canonicalizesMulScalarZeroToConstScalarInsideFusedPlan() {
        Tensor input = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor root = new Tensor(new int[]{2}, List.of(input), new mulScalar(0.0), "mul0", DataType.FLOAT64);

        FusedExpressionPlan plan = plan(root, List.of(input));

        FusedNodePlan node = plan.nodes().getFirst();
        assertEquals(operations.Operation.OpType.CONST_SCALAR, node.opType());
        assertEquals(List.of(), node.inputRefs());
        assertEquals(0.0d, ((ScalarDoubleAttribute) node.attributes()).value());
    }

    @Test
    void canonicalizesMulScalarOneToNoopInsideFusedPlan() {
        Tensor input = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor root = new Tensor(new int[]{2}, List.of(input), new mulScalar(1.0), "mul1", DataType.FLOAT64);

        FusedExpressionPlan plan = plan(root, List.of(input));

        FusedNodePlan node = plan.nodes().getFirst();
        assertEquals(operations.Operation.OpType.NOOP, node.opType());
        assertEquals(List.of(0), node.inputRefs());
        assertTrue(node.attributes() instanceof NoAttributes);
    }

    @Test
    void canonicalizesMulScalarNegativeOneToNegInsideFusedPlan() {
        Tensor input = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor root = new Tensor(new int[]{2}, List.of(input), new mulScalar(-1.0), "mulNeg1", DataType.FLOAT64);

        FusedExpressionPlan plan = plan(root, List.of(input));

        FusedNodePlan node = plan.nodes().getFirst();
        assertEquals(operations.Operation.OpType.NEG, node.opType());
        assertEquals(List.of(0), node.inputRefs());
        assertTrue(node.attributes() instanceof NoAttributes);
    }

    private static FusedExpressionPlan plan(Tensor root, List<Tensor> externalInputs) {
        List<Tensor> graph = root.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph, BackendIntentPlan.empty());
        List<Integer> externalInputNodeIds = externalInputs.stream()
                .map(graph::indexOf)
                .toList();
        return FusedPlanBuilder.build(
                List.of(graph.indexOf(root)),
                externalInputNodeIds,
                compiledNodes::get,
                CompiledTensorDescriptorBuilder.build(compiledNodes)
        );
    }
}
