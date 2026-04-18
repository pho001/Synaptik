package graph.codegen;

import operations.elementwise.unary.mulScalar;
import operations.elementwise.unary.pow;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FusedPlanBuilderCanonicalizationTest {

    @Test
    void canonicalizesPow2ToMulInsideFusedPlan() {
        Tensor input = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor root = new Tensor(new int[]{2}, List.of(input), new pow(2.0), "pow2", DataType.FLOAT64);

        FusedExpressionPlan plan = FusedPlanBuilder.build(List.of(root), List.of(input), root);

        FusedNodePlan node = plan.nodes().getFirst();
        assertEquals(operations.Operation.OpType.MUL, node.opType());
        assertEquals(List.of(0, 0), node.inputRefs());
        assertTrue(node.attributes() instanceof NoAttributes);
    }

    @Test
    void canonicalizesPow1ToNoopInsideFusedPlan() {
        Tensor input = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor root = new Tensor(new int[]{2}, List.of(input), new pow(1.0), "pow1", DataType.FLOAT64);

        FusedExpressionPlan plan = FusedPlanBuilder.build(List.of(root), List.of(input), root);

        FusedNodePlan node = plan.nodes().getFirst();
        assertEquals(operations.Operation.OpType.NOOP, node.opType());
        assertEquals(List.of(0), node.inputRefs());
        assertTrue(node.attributes() instanceof NoAttributes);
    }

    @Test
    void canonicalizesPowZeroToConstScalarInsideFusedPlan() {
        Tensor input = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor root = new Tensor(new int[]{2}, List.of(input), new pow(0.0), "pow0", DataType.FLOAT64);

        FusedExpressionPlan plan = FusedPlanBuilder.build(List.of(root), List.of(input), root);

        FusedNodePlan node = plan.nodes().getFirst();
        assertEquals(operations.Operation.OpType.CONST_SCALAR, node.opType());
        assertEquals(List.of(), node.inputRefs());
        assertEquals(1.0d, ((ScalarDoubleAttribute) node.attributes()).value());
    }

    @Test
    void canonicalizesPowNegativeOneToInvInsideFusedPlan() {
        Tensor input = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor root = new Tensor(new int[]{2}, List.of(input), new pow(-1.0), "powNeg1", DataType.FLOAT64);

        FusedExpressionPlan plan = FusedPlanBuilder.build(List.of(root), List.of(input), root);

        FusedNodePlan node = plan.nodes().getFirst();
        assertEquals(operations.Operation.OpType.INV, node.opType());
        assertEquals(List.of(0), node.inputRefs());
        assertTrue(node.attributes() instanceof NoAttributes);
    }

    @Test
    void canonicalizesMulScalarZeroToConstScalarInsideFusedPlan() {
        Tensor input = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor root = new Tensor(new int[]{2}, List.of(input), new mulScalar(0.0), "mul0", DataType.FLOAT64);

        FusedExpressionPlan plan = FusedPlanBuilder.build(List.of(root), List.of(input), root);

        FusedNodePlan node = plan.nodes().getFirst();
        assertEquals(operations.Operation.OpType.CONST_SCALAR, node.opType());
        assertEquals(List.of(), node.inputRefs());
        assertEquals(0.0d, ((ScalarDoubleAttribute) node.attributes()).value());
    }

    @Test
    void canonicalizesMulScalarOneToNoopInsideFusedPlan() {
        Tensor input = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor root = new Tensor(new int[]{2}, List.of(input), new mulScalar(1.0), "mul1", DataType.FLOAT64);

        FusedExpressionPlan plan = FusedPlanBuilder.build(List.of(root), List.of(input), root);

        FusedNodePlan node = plan.nodes().getFirst();
        assertEquals(operations.Operation.OpType.NOOP, node.opType());
        assertEquals(List.of(0), node.inputRefs());
        assertTrue(node.attributes() instanceof NoAttributes);
    }

    @Test
    void canonicalizesMulScalarNegativeOneToNegInsideFusedPlan() {
        Tensor input = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor root = new Tensor(new int[]{2}, List.of(input), new mulScalar(-1.0), "mulNeg1", DataType.FLOAT64);

        FusedExpressionPlan plan = FusedPlanBuilder.build(List.of(root), List.of(input), root);

        FusedNodePlan node = plan.nodes().getFirst();
        assertEquals(operations.Operation.OpType.NEG, node.opType());
        assertEquals(List.of(0), node.inputRefs());
        assertTrue(node.attributes() instanceof NoAttributes);
    }
}
