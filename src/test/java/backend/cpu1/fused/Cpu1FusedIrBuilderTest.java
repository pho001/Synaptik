package backend.cpu1.fused;

import backend.cpu1.fused.ir.Cpu1FusedAccessKind;
import backend.cpu1.fused.ir.Cpu1FusedExpressionPlan;
import backend.cpu1.fused.ir.Cpu1FusedInputPlan;
import backend.cpu1.fused.ir.Cpu1FusedIrBuilder;
import backend.cpu1.fused.ir.Cpu1FusedNodePlan;
import backend.cpu1.storage.Cpu1StorageAccessKind;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPlan;
import operations.Operation;
import operations.elementwise.unary.pow;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Cpu1FusedIrBuilderTest {
    @Test
    void canonicalizesPowTwoToMul() {
        Tensor input = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, new int[]{3}, null, "input", DataType.FLOAT32);
        Tensor pow2 = new Tensor(new int[]{3}, List.of(input), new pow(2.0f), "pow2", DataType.FLOAT32);
        Tensor output = pow2.relu();
        Fixture fixture = fixture(output);
        int powNodeId = fixture.nodeId(Operation.OpType.POW);
        int reluNodeId = fixture.nodeId(Operation.OpType.RELU);

        Cpu1FusedExpressionPlan plan = Cpu1FusedIrBuilder.build(
                List.of(powNodeId, reluNodeId),
                fixture::compiledNode,
                fixture.descriptorIndex()
        );

        Cpu1FusedNodePlan powPlan = plan.nodes().getFirst();
        assertEquals(Operation.OpType.MUL, powPlan.opType());
        assertEquals(2, powPlan.inputRefs().size());
        assertEquals(powPlan.inputRefs().getFirst(), powPlan.inputRefs().getLast());
        assertEquals(Operation.OpType.RELU, plan.nodes().getLast().opType());
        assertEquals(plan.nodes().getLast().outputRef(), plan.outputRef());
    }

    @Test
    void canonicalizesPowMinusTwoToMulThenInv() {
        Tensor input = new Tensor(new float[]{1.0f, 2.0f, 4.0f}, new int[]{3}, null, "input", DataType.FLOAT32);
        Tensor powMinusTwo = new Tensor(new int[]{3}, List.of(input), new pow(-2.0), "powMinusTwo", DataType.FLOAT32);
        Tensor output = powMinusTwo.relu();
        Fixture fixture = fixture(output);
        int powNodeId = fixture.nodeId(Operation.OpType.POW);
        int reluNodeId = fixture.nodeId(Operation.OpType.RELU);

        Cpu1FusedExpressionPlan plan = Cpu1FusedIrBuilder.build(
                List.of(powNodeId, reluNodeId),
                fixture::compiledNode,
                fixture.descriptorIndex()
        );

        assertEquals(3, plan.nodes().size());
        Cpu1FusedNodePlan mulPlan = plan.nodes().get(0);
        assertEquals(0, mulPlan.index());
        assertEquals(powNodeId, mulPlan.nodeId());
        assertEquals(Operation.OpType.MUL, mulPlan.opType());
        assertEquals(List.of(0, 0), mulPlan.inputRefs());
        assertEquals(1, mulPlan.outputRef());

        Cpu1FusedNodePlan invPlan = plan.nodes().get(1);
        assertEquals(1, invPlan.index());
        assertEquals(powNodeId, invPlan.nodeId());
        assertEquals(Operation.OpType.INV, invPlan.opType());
        assertEquals(List.of(mulPlan.outputRef()), invPlan.inputRefs());
        assertEquals(2, invPlan.outputRef());

        Cpu1FusedNodePlan reluPlan = plan.nodes().get(2);
        assertEquals(2, reluPlan.index());
        assertEquals(reluNodeId, reluPlan.nodeId());
        assertEquals(Operation.OpType.RELU, reluPlan.opType());
        assertEquals(List.of(invPlan.outputRef()), reluPlan.inputRefs());
        assertEquals(reluPlan.outputRef(), plan.outputRef());
        assertTrue(plan.nodes().stream().noneMatch(node -> node.opType() == Operation.OpType.POW));
    }

    @Test
    void buildsBroadcastEffectiveStrides() {
        Tensor left = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "left",
                DataType.FLOAT32
        );
        Tensor bias = new Tensor(new float[]{10.0f, 20.0f, 30.0f}, new int[]{3}, null, "bias", DataType.FLOAT32);
        Tensor output = left.add(bias);
        Fixture fixture = fixture(output);
        int addNodeId = fixture.nodeId(Operation.OpType.ADD);
        int biasNodeId = fixture.nodeId("bias");

        Cpu1FusedExpressionPlan plan = Cpu1FusedIrBuilder.build(
                List.of(addNodeId),
                fixture::compiledNode,
                fixture.descriptorIndex()
        );

        Cpu1FusedInputPlan biasPlan = plan.inputs().stream()
                .filter(input -> input.nodeId() == biasNodeId)
                .findFirst()
                .orElseThrow();
        assertArrayEquals(new int[]{3}, biasPlan.shape());
        assertArrayEquals(new int[]{2, 3}, biasPlan.logicalOutputShape());
        assertArrayEquals(new int[]{3, 1}, biasPlan.logicalOutputDenseStrides());
        assertArrayEquals(new int[]{0, 1}, biasPlan.effectiveStrides());
        assertEquals(Cpu1FusedAccessKind.BROADCAST_STRIDED, biasPlan.accessKind());
        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS, biasPlan.baseAccessPlan().kind());
        assertArrayEquals(new int[]{3}, biasPlan.baseAccessPlan().shape());
        assertArrayEquals(new int[]{1}, biasPlan.baseAccessPlan().strides());
        assertEquals(Cpu1StorageAccessKind.BROADCAST, biasPlan.logicalAccessPlan().kind());
        assertArrayEquals(new int[]{2, 3}, biasPlan.logicalAccessPlan().shape());
        assertArrayEquals(new int[]{0, 1}, biasPlan.logicalAccessPlan().strides());
        assertTrue(plan.inputs().stream()
                .filter(input -> input.nodeId() != biasNodeId)
                .allMatch(Cpu1FusedInputPlan::isLinearAccess));
    }

    @Test
    void capturesScalarParametersForSupportedScalarOps() {
        Tensor input = new Tensor(new float[]{-1.0f, 0.5f, 2.0f}, new int[]{3}, null, "input", DataType.FLOAT32);
        Tensor output = input.mul(0.25).clampMin(0.125);
        Fixture fixture = fixture(output);
        int mulScalarNodeId = fixture.nodeId(Operation.OpType.MUL_SCALAR);
        int clampMinNodeId = fixture.nodeId(Operation.OpType.CLAMP_MIN);

        Cpu1FusedExpressionPlan plan = Cpu1FusedIrBuilder.build(
                List.of(mulScalarNodeId, clampMinNodeId),
                fixture::compiledNode,
                fixture.descriptorIndex()
        );

        Cpu1FusedNodePlan mulScalar = plan.nodes().getFirst();
        assertEquals(Operation.OpType.MUL_SCALAR, mulScalar.opType());
        assertTrue(mulScalar.scalarParameter().present());
        assertEquals(0.25f, mulScalar.scalarParameter().f32());
        assertEquals(0.25d, mulScalar.scalarParameter().f64());

        Cpu1FusedNodePlan clampMin = plan.nodes().getLast();
        assertEquals(Operation.OpType.CLAMP_MIN, clampMin.opType());
        assertTrue(clampMin.scalarParameter().present());
        assertEquals(0.125f, clampMin.scalarParameter().f32());
        assertEquals(0.125d, clampMin.scalarParameter().f64());
        assertEquals(clampMin.outputRef(), plan.outputRef());
    }

    private static Fixture fixture(Tensor output) {
        List<CompiledNode> nodes = CompiledNode.snapshot(output.topologicalSort(), BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        return new Fixture(nodes, descriptorIndex);
    }

    private record Fixture(
            List<CompiledNode> nodes,
            CompiledTensorDescriptorIndex descriptorIndex
    ) {
        CompiledNode compiledNode(int nodeId) {
            return nodes.stream()
                    .filter(node -> node.id() == nodeId)
                    .findFirst()
                    .orElseThrow();
        }

        int nodeId(Operation.OpType opType) {
            return nodes.stream()
                    .filter(node -> node.operation() != null && node.operation().opType() == opType)
                    .findFirst()
                    .orElseThrow()
                    .id();
        }

        int nodeId(String label) {
            return nodes.stream()
                    .filter(node -> node.label().equals(label))
                    .findFirst()
                    .orElseThrow()
                    .id();
        }
    }
}
