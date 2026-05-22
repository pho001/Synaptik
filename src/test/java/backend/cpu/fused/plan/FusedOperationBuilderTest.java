package backend.cpu.fused.plan;

import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweringFamily;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import org.junit.jupiter.api.Test;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import graph.compile.intent.BackendIntentPlan;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class FusedOperationBuilderTest {
    @Test
    void buildsFusedPreparationFromLoweredExecutionUnit() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor out = add.relu();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(out.topologicalSort(), BackendIntentPlan.empty());

        FusedOperationPreparation preparation = FusedOperationBuilder.build(
                new LoweredExecutionUnit("unit", LoweringFamily.FUSED_NATIVE, List.of(2, 3), List.of(0, 1)),
                compiledNodes::get,
                CompiledTensorDescriptorBuilder.build(compiledNodes)
        );

        assertEquals(Operation.OpType.FUSED, preparation.operation().opType());
        assertEquals(List.of(0, 1), preparation.runtimeInputNodeIds());
    }

    @Test
    void cpuFusedExecutionStillUsesOperationOpTypeFused() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "builderA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "builderB", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor out = add.relu();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(out.topologicalSort(), BackendIntentPlan.empty());

        FusedOperationPreparation preparation = FusedOperationBuilder.build(
                new LoweredExecutionUnit("unit", LoweringFamily.FUSED_NATIVE, List.of(2, 3), List.of(0, 1)),
                compiledNodes::get,
                CompiledTensorDescriptorBuilder.build(compiledNodes)
        );

        assertEquals(Operation.OpType.FUSED, preparation.operation().opType());
    }

    @Test
    void resolvesViewExternalInputsToExternalValueNode() {
        Tensor base = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "base", DataType.FLOAT32);
        Tensor out = base.select(0, 1).relu().exp();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(out.topologicalSort(), BackendIntentPlan.empty());

        FusedOperationPreparation preparation = FusedOperationBuilder.build(
                new LoweredExecutionUnit("unit", LoweringFamily.FUSED_NATIVE, List.of(2, 3), List.of(0)),
                compiledNodes::get,
                CompiledTensorDescriptorBuilder.build(compiledNodes)
        );

        assertEquals(List.of(1), preparation.runtimeInputNodeIds());
    }

    @Test
    void rejectsEmptyLoweredUnit() {
        assertThrows(IllegalArgumentException.class, () -> FusedOperationBuilder.build(
                new LoweredExecutionUnit("unit", LoweringFamily.FUSED_NATIVE, List.of(), List.of()),
                ignored -> null,
                graph.compile.descriptor.CompiledTensorDescriptorIndex.empty()
        ));
    }
}
