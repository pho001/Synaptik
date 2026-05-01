package backend.cpu.fused.plan;

import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweringFamily;
import graph.CompiledNode;
import org.junit.jupiter.api.Test;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class LoweredFusedOperationBuilderTest {
    @Test
    void buildsFusedPreparationFromLoweredExecutionUnit() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor out = add.relu();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(out.topologicalSort());

        FusedOperationPreparation preparation = LoweredFusedOperationBuilder.build(
                new LoweredExecutionUnit("unit", LoweringFamily.FUSED_NATIVE, List.of(2, 3), List.of(0, 1)),
                compiledNodes::get
        );

        assertEquals(Operation.OpType.FUSED, preparation.operation().opType());
        assertEquals(2, preparation.runtimeInputs().size());
        assertSame(a, preparation.runtimeInputs().get(0));
        assertSame(b, preparation.runtimeInputs().get(1));
    }

    @Test
    void cpuFusedExecutionStillUsesOperationOpTypeFused() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "builderA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "builderB", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor out = add.relu();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(out.topologicalSort());

        FusedOperationPreparation preparation = LoweredFusedOperationBuilder.build(
                new LoweredExecutionUnit("unit", LoweringFamily.FUSED_NATIVE, List.of(2, 3), List.of(0, 1)),
                compiledNodes::get
        );

        assertEquals(Operation.OpType.FUSED, preparation.operation().opType());
    }

    @Test
    void resolvesViewExternalInputsToBackingTensor() {
        Tensor base = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "base", DataType.FLOAT32);
        Tensor out = base.select(0, 1).relu().exp();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(out.topologicalSort());

        FusedOperationPreparation preparation = LoweredFusedOperationBuilder.build(
                new LoweredExecutionUnit("unit", LoweringFamily.FUSED_NATIVE, List.of(2, 3), List.of(0)),
                compiledNodes::get
        );

        assertEquals(1, preparation.runtimeInputs().size());
        assertSame(base, preparation.runtimeInputs().getFirst());
    }

    @Test
    void rejectsEmptyLoweredUnit() {
        assertThrows(IllegalArgumentException.class, () -> LoweredFusedOperationBuilder.build(
                new LoweredExecutionUnit("unit", LoweringFamily.FUSED_NATIVE, List.of(), List.of()),
                ignored -> null
        ));
    }
}
