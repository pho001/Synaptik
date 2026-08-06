package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuKernelIrTest {
    @Test void recordsOrderedVirtualTopologyWithoutRouteOrExtentIdentity() {
        var small = CpuPartitionPreparerTest.analyze(Shape.of(2)).plan().units().getFirst()
                .portablePlan().kernelIr();
        var large = CpuPartitionPreparerTest.analyze(Shape.of(1024)).plan().units().getFirst()
                .portablePlan().kernelIr();
        assertAll(
                () -> assertEquals(small.structuralKey(), large.structuralKey()),
                () -> assertEquals(List.of(CpuKernelIr.Value.Kind.INPUT,
                                CpuKernelIr.Value.Kind.INPUT, CpuKernelIr.Value.Kind.INPUT,
                                CpuKernelIr.Value.Kind.VIRTUAL, CpuKernelIr.Value.Kind.VIRTUAL,
                                CpuKernelIr.Value.Kind.OUTPUT),
                        small.values().stream().map(CpuKernelIr.Value::kind).toList()),
                () -> assertEquals(List.of(CpuKernelIr.Instruction.Semantic.ADD,
                                CpuKernelIr.Instruction.Semantic.GELU_EXACT,
                                CpuKernelIr.Instruction.Semantic.MUL),
                        small.instructions().stream().map(CpuKernelIr.Instruction::semantic).toList()),
                () -> assertEquals(new CpuKernelIr.Loop("start", "end"), small.loop()));
    }
}
