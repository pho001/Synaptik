package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.datatype.DataType;
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
                () -> assertEquals(List.of(CpuPointwiseOpcode.ADD,
                                CpuPointwiseOpcode.GELU_EXACT,
                                CpuPointwiseOpcode.MUL),
                        small.instructions().stream().map(CpuKernelIr.Instruction::opcode).toList()),
                () -> assertEquals(new CpuKernelIr.Loop("start", "end"), small.loop()));
    }

    @Test void enforcesScalarPowerImmediateRealizationAndFloatingTypeTogether() {
        var immediate = new CpuKernelIr.ScalarImmediate(DataType.FLOAT32,
                Float.floatToRawIntBits(2.0f) & 0xffff_ffffL);
        assertAll(
                () -> assertDoesNotThrow(() -> powerIr(DataType.FLOAT32, immediate,
                        CpuKernelIr.PowerRealization.SQUARE)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuKernelIr.Instruction(CpuPointwiseOpcode.SCALAR_POW,
                                List.of(0), 1, immediate)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuKernelIr.Instruction(CpuPointwiseOpcode.SCALAR_DIV,
                                List.of(0), 1, immediate,
                                CpuKernelIr.PowerRealization.DIRECT)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> powerIr(DataType.FLOAT32, immediate,
                                CpuKernelIr.PowerRealization.DIRECT)),
                () -> assertThrows(IllegalArgumentException.class, () -> powerIr(DataType.INT32,
                        new CpuKernelIr.ScalarImmediate(DataType.INT32, 2),
                        CpuKernelIr.PowerRealization.SQUARE)));
    }

    @Test void realizationAndExactExponentBitsChangeStructuralIdentity() {
        var positive = new CpuKernelIr.ScalarImmediate(DataType.FLOAT64,
                Double.doubleToRawLongBits(+0.0d));
        var negative = new CpuKernelIr.ScalarImmediate(DataType.FLOAT64,
                Double.doubleToRawLongBits(-0.0d));
        var first = powerIr(DataType.FLOAT64, positive,
                CpuKernelIr.PowerRealization.POSITIVE_ONE);
        var bitsChanged = powerIr(DataType.FLOAT64, negative,
                CpuKernelIr.PowerRealization.POSITIVE_ONE);
        var identity = powerIr(DataType.FLOAT64,
                new CpuKernelIr.ScalarImmediate(DataType.FLOAT64,
                        Double.doubleToRawLongBits(1.0d)),
                CpuKernelIr.PowerRealization.IDENTITY);
        assertAll(
                () -> assertNotEquals(first.structuralKey(), bitsChanged.structuralKey()),
                () -> assertNotEquals(first.structuralKey(), identity.structuralKey()));
    }

    private static CpuKernelIr powerIr(DataType type, CpuKernelIr.ScalarImmediate immediate,
            CpuKernelIr.PowerRealization realization) {
        var read = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        var write = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,
                CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        return new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, type, CpuKernelIr.Value.Kind.INPUT, read),
                new CpuKernelIr.Value(1, type, CpuKernelIr.Value.Kind.OUTPUT, write)),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.SCALAR_POW,
                        List.of(0), 1, immediate, realization)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(1, 0)));
    }
}
