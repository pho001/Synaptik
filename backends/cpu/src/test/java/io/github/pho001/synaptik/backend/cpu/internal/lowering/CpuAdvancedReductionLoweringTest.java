package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAdvancedReductionIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.StatisticalReductionAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import org.junit.jupiter.api.Test;

public class CpuAdvancedReductionLoweringTest {
    @Test void derivesOrderedMembershipCountsAndExactStateOnlyForRequiredKinds() {
        var log = lower(AggregateReductionKind.LOG_SUM_EXP, DataType.FLOAT64, Shape.of(2, 3, 4),
                new MultiAxisReductionAttrs(java.util.List.of(2, 0), false), Shape.of(3));
        var variance = lower(AggregateReductionKind.VARIANCE, DataType.FLOAT32, Shape.of(2, 3, 4),
                new StatisticalReductionAttrs(java.util.List.of(2, 0), true, 1),
                Shape.of(1, 3, 1));
        assertAll(() -> assertArrayEquals(new int[] {2, 0},
                        ((CpuAdvancedReductionIr) log.portableKernelIr()).orderedAxes()),
                () -> assertArrayEquals(new boolean[] {true, false, true},
                        log.advancedReductionGeometry().orElseThrow().selectedAxes()),
                () -> assertEquals(8, log.advancedReductionGeometry().orElseThrow().domainCount()),
                () -> assertEquals(0, log.advancedReductionGeometry().orElseThrow().scratchSliceBytes()),
                () -> assertTrue(variance.advancedReductionGeometry().orElseThrow()
                        .scratchSliceBytes() > 0));
    }

    @Test void rejectsInvalidStaticStatisticalDenominator() {
        assertThrows(IllegalArgumentException.class, () -> lower(AggregateReductionKind.VARIANCE,
                DataType.FLOAT64, Shape.of(2),
                new StatisticalReductionAttrs(java.util.List.of(0), false, 2), Shape.scalar()));
    }

    public static CpuPartitionLowering.LoweredPartition lower(AggregateReductionKind kind,
            DataType type, Shape input, OperationAttrs attrs, Shape output) {
        return new CpuPartitionLowering().lower(CpuAggregateLoweringTest.context(
                kind, type, input, attrs, output));
    }
}
