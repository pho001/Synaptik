package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering.LoweredPartition;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import org.junit.jupiter.api.Test;

public class CpuSoftmaxLoweringTest {
    @Test void derivesCompleteSliceRangesAndPreservesEmptyNonSelectedExtent() {
        var dense = lower(SoftmaxKind.SOFTMAX, DataType.FLOAT64, Shape.of(2, 3, 4), 1);
        var empty = lower(SoftmaxKind.LOG_SOFTMAX, DataType.BFLOAT16, Shape.of(0, 3), 1);
        assertAll(() -> assertEquals(8, dense.elementCount()),
                () -> assertEquals(3, dense.softmaxGeometry().orElseThrow().sliceWidth()),
                () -> assertEquals(0, empty.elementCount()),
                () -> assertEquals(0, empty.softmaxGeometry().orElseThrow().elementCount()),
                () -> assertTrue(dense.aggregateGeometry().isEmpty()));
    }

    @Test void rejectsSelectedZeroExtent() {
        assertThrows(IllegalArgumentException.class, () -> lower(SoftmaxKind.SOFTMAX,
                DataType.FLOAT32, Shape.of(2, 0), 1));
    }

    public static LoweredPartition lower(SoftmaxKind kind, DataType type, Shape shape, int axis) {
        return new CpuPartitionLowering().lower(context(kind, type, shape, axis));
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(SoftmaxKind kind,
            DataType type, Shape shape, int axis) {
        return CpuScatterLoweringTest.context(new Operation(kind, new SoftmaxAttrs(axis)),
                List.of(0), List.of(CpuScatterLoweringTest.desc(type, shape)),
                CpuScatterLoweringTest.desc(type, shape));
    }
}
