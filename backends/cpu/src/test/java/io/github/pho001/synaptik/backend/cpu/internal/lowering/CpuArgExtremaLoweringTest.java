package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuArgExtremaIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaAttrs;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import org.junit.jupiter.api.Test;

public class CpuArgExtremaLoweringTest {
    @Test void lowersCompleteOutputCellsWithMixedBoundaryTypesAndNoResources() {
        var lowered = new CpuPartitionLowering().lower(context(AggregateReductionKind.ARG_MAX,
                DataType.BFLOAT16, Shape.of(2, 3, 4), 1, true,
                ArgExtremaTiePolicy.LAST_INDEX));
        var ir = (CpuArgExtremaIr) lowered.portableKernelIr();
        assertAll(() -> assertEquals(CpuArgExtremaIr.Kind.ARG_MAX, ir.kind()),
                () -> assertEquals(DataType.BFLOAT16, ir.inputType()),
                () -> assertEquals(DataType.INT64, lowered.boundaryDataTypes().getLast()),
                () -> assertEquals(8, lowered.elementCount()),
                () -> assertArrayEquals(new long[] {8}, lowered.extents()),
                () -> assertTrue(lowered.argExtremaGeometry().isPresent()),
                () -> assertEquals(3, lowered.argExtremaGeometry().orElseThrow().axisExtent()),
                () -> assertTrue(ir.narrowLogicalIndex()),
                () -> assertTrue(ir.narrowOutputIndex()),
                () -> assertTrue(lowered.aggregateGeometry().isEmpty()));
    }

    @Test void admitsFiveTypesKindsPoliciesAndFormsAndRejectsEmptySelectedAxis() {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16,
                DataType.INT32, DataType.INT64)) {
            for (AggregateReductionKind kind : List.of(AggregateReductionKind.ARG_MIN,
                    AggregateReductionKind.ARG_MAX)) {
                for (ArgExtremaTiePolicy tie : ArgExtremaTiePolicy.values()) {
                    for (boolean keep : List.of(false, true)) assertDoesNotThrow(() ->
                            new CpuPartitionLowering().lower(context(kind, type,
                                    Shape.of(2, 3), 1, keep, tie)));
                }
            }
        }
        assertThrows(IllegalArgumentException.class, () -> new CpuPartitionLowering().lower(
                context(AggregateReductionKind.ARG_MIN, DataType.FLOAT64, Shape.of(2, 0), 1,
                        false, ArgExtremaTiePolicy.FIRST_INDEX)));
        assertDoesNotThrow(() -> new CpuPartitionLowering().lower(context(
                AggregateReductionKind.ARG_MIN, DataType.FLOAT64, Shape.of(0, 2), 1, false,
                ArgExtremaTiePolicy.FIRST_INDEX)));
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(AggregateReductionKind kind,
            DataType type, Shape inputShape, int axis, boolean keep,
            ArgExtremaTiePolicy tie) {
        long[] input = inputShape.toLongArray();
        long[] output = new long[keep ? input.length : input.length - 1];
        for (int source = 0, target = 0; source < input.length; source++) {
            if (source == axis) {
                if (keep) output[target++] = 1;
            } else output[target++] = input[source];
        }
        return CpuScatterLoweringTest.context(new Operation(kind,
                        new ArgExtremaAttrs(axis, keep, tie)), List.of(0),
                List.of(CpuScatterLoweringTest.desc(type, inputShape)),
                CpuScatterLoweringTest.desc(DataType.INT64, Shape.of(output)));
    }
}
