package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMaskedReductionIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrs;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

public class CpuMaskedReductionLoweringTest {
    @Test void lowersDirectionalBroadcastToThreeBoundariesAndExactState() {
        var lowered = new CpuPartitionLowering().lower(context(AggregateReductionKind.MEAN,
                DataType.FLOAT32, Shape.of(2, 3, 4), Shape.of(3, 1), 1));
        var ir = (CpuMaskedReductionIr) lowered.portableKernelIr();
        var geometry = lowered.maskedReductionGeometry().orElseThrow();
        assertAll(() -> assertEquals(CpuMaskedReductionIr.Kind.MEAN, ir.kind()),
                () -> assertEquals(List.of(DataType.FLOAT32, DataType.BOOL, DataType.FLOAT32),
                        lowered.boundaryDataTypes()),
                () -> assertEquals(8, lowered.elementCount()),
                () -> assertEquals(3, geometry.maximumDomainCount()),
                () -> assertEquals(3, lowered.accessBindings().size()),
                () -> assertTrue(lowered.aggregateGeometry().isEmpty()),
                () -> assertTrue(lowered.argExtremaGeometry().isEmpty()));
    }

    @Test void admitsExactDirectionalBroadcastFormsAndRejectsEnlargingBroadcast() {
        for (Shape mask : List.of(Shape.scalar(), Shape.of(4), Shape.of(1, 4),
                Shape.of(2, 1), Shape.of(2, 4))) {
            assertDoesNotThrow(() -> new CpuPartitionLowering().lower(context(
                    AggregateReductionKind.SUM, DataType.FLOAT64,
                    Shape.of(2, 4), mask, 1)));
        }
        assertThrows(IllegalArgumentException.class, () -> new CpuPartitionLowering().lower(
                context(AggregateReductionKind.SUM, DataType.FLOAT64,
                        Shape.of(2, 4, 3), Shape.of(2, 4), 1)));
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(
            AggregateReductionKind kind, DataType type, Shape dataShape, Shape maskShape,
            int axis) {
        long[] source = dataShape.toLongArray();
        long[] result = new long[source.length - 1];
        for (int input = 0, output = 0; input < source.length; input++)
            if (input != axis) result[output++] = source[input];
        return CpuScatterLoweringTest.context(new Operation(kind, new MaskedReductionAttrs(axis)),
                List.of(0, 1), List.of(CpuScatterLoweringTest.desc(type, dataShape),
                        CpuScatterLoweringTest.desc(DataType.BOOL, maskShape)),
                CpuScatterLoweringTest.desc(type, Shape.of(result)));
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(
            AggregateReductionKind kind, DataType type, Shape dataShape, Shape maskShape,
            int axis, LayoutDescriptor dataLayout, LayoutDescriptor maskLayout,
            LayoutDescriptor outputLayout) {
        long[] source = dataShape.toLongArray();
        long[] result = new long[source.length - 1];
        for (int input = 0, output = 0; input < source.length; input++)
            if (input != axis) result[output++] = source[input];
        return CpuScatterLoweringTest.context(new Operation(kind, new MaskedReductionAttrs(axis)),
                List.of(0, 1), List.of(
                        new TensorDescriptor(type, dataShape, Optional.of(dataLayout), false),
                        new TensorDescriptor(DataType.BOOL, maskShape, Optional.of(maskLayout),
                                false)),
                new TensorDescriptor(type, Shape.of(result), Optional.of(outputLayout), false));
    }
}
