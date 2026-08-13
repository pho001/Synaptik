package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import org.junit.jupiter.api.Test;

public class CpuScanLoweringTest {
    @Test void lowersSliceDomainWithTwoBoundariesAndZeroWorkspaceGeometry() {
        var lowered = new CpuPartitionLowering().lower(context(CumulativeScanKind.CUM_PROD,
                DataType.BFLOAT16, Shape.of(2, 3, 4), 1, true, true));
        var ir = (CpuScanIr) lowered.portableKernelIr();
        assertAll(() -> assertEquals(CpuScanIr.Kind.CUM_PROD, ir.kind()),
                () -> assertEquals(8, lowered.elementCount()),
                () -> assertArrayEquals(new long[]{8}, lowered.extents()),
                () -> assertEquals(2, lowered.boundaryValues().size()),
                () -> assertTrue(lowered.virtualValues().isEmpty()),
                () -> assertTrue(lowered.scanGeometry().isPresent()),
                () -> assertTrue(lowered.randomGeometry().isEmpty()));
    }

    @Test void admitsFiveTypesBothKindsAndAllModesButRejectsBoolAndScalar() {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16,
                DataType.INT32, DataType.INT64)) for (CumulativeScanKind kind : CumulativeScanKind.values())
            for (boolean exclusive : List.of(false, true)) for (boolean reverse : List.of(false, true))
                assertDoesNotThrow(() -> new CpuPartitionLowering().lower(
                        context(kind, type, Shape.of(2, 3), 1, exclusive, reverse)));
        assertThrows(IllegalArgumentException.class, () -> new CpuPartitionLowering().lower(
                context(CumulativeScanKind.CUM_SUM, DataType.BOOL, Shape.of(2), 0, false, false)));
        assertThrows(IllegalArgumentException.class, () -> new CpuPartitionLowering().lower(
                context(CumulativeScanKind.CUM_SUM, DataType.FLOAT64, Shape.scalar(), 0, false, false)));
    }

    @Test void capabilityAndLoweringAgreeOnInterleavedAndCollidingOutputs() {
        Shape shape = Shape.of(3, 2);
        var input = CpuIndexingLoweringTest.descriptor(DataType.FLOAT64, shape);
        var interleaved = CpuIndexingLoweringTest.descriptor(DataType.FLOAT64, shape,
                LayoutDescriptor.of(shape, new long[] {2, 3}, 0, true));
        var colliding = CpuIndexingLoweringTest.descriptor(DataType.FLOAT64, shape,
                LayoutDescriptor.of(shape, new long[] {1, 1}, 0, true));
        var operation = new Operation(CumulativeScanKind.CUM_SUM,
                new CumulativeScanAttrs(1, false, false));
        var provider = new CpuCapabilityProvider();
        var accepted = new OperationCapabilityQuery(operation, List.of(input), List.of(interleaved));
        var rejected = new OperationCapabilityQuery(operation, List.of(input), List.of(colliding));

        assertAll(
                () -> assertTrue(provider.supports(accepted)),
                () -> assertDoesNotThrow(() -> new CpuPartitionLowering().lower(
                        CpuScatterLoweringTest.context(operation, List.of(0), List.of(input),
                                interleaved))),
                () -> assertFalse(provider.supports(rejected)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuPartitionLowering().lower(CpuScatterLoweringTest.context(
                                operation, List.of(0), List.of(input), colliding))));
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(CumulativeScanKind kind,
            DataType type, Shape shape, int axis, boolean exclusive, boolean reverse) {
        return CpuScatterLoweringTest.context(new Operation(kind,
                new CumulativeScanAttrs(axis, exclusive, reverse)), List.of(0),
                List.of(CpuScatterLoweringTest.desc(type, shape)),
                CpuScatterLoweringTest.desc(type, shape));
    }
}
