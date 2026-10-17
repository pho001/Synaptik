package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuIndexingIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import org.junit.jupiter.api.Test;

public class CpuIndexingLoweringTest {
    @Test void lowersEveryFamilyAndRetainsCompactGeometry() {
        var gather = lower(context(new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(1)),
                List.of(0, 1), List.of(descriptor(DataType.FLOAT32, Shape.of(2, 3)),
                        descriptor(DataType.INT64, Shape.of(2))),
                descriptor(DataType.FLOAT32, Shape.of(2, 2))));
        var elements = lower(context(new Operation(AxisGatherKind.GATHER_ELEMENTS,
                new IndexAxisAttrs(1)), List.of(0, 1),
                List.of(descriptor(DataType.INT64, Shape.of(2, 3)),
                        descriptor(DataType.INT32, Shape.of(2, 2))),
                descriptor(DataType.INT64, Shape.of(2, 2))));
        var nd = lower(context(new Operation(GatherNdKind.GATHER_ND, new GatherNdAttrs(0)),
                List.of(0, 1), List.of(descriptor(DataType.BOOL, Shape.of(2, 3)),
                        descriptor(DataType.INT32, Shape.of(2, 1))),
                descriptor(DataType.BOOL, Shape.of(2, 3))));
        var hot = lower(context(new Operation(OneHotKind.ONE_HOT, new OneHotAttrs(3)),
                List.of(0), List.of(descriptor(DataType.INT64, Shape.of(2))),
                descriptor(DataType.BOOL, Shape.of(2, 3))));
        assertAll(() -> assertEquals(CpuIndexingIr.Family.GATHER,
                        ((CpuIndexingIr) gather.portableKernelIr()).family()),
                () -> assertEquals(CpuIndexingIr.Family.GATHER_ELEMENTS,
                        ((CpuIndexingIr) elements.portableKernelIr()).family()),
                () -> assertInstanceOf(CpuIndexingLowering.Geometry.Nd.class,
                        nd.indexingGeometry().orElseThrow().variant()),
                () -> assertInstanceOf(CpuIndexingLowering.Geometry.Hot.class,
                        hot.indexingGeometry().orElseThrow().variant()),
                () -> assertTrue(gather.movementGeometry().isEmpty()));
    }

    @Test void lowersScalarBatchTupleAndDeduplicatedInputFormsWithoutPreparedTables() {
        var scalar = lower(context(new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(0)),
                List.of(0, 1), List.of(descriptor(DataType.INT64, Shape.of(3)),
                        descriptor(DataType.INT32, Shape.scalar())),
                descriptor(DataType.INT64, Shape.scalar())));
        var batched = lower(context(new Operation(GatherNdKind.GATHER_ND,
                        new GatherNdAttrs(1)), List.of(0, 1),
                List.of(descriptor(DataType.FLOAT32, Shape.of(2, 3, 4)),
                        descriptor(DataType.INT64, Shape.of(2, 5, 2))),
                descriptor(DataType.FLOAT32, Shape.of(2, 5))));
        var deduplicated = lower(context(new Operation(AxisGatherKind.GATHER,
                        new IndexAxisAttrs(0)), List.of(0, 0),
                List.of(descriptor(DataType.INT32, Shape.of(2))),
                descriptor(DataType.INT32, Shape.of(2))));
        var nd = assertInstanceOf(CpuIndexingLowering.Geometry.Nd.class,
                batched.indexingGeometry().orElseThrow().variant());
        assertAll(
                () -> assertEquals(1, scalar.elementCount()),
                () -> assertEquals(1, nd.batchDimensions()),
                () -> assertEquals(2, nd.tupleDepth()),
                () -> assertEquals(List.of(0, 0), ((CpuIndexingIr) deduplicated
                        .portableKernelIr()).occurrenceToBoundary()),
                () -> assertEquals(2, deduplicated.boundaryValues().size()),
                () -> assertEquals(3, batched.indexingGeometry().orElseThrow()
                        .boundaries().size()));
    }

    @Test void loweringFailsClosedForWrongFormulaTypeTupleAndOutputInjectivity() {
        var data = descriptor(DataType.FLOAT32, Shape.of(2, 3));
        var indices = descriptor(DataType.INT32, Shape.of(2));
        var nonInjective = descriptor(DataType.FLOAT32, Shape.of(2, 2),
                LayoutDescriptor.of(Shape.of(2, 2), new long[]{0, 1}, 0, true));
        assertAll(
                () -> assertThrows(IllegalArgumentException.class, () -> lower(context(
                        new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(1)),
                        List.of(0, 1), List.of(data, indices),
                        descriptor(DataType.FLOAT32, Shape.of(2, 3))))),
                () -> assertThrows(IllegalArgumentException.class, () -> lower(context(
                        new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(1)),
                        List.of(0, 1), List.of(data,
                                descriptor(DataType.FLOAT32, Shape.of(2))),
                        descriptor(DataType.FLOAT32, Shape.of(2, 2))))),
                () -> assertThrows(IllegalArgumentException.class, () -> lower(context(
                        new Operation(GatherNdKind.GATHER_ND, new GatherNdAttrs(0)),
                        List.of(0, 1), List.of(data,
                                descriptor(DataType.INT32, Shape.of(2, 3))),
                        descriptor(DataType.FLOAT32, Shape.of(2))))),
                () -> assertThrows(IllegalArgumentException.class, () -> lower(context(
                        new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(1)),
                        List.of(0, 1), List.of(data, indices), nonInjective))));
    }

    private static CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        return new CpuPartitionLowering().lower(context);
    }
    public static PrepareContext<CpuPartitionAnalysisInputs> context(Operation operation,
            List<Integer> occurrences,
            List<io.github.pho001.synaptik.model.tensor.TensorDescriptor> inputs,
            io.github.pho001.synaptik.model.tensor.TensorDescriptor output) {
        return CpuNonAffineMovementLoweringTest.context(operation, occurrences, inputs, output);
    }
    public static io.github.pho001.synaptik.model.tensor.TensorDescriptor descriptor(
            DataType type, Shape shape) {
        return CpuNonAffineMovementLoweringTest.descriptor(type, shape);
    }
    public static io.github.pho001.synaptik.model.tensor.TensorDescriptor descriptor(
            DataType type, Shape shape, LayoutDescriptor layout) {
        return CpuNonAffineMovementLoweringTest.descriptor(type, shape, layout);
    }
}
