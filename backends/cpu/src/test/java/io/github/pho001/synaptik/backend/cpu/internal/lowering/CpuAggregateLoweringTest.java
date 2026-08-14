package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import org.junit.jupiter.api.Test;

public class CpuAggregateLoweringTest {
    @Test void derivesFullSingleAndCanonicalMultiAxisOutputDomains() {
        var full = lower(AggregateReductionKind.MIN, DataType.FLOAT64, Shape.of(2,3,4),
                NoOperationAttrs.INSTANCE, Shape.scalar());
        var single = lower(AggregateReductionKind.MAX, DataType.INT64, Shape.of(2,3,4),
                new AxisReductionAttrs(1, true), Shape.of(2,1,4));
        var multi = lower(AggregateReductionKind.MIN, DataType.INT32, Shape.of(2,3,4),
                new MultiAxisReductionAttrs(List.of(2,0), false), Shape.of(3));
        assertAll(() -> assertEquals(1, full.elementCount()),
                () -> assertEquals(24, full.aggregateGeometry().orElseThrow().domainCount()),
                () -> assertEquals(8, single.elementCount()),
                () -> assertEquals(3, single.aggregateGeometry().orElseThrow().domainCount()),
                () -> assertArrayEquals(new int[] {0,2},
                        ((CpuAggregateIr) multi.portableKernelIr()).selectedAxes()),
                () -> assertEquals(3, multi.elementCount()),
                () -> assertEquals(8, multi.aggregateGeometry().orElseThrow().domainCount()),
                () -> assertEquals(2, multi.boundaryValues().size()),
                () -> assertTrue(multi.virtualValues().isEmpty()));
    }

    @Test void handlesEmptyAxesSelectedZeroUnselectedZeroAndScalar() {
        var point = lower(AggregateReductionKind.ANY, DataType.BOOL, Shape.of(2,3),
                new MultiAxisReductionAttrs(List.of(), false), Shape.of(2,3));
        var selectedEmpty = lower(AggregateReductionKind.ALL, DataType.BOOL, Shape.of(2,0,3),
                new AxisReductionAttrs(1, false), Shape.of(2,3));
        var unselectedEmpty = lower(AggregateReductionKind.MIN, DataType.FLOAT32, Shape.of(0,3),
                new AxisReductionAttrs(1, false), Shape.of(0));
        var scalar = lower(AggregateReductionKind.MAX, DataType.INT32, Shape.scalar(),
                NoOperationAttrs.INSTANCE, Shape.scalar());
        assertAll(() -> assertEquals(6, point.elementCount()),
                () -> assertEquals(1, point.aggregateGeometry().orElseThrow().domainCount()),
                () -> assertEquals(6, selectedEmpty.elementCount()),
                () -> assertEquals(0, selectedEmpty.aggregateGeometry().orElseThrow().domainCount()),
                () -> assertEquals(0, unselectedEmpty.elementCount()),
                () -> assertEquals(1, scalar.aggregateGeometry().orElseThrow().domainCount()));
    }

    @Test void capabilityIsExactAndAcceptsBoundedInjectiveNonDenseOutput() {
        Shape inputShape = Shape.of(2,3), outputShape = Shape.of(2,1);
        var input = CpuScatterLoweringTest.desc(DataType.FLOAT64, inputShape);
        var output = CpuIndexingLoweringTest.descriptor(DataType.FLOAT64, outputShape,
                LayoutDescriptor.of(outputShape, new long[] {2,1}, 1, true));
        var operation = new Operation(AggregateReductionKind.MIN, new AxisReductionAttrs(1, true));
        var query = new OperationCapabilityQuery(operation, List.of(input), List.of(output));
        assertAll(() -> assertTrue(new CpuCapabilityProvider().supports(query)),
                () -> assertDoesNotThrow(() -> new CpuPartitionLowering().lower(
                        CpuScatterLoweringTest.context(operation, List.of(0), List.of(input), output))),
                () -> assertFalse(new CpuCapabilityProvider().supports(new OperationCapabilityQuery(
                        new Operation(AggregateReductionKind.SUM, NoOperationAttrs.INSTANCE),
                        List.of(input), List.of(CpuScatterLoweringTest.desc(DataType.FLOAT64,
                            Shape.scalar()))))));
    }

    public static CpuPartitionLowering.LoweredPartition lower(AggregateReductionKind kind,
            DataType type, Shape input, OperationAttrs attrs, Shape output) {
        return new CpuPartitionLowering().lower(context(kind, type, input, attrs, output));
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(AggregateReductionKind kind,
            DataType type, Shape input, OperationAttrs attrs, Shape output) {
        return CpuScatterLoweringTest.context(new Operation(kind, attrs), List.of(0),
                List.of(CpuScatterLoweringTest.desc(type, input)),
                CpuScatterLoweringTest.desc(type, output));
    }
}
