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
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import org.junit.jupiter.api.Test;

public class CpuAggregateLoweringTest {
    /** Exact named Dimensions, not sampled batch extents, select SUM/MEAN axes. */
    @Test void commonAxisFactsPreserveNamedBatchAndActualSingletonFiltering() {
        var b = new io.github.pho001.synaptik.model.shape.DynamicDimension("B");
        var f = new io.github.pho001.synaptik.model.shape.StaticDimension(3);
        Shape input = Shape.ofDimensions(b,f), output = Shape.of(3);
        assertEquals(List.of(0),CpuAggregateLowering.normalizedAxes(input,output,new SumToShapeAttrs(output)));
        assertEquals(List.of(1),CpuAggregateLowering.normalizedAxes(input,Shape.ofDimensions(b),new AxisReductionAttrs(1,false)));
        assertEquals(List.of(0,1),CpuAggregateLowering.normalizedAxes(input,Shape.of(1,1),new MultiAxisReductionAttrs(List.of(1,0),true)));
        assertThrows(IllegalArgumentException.class,() -> CpuAggregateLowering.normalizedAxes(input,
                Shape.ofDimensions(new io.github.pho001.synaptik.model.shape.DynamicDimension("Other"),f),
                new AxisReductionAttrs(1,false)));
        var in = CpuPartitionLowering.batchCanonical(new io.github.pho001.synaptik.model.tensor.TensorDescriptor(
                DataType.FLOAT64,input,java.util.Optional.empty(),false),io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.AccessKind.READ);
        Shape target = Shape.of(1,3);
        var out = CpuPartitionLowering.batchCanonical(new io.github.pho001.synaptik.model.tensor.TensorDescriptor(
                DataType.FLOAT64,target,java.util.Optional.of(LayoutDescriptor.contiguous(target)),false),
                io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.AccessKind.WRITE);
        var geometry = new CpuAggregateLowering.BatchGeometry(CpuAggregateIr.Kind.SUM,DataType.FLOAT64,
                CpuAggregateLowering.reductionForm(new SumToShapeAttrs(target)),List.of(0),false,in,out);
        assertArrayEquals(new int[0],geometry.bind(1).selectedAxes());
        assertArrayEquals(new int[]{0},geometry.bind(7).selectedAxes());
        assertEquals(34,geometry.bind(0).stateLimbCount());
        assertEquals(280,geometry.byteSize(1));
    }
    @Test void derivesRightAlignedSumToShapeGeometryAndCopyResources() {
        var combined = lower(AggregateReductionKind.SUM, DataType.FLOAT64, Shape.of(2,3,4),
                new SumToShapeAttrs(Shape.of(3,1)), Shape.of(3,1));
        var copy = lower(AggregateReductionKind.SUM, DataType.FLOAT32, Shape.of(2,3),
                new SumToShapeAttrs(Shape.of(2,3)), Shape.of(2,3));
        var empty = lower(AggregateReductionKind.SUM, DataType.INT64, Shape.of(0,3,4),
                new SumToShapeAttrs(Shape.of(3,1)), Shape.of(3,1));
        assertAll(
                () -> assertEquals(CpuAggregateIr.Form.SUM_TO_SHAPE,
                        combined.aggregateGeometry().orElseThrow().form()),
                () -> assertArrayEquals(new int[]{0,2},
                        combined.aggregateGeometry().orElseThrow().selectedAxes()),
                () -> assertEquals(8, combined.aggregateGeometry().orElseThrow().domainCount()),
                () -> assertEquals(3, combined.elementCount()),
                () -> assertArrayEquals(new int[0],
                        copy.aggregateGeometry().orElseThrow().selectedAxes()),
                () -> assertEquals(0, copy.aggregateGeometry().orElseThrow().scratchSliceBytes()),
                () -> assertEquals(0, empty.aggregateGeometry().orElseThrow().domainCount()));
    }

    @Test void sumToShapeCoversLeadingAlignedScalarEqualityAndZeroGeometry() {
        var leading = lower(AggregateReductionKind.SUM, DataType.INT32, Shape.of(2,3),
                new SumToShapeAttrs(Shape.of(3)), Shape.of(3));
        var aligned = lower(AggregateReductionKind.SUM, DataType.INT64, Shape.of(2,3),
                new SumToShapeAttrs(Shape.of(2,1)), Shape.of(2,1));
        var scalar = lower(AggregateReductionKind.SUM, DataType.FLOAT64, Shape.scalar(),
                new SumToShapeAttrs(Shape.scalar()), Shape.scalar());
        var preservedOne = lower(AggregateReductionKind.SUM, DataType.FLOAT32, Shape.of(1,3),
                new SumToShapeAttrs(Shape.of(1,3)), Shape.of(1,3));
        var selectedZero = lower(AggregateReductionKind.SUM, DataType.FLOAT32, Shape.of(2,0,4),
                new SumToShapeAttrs(Shape.of(1,4)), Shape.of(1,4));
        var unselectedZero = lower(AggregateReductionKind.SUM, DataType.FLOAT32, Shape.of(2,0,4),
                new SumToShapeAttrs(Shape.of(0,4)), Shape.of(0,4));
        assertAll(
                () -> assertArrayEquals(new int[]{0}, leading.aggregateGeometry().orElseThrow()
                        .selectedAxes()),
                () -> assertArrayEquals(new int[]{1}, aligned.aggregateGeometry().orElseThrow()
                        .selectedAxes()),
                () -> assertArrayEquals(new int[0], scalar.aggregateGeometry().orElseThrow()
                        .selectedAxes()),
                () -> assertArrayEquals(new int[0], preservedOne.aggregateGeometry().orElseThrow()
                        .selectedAxes()),
                () -> assertEquals(0, selectedZero.aggregateGeometry().orElseThrow().domainCount()),
                () -> assertTrue(selectedZero.aggregateGeometry().orElseThrow()
                        .scratchSliceBytes() > 0),
                () -> assertEquals(0, unselectedZero.elementCount()),
                () -> assertThrows(IllegalArgumentException.class, () -> lower(
                        AggregateReductionKind.SUM, DataType.FLOAT32, Shape.of(2,3),
                        new SumToShapeAttrs(Shape.of(2)), Shape.of(2))));
    }

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

    @Test void derivesNumericalStateAndRetainsExactLowExtrema() {
        var sum = lower(AggregateReductionKind.SUM, DataType.FLOAT64, Shape.of(3),
                NoOperationAttrs.INSTANCE, Shape.scalar());
        var product = lower(AggregateReductionKind.PROD, DataType.FLOAT32, Shape.of(3),
                NoOperationAttrs.INSTANCE, Shape.scalar());
        var integral = lower(AggregateReductionKind.PROD, DataType.INT64, Shape.of(3),
                NoOperationAttrs.INSTANCE, Shape.scalar());
        var sumGeometry = sum.aggregateGeometry().orElseThrow();
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            for (var kind : List.of(AggregateReductionKind.MIN, AggregateReductionKind.MAX)) {
                var exact = lower(kind, type, Shape.of(65),
                        NoOperationAttrs.INSTANCE, Shape.scalar());
                assertEquals(65, exact.aggregateGeometry().orElseThrow().domainCount());
            }
        }
        var productGeometry = product.aggregateGeometry().orElseThrow();
        assertAll(
                () -> assertEquals(33, sumGeometry.stateLimbCount()),
                () -> assertEquals(272, sumGeometry.scratchSliceBytes()),
                () -> assertEquals(2, productGeometry.stateLimbCount()),
                () -> assertEquals(40, productGeometry.scratchSliceBytes()),
                () -> assertEquals(0, integral.aggregateGeometry().orElseThrow().stateLimbCount()));
    }

    @Test void capabilityIsExactAndAcceptsBoundedInjectiveNonDenseOutput() {
        Shape inputShape = Shape.of(2,3), outputShape = Shape.of(2,1);
        var input = CpuScatterLoweringTest.desc(DataType.FLOAT64, inputShape);
        var output = CpuIndexingLoweringTest.descriptor(DataType.FLOAT64, outputShape,
                LayoutDescriptor.of(outputShape, new long[] {2,1}, 1, true));
        var operation = new Operation(AggregateReductionKind.MIN, new AxisReductionAttrs(1, true));
        var query = new OperationCapabilityQuery(operation, List.of(input), List.of(output));
        assertAll(() -> io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest.assertCurrentSupport(new CpuCapabilityProvider(), query),
                () -> assertDoesNotThrow(() -> new CpuPartitionLowering().lower(
                        CpuScatterLoweringTest.context(operation, List.of(0), List.of(input), output))),
                () -> io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest.assertCurrentSupport(new CpuCapabilityProvider(), new OperationCapabilityQuery(new Operation(AggregateReductionKind.SUM, NoOperationAttrs.INSTANCE), List.of(input), List.of(CpuScatterLoweringTest.desc(DataType.FLOAT64,
                    Shape.scalar())))));
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
