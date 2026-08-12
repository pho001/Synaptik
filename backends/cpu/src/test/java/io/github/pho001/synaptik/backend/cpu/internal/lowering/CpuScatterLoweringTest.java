package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScatterIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import org.junit.jupiter.api.Test;

public class CpuScatterLoweringTest {
    @Test void lowersAllCurrentFamiliesAndExactGatherCompatibleShape() {
        var elements=lower(context(new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(1,ScatterReduction.NONE)),List.of(0,1,2),
                List.of(desc(DataType.INT64,Shape.of(2,3)),desc(DataType.INT32,Shape.of(2,2)),
                        desc(DataType.INT64,Shape.of(2,2))),desc(DataType.INT64,Shape.of(2,3))));
        var add=lower(context(new Operation(AxisScatterKind.SCATTER_ADD,new IndexAxisAttrs(1)),
                List.of(0,1,2),List.of(desc(DataType.FLOAT32,Shape.of(2,3,4)),
                        desc(DataType.INT64,Shape.of(5,2)),desc(DataType.FLOAT32,Shape.of(2,5,2,4))),
                desc(DataType.FLOAT32,Shape.of(2,3,4))));
        var nd=lower(context(new Operation(ScatterNdKind.SCATTER_ND,
                        new ScatterNdAttrs(1,ScatterReduction.MAX)),List.of(0,1,2),
                List.of(desc(DataType.FLOAT64,Shape.of(2,3,4)),
                        desc(DataType.INT32,Shape.of(2,5,1)),
                        desc(DataType.FLOAT64,Shape.of(2,5,4))),
                desc(DataType.FLOAT64,Shape.of(2,3,4))));
        assertAll(() -> assertEquals(CpuScatterIr.Family.SCATTER_ELEMENTS,
                        ((CpuScatterIr)elements.portableKernelIr()).family()),
                () -> assertEquals(CpuScatterIr.Family.SCATTER_ADD,
                        ((CpuScatterIr)add.portableKernelIr()).family()),
                () -> assertEquals(1,nd.scatterGeometry().orElseThrow().batchDimensions()),
                () -> assertEquals(1,nd.scatterGeometry().orElseThrow().tupleDepth()),
                () -> assertTrue(elements.indexingGeometry().isEmpty()));
    }

    @Test void deduplicatesInputOccurrencesAndDeclaresExactProductSliceOnlyWhenUseful() {
        var dedup=lower(context(new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0,ScatterReduction.ADD)),List.of(0,1,0),
                List.of(desc(DataType.INT32,Shape.of(2)),desc(DataType.INT32,Shape.of(2))),
                desc(DataType.INT32,Shape.of(2))));
        var product=lower(context(new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0,ScatterReduction.MUL)),List.of(0,1,2),
                List.of(desc(DataType.FLOAT64,Shape.of(3)),desc(DataType.INT64,Shape.of(5)),
                        desc(DataType.FLOAT64,Shape.of(5))),desc(DataType.FLOAT64,Shape.of(3))));
        var empty=lower(context(new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0,ScatterReduction.MUL)),List.of(0,1,2),
                List.of(desc(DataType.FLOAT32,Shape.of(3)),desc(DataType.INT32,Shape.of(0)),
                        desc(DataType.FLOAT32,Shape.of(0))),desc(DataType.FLOAT32,Shape.of(3))));
        assertAll(() -> assertEquals(List.of(0,1,0),((CpuScatterIr)dedup.portableKernelIr())
                        .occurrenceToBoundary()),
                () -> assertEquals(3,dedup.boundaryValues().size()),
                () -> assertEquals(64,product.scatterGeometry().orElseThrow().scratchSliceBytes()),
                () -> assertEquals(0,empty.scatterGeometry().orElseThrow().scratchSliceBytes()));
    }

    @Test void failsClosedForObsoleteShapeBoolArithmeticAndWrongNdFormula() {
        assertAll(() -> assertThrows(IllegalArgumentException.class,()->lower(context(
                        new Operation(AxisScatterKind.SCATTER_ADD,new IndexAxisAttrs(1)),List.of(0,1,2),
                        List.of(desc(DataType.FLOAT32,Shape.of(2,3)),desc(DataType.INT32,Shape.of(4)),
                                desc(DataType.FLOAT32,Shape.of(4))),
                        desc(DataType.FLOAT32,Shape.of(2,3))))),
                () -> assertThrows(IllegalArgumentException.class,()->lower(context(
                        new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                                new ScatterElementsAttrs(0,ScatterReduction.ADD)),List.of(0,1,2),
                        List.of(desc(DataType.BOOL,Shape.of(2)),desc(DataType.INT32,Shape.of(2)),
                                desc(DataType.BOOL,Shape.of(2))),desc(DataType.BOOL,Shape.of(2))))),
                () -> assertThrows(IllegalArgumentException.class,()->lower(context(
                        new Operation(ScatterNdKind.SCATTER_ND,new ScatterNdAttrs(0,ScatterReduction.NONE)),
                        List.of(0,1,2),List.of(desc(DataType.INT64,Shape.of(2,3)),
                                desc(DataType.INT32,Shape.of(2,1)),desc(DataType.INT64,Shape.of(2,2))),
                        desc(DataType.INT64,Shape.of(2,3))))));
    }

    @Test void productGeometryPacksReusableSeedsAndDisjointExactScratchSlices() {
        var product = lower(context(new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0, ScatterReduction.MUL)), List.of(0, 1, 2),
                List.of(desc(DataType.FLOAT32, Shape.of(5)),
                        desc(DataType.INT64, Shape.of(7)),
                        desc(DataType.FLOAT32, Shape.of(7))),
                desc(DataType.FLOAT32, Shape.of(5))));
        var geometry = product.scatterGeometry().orElseThrow();
        long[] first = geometry.pack(new long[]{3, 5, 7, 11}, 1, 3, 0);
        long[] second = geometry.pack(new long[]{3, 5, 7, 11}, 3, 5, 1);
        assertAll(
                () -> assertEquals(0, first[13]),
                () -> assertEquals(geometry.scratchSliceBytes(), second[13]),
                () -> assertEquals(geometry.scratchSliceBytes() * 2, geometry.workspaceBytes(2)),
                () -> assertEquals(1, first[16]),
                () -> assertEquals(1, first[17]));
    }

    @Test void geometryRejectsMalformedTypeShapeAndScratchFacts() {
        var layout3 = new CpuScatterLowering.Geometry.Layout(new long[]{3}, 0,
                new long[]{1});
        var layout2 = new CpuScatterLowering.Geometry.Layout(new long[]{2}, 0,
                new long[]{1});
        var layouts = List.of(layout3, layout2, layout2, layout3);
        var types = List.of(DataType.FLOAT32, DataType.INT32, DataType.FLOAT32,
                DataType.FLOAT32);
        long exactSlice = lower(context(new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0, ScatterReduction.MUL)), List.of(0, 1, 2),
                List.of(desc(DataType.FLOAT32, Shape.of(3)),
                        desc(DataType.INT32, Shape.of(2)),
                        desc(DataType.FLOAT32, Shape.of(2))),
                desc(DataType.FLOAT32, Shape.of(3)))).scatterGeometry().orElseThrow()
                .scratchSliceBytes();
        assertAll(
                () -> assertThrows(IllegalArgumentException.class, () ->
                        new CpuScatterLowering.Geometry(CpuScatterIr.Family.SCATTER_ELEMENTS,
                                ScatterReduction.MUL, List.of(0, 1, 2), layouts,
                                List.of(DataType.FLOAT32, DataType.INT32, DataType.INT64,
                                        DataType.FLOAT32), 0, 0, 0, 2, exactSlice)),
                () -> assertThrows(IllegalArgumentException.class, () ->
                        new CpuScatterLowering.Geometry(CpuScatterIr.Family.SCATTER_ELEMENTS,
                                ScatterReduction.MUL, List.of(0, 1, 2), layouts, types,
                                0, 0, 0, 3, exactSlice)),
                () -> assertThrows(IllegalArgumentException.class, () ->
                        new CpuScatterLowering.Geometry(CpuScatterIr.Family.SCATTER_ADD,
                                ScatterReduction.ADD, List.of(0, 1, 2),
                                List.of(layout3, layout2, layout3, layout3), types,
                                0, 0, 0, 2, 0)));
    }

    private static CpuPartitionLowering.LoweredPartition lower(PrepareContext<CpuPartitionAnalysisInputs> c){return new CpuPartitionLowering().lower(c);}
    public static PrepareContext<CpuPartitionAnalysisInputs> context(Operation operation,List<Integer> occurrences,List<io.github.pho001.synaptik.model.tensor.TensorDescriptor> inputs,io.github.pho001.synaptik.model.tensor.TensorDescriptor output){return CpuIndexingLoweringTest.context(operation,occurrences,inputs,output);}
    public static io.github.pho001.synaptik.model.tensor.TensorDescriptor desc(DataType type,Shape shape){return CpuIndexingLoweringTest.descriptor(type,shape);}
}
