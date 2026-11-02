package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.pooling.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

public class CpuPool3dLoweringTest {
    @Test void lowersLiteralCeilGeometryWithSchema56AndZeroWorkspace() {
        var context=context(Pool3dKind.AVERAGE_POOL3D,
                new AveragePool3dAttrs(2,2,2,2,2,2,2,2,2,1,1,1,true),DataType.FLOAT32,
                Shape.of(1,2,3,3,3),Shape.of(1,2,4,4,4));
        var lowered=new CpuPool3dLowering().lower(context);
        assertEquals(128,lowered.pool3dGeometry().orElseThrow().outputCount());
        var plan=new io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer()
                .analyze(context).plan();
        assertAll(()->assertEquals(56,plan.units().getFirst().portablePlan().specialization().classIdentitySchema()),
                ()->assertTrue(plan.units().getFirst().pool3dGeometry().isPresent()),
                ()->assertTrue(plan.workspaceDeclaration().isEmpty()));
    }

    @Test void rejectsOutputShapeDisagreement() {
        assertThrows(IllegalArgumentException.class,()->new CpuPool3dLowering().lower(context(
                Pool3dKind.MAX_POOL3D,new MaxPool3dAttrs(1,1,1,1,1,1,0,0,0,1,1,1,false),
                DataType.FLOAT64,Shape.of(1,1,2,2,2),Shape.of(1,1,1,1,1))));
    }

    @Test void lowersAsymmetricDepthHeightWidthFloorAndCeilGrids() {
        var floorAttrs=new MaxPool3dAttrs(2,3,4,2,3,2,1,2,1,2,1,2,false);
        var floor=new CpuPool3dLowering().lower(context(Pool3dKind.MAX_POOL3D,floorAttrs,
                DataType.FLOAT64,Shape.of(1,2,7,8,9),Shape.of(1,2,4,4,3)))
                .pool3dGeometry().orElseThrow();
        var ceilAttrs=new AveragePool3dAttrs(2,3,4,4,4,3,1,2,1,2,1,2,true);
        var ceil=new CpuPool3dLowering().lower(context(Pool3dKind.AVERAGE_POOL3D,ceilAttrs,
                DataType.FLOAT32,Shape.of(1,2,7,8,9),Shape.of(1,2,3,4,3)))
                .pool3dGeometry().orElseThrow();
        for (var type : java.util.List.of(DataType.BFLOAT16, DataType.FLOAT16))
            assertTrue(io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest
                    .assertLowUnsupported(context(Pool3dKind.AVERAGE_POOL3D,ceilAttrs,type,
                            Shape.of(1,2,7,8,9),Shape.of(1,2,3,4,3))));
        assertAll(()->assertEquals(24,floor.divisor()),
                ()->assertArrayEquals(new long[]{1,2,4,4,3},floor.output().extents()),
                ()->assertEquals(96,floor.outputCount()),
                ()->assertEquals(4,ceil.strideDepth()),
                ()->assertEquals(2,ceil.dilationDepth()),
                ()->assertArrayEquals(new long[]{1,2,3,4,3},ceil.output().extents()),
                ()->assertEquals(72,ceil.outputCount()));
    }

    @Test void zeroBatchProducesZeroCellsWithoutWorkspace() {
        var lowered=new CpuPool3dLowering().lower(context(Pool3dKind.AVERAGE_POOL3D,
                new AveragePool3dAttrs(1,1,1,1,1,1,0,0,0,1,1,1,false),DataType.FLOAT32,
                Shape.of(0,2,3,4,5),Shape.of(0,2,3,4,5)));
        assertAll(()->assertEquals(0,lowered.elementCount()),
                ()->assertEquals(0,lowered.pool3dGeometry().orElseThrow().outputCount()));
    }

    @Test void gradientMetadataPreservesForwardGeometryIdentityAndWorkspace() {
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (var kind : Pool3dKind.values()) {
                OperationAttrs attrs = kind == Pool3dKind.MAX_POOL3D
                        ? new MaxPool3dAttrs(2,2,2,1,2,2,1,1,1,2,1,1,true)
                        : new AveragePool3dAttrs(2,2,2,1,2,2,1,1,1,2,1,1,true);
                Shape input = Shape.of(1,2,3,4,5), output = Shape.of(1,2,3,3,4);
                var ordinary = new CpuPool3dLowering().lower(context(kind, attrs, type, input, output));
                var context = context(kind, attrs, type, input, output, true, true);
                var gradient = new CpuPool3dLowering().lower(context);
                assertEquals(ordinary.portableKernelIr().structuralKey(), gradient.portableKernelIr().structuralKey());
                assertArrayEquals(ordinary.pool3dGeometry().orElseThrow().pack(0, 0),
                        gradient.pool3dGeometry().orElseThrow().pack(0, 0));
                var plan = new io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer()
                        .analyze(context).plan();
                assertTrue(plan.workspaceDeclaration().isEmpty());
                assertTrue(plan.materializations().isEmpty());
                assertThrows(IllegalArgumentException.class, () -> new CpuPool3dLowering().lower(
                        context(kind, attrs, type, input, output, true, false)));
                assertThrows(IllegalArgumentException.class, () -> new CpuPool3dLowering().lower(
                        context(kind, attrs, type, input, output, false, true)));
            }
    }

    @Test void qualifiedBatchGradientRecipesDoNotGrantOrdinarySymbolicPooling() {
        var b = new io.github.pho001.synaptik.model.shape.DynamicDimension("B");
        Shape input = Shape.ofDimensions(b, new io.github.pho001.synaptik.model.shape.StaticDimension(1),
                new io.github.pho001.synaptik.model.shape.StaticDimension(3),
                new io.github.pho001.synaptik.model.shape.StaticDimension(3),
                new io.github.pho001.synaptik.model.shape.StaticDimension(3));
        Shape output = Shape.ofDimensions(b, new io.github.pho001.synaptik.model.shape.StaticDimension(1),
                new io.github.pho001.synaptik.model.shape.StaticDimension(2),
                new io.github.pho001.synaptik.model.shape.StaticDimension(2),
                new io.github.pho001.synaptik.model.shape.StaticDimension(2));
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (var kind : Pool3dKind.values()) {
                OperationAttrs attrs = kind == Pool3dKind.MAX_POOL3D
                        ? new MaxPool3dAttrs(2,2,2,1,1,1,0,0,0,1,1,1,false)
                        : new AveragePool3dAttrs(2,2,2,1,1,1,0,0,0,1,1,1,false);
                var context = context(kind, attrs, type, input, output, true, true);
                assertFalse(new io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider().supports(
                        new io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery(
                                context.nodes().getFirst().operation(),
                                List.of(context.values().getFirst().descriptor()),
                                List.of(context.values().getLast().descriptor()))));
                var recipe = new CpuPool3dLowering().lowerBatch(context).batchGeometry().orElseThrow();
                for (long batch : new long[]{0, 1, 3}) {
                    var resolved = recipe.bind(batch);
                    var fixed = new CpuPool3dLowering().lower(context(kind, attrs, type,
                            Shape.of(batch,1,3,3,3), Shape.of(batch,1,2,2,2), true, true));
                    assertEquals(fixed.portableKernelIr().structuralKey(), resolved.portableKernelIr().structuralKey());
                    assertArrayEquals(fixed.pool3dGeometry().orElseThrow().pack(0,0),
                            resolved.pool3dGeometry().orElseThrow().pack(0,0));
                }
                assertThrows(IllegalArgumentException.class, () -> new CpuPool3dLowering().lowerBatch(
                        context(kind, attrs, type, input, output, true, false)));
                assertThrows(IllegalArgumentException.class, () -> new CpuPool3dLowering().lowerBatch(
                        context(kind, attrs, type, input, output, false, true)));
            }
        for (var type : List.of(DataType.BFLOAT16, DataType.FLOAT16))
            assertThrows(IllegalArgumentException.class, () -> new CpuPool3dLowering().lowerBatch(
                    context(Pool3dKind.MAX_POOL3D,
                            new MaxPool3dAttrs(2,2,2,1,1,1,0,0,0,1,1,1,false),
                            type, input, output, true, true)));
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(Pool3dKind kind,
            OperationAttrs attrs,DataType type,Shape inputShape,Shape outputShape){
        return context(kind, attrs, type, inputShape, outputShape, false, false);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(Pool3dKind kind,
            OperationAttrs attrs, DataType type, Shape inputShape, Shape outputShape,
            boolean inputGrad, boolean outputGrad) {
        var input=new TensorDescriptor(type,inputShape,inputShape.isFullyStatic()
                ? Optional.of(LayoutDescriptor.contiguous(inputShape)) : Optional.empty(),inputGrad);
        var output=new TensorDescriptor(type,outputShape,outputShape.isFullyStatic()
                ? Optional.of(LayoutDescriptor.contiguous(outputShape)) : Optional.empty(),outputGrad);
        return CpuScatterLoweringTest.context(new Operation(kind,attrs),List.of(0),List.of(input),output);
    }
}
