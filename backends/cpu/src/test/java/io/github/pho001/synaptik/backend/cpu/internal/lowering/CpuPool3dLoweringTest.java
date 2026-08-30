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
                DataType.BFLOAT16,Shape.of(1,2,7,8,9),Shape.of(1,2,3,4,3)))
                .pool3dGeometry().orElseThrow();
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

    public static PrepareContext<CpuPartitionAnalysisInputs> context(Pool3dKind kind,
            OperationAttrs attrs,DataType type,Shape inputShape,Shape outputShape){
        var input=new TensorDescriptor(type,inputShape,Optional.of(LayoutDescriptor.contiguous(inputShape)),false);
        var output=new TensorDescriptor(type,outputShape,Optional.of(LayoutDescriptor.contiguous(outputShape)),false);
        return CpuScatterLoweringTest.context(new Operation(kind,attrs),List.of(0),List.of(input),output);
    }
}
