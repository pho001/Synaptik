package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv3dIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

public class CpuConv3dLoweringTest {
    @Test void derivesRankFiveGroupedZeroWorkspaceGeometry() {
        var lowered=new CpuPartitionLowering().lower(context(List.of(DataType.FLOAT32,DataType.BFLOAT16,DataType.FLOAT64),Shape.of(2,4,7,8,9),Shape.of(6,2,3,2,2),Shape.of(2,6,3,5,5),new Conv3dAttrs(2,2,2,1,1,1,2,1,1,2),null));
        var ir=(CpuConv3dIr)lowered.portableKernelIr();
        assertAll(()->assertEquals(DataType.FLOAT64,ir.resultType()),()->assertEquals(4,lowered.boundaryValues().size()),()->assertEquals(900,lowered.elementCount()),()->assertEquals(2,ir.groups()),()->assertTrue(lowered.virtualValues().isEmpty()),()->assertEquals(4,lowered.conv3dGeometry().orElseThrow().boundaries().size()),()->assertTrue(lowered.conv2dGeometry().isEmpty()));
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(List<DataType> types,
            Shape inputShape,Shape weightShape,Shape outputShape,Conv3dAttrs attrs,
            List<LayoutDescriptor> selectedLayouts){
        boolean bias=types.size()==3;var shapes=new ArrayList<Shape>();shapes.add(inputShape);shapes.add(weightShape);if(bias)shapes.add(Shape.of(weightShape.toLongArray()[0]));shapes.add(outputShape);
        var layouts=selectedLayouts==null?shapes.stream().map(LayoutDescriptor::contiguous).toList():List.copyOf(selectedLayouts);var descriptors=new ArrayList<TensorDescriptor>();
        for(int i=0;i<types.size();i++)descriptors.add(new TensorDescriptor(types.get(i),shapes.get(i),Optional.of(layouts.get(i)),false));
        DataType result=types.getFirst();for(int i=1;i<types.size();i++)result=io.github.pho001.synaptik.model.datatype.DataTypePromotion.promoteFloating(result,types.get(i));
        var output=new TensorDescriptor(result,outputShape,Optional.of(layouts.getLast()),false);List<Integer> occurrences=new ArrayList<>();for(int i=0;i<types.size();i++)occurrences.add(i);
        return CpuScatterLoweringTest.context(new Operation(Conv3dKind.CONV3D,attrs),occurrences,descriptors,output);
    }
}
