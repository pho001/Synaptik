package io.github.pho001.synaptik.backend.cpu;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CpuConv3dCapabilityTest {
    private final CpuCapabilityProvider provider=new CpuCapabilityProvider();

    @Test void admitsExactStaticGroupedPromotionAndRejectsExcludedBoundaries(){
        var attrs=new Conv3dAttrs(2,1,2,1,0,1,1,2,1,2);Shape x=Shape.of(2,4,5,6,7),w=Shape.of(6,2,3,2,2),b=Shape.of(6),y=Shape.of(2,6,3,4,4);
        assertTrue(provider.supports(query(attrs,List.of(descriptor(DataType.BFLOAT16,x,false),descriptor(DataType.FLOAT32,w,true),descriptor(DataType.FLOAT64,b,false)),descriptor(DataType.FLOAT64,y,true))));
        assertAll(()->assertFalse(provider.supports(query(attrs,List.of(descriptor(DataType.INT32,x,false),descriptor(DataType.FLOAT32,w,false)),descriptor(DataType.FLOAT32,y,false)))),()->assertFalse(provider.supports(query(attrs,List.of(unresolved(DataType.FLOAT32,x),descriptor(DataType.FLOAT32,w,false)),descriptor(DataType.FLOAT32,y,false)))),()->assertFalse(provider.supports(query(attrs,List.of(descriptor(DataType.FLOAT32,x,false),descriptor(DataType.FLOAT32,w,false)),descriptor(DataType.FLOAT32,Shape.of(2,6,3,4,5),false)))));
    }

    private static OperationCapabilityQuery query(Conv3dAttrs attrs,List<TensorDescriptor> inputs,TensorDescriptor output){return new OperationCapabilityQuery(new Operation(Conv3dKind.CONV3D,attrs),inputs,List.of(output));}
    private static TensorDescriptor descriptor(DataType type,Shape shape,boolean grad){return new TensorDescriptor(type,shape,Optional.of(LayoutDescriptor.contiguous(shape)),grad);}
    private static TensorDescriptor unresolved(DataType type,Shape shape){return new TensorDescriptor(type,shape,Optional.empty(),false);}
}
