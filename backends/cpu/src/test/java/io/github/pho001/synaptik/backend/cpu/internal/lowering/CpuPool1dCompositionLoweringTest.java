package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPool2dIr;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGenerator;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.pooling.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.*;
import org.junit.jupiter.api.Test;

class CpuPool1dCompositionLoweringTest {
    @Test void exactPrivateChainReusesSchema55Pool2dBodyAndVirtualizesViews(){
        var context=context(2,false);var lowered=new CpuPartitionLowering().lower(context);
        assertAll(()->assertInstanceOf(CpuPool2dIr.class,lowered.portableKernelIr()),
                ()->assertEquals(List.of(new ValueId(1),new ValueId(2)),lowered.virtualValues()),
                ()->assertEquals(List.of(new ValueId(0),new ValueId(3)),lowered.boundaryValues()),
                ()->assertTrue(lowered.pool2dGeometry().isPresent()),
                ()->assertTrue(lowered.pool3dGeometry().isEmpty()));
        var plan=new CpuPartitionPreparer().analyze(context).plan();
        assertAll(()->assertEquals(55,plan.units().getFirst().portablePlan().specialization().classIdentitySchema()),
                ()->assertTrue(plan.workspaceDeclaration().isEmpty()),
                ()->assertTrue(plan.materialization().isEmpty()));
    }
    @Test void recognizedCompositionHasByteIdenticalDirectPool2dBody(){
        var recognized=new CpuPartitionPreparer().analyze(context(2,false)).plan()
                .units().getFirst().portablePlan();
        var directContext=CpuPool2dLoweringTest.context(Pool2dKind.MAX_POOL2D,
                new MaxPool2dAttrs(1,2,1,1,0,0,1,1,false),DataType.FLOAT32,
                Shape.of(1,1,1,4),Shape.of(1,1,1,3));
        var direct=new CpuPartitionPreparer().analyze(directContext).plan()
                .units().getFirst().portablePlan();
        var generator=new CpuClassFileKernelGenerator();
        assertAll(()->assertEquals(direct.kernelIr(),recognized.kernelIr()),
                ()->assertEquals(direct.specialization(),recognized.specialization()),
                ()->assertArrayEquals(generator.generateClassBytes(direct.specialization(),
                                direct.kernelIr()),
                        generator.generateClassBytes(recognized.specialization(),
                                recognized.kernelIr())));
    }
    @Test void malformedAxisAndPublishedIntermediateFailClosed(){
        var published=context(2,true);var squeeze=published.nodes().get(2);
        var descriptors=published.values().stream().map(GraphValue::descriptor).toList();
        assertAll(()->assertThrows(IllegalArgumentException.class,()->new CpuPartitionLowering().lower(context(1,false))),
                ()->assertThrows(IllegalArgumentException.class,()->new CpuPartitionLowering().lower(context(2,true))),
                ()->assertTrue(new CpuCapabilityProvider().supports(
                        new io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery(
                                squeeze.operation(),List.of(descriptors.get(2)),List.of(descriptors.get(3))))),
                ()->assertTrue(new CpuPartitionDagDecomposer().decompose(validHeightNearMatch(),
                        new CpuPartitionLowering()).size()>1));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> validHeightNearMatch(){
        var base=context(2,false);var nodes=new ArrayList<>(base.nodes());var pool=nodes.get(1);
        nodes.set(1,new CompiledNode(pool.id(),new Operation(Pool2dKind.MAX_POOL2D,
                new MaxPool2dAttrs(3,2,1,1,1,0,1,1,false)),pool.inputs(),pool.outputs()));
        return new PrepareContext<>(base.partition(),nodes,base.values(),base.memoryRequirements(),
                base.constants(),base.backendInputs());
    }

    static PrepareContext<CpuPartitionAnalysisInputs> context(int axis,boolean publishIntermediate){
        Shape x=Shape.of(1,1,4),ex=Shape.of(1,1,1,4),py=Shape.of(1,1,1,3),y=Shape.of(1,1,3);
        List<ValueId> ids=java.util.stream.LongStream.range(0,4).mapToObj(ValueId::new).toList();
        List<CompiledNode> nodes=List.of(
                new CompiledNode(new NodeId(0),new Operation(AxisTransformKind.EXPAND_DIMS,new AxisTransformAttrs(axis)),List.of(ids.get(0)),List.of(ids.get(1))),
                new CompiledNode(new NodeId(1),new Operation(Pool2dKind.MAX_POOL2D,new MaxPool2dAttrs(1,2,1,1,0,0,1,1,false)),List.of(ids.get(1)),List.of(ids.get(2))),
                new CompiledNode(new NodeId(2),new Operation(AxisTransformKind.SQUEEZE,new AxisTransformAttrs(2)),List.of(ids.get(2)),List.of(ids.get(3))));
        var partition=new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,nodes.stream().map(CompiledNode::id).toList());
        List<TensorDescriptor> descriptors=List.of(
                new TensorDescriptor(DataType.FLOAT32,x,Optional.of(LayoutDescriptor.contiguous(x)),false),
                new TensorDescriptor(DataType.FLOAT32,ex,Optional.of(LayoutDescriptor.of(ex,new long[]{4,4,4,1},0,true)),false),
                new TensorDescriptor(DataType.FLOAT32,py,Optional.of(LayoutDescriptor.of(py,new long[]{3,3,3,1},0,true)),false),
                new TensorDescriptor(DataType.FLOAT32,y,
                        Optional.of(LayoutDescriptor.of(y,new long[]{3,3,1},0,true)),false));
        var values=new ArrayList<GraphValue>();var memory=new ArrayList<LogicalMemoryRequirement>();
        for(int i=0;i<4;i++){values.add(new GraphValue(ids.get(i),descriptors.get(i)));boolean produced=i>0;
            boolean published=i==3||publishIntermediate&&i==1;
            memory.add(new LogicalMemoryRequirement(ids.get(i),descriptors.get(i),
                    produced?Optional.of(partition):Optional.empty(),i<3?List.of(partition):List.of(),
                    published));}
        return new PrepareContext<>(partition,nodes,values,memory,Map.of(),CpuPartitionAnalysisInputs.DEFAULT);
    }
}
