package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGenerator;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv2dIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CpuConv1dCompositionLoweringTest {
    @Test void foldsOnlyExactVisibleTopologyAndReusesConv2dBody() throws Throwable {
        var context=context(2);var lowered=new CpuPartitionLowering().lower(context);
        assertAll(()->assertInstanceOf(CpuConv2dIr.class,lowered.portableKernelIr()),()->assertEquals(3,lowered.virtualValues().size()),()->assertEquals(4,lowered.boundaryValues().size()),()->assertEquals(0,lowered.affineAddressPairs().length),()->assertTrue(lowered.conv2dGeometry().isPresent()),()->assertTrue(lowered.conv3dGeometry().isEmpty()));
        var plan=new CpuPartitionPreparer().analyze(context).plan();var unit=plan.units().getFirst();var route=unit.portablePlan();var generator=new CpuClassFileKernelGenerator();var handle=generator.defineClassBytes(route.specialization(),generator.generateClassBytes(route.specialization(),route.kernelIr())).entryPoint();
        float[] input={1,2,3,4,5,6,7,8},weight={1,0,-1,2,1,0,-1,0,1,1,1,1},bias={.5f,-1f},output=new float[8];handle.invokeExact(input,weight,bias,output,unit.conv2dGeometry().orElseThrow().pack(new long[4]),0L,8L);assertArrayEquals(new float[]{3.5f,14.5f,17.5f,25.5f,12f,19f,22f,11f},output);
        assertAll(()->assertTrue(plan.workspaceDeclaration().isEmpty()),()->assertTrue(plan.materialization().isEmpty()));
    }

    @Test void wrongAxisFailsClosed(){assertThrows(IllegalArgumentException.class,()->new CpuPartitionLowering().lower(context(1)));}

    static PrepareContext<CpuPartitionAnalysisInputs> context(int axis){
        Shape x=Shape.of(1,2,4),w=Shape.of(2,2,3),b=Shape.of(2),ex=Shape.of(1,2,1,4),ew=Shape.of(2,2,1,3),cy=Shape.of(1,2,1,4),y=Shape.of(1,2,4);
        List<Shape> shapes=List.of(x,w,b,ex,ew,cy,y);List<DataType> types=java.util.Collections.nCopies(7,DataType.FLOAT32);List<ValueId> ids=java.util.stream.LongStream.range(0,7).mapToObj(ValueId::new).toList();
        List<CompiledNode> nodes=List.of(new CompiledNode(new NodeId(0),new Operation(AxisTransformKind.EXPAND_DIMS,new AxisTransformAttrs(axis)),List.of(ids.get(0)),List.of(ids.get(3))),new CompiledNode(new NodeId(1),new Operation(AxisTransformKind.EXPAND_DIMS,new AxisTransformAttrs(2)),List.of(ids.get(1)),List.of(ids.get(4))),new CompiledNode(new NodeId(2),new Operation(Conv2dKind.CONV2D,new Conv2dAttrs(1,1,0,1,1,1,1)),List.of(ids.get(3),ids.get(4),ids.get(2)),List.of(ids.get(5))),new CompiledNode(new NodeId(3),new Operation(AxisTransformKind.SQUEEZE,new AxisTransformAttrs(2)),List.of(ids.get(5)),List.of(ids.get(6))));
        var partition=new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,nodes.stream().map(CompiledNode::id).toList());var values=new ArrayList<GraphValue>();var memory=new ArrayList<LogicalMemoryRequirement>();
        for(int i=0;i<ids.size();i++){var descriptor=new TensorDescriptor(types.get(i),shapes.get(i),Optional.of(LayoutDescriptor.contiguous(shapes.get(i))),false);values.add(new GraphValue(ids.get(i),descriptor));boolean produced=i>=3,published=i==6;memory.add(new LogicalMemoryRequirement(ids.get(i),descriptor,produced?Optional.of(partition):Optional.empty(),published?List.of():List.of(partition),published));}
        return new PrepareContext<>(partition,nodes,values,memory,Map.of(),new CpuPartitionAnalysisInputs(false,java.util.Collections.nCopies(4,CarrierAccess.FLOAT_ARRAY)));
    }

    static PrepareContext<CpuPartitionAnalysisInputs> contextWithSuffix(boolean thirdOperation) {
        var base = context(2);
        var nodes = new ArrayList<>(base.nodes());
        nodes.add(new CompiledNode(new NodeId(4), new Operation(BinaryArithmeticKind.ADD,
                NoOperationAttrs.INSTANCE), List.of(new ValueId(6), new ValueId(7)),
                List.of(new ValueId(8))));
        nodes.add(new CompiledNode(new NodeId(5), new Operation(UnaryElementwiseKind.RELU,
                NoOperationAttrs.INSTANCE), List.of(new ValueId(8)), List.of(new ValueId(9))));
        if (thirdOperation) nodes.add(new CompiledNode(new NodeId(6),
                new Operation(UnaryElementwiseKind.TANH, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(9)), List.of(new ValueId(10))));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var values = new ArrayList<>(base.values());
        Shape y = Shape.of(1, 2, 4);
        for (long id = 7; id <= (thirdOperation ? 10 : 9); id++) {
            values.add(new GraphValue(new ValueId(id), new TensorDescriptor(DataType.FLOAT32, y,
                    Optional.of(LayoutDescriptor.contiguous(y)), false)));
        }
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (GraphValue value : values) {
            ValueId id = value.id();
            boolean produced = id.value() >= 3 && id.value() != 7;
            boolean published = id.value() == (thirdOperation ? 10 : 9);
            boolean consumed = nodes.stream().anyMatch(node -> node.inputs().contains(id));
            memory.add(new LogicalMemoryRequirement(id, value.descriptor(),
                    produced ? Optional.of(partition) : Optional.empty(),
                    consumed && !published ? List.of(partition) : List.of(), published));
        }
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(),
                CpuPartitionAnalysisInputs.DEFAULT);
    }
}
