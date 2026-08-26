package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.*;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.normalization.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.*;
import org.junit.jupiter.api.Test;

public class CpuBatchNormTrainingLoweringTest {
 @Test void derivesCompleteChannelGeometryFiveOutputsAndExactState(){var l=new CpuPartitionLowering().lower(context(Shape.of(2,3,4),1));var g=l.batchNormTrainingGeometry().orElseThrow();assertAll(()->assertEquals(2,g.prefixCount()),()->assertEquals(3,g.channelCount()),()->assertEquals(4,g.suffixCount()),()->assertEquals(8,g.reductionCount()),()->assertEquals(10,l.boundaryValues().size()),()->assertTrue(g.scratchSliceBytes()>0));}
 public static PrepareContext<CpuPartitionAnalysisInputs> context(Shape shape,int axis){Shape vector=Shape.of(shape.toLongArray()[axis]);var layouts=new ArrayList<LayoutDescriptor>();layouts.add(LayoutDescriptor.contiguous(shape));for(int i=1;i<5;i++)layouts.add(LayoutDescriptor.contiguous(vector));layouts.add(LayoutDescriptor.contiguous(shape));for(int i=1;i<5;i++)layouts.add(LayoutDescriptor.contiguous(vector));return context(Collections.nCopies(5,DataType.FLOAT32),shape,axis,List.of(0,1,2,3,4),layouts);}
 public static PrepareContext<CpuPartitionAnalysisInputs> context(List<DataType> types,Shape shape,int axis,List<Integer> occurrences,List<LayoutDescriptor> layouts){
  if(types.size()!=5||occurrences.size()!=5||layouts.size()!=10)throw new IllegalArgumentException("training test facts");Shape vector=Shape.of(shape.toLongArray()[axis]);DataType result=types.getFirst();for(int i=1;i<5;i++)result=DataTypePromotion.promoteFloating(result,types.get(i));var inputs=new ArrayList<TensorDescriptor>();for(int i=0;i<5;i++)inputs.add(new TensorDescriptor(types.get(i),i==0?shape:vector,Optional.of(layouts.get(i)),false));
  var outputs=List.of(new TensorDescriptor(result,shape,Optional.of(layouts.get(5)),false),new TensorDescriptor(result,vector,Optional.of(layouts.get(6)),false),new TensorDescriptor(result,vector,Optional.of(layouts.get(7)),false),new TensorDescriptor(result,vector,Optional.of(layouts.get(8)),false),new TensorDescriptor(result,vector,Optional.of(layouts.get(9)),false));
  List<ValueId> inIds=occurrences.stream().map(ValueId::new).toList();int unique=occurrences.stream().mapToInt(Integer::intValue).max().orElseThrow()+1;List<ValueId> outIds=java.util.stream.IntStream.range(unique,unique+5).mapToObj(ValueId::new).toList();ScalarValue momentum=result==DataType.FLOAT64?ScalarValue.float64(.25):result==DataType.FLOAT32?ScalarValue.float32(.25f):ScalarValue.bfloat16Bits((short)0x3e80);ScalarValue epsilon=result==DataType.FLOAT64?ScalarValue.float64(1e-5):result==DataType.FLOAT32?ScalarValue.float32(1e-5f):ScalarValue.bfloat16Bits((short)0x3728);
  var op=new Operation(BatchNormKind.BATCH_NORM_TRAINING,new BatchNormTrainingAttrs(axis,momentum,epsilon));var node=new CompiledNode(new NodeId(0),op,inIds,outIds);var partition=new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,List.of(node.id()));var values=new ArrayList<GraphValue>();var memory=new ArrayList<LogicalMemoryRequirement>();
  for(int i=0;i<5;i++)if(occurrences.indexOf(occurrences.get(i))==i){values.add(new GraphValue(inIds.get(i),inputs.get(i)));memory.add(new LogicalMemoryRequirement(inIds.get(i),inputs.get(i),Optional.empty(),List.of(partition),false));}
  for(int i=0;i<5;i++){values.add(new GraphValue(outIds.get(i),outputs.get(i)));memory.add(new LogicalMemoryRequirement(outIds.get(i),outputs.get(i),Optional.of(partition),List.of(),true));}
  return new PrepareContext<>(partition,List.of(node),values,memory,Map.of(),CpuPartitionAnalysisInputs.DEFAULT);}
}
