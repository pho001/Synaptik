package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFoldIr;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBorrowedBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.*;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.runtime.memory.*;
import io.github.pho001.synaptik.runtime.run.*;
import java.lang.foreign.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Consumer proof of compact declarations, fresh injection and complete fail-before-write binding. */
class CpuBroadcastFoldConformanceTest {
    @Test void compactResourcesBindFreshValuesAndRejectInvalidPhysicalStorageBeforeWrites() {
        for(var type:List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            var analysis=new CpuPartitionPreparer().analyze(context(type));var plan=analysis.plan();
            assertEquals(1,plan.units().size());assertEquals(2,analysis.requirements().size());
            assertEquals(List.of(new ValueId(0),new ValueId(6)),plan.boundaryValues());
            assertEquals(2L*3*28*type.byteWidth(),plan.bufferDeclarations().getFirst().byteSize());
            assertEquals(2L*3*5*7*type.byteWidth(),plan.bufferDeclarations().getLast().byteSize());
            assertTrue(((CpuFoldIr)plan.units().getFirst().portablePlan().portableKernelIr()).kernelBroadcastInput());
            assertTrue(plan.workspaceDeclaration().isEmpty());assertTrue(plan.materialization().isEmpty());
            var buffers=new ArrayList<PreparedMemoryPlan.BufferEntry>();var assignments=new ArrayList<PreparationResourceAssignment>();
            for(var declaration:plan.bufferDeclarations()) {
                var slot=new BufferSlot(buffers.size());
                buffers.add(new PreparedMemoryPlan.BufferEntry(slot,declaration.byteSize(),declaration.byteAlignment()));
                assignments.add(new PreparationResourceAssignment.Buffer(declaration,slot,buffers.size()-1,0));
            }
            var memory=new PreparedMemoryPlan(buffers,List.of());
            var executable=new CpuPartitionFinalizer().finalizePartition(new BackendPartitionFinalization<>(analysis,memory,assignments)).executable();
            try(var arena=Arena.ofConfined()) {
                for(int run=1;run<=2;run++) {
                    var source=arena.allocate(168L*type.byteWidth(),type.byteWidth());
                    var output=arena.allocate(210L*type.byteWidth(),type.byteWidth());
                    for(int i=0;i<168;i++) write(type,source,i,run*(i%13-6)/7d);
                    try(var in=CpuBorrowedBuffer.borrow(new MemorySegmentStorage(type,168,source));
                            var out=CpuBorrowedBuffer.borrow(new MemorySegmentStorage(type,210,output))) {
                        executable.bind(new RunState(memory,List.of(List.of(new BufferRepresentationBinding(in,RunResourceOwnership.BORROWED)),
                                List.of(new BufferRepresentationBinding(out,RunResourceOwnership.BORROWED))),List.of())).execute();
                        double[] expected=new double[210];
                        for(int nc=0;nc<6;nc++) for(int ky=0;ky<2;ky++) for(int kx=0;kx<3;kx++)
                            for(int p=0;p<28;p++) {
                                int y=p/7+ky,x=p%7-1+kx;
                                if(x<0||x>=7)continue;
                                int target=nc*35+y*7+x;
                                double sum=expected[target]+read(type,source,nc*28+p);
                                expected[target]=type==DataType.FLOAT32?(float)sum:sum;
                            }
                        for(int i=0;i<210;i++) assertEquals(expected[i],read(type,output,i));
                    }
                }
                // Heap-backed segment aliases, short spans, misalignment, type and output access.
                for(int bad=0;bad<6;bad++) {
                    MemorySegment output=type==DataType.FLOAT32?MemorySegment.ofArray(new float[210]):MemorySegment.ofArray(new double[210]);
                    output.fill((byte)0x55);byte[] unchanged=output.toArray(ValueLayout.JAVA_BYTE);
                    MemorySegment source=arena.allocate(168L*type.byteWidth(),type.byteWidth());
                    if(bad==0)source=output.asSlice(0,168L*type.byteWidth());
                    if(bad==1)source=source.asSlice(0,source.byteSize()-type.byteWidth());
                    if(bad==2)source=arena.allocate(source.byteSize()+1,8).asSlice(1,source.byteSize());
                    MemorySegment selectedOutput=bad==3?output.asReadOnly():bad==4?output.asSlice(0,output.byteSize()-type.byteWidth()):output;
                    var inputType=bad==5?(type==DataType.FLOAT32?DataType.INT32:DataType.INT64):type;
                    try(var in=CpuBorrowedBuffer.borrow(new MemorySegmentStorage(inputType,source.byteSize()/type.byteWidth(),source));
                            var out=CpuBorrowedBuffer.borrow(new MemorySegmentStorage(type,selectedOutput.byteSize()/type.byteWidth(),selectedOutput))) {
                        var state=new RunState(memory,List.of(List.of(new BufferRepresentationBinding(in,RunResourceOwnership.BORROWED)),
                                List.of(new BufferRepresentationBinding(out,RunResourceOwnership.BORROWED))),List.of());
                        assertThrows(IllegalArgumentException.class,()->executable.bind(state));
                        assertArrayEquals(unchanged,output.toArray(ValueLayout.JAVA_BYTE));
                    }
                }
            }
        }
    }

    @Test void lowFormatsAccumulateEveryFoldContributorOnceAndRejectInvalidStorageBeforeWrites() {
        for(var type:LowPrecisionValues.TYPES) {
            var analysis=new CpuPartitionPreparer().analyze(context(type));var plan=analysis.plan();
            var buffers=new ArrayList<PreparedMemoryPlan.BufferEntry>();var assignments=new ArrayList<PreparationResourceAssignment>();
            for(var declaration:plan.bufferDeclarations()) {
                var slot=new BufferSlot(buffers.size());
                buffers.add(new PreparedMemoryPlan.BufferEntry(slot,declaration.byteSize(),declaration.byteAlignment()));
                assignments.add(new PreparationResourceAssignment.Buffer(declaration,slot,buffers.size()-1,0));
            }
            var memory=new PreparedMemoryPlan(buffers,List.of());
            var executable=new CpuPartitionFinalizer().finalizePartition(new BackendPartitionFinalization<>(analysis,memory,assignments)).executable();
            try(var arena=Arena.ofConfined()) {
                for(int run=1;run<=2;run++) {
                    var bindings=new ArrayList<List<BufferRepresentationBinding>>();
                    var segments=new HashMap<ValueId,MemorySegment>();
                    for(var declaration:plan.bufferDeclarations()) {
                        var segment=arena.allocate(declaration.byteSize(),declaration.byteAlignment());
                        segments.put(declaration.valueId(),segment);
                        bindings.add(List.of(new BufferRepresentationBinding(CpuBorrowedBuffer.borrow(
                                new MemorySegmentStorage(type,declaration.byteSize()/type.byteWidth(),segment)),RunResourceOwnership.RUN_OWNED)));
                    }
                    var source=segments.get(new ValueId(0));var output=segments.get(new ValueId(6));
                    float half=Math.scalb(1f,type==DataType.BFLOAT16?-8:type==DataType.FLOAT16?-11:
                            type==DataType.FLOAT8_E4M3?-4:-3);
                    for(int i=0;i<168;i++) LowPrecisionValues.put(type,source,i,
                            i%7==0?run:i%7==1?half:i%7==2?-run:(i%13-6)/7f);
                    try(var state=new RunState(memory,bindings,List.of())) {
                        executable.bind(state).execute();
                    }
                    float[] expected=new float[210];
                    for(int nc=0;nc<6;nc++)for(int ky=0;ky<2;ky++)for(int kx=0;kx<3;kx++)for(int p=0;p<28;p++) {
                        int y=p/7+ky,x=p%7-1+kx;
                        if(x<0||x>=7)continue;
                        expected[nc*35+y*7+x]+=LowPrecisionValues.decode(type,LowPrecisionValues.raw(type,source,nc*28+p));
                    }
                    for(int i=0;i<210;i++)assertEquals(LowPrecisionValues.encode(type,expected[i]),LowPrecisionValues.raw(type,output,i));
                }
                for(int bad=0;bad<6;bad++) {
                    if(bad==2&&type.byteWidth()==1)continue; // Every byte address is naturally aligned.
                    var segments=new HashMap<ValueId,MemorySegment>();
                    for(var d:plan.bufferDeclarations())segments.put(d.valueId(),arena.allocate(d.byteSize(),d.byteAlignment()));
                    var output=segments.get(new ValueId(6));output.fill((byte)0x55);
                    byte[] unchanged=output.toArray(ValueLayout.JAVA_BYTE);
                    var source=segments.get(new ValueId(0));
                    if(bad==0)source=output.asSlice(0,source.byteSize());
                    if(bad==1)source=source.asSlice(0,source.byteSize()-type.byteWidth());
                    if(bad==2)source=arena.allocate(source.byteSize()+1,type.byteWidth()).asSlice(1,source.byteSize());
                    segments.put(new ValueId(0),source);
                    if(bad==3)segments.put(new ValueId(6),output.asReadOnly());
                    if(bad==4)segments.put(new ValueId(6),output.asSlice(0,output.byteSize()-type.byteWidth()));
                    var bindings=new ArrayList<List<BufferRepresentationBinding>>();
                    for(var d:plan.bufferDeclarations()) {
                        var s=segments.get(d.valueId());
                        var physical=bad==5&&d.valueId().equals(new ValueId(0))
                                ?(type==DataType.BFLOAT16?DataType.FLOAT16:type==DataType.FLOAT16?DataType.BFLOAT16:
                                type==DataType.FLOAT8_E4M3?DataType.FLOAT8_E5M2:DataType.FLOAT8_E4M3):type;
                        bindings.add(List.of(new BufferRepresentationBinding(CpuBorrowedBuffer.borrow(
                                new MemorySegmentStorage(physical,s.byteSize()/type.byteWidth(),s)),RunResourceOwnership.RUN_OWNED)));
                    }
                    try(var state=new RunState(memory,bindings,List.of())) {
                        assertThrows(IllegalArgumentException.class,()->executable.bind(state));
                        assertArrayEquals(unchanged,output.toArray(ValueLayout.JAVA_BYTE));
                    }
                }
            }
        }
    }


    /**
     * Builds the raw rank-edit/repetition/copy chain with resolved final output and private mappings.
     * This metadata-only fixture borrows no physical storage and allocates no runtime resources.
     * @param type non-null F32/F64 compact execution type or low floating ordinary execution type
     * @return new immutable CPU context for [2,3,28] source and [2,3,5,7] output; never null
     * @throws NullPointerException if type is null
     */
    private static PrepareContext<CpuPartitionAnalysisInputs> context(DataType type) {
        Shape source=Shape.of(2,3,28),output=Shape.of(2,3,5,7);
        var tensors=new ArrayList<Tensor>();
        tensors.add(TensorFactory.create(new TensorDescriptor(type,source,Optional.of(LayoutDescriptor.contiguous(source)),false)));
        tensors.add(tensors.getLast().expandDims(2));tensors.add(tensors.getLast().expand(Shape.of(2,3,6,28)));
        tensors.add(tensors.getLast().contiguous());tensors.add(tensors.getLast().reshape(Shape.of(2,18,28)));
        tensors.add(tensors.getLast().contiguous());tensors.add(tensors.getLast().fold2d(output,new Window2dAttrs(2,3,1,1,0,1,1,1,false)));
        var nodes=new ArrayList<CompiledNode>();var values=new ArrayList<GraphValue>();
        for(int i=1;i<7;i++)nodes.add(new CompiledNode(new NodeId(i-1),tensors.get(i).provenance().orElseThrow().operation(),List.of(new ValueId(i-1)),List.of(new ValueId(i))));
        var partition=new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,nodes.stream().map(CompiledNode::id).toList());
        var memory=new ArrayList<LogicalMemoryRequirement>();
        for(int i=0;i<7;i++) {
            var descriptor=i==6?new TensorDescriptor(type,output,Optional.of(LayoutDescriptor.contiguous(output)),false):tensors.get(i).descriptor();
            var id=new ValueId(i);values.add(new GraphValue(id,descriptor));
            memory.add(new LogicalMemoryRequirement(id,descriptor,i==0?Optional.empty():Optional.of(partition),i==6?List.of():List.of(partition),i==6));
        }
        return new PrepareContext<>(partition,nodes,values,memory,Map.of(),CpuPartitionAnalysisInputs.DEFAULT);
    }
    private static void write(DataType t,MemorySegment s,int i,double v) {if(t==DataType.FLOAT32)s.setAtIndex(ValueLayout.JAVA_FLOAT,i,(float)v);else s.setAtIndex(ValueLayout.JAVA_DOUBLE,i,v);}
    private static double read(DataType t,MemorySegment s,int i) {return t==DataType.FLOAT32?s.getAtIndex(ValueLayout.JAVA_FLOAT,i):s.getAtIndex(ValueLayout.JAVA_DOUBLE,i);}
}
