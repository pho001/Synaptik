package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.*;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.*;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.*;
import io.github.pho001.synaptik.prepare.analysis.*;
import io.github.pho001.synaptik.runtime.memory.*;
import io.github.pho001.synaptik.runtime.run.*;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBorrowedBuffer;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** CPU physical conformance for exact source facts, output-only units and mutable fallback. */
class CpuStaticTypedOneCountConformanceTest {
    /** Executes retained count writes and explicitly rejects both low SUM-derived count families. */
    @Test void staticExplicitCountWritesOnlyItsPublicResultAndRetainsTheOriginalSource() {
        for(var type:List.of(DataType.FLOAT32,DataType.FLOAT64,DataType.BFLOAT16,DataType.FLOAT16,DataType.INT32,DataType.INT64)) {
            var context=context(type,true);
            if (CpuProfileFreeConformanceTest.assertLockedNumerical(context)) continue;
            var analysis=new CpuPartitionPreparer().analyze(context);
            var plan=analysis.plan();var unit=plan.units().getFirst();
            assertEquals(1,plan.units().size());assertEquals(List.of(0,1,2),unit.memberNodeOrdinals());
            assertInstanceOf(CpuConstantWriteIr.class,unit.portablePlan().portableKernelIr());
            assertEquals(List.of(new ValueId(3)),unit.boundaryValues());
            assertEquals(Set.of(new ValueId(0),new ValueId(3)),new HashSet<>(plan.boundaryValues()));
            assertTrue(plan.workspaceDeclaration().isEmpty());assertFalse(unit.portablePlan().specialization().scratchParameter());
            var buffers=new ArrayList<PreparedMemoryPlan.BufferEntry>();var assignments=new ArrayList<PreparationResourceAssignment>();
            for(var declaration:plan.bufferDeclarations()) {
                var slot=new BufferSlot(buffers.size());buffers.add(new PreparedMemoryPlan.BufferEntry(slot,declaration.byteSize(),declaration.byteAlignment()));
                assignments.add(new PreparationResourceAssignment.Buffer(declaration,slot,buffers.size()-1,0));
            }
            var memory=new PreparedMemoryPlan(buffers,List.of());
            var executable=new CpuPartitionFinalizer().finalizePartition(new BackendPartitionFinalization<>(analysis,memory,assignments)).executable();
            try(var arena=Arena.ofConfined()) {
                var representations=new ArrayList<List<BufferRepresentationBinding>>();var borrowed=new ArrayList<CpuBorrowedBuffer>();
                MemorySegment output=null;
                for(var declaration:plan.bufferDeclarations()) {
                    var segment=arena.allocate(declaration.byteSize(),declaration.byteAlignment());
                    // This physical test deliberately poisons the unused retained source;
                    // GraphPreparation owns normal initialization from the explicit fact.
                    segment.fill((byte)0x55);
                    if(declaration.valueId().equals(new ValueId(3))) output=segment;
                    var representation=CpuBorrowedBuffer.borrow(new MemorySegmentStorage(type,declaration.byteSize()/type.byteWidth(),segment));
                    borrowed.add(representation);representations.add(List.of(new BufferRepresentationBinding(representation,RunResourceOwnership.BORROWED)));
                }
                try {
                    executable.bind(new RunState(memory,representations,List.of())).execute();
                    assertNotNull(output);
                    for(int index=0;index<2;index++) assertEquals(3,read(type,output,index));
                } finally { borrowed.forEach(CpuBorrowedBuffer::close); }
            }
        }
    }

    /** Keeps the ordinary materialization and reduction path when no immutable fact is supplied. */
    @Test void absentFactsKeepTheOrdinaryMaterializationAndReductionTopology() {
        var plan=new CpuPartitionPreparer().analyze(context(DataType.FLOAT32,false)).plan();
        assertTrue(plan.units().stream().noneMatch(u->u.portablePlan().portableKernelIr() instanceof CpuConstantWriteIr));
        assertTrue(plan.boundaryValues().contains(new ValueId(2)));
        assertTrue(plan.units().stream().anyMatch(u->u.portablePlan().portableKernelIr() instanceof CpuAggregateIr));
    }

    /**
     * Builds an EXPAND/CONTIGUOUS/axis-SUM fixture with a public two-cell result.
     * @param type non-null supported numeric count type
     * @param explicit whether to project the exact immutable source-one fact
     * @return non-null immutable static CPU projection without physical allocation
     */
    private static PrepareContext<CpuPartitionAnalysisInputs> context(DataType type,boolean explicit) {
        Shape domain=Shape.of(2,3),output=Shape.of(2);
        var descriptors=List.of(descriptor(type,Shape.scalar(),LayoutDescriptor.contiguous(Shape.scalar())),
                descriptor(type,domain,LayoutDescriptor.of(domain,new long[]{0,0},0,true)),
                descriptor(type,domain,LayoutDescriptor.contiguous(domain)),descriptor(type,output,LayoutDescriptor.contiguous(output)));
        var nodes=List.of(new CompiledNode(new NodeId(0),new Operation(ShapeTransformKind.EXPAND,new TargetShapeAttrs(domain)),List.of(new ValueId(0)),List.of(new ValueId(1))),
                new CompiledNode(new NodeId(1),new Operation(ContiguousKind.CONTIGUOUS,NoOperationAttrs.INSTANCE),List.of(new ValueId(1)),List.of(new ValueId(2))),
                new CompiledNode(new NodeId(2),new Operation(AggregateReductionKind.SUM,new AxisReductionAttrs(1,false)),List.of(new ValueId(2)),List.of(new ValueId(3))));
        var partition=new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,nodes.stream().map(CompiledNode::id).toList());
        var values=new ArrayList<GraphValue>();var memory=new ArrayList<LogicalMemoryRequirement>();
        for(int i=0;i<4;i++) {
            var id=new ValueId(i);var descriptor=descriptors.get(i);values.add(new GraphValue(id,descriptor));
            memory.add(new LogicalMemoryRequirement(id,descriptor,i==0?Optional.empty():Optional.of(partition),i==3?List.of():List.of(partition),i==3));
        }
        var one=switch(type) { case FLOAT32->ScalarValue.float32(1);case FLOAT64->ScalarValue.float64(1);case BFLOAT16->ScalarValue.bfloat16(1);
            case FLOAT16->ScalarValue.float16(1);case INT32->ScalarValue.int32(1);case INT64->ScalarValue.int64(1);default->throw new AssertionError(); };
        return new PrepareContext<>(partition,nodes,values,memory,explicit?Map.of(new ValueId(0),one):Map.of(),CpuPartitionAnalysisInputs.DEFAULT);
    }
    /**
     * Creates one non-gradient descriptor with its exact resolved layout.
     * @param type non-null represented type
     * @param shape non-null fully static logical shape
     * @param layout non-null resolved layout for {@code shape}
     * @return non-null immutable descriptor with no storage association
     */
    private static TensorDescriptor descriptor(DataType type,Shape shape,LayoutDescriptor layout) { return new TensorDescriptor(type,shape,Optional.of(layout),false); }
    /**
     * Reads the small exact count result for this fixture, not a large-count conversion oracle.
     * @param type non-null supported numeric type
     * @param segment non-null live segment covering the requested represented cell
     * @param index nonnegative element index, not a byte offset
     * @return decoded cell value, exactly representable as double for this fixture
     */
    private static double read(DataType type,MemorySegment segment,int index) { return switch(type) {
        case FLOAT32->segment.getAtIndex(ValueLayout.JAVA_FLOAT,index);case FLOAT64->segment.getAtIndex(ValueLayout.JAVA_DOUBLE,index);
        case BFLOAT16->BFloat16Bits.toFloat(segment.getAtIndex(ValueLayout.JAVA_SHORT,index));case FLOAT16->Float16Bits.toFloat(segment.getAtIndex(ValueLayout.JAVA_SHORT,index));
        case INT32->segment.getAtIndex(ValueLayout.JAVA_INT,index);case INT64->segment.getAtIndex(ValueLayout.JAVA_LONG,index);default->throw new AssertionError();
    }; }
}
