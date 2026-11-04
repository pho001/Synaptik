package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuStaticTypedOneCountFoldingTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture;
import io.github.pho001.synaptik.model.datatype.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.foreign.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Exact typed folded-count stores and their ordinary CPU SUM numerical boundary. */
class CpuConstantWriteGeneratedKernelTest {
    /**
     * Checks direct heap stores against Java with one-time narrowing and no runtime method calls.
     * @throws Throwable if generated class definition or entry invocation fails
     */
    @Test void denseHeapStoresUseOnlyIntegerInductionAndNoRuntimeCalls() throws Throwable {
        for(var type:CpuStaticTypedOneCountFoldingTest.types()) {
            var c=CpuStaticTypedOneCountFoldingTest.context(type,Shape.of(4,3),new AxisReductionAttrs(1,false),Shape.of(4),0,true);
            c=new PrepareContext<>(c.partition(),c.nodes(),c.values(),c.memoryRequirements(),c.constants(),
                    new CpuPartitionAnalysisInputs(false,List.of(arrayCarrier(type),CarrierAccess.MEMORY_SEGMENT)));
            
            var route=new CpuPartitionPreparer().analyze(c).plan().units().getFirst().portablePlan();
            var generator=new CpuClassFileKernelGenerator();var bytes=generator.generateClassBytes(route.specialization(),route.kernelIr());
            Object array=switch(type) { case FLOAT32->new float[10];case FLOAT64->new double[10];case INT64->new long[10];
                case INT32->new int[10];case FLOAT8_E4M3,FLOAT8_E5M2->new byte[10];default->new short[10]; };
            var storage=segment(array);storage.fill((byte)0x55);
            var artifact=generator.defineClassBytes(route.specialization(), route.kernelIr(), bytes);
            artifact.entryPoint().invokeWithArguments(array,new long[]{4,0,0,1,0,4},0L,4L);
            var expected=MemorySegment.ofArray(new byte[(int)storage.byteSize()]);expected.fill((byte)0x55);
            directStores(type,expected,3,0,1);
            assertEquals(-1,storage.mismatch(expected));
        }
    }

    /**
     * Compares folded executions with exact CPU SUM at represented output boundaries.
     * @throws Throwable if generated class definition or entry invocation fails
     */
    @Test void generatedCountMatchesTheExistingExactCpuSumAcrossRoundingAndOverflowSites() throws Throwable {
        for(var type:CpuStaticTypedOneCountFoldingTest.types()) for(int n:new int[]{0,1,257,259,2049,2051,65519,65520}) {
            var constant=CpuStaticTypedOneCountFoldingTest.context(type,Shape.of(n),NoOperationAttrs.INSTANCE,Shape.scalar(),0,true);
            var ordinary=new PrepareContext<>(constant.partition(),constant.nodes(),constant.values(),constant.memoryRequirements(),Map.of(),constant.backendInputs());
            var folded=new CpuPartitionPreparer().analyze(constant);
            var fallback=new CpuPartitionPreparer().analyze(ordinary);
            try(var arena=Arena.ofConfined();var expected=new Fixture(fallback,1,false,false,Map.of())) {
                for(int i=0;i<n;i++) putOne(type,expected.segment(0),i);
                var route=folded.plan().units().getFirst().portablePlan();var generator=new CpuClassFileKernelGenerator();
                var artifact=generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
                var actual=arena.allocate(type.byteWidth(),type.byteWidth());
                artifact.entryPoint().invokeWithArguments(actual,new long[]{0,0,1},0L,1L);
                expected.executable.bind(expected.state).execute();
                assertEquals(-1,actual.mismatch(expected.segment(1)),type+":"+n);
            }
        }
    }

    /**
     * Checks the one-output entry and unchanged surrounding bytes for strided, partial writes.
     * @throws Throwable if generated class definition or entry invocation fails
     */
    @Test void oneOutputAbiStoresExactBitsToHeapOrSegmentWithStridesOffsetsAndSubranges() throws Throwable {
        for(var type:CpuStaticTypedOneCountFoldingTest.types()) for(boolean heap:List.of(false,true)) {
            var base=CpuStaticTypedOneCountFoldingTest.context(type,Shape.of(4,3),new AxisReductionAttrs(1,false),Shape.of(4),0,true);
            var descriptor=new TensorDescriptor(type,Shape.of(4),Optional.of(LayoutDescriptor.of(Shape.of(4),new long[]{2},1,true)),false);
            var values=List.of(base.values().getFirst(),new GraphValue(new ValueId(1),descriptor));
            var memory=List.of(base.memoryRequirements().getFirst(),new LogicalMemoryRequirement(new ValueId(1),descriptor,Optional.of(base.partition()),List.of(),true));
            var carrier=heap?arrayCarrier(type):CarrierAccess.MEMORY_SEGMENT;
            var c=new PrepareContext<>(base.partition(),base.nodes(),values,memory,base.constants(),new CpuPartitionAnalysisInputs(false,List.of(carrier,carrier)));
            
            var unit=new CpuPartitionPreparer().analyze(c).plan().units().getFirst();
            var route=unit.portablePlan();
            var generator=new CpuClassFileKernelGenerator();byte[] bytes=generator.generateClassBytes(route.specialization(),route.kernelIr());
            assertArrayEquals(bytes,generator.generateClassBytes(route.specialization(),route.kernelIr()));
            var artifact=generator.defineClassBytes(route.specialization(), route.kernelIr(), bytes);
            Object array=switch(type) {
                case FLOAT32->new float[10];case FLOAT64->new double[10];case INT64->new long[10];
                case INT32->new int[10];case FLOAT8_E4M3,FLOAT8_E5M2->new byte[10];default->new short[10];
            };
            MemorySegment storage=segment(array);storage.fill((byte)0x55);
            Object output=heap?array:storage;
            for(long[] range:List.of(new long[]{0,1},new long[]{1,1},new long[]{1,4})) {
                long start=range[0],end=range[1];
                long[] geometry={4,start,1+2*start,2,0,1};
                artifact.entryPoint().invokeWithArguments(output,geometry,start,end);
            }
            MemorySegment expected=MemorySegment.ofArray(new byte[(int)storage.byteSize()]);expected.fill((byte)0x55);
            // Optimal specialized direct Java oracle: only output stores, no source traversal.
            directStores(type,expected,3,1,2);
            assertEquals(-1,storage.mismatch(expected),type+":"+heap);
        }
    }

    /**
     * Chooses the matching primitive heap form without changing represented low bits.
     * @param type non-null supported numeric count type
     * @return non-null matching array carrier
     */
    private static CarrierAccess arrayCarrier(DataType type) { return switch(type) {
        case FLOAT32->CarrierAccess.FLOAT_ARRAY;case FLOAT64->CarrierAccess.DOUBLE_ARRAY;case INT32->CarrierAccess.INT_ARRAY;
        case INT64->CarrierAccess.LONG_ARRAY;case FLOAT8_E4M3,FLOAT8_E5M2->CarrierAccess.BYTE_ARRAY;default->CarrierAccess.SHORT_ARRAY;
    }; }
    /**
     * Views a test primitive array as bytes for exact comparison; ownership stays with the test.
     * @param array non-null float, double, int, long or short array, not copied
     * @return non-null segment over the same array storage
     * @throws AssertionError if {@code array} has an unsupported type
     */
    private static MemorySegment segment(Object array) { return switch(array) {
        case float[] a->MemorySegment.ofArray(a);case double[] a->MemorySegment.ofArray(a);case int[] a->MemorySegment.ofArray(a);
        case long[] a->MemorySegment.ofArray(a);case short[] a->MemorySegment.ofArray(a);case byte[] a->MemorySegment.ofArray(a);default->throw new AssertionError();
    }; }
    /**
     * Initializes one represented +1 contributor for the existing exact-SUM reference route.
     * @param type non-null supported numeric type
     * @param s non-null live writable segment with a cell of the requested type at {@code i}
     * @param i nonnegative element index, not a byte offset
     */
    private static void putOne(DataType type,MemorySegment s,int i) { switch(type) {
        case FLOAT32->s.setAtIndex(ValueLayout.JAVA_FLOAT,i,1);case FLOAT64->s.setAtIndex(ValueLayout.JAVA_DOUBLE,i,1);
        case INT32->s.setAtIndex(ValueLayout.JAVA_INT,i,1);case INT64->s.setAtIndex(ValueLayout.JAVA_LONG,i,1);
        case BFLOAT16->s.setAtIndex(ValueLayout.JAVA_SHORT,i,(short)0x3f80);case FLOAT16->s.setAtIndex(ValueLayout.JAVA_SHORT,i,(short)0x3c00);
        case FLOAT8_E4M3->s.setAtIndex(ValueLayout.JAVA_BYTE,i,(byte)0x38);
        case FLOAT8_E5M2->s.setAtIndex(ValueLayout.JAVA_BYTE,i,(byte)0x3c);
        default->throw new AssertionError();
    } }
    /**
     * Stores four cells using one cold type selection and literal, then only direct Java stores.
     * The small literal is exactly representable; large-count rounding is tested separately.
     * @param type non-null supported numeric type
     * @param s non-null live writable segment covering all four selected cells
     * @param count nonnegative small integer literal, not the number of output cells
     * @param base nonnegative first element address
     * @param stride nonnegative address increment in elements
     */
    private static void directStores(DataType type,MemorySegment s,int count,int base,int stride) { switch(type) {
        case FLOAT32->{ float literal=count;for(int i=0,address=base;i<4;i++,address+=stride) s.setAtIndex(ValueLayout.JAVA_FLOAT_UNALIGNED,address,literal); }
        case FLOAT64->{ double literal=count;for(int i=0,address=base;i<4;i++,address+=stride) s.setAtIndex(ValueLayout.JAVA_DOUBLE_UNALIGNED,address,literal); }
        case INT32->{ for(int i=0,address=base;i<4;i++,address+=stride) s.setAtIndex(ValueLayout.JAVA_INT_UNALIGNED,address,count); }
        case INT64->{ long literal=count;for(int i=0,address=base;i<4;i++,address+=stride) s.setAtIndex(ValueLayout.JAVA_LONG_UNALIGNED,address,literal); }
        case BFLOAT16->{ short literal=BFloat16Bits.fromFloat(count);for(int i=0,address=base;i<4;i++,address+=stride) s.setAtIndex(ValueLayout.JAVA_SHORT_UNALIGNED,address,literal); }
        case FLOAT16->{ short literal=Float16Bits.fromFloat(count);for(int i=0,address=base;i<4;i++,address+=stride) s.setAtIndex(ValueLayout.JAVA_SHORT_UNALIGNED,address,literal); }
        case FLOAT8_E4M3->{ byte literal=io.github.pho001.synaptik.model.datatype.Float8E4M3Bits.fromFloat(count);for(int i=0,address=base;i<4;i++,address+=stride) s.setAtIndex(ValueLayout.JAVA_BYTE,address,literal); }
        case FLOAT8_E5M2->{ byte literal=io.github.pho001.synaptik.model.datatype.Float8E5M2Bits.fromFloat(count);for(int i=0,address=base;i<4;i++,address+=stride) s.setAtIndex(ValueLayout.JAVA_BYTE,address,literal); }
        default->throw new AssertionError();
    } }
}
