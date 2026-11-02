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

/** Admitted exact typed stores plus explicit rejection of each low SUM-derived count fixture. */
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
            if (io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest
                    .assertLowUnsupported(c)) continue;
            var route=new CpuPartitionPreparer().analyze(c).plan().units().getFirst().portablePlan();
            var generator=new CpuClassFileKernelGenerator();var bytes=generator.generateClassBytes(route.specialization(),route.kernelIr());
            var instructions=ClassFile.of().parse(bytes).methods().getFirst().code().orElseThrow().elementStream()
                    .filter(Instruction.class::isInstance).map(Instruction.class::cast).toList();
            assertTrue(instructions.stream().noneMatch(InvokeInstruction.class::isInstance));
            assertEquals(3,instructions.stream().filter(i->i.opcode()==Opcode.L2I).count(),"bounds/address narrow only once before induction");
            assertTrue(instructions.stream().noneMatch(i->i.opcode()==Opcode.LADD));
            Object array=switch(type) { case FLOAT32->new float[10];case FLOAT64->new double[10];case INT64->new long[10];
                case INT32->new int[10];default->new short[10]; };
            var storage=segment(array);storage.fill((byte)0x55);
            var artifact=generator.defineClassBytes(route.specialization(), route.kernelIr(), bytes);
            artifact.entryPoint().invokeWithArguments(array,new long[]{4,0,0,1,0,4},0L,4L);
            var expected=MemorySegment.ofArray(new byte[(int)storage.byteSize()]);expected.fill((byte)0x55);
            directStores(type,expected,3,0,1);
            assertEquals(-1,storage.mismatch(expected));
        }
    }

    /**
     * Compares admitted folded executions with exact CPU SUM and witnesses each low-type rejection.
     * @throws Throwable if generated class definition or entry invocation fails
     */
    @Test void generatedCountMatchesTheExistingExactCpuSumAcrossRoundingAndOverflowSites() throws Throwable {
        for(var type:CpuStaticTypedOneCountFoldingTest.types()) for(int n:new int[]{0,1,257,259,2049,2051,65519,65520}) {
            var constant=CpuStaticTypedOneCountFoldingTest.context(type,Shape.of(n),NoOperationAttrs.INSTANCE,Shape.scalar(),0,true);
            var ordinary=new PrepareContext<>(constant.partition(),constant.nodes(),constant.values(),constant.memoryRequirements(),Map.of(),constant.backendInputs());
            boolean foldedUnsupported = io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest
                    .assertLowUnsupported(constant);
            boolean ordinaryUnsupported = io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest
                    .assertLowUnsupported(ordinary);
            assertEquals(foldedUnsupported, ordinaryUnsupported, type + ":" + n);
            if (foldedUnsupported) continue;
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
            if (io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest
                    .assertLowUnsupported(c)) continue;
            var unit=new CpuPartitionPreparer().analyze(c).plan().units().getFirst();
            var route=unit.portablePlan();
            var generator=new CpuClassFileKernelGenerator();byte[] bytes=generator.generateClassBytes(route.specialization(),route.kernelIr());
            assertArrayEquals(bytes,generator.generateClassBytes(route.specialization(),route.kernelIr()));
            inspect(bytes,heap);
            var artifact=generator.defineClassBytes(route.specialization(), route.kernelIr(), bytes);
            Object array=switch(type) {
                case FLOAT32->new float[10];case FLOAT64->new double[10];case INT64->new long[10];
                case INT32->new int[10];default->new short[10];
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

    /** Separates baked count literals while retaining absent/non-one fallback keys and bytes. */
    @Test void literalBitsChangeBothIdentityStreamsButUnprovedControlsRetainKeysAndBytes() {
        var generator=new CpuClassFileKernelGenerator();
        for(var type:CpuStaticTypedOneCountFoldingTest.types()) {
            if (type == DataType.BFLOAT16 || type == DataType.FLOAT16) {
                for (int count : List.of(3, 4))
                    assertThrows(IllegalArgumentException.class, () -> route(type, count, true));
                var absent = CpuStaticTypedOneCountFoldingTest.context(type, Shape.of(3),
                        NoOperationAttrs.INSTANCE, Shape.scalar(), 0, false);
                var nonOne = new PrepareContext<>(absent.partition(), absent.nodes(), absent.values(),
                        absent.memoryRequirements(), Map.of(new ValueId(0), two(type)), absent.backendInputs());
                assertTrue(io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest
                        .assertLowUnsupported(absent));
                assertTrue(io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest
                        .assertLowUnsupported(nonOne));
                continue;
            }
            var a=route(type,3,true);var b=route(type,4,true);
            assertNotEquals(a.specialization().structuralKey(),b.specialization().structuralKey());
            assertFalse(Arrays.equals(a.specialization().compatibilityBytes(),b.specialization().compatibilityBytes()));
            assertFalse(Arrays.equals(a.specialization().classIdentityBytes(),b.specialization().classIdentityBytes()));
            assertFalse(Arrays.equals(generator.generateClassBytes(a.specialization(),a.kernelIr()),generator.generateClassBytes(b.specialization(),b.kernelIr())));
            var c=CpuStaticTypedOneCountFoldingTest.context(type,Shape.of(3),NoOperationAttrs.INSTANCE,Shape.scalar(),0,false);
            var nonOne=new PrepareContext<>(c.partition(),c.nodes(),c.values(),c.memoryRequirements(),Map.of(new ValueId(0),two(type)),c.backendInputs());
            var absent=new CpuPartitionPreparer().analyze(c).plan().units().getFirst().portablePlan();
            var unchanged=new CpuPartitionPreparer().analyze(nonOne).plan().units().getFirst().portablePlan();
            assertArrayEquals(absent.specialization().compatibilityBytes(),unchanged.specialization().compatibilityBytes());
            assertArrayEquals(absent.specialization().classIdentityBytes(),unchanged.specialization().classIdentityBytes());
            assertArrayEquals(generator.generateClassBytes(absent.specialization(),absent.kernelIr()),generator.generateClassBytes(unchanged.specialization(),unchanged.kernelIr()));
        }
    }

    /**
     * Inspects actual ClassFile instructions for the one-output, store-only entry contract.
     * @param bytes non-null generated class bytes, not mutated
     * @param heap whether the entry uses a primitive array rather than a segment
     */
    private static void inspect(byte[] bytes,boolean heap) {
        var model=ClassFile.of().parse(bytes);assertEquals(1,model.methods().size());
        var method=model.methods().getFirst();
        assertTrue(method.methodType().stringValue().endsWith("[JJJ)V"));
        var instructions=method.code().orElseThrow().elementStream().filter(Instruction.class::isInstance).map(Instruction.class::cast).toList();
        for(var instruction:instructions) {
            assertFalse(Set.of(Opcode.NEW,Opcode.NEWARRAY,Opcode.ANEWARRAY,Opcode.INVOKEDYNAMIC,Opcode.MULTIANEWARRAY).contains(instruction.opcode()));
            assertFalse(Set.of(Opcode.FALOAD,Opcode.DALOAD,Opcode.SALOAD,Opcode.IALOAD).contains(instruction.opcode()));
            if(instruction instanceof InvokeInstruction invoke) {
                assertFalse(heap,"heap constant-store loop needs no method calls");
                assertEquals("set",invoke.name().stringValue());
                assertEquals("java/lang/foreign/MemorySegment",invoke.owner().asInternalName());
            }
        }
        assertEquals(1,instructions.stream().filter(i->Set.of(Opcode.GOTO,Opcode.GOTO_W).contains(i.opcode())).count(),"only output induction, never a contributor loop");
    }
    /**
     * Analyzes a direct full-SUM fixture for artifact-identity comparisons.
     * @param type non-null numeric count type, including explicit rejection fixtures
     * @param count nonnegative static source cardinality
     * @param explicit whether to project the immutable +1 fact
     * @return non-null first unit's selected portable route
     * @throws IllegalArgumentException if the source SUM uses a currently unsupported low type
     */
    private static io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route(DataType type,int count,boolean explicit) {
        return new CpuPartitionPreparer().analyze(CpuStaticTypedOneCountFoldingTest.context(type,Shape.of(count),NoOperationAttrs.INSTANCE,Shape.scalar(),0,explicit)).plan().units().getFirst().portablePlan();
    }
    /**
     * Chooses the matching primitive heap form without changing represented low bits.
     * @param type non-null supported numeric count type
     * @return non-null matching array carrier
     */
    private static CarrierAccess arrayCarrier(DataType type) { return switch(type) {
        case FLOAT32->CarrierAccess.FLOAT_ARRAY;case FLOAT64->CarrierAccess.DOUBLE_ARRAY;case INT32->CarrierAccess.INT_ARRAY;
        case INT64->CarrierAccess.LONG_ARRAY;default->CarrierAccess.SHORT_ARRAY;
    }; }
    /**
     * Views a test primitive array as bytes for exact comparison; ownership stays with the test.
     * @param array non-null float, double, int, long or short array, not copied
     * @return non-null segment over the same array storage
     * @throws AssertionError if {@code array} has an unsupported type
     */
    private static MemorySegment segment(Object array) { return switch(array) {
        case float[] a->MemorySegment.ofArray(a);case double[] a->MemorySegment.ofArray(a);case int[] a->MemorySegment.ofArray(a);
        case long[] a->MemorySegment.ofArray(a);case short[] a->MemorySegment.ofArray(a);default->throw new AssertionError();
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
        default->throw new AssertionError();
    } }
    /**
     * Creates an exact non-one fact to exercise the unchanged ordinary reduction fallback.
     * @param type non-null supported numeric type
     * @return non-null immutable typed value two
     */
    private static ScalarValue two(DataType type) { return switch(type) {
        case FLOAT32->ScalarValue.float32(2);case FLOAT64->ScalarValue.float64(2);case INT32->ScalarValue.int32(2);case INT64->ScalarValue.int64(2);
        case BFLOAT16->ScalarValue.bfloat16(2);case FLOAT16->ScalarValue.float16(2);default->throw new AssertionError();
    }; }
}
