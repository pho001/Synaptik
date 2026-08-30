package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool3dLowering;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuPool3dReferenceKernel;
import io.github.pho001.synaptik.model.datatype.*;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.security.MessageDigest;
import java.util.*;
import org.junit.jupiter.api.Test;

class CpuPool3dGeneratedKernelTest {
    @Test
    void productionSelectionEnumeratesExactlyTwentyFourDistinctBodies() throws Throwable {
        Set<String> digests=new LinkedHashSet<>();
        for(DataType type:List.of(DataType.BFLOAT16,DataType.FLOAT32,DataType.FLOAT64))
            for(CpuPool3dIr.Kind kind:CpuPool3dIr.Kind.values())
                for(boolean inputSegment:List.of(false,true))
                    for(boolean outputSegment:List.of(false,true)){
                        CarrierAccess input=inputSegment?CarrierAccess.MEMORY_SEGMENT:arrayAccess(type);
                        CarrierAccess output=outputSegment?CarrierAccess.MEMORY_SEGMENT:arrayAccess(type);
                        Fixture fixture=fixture(type,kind,input,output);
                        digests.add(HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                                .digest(fixture.bytes())));
                        var model=ClassFile.of().parse(fixture.bytes());
                        assertTrue(model.fields().isEmpty());
                        assertEquals(1,model.methods().size());
                    }
        assertEquals(24,digests.size());
    }

    @Test
    void allTwentyFourBodyFamiliesMatchDirectOracle() throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64))
                for (CpuPool3dIr.Kind kind : CpuPool3dIr.Kind.values())
                    for (List<CarrierAccess> pair : List.of(
                            List.of(arrayAccess(type), arrayAccess(type)),
                            List.of(arrayAccess(type), CarrierAccess.MEMORY_SEGMENT),
                            List.of(CarrierAccess.MEMORY_SEGMENT, arrayAccess(type)),
                            List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT))) {
                        Fixture f = fixture(type, kind, pair.get(0), pair.get(1));
                        Object input = carrier(type, pair.get(0), arena, 256);
                        Object actual = carrier(type, pair.get(1), arena, 320);
                        Object expected = carrier(type, pair.get(1), arena, 320);
                        fill(type, input, 256);
                        CpuPool3dReferenceKernel.evaluate(f.geometry(), input, expected, 0,
                                f.geometry().outputCount());
                        long middle = f.geometry().outputCount() / 2;
                        f.handle().invokeWithArguments(input, actual, f.geometry().pack(0, 0),
                                0L, middle);
                        f.handle().invokeWithArguments(input, actual, f.geometry().pack(0, 0),
                                middle, f.geometry().outputCount());
                        assertCarrierEquals(type, expected, actual, kind + " " + pair);
                    }
        }
    }

    @Test
    void generatedClassIsFinalFieldFreeTypedAndHasNoSynaptikHotCall() throws Throwable {
        Fixture fixture = fixture(DataType.FLOAT32, CpuPool3dIr.Kind.AVERAGE,
                CarrierAccess.FLOAT_ARRAY, CarrierAccess.MEMORY_SEGMENT);
        var model = ClassFile.of().parse(fixture.bytes());
        assertTrue(model.flags().has(java.lang.reflect.AccessFlag.FINAL));
        assertTrue(model.fields().isEmpty());
        assertEquals(1, model.methods().stream().filter(method -> method.methodName().stringValue()
                .equals(CpuGeneratorSchema.ENTRY_NAME)).count());
        String refs = java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast)
                .map(entry -> entry.owner().asInternalName()).reduce("", (a, b) -> a + '\n' + b);
        assertFalse(refs.contains("io/github/pho001/synaptik"));
        assertFalse(refs.contains("java/util/"));
        assertFalse(refs.contains("java/lang/reflect"));
    }

    private static Fixture fixture(DataType type, CpuPool3dIr.Kind kind,
            CarrierAccess inputCarrier, CarrierAccess outputCarrier) throws Throwable {
        long[] ie={1,1,3,4,5}, oe={1,1,4,3,5};
        long[] is=inputCarrier==CarrierAccess.MEMORY_SEGMENT
                ?new long[]{256,256,60,13,2}:new long[]{60,60,20,5,1};
        long[] os=outputCarrier==CarrierAccess.MEMORY_SEGMENT
                ?new long[]{320,320,70,17,3}:new long[]{60,60,15,5,1};
        var geometry = new CpuPool3dLowering.Geometry(kind,type,
                new CpuPool3dLowering.Layout(ie,0,is),new CpuPool3dLowering.Layout(oe,0,os),
                2,3,2, 2,2,2, 2,2,2, 2,2,1, 12,60);
        CpuAccessPlan.Regime regime=inputCarrier==CarrierAccess.MEMORY_SEGMENT
                ||outputCarrier==CarrierAccess.MEMORY_SEGMENT
                ?CpuAccessPlan.Regime.GENERAL_ODOMETER:CpuAccessPlan.Regime.DENSE_LINEAR;
        var roles=Collections.nCopies(5,regime==CpuAccessPlan.Regime.DENSE_LINEAR
                ?CpuAccessPlan.AxisRole.CONTIGUOUS:CpuAccessPlan.AxisRole.STRIDED);
        var pool=new CpuPool3dIr(kind,type,CpuPool3dIr.Realization.DIRECT_SCALAR,
                new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,regime,5,roles,
                        regime==CpuAccessPlan.Regime.DENSE_LINEAR?5:0),
                new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,regime,5,roles,
                        regime==CpuAccessPlan.Regime.DENSE_LINEAR?5:0));
        var ir=pool.encodedKernelIr();
        var specialization=new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,List.of(type,type),
                List.of(inputCarrier,outputCarrier),0,-1,List.of(),false,56);
        var generator=new CpuClassFileKernelGenerator();
        byte[] bytes=generator.generateClassBytes(specialization,ir);
        return new Fixture(geometry,generator.defineClassBytes(specialization,bytes).entryPoint(),bytes);
    }

    private static CarrierAccess arrayAccess(DataType type){return switch(type){
        case BFLOAT16->CarrierAccess.SHORT_ARRAY;case FLOAT32->CarrierAccess.FLOAT_ARRAY;
        case FLOAT64->CarrierAccess.DOUBLE_ARRAY;default->throw new AssertionError(type);};}
    private static Object carrier(DataType type,CarrierAccess access,Arena arena,int count){
        if(access==CarrierAccess.MEMORY_SEGMENT)return arena.allocate((long)count*type.byteWidth(),type.byteWidth());
        return switch(type){case BFLOAT16->new short[count];case FLOAT32->new float[count];
            case FLOAT64->new double[count];default->throw new AssertionError(type);};}
    private static void fill(DataType type,Object carrier,int count){for(int i=0;i<count;i++){
        double value=i==7?Double.NaN:i==8?Double.POSITIVE_INFINITY:i==9?Double.NEGATIVE_INFINITY:i%11-5.25;
        long offset=(long)i*type.byteWidth();
        if(carrier instanceof MemorySegment s)switch(type){
            case BFLOAT16->s.set(ValueLayout.JAVA_SHORT_UNALIGNED,offset,BFloat16Bits.fromFloat((float)value));
            case FLOAT32->s.set(ValueLayout.JAVA_FLOAT_UNALIGNED,offset,(float)value);
            case FLOAT64->s.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,offset,value);default->throw new AssertionError();}
        else switch(type){case BFLOAT16->((short[])carrier)[i]=BFloat16Bits.fromFloat((float)value);
            case FLOAT32->((float[])carrier)[i]=(float)value;case FLOAT64->((double[])carrier)[i]=value;
            default->throw new AssertionError();}}}
    private static void assertCarrierEquals(DataType type,Object expected,Object actual,String message){
        if(expected instanceof MemorySegment e){assertEquals(-1,e.mismatch((MemorySegment)actual),message);return;}
        switch(type){case BFLOAT16->assertArrayEquals((short[])expected,(short[])actual,message);
            case FLOAT32->assertArrayEquals(bits((float[])expected),bits((float[])actual),message);
            case FLOAT64->assertArrayEquals(bits((double[])expected),bits((double[])actual),message);
            default->throw new AssertionError();}}
    private static int[] bits(float[] values){int[] r=new int[values.length];for(int i=0;i<r.length;i++)r[i]=Float.floatToRawIntBits(values[i]);return r;}
    private static long[] bits(double[] values){long[] r=new long[values.length];for(int i=0;i<r.length;i++)r[i]=Double.doubleToRawLongBits(values[i]);return r;}
    private record Fixture(CpuPool3dLowering.Geometry geometry,MethodHandle handle,byte[] bytes){}
}
