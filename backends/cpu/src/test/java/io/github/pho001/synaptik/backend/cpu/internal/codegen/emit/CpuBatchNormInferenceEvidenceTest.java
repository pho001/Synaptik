package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormInferenceLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Frozen CPU 0007F1 generated/direct evidence probe and isolated-fork driver. */
public final class CpuBatchNormInferenceEvidenceTest {
    private static final long MIN_BATCH_NANOS = 25_000_000L;
    private static volatile long sink;
    private static final ValueLayout.OfShort SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfFloat FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfDouble DOUBLE = ValueLayout.JAVA_DOUBLE_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());
    @FunctionalInterface private interface Action { long run() throws Throwable; }
    private record Case(String name, Action generated, Action direct, Runnable verify,
            long roots, long reads, long[] generatedRounds, long[] directRounds) { }
    private record Prepared(MethodHandle handle, long[] geometry,
            CpuBatchNormInferenceIr.RangeForm form, long items, byte[] bytes) { }

    @Test void frozenInventoryAndOperationBoundsAreExact() throws Throwable {
        List<Case> cases = cases();
        assertEquals(List.of("BN-BF16-A1", "BN-F32-A1", "BN-F64-A1", "BN-F32-A0",
                "BN-F32-A2", "BN-MIX-F64", "BN-MIX-F32", "BN-REPEAT-C1"),
                cases.stream().map(Case::name).toList());
        for (Case value : cases) {
            value.generated.run(); value.direct.run(); value.verify.run();
            assertTrue(value.roots <= value.reads);
        }
    }

    @Test void retainedIsolatedFork() throws Throwable {
        String fork = System.getProperty("synaptik.cpu.batchnorm.fork");
        if (fork == null) fork = System.getenv("SYNAPTIK_CPU_BATCHNORM_FORK");
        org.junit.jupiter.api.Assumptions.assumeTrue(fork != null);
        main(new String[] {fork});
    }

    /** Runs one deterministic isolated five-warmup/nine-measurement fork. */
    public static void main(String[] args) throws Throwable {
        long fork = args.length == 0 ? 0 : Long.parseLong(args[0]);
        List<Case> cases = cases();
        int[] repetitions = new int[cases.size() * 2];
        for (int i = 0; i < cases.size(); i++) {
            cases.get(i).generated.run(); cases.get(i).direct.run(); cases.get(i).verify.run();
            repetitions[2 * i] = repetitions(cases.get(i).generated);
            repetitions[2 * i + 1] = repetitions(cases.get(i).direct);
        }
        Random random = new Random(0x0007f120260825L ^ fork * 0x9e3779b97f4a7c15L);
        for (int round = -10; round < 9; round++) {
            List<Integer> order = new ArrayList<>();
            for (int i = 0; i < cases.size(); i++) order.add(i);
            Collections.shuffle(order, random);
            for (int i : order) {
                Case value = cases.get(i); long generated, direct;
                if (random.nextBoolean()) {
                    generated = time(value.generated, repetitions[2 * i]);
                    direct = time(value.direct, repetitions[2 * i + 1]);
                } else {
                    direct = time(value.direct, repetitions[2 * i + 1]);
                    generated = time(value.generated, repetitions[2 * i]);
                }
                if (round >= 0) {
                    value.generatedRounds[round] = generated / repetitions[2 * i];
                    value.directRounds[round] = direct / repetitions[2 * i + 1];
                    value.verify.run();
                }
            }
        }
        int failures = 0; StringBuilder summary = new StringBuilder();
        for (Case value : cases) {
            long generated = median(value.generatedRounds), direct = median(value.directRounds);
            double ratio = (double) generated / direct;
            if (ratio > 1.15) failures++;
            System.out.printf(Locale.ROOT, "RESULT,%s,%d,%d,%.9f,roots=%d,reads=%d,%s,%s%n",
                    value.name, generated, direct, ratio, value.roots, value.reads,
                    Arrays.toString(value.generatedRounds), Arrays.toString(value.directRounds));
            summary.append(String.format(Locale.ROOT,
                    "RESULT,%s,%d,%d,%.9f,roots=%d,reads=%d,%s,%s%n", value.name,
                    generated, direct, ratio, value.roots, value.reads,
                    Arrays.toString(value.generatedRounds), Arrays.toString(value.directRounds)));
        }
        System.out.println("SINK," + sink);
        Path evidence = Path.of("/private/tmp/synaptik-cpu-0007f1-retained-evidence-20260825");
        Files.createDirectories(evidence.resolve("forks"));
        Files.writeString(evidence.resolve("forks/fork-" + fork + ".csv"), summary.toString());
        if (failures != 0) throw new AssertionError("ratio failures " + failures);
    }

    private static List<Case> cases() throws Throwable {
        List<Case> result = new ArrayList<>();
        result.add(dense("BN-BF16-A1", DataType.BFLOAT16, 32, 64, 256, 1));
        result.add(dense("BN-F32-A1", DataType.FLOAT32, 32, 64, 256, 1));
        result.add(dense("BN-F64-A1", DataType.FLOAT64, 32, 64, 256, 1));
        result.add(dense("BN-F32-A0", DataType.FLOAT32, 4096, 128, 1, 0));
        result.add(dense("BN-F32-A2", DataType.FLOAT32, 32, 256, 64, 2));
        result.add(mixedDense("BN-MIX-F64", true));
        result.add(mixedDense("BN-MIX-F32", false));
        result.add(repeated());
        return result;
    }

    private static Case dense(String name, DataType type, int a, int b, int c, int axis)
            throws Throwable {
        long[] shape = c == 1 && axis == 0 ? new long[] {a, b} : new long[] {a, b, c};
        Shape modelShape = Shape.of(shape); int channels = Math.toIntExact(shape[axis]);
        int elements = Math.toIntExact(Arrays.stream(shape).reduce(1, Math::multiplyExact));
        Prepared prepared = prepare(name, Collections.nCopies(5, type), modelShape, axis,
                List.of(0, 1, 2, 3, 4), Collections.nCopies(6, array(type)));
        Object input = array(type, elements), scale = array(type, channels), bias = array(type, channels),
                mean = array(type, channels), variance = array(type, channels),
                generatedOutput = array(type, elements), directOutput = array(type, elements);
        fill(type, input, 17); fill(type, scale, 3); fill(type, bias, 7); fill(type, mean, 11);
        fillPositive(type, variance);
        Action generated = () -> invoke(type, prepared, input, scale, bias, mean, variance,
                generatedOutput);
        Action direct = () -> direct(type, prepared, shape, axis, input, scale, bias, mean,
                variance, directOutput);
        long roots = prepared.form == CpuBatchNormInferenceIr.RangeForm.CHANNEL_RANGE
                ? channels : (long) channels * 4;
        long channelsRead = roots * 4;
        return new Case(name, generated, direct, () -> equal(type, generatedOutput, directOutput),
                roots, channelsRead, new long[9], new long[9]);
    }

    private static Case mixedDense(String name, boolean f64) throws Throwable {
        List<DataType> types = f64 ? List.of(DataType.FLOAT32, DataType.FLOAT64,
                DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)
                : List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.BFLOAT16,
                        DataType.FLOAT32, DataType.BFLOAT16);
        DataType result = f64 ? DataType.FLOAT64 : DataType.FLOAT32;
        Shape shape = Shape.of(16, 32, 64), vector = Shape.of(32);
        long[] inputStrides = f64 ? new long[] {5000, 137, 2} : new long[] {5100, 139, 2};
        long inputOffset = f64 ? 11 : 9, outputOffset = f64 ? 13 : 17;
        long[] outputStrides = f64 ? new long[] {6000, 151, 2} : new long[] {6100, 157, 2};
        long[] vectorOffsets = f64 ? new long[] {3, 5, 7, 11} : new long[] {2, 4, 6, 10};
        long[] vectorStrides = f64 ? new long[] {2, 0, 3, 1} : new long[] {2, 0, 3, 1};
        var layouts = List.of(LayoutDescriptor.of(shape, inputStrides, inputOffset, true),
                LayoutDescriptor.of(vector, new long[] {vectorStrides[0]}, vectorOffsets[0], true),
                LayoutDescriptor.of(vector, new long[] {vectorStrides[1]}, vectorOffsets[1], true),
                LayoutDescriptor.of(vector, new long[] {vectorStrides[2]}, vectorOffsets[2], true),
                LayoutDescriptor.of(vector, new long[] {vectorStrides[3]}, vectorOffsets[3], true),
                LayoutDescriptor.of(shape, outputStrides, outputOffset, true));
        List<CarrierAccess> carriers = f64 ? List.of(CarrierAccess.FLOAT_ARRAY,
                CarrierAccess.MEMORY_SEGMENT, CarrierAccess.SHORT_ARRAY,
                CarrierAccess.MEMORY_SEGMENT, CarrierAccess.DOUBLE_ARRAY,
                CarrierAccess.MEMORY_SEGMENT) : List.of(CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.FLOAT_ARRAY, CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.FLOAT_ARRAY, CarrierAccess.SHORT_ARRAY, CarrierAccess.FLOAT_ARRAY);
        Prepared prepared = prepare(name, types, shape, 1, List.of(0, 1, 2, 3, 4), carriers,
                layouts);
        int inputSpan = Math.toIntExact(layouts.getFirst().referencedElementSpan());
        int outputSpan = Math.toIntExact(layouts.getLast().referencedElementSpan());
        Object input = f64 ? new float[inputSpan] : Arena.global().allocate(inputSpan * 2L, 2);
        Object scale = f64 ? Arena.global().allocate(layouts.get(1).referencedElementSpan()*8L,8)
                : new float[Math.toIntExact(layouts.get(1).referencedElementSpan())];
        Object bias = f64 ? new short[Math.toIntExact(layouts.get(2).referencedElementSpan())]
                : Arena.global().allocate(layouts.get(2).referencedElementSpan()*2L,2);
        Object mean = f64 ? Arena.global().allocate(layouts.get(3).referencedElementSpan()*4L,4)
                : new float[Math.toIntExact(layouts.get(3).referencedElementSpan())];
        Object variance = f64 ? new double[Math.toIntExact(layouts.get(4).referencedElementSpan())]
                : new short[Math.toIntExact(layouts.get(4).referencedElementSpan())];
        Object generatedOutput = f64 ? Arena.global().allocate(outputSpan*8L,8)
                : new float[outputSpan];
        Object directOutput = f64 ? Arena.global().allocate(outputSpan*8L,8)
                : new float[outputSpan];
        initializeMixed(f64,input,scale,bias,mean,variance,inputOffset,inputStrides,vectorOffsets,
                vectorStrides);
        Action generated=f64
                ?()->invokeMixed64(prepared,(float[])input,(MemorySegment)scale,(short[])bias,
                        (MemorySegment)mean,(double[])variance,(MemorySegment)generatedOutput)
                :()->invokeMixed32(prepared,(MemorySegment)input,(float[])scale,
                        (MemorySegment)bias,(float[])mean,(short[])variance,
                        (float[])generatedOutput);
        Action direct=()->directMixed(f64,prepared,input,scale,bias,mean,variance,directOutput,
                inputOffset,inputStrides,vectorOffsets,vectorStrides,outputOffset,outputStrides);
        Runnable verify=()->equalMixed(f64,generatedOutput,directOutput,outputSpan);
        return new Case(name,generated,direct,verify,128,512,new long[9],new long[9]);
    }

    private static Case repeated() throws Throwable {
        Shape shape=Shape.of(65536,1); Prepared prepared=prepare("BN-REPEAT-C1",
                Collections.nCopies(5,DataType.FLOAT32),shape,1,List.of(0,1,1,1,1),
                List.of(CarrierAccess.FLOAT_ARRAY,CarrierAccess.FLOAT_ARRAY,CarrierAccess.FLOAT_ARRAY));
        float[] input=new float[65536], shared={2}, generated=new float[65536], direct=new float[65536];
        fill(DataType.FLOAT32,input,17);
        Action g=()->{ long value=0; for(long[] r:ranges(prepared.items)) {
            prepared.handle.invokeExact(input,shared,generated,prepared.geometry,r[0],r[1]); } return checksum(DataType.FLOAT32,generated);};
        Action d=()->{ for(long[] r:ranges(prepared.items)) for(long q=r[0];q<r[1];q++) {
            float denominator=(float)Math.sqrt(shared[0]+1e-5f); float centered=input[(int)q]-shared[0];
            float standardized=centered/denominator; float scaled=standardized*shared[0];
            direct[(int)q]=scaled+shared[0]; } return checksum(DataType.FLOAT32,direct);};
        return new Case("BN-REPEAT-C1",g,d,()->assertArrayEquals(generated,direct),4,16,
                new long[9],new long[9]);
    }

    private static Prepared prepare(String name,List<DataType> types,Shape shape,int axis,
            List<Integer> map,List<CarrierAccess> carriers)throws Exception{
        return prepare(name,types,shape,axis,map,carriers,null);
    }
    private static Prepared prepare(String name,List<DataType> types,Shape shape,int axis,
            List<Integer> map,List<CarrierAccess> carriers,List<LayoutDescriptor> layouts)throws Exception{
        var base=layouts==null?CpuBatchNormInferenceLoweringTest.context(types,shape,axis,map)
                :CpuBatchNormInferenceLoweringTest.context(types,shape,axis,map,layouts);
        var config=new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,4,4,4096);
        var context=new PrepareContext<>(base.partition(),base.nodes(),base.values(),base.memoryRequirements(),
                base.constants(),new CpuPartitionAnalysisInputs(false,carriers,config));
        var plan=new CpuPartitionPreparer().analyze(context).plan(); assertEquals(4,plan.selectedRangeCount());
        var route=plan.units().getFirst().portablePlan(); var generator=new CpuClassFileKernelGenerator();
        byte[] bytes=generator.generateClassBytes(route.specialization(),route.kernelIr());
        retain(name,bytes,route.specialization().compatibilityBytes(),route.specialization().toString());
        return new Prepared(generator.defineClassBytes(route.specialization(),bytes).entryPoint(),
                plan.batchNormInferenceGeometry().orElseThrow().pack(new long[carriers.size()]),
                plan.batchNormInferenceGeometry().orElseThrow().rangeForm(),plan.elementCount(),bytes);
    }

    private static long invoke(DataType type,Prepared p,Object i,Object s,Object b,Object m,Object v,Object o)throws Throwable{
        for(long[] r:ranges(p.items)){ if(type==DataType.FLOAT64)p.handle.invokeExact((double[])i,(double[])s,(double[])b,(double[])m,(double[])v,(double[])o,p.geometry,r[0],r[1]);
            else if(type==DataType.FLOAT32)p.handle.invokeExact((float[])i,(float[])s,(float[])b,(float[])m,(float[])v,(float[])o,p.geometry,r[0],r[1]);
            else p.handle.invokeExact((short[])i,(short[])s,(short[])b,(short[])m,(short[])v,(short[])o,p.geometry,r[0],r[1]);} return checksum(type,o); }
    private static long invokeMixed64(Prepared p,float[]i,MemorySegment s,short[]b,
            MemorySegment m,double[]v,MemorySegment o)throws Throwable{for(long[]r:ranges(p.items))
        p.handle.invokeExact(i,s,b,m,v,o,p.geometry,r[0],r[1]);return checksumMixed(true,o);}
    private static long invokeMixed32(Prepared p,MemorySegment i,float[]s,MemorySegment b,
            float[]m,short[]v,float[]o)throws Throwable{for(long[]r:ranges(p.items))
        p.handle.invokeExact(i,s,b,m,v,o,p.geometry,r[0],r[1]);return checksumMixed(false,o);}

    private static long direct(DataType type,Prepared p,long[] shape,int axis,Object i,Object s,Object b,Object m,Object v,Object o){
        int c=(int)shape[axis];long suffix=1;for(int x=axis+1;x<shape.length;x++)suffix*=shape[x];long non=p.form==CpuBatchNormInferenceIr.RangeForm.CHANNEL_RANGE?Arrays.stream(shape).reduce(1,Math::multiplyExact)/c:p.items;
        for(long[]r:ranges(p.items))if(p.form==CpuBatchNormInferenceIr.RangeForm.CHANNEL_RANGE)for(long ch=r[0];ch<r[1];ch++)for(long q=0;q<non;q++)formula(type,i,s,b,m,v,o,(q/suffix*c+ch)*suffix+q%suffix,(int)ch);
        else for(long ch=0;ch<c;ch++)for(long q=r[0];q<r[1];q++)formula(type,i,s,b,m,v,o,(q/suffix*c+ch)*suffix+q%suffix,(int)ch);return checksum(type,o);}
    private static long directMixed(boolean f64,Prepared p,Object i,Object s,Object b,Object m,Object v,Object o,
            long io,long[]is,long[]vo,long[]vs,long oo,long[]os){for(long[]r:ranges(p.items))for(int c=0;c<32;c++)for(long q=r[0];q<r[1];q++){long n=q/64,z=q%64,ia=io+n*is[0]+c*is[1]+z*is[2],oa=oo+n*os[0]+c*os[1]+z*os[2];if(f64){double centered=((float[])i)[(int)ia]-((MemorySegment)m).get(FLOAT,(vo[2]+c*vs[2])*4);double rad=((double[])v)[(int)(vo[3]+c*vs[3])]+1e-5;double den=Math.sqrt(rad),std=centered/den,scaled=std*((MemorySegment)s).get(DOUBLE,(vo[0]+c*vs[0])*8);((MemorySegment)o).set(DOUBLE,oa*8,scaled+Float.intBitsToFloat(Short.toUnsignedInt(((short[])b)[(int)vo[1]])<<16));}else{float centered=Float.intBitsToFloat(Short.toUnsignedInt(((MemorySegment)i).get(SHORT,ia*2))<<16)-((float[])m)[(int)(vo[2]+c*vs[2])];float rad=Float.intBitsToFloat(Short.toUnsignedInt(((short[])v)[(int)(vo[3]+c*vs[3])])<<16)+1e-5f;float den=(float)Math.sqrt(rad),std=centered/den,scaled=std*((float[])s)[(int)(vo[0]+c*vs[0])];((float[])o)[(int)oa]=scaled+Float.intBitsToFloat(Short.toUnsignedInt(((MemorySegment)b).get(SHORT,vo[1]*2))<<16);}}return checksumMixed(f64,o);}
    private static void formula(DataType result,Object i,Object s,Object b,Object m,Object v,Object o,long x,int c){
        if(result==DataType.FLOAT64){double centered=read(i,x)-read(m,c),radicand=read(v,c)+1e-5,den=Math.sqrt(radicand),standard=centered/den,scaled=standard*read(s,c);((double[])o)[(int)x]=scaled+read(b,c);}
        else{float centered=(float)((float)read(i,x)-(float)read(m,c));float radicand=(float)((float)read(v,c)+1e-5f);float den=(float)Math.sqrt(radicand);float standard=centered/den;float scaled=standard*(float)read(s,c);float value=scaled+(float)read(b,c);if(result==DataType.BFLOAT16)((short[])o)[(int)x]=bf(value);else((float[])o)[(int)x]=value;}}
    private static void initializeMixed(boolean f64,Object i,Object s,Object b,Object m,Object v,
            long io,long[]is,long[]vo,long[]vs){for(int n=0;n<16;n++)for(int c=0;c<32;c++)for(int z=0;z<64;z++){float value=n*.125f+c*.03125f+z*.0078125f-1;if(f64)((float[])i)[(int)(io+n*is[0]+c*is[1]+z*is[2])]=value;else((MemorySegment)i).set(SHORT,(io+n*is[0]+c*is[1]+z*is[2])*2,bf(value));}for(int c=0;c<32;c++){float scale=.5f+c*.015625f,bias=.125f,mean=-.25f+c*.0078125f,variance=.5f+c*.03125f;if(f64){((MemorySegment)s).set(DOUBLE,(vo[0]+c*vs[0])*8,scale);((short[])b)[(int)vo[1]]=bf(bias);((MemorySegment)m).set(FLOAT,(vo[2]+c*vs[2])*4,mean);((double[])v)[(int)(vo[3]+c*vs[3])]=variance;}else{((float[])s)[(int)(vo[0]+c*vs[0])]=scale;((MemorySegment)b).set(SHORT,vo[1]*2,bf(bias));((float[])m)[(int)(vo[2]+c*vs[2])]=mean;((short[])v)[(int)(vo[3]+c*vs[3])]=bf(variance);}}}
    private static long checksumMixed(boolean f64,Object o){long h=0;if(f64){MemorySegment x=(MemorySegment)o;for(long p=0;p<x.byteSize();p+=8)h=Long.rotateLeft(h,1)^Double.doubleToRawLongBits(x.get(DOUBLE,p));}else for(float x:(float[])o)h=Long.rotateLeft(h,1)^Integer.toUnsignedLong(Float.floatToRawIntBits(x));return h;}
    private static void equalMixed(boolean f64,Object a,Object b,int span){if(f64){for(int x=0;x<span;x++)assertEquals(Double.doubleToRawLongBits(((MemorySegment)a).get(DOUBLE,x*8L)),Double.doubleToRawLongBits(((MemorySegment)b).get(DOUBLE,x*8L)),"mixed output "+x);}else assertArrayEquals((float[])a,(float[])b);}
    private static double read(Object a,long i){if(a instanceof double[]x)return x[(int)i];if(a instanceof float[]x)return x[(int)i];return Float.intBitsToFloat(Short.toUnsignedInt(((short[])a)[(int)i])<<16);}
    private static CarrierAccess array(DataType t){return t==DataType.FLOAT64?CarrierAccess.DOUBLE_ARRAY:t==DataType.FLOAT32?CarrierAccess.FLOAT_ARRAY:CarrierAccess.SHORT_ARRAY;}
    private static Object array(DataType t,int n){return t==DataType.FLOAT64?new double[n]:t==DataType.FLOAT32?new float[n]:new short[n];}
    private static void fill(DataType t,Object a,int seed){Random r=new Random(seed);if(t==DataType.FLOAT64){double[]x=(double[])a;for(int i=0;i<x.length;i++)x[i]=r.nextDouble()*2-1;}else if(t==DataType.FLOAT32){float[]x=(float[])a;for(int i=0;i<x.length;i++)x[i]=r.nextFloat()*2-1;}else{short[]x=(short[])a;for(int i=0;i<x.length;i++)x[i]=bf(r.nextFloat()*2-1);}}
    private static void fillPositive(DataType t,Object a){fill(t,a,29);if(t==DataType.FLOAT64){double[]x=(double[])a;for(int i=0;i<x.length;i++)x[i]=Math.abs(x[i])+0.25;}else if(t==DataType.FLOAT32){float[]x=(float[])a;for(int i=0;i<x.length;i++)x[i]=Math.abs(x[i])+0.25f;}else{short[]x=(short[])a;for(int i=0;i<x.length;i++)x[i]=bf(Math.abs((float)read(x,i))+0.25f);}}
    private static short bf(float v){int bits=Float.floatToRawIntBits(v),upper=bits>>>16,lower=bits&0xffff;if(lower>0x8000||lower==0x8000&&(upper&1)!=0)upper++;return(short)upper;}
    private static long[][] ranges(long n){long[][]r=new long[4][2];for(int i=0;i<4;i++){r[i][0]=n*i/4;r[i][1]=n*(i+1)/4;}return r;}
    private static long checksum(DataType t,Object a){long h=0;if(t==DataType.FLOAT64)for(double v:(double[])a)h=Long.rotateLeft(h,1)^Double.doubleToRawLongBits(v);else if(t==DataType.FLOAT32)for(float v:(float[])a)h=Long.rotateLeft(h,1)^Integer.toUnsignedLong(Float.floatToRawIntBits(v));else for(short v:(short[])a)h=Long.rotateLeft(h,1)^Short.toUnsignedLong(v);return h;}
    private static void equal(DataType t,Object a,Object b){if(t==DataType.FLOAT64)assertArrayEquals((double[])a,(double[])b);else if(t==DataType.FLOAT32)assertArrayEquals((float[])a,(float[])b);else assertArrayEquals((short[])a,(short[])b);}
    private static long time(Action a,int n)throws Throwable{long start=System.nanoTime(),v=0;for(int i=0;i<n;i++)v^=a.run();sink^=v;return System.nanoTime()-start;}
    private static int repetitions(Action a)throws Throwable{int n=1;while(time(a,n)<MIN_BATCH_NANOS)n=Math.multiplyExact(n,2);return n;}
    private static long median(long[]a){long[]c=a.clone();Arrays.sort(c);return c[4];}
    private static void retain(String name,byte[]bytes,byte[]compat,String specialization)throws Exception{var model=ClassFile.of().parse(bytes);StringBuilder members=new StringBuilder();java.util.stream.StreamSupport.stream(model.constantPool().spliterator(),false).filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast).forEach(m->members.append(m.owner().asInternalName()).append('.').append(m.name().stringValue()).append(m.type().stringValue()).append('\n'));Path dir=Path.of("/private/tmp/synaptik-cpu-0007f1-retained-evidence-20260825/generated");Files.createDirectories(dir);Files.write(dir.resolve(name+".class"),bytes);Files.write(dir.resolve(name+".compatibility"),compat);Files.writeString(dir.resolve(name+".specialization"),specialization+"\n");Files.writeString(dir.resolve(name+".members"),members.toString());}
}
