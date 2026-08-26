package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormTrainingLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormTrainingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.*;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/** Frozen generated/direct five-fork performance driver for CPU 0007F2. */
public final class CpuBatchNormTrainingPerformanceTest {
    private static final long MIN_BATCH_NANOS = 25_000_000L;
    private static final Path EVIDENCE = Path.of(
            "/private/tmp/synaptik-cpu-0007f2-retained-evidence-20260825");
    private static final ValueLayout.OfShort SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfFloat FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfDouble DOUBLE = ValueLayout.JAVA_DOUBLE_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());
    private static volatile long sink;

    @FunctionalInterface interface BenchmarkAction { long run() throws Throwable; }
    @FunctionalInterface interface Verification { void run() throws Throwable; }
    record BenchmarkCase(String name, BenchmarkAction generated, BenchmarkAction direct,
            Verification verify) { }
    private record Prepared(MethodHandle handle, MemorySegment generatedScratch,
            MemorySegment directScratch, long[] packed, CpuBatchNormTrainingLowering.Geometry g,
            int ranges) { }

    static BenchmarkCase adapt(String name, BenchmarkAction generated, BenchmarkAction direct,
            Verification verify) {
        return new BenchmarkCase(name, generated, direct, verify);
    }

    @Test void frozenInventoryRunsSemanticAndChecksumGates() throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            List<BenchmarkCase> cases = cases(arena);
            assertEquals(List.of("BNT-BF16-A1", "BNT-F32-A1", "BNT-F64-A1", "BNT-F32-A0",
                    "BNT-F32-A2", "BNT-MIX-F64", "BNT-MIX-F32", "BNT-REPEAT",
                    "CONTROL_F32_BATCH_INFERENCE", "CONTROL_F32_LAYER_NORM",
                    "CONTROL_F32_VARIANCE", "CONTROL_F32_ADD"),
                    cases.stream().map(BenchmarkCase::name).toList());
            for (BenchmarkCase value : cases) gate(value);
        }
    }

    @Test void retainedIsolatedFork() throws Throwable {
        String fork = System.getProperty("synaptik.cpu.batchnorm.training.fork");
        if (fork == null) fork = System.getenv("SYNAPTIK_CPU_BATCHNORM_TRAINING_FORK");
        Assumptions.assumeTrue(fork != null);
        main(new String[] {fork, System.getProperty("synaptik.cpu.batchnorm.training.attempt", "0")});
    }

    /** Runs one isolated sample, or aggregates accepted fork zero through four. */
    public static void main(String[] args) throws Throwable {
        if (args.length > 0 && args[0].equals("aggregate")) { aggregate(); return; }
        int fork = args.length == 0 ? 0 : Integer.parseInt(args[0]);
        int attempt = args.length < 2 ? 0 : Integer.parseInt(args[1]);
        StringBuilder report = new StringBuilder();
        try (Arena arena = Arena.ofConfined()) {
            environment(report);
            List<BenchmarkCase> cases = cases(arena);
            int[] repetitions = new int[cases.size() * 2];
            for (int i = 0; i < cases.size(); i++) {
                gate(cases.get(i));
                repetitions[2 * i] = repetitions(cases.get(i).generated);
                repetitions[2 * i + 1] = repetitions(cases.get(i).direct);
            }
            long[][] generated = new long[cases.size()][9], direct = new long[cases.size()][9];
            Random random = new Random(0x0007f220260826L ^ fork * 0x9e3779b97f4a7c15L
                    ^ attempt * 0xd1b54a32d192ed03L);
            for (int round = -5; round < 9; round++) {
                List<Integer> order = new ArrayList<>();
                for (int i = 0; i < cases.size(); i++) order.add(i);
                Collections.shuffle(order, random);
                for (int i : order) {
                    long gt, dt;
                    if (random.nextBoolean()) {
                        gt = time(cases.get(i).generated, repetitions[2 * i]);
                        dt = time(cases.get(i).direct, repetitions[2 * i + 1]);
                    } else {
                        dt = time(cases.get(i).direct, repetitions[2 * i + 1]);
                        gt = time(cases.get(i).generated, repetitions[2 * i]);
                    }
                    if (round >= 0) {
                        generated[i][round] = gt / repetitions[2 * i];
                        direct[i][round] = dt / repetitions[2 * i + 1];
                        gate(cases.get(i));
                    }
                }
            }
            int failures = 0;
            for (int i = 0; i < cases.size(); i++) {
                long gm = median(generated[i]), dm = median(direct[i]);
                double ratio = (double) gm / dm;
                if (!(ratio <= 1.15)) failures++;
                report.append(String.format(Locale.ROOT, "RESULT,%s,%d,%d,%.9f,%s,%s%n",
                        cases.get(i).name, gm, dm, ratio, Arrays.toString(generated[i]),
                        Arrays.toString(direct[i])));
            }
            report.append("SINK,").append(sink).append('\n');
            if (failures != 0) throw new AssertionError("ratio failures " + failures);
        } catch (Throwable failure) {
            report.append("REJECTED,").append(failure.getClass().getName()).append(',')
                    .append(clean(failure.getMessage())).append('\n');
            retain(false, fork, attempt, report.toString());
            System.out.print(report); throw failure;
        }
        retain(true, fork, attempt, report.toString());
        System.out.print(report);
    }

    private static List<BenchmarkCase> cases(Arena arena) throws Throwable {
        List<BenchmarkCase> result = new ArrayList<>();
        result.add(dense(arena, "BNT-BF16-A1", DataType.BFLOAT16, Shape.of(32,64,256),1,4));
        result.add(dense(arena, "BNT-F32-A1", DataType.FLOAT32, Shape.of(32,64,256),1,4));
        result.add(dense(arena, "BNT-F64-A1", DataType.FLOAT64, Shape.of(32,64,256),1,1));
        result.add(dense(arena, "BNT-F32-A0", DataType.FLOAT32, Shape.of(128,4096),0,4));
        result.add(dense(arena, "BNT-F32-A2", DataType.FLOAT32, Shape.of(32,256,64),2,4));
        result.add(mixed(arena, true)); result.add(mixed(arena, false)); result.add(repeated(arena));
        result.add(CpuBatchNormInferenceEvidenceTest.float32BatchInferenceControl());
        result.addAll(CpuTrailingNormalizationEvidenceTest.trainingControls(arena));
        return List.copyOf(result);
    }

    private static BenchmarkCase dense(Arena arena, String name, DataType type, Shape shape,
            int axis, int ranges) throws Throwable {
        long[] e = shape.toLongArray(); int channels = Math.toIntExact(e[axis]);
        int count = Math.toIntExact(Arrays.stream(e).reduce(1, Math::multiplyExact));
        Shape vector = Shape.of(channels); List<LayoutDescriptor> layouts = new ArrayList<>();
        layouts.add(LayoutDescriptor.contiguous(shape));
        for (int i=0;i<4;i++) layouts.add(LayoutDescriptor.contiguous(vector));
        layouts.add(LayoutDescriptor.contiguous(shape));
        for (int i=0;i<4;i++) layouts.add(LayoutDescriptor.contiguous(vector));
        Prepared p=prepare(arena,Collections.nCopies(5,type),shape,axis,List.of(0,1,2,3,4),
                layouts,Collections.nCopies(10,arrayCarrier(type)),ranges);
        Object input=array(type,count),scale=array(type,channels),bias=array(type,channels),
                oldMean=array(type,channels),oldVar=array(type,channels);
        Object[] go=outputs(type,count,channels),do_=outputs(type,count,channels);
        fill(type,input,scale,bias,oldMean,oldVar);
        if(type==DataType.FLOAT64)return adapt(name,()->invoke64(p,(double[])input,(double[])scale,
                (double[])bias,(double[])oldMean,(double[])oldVar,go),()->direct64(p,(double[])input,
                (double[])scale,(double[])bias,(double[])oldMean,(double[])oldVar,do_),
                ()->equal(type,go,do_));
        if(type==DataType.FLOAT32)return adapt(name,()->invoke32(p,(float[])input,(float[])scale,
                (float[])bias,(float[])oldMean,(float[])oldVar,go),()->direct32(p,(float[])input,
                (float[])scale,(float[])bias,(float[])oldMean,(float[])oldVar,do_),
                ()->equal(type,go,do_));
        return adapt(name,()->invoke16(p,(short[])input,(short[])scale,(short[])bias,
                (short[])oldMean,(short[])oldVar,go),()->direct16(p,(short[])input,(short[])scale,
                (short[])bias,(short[])oldMean,(short[])oldVar,do_),()->equal(type,go,do_));
    }

    private static BenchmarkCase mixed(Arena arena, boolean f64) throws Throwable {
        Shape shape=Shape.of(16,32,64),vector=Shape.of(32);
        List<LayoutDescriptor> ls=layouts(shape,vector,new long[]{5000,137,2},11,
                new long[]{6000,151,2},13,new long[]{3,5,7,11},new long[]{2,1,3,1});
        if(f64){
            Prepared p=prepare(arena,List.of(DataType.FLOAT32,DataType.FLOAT64,DataType.BFLOAT16,
                    DataType.FLOAT32,DataType.FLOAT64),shape,1,List.of(0,1,2,3,4),ls,List.of(
                    CarrierAccess.FLOAT_ARRAY,CarrierAccess.MEMORY_SEGMENT,CarrierAccess.SHORT_ARRAY,
                    CarrierAccess.MEMORY_SEGMENT,CarrierAccess.DOUBLE_ARRAY,CarrierAccess.MEMORY_SEGMENT,
                    CarrierAccess.DOUBLE_ARRAY,CarrierAccess.MEMORY_SEGMENT,CarrierAccess.DOUBLE_ARRAY,
                    CarrierAccess.MEMORY_SEGMENT),4);
            float[] x=new float[span(ls.get(0))];MemorySegment s=arena.allocate(span(ls.get(1))*8L,8);
            short[] b=new short[span(ls.get(2))];MemorySegment m=arena.allocate(span(ls.get(3))*4L,4);
            double[] v=new double[span(ls.get(4))];Object[] go=mixedOutputs(arena,ls,true),do_=mixedOutputs(arena,ls,true);
            fillMixed64(p.g,x,s,b,m,v);
            return adapt("BNT-MIX-F64",()->invokeMixed64(p,x,s,b,m,v,go),
                    ()->directMixed64(p,x,s,b,m,v,do_),()->equal(DataType.FLOAT64,go,do_));
        }
        Prepared p=prepare(arena,List.of(DataType.BFLOAT16,DataType.FLOAT32,DataType.BFLOAT16,
                DataType.FLOAT32,DataType.BFLOAT16),shape,1,List.of(0,1,2,3,4),ls,List.of(
                CarrierAccess.MEMORY_SEGMENT,CarrierAccess.FLOAT_ARRAY,CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.FLOAT_ARRAY,CarrierAccess.SHORT_ARRAY,CarrierAccess.FLOAT_ARRAY,
                CarrierAccess.MEMORY_SEGMENT,CarrierAccess.FLOAT_ARRAY,CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.FLOAT_ARRAY),4);
        MemorySegment x=arena.allocate(span(ls.get(0))*2L,2);float[] s=new float[span(ls.get(1))];
        MemorySegment b=arena.allocate(span(ls.get(2))*2L,2);float[] m=new float[span(ls.get(3))];
        short[] v=new short[span(ls.get(4))];Object[] go=mixedOutputs(arena,ls,false),do_=mixedOutputs(arena,ls,false);
        fillMixed32(p.g,x,s,b,m,v);
        return adapt("BNT-MIX-F32",()->invokeMixed32(p,x,s,b,m,v,go),
                ()->directMixed32(p,x,s,b,m,v,do_),()->equal(DataType.FLOAT32,go,do_));
    }

    private static BenchmarkCase repeated(Arena arena)throws Throwable{
        Shape shape=Shape.of(32,64,128),vector=Shape.of(64);List<LayoutDescriptor> ls=layouts(shape,
                vector,new long[]{8192,128,1},0,new long[]{8192,128,1},0,
                new long[]{2,3,4,5},new long[]{2,2,2,2});
        Prepared p=prepare(arena,Collections.nCopies(5,DataType.FLOAT32),shape,1,List.of(0,1,1,1,1),ls,
                List.of(CarrierAccess.FLOAT_ARRAY,CarrierAccess.FLOAT_ARRAY,CarrierAccess.FLOAT_ARRAY,
                CarrierAccess.MEMORY_SEGMENT,CarrierAccess.FLOAT_ARRAY,CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.FLOAT_ARRAY),4);
        float[] x=new float[32*64*128],shared=new float[span(ls.get(1))];
        fill(DataType.FLOAT32,x,shared,shared,shared,shared);
        Object[] go={new float[x.length],arena.allocate(span(ls.get(6))*4L,4),new float[span(ls.get(7))],
                arena.allocate(span(ls.get(8))*4L,4),new float[span(ls.get(9))]};
        Object[] do_={new float[x.length],arena.allocate(span(ls.get(6))*4L,4),new float[span(ls.get(7))],
                arena.allocate(span(ls.get(8))*4L,4),new float[span(ls.get(9))]};
        return adapt("BNT-REPEAT",()->invokeRepeat(p,x,shared,go),
                ()->directRepeat(p,x,shared,do_),()->equal(DataType.FLOAT32,go,do_));
    }

    private static Prepared prepare(Arena arena,List<DataType> types,Shape shape,int axis,
            List<Integer> occurrences,List<LayoutDescriptor> layouts,List<CarrierAccess> carriers,
            int ranges)throws Exception{
        var base=CpuBatchNormTrainingLoweringTest.context(types,shape,axis,occurrences,layouts);
        var config=new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,ranges,ranges,4096);
        var context=new PrepareContext<>(base.partition(),base.nodes(),base.values(),base.memoryRequirements(),
                base.constants(),new CpuPartitionAnalysisInputs(false,carriers,config));
        var plan=new CpuPartitionPreparer().analyze(context).plan();assertEquals(ranges,plan.selectedRangeCount());
        var route=plan.units().getFirst().portablePlan();var gen=new CpuClassFileKernelGenerator();
        byte[] bytes=gen.generateClassBytes(route.specialization(),route.kernelIr());var g=plan.batchNormTrainingGeometry().orElseThrow();
        return new Prepared(gen.defineClassBytes(route.specialization(),bytes).entryPoint(),
                arena.allocate(g.scratchSliceBytes()*ranges,8),arena.allocate(g.scratchSliceBytes()*ranges,8),
                g.pack(new long[carriers.size()],0),g,ranges);
    }

    private static MemorySegment scratch(MemorySegment whole,Prepared p,int range){long n=p.g.scratchSliceBytes();return whole.asSlice(n*range,n);}
    private static long start(Prepared p,int r){return p.g.channelCount()*r/p.ranges;}
    private static long end(Prepared p,int r){return p.g.channelCount()*(r+1)/p.ranges;}

    private static long invoke64(Prepared p,double[]x,double[]s,double[]b,double[]m,double[]v,Object[]o)throws Throwable{for(int r=0;r<p.ranges;r++)p.handle.invokeExact(x,s,b,m,v,(double[])o[0],(double[])o[1],(double[])o[2],(double[])o[3],(double[])o[4],scratch(p.generatedScratch,p,r),p.packed,start(p,r),end(p,r));return checksum(DataType.FLOAT64,o);}
    private static long invoke32(Prepared p,float[]x,float[]s,float[]b,float[]m,float[]v,Object[]o)throws Throwable{for(int r=0;r<p.ranges;r++)p.handle.invokeExact(x,s,b,m,v,(float[])o[0],(float[])o[1],(float[])o[2],(float[])o[3],(float[])o[4],scratch(p.generatedScratch,p,r),p.packed,start(p,r),end(p,r));return checksum(DataType.FLOAT32,o);}
    private static long invoke16(Prepared p,short[]x,short[]s,short[]b,short[]m,short[]v,Object[]o)throws Throwable{for(int r=0;r<p.ranges;r++)p.handle.invokeExact(x,s,b,m,v,(short[])o[0],(short[])o[1],(short[])o[2],(short[])o[3],(short[])o[4],scratch(p.generatedScratch,p,r),p.packed,start(p,r),end(p,r));return checksum(DataType.BFLOAT16,o);}
    private static long invokeMixed64(Prepared p,float[]x,MemorySegment s,short[]b,MemorySegment m,double[]v,Object[]o)throws Throwable{for(int r=0;r<4;r++)p.handle.invokeExact(x,s,b,m,v,(MemorySegment)o[0],(double[])o[1],(MemorySegment)o[2],(double[])o[3],(MemorySegment)o[4],scratch(p.generatedScratch,p,r),p.packed,start(p,r),end(p,r));return checksum(DataType.FLOAT64,o);}
    private static long invokeMixed32(Prepared p,MemorySegment x,float[]s,MemorySegment b,float[]m,short[]v,Object[]o)throws Throwable{for(int r=0;r<4;r++)p.handle.invokeExact(x,s,b,m,v,(float[])o[0],(MemorySegment)o[1],(float[])o[2],(MemorySegment)o[3],(float[])o[4],scratch(p.generatedScratch,p,r),p.packed,start(p,r),end(p,r));return checksum(DataType.FLOAT32,o);}
    private static long invokeRepeat(Prepared p,float[]x,float[]s,Object[]o)throws Throwable{for(int r=0;r<4;r++)p.handle.invokeExact(x,s,(float[])o[0],(MemorySegment)o[1],(float[])o[2],(MemorySegment)o[3],(float[])o[4],scratch(p.generatedScratch,p,r),p.packed,start(p,r),end(p,r));return checksum(DataType.FLOAT32,o);}

    private static long direct64(Prepared p,double[]x,double[]s,double[]b,double[]m,double[]v,Object[]o){for(int r=0;r<p.ranges;r++)for(int c=(int)start(p,r);c<end(p,r);c++)channel64(p,scratch(p.directScratch,p,r),c,x,s,b,m,v,(double[])o[0],(double[])o[1],(double[])o[2],(double[])o[3],(double[])o[4]);return checksum(DataType.FLOAT64,o);}
    private static long direct32(Prepared p,float[]x,float[]s,float[]b,float[]m,float[]v,Object[]o){for(int r=0;r<p.ranges;r++)for(int c=(int)start(p,r);c<end(p,r);c++)channel32(p,scratch(p.directScratch,p,r),c,x,s,b,m,v,(float[])o[0],(float[])o[1],(float[])o[2],(float[])o[3],(float[])o[4],false);return checksum(DataType.FLOAT32,o);}
    private static long direct16(Prepared p,short[]x,short[]s,short[]b,short[]m,short[]v,Object[]o){for(int r=0;r<p.ranges;r++)for(int c=(int)start(p,r);c<end(p,r);c++)channel16(p,scratch(p.directScratch,p,r),c,x,s,b,m,v,(short[])o[0],(short[])o[1],(short[])o[2],(short[])o[3],(short[])o[4]);return checksum(DataType.BFLOAT16,o);}

    private static void channel64(Prepared p,MemorySegment state,int c,double[]x,double[]scale,double[]bias,double[]oldMean,double[]oldVar,double[]out,double[]nextMean,double[]nextVar,double[]savedMean,double[]savedInv){var g=p.g;Exact.reset(state,g.stateLimbCount());for(int a=0;a<g.prefixCount();a++){int q=(int)((a*g.channelCount()+c)*g.suffixCount()),z=(int)(q+g.suffixCount());for(;q<z;q++)Exact.add64(state,x[q],g.stateLimbCount());}double mean=Double.longBitsToDouble(Exact.mean(state,Exact.F64,g.stateLimbCount(),g.reductionCount()));double ds=0,dc=0,ss=0,sc=0;for(int a=0;a<g.prefixCount();a++){int q=(int)((a*g.channelCount()+c)*g.suffixCount()),z=(int)(q+g.suffixCount());for(;q<z;q++){double d=x[q]-mean,y=d-dc,t=ds+y;dc=(t-ds)-y;ds=t;double sq=d*d; y=sq-sc;t=ss+y;sc=(t-ss)-y;ss=t;}}double n=ss-ds*ds/g.reductionCount();if(n<0)n=0;double bi=n/g.reductionCount(),un=n/(g.reductionCount()-1),inv=1/Math.sqrt(bi+1e-5),om=1-.25;nextMean[c]=om*oldMean[c]+.25*mean;nextVar[c]=om*oldVar[c]+.25*un;savedMean[c]=mean;savedInv[c]=inv;for(int a=0;a<g.prefixCount();a++){int q=(int)((a*g.channelCount()+c)*g.suffixCount()),z=(int)(q+g.suffixCount());for(;q<z;q++){double w=x[q]-mean;w=w*inv;w=w*scale[c];out[q]=w+bias[c];}}}
    private static void channel32(Prepared p,MemorySegment state,int c,float[]x,float[]scale,float[]bias,float[]oldMean,float[]oldVar,float[]out,float[]nextMean,float[]nextVar,float[]savedMean,float[]savedInv,boolean ignored){var g=p.g;Exact.reset(state,g.stateLimbCount());for(int a=0;a<g.prefixCount();a++){int q=(int)((a*g.channelCount()+c)*g.suffixCount()),z=(int)(q+g.suffixCount());for(;q<z;q++)Exact.add32(state,x[q],g.stateLimbCount());}float mean=Float.intBitsToFloat((int)Exact.mean(state,Exact.F32,g.stateLimbCount(),g.reductionCount()));double ds=0,dc=0,ss=0,sc=0;for(int a=0;a<g.prefixCount();a++){int q=(int)((a*g.channelCount()+c)*g.suffixCount()),z=(int)(q+g.suffixCount());for(;q<z;q++){double d=(float)(x[q]-mean),y=d-dc,t=ds+y;dc=(t-ds)-y;ds=t;double sq=d*d;y=sq-sc;t=ss+y;sc=(t-ss)-y;ss=t;}}double n=ss-ds*ds/g.reductionCount();if(n<0)n=0;float bi=(float)((float)n/(float)g.reductionCount()),un=(float)((float)n/(float)(g.reductionCount()-1)),inv=1f/(float)Math.sqrt(bi+1e-5f),om=1f-.25f;nextMean[c]=om*oldMean[c]+.25f*mean;nextVar[c]=om*oldVar[c]+.25f*un;savedMean[c]=mean;savedInv[c]=inv;for(int a=0;a<g.prefixCount();a++){int q=(int)((a*g.channelCount()+c)*g.suffixCount()),z=(int)(q+g.suffixCount());for(;q<z;q++){float w=x[q]-mean;w=w*inv;w=w*scale[c];out[q]=w+bias[c];}}}
    private static void channel16(Prepared p,MemorySegment state,int c,short[]x,short[]scale,short[]bias,short[]oldMean,short[]oldVar,short[]out,short[]nextMean,short[]nextVar,short[]savedMean,short[]savedInv){var g=p.g;Exact.reset(state,g.stateLimbCount());for(int a=0;a<g.prefixCount();a++){int q=(int)((a*g.channelCount()+c)*g.suffixCount()),z=(int)(q+g.suffixCount());for(;q<z;q++)Exact.add32(state,decode(x[q]),g.stateLimbCount());}float mean=Float.intBitsToFloat((int)Exact.mean(state,Exact.F32,g.stateLimbCount(),g.reductionCount()));double ds=0,dc=0,ss=0,sc=0;for(int a=0;a<g.prefixCount();a++){int q=(int)((a*g.channelCount()+c)*g.suffixCount()),z=(int)(q+g.suffixCount());for(;q<z;q++){double d=(float)(decode(x[q])-mean),y=d-dc,t=ds+y;dc=(t-ds)-y;ds=t;double sq=d*d;y=sq-sc;t=ss+y;sc=(t-ss)-y;ss=t;}}double n=ss-ds*ds/g.reductionCount();if(n<0)n=0;float bi=(float)((float)n/(float)g.reductionCount()),un=(float)((float)n/(float)(g.reductionCount()-1)),inv=1f/(float)Math.sqrt(bi+decode((short)0x3728)),om=1f-.25f;nextMean[c]=bf(om*decode(oldMean[c])+.25f*mean);nextVar[c]=bf(om*decode(oldVar[c])+.25f*un);savedMean[c]=bf(mean);savedInv[c]=bf(inv);float s=decode(scale[c]),b=decode(bias[c]);for(int a=0;a<g.prefixCount();a++){int q=(int)((a*g.channelCount()+c)*g.suffixCount()),z=(int)(q+g.suffixCount());for(;q<z;q++){float w=decode(x[q])-mean;w=w*inv;w=w*s;out[q]=bf(w+b);}}}

    private static long directMixed64(Prepared p,float[]x,MemorySegment scale,short[]bias,
            MemorySegment oldMean,double[]oldVar,Object[]o){var g=p.g;var in=g.inputs();var out=g.outputs();
        for(int r=0;r<4;r++){MemorySegment state=scratch(p.directScratch,p,r);for(int c=(int)start(p,r);c<end(p,r);c++){
            Exact.reset(state,g.stateLimbCount());long ib=in.get(0).offset()+c*in.get(0).strides()[1];
            for(int a=0;a<16;a++){long q=ib+a*in.get(0).strides()[0];for(int z=0;z<64;z++,q+=in.get(0).strides()[2])Exact.add32(state,x[(int)q],g.stateLimbCount());}
            double mean=Float.intBitsToFloat((int)Exact.mean(state,Exact.F32,g.stateLimbCount(),1024)),ds=0,dc=0,ss=0,sc=0;
            for(int a=0;a<16;a++){long q=ib+a*in.get(0).strides()[0];for(int z=0;z<64;z++,q+=in.get(0).strides()[2]){double d=x[(int)q]-mean,y=d-dc,t=ds+y;dc=(t-ds)-y;ds=t;double sq=d*d;y=sq-sc;t=ss+y;sc=(t-ss)-y;ss=t;}}
            double n=ss-ds*ds/1024.;if(n<0)n=0;double bi=n/1024.,un=n/1023.,inv=1/Math.sqrt(bi+1e-5),s=scale.get(DOUBLE,vecByte(in.get(1),c,8)),b=decode(bias[(int)vec(in.get(2),c)]),om=oldMean.get(FLOAT,vecByte(in.get(3),c,4)),ov=oldVar[(int)vec(in.get(4),c)];
            store64(o[1],out.get(1),c,.75*om+.25*mean);store64(o[2],out.get(2),c,.75*ov+.25*un);store64(o[3],out.get(3),c,mean);store64(o[4],out.get(4),c,inv);
            long ob=out.get(0).offset()+c*out.get(0).strides()[1];for(int a=0;a<16;a++){long q=ib+a*in.get(0).strides()[0],w=ob+a*out.get(0).strides()[0];for(int z=0;z<64;z++,q+=in.get(0).strides()[2],w+=out.get(0).strides()[2]){double value=x[(int)q]-mean;value=value*inv;value=value*s;((MemorySegment)o[0]).set(DOUBLE,w*8,value+b);}}
        }}return checksum(DataType.FLOAT64,o);}

    private static long directMixed32(Prepared p,MemorySegment x,float[]scale,MemorySegment bias,
            float[]oldMean,short[]oldVar,Object[]o){var g=p.g;var in=g.inputs();var out=g.outputs();
        for(int r=0;r<4;r++){MemorySegment state=scratch(p.directScratch,p,r);for(int c=(int)start(p,r);c<end(p,r);c++){
            Exact.reset(state,g.stateLimbCount());long ib=in.get(0).offset()+c*in.get(0).strides()[1];
            for(int a=0;a<16;a++){long q=ib+a*in.get(0).strides()[0];for(int z=0;z<64;z++,q+=in.get(0).strides()[2])Exact.add32(state,decode(x.get(SHORT,q*2)),g.stateLimbCount());}
            float mean=Float.intBitsToFloat((int)Exact.mean(state,Exact.F32,g.stateLimbCount(),1024));double ds=0,dc=0,ss=0,sc=0;
            for(int a=0;a<16;a++){long q=ib+a*in.get(0).strides()[0];for(int z=0;z<64;z++,q+=in.get(0).strides()[2]){double d=(float)(decode(x.get(SHORT,q*2))-mean),y=d-dc,t=ds+y;dc=(t-ds)-y;ds=t;double sq=d*d;y=sq-sc;t=ss+y;sc=(t-ss)-y;ss=t;}}
            double n=ss-ds*ds/1024.;if(n<0)n=0;float bi=(float)((float)n/1024f),un=(float)((float)n/1023f),inv=1f/(float)Math.sqrt(bi+1e-5f),s=scale[(int)vec(in.get(1),c)],b=decode(bias.get(SHORT,vecByte(in.get(2),c,2))),om=oldMean[(int)vec(in.get(3),c)],ov=decode(oldVar[(int)vec(in.get(4),c)]);
            store32(o[1],out.get(1),c,.75f*om+.25f*mean);store32(o[2],out.get(2),c,.75f*ov+.25f*un);store32(o[3],out.get(3),c,mean);store32(o[4],out.get(4),c,inv);
            long ob=out.get(0).offset()+c*out.get(0).strides()[1];for(int a=0;a<16;a++){long q=ib+a*in.get(0).strides()[0],w=ob+a*out.get(0).strides()[0];for(int z=0;z<64;z++,q+=in.get(0).strides()[2],w+=out.get(0).strides()[2]){float value=decode(x.get(SHORT,q*2))-mean;value=value*inv;value=value*s;((float[])o[0])[(int)w]=value+b;}}
        }}return checksum(DataType.FLOAT32,o);}

    private static long directRepeat(Prepared p,float[]x,float[]shared,Object[]o){var g=p.g;var in=g.inputs().get(1);var out=g.outputs();
        for(int r=0;r<4;r++){MemorySegment state=scratch(p.directScratch,p,r);for(int c=(int)start(p,r);c<end(p,r);c++){
            Exact.reset(state,g.stateLimbCount());for(int a=0;a<32;a++){int q=(a*64+c)*128,z=q+128;for(;q<z;q++)Exact.add32(state,x[q],g.stateLimbCount());}
            float mean=Float.intBitsToFloat((int)Exact.mean(state,Exact.F32,g.stateLimbCount(),4096));double ds=0,dc=0,ss=0,sc=0;
            for(int a=0;a<32;a++){int q=(a*64+c)*128,z=q+128;for(;q<z;q++){double d=(float)(x[q]-mean),y=d-dc,t=ds+y;dc=(t-ds)-y;ds=t;double sq=d*d;y=sq-sc;t=ss+y;sc=(t-ss)-y;ss=t;}}
            double n=ss-ds*ds/4096.;if(n<0)n=0;float bi=(float)((float)n/4096f),un=(float)((float)n/4095f),inv=1f/(float)Math.sqrt(bi+1e-5f),v=shared[(int)vec(in,c)];
            store32(o[1],out.get(1),c,.75f*v+.25f*mean);store32(o[2],out.get(2),c,.75f*v+.25f*un);store32(o[3],out.get(3),c,mean);store32(o[4],out.get(4),c,inv);
            for(int a=0;a<32;a++){int q=(a*64+c)*128,z=q+128;for(;q<z;q++){float w=x[q]-mean;w=w*inv;w=w*v;((float[])o[0])[q]=w+v;}}
        }}return checksum(DataType.FLOAT32,o);}

    private static long vec(CpuBatchNormTrainingLowering.Layout l,int c){return l.offset()+c*l.strides()[0];}
    private static long vecByte(CpuBatchNormTrainingLowering.Layout l,int c,int width){return vec(l,c)*width;}
    private static void store64(Object carrier,CpuBatchNormTrainingLowering.Layout l,int c,double v){long a=vec(l,c);if(carrier instanceof double[]x)x[(int)a]=v;else((MemorySegment)carrier).set(DOUBLE,a*8,v);}
    private static void store32(Object carrier,CpuBatchNormTrainingLowering.Layout l,int c,float v){long a=vec(l,c);if(carrier instanceof float[]x)x[(int)a]=v;else((MemorySegment)carrier).set(FLOAT,a*4,v);}

    private static final class Exact{
        static final int F64=0,F32=1;
        static void reset(MemorySegment s,int n){for(int i=0;i<=n;i++)s.set(JAVA_LONG,i*8L,0);}
        static void add64(MemorySegment s,double v,int n){long b=Double.doubleToRawLongBits(v),f=b&0xfffffffffffffL,e=b>>>52&0x7ff;if(e==0x7ff||e==0&&f==0)return;add(s,n,e==0?f:(1L<<52)|f,e==0?0:(int)e-1023-52+1074,b<0);}
        static void add32(MemorySegment s,float v,int n){int b=Float.floatToRawIntBits(v);long f=b&0x7fffff,e=b>>>23&0xff;if(e==0xff||e==0&&f==0)return;add(s,n,e==0?f:(1L<<23)|f,e==0?0:(int)e-127-23+149,b<0);}
        static void add(MemorySegment s,int n,long sig,int shift,boolean neg){int word=shift>>>6,bit=shift&63;long carry=neg?1:0;for(int i=0;i<n;i++){long part=i==word?sig<<bit:bit!=0&&i==word+1?sig>>>(64-bit):0;if(neg)part=~part;long wc=part+carry,first=Long.compareUnsigned(wc,part)<0?1:0,old=s.get(JAVA_LONG,8L+i*8L),sum=old+wc,second=Long.compareUnsigned(sum,old)<0?1:0;carry=first|second;s.set(JAVA_LONG,8L+i*8L,sum);}}
        static long mean(MemorySegment s,int type,int n,long divisor){int min=type==F64?-1074:-149,max=type==F64?1023:127,frac=type==F64?52:23,bias=type==F64?1023:127,precision=frac+1;boolean neg=s.get(JAVA_LONG,n*8L)<0;if(neg){long carry=1;for(int i=0;i<n;i++){long old=~s.get(JAVA_LONG,8L+i*8L),next=old+carry;s.set(JAVA_LONG,8L+i*8L,next);carry=Long.compareUnsigned(next,old)<0?1:0;}}long rem=0;for(int w=n-1;w>=0;w--){long old=s.get(JAVA_LONG,8L+w*8L),q=0;for(int bit=63;bit>=0;bit--){long next=rem<<1|old>>>bit&1;if(Long.compareUnsigned(next,divisor)>=0){next-=divisor;q|=1L<<bit;}rem=next;}s.set(JAVA_LONG,8L+w*8L,q);}int top=n-1;while(top>0&&s.get(JAVA_LONG,8L+top*8L)==0)top--;long tw=s.get(JAVA_LONG,8L+top*8L);if(tw==0&&rem==0)return neg?(type==F64?Long.MIN_VALUE:1L<<31):0;int length=tw==0?0:top*64+64-Long.numberOfLeadingZeros(tw);long unbiased=(long)min+length-1;int shift=unbiased>=1-bias?length-precision:0;long rounded=extract(s,n,Math.max(0,shift));int guard;boolean sticky;if(shift>0){guard=bit(s,shift-1);sticky=shift>1&&below(s,shift-1)||rem!=0;}else{int cmp=Long.compareUnsigned(rem<<1,divisor);guard=cmp>=0?1:0;sticky=cmp>0;}if(guard!=0&&(sticky||(rounded&1)!=0))rounded++;if(unbiased>=1-bias&&rounded==1L<<precision){rounded>>>=1;unbiased++;}long sign=neg?(type==F64?Long.MIN_VALUE:1L<<31):0,exp=type==F64?0x7ff:0xff,mask=(1L<<frac)-1;if(unbiased>max)return sign|exp<<frac;if(unbiased>=1-bias)return sign|(unbiased+bias)<<frac|rounded&mask;if(rounded>=1L<<frac)return sign|1L<<frac;return sign|rounded;}
        static long extract(MemorySegment s,int n,int shift){int w=shift>>>6,b=shift&63;if(w>=n)return 0;long r=s.get(JAVA_LONG,8L+w*8L)>>>b;if(b!=0&&w+1<n)r|=s.get(JAVA_LONG,8L+(w+1)*8L)<<(64-b);return r;}
        static int bit(MemorySegment s,int p){return(int)(s.get(JAVA_LONG,8L+(p>>>6)*8L)>>>(p&63)&1);}
        static boolean below(MemorySegment s,int count){int full=count>>>6,b=count&63;for(int i=0;i<full;i++)if(s.get(JAVA_LONG,8L+i*8L)!=0)return true;return b!=0&&(s.get(JAVA_LONG,8L+full*8L)&(1L<<b)-1)!=0;}
    }

    private static List<LayoutDescriptor> layouts(Shape shape,Shape vector,long[]is,long io,
            long[]os,long oo,long[]vo,long[]vs){List<LayoutDescriptor>r=new ArrayList<>();
        r.add(LayoutDescriptor.of(shape,is,io,true));for(int i=0;i<4;i++)r.add(LayoutDescriptor.of(vector,new long[]{vs[i]},vo[i],true));
        r.add(LayoutDescriptor.of(shape,os,oo,true));for(int i=0;i<4;i++)r.add(LayoutDescriptor.of(vector,new long[]{vs[i]+1},vo[i]+1,true));return List.copyOf(r);}
    private static int span(LayoutDescriptor l){return Math.toIntExact(l.referencedElementSpan());}
    private static CarrierAccess arrayCarrier(DataType t){return t==DataType.FLOAT64?CarrierAccess.DOUBLE_ARRAY:t==DataType.FLOAT32?CarrierAccess.FLOAT_ARRAY:CarrierAccess.SHORT_ARRAY;}
    private static Object array(DataType t,int n){return t==DataType.FLOAT64?new double[n]:t==DataType.FLOAT32?new float[n]:new short[n];}
    private static Object[] outputs(DataType t,int n,int c){return new Object[]{array(t,n),array(t,c),array(t,c),array(t,c),array(t,c)};}
    private static Object[] mixedOutputs(Arena a,List<LayoutDescriptor>l,boolean f64){return f64?new Object[]{a.allocate(span(l.get(5))*8L,8),new double[span(l.get(6))],a.allocate(span(l.get(7))*8L,8),new double[span(l.get(8))],a.allocate(span(l.get(9))*8L,8)}:new Object[]{new float[span(l.get(5))],a.allocate(span(l.get(6))*4L,4),new float[span(l.get(7))],a.allocate(span(l.get(8))*4L,4),new float[span(l.get(9))]};}
    private static float value(int i){return(i%37-18)*.03125f+(i/2048%7-3)*.0078125f;}
    private static short bf(float v){int b=Float.floatToRawIntBits(v);if((b&0x7fffffff)>0x7f800000)return(short)0x7fc0;return(short)((b+0x7fff+(b>>>16&1))>>>16);}
    private static float decode(short v){return Float.intBitsToFloat(Short.toUnsignedInt(v)<<16);}
    private static void fill(DataType t,Object input,Object scale,Object bias,Object mean,Object variance){
        if(t==DataType.FLOAT64){double[]x=(double[])input,s=(double[])scale,b=(double[])bias,m=(double[])mean,v=(double[])variance;for(int i=0;i<x.length;i++)x[i]=value(i);for(int i=0;i<s.length;i++){s[i]=.75+(i%13)*.03125;b[i]=(i%11-5)*.015625;m[i]=(i%9-4)*.0625;v[i]=.5+(i%7)*.125;}}
        else if(t==DataType.FLOAT32){float[]x=(float[])input,s=(float[])scale,b=(float[])bias,m=(float[])mean,v=(float[])variance;for(int i=0;i<x.length;i++)x[i]=value(i);for(int i=0;i<s.length;i++){s[i]=.75f+(i%13)*.03125f;b[i]=(i%11-5)*.015625f;m[i]=(i%9-4)*.0625f;v[i]=.5f+(i%7)*.125f;}}
        else{short[]x=(short[])input,s=(short[])scale,b=(short[])bias,m=(short[])mean,v=(short[])variance;for(int i=0;i<x.length;i++)x[i]=bf(value(i));for(int i=0;i<s.length;i++){s[i]=bf(.75f+(i%13)*.03125f);b[i]=bf((i%11-5)*.015625f);m[i]=bf((i%9-4)*.0625f);v[i]=bf(.5f+(i%7)*.125f);}}}
    private static void fillMixed64(CpuBatchNormTrainingLowering.Geometry g,float[]x,MemorySegment s,short[]b,MemorySegment m,double[]v){var in=g.inputs();for(int a=0;a<16;a++)for(int c=0;c<32;c++)for(int z=0;z<64;z++)x[(int)(in.get(0).offset()+a*in.get(0).strides()[0]+c*in.get(0).strides()[1]+z*in.get(0).strides()[2])]=value((a*32+c)*64+z);for(int c=0;c<32;c++){s.set(DOUBLE,vecByte(in.get(1),c,8),.75+c*.015625);b[(int)vec(in.get(2),c)]=bf((c%7-3)*.03125f);m.set(FLOAT,vecByte(in.get(3),c,4),(c%9-4)*.0625f);v[(int)vec(in.get(4),c)]=.5+(c%5)*.125;}}
    private static void fillMixed32(CpuBatchNormTrainingLowering.Geometry g,MemorySegment x,float[]s,MemorySegment b,float[]m,short[]v){var in=g.inputs();for(int a=0;a<16;a++)for(int c=0;c<32;c++)for(int z=0;z<64;z++){long q=in.get(0).offset()+a*in.get(0).strides()[0]+c*in.get(0).strides()[1]+z*in.get(0).strides()[2];x.set(SHORT,q*2,bf(value((a*32+c)*64+z)));}for(int c=0;c<32;c++){s[(int)vec(in.get(1),c)]=.75f+c*.015625f;b.set(SHORT,vecByte(in.get(2),c,2),bf((c%7-3)*.03125f));m[(int)vec(in.get(3),c)]=(c%9-4)*.0625f;v[(int)vec(in.get(4),c)]=bf(.5f+(c%5)*.125f);}}

    private static void gate(BenchmarkCase c)throws Throwable{long g=c.generated.run(),d=c.direct.run();if(g!=d)throw new AssertionError(c.name+" checksum "+g+"/"+d);c.verify.run();}
    private static long checksum(DataType t,Object[]o){long h=0;for(Object x:o)h=Long.rotateLeft(h,7)^checksum(t,x);return h;}
    private static long checksum(DataType t,Object o){long h=0;if(o instanceof MemorySegment s){int w=t.byteWidth();for(long p=0;p<s.byteSize();p+=w){long b=t==DataType.FLOAT64?Double.doubleToRawLongBits(s.get(DOUBLE,p)):Integer.toUnsignedLong(Float.floatToRawIntBits(s.get(FLOAT,p)));h=Long.rotateLeft(h,1)^b;}}else if(t==DataType.FLOAT64)for(double v:(double[])o)h=Long.rotateLeft(h,1)^Double.doubleToRawLongBits(v);else if(t==DataType.FLOAT32)for(float v:(float[])o)h=Long.rotateLeft(h,1)^Integer.toUnsignedLong(Float.floatToRawIntBits(v));else for(short v:(short[])o)h=Long.rotateLeft(h,1)^Short.toUnsignedLong(v);return h;}
    private static void equal(DataType t,Object[]a,Object[]b){for(int i=0;i<5;i++){if(a[i]instanceof MemorySegment x){if(x.mismatch((MemorySegment)b[i])!=-1)throw new AssertionError("output "+i);}else if(t==DataType.FLOAT64){if(!Arrays.equals((double[])a[i],(double[])b[i]))throw new AssertionError("output "+i);}else if(t==DataType.FLOAT32){if(!Arrays.equals((float[])a[i],(float[])b[i]))throw new AssertionError("output "+i);}else if(!Arrays.equals((short[])a[i],(short[])b[i]))throw new AssertionError("output "+i);}}
    private static long time(BenchmarkAction a,int n)throws Throwable{long s=System.nanoTime(),v=0;for(int i=0;i<n;i++)v^=a.run();sink^=v;return System.nanoTime()-s;}
    private static int repetitions(BenchmarkAction a)throws Throwable{int n=1;while(time(a,n)<MIN_BATCH_NANOS)n=Math.multiplyExact(n,2);return n;}
    private static long median(long[]a){long[]c=a.clone();Arrays.sort(c);return c[4];}
    private static void environment(StringBuilder out){long total=Runtime.getRuntime().totalMemory(),max=Runtime.getRuntime().maxMemory();String java=System.getProperty("java.version");out.append("ENV,java=").append(java).append(",vm=").append(System.getProperty("java.vm.name")).append(",processors=").append(Runtime.getRuntime().availableProcessors()).append(",totalMemory=").append(total).append(",maxMemory=").append(max).append(",byteOrder=").append(ByteOrder.nativeOrder()).append('\n');long low=900L<<20,high=1100L<<20;if(!java.startsWith("26")||total<low||max<low||max>high)throw new AssertionError("requires Java 26 -Xms1g -Xmx1g");}
    private static void retain(boolean accepted,int fork,int attempt,String text)throws Exception{Path dir=EVIDENCE.resolve(accepted?"forks":"rejected-samples");Files.createDirectories(dir);String name=accepted?"fork-"+fork+".csv":"fork-"+fork+"-attempt-"+attempt+"-"+Instant.now().toEpochMilli()+".csv";Files.writeString(dir.resolve(name),text);}
    private static void aggregate()throws Exception{List<String>names=null;double[][]ratios=new double[5][12];for(int f=0;f<5;f++){List<String>ns=new ArrayList<>();int i=0;for(String line:Files.readAllLines(EVIDENCE.resolve("forks/fork-"+f+".csv")))if(line.startsWith("RESULT,")){String[]x=line.split(",",6);ns.add(x[1]);ratios[f][i++]=Double.parseDouble(x[4]);}if(i!=12)throw new AssertionError("fork "+f+" count "+i);if(names==null)names=List.copyOf(ns);else if(!names.equals(ns))throw new AssertionError("inventory");}StringBuilder s=new StringBuilder();int failures=0;for(int i=0;i<12;i++){double[]v=new double[5];for(int f=0;f<5;f++)v[f]=ratios[f][i];Arrays.sort(v);if(v[2]>1.15)failures++;s.append(String.format(Locale.ROOT,"AGGREGATE,%s,%.9f,%s%n",names.get(i),v[2],Arrays.toString(v)));}long rejected=Files.exists(EVIDENCE.resolve("rejected-samples"))?Files.list(EVIDENCE.resolve("rejected-samples")).count():0;s.append("COUNTS,accepted=5,rejected=").append(rejected).append('\n');Files.writeString(EVIDENCE.resolve("summary.csv"),s);System.out.print(s);if(failures!=0)throw new AssertionError("aggregate failures "+failures);}
    private static String clean(String s){return s==null?"":s.replace('\n',' ').replace('\r',' ').replace(',',';');}
}
