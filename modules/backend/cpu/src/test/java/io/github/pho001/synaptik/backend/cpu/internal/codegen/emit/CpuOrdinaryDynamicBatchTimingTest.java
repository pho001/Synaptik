package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.lang.management.ManagementFactory;
import java.util.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/** Bounded single-JVM direct-entry comparison, not a throughput guarantee or benchmark matrix. */
@EnabledIfSystemProperty(named="synaptik.cpu.dynamicBatch.performance",matches="true")
class CpuOrdinaryDynamicBatchTimingTest {
    private static volatile int sink;
    @Test void generatedVersusFixedFactCleanJavaIncludingExactMean() throws Throwable {
        long launched=System.nanoTime();
        System.out.println("dynamic-batch kernel JVM="+System.getProperty("java.version")+" options="
                +ManagementFactory.getRuntimeMXBean().getInputArguments()+" forks=1 samples=3 serial, native F32 segments; output lifetime reused equally");
        for(int[] size:new int[][]{{4,3},{128,128}}) {
            int f=size[0],h=size[1];
            long cold=System.nanoTime();
            var routes=CpuOrdinaryDynamicBatchTest.routes(f,h).stream().filter(r->r.batchMatmul().isPresent()).toList();
            var entries=new ArrayList<MethodHandle>();
            for(var route:routes) entries.add(new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(), route.kernelIr(), CpuOrdinaryDynamicBatchTest.bytes(route)).entryPoint());
            System.out.printf(Locale.ROOT,"cold compile/lower/generate/define F%d H%d %.3fms (untimed in ratios)%n",f,h,(System.nanoTime()-cold)/1e6);
            for(int b:f==4?new int[]{1,40,64}:new int[]{64}) try(Arena arena=Arena.ofConfined()) {
                MemorySegment x=arena.allocate(4L*b*f,4),w=arena.allocate(4L*f*h,4),dy=arena.allocate(4L*b*h,4);
                fill(x,1);fill(w,1);fill(dy,1f/(b*h));
                for(int index=0;index<routes.size();index++) {
                    var route=routes.get(index); var recipe=route.batchMatmul().orElseThrow();var g=recipe.bind(b);
                    int family=recipe.left().extentBatchFactors().get(1)==1?2
                            :recipe.right().extentConstants().get(1)==f&&recipe.right().strideConstants().get(0)==1?1:0;
                    MemorySegment left=family==1?dy:x,right=family==0?w:family==1?w:dy;
                    MemorySegment actual=arena.allocate(4*g.outputCount(),4),expected=arena.allocate(4*g.outputCount(),4);
                    long[] geometry=g.pack(0,0,0);MethodHandle entry=entries.get(index);
                    Runnable generated=()->invoke(entry,left,right,actual,geometry,g.outputCount());
                    Runnable direct=()->CpuOrdinaryDynamicBatchTest.oracle(f,h,family,x,w,dy,expected,b);
                    generated.run();direct.run();assertEquals(-1,actual.mismatch(expected));
                    measure("B"+b+" F"+f+" H"+h+" matrix"+family,generated,direct);
                    sink^=Float.floatToRawIntBits(actual.get(java.lang.foreign.ValueLayout.JAVA_FLOAT,0));
                }
                CpuPortableRoutePlan mean=CpuOrdinaryDynamicBatchTest.aggregate(new Operation(AggregateReductionKind.MEAN,
                        NoOperationAttrs.INSTANCE),CpuOrdinaryDynamicBatchTest.symbolic(h),Shape.scalar());
                var entry=new CpuClassFileKernelGenerator().defineClassBytes(mean.specialization(), mean.kernelIr(), CpuOrdinaryDynamicBatchTest.bytes(mean)).entryPoint();
                var g=mean.batchAggregate().orElseThrow().bind(b);long[] geometry=g.pack(new long[2]);
                MemorySegment a=arena.allocate(4,4),e=arena.allocate(4,4),scratch=arena.allocate(48,8),oracleScratch=arena.allocate(48,8);
                Runnable generated=()->invokeAggregate(entry,dy,a,scratch,geometry);
                Runnable direct=()->CpuOrdinaryDynamicBatchCleanJavaOracle.reduce(dy,e,oracleScratch,1,b*h,0,1,true);
                generated.run();direct.run();assertEquals(-1,a.mismatch(e));
                measure("B"+b+" F"+f+" H"+h+" exactMean-five-limb",generated,direct);
            }
        }
        System.out.printf(Locale.ROOT,"kernel bounded total %.3fs sink=%d%n",(System.nanoTime()-launched)/1e9,sink);
        assertTrue(System.nanoTime()-launched<30_000_000_000L);
    }
    static void measure(String label,Runnable generated,Runnable direct) {
        sample(generated,60_000_000);sample(direct,60_000_000);
        double[] g=new double[3],d=new double[3];
        for(int i=0;i<3;i++) { if((i&1)==0){g[i]=sample(generated,45_000_000);d[i]=sample(direct,45_000_000);}
            else{d[i]=sample(direct,45_000_000);g[i]=sample(generated,45_000_000);} }
        Arrays.sort(g);Arrays.sort(d);
        System.out.printf(Locale.ROOT,"%s generated/direct ns median %.1f/%.1f ratio %.3f ranges [%.1f,%.1f]/[%.1f,%.1f]%n",
                label,g[1],d[1],g[1]/d[1],g[0],g[2],d[0],d[2]);
    }
    private static double sample(Runnable work,long duration) {
        long start=System.nanoTime(),now=start,count=0;
        do {work.run();count++;now=System.nanoTime();}while(now-start<duration);
        return (double)(now-start)/count;
    }
    static void fill(MemorySegment s,float value) { for(long i=0;i<s.byteSize()/4;i++)s.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT,i,value); }
    private static void invoke(MethodHandle entry,MemorySegment a,MemorySegment b,MemorySegment c,long[] geometry,long end) {
        try{entry.invokeExact(a,b,c,geometry,0L,end);}catch(Throwable failure){throw new AssertionError(failure);}
    }
    private static void invokeAggregate(MethodHandle entry,MemorySegment a,MemorySegment b,MemorySegment scratch,long[] geometry) {
        try{entry.invokeExact(a,b,scratch,geometry,0L,1L);}catch(Throwable failure){throw new AssertionError(failure);}
    }
}
