package io.github.pho001.synaptik.engine;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.prepare.*;
import java.lang.management.ManagementFactory;
import java.util.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/** Serial end-to-end comparison with equal fresh-run/result/materialization lifetimes. */
@EnabledIfSystemProperty(named="synaptik.cpu.dynamicBatch.performance",matches="true")
class EngineOrdinaryDynamicBatchTimingTest {
    private static volatile int sink;
    @Test void staticVersusDynamicScalarPortableFullActualBackward() {
        long start=System.nanoTime();
        System.out.println("dynamic-batch Engine JVM="+System.getProperty("java.version")+" options="
                +ManagementFactory.getRuntimeMXBean().getInputArguments()+" forks=1 samples=3 serial; F32 segments, scalar single range, no native route");
        for(int[] c:new int[][]{{1,4,3},{40,4,3},{64,4,3},{64,128,1},{64,128,128}}) {
            int b=c[0],f=c[1],h=c[2];
            try(Engine engine=scalarEngine()) {
                Tensor dynamicX=TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,
                        Shape.ofDimensions(new DynamicDimension("B"),new StaticDimension(f)),Optional.empty(),true));
                Tensor staticX=TensorFactory.create(EngineOrdinaryDynamicBatchTest.descriptor(Shape.of(b,f),true));
                Tensor w=TensorFactory.create(EngineOrdinaryDynamicBatchTest.descriptor(Shape.of(f,h),true));
                Tensor seed=EngineOrdinaryDynamicBatchTest.ones(Shape.scalar(),false);
                long cold=System.nanoTime();
                var dc=engine.compile(List.of(dynamicX.matmul(w).mean()),List.of(seed),List.of(dynamicX,w));
                var sc=engine.compile(List.of(staticX.matmul(w).mean()),List.of(seed),List.of(staticX,w));
                try(var dp=engine.prepare(dc);var sp=engine.prepare(sc)) {
                    var dynamicRealizations=realizations(dp);var staticRealizations=realizations(sp);
                    boolean matched=!(f==128&&h==128);
                    assertEquals(3,dynamicRealizations.size());assertEquals(3,staticRealizations.size());
                    if(matched) assertEquals(dynamicRealizations,staticRealizations,"matched rows require equal actual MATMUL realizations");
                    else assertNotEquals(dynamicRealizations,staticRealizations,"retained medium route-policy diagnostic");
                    System.out.println((matched?"MATCHED algorithms":"UNMATCHED route-policy diagnostic, not B-overhead")
                            +" dynamic="+dynamicRealizations+" static="+staticRealizations);
                    System.out.printf(Locale.ROOT,"cold static+dynamic compile/prepare B%d F%d H%d %.3fms; buffers dynamic/static %d/%d workspaces %d/%d (actual topology cost retained)%n",
                            b,f,h,(System.nanoTime()-cold)/1e6,dp.execution().memoryPlan().buffers().size(),sp.execution().memoryPlan().buffers().size(),
                            dp.execution().memoryPlan().workspaces().size(),sp.execution().memoryPlan().workspaces().size());
                    Tensor xValue=EngineOrdinaryDynamicBatchTest.ones(Shape.of(b,f),true),wValue=EngineOrdinaryDynamicBatchTest.ones(Shape.of(f,h),true);
                    var di=dc.inputs().stream().map(i->i.tensorId().equals(dynamicX.id())?xValue:i.tensorId().equals(w.id())?wValue:seed).toList();
                    var si=sc.inputs().stream().map(i->i.tensorId().equals(staticX.id())?xValue:i.tensorId().equals(w.id())?wValue:seed).toList();
                    try(var dr=engine.runValues(dp,di);var sr=engine.runValues(sp,si)) {
                        for(int i=0;i<3;i++) assertEquals(dr.materialize(dr.publications().get(i),1000000).bytes(),
                                sr.materialize(sr.publications().get(i),1000000).bytes());
                    }
                    Runnable dynamic=()->run(engine,dp,di),fixed=()->run(engine,sp,si);
                    sample(dynamic,60_000_000);sample(fixed,60_000_000);
                    double[] d=new double[3],s=new double[3];
                    for(int i=0;i<3;i++) { if((i&1)==0){d[i]=sample(dynamic,45_000_000);s[i]=sample(fixed,45_000_000);}
                        else{s[i]=sample(fixed,45_000_000);d[i]=sample(dynamic,45_000_000);} }
                    Arrays.sort(d);Arrays.sort(s);
                    System.out.printf(Locale.ROOT,"Engine %s B%d F%d H%d dynamic/static ns median %.1f/%.1f ratio %.3f ranges [%.1f,%.1f]/[%.1f,%.1f]; each run materializes loss,dX,dW then closes result; setup/allocation topology differences included, not isolated B-propagation overhead%n",
                            matched?"MATCHED":"UNMATCHED",b,f,h,d[1],s[1],d[1]/s[1],d[0],d[2],s[0],s[2]);
                }
            }
        }
        System.out.printf(Locale.ROOT,"Engine bounded total %.3fs sink=%d%n",(System.nanoTime()-start)/1e9,sink);
        assertTrue(System.nanoTime()-start<30_000_000_000L);
    }
    private static List<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Realization> realizations(PreparedExecution p) {
        var result=new ArrayList<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Realization>();
        for(var step:p.execution().schedule().steps()) if(step instanceof io.github.pho001.synaptik.runtime.schedule.PreparedSchedule.ExecutionStep execution)
            collect(execution.executable(),result);
        return List.copyOf(result);
    }
    private static void collect(io.github.pho001.synaptik.runtime.execution.PreparedExecutable executable,
            List<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Realization> result) {
        if(executable instanceof io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedPartitionExecutable partition)
            partition.children().forEach(child->collect(child,result));
        else if(executable instanceof io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutable cpu)
            cpu.artifact().specialization().matmulIr().ifPresent(ir->result.add(ir.realization()));
    }
    private static Engine scalarEngine() {
        var cpu=CpuBackendIntegration.open();var composition=new CpuEngineBackendComposition(cpu);
        // Existing package-private test seam fixes both paths to the same public CPU scalar
        // analysis inputs, with OpenBLAS/tuning disabled. No production route switch is added.
        var preparation=new PartitionPreparation<>(CpuPartitionAnalysisInputs.DEFAULT,new CpuPartitionPreparer(),new CpuPartitionFinalizer());
        return new Engine(new AdvancedEngine(composition,artifacts->GraphPreparation.prepare(artifacts,
                Collections.nCopies(artifacts.partitions().size(),preparation),cpu.scheduleAssembler())));
    }
    private static void run(Engine engine,PreparedExecution p,List<Tensor> inputs) {
        try(var result=engine.runValues(p,inputs)) {
            for(var publication:result.publications()) sink^=result.materialize(publication,1000000).bytes().hashCode();
        }
    }
    private static double sample(Runnable work,long duration) {
        long start=System.nanoTime(),now=start,count=0;
        do{work.run();count++;now=System.nanoTime();}while(now-start<duration);
        return (double)(now-start)/count;
    }
}
