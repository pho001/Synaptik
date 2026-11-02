package io.github.pho001.synaptik.engine;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.prepare.*;
import io.github.pho001.synaptik.runtime.run.RunMemoryGeometry;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.lang.management.ManagementFactory;
import java.util.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/** Fixed bounded full-admission and fresh-run probe; no memo, adaptive iteration or parity claim. */
@EnabledIfSystemProperty(named="synaptik.cpu.dynamicBatch.performance",matches="true")
class EngineSymbolicScanTimingTest {
    private static volatile long sink;
    @Test void fullBindingAndFreshEngineLifetimes() {
        long start=System.nanoTime(),deadline=start+30_000_000_000L;
        System.out.println("symbolic Scan Engine JVM="+System.getProperty("java.version")+" options="
                +ManagementFactory.getRuntimeMXBean().getInputArguments()+" forks=1 warmup=100 samples=3x300 alternating pairs cap=30s including cold");
        for (int[] c : new int[][] {{3,1,4,3},{3,40,4,3},{3,64,4,3},{4,64,32,32}}) {
            if (System.nanoTime()>=deadline) { System.out.println("CENSORED remaining Engine cases"); break; }
            try (var engine=scalarEngine()) {
                long cold=System.nanoTime(); var symbolic=compile(engine,c,true); var fixed=compile(engine,c,false);
                try (var dp=engine.prepare(symbolic.graph); var sp=engine.prepare(fixed.graph)) {
                    System.out.printf(Locale.ROOT,"cold BOTH compile/prepare T%d B%d F%d H%d %.3fms; scalar F32 native run-owned resources; dynamic/static buffers=%d/%d workspaces=%d/%d%n",
                            c[0],c[1],c[2],c[3],(System.nanoTime()-cold)/1e6,dp.execution().memoryPlan().buffers().size(),sp.execution().memoryPlan().buffers().size(),
                            dp.execution().memoryPlan().workspaces().size(),sp.execution().memoryPlan().workspaces().size());
                    try (var dr=engine.runValues(dp,symbolic.inputs); var sr=engine.runValues(sp,fixed.inputs)) {
                        assertEquals(dr.resultCount(),sr.resultCount());
                        for (int p=0;p<dr.resultCount();p++) assertEquals(dr.materialize(dr.publications().get(p),10_000_000).bytes(),sr.materialize(sr.publications().get(p),10_000_000).bytes());
                    }
                    measure("BIND symbolic full semantic/scoped/numeric/aggregate/current-storage admission (both independent identical calls) "+Arrays.toString(c),
                            ()->bind(dp,symbolic.inputs),()->bind(dp,symbolic.inputs),deadline);
                    measure("Engine symbolic/static fresh run + all five materializations + result close "+Arrays.toString(c),
                            ()->run(engine,dp,symbolic.inputs),()->run(engine,sp,fixed.inputs),deadline);
                }
            }
        }
        System.out.printf(Locale.ROOT,"Engine total %.3fs sink=%d; allocation topology included, no parity/B-overhead claim%n",(System.nanoTime()-start)/1e9,sink);
    }
    /** Same scalar test seam as the retained ordinary probe; no native policy switch. */
    private static Engine scalarEngine() {
        var cpu=CpuBackendIntegration.open(); var composition=new CpuEngineBackendComposition(cpu);
        var preparation=new PartitionPreparation<>(CpuPartitionAnalysisInputs.DEFAULT,new CpuPartitionPreparer(),new CpuPartitionFinalizer());
        return new Engine(new AdvancedEngine(composition,a->GraphPreparation.prepare(a,Collections.nCopies(a.partitions().size(),preparation),cpu.scheduleAssembler())));
    }
    private record Case(CompiledGraph graph,List<Tensor> inputs) { }
    private static Case compile(Engine engine,int[] c,boolean dynamic) {
        int t=c[0],b=c[1],f=c[2],h=c[3]; Dimension batch=dynamic?new DynamicDimension("B"):new StaticDimension(b);
        Tensor sh=leaf(true,batch,new StaticDimension(h)), sx=leaf(true,batch,new StaticDimension(f)), sw=leaf(true,new StaticDimension(f),new StaticDimension(h));
        Tensor next=sx.matmul(sw).add(sh).tanh();
        Tensor initial=leaf(true,batch,new StaticDimension(h)), sequence=leaf(true,new StaticDimension(t),batch,new StaticDimension(f)), weight=leaf(true,new StaticDimension(f),new StaticDimension(h));
        var scan=Scan.apply(List.of(initial),List.of(sequence),List.of(weight),new ScanBody(List.of(sh),List.of(sx),List.of(sw),List.of(next),List.of(next)),false);
        Tensor fs=leaf(false,batch,new StaticDimension(h)), es=leaf(false,new StaticDimension(t),batch,new StaticDimension(h));
        var graph=engine.compile(List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst()),List.of(fs,es),List.of(initial,sequence,weight));
        var values=Map.of(initial.id(),value(Shape.of(b,h),true,.01f),sequence.id(),value(Shape.of(t,b,f),true,.02f),weight.id(),value(Shape.of(f,h),true,.03f),
                fs.id(),value(Shape.of(b,h),false,.1f),es.id(),value(Shape.of(t,b,h),false,.05f));
        return new Case(graph,graph.inputs().stream().map(i->values.get(i.tensorId())).toList());
    }
    /** Mirrors the complete existing pre-borrow admission, without acquiring a lease or storage. */
    static void bind(PreparedExecution p,List<Tensor> inputs) {
        var binding=p.batchPlan().bind(inputs.stream().map(Tensor::descriptor).toList());
        p.validateBatchGeometry(binding.batchSize()); var plan=p.execution().memoryPlan();
        var geometry=RunMemoryGeometry.resolve(plan,OptionalLong.of(binding.batchSize()));
        var descriptors=p.publicationDescriptors(binding); long total=0;
        long[] prefixes=new long[inputs.size()];
        for (int i=0;i<inputs.size();i++) prefixes[i]=bytes(inputs.get(i).descriptor());
        for (var descriptor:descriptors) bytes(descriptor);
        var creation=(PreparedSchedule.RepresentationCreationStep)p.execution().schedule().steps().getFirst();
        for (int i=0;i<plan.buffers().size();i++) total=Math.addExact(total,Math.multiplyExact(Math.addExact(geometry.bufferByteSize(i),plan.buffers().get(i).byteAlignment()-1),creation.representationPlan().bufferPreparations().get(i).size()));
        for (int i=0;i<plan.workspaces().size();i++) total=Math.addExact(total,Math.addExact(geometry.workspaceByteSize(i),plan.workspaces().get(i).byteAlignment()-1));
        var storage=inputs.stream().map(i->i.hostStorage().orElseThrow()).toList();
        for (int i=0;i<storage.size();i++) {
            var s=storage.get(i); if (s.dataType()!=inputs.get(i).descriptor().dataType() || !s.isAlive() || !s.segment().isAccessibleBy(Thread.currentThread()) || s.byteSize()<prefixes[i]) throw new AssertionError("invalid storage");
        }
        sink^=total;
    }
    private static long bytes(TensorDescriptor d) {
        long suffix=1; for (int axis=d.shape().rank()-1;axis>=0;axis--) suffix=Math.toIntExact(Math.multiplyExact(suffix,Math.toIntExact(d.shape().dimension(axis).staticSize().orElseThrow())));
        var layout=d.layout().orElseGet(()->LayoutDescriptor.contiguous(d.shape())); Math.toIntExact(layout.storageOffset());
        for (long stride:layout.strides()) Math.toIntExact(stride); Math.toIntExact(layout.referencedElementSpan());
        return Math.multiplyExact(layout.referencedElementSpan(),d.dataType().byteWidth());
    }
    private static Tensor leaf(boolean grad,Dimension... d) {
        var shape=Shape.ofDimensions(d); return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,shape,shape.isFullyStatic()?Optional.of(LayoutDescriptor.contiguous(shape)):Optional.empty(),grad));
    }
    private static Tensor value(Shape shape,boolean grad,float scale) { return EngineSymbolicScanTest.stored(grad,shape,EngineSymbolicScanTest.data(Math.toIntExact(shape.knownElementCount().orElseThrow()),scale)); }
    private static void run(Engine e,PreparedExecution p,List<Tensor> inputs) { try (var r=e.runValues(p,inputs)) { for (var publication:r.publications()) sink^=r.materialize(publication,10_000_000).bytes().hashCode(); } }
    private static void measure(String label,Runnable a,Runnable b,long deadline) {
        for (int i=0;i<100;i++) { if (System.nanoTime()>=deadline) { System.out.println(label+" CENSORED warmup"); return; } a.run(); b.run(); }
        double[] x=new double[3],y=new double[3];
        for (int i=0;i<3;i++) {
            if ((i&1)==0) { x[i]=sample(a,deadline); y[i]=sample(b,deadline); } else { y[i]=sample(b,deadline); x[i]=sample(a,deadline); }
            if (Double.isNaN(x[i]) || Double.isNaN(y[i])) { System.out.println(label+" CENSORED sample="+i); return; }
        }
        System.out.println(label+" A ns="+Arrays.toString(x)+" B ns="+Arrays.toString(y));
    }
    private static double sample(Runnable work,long deadline) { long start=System.nanoTime(); for (int i=0;i<300;i++) { if (System.nanoTime()>=deadline) return Double.NaN; work.run(); } return (System.nanoTime()-start)/300.0; }
}
