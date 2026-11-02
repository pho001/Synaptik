package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.contract.*;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.compiler.*;
import io.github.pho001.synaptik.config.compile.*;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.lang.foreign.*;
import java.lang.management.ManagementFactory;
import java.util.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/** Fixed-count production bound-invocation observations against the same clean-Java numerical oracle. */
@EnabledIfSystemProperty(named="synaptik.cpu.dynamicBatch.performance",matches="true")
class CpuSymbolicScanTimingTest {
    @Test void boundedForwardAndActualReverseAgainstDirectJava() {
        long start=System.nanoTime(), deadline=start+30_000_000_000L;
        System.out.println("symbolic Scan JVM="+System.getProperty("java.version")+" options="
                +ManagementFactory.getRuntimeMXBean().getInputArguments()
                +" forks=1 bodyCap=30s startupPairs=20000/3000 at B64 warmPairs=1000 per case"
                +" samples=3x300 per variant orders=G-D,D-G,G-D; F32 scalar native"
                +" post-fixed-warmup observations; JIT readiness unverified/transitional without log analysis"
                +" generated=production bound invocation direct=one specialized Java call; not pure-loop/Engine");
        for (int[] dims:new int[][] {{3,4,3},{4,32,32}}) {
            String label="T"+dims[0]+"F"+dims[1]+"H"+dims[2];
            if (capped(label+" cold not started",start,deadline)) break;
            boundary(label+" cold compile/lower/generate/define BEGIN",start);
            long cold=System.nanoTime();
            var family=new PreparedFamily(dims[0],dims[1],dims[2],false);
            boundary(label+" cold END elapsedMs="+(System.nanoTime()-cold)/1e6+" compilations=1",start);
            if (capped(label+" noninterruptible cold setup overshoot",start,deadline)) break;
            try (var fixture=bind(family,64,label+" startup",start)) {
                if (capped(label+" noninterruptible startup allocation/binding overshoot",start,deadline)) break;
                Runnable direct=fixture.timedDirect();
                fixture.verifyWith(direct);
                if (capped(label+" startup binding/check overshoot",start,deadline)) break;
                boolean complete=pairs(label+" STARTUP B64",fixture::generated,direct,dims[0]==3?20_000:3_000,start,deadline);
                fixture.assertMatched();
                boundary(label+" startup after-check all5",start);
                if (!complete || capped(label+" startup after-check overshoot",start,deadline)) break;
            }
            for (int batch:dims[0]==3?new int[]{1,40,64}:new int[]{64}) {
                String name=label+" B"+batch;
                if (capped(name+" case not started",start,deadline)) break;
                try (var fixture=bind(family,batch,name,start)) {
                    if (capped(name+" noninterruptible allocation/binding overshoot",start,deadline)) break;
                    Runnable direct=fixture.timedDirect();
                    fixture.verifyWith(direct);
                    boundary(name+" before-check all5+public-buffer separation",start);
                    if (capped(name+" binding/check overshoot",start,deadline)) break;
                    boolean complete=measure(name,fixture::generated,direct,start,deadline);
                    // Read-only post-check even on censorship; no post-cap kernel call.
                    fixture.assertMatched();
                    boundary(name+" after-check all5+public-buffer separation",start);
                    if (!complete || capped(name+" after-check overshoot",start,deadline)) break;
                    if (!measureBinding(name,fixture,start,deadline)) break;
                }
            }
        }
        boundary("BODY END elapsedSeconds="+(System.nanoTime()-start)/1e9
                +"; setup/check/call may be noninterruptible; no parity/B-overhead/steady-state guarantee",start);
    }

    /**
     * Allocates fresh invocation state only, reporting cost separately from kernel windows.
     * @param family non-null immutable method/metadata owner, borrowed
     * @param batch non-negative invocation B
     * @param label case name
     * @param start monotonic body origin, nanoseconds
     * @return new native fixture which the caller must close
     */
    private static Fixture bind(PreparedFamily family,int batch,String label,long start) {
        boundary(label+" allocation/binding BEGIN",start);
        long began=System.nanoTime();
        var fixture=new Fixture(family,batch);
        boundary(label+" allocation/binding END elapsedMs="+(System.nanoTime()-began)/1e6
                +" oracleAllocatedBytes="+fixture.capacity+" mandatoryReadBytes="+fixture.readBytes
                +" publicWriteBytes="+fixture.generatedWriteBytes+" directWriteBytes="+fixture.directWriteBytes
                +"; production preparation and binding included",start);
        return fixture;
    }

    /**
     * Executes fixed alternating-order warmup, with honest partial call counts on censorship.
     * @param label phase/case name
     * @param generated complete forward/reverse generated invocation
     * @param direct complete specialized Java invocation
     * @param count fixed pair count, never adaptively extended
     * @param start monotonic body origin, nanoseconds
     * @param deadline common body deadline, nanoseconds
     * @return whether all pairs finished before the cap
     */
    private static boolean pairs(String label,Runnable generated,Runnable direct,int count,long start,long deadline) {
        boundary(label+" BEGIN requestedPairs="+count,start);
        int g=0,d=0;
        outer: for (int i=0;i<count;i++) for (int position=0;position<2;position++) {
            if (System.nanoTime()>=deadline) break outer;
            if (((i+position)&1)==0) { generated.run(); g++; } else { direct.run(); d++; }
        }
        boolean complete=g==count && d==count && System.nanoTime()<deadline;
        boundary(label+" END generatedCalls="+g+" directCalls="+d
                +(complete?" COMPLETE (counts are not compilation readiness)":" CENSORED"),start);
        return complete;
    }

    /**
     * Observes three ordered pairs after fixed warmup; unfinished pairs have no paired result.
     * @param label case name
     * @param generated complete generated invocation
     * @param direct complete specialized Java invocation
     * @param start monotonic body origin, nanoseconds
     * @param deadline common body deadline, nanoseconds
     * @return whether all warmup and sample calls completed before the cap
     */
    private static boolean measure(String label,Runnable generated,Runnable direct,long start,long deadline) {
        if (!pairs(label+" WARM",generated,direct,1_000,start,deadline)) return false;
        for (int i=0;i<3;i++) {
            boolean gFirst=(i&1)==0;
            String pair=label+" SAMPLE pair="+i+" order="+(gFirst?"G-D":"D-G");
            if (!sample(pair+(gFirst?" generated":" direct"),gFirst?generated:direct,start,deadline)
                    || !sample(pair+(gFirst?" direct":" generated"),gFirst?direct:generated,start,deadline)) {
                boundary(pair+" CENSORED pair/remaining windows (not a complete paired sample)",start);
                return false;
            }
        }
        return true;
    }

    /**
     * Measures 300 calls with per-call cap checks; partial/late windows report no sample mean.
     * @param label variant and ordered window
     * @param work non-null complete invocation
     * @param start monotonic body origin, nanoseconds
     * @param deadline common body deadline, nanoseconds
     * @return whether all 300 calls completed before the cap
     */
    private static boolean sample(String label,Runnable work,long start,long deadline) {
        if (capped(label+" window not started",start,deadline)) return false;
        boundary(label+" WINDOW BEGIN requestedCalls=300",start);
        long began=System.nanoTime(); int calls=0;
        for (;calls<300 && System.nanoTime()<deadline;calls++) work.run();
        long end=System.nanoTime(); boolean complete=calls==300 && end<deadline;
        boundary(label+" WINDOW END calls="+calls+(complete?" meanNs="+(end-began)/300.0
                :" CENSORED elapsedNs="+(end-began)+" no sample mean"),start);
        return complete;
    }

    /**
     * Observes production invocation binding separately from execution.
     * @param label case name
     * @param fixture live invocation providing B and immutable recipes
     * @param start monotonic body origin, nanoseconds
     * @param deadline the same deadline used for setup/kernels
     * @return whether 100 warmup calls and three 300-call windows completed before the cap
     */
    private static boolean measureBinding(String label,Fixture fixture,long start,long deadline) {
        label+=" PRODUCTION BIND (execution excluded)";
        if (capped(label+" omitted",start,deadline)) return false;
        Runnable pack=fixture.consumer::bind;
        boundary(label+" WARM BEGIN requestedCalls=100",start);
        int calls=0; for (;calls<100 && System.nanoTime()<deadline;calls++) pack.run();
        boolean complete=calls==100 && System.nanoTime()<deadline;
        boundary(label+" WARM END calls="+calls+(complete?" COMPLETE":" CENSORED"),start);
        if (!complete) return false;
        for (int i=0;i<3;i++) if (!sample(label+" sample="+i,pack,start,deadline)) return false;
        return true;
    }

    /**
     * Logs outside inner loops, with JVM uptime to correlate compilation-log stamps.
     * @param label boundary description
     * @param start monotonic body origin, nanoseconds
     */
    private static void boundary(String label,long start) {
        System.out.printf(Locale.ROOT,"bodyMs=%.3f uptimeMs=%d %s%n",(System.nanoTime()-start)/1e6,
                ManagementFactory.getRuntimeMXBean().getUptime(),label);
    }

    /**
     * Reports unfinished work/overshoot without a retry or partial successful sample.
     * @param label phase description
     * @param start monotonic body origin, nanoseconds
     * @param deadline common monotonic body deadline, nanoseconds
     * @return whether the deadline has been reached
     */
    private static boolean capped(String label,long start,long deadline) {
        if (System.nanoTime()<deadline) return false;
        boundary("CENSORED "+label+"; remaining phases/cases omitted",start); return true;
    }

    /** Immutable compiled graph; invocation storage remains owned by the production consumer. */
    static final class PreparedFamily {
        final int t,f,h;
        final boolean sigmoid;
        final CompileArtifacts compiled;
        final List<Tensor> inputs;

        /**
         * Compiles the complete forward and actual reverse graph once. Only T=0 requests
         * disconnected sequence/parameter zeros; every positive-T request retains ERROR.
         * @param t fixed non-negative time extent
         * @param f fixed input width
         * @param h fixed carry width
         * @param sigmoid whether the generic family uses SIGMOID instead of TANH
         */
        PreparedFamily(int t,int f,int h,boolean sigmoid) {
            this.t=t; this.f=f; this.h=h; this.sigmoid=sigmoid;
            var batch=new DynamicDimension("B");
            Tensor sh=symbolic(true,batch,new StaticDimension(h)), sx=symbolic(true,batch,new StaticDimension(f)), sw=fixed(true,f,h);
            Tensor sum=sx.matmul(sw).add(sh), next=sigmoid?sum.sigmoid():sum.tanh();
            Tensor ih=symbolic(true,batch,new StaticDimension(h)), seq=symbolic(true,new StaticDimension(t),batch,new StaticDimension(f)), weight=fixed(true,f,h);
            var scan=Scan.apply(List.of(ih),List.of(seq),List.of(weight),new ScanBody(List.of(sh),List.of(sx),List.of(sw),List.of(next),List.of(next)),false);
            Tensor finalSeed=symbolic(false,batch,new StaticDimension(h)), emissionSeed=symbolic(false,new StaticDimension(t),batch,new StaticDimension(h));
            inputs=List.of(ih,seq,weight,finalSeed,emissionSeed);
            var roots=List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
            var stage=new FunctionalGradientRequest.Stage(roots.stream().map(r->(FunctionalGradientRequest.OutputReference)new FunctionalGradientRequest.ForwardTensorReference(r)).toList(),
                    List.of(Optional.of(finalSeed),Optional.of(emissionSeed)),List.of(ih,seq,weight),false,
                    t == 0 ? FunctionalGradientRequest.DisconnectedPolicy.ZERO : FunctionalGradientRequest.DisconnectedPolicy.ERROR);
            var cpu=new CpuCapabilityProvider(); var backend=cpu.backendId();
            compiled=GraphCompilationPort.compile(CompileMode.FORWARD_AND_BACKWARD,roots,Optional.of(new FunctionalGradientRequest(List.of(stage))),
                    GraphOptimizationConfig.disabled(),BackendIntent.unconstrained(),PartitionScoringConfig.neutral(),List.of(cpu),
                    List.of(new BackendAvailabilitySnapshot(backend,Map.of(new BackendDeviceId(backend,"cpu"),DeviceClass.CPU))));
        }

    }

    /** Production assigned-slot invocation and independent clean-Java reference storage. */
    static final class Fixture implements AutoCloseable {
        final PreparedFamily family;
        final int t,b,f,h;
        final boolean sigmoid;
        final Arena arena;
        final io.github.pho001.synaptik.backend.cpu.internal.executable.CpuTypedEmptyScanConsumerTest.Consumer consumer;
        final List<MemorySegment> actual=new ArrayList<>(), expected=new ArrayList<>(), allocations=new ArrayList<>();
        final MemorySegment initial,x,w,fs,es,carry,history,savedHistory,dh,dx,dw;
        long capacity,readBytes,generatedWriteBytes,directWriteBytes;

        /**
         * Preserves generic TANH fixture construction with a private fresh family.
         * @param t fixed non-negative time
         * @param b non-negative runtime batch
         * @param f fixed positive input width
         * @param h fixed positive carry width
         */
        Fixture(int t,int b,int f,int h) { this(t,b,f,h,false); }

        /**
         * Preserves arbitrary generic dimensions and both activation consumers.
         * @param t fixed non-negative time
         * @param b non-negative runtime batch
         * @param f fixed positive input width
         * @param h fixed positive carry width
         * @param sigmoid SIGMOID rather than TANH
         */
        Fixture(int t,int b,int f,int h,boolean sigmoid) { this(new PreparedFamily(t,f,h,sigmoid),b); }

        /**
         * Prepares and binds the complete compiled graph with fresh production-owned storage.
         * @param family immutable compiled owner, borrowed beyond fixture closure
         * @param b non-negative invocation batch
         * @throws IllegalArgumentException if B is negative
         */
        Fixture(PreparedFamily family,int b) {
            if (b<0) throw new IllegalArgumentException("negative B");
            this.family=family; t=family.t; this.b=b; f=family.f; h=family.h; sigmoid=family.sigmoid;
            consumer=new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuTypedEmptyScanConsumerTest.Consumer(family.compiled,OptionalLong.of(b));
            arena=Arena.ofConfined();
            try {
                initial=consumer.inputStorage(family.inputs.get(0));
                // Empty-time primals are reference-only inputs, not invented prepared slots.
                x=t == 0 ? allocate(0) : consumer.inputStorage(family.inputs.get(1));
                w=t == 0 ? allocate(4L*f*h) : consumer.inputStorage(family.inputs.get(2));
                fs=consumer.inputStorage(family.inputs.get(3));
                es=t == 0 ? allocate(0) : consumer.inputStorage(family.inputs.get(4));
                fill(initial,.01f); fill(x,.02f); fill(w,.03f); fill(fs,.1f); fill(es,.05f);
                var publications=new ArrayList<ValueId>();
                family.compiled.publication().forwardBindings().forEach(binding -> publications.add(binding.valueId()));
                family.compiled.publication().gradientBindings().forEach(binding -> publications.add(binding.valueId()));
                for (var value:publications) actual.add(consumer.valueStorage(value));
                consumer.bind();
                // A published view's backing capacity is not its logical shape. In particular,
                // an empty disconnected gradient can retain a nonempty typed-zero backing.
                for (var value:publications) expected.add(allocate(logicalByteSize(value)));
                carry=expected.get(0); history=expected.get(1); dh=expected.get(2); dx=expected.get(3); dw=expected.get(4);
                savedHistory=allocate(history.byteSize());
                generatedWriteBytes=actual.stream().mapToLong(MemorySegment::byteSize).sum();
                directWriteBytes=expected.stream().mapToLong(MemorySegment::byteSize).sum()+savedHistory.byteSize();
                readBytes=List.of(initial,x,w,fs,es).stream().mapToLong(MemorySegment::byteSize).sum();
            } catch (RuntimeException | Error failure) { consumer.close(); arena.close(); throw failure; }
        }

        /** Checks public result separation without assuming the compiler's saved-site inventory. */
        void assertFootprint() {
            for (var a:actual) for (var e:expected) if (a.byteSize()>0 && e.byteSize()>0)
                assertTrue(a.asOverlappingSlice(e).isEmpty());
        }

        /**
         * Resolves publication geometry from the actual complete compiled value, not a plan
         * position or its physical storage span.
         * @param value exact forward/gradient publication value identity
         * @return resolved logical byte count, including empty and broadcast views
         */
        private long logicalByteSize(ValueId value) {
            var descriptor=family.compiled.graph().values().stream().filter(v -> v.id().equals(value))
                    .findFirst().orElseThrow().descriptor();
            long elements=1;
            for (var dimension:descriptor.shape().dimensions())
                elements=Math.multiplyExact(elements,dimension instanceof DynamicDimension ? b : dimension.staticSize().orElseThrow());
            return Math.multiplyExact(elements,descriptor.dataType().byteWidth());
        }

        /**
         * Allocates zero-filled four-byte-aligned invocation storage.
         * @param bytes checked non-negative byte count, including structural zero resources
         * @return new fixture-owned segment
         */
        private MemorySegment allocate(long bytes) {
            capacity=Math.addExact(capacity,bytes); var s=arena.allocate(bytes,4); s.fill((byte)0); allocations.add(s); return s;
        }


        /** Executes the real prepared and bound forward/reverse graph. */
        void generated() { consumer.execute(); }

        /**
         * Selects a fixed-T/F/H TANH method once, outside invocation/hot loops.
         * @return new wrapper over this fixture's native storage
         * @throws IllegalArgumentException if not one of the two timed families
         */
        Runnable timedDirect() {
            if (!sigmoid && t==3 && f==4 && h==3) return ()->CpuSymbolicScanCleanJavaOracle.t3f4h3(b,initial,x,w,fs,es,carry,history,savedHistory,dh,dx,dw);
            if (!sigmoid && t==4 && f==32 && h==32) return ()->CpuSymbolicScanCleanJavaOracle.t4f32h32(b,initial,x,w,fs,es,carry,history,savedHistory,dh,dx,dw);
            throw new IllegalArgumentException("not a fixed timed TANH family");
        }

        /**
         * Poisons writable outputs/sites only; inputs/constants remain unchanged.
         * @param poison byte pattern exposing omitted reset or empty-contraction writes
         */
        void reset(byte poison) {
            consumer.poisonProducedPublications(poison);
            expected.forEach(s->s.fill(poison)); savedHistory.fill(poison);
        }

        /** Checks the retained generic reference for arbitrary dimensions and activation. */
        void verify() { verifyWith(this::direct); }

        /**
         * Invokes both variants after poison and checks every output/saved site.
         * @param direct non-null selected specialized or generic Java invocation
         */
        void verifyWith(Runnable direct) { reset((byte)0x5a); generated(); direct.run(); assertMatched(); }

        /** Read-only all-five-output, represented saved-site and footprint checks. */
        void assertMatched() {
            assertFootprint();
            var logical=new ArrayList<double[]>();
            for (int p=0;p<family.compiled.publication().forwardBindings().size();p++) logical.add(consumer.forward(p));
            for (int p=0;p<family.compiled.publication().gradientBindings().size();p++) logical.add(consumer.gradient(p));
            for (int p=0;p<logical.size();p++) {
                assertEquals(expected.get(p).byteSize()/4,logical.get(p).length,"logical publication cells port="+p);
                for (int i=0;i<logical.get(p).length;i++)
                    assertEquals(get(expected.get(p),i),logical.get(p)[i],3e-5,"port="+p+" cell="+i);
            }
            assertEquals(-1,history.mismatch(savedHistory),"Java public/saved histories preserve identical represented bits");
            if (b==0) for (int i=0;i<logical.get(4).length;i++) {
                assertEquals(0,Float.floatToRawIntBits(get(dw,i)),"nonempty dW typed zero");
                assertEquals(0,Float.floatToRawIntBits((float)logical.get(4)[i]),"generated nonempty dW typed zero");
            }
        }

        /**
         * Generic semantic reference, never timed. Supports both activations and T=0; every call
         * overwrites complete public/saved outputs, including nonempty parameter gradients at B=0.
         */
        void direct() {
            for (int time=0;time<t;time++) for (int row=0;row<b;row++) for (int col=0;col<h;col++) {
                float sum=0; for (int k=0;k<f;k++) sum+=get(x,(time*b+row)*f+k)*get(w,k*h+col);
                float previous=time==0?get(initial,row*h+col):get(history,((time-1)*b+row)*h+col);
                float v=sum+previous;
                float y=sigmoid?(float)(v>=0?1.0/(1.0+StrictMath.exp(-v)):StrictMath.exp(v)/(1.0+StrictMath.exp(v))):(float)StrictMath.tanh(v);
                set(history,(time*b+row)*h+col,y); set(savedHistory,(time*b+row)*h+col,y);
            }
            MemorySegment.copy(t==0?initial:history,t==0?0:4L*(t-1)*b*h,carry,0,carry.byteSize());
            if (t==0) { MemorySegment.copy(fs,0,dh,0,fs.byteSize()); dw.fill((byte)0); return; }
            for (int time=t-1;time>=0;time--) {
                MemorySegment oldH=time==t-1?fs:dh;
                for (int row=0;row<b;row++) for (int col=0;col<h;col++) {
                    int i=row*h+col; float y=get(savedHistory,time*b*h+i);
                    set(dh,i,(get(oldH,i)+get(es,time*b*h+i))*(sigmoid?y*(1-y):1-y*y));
                }
                for (int row=0;row<b;row++) for (int k=0;k<f;k++) {
                    float sum=0; for (int col=0;col<h;col++) sum+=get(dh,row*h+col)*get(w,k*h+col); set(dx,(time*b+row)*f+k,sum);
                }
                for (int k=0;k<f;k++) for (int col=0;col<h;col++) {
                    float sum=0; for (int row=0;row<b;row++) sum+=get(x,(time*b+row)*f+k)*get(dh,row*h+col);
                    set(dw,k*h+col,(time==t-1?0f:get(dw,k*h+col))+sum);
                }
            }
        }

        /**
         * Reads a represented cell for generic reference/checks.
         * @param s live non-null segment
         * @param i valid zero-based FLOAT32 index
         * @return represented FLOAT32 value
         */
        private static float get(MemorySegment s,long i) { return s.getAtIndex(ValueLayout.JAVA_FLOAT,i); }
        /**
         * Writes a represented cell for generic reference/setup.
         * @param s live writable segment
         * @param i valid zero-based FLOAT32 index
         * @param value represented FLOAT32 value
         */
        private static void set(MemorySegment s,long i,float value) { s.setAtIndex(ValueLayout.JAVA_FLOAT,i,value); }
        /**
         * Initializes nonuniform input outside kernel windows.
         * @param s live writable segment
         * @param scale finite FLOAT32 input scale
         */
        private static void fill(MemorySegment s,float scale) { for (long i=0;i<s.byteSize()/4;i++) set(s,i,(i%11-5)*scale); }
        /** Releases only this fixture's resources; repeated closure is harmless. */
        @Override public void close() {
            if (arena.scope().isAlive()) {
                try { consumer.close(); } finally { arena.close(); }
            }
        }
    }

    /**
     * Constructs a storage-free symbolic FLOAT32 source leaf.
     * @param grad gradient eligibility
     * @param dimensions non-null fixed/one-B dimensions
     * @return new non-null source Tensor
     */
    private static Tensor symbolic(boolean grad,Dimension... dimensions) { return CpuSymbolicScanGeneratedKernelTest.symbolic(grad,dimensions); }
    /**
     * Constructs a canonical fixed FLOAT32 source leaf.
     * @param grad gradient eligibility
     * @param dimensions non-negative fixed extents
     * @return new non-null source Tensor
     */
    private static Tensor fixed(boolean grad,long... dimensions) { return CpuSymbolicScanGeneratedKernelTest.fixed(grad,dimensions); }
}
