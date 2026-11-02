package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.*;
import io.github.pho001.synaptik.backend.cpu.internal.executable.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import java.lang.classfile.*;
import java.lang.classfile.instruction.*;
import java.lang.foreign.*;
import java.nio.file.*;
import java.util.*;
import jdk.incubator.vector.*;
import org.junit.jupiter.api.Test;

/**
 * Unmeasured controls and opt-in finite observations. Ordinary arms retain their selected
 * regional vector/parallel proof; Scan arms execute the complete production-bound forward
 * and reverse graph against the independent clean-Java reference, without a legacy phase ABI.
 */
class CpuRowFusionSimdParallelTimingTest {
    private static volatile long sink;
    private static final com.sun.management.ThreadMXBean ALLOCATIONS=allocationBean();

    @Test void emptyOrdinaryJavaArmsNeverReadAbsentWeightsOrSavedInputs() {
        var family=new CpuSymbolicLengthScanFixture.Family(DataType.FLOAT32,false,0,3,config(false,false));
        try(var run=new CpuRecurrentScanExecutionTest.Fixture(family.compiled,family.analysisInputs,family.analysis,1,true,true,Map.of(),OptionalLong.of(0));var arena=Arena.ofShared()) {
            for(var type:List.of(DataType.FLOAT32,DataType.FLOAT64))for(boolean vector:List.of(false,true))for(boolean parallel:List.of(false,true)) {
                var dw=arena.allocate(6L*type.byteWidth(),type.byteWidth());var du=arena.allocate(4L*type.byteWidth(),type.byteWidth());
                dw.fill((byte)0x55);du.fill((byte)0x55);
                var empty=MemorySegment.NULL;
                var data=new CpuRowFusionSimdParallelJavaOracle.Data(empty,empty,empty,empty,empty,empty,empty,empty,empty,
                        empty,empty,dw,du,empty,0,1,false,false,1,1,0);
                CpuRowFusionSimdParallelJavaOracle.bind(data,type==DataType.FLOAT64,vector,parallel,1,run,arena).run();
                for(byte value:dw.toArray(ValueLayout.JAVA_BYTE))assertEquals(0,value);
                for(byte value:du.toArray(ValueLayout.JAVA_BYTE))assertEquals(0,value);
            }
        }
    }

    /**
     * MAIN-only direct-JVM entry: one measured harness, no Gradle parent/child or retry.
     * A daemon watchdog preserves already appended samples and halts this JVM after sixty seconds.
     * @param args must be empty; evidence/release arrive through the same environment gate
     * @throws Exception if controls, generation, inspection or evidence persistence fails
     */
    public static void main(String[] args) throws Exception {
        if(args.length!=0||!"true".equals(System.getenv("SYNAPTIK_CPU0074_MEASURE")))
            throw new IllegalArgumentException("MAIN release and no arguments required");
        Path root=Path.of(Objects.requireNonNull(System.getenv("SYNAPTIK_CPU0074_EVIDENCE")));
        Files.createDirectories(root);
        Thread.ofPlatform().daemon(true).name("cpu0074-sole-jvm-watchdog").start(()->{
            try {Thread.sleep(60_000);}catch(InterruptedException finished){return;}
            try {Files.writeString(root.resolve("CENSORED.txt"),"60-second sole-JVM watchdog; no retry; completed samples retained.\n");}
            catch(java.io.IOException failure){failure.printStackTrace();}
            Runtime.getRuntime().halt(124);
        });
        try {new CpuRowFusionSimdParallelTimingTest().soleMeasuredMatrixRequiresMainRelease();}
        catch(Throwable failure){Files.writeString(root.resolve("FAILED.txt"),failure.toString()+"\n");throw failure;}
    }

    @Test void independentJavaControlsAndSelectedClassFileInspection() throws Exception {
        String parent=System.getenv("SYNAPTIK_CPU0074_EVIDENCE");
        Path root=parent==null?Files.createTempDirectory("cpu0074-unmeasured-inspection.")
                :Files.createTempDirectory(Files.createDirectories(Path.of(parent)),"unmeasured-inspection.");
        matrix(root,false);
        System.out.println("CPU0074 unmeasured controls/inspection: "+root);
    }

    @Test void soleMeasuredMatrixRequiresMainRelease() throws Exception {
        org.junit.jupiter.api.Assumptions.assumeTrue("true".equals(System.getenv("SYNAPTIK_CPU0074_MEASURE")));
        String supplied=Objects.requireNonNull(System.getenv("SYNAPTIK_CPU0074_EVIDENCE"),"fresh evidence directory required");
        Path root=Path.of(supplied);Files.createDirectories(root);
        matrix(root,true);
    }

    /**
     * Runs exactly eight rows; no calibration, retry, fallback measurement or extra fork.
     * @param root fresh private candidate artifact/evidence directory
     * @param measure whether MAIN has released the single 50ms+3x50ms-per-arm budget
     */
    private static void matrix(Path root,boolean measure) throws Exception {
        Files.writeString(root.resolve("environment.txt"),"java="+System.getProperty("java.runtime.version")+"\n"
                +"F32 species="+FloatVector.SPECIES_PREFERRED+" F64 species="+DoubleVector.SPECIES_PREFERRED+"\n"
                +"availableProcessors="+Runtime.getRuntime().availableProcessors()+"\n"
                +"scope=ordinary selected region versus matching Java; Scan complete bound forward/reverse versus generic clean Java; native FFM.\n"
                +"Ordinary Java parallel borrows the same worker group/budget; Scan policy requests are not strategy-selection claims.\n"
                +"Controls, compilation, binding, checksums and result comparison are outside timing.\n"
                +"Allocation: caller ThreadMXBean="+(ALLOCATIONS==null?"unsupported":"enabled; worker allocations excluded")+". Cold native/setup bytes reported per arm.\n"
                +"No assembly/JIT guarantee or C2 claim; 50ms warm is short, tiering and host noise may dominate.\n"
                +"System.nanoTime loop overhead is shared and not subtracted; asynchronous host activity is uncontrolled.\n"
                +"Production preflight/admission versus Java equation-only admission is retained, not subtracted.\n");
        Path samples=root.resolve("samples.csv");
        Files.writeString(samples,"case,type,B,mode,phase,sample,arm,order,calls,elapsed_ns,ns_per_call,caller_allocated_bytes,caller_bytes_per_call\n");
        for(boolean scan:List.of(false,true))for(var type:List.of(DataType.FLOAT32,DataType.FLOAT64))
            for(int batch:List.of(32,256))for(int mode=0;mode<4;mode++) {
                boolean vector=(mode&1)!=0,parallel=(mode&2)!=0;
                String id=(scan?"scan":"ordinary")+"-"+type+"-B"+batch+"-"+mode;
                Path dir=Files.createDirectory(root.resolve(id));
                long setupBegin=System.nanoTime(),allocatedBefore=allocatedBytes();
                try(var arm=scan?scan(type,batch,vector,parallel,dir):ordinary(type,batch,vector,parallel,dir)) {
                    long setupNanos=System.nanoTime()-setupBegin,allocatedAfter=allocatedBytes();
                    long workspaceBytes=arm.run.state.memoryPlan().workspaces().stream().mapToLong(w->w.size().byteSize(batch)).sum();
                    long oraclePrivateBytes=scan?0:(vector?10L*type.byteWidth():0)+(parallel?20L*type.byteWidth():0);
                    Files.writeString(dir.resolve("setup.txt"),"cold_compile_generate_define_finalize_bind_ns="+setupNanos
                            +"\ncaller_setup_allocated_bytes="+allocationDelta(allocatedBefore,allocatedAfter)
                            +"\ngenerated_all_partition_buffer_native_bytes="+arm.run.storage.values().stream().mapToLong(MemorySegment::byteSize).sum()
                            +"\ngenerated_all_partition_workspace_native_bytes="+workspaceBytes
                            +"\noracle_transpose_and_private_partial_native_bytes="+oraclePrivateBytes
                            +"\nScan compares the complete production invocation; ordinary retains selected-region staging and range partials.\n");
                    arm.generated.run();arm.java.run();arm.check.run();
                    if(measure) {
                        persist(samples,id,type,batch,mode,"warm",-1,"generated",0,sample(arm.generated,50_000_000));
                        persist(samples,id,type,batch,mode,"warm",-1,"java",1,sample(arm.java,50_000_000));
                        for(int sample=0;sample<3;sample++) {
                            boolean generatedFirst=(sample&1)==0;
                            persist(samples,id,type,batch,mode,"sample",sample,generatedFirst?"generated":"java",0,
                                    sample(generatedFirst?arm.generated:arm.java,50_000_000));
                            persist(samples,id,type,batch,mode,"sample",sample,generatedFirst?"java":"generated",1,
                                    sample(generatedFirst?arm.java:arm.generated,50_000_000));
                        }
                    }
                    arm.check.run();sink^=arm.run.storage.values().stream().mapToLong(MemorySegment::byteSize).sum();
                }
            }
        Files.writeString(root.resolve("COMPLETE.txt"),measure?"All approved samples completed; no retry/fork.\n":"All unmeasured controls completed; no timing windows.\n");
    }

    /**
     * Measures one fixed window without controls or reporting in the loop.
     * @param action prebound equation/selected invocation
     * @param windowNanos exactly 50,000,000 nanoseconds
     * @return completed call count, elapsed time and caller allocation delta; unsupported is -1
     */
    private static Sample sample(Runnable action,long windowNanos) {
        long allocatedBefore=allocatedBytes(),begin=System.nanoTime(),count=0,now;
        do {action.run();count++;now=System.nanoTime();}while(now-begin<windowNanos);
        long allocatedAfter=allocatedBytes();sink^=count;
        return new Sample(count,now-begin,allocationDelta(allocatedBefore,allocatedAfter));
    }

    /** Completed windows are appended immediately outside timing; a watchdog retains prior rows. */
    private static void persist(Path path,String id,DataType type,int batch,int mode,String phase,int sample,
            String arm,int order,Sample value) throws Exception {
        Files.writeString(path,id+","+type+","+batch+","+mode+","+phase+","+sample+","+arm+","+order+","+value.calls+","+value.elapsed
                +","+(double)value.elapsed/value.calls+","+value.allocated+","+(value.allocated<0?-1:(double)value.allocated/value.calls)+"\n",StandardOpenOption.APPEND);
    }
    /** @return available enabled caller allocation counter, or null when unsupported/forbidden */
    private static com.sun.management.ThreadMXBean allocationBean() {
        try {
            var bean=java.lang.management.ManagementFactory.getThreadMXBean();
            if(bean instanceof com.sun.management.ThreadMXBean allocation&&allocation.isThreadAllocatedMemorySupported()) {
                if(!allocation.isThreadAllocatedMemoryEnabled())allocation.setThreadAllocatedMemoryEnabled(true);
                return allocation;
            }
        }catch(UnsupportedOperationException|SecurityException unavailable){ }
        return null;
    }
    private static long allocatedBytes(){return ALLOCATIONS==null?-1:ALLOCATIONS.getThreadAllocatedBytes(Thread.currentThread().threadId());}
    private static long allocationDelta(long before,long after){return before<0||after<0?-1:after-before;}
    private record Sample(long calls,long elapsed,long allocated) { }

    private static CpuPartitionAnalysisInputs.PortableExecutionConfig config(boolean vector,boolean parallel) {
        return new CpuPartitionAnalysisInputs.PortableExecutionConfig(vector
                ? CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                : CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,parallel?2:1,2,1);
    }

    private static void selected(CpuPartitionPreparationPlan.ExecutionUnitPlan unit,boolean vector,boolean parallel) {
        var expected=parallel?vector?CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_VECTOR
                :CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_SCALAR:vector
                ?CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR:CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR;
        assertEquals(expected,unit.executionStrategy(),"no scalar fallback may qualify an arm");
        assertEquals(parallel?2:1,unit.selectedRangeCount());
    }

    private static Arm scan(DataType type,int b,boolean vector,boolean parallel,Path dir) throws Exception {
        var family=new CpuSymbolicLengthScanFixture.Family(type,false,4,3,config(vector,parallel));
        var source=new CpuSymbolicLengthScanFixture(family,b);
        var run=new CpuRecurrentScanExecutionTest.Fixture(family.compiled,family.analysisInputs,family.analysis,1,true,true,Map.of(),OptionalLong.of(b),null,Optional.of(dir.resolve("cache")));
        var arena=Arena.ofShared();
        try {
            CpuRowFusionSimdParallelTest.copyReferenceInputs(source,run);
            var invocation=run.executable.bind(run.state);
            var actual=new ArrayList<MemorySegment>();
            family.compiled.publication().forwardBindings().forEach(p->actual.add(run.storage.get(p.valueId())));
            family.compiled.publication().gradientBindings().forEach(p->actual.add(run.storage.get(p.valueId())));
            Runnable java=()->{source.directForward.run();source.directReverse.run();};
            Files.writeString(dir.resolve("oracle-storage.txt"),"independent_public_output_native_bytes="
                    +source.expected.stream().mapToLong(MemorySegment::byteSize).sum()
                    +"\nindependent_oracle_work_native_bytes="+(source.oraclePrevious.byteSize()+source.oracleSaved.byteSize()
                            +source.oracleForwardScratch.byteSize()+source.oracleReverseScratch.byteSize())+"\n");
            return new Arm(run,source,arena,invocation::execute,java,()->compare(type,actual,source.expected));
        } catch(Throwable failure){arena.close();run.close();source.close();throw failure;}
    }

    private static Arm ordinary(DataType type,int b,boolean vector,boolean parallel,Path dir) throws Exception {
        Tensor x=CpuSymbolicLengthScanFixture.tensor(type,true,Shape.of(b,3)),h=CpuSymbolicLengthScanFixture.tensor(type,true,Shape.of(b,2));
        Tensor w=CpuSymbolicLengthScanFixture.tensor(type,true,Shape.of(3,2)),u=CpuSymbolicLengthScanFixture.tensor(type,true,Shape.of(2,2));
        Tensor bias=CpuSymbolicLengthScanFixture.tensor(type,true,Shape.of(2));
        Tensor y=x.matmul(w).add(h.matmul(u)).add(bias).tanh(),loss=y.mul(y).sum();
        var inputs=List.of(x,h,w,u,bias);
        var compiled=CpuGeneralScanCapabilityTest.compile(List.of(loss,y),Optional.of(CpuGeneralScanCapabilityTest.request(List.of(loss),inputs)),false,new CpuCapabilityProvider());
        var analysisInputs=new CpuPartitionAnalysisInputs(false,List.of(),config(vector,parallel));
        var analysis=new CpuPartitionPreparer().analyze(GraphPreparation.project(compiled,compiled.partitions().getFirst(),analysisInputs));
        var unit=analysis.plan().units().stream().filter(v->v.portablePlan().specialization().rowMapReductionIr().isPresent()).findFirst().orElseThrow();
        selected(unit,vector,parallel);
        var run=new CpuRecurrentScanExecutionTest.Fixture(compiled,analysisInputs,analysis,1,true,true,Map.of(),OptionalLong.empty(),null,Optional.of(dir.resolve("cache")));
        var arena=Arena.ofShared();
        try {
            var data=new ArrayList<MemorySegment>();
            for(var binding:compiled.constants().bindableInputBindings()) {
                var s=run.storage.get(binding.valueId());
                for(int k=0;k<s.byteSize()/type.byteWidth();k++)put(type,s,k,1);
            }
            for(var input:inputs) {
                var binding=compiled.constants().bindableInputBindings().stream().filter(v->v.tensorId().equals(input.id())).findFirst().orElseThrow();
                var s=run.storage.get(binding.valueId());data.add(s);
                for(int k=0;k<s.byteSize()/type.byteWidth();k++)put(type,s,k,(k%11-5)*.035);
            }
            run.executable.bind(run.state).execute();
            var saved=run.storage.get(compiled.publication().forwardBindings().get(1).valueId());
            var actual=compiled.publication().gradientBindings().stream().limit(4).map(p->run.storage.get(p.valueId())).toList();
            var out=actual.stream().map(s->arena.allocate(s.byteSize(),type.byteWidth())).toList();
            var delta=arena.allocate(2L*b*type.byteWidth(),type.byteWidth());
            Files.writeString(dir.resolve("oracle-storage.txt"),"independent_output_native_bytes="+(delta.byteSize()+out.stream().mapToLong(MemorySegment::byteSize).sum())+"\n");
            var equation=new CpuRowFusionSimdParallelJavaOracle.Data(data.get(0),data.get(1),saved,data.get(2),data.get(3),
                    MemorySegment.NULL,MemorySegment.NULL,MemorySegment.NULL,delta,out.get(0),out.get(1),out.get(2),out.get(3),MemorySegment.NULL,
                    b,1,false,false,1,1,0);
            var java=CpuRowFusionSimdParallelJavaOracle.bind(equation,type==DataType.FLOAT64,vector,parallel,unit.minimumElementsPerWorker(),run,arena);
            var child=((CpuPreparedPartitionExecutable)run.executable).children().get(analysis.plan().units().indexOf(unit)).bind(run.state);
            inspect(unit,dir);
            return new Arm(run,null,arena,child::execute,java,()->compare(type,actual,out));
        }catch(Throwable failure){arena.close();run.close();throw failure;}
    }

    /**
     * Inspects every method, including range/merge, and saves the exact candidate bytes and calls.
     * @param unit selected execution unit
     * @param dir fresh private evidence directory
     */
    private static void inspect(CpuPartitionPreparationPlan.ExecutionUnitPlan unit,Path dir) throws Exception {
        var spec=unit.portablePlan().specialization();var generator=new CpuClassFileKernelGenerator();
        byte[] bytes=generator.generateClassBytes(spec,unit.portablePlan().kernelIr());
        // Defining authenticates all entry/helper descriptors and code guards as well as parsing.
        generator.defineClassBytes(spec, unit.portablePlan().kernelIr(), bytes);
        Files.write(dir.resolve("selected.class"),bytes);
        var report=new StringBuilder("strategy="+unit.executionStrategy()+"\nW="+unit.selectedRangeCount()+" m="+unit.minimumElementsPerWorker()
                +"\nworkspace="+unit.runtimeFacts().workspaceDeclaration()+"\n");
        for(var method:ClassFile.of().parse(bytes).methods())if(method.code().isPresent()) {
            var code=method.code().orElseThrow();int vectors=0,masks=0,allocations=0,divisions=0;
            var attribute=(java.lang.classfile.attribute.CodeAttribute)code;
            report.append(method.methodName().stringValue()).append(" maxLocals=").append(attribute.maxLocals()).append(" codeBytes=").append(attribute.codeLength()).append('\n');
            for(var e:code) {
                if(e instanceof Instruction i && i.opcode()==Opcode.LDIV)divisions++;
                if(e instanceof NewObjectInstruction object) {
                    // Existing range guards may allocate only on their cold throwing path.
                    assertTrue(object.className().asInternalName().endsWith("Exception"),"nonexception generated allocation");
                    report.append("  cold exception allocation ").append(object.className().asInternalName()).append('\n');
                }
                if(e instanceof NewReferenceArrayInstruction || e instanceof NewPrimitiveArrayInstruction)allocations++;
                if(e instanceof InvokeInstruction call) {
                    String owner=call.owner().asInternalName(),name=call.name().stringValue();
                    report.append("  ").append(owner).append('.').append(name).append(call.type().stringValue()).append('\n');
                    assertFalse(Set.of("fma","reduceLanes","withLane").contains(name),name);
                    assertFalse(owner.startsWith("java/lang/reflect")||owner.equals("java/lang/Double")||owner.equals("java/lang/Float"),owner);
                    if(owner.startsWith("io/github/pho001/synaptik/"))assertTrue(owner.endsWith("CpuGeneralScanPreparedExecutable$StepDispatch"),"hidden Synaptik arithmetic/dispatch helper: "+owner);
                    if(owner.endsWith("Vector")&&Set.of("mul","add","neg","sub").contains(name))vectors++;
                    if(owner.endsWith("VectorMask")&&name.equals("fromLong"))masks++;
                }
            }
            assertEquals(0,allocations,"no generated body/step allocation");
            assertTrue(attribute.codeLength()<=48000,"existing emitted-method limit");
            if(spec.rowFusionEmission().isPresent())assertEquals(0,divisions,"selected checked byte access has no /width roundtrip");
            report.append("  vectorArithmetic=").append(vectors).append(" masks=").append(masks).append(" LDIV=").append(divisions).append('\n');
            boolean vector=unit.executionStrategy().compute()==CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR;
            if(vector&&(method.methodName().stringValue().equals("ordinaryRowRange")
                    ||method.methodName().stringValue().equals("scanRowRange")
                    ||(unit.executionStrategy().orchestration()==CpuPartitionPreparationPlan.ExecutionStrategy.Orchestration.SINGLE_THREAD&&method.methodName().stringValue().equals("invoke"))))
                assertTrue(vectors>0,"the executed row method, not a dead instruction elsewhere, must contain vector arithmetic");
        }
        Files.writeString(dir.resolve("inspection.txt"),report);
    }

    private static void compare(DataType type,List<MemorySegment> actual,List<MemorySegment> expected) {
        assertEquals(actual.size(),expected.size());
        for(int p=0;p<actual.size();p++)for(int k=0;k<actual.get(p).byteSize()/type.byteWidth();k++) {
            double a=get(type,actual.get(p),k),e=get(type,expected.get(p),k);
            assertEquals(e,a,type==DataType.FLOAT64?2e-14:2e-6,"port="+p+" cell="+k);
        }
    }
    private static double get(DataType type,MemorySegment s,int k) {
        return type==DataType.FLOAT64?s.getAtIndex(ValueLayout.JAVA_DOUBLE,k):s.getAtIndex(ValueLayout.JAVA_FLOAT,k);
    }
    private static void put(DataType type,MemorySegment s,int k,double value) {
        if(type==DataType.FLOAT64)s.setAtIndex(ValueLayout.JAVA_DOUBLE,k,value);else s.setAtIndex(ValueLayout.JAVA_FLOAT,k,(float)value);
    }

    /** One cold bound pair with independent writable outputs and borrowed immutable saved inputs. */
    private record Arm(CpuRecurrentScanExecutionTest.Fixture run,AutoCloseable source,Arena arena,
            Runnable generated,Runnable java,Runnable check) implements AutoCloseable {
        @Override public void close() throws Exception {try{arena.close();}finally{try{run.close();}finally{if(source!=null)source.close();}}}
    }
}

