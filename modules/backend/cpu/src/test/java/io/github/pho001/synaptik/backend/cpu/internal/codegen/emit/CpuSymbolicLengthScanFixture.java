package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.CpuGeneralScanCapabilityTest;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuTypedEmptyScanConsumerTest.Consumer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.compiler.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import java.lang.foreign.*;
import java.util.*;

/** Production-bound joint forward/AD consumers with independently owned numerical references. */
final class CpuSymbolicLengthScanFixture implements AutoCloseable {
    static final class Family {
        final DataType type;
        final boolean reverse, symbolic, lengthAware;
        final int time, features;
        final CompileArtifacts compiled;
        final List<TensorId> inputs;
        private final List<Tensor> tensors;
        final CpuPartitionAnalysisInputs analysisInputs;
        final BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis;
        final long compileAnalyzeNanos;

        Family(DataType type,boolean reverse) { this(type,reverse,4); }
        Family(DataType type,boolean reverse,int time) { this(type,reverse,time,3); }
        Family(DataType type,boolean reverse,int time,int features) {
            this(type,reverse,time,features,CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT);
        }
        Family(DataType type,boolean reverse,int time,int features,CpuPartitionAnalysisInputs.PortableExecutionConfig execution) {
            this(type,reverse,time,features,execution,true,true,0);
        }
        Family(DataType type,boolean reverse,int time,int features,CpuPartitionAnalysisInputs.PortableExecutionConfig execution,
                boolean symbolic,boolean lengthAware,int fixedBatch) {
            if (time < 0) throw new IllegalArgumentException("negative fixed T");
            if (features < 1) throw new IllegalArgumentException("nonpositive fixed F");
            long started = System.nanoTime();
            this.type=type; this.reverse=reverse; this.time=time; this.features=features;
            this.symbolic=symbolic; this.lengthAware=lengthAware;
            Dimension b = symbolic ? new DynamicDimension("B") : new StaticDimension(fixedBatch);
            var row = Shape.ofDimensions(b,new StaticDimension(2));
            var inputRow = Shape.ofDimensions(b,new StaticDimension(features));
            var sequence = Shape.ofDimensions(new StaticDimension(time),b,new StaticDimension(features));
            Tensor h=tensor(type,true,row), x=tensor(type,true,inputRow), w=tensor(type,true,Shape.of(features,2));
            Tensor u=tensor(type,true,Shape.of(2,2)), bias=tensor(type,true,Shape.of(2));
            Tensor next=x.matmul(w).add(h.matmul(u)).add(bias).tanh();
            Tensor initial=tensor(type,true,row), seq=tensor(type,true,sequence), wx=tensor(type,true,Shape.of(features,2));
            Tensor wh=tensor(type,true,Shape.of(2,2)), offset=tensor(type,true,Shape.of(2));
            Tensor lengths=tensor(DataType.INT64,false,Shape.ofDimensions(b));
            Tensor fs=tensor(type,false,row), es=tensor(type,false,Shape.ofDimensions(new StaticDimension(time),b,new StaticDimension(2)));
            tensors=List.of(initial,seq,wx,wh,offset,lengths,fs,es);
            inputs=tensors.stream().map(Tensor::id).toList();
            var body=new ScanBody(List.of(h),List.of(x),List.of(w,u,bias),List.of(next),List.of(next));
            var scan=lengthAware ? Scan.apply(List.of(initial),List.of(seq),List.of(wx,wh,offset),lengths,1,List.of(OptionalInt.of(0)),body,reverse)
                    : Scan.apply(List.of(initial),List.of(seq),List.of(wx,wh,offset),body,reverse);
            var roots=List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
            // No transition exists at T=0: only the initial-carry target remains connected.
            // Positive-T requests keep ERROR; the empty sequence/parameter targets explicitly
            // request their mathematically disconnected typed zeros.
            var request=new FunctionalGradientRequest.Stage(roots.stream().map(r -> (FunctionalGradientRequest.OutputReference)new FunctionalGradientRequest.ForwardTensorReference(r)).toList(),
                    List.of(Optional.of(fs),Optional.of(es)),List.of(initial,seq,wx,wh,offset),false,
                    time == 0 ? FunctionalGradientRequest.DisconnectedPolicy.ZERO : FunctionalGradientRequest.DisconnectedPolicy.ERROR);
            compiled=CpuGeneralScanCapabilityTest.compile(roots,Optional.of(new FunctionalGradientRequest(List.of(request))),false,new CpuCapabilityProvider());
            analysisInputs=new CpuPartitionAnalysisInputs(false,List.of(),execution);
            analysis=new CpuPartitionPreparer().analyze(GraphPreparation.project(compiled,compiled.partitions().getFirst(),analysisInputs));
            compileAnalyzeNanos=System.nanoTime()-started;
        }
    }

    final Family family;
    final int b;
    final Arena arena=Arena.ofShared();
    final Consumer consumer;
    final List<MemorySegment> actual=new ArrayList<>(), expected=new ArrayList<>();
    final MemorySegment initial,x,w,u,bias,lengths,fs,es,oraclePrevious,oracleSaved,oracleForwardScratch,oracleReverseScratch;
    final Runnable directForward,directReverse;

    CpuSymbolicLengthScanFixture(Family family,int batch) {
        this.family=family; b=batch;
        consumer=new Consumer(family.compiled,family.symbolic ? OptionalLong.of(batch) : OptionalLong.empty(),Map.of(),family.analysisInputs);
        try {
            initial=consumer.inputStorage(family.tensors.get(0));
            // At T=0 these primals have no physical read. Keep independent oracle inputs,
            // never fabricated assigned buffers or dependencies in the compiled graph.
            x=family.time == 0 ? arena.allocate(0,family.type.byteWidth()) : consumer.inputStorage(family.tensors.get(1));
            w=family.time == 0 ? arena.allocate(2L*family.features*family.type.byteWidth(),family.type.byteWidth()) : consumer.inputStorage(family.tensors.get(2));
            u=family.time == 0 ? arena.allocate(4L*family.type.byteWidth(),family.type.byteWidth()) : consumer.inputStorage(family.tensors.get(3));
            bias=family.time == 0 ? arena.allocate(2L*family.type.byteWidth(),family.type.byteWidth()) : consumer.inputStorage(family.tensors.get(4));
            lengths=family.lengthAware ? consumer.inputStorage(family.tensors.get(5)) : arena.allocate(8L*batch,8);
            fs=consumer.inputStorage(family.tensors.get(6));
            es=family.time == 0 ? arena.allocate(0,family.type.byteWidth()) : consumer.inputStorage(family.tensors.get(7));
            fill(initial,.01); fill(x,.02); fill(w,.03); fill(u,.025); fill(bias,.04); fill(fs,.1); fill(es,.05);
            for (int row=0; row<batch; row++) lengths.setAtIndex(ValueLayout.JAVA_LONG,row,family.lengthAware ? Math.min(row%5,family.time) : family.time);
            for (int p=0; p<family.compiled.publication().forwardBindings().size(); p++) actual.add(consumer.forwardStorage(p));
            for (int p=0; p<family.compiled.publication().gradientBindings().size(); p++) actual.add(consumer.gradientStorage(p));
            for (var output:actual) expected.add(arena.allocate(output.byteSize(),family.type.byteWidth()));
            oraclePrevious=arena.allocate(2L*family.time*batch*family.type.byteWidth(),family.type.byteWidth());
            oracleSaved=arena.allocate(oraclePrevious.byteSize(),family.type.byteWidth());
            oracleForwardScratch=arena.allocate(4L*batch*family.type.byteWidth(),family.type.byteWidth());
            oracleReverseScratch=arena.allocate(2L*batch*family.type.byteWidth(),family.type.byteWidth());
            directForward=this::oracleForward; directReverse=this::oracleReverse;
            consumer.bind();
        } catch (Throwable failure) { consumer.close(); arena.close(); throw failure; }
    }

    /** Executes the complete production-bound forward and requested AD publication pipeline. */
    void execute() { consumer.execute(); }
    /** Joint execution, not a separately callable forward/reverse kernel ABI. */
    void generated() { consumer.execute(); }
    Runnable preboundExecution() { return consumer::execute; }

    /** Uses the selected scalar StrictMath TANH producer while preserving all typed operation boundaries. */
    void oracleForward() {
        if (family.time==4 && family.features==3) {
            if (family.type==DataType.FLOAT64) CpuSymbolicLengthScanCleanJavaOracle.forwardFloat64(initial,x,w,u,bias,lengths,expected.get(0),expected.get(1),oraclePrevious,oracleSaved,oracleForwardScratch,b,family.reverse);
            else CpuSymbolicLengthScanCleanJavaOracle.forwardFloat32(initial,x,w,u,bias,lengths,expected.get(0),expected.get(1),oraclePrevious,oracleSaved,oracleForwardScratch,b,family.reverse);
            return;
        }
        var result=expected.get(0); var emissions=expected.get(1);
        result.copyFrom(initial); emissions.fill((byte)0); oraclePrevious.fill((byte)0); oracleSaved.fill((byte)0);
        for (int step=0; step<family.time; step++) {
            int time=family.reverse ? family.time-1-step : step;
            for (int row=0; row<b; row++) if (active(time,row)) for (int col=0; col<2; col++) {
                put(oraclePrevious,(time*b+row)*2L+col,get(result,row*2L+col));
                double sum=0;
                for (int k=0; k<family.features; k++) sum=round(sum+round(get(x,(time*b+row)*(long)family.features+k)*get(w,k*2L+col)));
                put(oracleForwardScratch,row*2L+col,sum);
            }
            for (int row=0; row<b; row++) if (active(time,row)) for (int col=0; col<2; col++) {
                double sum=0;
                for (int k=0; k<2; k++) sum=round(sum+round(get(result,row*2L+k)*get(u,k*2L+col)));
                double next=round(StrictMath.tanh(round(round(get(oracleForwardScratch,row*2L+col)+sum)+get(bias,col))));
                put(oracleForwardScratch,2L*b+row*2L+col,next);
                put(emissions,(time*b+row)*2L+col,next); put(oracleSaved,(time*b+row)*2L+col,next);
            }
            for (int row=0; row<b; row++) if (active(time,row)) for (int col=0; col<2; col++)
                put(result,row*2L+col,get(oracleForwardScratch,2L*b+row*2L+col));
        }
    }

    void oracleReverse() {
        if (family.time==4 && family.features==3) {
            if (family.type==DataType.FLOAT64) CpuSymbolicLengthScanCleanJavaOracle.reverseFloat64(x,w,u,lengths,fs,es,oraclePrevious,oracleSaved,expected.get(2),expected.get(3),expected.get(4),expected.get(5),expected.get(6),oracleReverseScratch,b,family.reverse);
            else CpuSymbolicLengthScanCleanJavaOracle.reverseFloat32(x,w,u,lengths,fs,es,oraclePrevious,oracleSaved,expected.get(2),expected.get(3),expected.get(4),expected.get(5),expected.get(6),oracleReverseScratch,b,family.reverse);
            return;
        }
        var dh=expected.get(2); var dx=expected.get(3); var dw=expected.get(4); var du=expected.get(5); var db=expected.get(6);
        dh.copyFrom(fs); dx.fill((byte)0); dw.fill((byte)0); du.fill((byte)0); db.fill((byte)0);
        for (int step=0; step<family.time; step++) {
            int time=family.reverse ? step : family.time-1-step;
            boolean present=false;
            for (int row=0; row<b; row++) if (active(time,row)) { present=true; break; }
            if (!present) continue;
            for (int row=0; row<b; row++) if (active(time,row)) for (int col=0; col<2; col++) {
                double y=get(oracleSaved,(time*b+row)*2L+col);
                double seed=round(get(dh,row*2L+col)+get(es,(time*b+row)*2L+col));
                put(oracleReverseScratch,row*2L+col,round(seed*round(1-round(y*y))));
            }
            for (int row=0; row<b; row++) if (active(time,row)) for (int k=0; k<family.features; k++) {
                double sum=0;
                for (int col=0; col<2; col++) sum=round(sum+round(get(oracleReverseScratch,row*2L+col)*get(w,k*2L+col)));
                put(dx,(time*b+row)*(long)family.features+k,sum);
            }
            for (int k=0; k<family.features; k++) for (int col=0; col<2; col++) {
                double sum=0;
                for (int row=0; row<b; row++) if (active(time,row)) sum=round(sum+round(get(x,(time*b+row)*(long)family.features+k)*get(oracleReverseScratch,row*2L+col)));
                put(dw,k*2L+col,round(get(dw,k*2L+col)+sum));
            }
            for (int k=0; k<2; k++) for (int col=0; col<2; col++) {
                double sum=0;
                for (int row=0; row<b; row++) if (active(time,row)) sum=round(sum+round(get(oraclePrevious,(time*b+row)*2L+k)*get(oracleReverseScratch,row*2L+col)));
                put(du,k*2L+col,round(get(du,k*2L+col)+sum));
            }
            for (int col=0; col<2; col++) {
                double sum=b==1 ? get(oracleReverseScratch,col) : 0;
                if (b!=1) for (int row=0; row<b; row++) if (active(time,row)) sum=round(sum+get(oracleReverseScratch,row*2L+col));
                put(db,col,round(get(db,col)+sum));
            }
            for (int row=0; row<b; row++) if (active(time,row)) for (int k=0; k<2; k++) {
                double sum=0;
                for (int col=0; col<2; col++) sum=round(sum+round(get(oracleReverseScratch,row*2L+col)*get(u,k*2L+col)));
                put(dh,row*2L+k,sum);
            }
        }
    }

    private boolean active(int time,int row) { return time<lengths.getAtIndex(ValueLayout.JAVA_LONG,row); }
    private double round(double value) { return family.type==DataType.FLOAT32 ? (float)value : value; }
    void verifyLatest() {
        for (int p=0; p<actual.size(); p++) for (long i=0; i<actual.get(p).byteSize()/family.type.byteWidth(); i++)
            assertEquals(get(expected.get(p),i),get(actual.get(p),i),family.type==DataType.FLOAT64 ? 2e-14 : 2e-6,"port="+p+" cell="+i);
    }
    void verify() { execute(); oracleForward(); oracleReverse(); verifyLatest(); }
    void fill(MemorySegment segment,double scale) { for (long i=0; i<segment.byteSize()/family.type.byteWidth(); i++) put(segment,i,(i%11-5)*scale); }
    double get(MemorySegment segment,long i) { return family.type==DataType.FLOAT64 ? segment.getAtIndex(ValueLayout.JAVA_DOUBLE,i) : segment.getAtIndex(ValueLayout.JAVA_FLOAT,i); }
    void put(MemorySegment segment,long i,double value) { if (family.type==DataType.FLOAT64) segment.setAtIndex(ValueLayout.JAVA_DOUBLE,i,value); else segment.setAtIndex(ValueLayout.JAVA_FLOAT,i,(float)value); }
    static Tensor tensor(DataType type,boolean gradient,Shape shape) { return TensorFactory.create(new TensorDescriptor(type,shape,shape.isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(),gradient)); }
    @Override public void close() { try { consumer.close(); } finally { arena.close(); } }
}
