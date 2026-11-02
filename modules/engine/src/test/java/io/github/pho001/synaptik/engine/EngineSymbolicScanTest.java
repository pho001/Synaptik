package io.github.pho001.synaptik.engine;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.contract.*;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.compiler.*;
import io.github.pho001.synaptik.config.compile.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Public fresh-run publication of independently qualified forward and actual reverse scans. */
class EngineSymbolicScanTest {
    @Test void lengthMeanUsesDenseBatchDenominatorAndFalseCarryFlagStillPropagates() {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (int time : new int[]{0,4}) {
            var b = new DynamicDimension("B"); var row = Shape.ofDimensions(b,new StaticDimension(3));
            Tensor h = typed(type,false,b,new StaticDimension(3)), x = typed(type,true,b,new StaticDimension(3)), w = typed(type,true,new StaticDimension(3));
            Tensor initial = typed(type,false,b,new StaticDimension(3)), seq = typed(type,true,new StaticDimension(time),b,new StaticDimension(3));
            Tensor weight = typed(type,true,new StaticDimension(3)), lengths = typed(DataType.INT64,false,b);
            var scan = Scan.apply(List.of(initial),List.of(seq),List.of(weight),lengths,1,List.of(OptionalInt.of(0)),
                    new ScanBody(List.of(h),List.of(x),List.of(w),List.of(h.add(x).add(w)),List.of()),false);
            Tensor seed = typed(type,false);
            try (var engine = Engine.standard()) {
                // No step can connect seq or weight to the selected final carry at T=0.
                var compiled = compileScanGradients(engine,List.of(scan.finalCarry().getFirst().mean()),List.of(seed),List.of(seq,weight),time == 0);
                try (var prepared = engine.prepare(compiled)) {
                    for (int batch : new int[]{0,1,7,32,7}) {
                        long[] ls = new long[batch]; double sumLengths = 0;
                        for (int r = 0; r < batch; r++) { ls[r] = time == 0 ? 0 : r%5; sumLengths += ls[r]; }
                        double[] ih = new double[batch*3], sx = new double[time*batch*3]; Arrays.fill(ih,.125); Arrays.fill(sx,.5);
                        var supplied = Map.of(initial.id(),typedStored(type,false,Shape.of(batch,3),ih),seq.id(),typedStored(type,true,Shape.of(time,batch,3),sx),
                                weight.id(),typedStored(type,true,Shape.of(3),new double[]{.25,.25,.25}),seed.id(),typedStored(type,false,Shape.scalar(),new double[]{1}),
                                lengths.id(),TensorFactory.fromFlatArray(new TensorDescriptor(DataType.INT64,Shape.of(batch),Optional.of(LayoutDescriptor.contiguous(Shape.of(batch))),false),Optional.empty(),ls));
                        try (var result = engine.runValues(prepared,compiled.inputs().stream().map(i -> supplied.get(i.tensorId())).toList())) {
                            var mean = result.materialize(result.publications().get(0),8).bytes();
                            assertEquals(batch == 0 ? Double.NaN : .125+.75*sumLengths/batch,type == DataType.FLOAT64 ? mean.getDouble() : mean.getFloat(),1e-7);
                            var dx = result.materialize(result.publications().get(1),1_000_000).bytes();
                            for (int t = 0; t < time; t++) for (int r = 0; r < batch; r++) for (int c = 0; c < 3; c++)
                                assertEquals(t < ls[r] ? 1d/(batch*3) : 0,type == DataType.FLOAT64 ? dx.getDouble() : dx.getFloat(),1e-7);
                            var dw = result.materialize(result.publications().get(2),24).bytes();
                            for (int c = 0; c < 3; c++) assertEquals(batch == 0 ? 0 : sumLengths/(batch*3),type == DataType.FLOAT64 ? dw.getDouble() : dw.getFloat(),1e-7);
                        }
                    }
                }
            }
        }
    }

    @Test void symbolicLengthMixedBatchConcurrentResultsAndProducerFailureRetryStayIsolated() throws Exception {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            var b = new DynamicDimension("B");
            Tensor h = typed(type,false,b,new StaticDimension(2)), x = typed(type,true,b,new StaticDimension(2));
            Tensor w = typed(type,true,new StaticDimension(2)), u = typed(type,true,new StaticDimension(2));
            Tensor initial = typed(type,false,b,new StaticDimension(2)), seq = typed(type,true,new StaticDimension(4),b,new StaticDimension(2));
            Tensor weight = typed(type,true,new StaticDimension(2)), raw = typed(type,false,b), length = raw.cast(DataType.INT64);
            var body = new ScanBody(List.of(h),List.of(x),List.of(w,u),List.of(h.add(x.mul(w)).add(x.mul(u))),List.of());
            var roots = new ArrayList<Tensor>();
            for (boolean reverse : List.of(false,true)) roots.addAll(Scan.apply(List.of(initial),List.of(seq),List.of(weight,weight),length,1,List.of(OptionalInt.of(0)),body,reverse).finalCarry());
            Tensor seedA = typed(type,false,b,new StaticDimension(2)), seedB = typed(type,false,b,new StaticDimension(2));
            try (var engine = Engine.standard()) {
                var compiled = engine.compile(roots,List.of(seedA,seedB),List.of(seq,weight));
                try (var prepared = engine.prepare(compiled); var executor = java.util.concurrent.Executors.newFixedThreadPool(2)) {
                    java.util.function.BiFunction<Integer,Boolean,List<Tensor>> inputs = (batch,invalid) -> {
                        double[] ih = new double[batch*2], sx = new double[8*batch], seeds = new double[batch*2], ls = new double[batch];
                        Arrays.fill(sx,.5); Arrays.fill(seeds,1); for (int r = 0; r < batch; r++) ls[r] = r%5;
                        if (invalid && batch > 0) ls[batch-1] = 5;
                        var values = Map.of(initial.id(),typedStored(type,false,Shape.of(batch,2),ih),seq.id(),typedStored(type,true,Shape.of(4,batch,2),sx),
                                weight.id(),typedStored(type,true,Shape.of(2),new double[]{.25,.25}),raw.id(),typedStored(type,false,Shape.of(batch),ls),
                                seedA.id(),typedStored(type,false,Shape.of(batch,2),seeds),seedB.id(),typedStored(type,false,Shape.of(batch,2),seeds));
                        return compiled.inputs().stream().map(i -> values.get(i.tensorId())).toList();
                    };
                    assertThrows(IllegalArgumentException.class,() -> engine.runValues(prepared,inputs.apply(7,true)));
                    var large = executor.submit(() -> engine.runValues(prepared,inputs.apply(32,false)));
                    var small = executor.submit(() -> engine.runValues(prepared,inputs.apply(7,false)));
                    try (var a = large.get(); var z = small.get(); var retry = engine.runValues(prepared,inputs.apply(1,false))) {
                        for (var result : List.of(a,z,retry)) {
                            int batch = (int)result.publications().getFirst().descriptor().shape().toLongArray()[0];
                            for (int port = 0; port < 2; port++) { var data = result.materialize(result.publications().get(port),100_000).bytes();
                                for (int r = 0; r < batch; r++) for (int j = 0; j < 2; j++) assertEquals(.25*(r%5),type == DataType.FLOAT64 ? data.getDouble() : data.getFloat()); }
                            var dx = result.materialize(result.publications().get(2),100_000).bytes();
                            for (int t = 0; t < 4; t++) for (int r = 0; r < batch; r++) for (int j = 0; j < 2; j++)
                                assertEquals(t < r%5 ? 1d : 0,type == DataType.FLOAT64 ? dx.getDouble() : dx.getFloat());
                            double total = 0; for (int r = 0; r < batch; r++) total += 2*(r%5);
                            var dw = result.materialize(result.publications().get(3),16).bytes();
                            for (int j = 0; j < 2; j++) assertEquals(total,type == DataType.FLOAT64 ? dw.getDouble() : dw.getFloat());
                        }
                        prepared.close(); assertThrows(IllegalStateException.class,() -> engine.runValues(prepared,null));
                        assertEquals(Shape.of(7,2),z.publications().getFirst().descriptor().shape());
                        assertDoesNotThrow(() -> z.materialize(z.publications().getFirst(),1000));
                    }
                }
            }
        }
    }

    @Test void symbolicLengthsTypedForwardAndActualReverseReuseOnePreparedOwner() {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64))
            for (boolean reverse : List.of(false,true)) for (boolean produced : List.of(false,true))
                for (int time : new int[]{4,0}) {
            var b = new DynamicDimension("batch"); int f = 3, h = 2;
            Tensor state = typed(type,true,b,new StaticDimension(h)), step = typed(type,true,b,new StaticDimension(f));
            Tensor w = typed(type,true,new StaticDimension(f),new StaticDimension(h));
            Tensor u = typed(type,true,new StaticDimension(h),new StaticDimension(h)), bias = typed(type,true,new StaticDimension(h));
            Tensor next = step.matmul(w).add(state.matmul(u)).add(bias).tanh();
            Tensor initial = typed(type,true,b,new StaticDimension(h)), sequence = typed(type,true,new StaticDimension(time),b,new StaticDimension(f));
            Tensor weight = typed(type,true,new StaticDimension(f),new StaticDimension(h));
            Tensor recurrent = typed(type,true,new StaticDimension(h),new StaticDimension(h)), offset = typed(type,true,new StaticDimension(h));
            Tensor lengths = typed(produced ? type : DataType.INT64,false,b);
            var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(weight,recurrent,offset),
                    produced ? lengths.cast(DataType.INT64) : lengths,1,List.of(OptionalInt.of(0)),
                    new ScanBody(List.of(state),List.of(step),List.of(w,u,bias),List.of(next),List.of(next)),reverse);
            Tensor finalSeed = typed(type,false,b,new StaticDimension(h)), emissionSeed = typed(type,false,new StaticDimension(time),b,new StaticDimension(h));
            try (var engine = Engine.standard()) {
                var compiled = compileScanGradients(engine,List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst()),
                        List.of(finalSeed,emissionSeed),List.of(initial,sequence,weight,recurrent,offset),time == 0);
                try (var prepared = engine.prepare(compiled)) {
                    var plan = prepared.execution().memoryPlan(); var schedule = prepared.execution().schedule();
                    var steps = List.copyOf(schedule.steps());
                    int run = 0;
                    for (int batch : new int[]{32,32,7,1,32,32,0,3,2}) {
                        long[] ls = new long[batch];
                        for (int row = 0; row < batch; row++) ls[row] = time == 0 ? 0 : run == 1 ? 4-row%5 : run >= 4 ? 4 : row%5;
                        if (batch == 1) ls[0] = 0;
                        double[] ih = typedData(type,batch*h,.01), sx = typedData(type,time*batch*f,.02);
                        double[] sw = typedData(type,f*h,.03), su = typedData(type,h*h,.025), sb = typedData(type,h,.04);
                        double[] fs = typedData(type,batch*h,.1), es = typedData(type,time*batch*h,.05);
                        for (int t = 0; t < time; t++) for (int row = 0; row < batch; row++) if (t >= ls[row]) {
                            Arrays.fill(sx,(t*batch+row)*f,(t*batch+row+1)*f,Double.NaN);
                            Arrays.fill(es,(t*batch+row)*h,(t*batch+row+1)*h,Double.NaN);
                        }
                        if (batch > 0 && ls[0] == 0) { ih[0] = -0d; fs[0] = -0d; }
                        var actual = new HashMap<TensorId,Tensor>();
                        actual.put(initial.id(),typedStored(type,true,Shape.of(batch,h),ih)); actual.put(sequence.id(),typedStored(type,true,Shape.of(time,batch,f),sx));
                        actual.put(weight.id(),typedStored(type,true,Shape.of(f,h),sw)); actual.put(recurrent.id(),typedStored(type,true,Shape.of(h,h),su));
                        actual.put(offset.id(),typedStored(type,true,Shape.of(h),sb)); actual.put(finalSeed.id(),typedStored(type,false,Shape.of(batch,h),fs));
                        actual.put(emissionSeed.id(),typedStored(type,false,Shape.of(time,batch,h),es));
                        actual.put(lengths.id(),produced ? typedStored(type,false,Shape.of(batch),Arrays.stream(ls).asDoubleStream().toArray())
                                : TensorFactory.fromFlatArray(new TensorDescriptor(DataType.INT64,Shape.of(batch),Optional.of(LayoutDescriptor.contiguous(Shape.of(batch))),false),Optional.empty(),ls));
                        var supplied = compiled.inputs().stream().map(i -> Objects.requireNonNull(actual.get(i.tensorId()),i.toString())).toList();
                        double[][] expected = lengthOracle(type,time,batch,f,h,reverse,ls,ih,sx,sw,su,sb,fs,es);
                        try (var result = engine.runValues(prepared,supplied)) {
                            assertEquals(7,result.resultCount());
                            for (int port = 0; port < expected.length; port++) {
                                assertPublication(compiled,result.publications().get(port),port,batch);
                                var value = result.materialize(result.publications().get(port),1_000_000);
                                assertEquals(expected[port].length*(long)type.byteWidth(),value.byteSize());
                                for (int i = 0; i < expected[port].length; i++) {
                                    double observed = type == DataType.FLOAT64 ? value.bytes().getDouble(i*8) : value.bytes().getFloat(i*4);
                                    assertEquals(expected[port][i],observed,type == DataType.FLOAT64 ? 2e-14 : 2e-6,
                                            type+" reverse="+reverse+" produced="+produced+" T="+time+" B="+batch+" port="+port+" cell="+i);
                                    if (expected[port][i] == 0) assertEquals(Double.doubleToRawLongBits(expected[port][i]),Double.doubleToRawLongBits(observed),"raw identity/+0");
                                }
                            }
                        }
                        assertSame(plan,prepared.execution().memoryPlan()); assertSame(schedule,prepared.execution().schedule());
                        for (int i = 0; i < steps.size(); i++) assertSame(steps.get(i),schedule.steps().get(i));
                        run++;
                    }
                }
            }
        }
    }

    static Tensor typed(DataType type,boolean grad,Dimension... dimensions) {
        Shape shape = Shape.ofDimensions(dimensions);
        return TensorFactory.create(new TensorDescriptor(type,shape,shape.isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(),grad));
    }
    static Tensor typedStored(DataType type,boolean grad,Shape shape,double[] values) {
        var d = new TensorDescriptor(type,shape,Optional.of(LayoutDescriptor.contiguous(shape)),grad);
        if (type == DataType.FLOAT64) return TensorFactory.fromFlatArray(d,Optional.empty(),values);
        float[] floats = new float[values.length]; for (int i = 0; i < values.length; i++) floats[i] = (float)values[i];
        return TensorFactory.fromFlatArray(d,Optional.empty(),floats);
    }
    static double[] typedData(DataType type,int n,double scale) {
        double[] values = new double[n]; for (int i = 0; i < n; i++) values[i] = represented(type,(i%11-5)*scale); return values;
    }
    static double represented(DataType type,double value) { return type == DataType.FLOAT64 ? value : (float)value; }

    /** Independent active-prefix unrolling; actual production reverse always comes from Compiler. */
    static double[][] lengthOracle(DataType type,int t,int b,int f,int h,boolean reverse,long[] lengths,
            double[] initial,double[] x,double[] w,double[] u,double[] bias,double[] finalSeed,double[] seed) {
        double[] carry = initial.clone(), history = new double[t*b*h], previous = new double[t*b*h];
        for (int step = 0; step < t; step++) {
            int time = reverse ? t-1-step : step; double[] updated = carry.clone();
            for (int row = 0; row < b; row++) if (time < lengths[row]) for (int col = 0; col < h; col++) {
                previous[(time*b+row)*h+col] = carry[row*h+col];
                double sum = 0, recurrent = 0;
                for (int k = 0; k < f; k++) sum = represented(type,sum+represented(type,x[(time*b+row)*f+k]*w[k*h+col]));
                for (int k = 0; k < h; k++) recurrent = represented(type,recurrent+represented(type,carry[row*h+k]*u[k*h+col]));
                updated[row*h+col] = represented(type,Math.tanh(represented(type,represented(type,sum+recurrent)+bias[col])));
                history[(time*b+row)*h+col] = updated[row*h+col];
            }
            carry = updated;
        }
        double[] dh = finalSeed.clone(), dx = new double[x.length], dw = new double[w.length], du = new double[u.length], db = new double[h];
        for (int step = 0; step < t; step++) {
            int time = reverse ? step : t-1-step; double[] local = new double[b*h]; boolean active = false;
            for (int row = 0; row < b; row++) if (time < lengths[row]) {
                active = true;
                for (int col = 0; col < h; col++) { int i = row*h+col; double y = history[time*b*h+i];
                    local[i] = represented(type,represented(type,dh[i]+seed[time*b*h+i])*represented(type,1-represented(type,y*y))); }
            }
            if (!active) continue;
            for (int row = 0; row < b; row++) if (time < lengths[row]) {
                for (int k = 0; k < f; k++) { double sum = 0;
                    for (int col = 0; col < h; col++) sum = represented(type,sum+represented(type,local[row*h+col]*w[k*h+col]));
                    dx[(time*b+row)*f+k] = sum; }
            }
            for (int k = 0; k < f; k++) for (int col = 0; col < h; col++) { double sum = 0;
                for (int row = 0; row < b; row++) if (time < lengths[row]) sum = represented(type,sum+represented(type,x[(time*b+row)*f+k]*local[row*h+col]));
                dw[k*h+col] = represented(type,dw[k*h+col]+sum); }
            for (int k = 0; k < h; k++) for (int col = 0; col < h; col++) { double sum = 0;
                for (int row = 0; row < b; row++) if (time < lengths[row]) sum = represented(type,sum+represented(type,previous[(time*b+row)*h+k]*local[row*h+col]));
                du[k*h+col] = represented(type,du[k*h+col]+sum); }
            for (int col = 0; col < h; col++) { double sum = 0;
                for (int row = 0; row < b; row++) if (time < lengths[row]) sum = represented(type,sum+local[row*h+col]);
                db[col] = represented(type,db[col]+sum); }
            double[] updated = dh.clone();
            for (int row = 0; row < b; row++) if (time < lengths[row]) for (int k = 0; k < h; k++) { double sum = 0;
                for (int col = 0; col < h; col++) sum = represented(type,sum+represented(type,local[row*h+col]*u[k*h+col]));
                updated[row*h+k] = sum; }
            dh = updated;
        }
        return new double[][]{carry,history,dh,dx,dw,du,db};
    }

    @Test void finalCarryMeanUsesBatchTimesHiddenIncludingEmptyAndZeroTime() {
        var b=new DynamicDimension("B"); int width=3;
        for (int time:new int[]{0,3}) {
            Tensor h=symbolic(true,b,new StaticDimension(width)), x=symbolic(true,b,new StaticDimension(width));
            Tensor initial=symbolic(true,b,new StaticDimension(width)), sequence=symbolic(true,new StaticDimension(time),b,new StaticDimension(width));
            var scan=Scan.apply(List.of(initial),List.of(sequence),List.of(),new ScanBody(List.of(h),List.of(x),List.of(),List.of(h.add(x)),List.of()),false);
            Tensor seed=tensor(false);
            try (var engine=Engine.standard()) {
                var compiled=compileScanGradients(engine,List.of(scan.finalCarry().getFirst().mean()),List.of(seed),List.of(initial,sequence),time == 0);
                try (var p=engine.prepare(compiled)) {
                    for (int batch:new int[]{0,1,40,64}) {
                        float[] ih=data(batch*width,.125f), sx=data(time*batch*width,.0625f), carry=ih.clone();
                        for (int t=0;t<time;t++) for (int i=0;i<carry.length;i++) carry[i]+=sx[t*carry.length+i];
                        double sum=0; for (float cell:carry) sum+=cell;
                        var values=Map.of(initial.id(),stored(true,Shape.of(batch,width),ih),sequence.id(),stored(true,Shape.of(time,batch,width),sx),seed.id(),stored(false,Shape.scalar(),new float[]{1}));
                        try (var result=engine.runValues(p,compiled.inputs().stream().map(i->values.get(i.tensorId())).toList())) {
                            assertEquals((float)(sum/(batch*width)),result.materialize(result.publications().getFirst(),4).bytes().getFloat());
                            for (int port=1;port<3;port++) { var out=result.materialize(result.publications().get(port),1_000_000);
                                for (int i=0;i<out.byteSize()/4;i++) assertEquals(1f/(batch*width),out.bytes().getFloat(i*4)); }
                        }
                    }
                }
            }
        }
    }
    @Test void symbolicBorrowFailureAttemptsReverseCleanupAndCloseRaceRollsBackPublication() throws Exception {
        for (boolean race:List.of(false,true)) {
            var delegate=new CpuEngineBackendComposition(io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration.open());
            var probe=new BorrowProbe();
            // Test-only delegation intercepts exactly the existing prefix-borrow seam. All
            // capability/preparation/publication work remains the real CPU implementation.
            var composition=(EngineBackendComposition)java.lang.reflect.Proxy.newProxyInstance(getClass().getClassLoader(),new Class<?>[]{EngineBackendComposition.class},(proxy,method,args)->{
                if (method.getName().equals("borrow") && args.length==2) return probe.borrow((io.github.pho001.synaptik.model.storage.HostTensorStorage)args[0],(long)args[1]);
                try { return method.invoke(delegate,args); } catch (java.lang.reflect.InvocationTargetException failure) { throw failure.getCause(); }
            });
            try (var engine=new Engine(new AdvancedEngine(composition))) {
                var b=new DynamicDimension("B"); Tensor h=symbolic(false,b,new StaticDimension(3)), x=symbolic(false,b,new StaticDimension(3)), bias=tensor(false,3);
                Tensor initial=symbolic(false,b,new StaticDimension(3)), sequence=symbolic(false,new StaticDimension(3),b,new StaticDimension(3)), offset=tensor(false,3);
                var scan=Scan.apply(List.of(initial),List.of(sequence),List.of(offset),new ScanBody(List.of(h),List.of(x),List.of(bias),List.of(h.add(x).add(bias)),List.of()),false);
                var compiled=engine.compile(scan.finalCarry());
                try (var p=engine.prepare(compiled)) {
                    var supplied=Map.of(initial.id(),stored(false,Shape.of(1,3),new float[]{1,2,3}),sequence.id(),stored(false,Shape.of(3,1,3),new float[9]),offset.id(),stored(false,Shape.of(3),new float[3]));
                    var inputs=compiled.inputs().stream().map(i->supplied.get(i.tensorId())).toList();
                    var missing=new ArrayList<>(inputs); var old=missing.getLast(); missing.set(missing.size()-1,TensorFactory.create(old.descriptor()));
                    assertThrows(IllegalStateException.class,()->engine.runValues(p,missing)); assertEquals(0,probe.borrows,"ALL storage validates before first borrow");
                    if (!race) {
                        probe.failBorrow=2; probe.failClose=true;
                        var failure=assertThrows(RuntimeException.class,()->engine.runValues(p,inputs)); assertSame(probe.primary,failure);
                        assertEquals(List.of(1,0),probe.closed); assertEquals(2,failure.getSuppressed().length);
                        probe.reset();
                        try (var result=engine.runValues(p,inputs)) { assertEquals(1,result.resultCount()); assertEquals(1,result.materialize(result.publications().getFirst(),12).bytes().getFloat()); }
                        assertEquals(List.of(2,1,0),probe.closed); assertTrue(inputs.stream().allMatch(t->t.hostStorage().orElseThrow().isAlive()));
                    } else {
                        probe.block=true;
                        try (var executor=java.util.concurrent.Executors.newFixedThreadPool(2)) {
                            var run=executor.submit(()->engine.runValues(p,inputs));
                            assertTrue(probe.entered.await(10,java.util.concurrent.TimeUnit.SECONDS));
                            var close=executor.submit(()->{ engine.close(); return null; });
                            long limit=System.nanoTime()+10_000_000_000L;
                            while (!engine.isClosed() && System.nanoTime()<limit) Thread.onSpinWait();
                            assertTrue(engine.isClosed()); probe.release.countDown();
                            var failure=assertThrows(java.util.concurrent.ExecutionException.class,()->run.get(10,java.util.concurrent.TimeUnit.SECONDS));
                            assertEquals("advanced engine is closed",failure.getCause().getMessage()); close.get(10,java.util.concurrent.TimeUnit.SECONDS);
                            assertEquals(List.of(2,1,0),probe.closed); assertTrue(inputs.stream().allMatch(t->t.hostStorage().orElseThrow().isAlive()));
                        } finally { probe.release.countDown(); }
                    }
                }
            }
        }
    }
    private static final class BorrowProbe {
        int borrows,failBorrow=-1; boolean failClose,block;
        final RuntimeException primary=new RuntimeException("injected borrow failure");
        final List<Integer> closed=new ArrayList<>();
        final java.util.concurrent.CountDownLatch entered=new java.util.concurrent.CountDownLatch(1),release=new java.util.concurrent.CountDownLatch(1);
        io.github.pho001.synaptik.runtime.resource.BufferRepresentation borrow(io.github.pho001.synaptik.model.storage.HostTensorStorage s,long bytes) throws InterruptedException {
            int index=borrows++; if (index==failBorrow) throw primary;
            if (block && index==1) { entered.countDown(); if (!release.await(10,java.util.concurrent.TimeUnit.SECONDS)) throw new AssertionError("borrow barrier timed out"); }
            return new io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferRepresentation(s.dataType(),bytes,s.segment().asSlice(0,bytes)) {
                boolean closedFlag;
                @Override protected boolean isClosed() { return closedFlag; }
                @Override public void close() { if (!closedFlag) { closedFlag=true; closed.add(index); if (failClose) throw new RuntimeException("close-"+index); } }
            };
        }
        void reset() { borrows=0; failBorrow=-1; failClose=false; closed.clear(); }
    }

    @Test void emissionOnlySigmoidScalarAlgebraAndMeanUseActualBatchDivisor() {
        for (boolean reverse:List.of(false,true)) for (int time:new int[]{0,3}) {
            var b=new DynamicDimension("B"); int width=3;
            Tensor h=symbolic(true,b,new StaticDimension(width)), x=symbolic(true,b,new StaticDimension(width));
            Tensor initial=symbolic(true,b,new StaticDimension(width)), sequence=symbolic(true,new StaticDimension(time),b,new StaticDimension(width));
            Tensor next=h.add(x).sub(io.github.pho001.synaptik.model.datatype.ScalarValue.float32(.125f))
                    .mul(io.github.pho001.synaptik.model.datatype.ScalarValue.float32(.5f))
                    .div(io.github.pho001.synaptik.model.datatype.ScalarValue.float32(2f))
                    .neg().sigmoid().cast(DataType.FLOAT32).reshape(Shape.ofDimensions(b,new StaticDimension(width))).contiguous();
            var scan=Scan.apply(List.of(initial),List.of(sequence),List.of(),new ScanBody(List.of(h),List.of(x),List.of(),List.of(next),List.of(next)),reverse);
            Tensor emission=scan.outputs().getFirst(), mean=emission.mean();
            Tensor es=symbolic(false,new StaticDimension(time),b,new StaticDimension(width)), ms=tensor(false);
            try (var engine=Engine.standard()) {
                // An empty emission has no route even to initial; a mean does not add one.
                var compiled=compileScanGradients(engine,List.of(emission,mean),List.of(es,ms),List.of(initial,sequence),time == 0);
                try (var p=engine.prepare(compiled)) {
                    for (int batch:new int[]{64,40,1,64,0,3}) {
                        float[] ih=data(batch*width,.05f), sx=data(time*batch*width,.03f), seed=data(sx.length,.01f), carry=ih.clone(), history=new float[sx.length];
                        for (int step=0;step<time;step++) {
                            int t=reverse?time-1-step:step;
                            for (int i=0;i<carry.length;i++) { float v=-(((carry[i]+sx[t*carry.length+i])-.125f)*.5f/2f); carry[i]=(float)(1.0/(1.0+Math.exp(-v))); history[t*carry.length+i]=carry[i]; }
                        }
                        double total=0; for (float cell:history) total+=cell;
                        float expectedMean=(float)(total/history.length); float[] dh=new float[carry.length], dx=new float[sx.length];
                        float divisor=1f/history.length;
                        for (int step=0;step<time;step++) {
                            int t=reverse?step:time-1-step;
                            for (int i=0;i<dh.length;i++) { float y=history[t*dh.length+i]; dh[i]=((dh[i]+(seed[t*dh.length+i]+divisor))*(y*(1-y)))*-.25f; dx[t*dh.length+i]=dh[i]; }
                        }
                        var values=Map.of(initial.id(),stored(true,Shape.of(batch,width),ih),sequence.id(),stored(true,Shape.of(time,batch,width),sx),
                                es.id(),stored(false,Shape.of(time,batch,width),seed),ms.id(),stored(false,Shape.scalar(),new float[]{1}));
                        try (var result=engine.runValues(p,compiled.inputs().stream().map(i->values.get(i.tensorId())).toList())) {
                            float[][] expected={history,new float[]{expectedMean},dh,dx};
                            for (int port=0;port<expected.length;port++) {
                                var out=result.materialize(result.publications().get(port),1_000_000);
                                assertEquals(expected[port].length*4L,out.byteSize());
                                for (int i=0;i<expected[port].length;i++) assertEquals(expected[port][i],out.bytes().getFloat(i*4),3e-6,"port="+port+" B="+batch);
                            }
                        }
                    }
                }
            }
        }
    }

    @Test void scopedOnlySymbolsFailBeforeBackendPreparationAndUnusedTrailingDomainsFailBeforeStorage() throws Exception {
        var cpu=io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration.open();
        var calls=new java.util.concurrent.atomic.AtomicInteger();
        try (var engine=new Engine(new AdvancedEngine(new CpuEngineBackendComposition(cpu),artifacts->{ calls.incrementAndGet(); throw new AssertionError("analyzer must not run"); }))) {
            Tensor h=tensor(false,3), x=tensor(false,3);
            var scan=Scan.apply(List.of(tensor(false,3)),List.of(tensor(false,0,3)),List.of(),new ScanBody(List.of(h),List.of(x),List.of(),List.of(h.add(x)),List.of()),false);
            var compiled=engine.compile(scan.finalCarry());
            var forged=withUnusedLocal(engine,compiled,new TensorDescriptor(DataType.FLOAT32,Shape.ofDimensions(new DynamicDimension("B")),Optional.empty(),false));
            assertEquals("scoped batch symbol has no outer binding",assertThrows(IllegalArgumentException.class,()->engine.prepare(forged)).getMessage());
            assertEquals(0,calls.get());
        }
        var b=new DynamicDimension("B"); Tensor h=symbolic(false,b,new StaticDimension(3)), x=symbolic(false,b,new StaticDimension(3));
        var scan=Scan.apply(List.of(symbolic(false,b,new StaticDimension(3))),List.of(symbolic(false,new StaticDimension(0),b,new StaticDimension(3))),List.of(),
                new ScanBody(List.of(h),List.of(x),List.of(),List.of(h.add(x)),List.of()),false);
        try (var engine=Engine.standard()) {
            var compiled=engine.compile(scan.finalCarry());
            try (var p=engine.prepare(compiled)) {
                var forged=withUnusedLocal(engine,compiled,new TensorDescriptor(DataType.FLOAT32,Shape.of(0,Integer.MAX_VALUE,Integer.MAX_VALUE),Optional.empty(),false));
                // Existing test-only construction seam isolates the scoped preindex from CPU body
                // qualification, which independently rejects this forged dead NEG geometry.
                assertThrows(ArithmeticException.class,()->new PreparedExecution(engine,forged,p.execution(),p.inputAdapters(),p.publicationAdapters(),p.batchPlan()),
                        "fixed nonzero features overflow independently of outer T0/B0 during preindexing");
            }
        }
    }

    private static CompiledGraph withUnusedLocal(Engine owner,CompiledGraph source,TensorDescriptor descriptor) throws Exception {
        var a=source.artifacts(); var graph=a.graph(); var first=graph.nodes().getFirst(); var body=first.scanBody().orElseThrow(); var local=body.graph();
        var values=new ArrayList<>(local.values()); var id=new io.github.pho001.synaptik.model.graph.ValueId(values.stream().mapToLong(v->v.id().value()).max().orElseThrow()+1);
        values.add(new io.github.pho001.synaptik.model.graph.GraphValue(id,descriptor));
        var nodes=new ArrayList<>(local.nodes()); var nodeId=new io.github.pho001.synaptik.model.graph.NodeId(nodes.stream().mapToLong(n->n.id().value()).max().orElse(-1)+1);
        var extra=new io.github.pho001.synaptik.model.graph.CompiledNode(nodeId,new io.github.pho001.synaptik.model.operation.Operation(
                io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind.NEG,io.github.pho001.synaptik.model.operation.NoOperationAttrs.INSTANCE),List.of(local.inputs().getFirst()),List.of(id));
        nodes.add(extra); var phases=new LinkedHashMap<>(local.nodePhases()); phases.put(nodeId,io.github.pho001.synaptik.model.graph.GraphPhase.FORWARD);
        var scoped=new io.github.pho001.synaptik.model.graph.CompiledScanBody(new io.github.pho001.synaptik.model.graph.CompiledGraphModel(values,nodes,local.inputs(),local.outputs(),phases),body.nextCarry(),body.emissions(),body.residuals(),body.demandedOutputIndices());
        var outerNodes=new ArrayList<>(graph.nodes()); outerNodes.set(0,new io.github.pho001.synaptik.model.graph.CompiledNode(first.id(),first.operation(),first.inputs(),first.outputs(),Optional.of(scoped)));
        var replacement=new io.github.pho001.synaptik.model.graph.CompiledGraphModel(graph.values(),outerNodes,graph.inputs(),graph.outputs(),graph.nodePhases());
        var constructor=PublicationPlan.class.getDeclaredConstructor(io.github.pho001.synaptik.model.graph.CompiledGraphModel.class,List.class,List.class); constructor.setAccessible(true);
        var publication=constructor.newInstance(replacement,a.publication().forwardBindings(),a.publication().gradientBindings());
        return new CompiledGraph(owner,new CompileArtifacts(a.mode(),replacement,a.partitions(),io.github.pho001.synaptik.planning.memory.LogicalMemoryPlanning.plan(replacement,a.partitions()),publication,a.constants(),a.diagnostics(),new DerivativeGraphMetadata(replacement,a.derivatives().derivativeOrderByNode())));
    }

    @Test void sharedCarryHistoryAndGradientExecuteBesideEmptyRowsAndT0CopiesRaw() {
        for (boolean reverse : List.of(false,true)) for (int time : new int[] {0,3}) {
            var b = new DynamicDimension("B"); Tensor h = symbolic(true,b,new StaticDimension(3)), x = symbolic(true,b,new StaticDimension(3));
            Tensor q = tensor(true,3,3), w = tensor(true,3,3), unusedFormal = tensor(true,2);
            var body = new ScanBody(List.of(h,q),List.of(x),List.of(w,unusedFormal),List.of(h.add(x),q.add(w)),List.of(q));
            Tensor initial = symbolic(true,b,new StaticDimension(3)), initialQ = tensor(true,3,3), weight = tensor(true,3,3), unused = tensor(true,2);
            Tensor sequence = symbolic(true,new StaticDimension(time),b,new StaticDimension(3));
            var scan = Scan.apply(List.of(initial,initialQ),List.of(sequence),List.of(weight,unused),body,reverse);
            Tensor hs = symbolic(false,b,new StaticDimension(3)), qs = tensor(false,3,3), es = tensor(false,time,3,3);
            var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
            try (Engine engine = Engine.standard()) {
                // weight reaches final Q at positive T; unused has no body route at any T.
                assertThrows(IllegalArgumentException.class,() -> engine.compile(roots,List.of(hs,qs,es),List.of(weight,unused)),
                        "the unused actual has no selected finite-time route");
                var compiled = compileScanGradients(engine,roots,List.of(hs,qs,es),List.of(weight,unused),true);
                try (var p = engine.prepare(compiled)) {
                    for (int batch : new int[] {0,1,3,40}) {
                        float[] qv = data(9,.05f), wv = data(9,.025f), qsv = data(9,.1f), esv = data(time*9,.01f);
                        if (time == 0) { qv[0] = -0f; qv[1] = Float.intBitsToFloat(0x7fc12345); qv[2] = Float.intBitsToFloat(1); }
                        Map<TensorId,Tensor> supplied = Map.of(initial.id(),stored(true,Shape.of(batch,3),data(batch*3,.01f)),
                                initialQ.id(),stored(true,Shape.of(3,3),qv),weight.id(),stored(true,Shape.of(3,3),wv),unused.id(),stored(true,Shape.of(2),new float[] {9,10}),
                                sequence.id(),stored(true,Shape.of(time,batch,3),data(time*batch*3,.02f)),hs.id(),stored(false,Shape.of(batch,3),data(batch*3,.1f)),
                                qs.id(),stored(false,Shape.of(3,3),qsv),es.id(),stored(false,Shape.of(time,3,3),esv));
                        float[] state = qv.clone(), history = new float[time*9], dw = new float[9], dq = qsv.clone();
                        for (int step = 0; step < time; step++) {
                            int t = reverse ? time-1-step : step; System.arraycopy(state,0,history,t*9,9);
                            for (int i = 0; i < 9; i++) state[i] += wv[i];
                        }
                        for (int step = 0; step < time; step++) {
                            int t = reverse ? step : time-1-step;
                            for (int i = 0; i < 9; i++) { dw[i] += dq[i]; dq[i] += esv[t*9+i]; }
                        }
                        try (var r = engine.runValues(p,compiled.inputs().stream().map(i -> Objects.requireNonNull(supplied.get(i.tensorId()))).toList())) {
                            assertEquals(5,r.resultCount()); assertEquals(Shape.of(time,3,3),r.publications().get(2).descriptor().shape());
                            for (int port : new int[] {1,2,3,4}) {
                                float[] expected = port == 1 ? state : port == 2 ? history : port == 3 ? dw : new float[2];
                                var value = r.materialize(r.publications().get(port),1_000_000); assertEquals(expected.length*4L,value.byteSize());
                                for (int i = 0; i < expected.length; i++) {
                                    float actual = value.bytes().getFloat(i*4);
                                    if (time == 0 && port == 1) assertEquals(Float.floatToRawIntBits(expected[i]),Float.floatToRawIntBits(actual));
                                    else assertEquals(expected[i],actual,2e-7f);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Test void failedAdmissionRetryConcurrentMixedBatchAndOpenResultsRetainExistingLifecycle() throws Exception {
        var b = new DynamicDimension("B"); Tensor h = symbolic(false,b,new StaticDimension(3)), x = symbolic(false,b,new StaticDimension(3));
        Tensor initial = symbolic(false,b,new StaticDimension(3)), sequence = symbolic(false,new StaticDimension(3),b,new StaticDimension(3));
        var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(),new ScanBody(List.of(h),List.of(x),List.of(),List.of(h.add(x).sigmoid()),List.of()),false);
        try (Engine engine = Engine.standard()) {
            var compiled = engine.compile(scan.finalCarry()); var p = engine.prepare(compiled);
            java.util.function.IntFunction<List<Tensor>> inputs = batch -> compiled.inputs().stream().map(i -> i.tensorId().equals(initial.id())
                    ? stored(false,Shape.of(batch,3),data(batch*3,.01f)) : stored(false,Shape.of(3,batch,3),data(9*batch,.02f))).toList();
            var bad = new ArrayList<>(inputs.apply(1)); bad.set(0,tensor(false,1,3));
            assertThrows(IllegalStateException.class,() -> engine.runValues(p,bad));
            var huge = compiled.inputs().stream().map(i -> i.tensorId().equals(initial.id()) ? tensor(false,(long)Integer.MAX_VALUE+1,3)
                    : tensor(false,3,(long)Integer.MAX_VALUE+1,3)).toList();
            assertThrows(ArithmeticException.class,() -> engine.runValues(p,huge),"numeric rejection precedes absent storage");
            try (var first = engine.runValues(p,inputs.apply(1)); var executor = java.util.concurrent.Executors.newFixedThreadPool(2)) {
                var a = executor.submit(() -> { try (var r = engine.runValues(p,inputs.apply(40))) { return r.materialize(r.publications().getFirst(),1_000_000).byteSize(); } });
                var c = executor.submit(() -> { try (var r = engine.runValues(p,inputs.apply(64))) { return r.materialize(r.publications().getFirst(),1_000_000).byteSize(); } });
                assertEquals(480,a.get(10,java.util.concurrent.TimeUnit.SECONDS)); assertEquals(768,c.get(10,java.util.concurrent.TimeUnit.SECONDS));
                p.close(); assertEquals(12,first.materialize(first.publications().getFirst(),1_000_000).byteSize());
                assertEquals("prepared execution is closed",assertThrows(IllegalStateException.class,() -> engine.runValues(p,null)).getMessage());
                assertThrows(IllegalStateException.class,() -> engine.run(p,List.of()));
            } finally { p.close(); }
        }
    }
    @Test void expandedSequenceTimeStrideIsIndependentOfNominalFramesInForwardAndActualReverse() {
        for (boolean reverse : List.of(false,true)) for (boolean repeatTime : List.of(false,true)) for (int time : new int[] {0,3}) {
            var b = new DynamicDimension("B"); int width = 4;
            Tensor h = symbolic(true,b,new StaticDimension(width)), x = symbolic(true,b,new StaticDimension(width));
            Tensor initial = symbolic(true,b,new StaticDimension(width));
            Tensor source = repeatTime ? symbolic(true,b,new StaticDimension(width)) : symbolic(true,new StaticDimension(time),b,new StaticDimension(1));
            Tensor sequence = source.expand(Shape.ofDimensions(new StaticDimension(time),b,new StaticDimension(width)));
            Tensor next = h.add(x).tanh();
            var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(),new ScanBody(List.of(h),List.of(x),List.of(),List.of(next),List.of(next)),reverse);
            Tensor fs = symbolic(false,b,new StaticDimension(width)), es = symbolic(false,new StaticDimension(time),b,new StaticDimension(width));
            try (Engine engine = Engine.standard()) {
                var compiled = compileScanGradients(engine,List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst()),List.of(fs,es),List.of(initial,source),time == 0);
                try (var p = engine.prepare(compiled)) {
                    for (int batch : new int[] {0,1,4,7}) {
                        var sourceShape = repeatTime ? Shape.of(batch,width) : Shape.of(time,batch,1);
                        float[] ih = data(batch*width,.025f), sx = data(Math.toIntExact(sourceShape.knownElementCount().orElseThrow()),.04f),
                                fsv = data(batch*width,.1f), esv = data(time*batch*width,.03f);
                        Map<TensorId,Tensor> actual = Map.of(initial.id(),stored(true,Shape.of(batch,width),ih),source.id(),stored(true,sourceShape,sx),
                                fs.id(),stored(false,Shape.of(batch,width),fsv),es.id(),stored(false,Shape.of(time,batch,width),esv));
                        float[] carry = ih.clone(), history = new float[time*batch*width];
                        for (int step = 0; step < time; step++) {
                            int t = reverse ? time-1-step : step;
                            for (int row = 0; row < batch; row++) for (int col = 0; col < width; col++) {
                                int index = row*width+col, from = repeatTime ? index : t*batch+row;
                                carry[index] = (float)Math.tanh(carry[index]+sx[from]); history[t*batch*width+index] = carry[index];
                            }
                        }
                        float[] dh = fsv.clone(), ds = new float[sx.length];
                        for (int step = 0; step < time; step++) {
                            int t = reverse ? step : time-1-step;
                            for (int row = 0; row < batch; row++) for (int col = 0; col < width; col++) {
                                int index = row*width+col; float y = history[t*batch*width+index];
                                dh[index] = (dh[index]+esv[t*batch*width+index])*(1-y*y);
                                ds[repeatTime ? index : t*batch+row] += dh[index];
                            }
                        }
                        float[][] expected = {carry,history,dh,ds};
                        try (var r = engine.runValues(p,compiled.inputs().stream().map(i -> Objects.requireNonNull(actual.get(i.tensorId()))).toList())) {
                            for (int port = 0; port < expected.length; port++) {
                                var out = r.materialize(r.publications().get(port),1_000_000); assertEquals(expected[port].length*4L,out.byteSize());
                                for (int i = 0; i < expected[port].length; i++) assertEquals(expected[port][i],out.bytes().getFloat(i*4),3e-6,"port="+port+" B="+batch+" repeatTime="+repeatTime);
                            }
                        }
                    }
                }
            }
        }
    }
    @Test void outerTransposeAndExpandedInitialKeepActualMappingThroughBothScans() {
        for (int time : new int[] {0,3}) {
            var b = new DynamicDimension("B"); int f = 4, h = 3;
            Tensor state = symbolic(true,b,new StaticDimension(h)), step = symbolic(true,b,new StaticDimension(f)), w = tensor(true,f,h);
            Tensor next = step.matmul(w).add(state).tanh();
            Tensor scalar = tensor(true), weight = tensor(true,h,f), seq = symbolic(true,new StaticDimension(time),b,new StaticDimension(f));
            var initial = scalar.expand(Shape.ofDimensions(b,new StaticDimension(h)));
            var scan = Scan.apply(List.of(initial),List.of(seq),List.of(weight.permute(1,0)),
                    new ScanBody(List.of(state),List.of(step),List.of(w),List.of(next),List.of(next)),false);
            Tensor finalSeed = symbolic(false,b,new StaticDimension(h));
            var request = new FunctionalGradientRequest.Stage(List.of(new FunctionalGradientRequest.ForwardTensorReference(scan.finalCarry().getFirst())),
                    List.of(Optional.of(finalSeed)),List.of(seq,weight,scalar),false,time == 0
                            ? FunctionalGradientRequest.DisconnectedPolicy.ZERO : FunctionalGradientRequest.DisconnectedPolicy.ERROR);
            var cpu = new CpuCapabilityProvider(); var backend = CpuCapabilityProvider.CPU_BACKEND_ID;
            var artifacts = GraphCompilationPort.compile(CompileMode.FORWARD_AND_BACKWARD,scan.finalCarry(),Optional.of(new FunctionalGradientRequest(List.of(request))),
                    GraphOptimizationConfig.disabled(),BackendIntent.unconstrained(),PartitionScoringConfig.neutral(),List.of(cpu),
                    List.of(new BackendAvailabilitySnapshot(backend,Map.of(new BackendDeviceId(backend,"cpu"),DeviceClass.CPU))));
            try (Engine engine = Engine.standard()) {
                var compiled = new CompiledGraph(engine,artifacts);
                try (var p = engine.prepare(compiled)) {
                    for (int batch : new int[] {1,7,3,4,0}) {
                        float[] sx = data(time*batch*f,.01f), sw = data(h*f,.03f), fs = data(batch*h,.04f);
                        Map<TensorId,Tensor> actual = Map.of(scalar.id(),stored(true,Shape.scalar(),new float[] {-.125f}),weight.id(),stored(true,Shape.of(h,f),sw),
                                seq.id(),stored(true,Shape.of(time,batch,f),sx),finalSeed.id(),stored(false,Shape.of(batch,h),fs));
                        var supplied = compiled.inputs().stream().map(i -> Objects.requireNonNull(actual.get(i.tensorId()))).toList();
                        float[] carry = new float[batch*h], history = new float[time*batch*h]; Arrays.fill(carry,-.125f);
                        for (int t = 0; t < time; t++) for (int row = 0; row < batch; row++) for (int col = 0; col < h; col++) {
                            float sum = 0; for (int k = 0; k < f; k++) sum += sx[(t*batch+row)*f+k]*sw[col*f+k];
                            carry[row*h+col] = (float)Math.tanh(sum+carry[row*h+col]); history[(t*batch+row)*h+col] = carry[row*h+col];
                        }
                        float[] dh = fs.clone(), dx = new float[sx.length], dw = new float[sw.length];
                        for (int t = time-1; t >= 0; t--) {
                            for (int i = 0; i < dh.length; i++) dh[i] *= 1-history[t*batch*h+i]*history[t*batch*h+i];
                            for (int row = 0; row < batch; row++) for (int k = 0; k < f; k++) for (int col = 0; col < h; col++) dx[(t*batch+row)*f+k] += dh[row*h+col]*sw[col*f+k];
                            for (int col = 0; col < h; col++) for (int k = 0; k < f; k++) {
                                float sum = 0; for (int row = 0; row < batch; row++) sum += sx[(t*batch+row)*f+k]*dh[row*h+col]; dw[col*f+k] += sum;
                            }
                        }
                        float ds = 0; for (float cell : dh) ds += cell;
                        float[][] expected = {carry,dx,dw,new float[] {ds}};
                        try (var r = engine.runValues(p,supplied)) {
                            assertEquals(expected.length,r.resultCount());
                            for (int port = 0; port < expected.length; port++) {
                                var out = r.materialize(r.publications().get(port),1_000_000); assertEquals(expected[port].length*4L,out.byteSize());
                                for (int i = 0; i < expected[port].length; i++) assertEquals(expected[port][i],out.bytes().getFloat(i*4),2e-5,"port="+port+" B="+batch);
                            }
                        }
                    }
                }
            }
        }
    }
    @Test void forwardAndActualCompilerReverseReuseOnePreparation() {
        for (boolean reverse : List.of(false,true)) for (int time : new int[] {0,3}) for (int features : new int[] {3,4}) {
            int hidden = 3;
            var b = new DynamicDimension("B");
            Tensor h = symbolic(true,b,new StaticDimension(hidden)), x = symbolic(true,b,new StaticDimension(features));
            Tensor w = tensor(true,features,hidden), u = tensor(true,hidden,hidden), bias = tensor(true,hidden);
            Tensor next = x.matmul(w).add(h.matmul(u)).add(bias).tanh();
            var body = new ScanBody(List.of(h),List.of(x),List.of(w,u,bias),List.of(next),List.of(next));
            Tensor initial = symbolic(true,b,new StaticDimension(hidden));
            Tensor sequence = symbolic(true,new StaticDimension(time),b,new StaticDimension(features));
            Tensor weight = tensor(true,features,hidden), recurrent = tensor(true,hidden,hidden), offset = tensor(true,hidden);
            var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(weight,recurrent,offset),body,reverse);
            var roots = List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
            Tensor finalSeed = symbolic(false,b,new StaticDimension(hidden));
            Tensor emissionSeed = symbolic(false,new StaticDimension(time),b,new StaticDimension(hidden));
            var stage = new FunctionalGradientRequest.Stage(roots.stream().map(t -> (FunctionalGradientRequest.OutputReference)
                    new FunctionalGradientRequest.ForwardTensorReference(t)).toList(),
                    List.of(Optional.of(finalSeed),Optional.of(emissionSeed)),List.of(initial,sequence,weight,recurrent,offset),false,
                    time == 0 ? FunctionalGradientRequest.DisconnectedPolicy.ZERO : FunctionalGradientRequest.DisconnectedPolicy.ERROR);
            var cpu = new CpuCapabilityProvider(); var backend = CpuCapabilityProvider.CPU_BACKEND_ID;
            var checked = new io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider() {
                public BackendId backendId() { return backend; }
                public boolean supports(io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery q) {
                    if (q.scanBody().isPresent()) io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLowering.analyze(q);
                    return cpu.supports(q);
                }
            };
            var artifacts = GraphCompilationPort.compile(CompileMode.FORWARD_AND_BACKWARD,roots,
                    Optional.of(new FunctionalGradientRequest(List.of(stage))),GraphOptimizationConfig.disabled(),
                    BackendIntent.unconstrained(),PartitionScoringConfig.neutral(),List.of(checked),
                    List.of(new BackendAvailabilitySnapshot(backend,Map.of(new BackendDeviceId(backend,"cpu"),DeviceClass.CPU))));
            try (Engine engine = Engine.standard()) {
                var compiled = new CompiledGraph(engine,artifacts);
                try (var prepared = engine.prepare(compiled)) {
                    var plan = prepared.execution().memoryPlan(); var schedule = prepared.execution().schedule();
                    var retainedSteps=List.copyOf(schedule.steps());
                    for (int batch : new int[] {64,40,1,64,0,features,hidden}) {
                        float[] ih = data(batch*hidden,.01f), sx = data(time*batch*features,.02f),
                                sw = data(features*hidden,.03f), su = data(hidden*hidden,.025f), sb = data(hidden,.04f),
                                fs = data(batch*hidden,.1f), es = data(time*batch*hidden,.05f);
                        Map<io.github.pho001.synaptik.model.tensor.TensorId,Tensor> actual = Map.of(
                                initial.id(),stored(true,Shape.of(batch,hidden),ih),sequence.id(),stored(true,Shape.of(time,batch,features),sx),
                                weight.id(),stored(true,Shape.of(features,hidden),sw),offset.id(),stored(true,Shape.of(hidden),sb),
                                recurrent.id(),stored(true,Shape.of(hidden,hidden),su),
                                finalSeed.id(),stored(false,Shape.of(batch,hidden),fs),emissionSeed.id(),stored(false,Shape.of(time,batch,hidden),es));
                        var supplied = compiled.inputs().stream().map(i -> Objects.requireNonNull(actual.get(i.tensorId()),"input "+i)).toList();
                        float[][] expected = oracle(time,batch,features,hidden,reverse,ih,sx,sw,su,sb,fs,es);
                        try (var result = engine.runValues(prepared,supplied)) {
                            assertEquals(expected.length,result.resultCount());
                            for (int p = 0; p < expected.length; p++) {
                                assertPublication(compiled,result.publications().get(p),p,batch);
                                var value = result.materialize(result.publications().get(p),10_000_000);
                                assertEquals(expected[p].length*4L,value.byteSize(),"port "+p);
                                for (int i = 0; i < expected[p].length; i++) assertEquals(expected[p][i],value.bytes().getFloat(i*4),2e-5f,"B="+batch+" port="+p+" cell="+i);
                            }
                        }
                        assertSame(plan,prepared.execution().memoryPlan()); assertSame(schedule,prepared.execution().schedule());
                        for (int s=0;s<retainedSteps.size();s++) assertSame(retainedSteps.get(s),schedule.steps().get(s));
                    }
                }
            }
        }
    }

    private static void assertPublication(CompiledGraph graph,RunResult.Publication publication,int index,int batch) {
        var spec=graph.publicationSpecs().get(index); var template=spec.descriptor;
        Shape shape=Shape.of(template.shape().dimensions().stream().mapToLong(d->d instanceof DynamicDimension?batch:d.staticSize().orElseThrow()).toArray());
        assertEquals(index,publication.index()); assertEquals(spec.tensorId,publication.tensorId()); assertEquals(spec.role,publication.role());
        assertEquals(template.dataType(),publication.descriptor().dataType());
        assertEquals(shape,publication.descriptor().shape());
        assertEquals(template.requiresGrad(),publication.descriptor().requiresGrad());
        assertEquals(spec.role==RunResult.Role.GRADIENT?OptionalInt.of(spec.derivativeOrder):OptionalInt.empty(),publication.derivativeOrder());
        assertEquals(spec.role==RunResult.Role.GRADIENT?OptionalInt.of(spec.targetIndex):OptionalInt.empty(),publication.targetIndex());
    }


    /**
     * Uses the ordinary strict entry unless a target has a source-proved absent finite route.
     * Runtime B=0 or all-zero lengths do not change this compile-time policy.
     */
    private static CompiledGraph compileScanGradients(Engine engine,List<Tensor> roots,List<Tensor> seeds,
            List<Tensor> targets,boolean hasDisconnectedTarget) {
        if (!hasDisconnectedTarget) return engine.compile(roots,seeds,targets);
        var stage = new FunctionalGradientRequest.Stage(roots.stream().map(root ->
                (FunctionalGradientRequest.OutputReference)new FunctionalGradientRequest.ForwardTensorReference(root)).toList(),
                seeds.stream().map(Optional::of).toList(),targets,false,FunctionalGradientRequest.DisconnectedPolicy.ZERO);
        var cpu = new CpuCapabilityProvider(); var backend = cpu.backendId();
        var artifacts = GraphCompilationPort.compile(CompileMode.FORWARD_AND_BACKWARD,roots,
                Optional.of(new FunctionalGradientRequest(List.of(stage))),GraphOptimizationConfig.disabled(),
                BackendIntent.unconstrained(),PartitionScoringConfig.neutral(),List.of(cpu),
                List.of(new BackendAvailabilitySnapshot(backend,Map.of(new BackendDeviceId(backend,"cpu"),DeviceClass.CPU))));
        return new CompiledGraph(engine,artifacts);
    }

    static Tensor symbolic(boolean grad,Dimension... dimensions) {
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,Shape.ofDimensions(dimensions),Optional.empty(),grad));
    }
    static Tensor tensor(boolean grad,long... dimensions) {
        var shape = Shape.of(dimensions);
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,shape,Optional.of(LayoutDescriptor.contiguous(shape)),grad));
    }
    static Tensor stored(boolean grad,Shape shape,float[] data) {
        return TensorFactory.fromFlatArray(new TensorDescriptor(DataType.FLOAT32,shape,Optional.of(LayoutDescriptor.contiguous(shape)),grad),Optional.empty(),data);
    }
    static float[] data(int n,float scale) {
        float[] data = new float[n]; for (int i = 0; i < n; i++) data[i] = (i%11-5)*scale; return data;
    }
    static float[][] oracle(int t,int b,int f,int h,boolean reverse,float[] initial,float[] x,float[] w,float[] u,float[] bias,float[] finalSeed,float[] seed) {
        float[] carry = initial.clone(), history = new float[t*b*h];
        for (int step = 0; step < t; step++) {
            int time = reverse ? t-1-step : step;
            float[] updated = new float[carry.length];
            for (int row = 0; row < b; row++) for (int col = 0; col < h; col++) {
                float sum = 0; for (int k = 0; k < f; k++) sum += x[(time*b+row)*f+k]*w[k*h+col];
                float recurrent = 0; for (int k = 0; k < h; k++) recurrent += carry[row*h+k]*u[k*h+col];
                float next = (float)Math.tanh((sum+recurrent)+bias[col]);
                updated[row*h+col] = next; history[(time*b+row)*h+col] = next;
            }
            carry = updated;
        }
        float[] dh = finalSeed.clone(), dx = new float[x.length], dw = new float[w.length], du = new float[u.length], db = new float[h];
        for (int step = 0; step < t; step++) {
            int time = reverse ? step : t-1-step;
            float[] local = new float[b*h];
            for (int i = 0; i < local.length; i++) { float y = history[time*b*h+i]; local[i] = (dh[i]+seed[time*b*h+i])*(1-y*y); }
            for (int row = 0; row < b; row++) for (int k = 0; k < f; k++) {
                float sum = 0; for (int col = 0; col < h; col++) sum += local[row*h+col]*w[k*h+col]; dx[(time*b+row)*f+k] = sum;
            }
            for (int k = 0; k < f; k++) for (int col = 0; col < h; col++) {
                float sum = 0; for (int row = 0; row < b; row++) sum += x[(time*b+row)*f+k]*local[row*h+col]; dw[k*h+col] += sum;
            }
            for (int col = 0; col < h; col++) { float sum = 0; for (int row = 0; row < b; row++) sum += local[row*h+col]; db[col] += sum; }
            int previous = reverse ? time+1 : time-1;
            for (int k = 0; k < h; k++) for (int col = 0; col < h; col++) {
                float sum = 0; for (int row = 0; row < b; row++) sum +=
                        (previous < 0 || previous >= t ? initial[row*h+k] : history[(previous*b+row)*h+k])*local[row*h+col];
                du[k*h+col] += sum;
            }
            dh = new float[b*h];
            for (int row = 0; row < b; row++) for (int k = 0; k < h; k++) {
                float sum = 0; for (int col = 0; col < h; col++) sum += local[row*h+col]*u[k*h+col]; dh[row*h+k] = sum;
            }
        }
        return new float[][] {carry,history,dh,dx,dw,du,db};
    }
}
