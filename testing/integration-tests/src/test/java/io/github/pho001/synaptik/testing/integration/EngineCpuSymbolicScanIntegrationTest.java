package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.engine.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Public actual symbolic AD compared with independently compiled ordinary Tensor unrolling. */
class EngineCpuSymbolicScanIntegrationTest {
    @Test void directLengthsMixedRowHistoriesAndSharedTransposeMatchIndependentActivePrefixTensorAd() {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean reverse : List.of(false,true)) {
            var b = new DynamicDimension("B"); var row = Shape.ofDimensions(b,new StaticDimension(2));
            var stack = Shape.ofDimensions(new StaticDimension(4),b,new StaticDimension(2));
            Tensor h = typed(type,true,row), x = typed(type,true,row), w = typed(type,true,Shape.of(2,2));
            Tensor sharedTranspose = w.permute(1,0), activation = h.add(x.matmul(sharedTranspose)).permute(1,0).tanh();
            Tensor next = activation.sigmoid().permute(1,0);
            var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),List.of(next));
            Tensor initial = typed(type,true,row), sequence = typed(type,true,stack), weight = typed(type,true,Shape.of(2,2));
            Tensor lengths = typed(DataType.INT64,false,Shape.ofDimensions(b)), fs = typed(type,false,row), es = typed(type,false,stack);
            var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(weight),lengths,1,List.of(OptionalInt.of(0)),body,reverse);
            try (var engine = Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
                var compiled = engine.compile(List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst()),List.of(fs,es),List.of(initial,sequence,weight));
                try (var prepared = engine.prepare(compiled)) {
                    for (int batch : new int[]{5,2,0,1,5}) {
                        long[] ls = new long[batch]; for (int r = 0; r < batch; r++) ls[r] = r%5;
                        Tensor iv = typedValue(type,true,Shape.of(batch,2),.01), xv = typedValue(type,true,Shape.of(4,batch,2),.02);
                        Tensor wv = typedValue(type,true,Shape.of(2,2),.03), fsv = typedValue(type,false,Shape.of(batch,2),.1), esv = typedValue(type,false,Shape.of(4,batch,2),.05);
                        double[][] expected = {new double[batch*2],new double[8*batch],new double[batch*2],new double[8*batch],new double[4]};
                        for (int r = 0; r < batch; r++) {
                            // This oracle has no Scan or mask: construct exactly each active
                            // prefix as ordinary Tensor operations, then sum its parameter AD.
                            if (ls[r] == 0) { for (int j = 0; j < 2; j++) { expected[0][r*2+j] = cell(type,iv,r*2+j); expected[2][r*2+j] = cell(type,fsv,r*2+j); } continue; }
                            Tensor state = rowValue(type,true,iv,r*2), ih = state; var steps = new ArrayList<Tensor>();
                            for (int t = 0; t < ls[r]; t++) steps.add(rowValue(type,true,xv,(t*batch+r)*2));
                            var history = new Tensor[(int)ls[r]];
                            for (int visit = 0; visit < ls[r]; visit++) { int t = reverse ? (int)ls[r]-1-visit : visit;
                                state = state.add(steps.get(t).matmul(wv.permute(1,0))).tanh().sigmoid(); history[t] = state; }
                            var roots = new ArrayList<Tensor>(); roots.add(state.neg()); roots.addAll(List.of(history));
                            var seeds = new ArrayList<Tensor>(); seeds.add(stored(type,false,Shape.of(1,2),new double[]{-cell(type,fsv,r*2),-cell(type,fsv,r*2+1)}));
                            for (int t = 0; t < ls[r]; t++) seeds.add(rowValue(type,false,esv,(t*batch+r)*2));
                            var targets = new ArrayList<Tensor>(); targets.add(ih); targets.addAll(steps); targets.add(wv);
                            var reference = engine.compile(roots,seeds,targets); var pool = new ArrayList<>(List.of(ih,wv)); pool.addAll(steps); pool.addAll(seeds);
                            try (var rp = engine.prepare(reference); var rr = engine.run(rp,ordered(reference,pool))) {
                                var values = typedValues(type,rr); for (int j = 0; j < 2; j++) expected[0][r*2+j] = -values.getFirst()[j];
                                for (int t = 0; t < ls[r]; t++) System.arraycopy(values.get(1+t),0,expected[1],(t*batch+r)*2,2);
                                System.arraycopy(values.get(1+(int)ls[r]),0,expected[2],r*2,2);
                                for (int t = 0; t < ls[r]; t++) System.arraycopy(values.get(2+(int)ls[r]+t),0,expected[3],(t*batch+r)*2,2);
                                for (int j = 0; j < 4; j++) expected[4][j] += values.getLast()[j];
                            }
                        }
                        Tensor lv = TensorFactory.fromFlatArray(new TensorDescriptor(DataType.INT64,Shape.of(batch),Optional.of(LayoutDescriptor.contiguous(Shape.of(batch))),false),Optional.empty(),ls);
                        var values = Map.of(initial.id(),iv,sequence.id(),xv,weight.id(),wv,lengths.id(),lv,fs.id(),fsv,es.id(),esv);
                        try (var result = engine.runValues(prepared,compiled.inputs().stream().map(i -> values.get(i.tensorId())).toList())) {
                            var actual = typedValues(type,result); assertEquals(expected.length,actual.size());
                            for (int p = 0; p < expected.length; p++) assertArrayEquals(expected[p],actual.get(p),type == DataType.FLOAT64 ? 2e-13 : 3e-6,"B="+batch+" port="+p);
                        }
                    }
                }
            }
        }
    }
    private static Tensor typed(DataType type,boolean grad,Shape shape) {
        return TensorFactory.create(new TensorDescriptor(type,shape,shape.isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(),grad));
    }
    private static Tensor typedValue(DataType type,boolean grad,Shape shape,double scale) {
        int n = (int)shape.knownElementCount().orElseThrow(); double[] data = new double[n]; for (int i = 0; i < n; i++) data[i] = (i%11-5)*scale;
        return stored(type,grad,shape,data);
    }
    private static Tensor stored(DataType type,boolean grad,Shape shape,double[] data) {
        var descriptor = new TensorDescriptor(type,shape,Optional.of(LayoutDescriptor.contiguous(shape)),grad);
        if (type == DataType.FLOAT64) return TensorFactory.fromFlatArray(descriptor,Optional.empty(),data);
        float[] values = new float[data.length]; for (int i = 0; i < values.length; i++) values[i] = (float)data[i];
        return TensorFactory.fromFlatArray(descriptor,Optional.empty(),values);
    }
    private static double cell(DataType type,Tensor tensor,int i) {
        var segment = tensor.hostStorage().orElseThrow().segment(); return type == DataType.FLOAT64 ? segment.getAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE,i) : segment.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT,i);
    }
    private static Tensor rowValue(DataType type,boolean grad,Tensor input,int offset) {
        return stored(type,grad,Shape.of(1,2),new double[]{cell(type,input,offset),cell(type,input,offset+1)});
    }
    private static List<double[]> typedValues(DataType type,RunResult result) {
        return result.publications().stream().map(p -> { var v = result.materialize(p,1_000_000); var bytes = v.bytes(); double[] data = new double[(int)(v.byteSize()/type.byteWidth())];
            for (int i = 0; i < data.length; i++) data[i] = type == DataType.FLOAT64 ? bytes.getDouble() : bytes.getFloat(); return data; }).toList();
    }

    @Test void nonuniformJointSeedsAndTiedInvariantRolesMatchOrdinaryTensorUnrolling() {
        for (boolean reverse : List.of(false,true)) for (boolean tied : List.of(false,true)) {
            int time = 3, features = tied ? 3 : 4, hidden = 3;
            var b = new DynamicDimension("B");
            Tensor h = symbolic(true,b,new StaticDimension(hidden)), x = symbolic(true,b,new StaticDimension(features));
            Tensor w = fixed(true,features,hidden), u = fixed(true,hidden,hidden), bias = fixed(true,hidden);
            Tensor next = x.matmul(w).add(h.matmul(u.permute(1,0).contiguous())).add(bias).tanh();
            var body = new ScanBody(List.of(h),List.of(x),List.of(w,u,bias),List.of(next),List.of(next));
            Tensor initial = symbolic(true,b,new StaticDimension(hidden));
            Tensor sequence = symbolic(true,new StaticDimension(time),b,new StaticDimension(features));
            Tensor weight = fixed(true,features,hidden), recurrent = tied ? weight : fixed(true,hidden,hidden), offset = fixed(true,hidden);
            var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(weight,recurrent,offset),body,reverse);
            var roots = List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
            Tensor fs = symbolic(false,b,new StaticDimension(hidden)), es = symbolic(false,new StaticDimension(time),b,new StaticDimension(hidden));
            var targets = tied ? List.of(initial,sequence,weight,offset) : List.of(initial,sequence,weight,recurrent,offset);
            try (Engine engine = Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
                var compiled = engine.compile(roots,List.of(fs,es),targets);
                try (var prepared = engine.prepare(compiled)) {
                    for (int batch : new int[] {64,40,1,64,0,3,4}) {
                        Tensor ih = value(true,Shape.of(batch,hidden),.01f), sx = value(true,Shape.of(time,batch,features),.02f),
                                sw = value(true,Shape.of(features,hidden),.03f), su = tied ? sw : value(true,Shape.of(hidden,hidden),.025f),
                                sb = value(true,Shape.of(hidden),.04f), fsv = value(false,Shape.of(batch,hidden),.1f),
                                esv = value(false,Shape.of(time,batch,hidden),.05f);
                        Tensor state = ih; Tensor[] history = new Tensor[time]; var steps = new ArrayList<Tensor>();
                        float[] sequenceCells = sx.hostStorage().orElseThrow().segment().toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT);
                        for (int t = 0; t < time; t++) steps.add(TensorFactory.fromFlatArray(
                                new TensorDescriptor(DataType.FLOAT32,Shape.of(batch,features),Optional.of(LayoutDescriptor.contiguous(Shape.of(batch,features))),true),
                                Optional.empty(),Arrays.copyOfRange(sequenceCells,t*batch*features,(t+1)*batch*features)));
                        for (int step = 0; step < time; step++) {
                            int t = reverse ? time-1-step : step;
                            state = steps.get(t).matmul(sw).add(state.matmul(su.permute(1,0).contiguous())).add(sb).tanh();
                            history[t] = state;
                        }
                        var referenceTargets = new ArrayList<Tensor>(); referenceTargets.add(ih); referenceTargets.addAll(steps); referenceTargets.add(sw);
                        if (!tied) referenceTargets.add(su); referenceTargets.add(sb);
                        // Independent emission roots avoid STACK's static-empty SELECT adjoint.
                        // The tested Scan still publishes its one canonical stacked emission.
                        var referenceRoots = new ArrayList<Tensor>(); referenceRoots.add(state.neg()); referenceRoots.addAll(List.of(history));
                        float[] negativeFinalSeed = fsv.hostStorage().orElseThrow().segment().toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT);
                        for (int i = 0; i < negativeFinalSeed.length; i++) negativeFinalSeed[i] = -negativeFinalSeed[i];
                        var referenceSeeds = new ArrayList<Tensor>(); referenceSeeds.add(TensorFactory.fromFlatArray(
                                new TensorDescriptor(DataType.FLOAT32,Shape.of(batch,hidden),Optional.of(LayoutDescriptor.contiguous(Shape.of(batch,hidden))),false),
                                Optional.empty(),negativeFinalSeed));
                        float[] emissionSeeds = esv.hostStorage().orElseThrow().segment().toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT);
                        for (int t = 0; t < time; t++) referenceSeeds.add(TensorFactory.fromFlatArray(
                                new TensorDescriptor(DataType.FLOAT32,Shape.of(batch,hidden),Optional.of(LayoutDescriptor.contiguous(Shape.of(batch,hidden))),false),
                                Optional.empty(),Arrays.copyOfRange(emissionSeeds,t*batch*hidden,(t+1)*batch*hidden)));
                        var reference = engine.compile(referenceRoots,referenceSeeds,referenceTargets);
                        var referencePool = new ArrayList<>(List.of(ih,sw,su,sb)); referencePool.addAll(referenceSeeds); referencePool.addAll(steps); List<float[]> expected;
                        try (var rp = engine.prepare(reference); var rr = engine.run(rp,ordered(reference,referencePool))) {
                            var raw = values(rr); expected = new ArrayList<>(); float[] finalValue = raw.getFirst().clone();
                            for (int i = 0; i < finalValue.length; i++) finalValue[i] = -finalValue[i]; expected.add(finalValue);
                            float[] emissions = new float[time*batch*hidden];
                            for (int t = 0; t < time; t++) System.arraycopy(raw.get(1+t),0,emissions,t*batch*hidden,batch*hidden);
                            expected.add(emissions); expected.add(raw.get(1+time));
                            float[] dx = new float[time*batch*features];
                            for (int t = 0; t < time; t++) System.arraycopy(raw.get(2+time+t),0,dx,t*batch*features,batch*features);
                            expected.add(dx); expected.addAll(raw.subList(2+2*time,raw.size()));
                        }
                        var supplied = new HashMap<TensorId,Tensor>();
                        supplied.put(initial.id(),ih); supplied.put(sequence.id(),sx); supplied.put(weight.id(),sw); supplied.put(recurrent.id(),su);
                        supplied.put(offset.id(),sb); supplied.put(fs.id(),fsv); supplied.put(es.id(),esv);
                        try (var result = engine.runValues(prepared,compiled.inputs().stream().map(i -> Objects.requireNonNull(supplied.get(i.tensorId()))).toList())) {
                            var actual = values(result); assertEquals(expected.size(),actual.size());
                            for (int p = 0; p < actual.size(); p++) assertArrayEquals(expected.get(p),actual.get(p),3e-5f,"B="+batch+" tied="+tied+" port="+p);
                            assertEquals(Shape.of(features,hidden),result.publications().get(4).descriptor().shape(),"parameter gradient never has a batch axis");
                        }
                    }
                }
            }
        }
    }

    private static List<Tensor> ordered(CompiledGraph graph,List<Tensor> pool) {
        return graph.inputs().stream().map(i -> pool.stream().filter(t -> t.id().equals(i.tensorId())).findFirst().orElseThrow()).toList();
    }
    private static List<float[]> values(RunResult result) {
        return result.publications().stream().map(p -> {
            var v = result.materialize(p,10_000_000); float[] cells = new float[Math.toIntExact(v.byteSize()/4)];
            var bytes = v.bytes(); for (int i = 0; i < cells.length; i++) cells[i] = bytes.getFloat(); return cells;
        }).toList();
    }
    private static Tensor symbolic(boolean grad,Dimension... dimensions) {
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,Shape.ofDimensions(dimensions),Optional.empty(),grad));
    }
    private static Tensor fixed(boolean grad,long... dimensions) {
        var shape = Shape.of(dimensions); return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,shape,Optional.of(LayoutDescriptor.contiguous(shape)),grad));
    }
    private static Tensor value(boolean grad,Shape shape,float scale) {
        float[] cells = new float[Math.toIntExact(shape.knownElementCount().orElseThrow())];
        for (int i = 0; i < cells.length; i++) cells[i] = (i%11-5)*scale;
        return TensorFactory.fromFlatArray(new TensorDescriptor(DataType.FLOAT32,shape,Optional.of(LayoutDescriptor.contiguous(shape)),grad),Optional.empty(),cells);
    }
}
