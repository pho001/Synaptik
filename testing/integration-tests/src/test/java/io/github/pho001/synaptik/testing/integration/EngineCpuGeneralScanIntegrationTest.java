package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.compiler.FunctionalGradientRequest;
import io.github.pho001.synaptik.config.compile.*;
import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.engine.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.*;
import java.lang.foreign.*;
import java.util.*;
import java.util.concurrent.*;
import org.junit.jupiter.api.Test;

/**
 * Public CPU-only Engine forward and actual Compiler AD, with independent unrolled Tensor AD.
 * Unrolled references target only inputs present in their own forward graph; gradients for
 * unused declared scan inputs are checked separately on the actual scan publications.
 */
final class EngineCpuGeneralScanIntegrationTest {
    @Test void symbolicLengthProducedControlFiveRunSequenceUsesSamePublicPreparation() {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean reverse : List.of(false,true)) {
            var b = new io.github.pho001.synaptik.model.shape.DynamicDimension("B");
            var row = Shape.ofDimensions(b,new io.github.pho001.synaptik.model.shape.StaticDimension(2));
            var stacked = Shape.ofDimensions(new io.github.pho001.synaptik.model.shape.StaticDimension(4),b,new io.github.pho001.synaptik.model.shape.StaticDimension(2));
            Tensor h = unresolved(type,true,row), x = unresolved(type,true,row), w = formal(type,true,Shape.of(2));
            Tensor initial = unresolved(type,true,row), sequence = unresolved(type,true,stacked), weight = formal(type,true,Shape.of(2));
            Tensor rawLengths = unresolved(type,false,Shape.ofDimensions(b));
            Tensor next = h.add(x.mul(w));
            var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(weight),rawLengths.cast(DataType.INT64),1,List.of(OptionalInt.of(0)),
                    new ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),List.of(next)),reverse);
            Tensor fs = unresolved(type,false,row), es = unresolved(type,false,stacked);
            try (var engine = Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
                var compiled = engine.compile(List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst()),List.of(fs,es),List.of(initial,sequence,weight));
                try (var prepared = engine.prepare(compiled)) {
                    int run = 0;
                    for (int batch : new int[]{32,32,7,1,32,32,0}) {
                        long[] lengths = new long[batch]; for (int r = 0; r < batch; r++) lengths[r] = run == 1 ? 4-r%5 : run >= 4 ? 4 : r%5;
                        var supplied = Map.of(initial.id(),value(type,true,Shape.of(batch,2),-0d),sequence.id(),value(type,true,Shape.of(4,batch,2),.5),
                                weight.id(),value(type,true,Shape.of(2),.25),rawLengths.id(),lengthValue(type,lengths),
                                fs.id(),value(type,false,Shape.of(batch,2),1),es.id(),value(type,false,Shape.of(4,batch,2),.25));
                        try (var result = engine.runValues(prepared,compiled.inputs().stream().map(i -> supplied.get(i.tensorId())).toList())) {
                            assertLengthLinear(materializeValues(result),lengths,reverse,4);
                            assertEquals(Shape.of(batch,2),result.publications().getFirst().descriptor().shape());
                            assertEquals(Shape.of(4,batch,2),result.publications().get(3).descriptor().shape());
                            assertEquals(Shape.of(2),result.publications().get(4).descriptor().shape());
                        }
                        run++;
                    }
                }
            }
        }
    }
    private static Tensor unresolved(DataType type,boolean gradient,Shape shape) {
        return TensorFactory.create(new TensorDescriptor(type,shape,Optional.empty(),gradient));
    }

    /**
     * One prepared public Engine execution keeps fixed code/resources while produced lengths
     * vary. Both actual Compiler siblings use finite sequence values and cotangent seeds;
     * active-prefix gradients and result/concurrent-run isolation are checked independently.
     * Poisoned-padding/seed rejection is covered by the focused CPU execution tests.
     * @param type independently qualified FLOAT32 or FLOAT64 body type
     * @throws Exception if concurrent execution or retrieval of a completed result fails
     */
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void lengthAwareProducedControlActualReverseReuseAndConcurrentResults(DataType type) throws Exception {
        for (boolean reverse : List.of(false,true)) {
            Tensor h = formal(type,true,Shape.of(3,2)), x = formal(type,true,Shape.of(3,2)), w = formal(type,true,Shape.of(2));
            Tensor next = h.add(x.mul(w));
            var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),List.of(next));
            Tensor initial = value(type,true,Shape.of(3,2),-0d), sequence = value(type,true,Shape.of(5,3,2),.5);
            Tensor weight = value(type,true,Shape.of(2),.25), rawLengths = lengthValue(type,5,3,0);
            Tensor lengths = rawLengths.cast(DataType.INT64);
            var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(weight),lengths,1,List.of(OptionalInt.of(0)),body,reverse);
            var roots = List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
            Tensor finalSeed = value(type,false,Shape.of(3,2),1), outputSeed = value(type,false,Shape.of(5,3,2),.25);
            var pool = List.of(initial,sequence,weight,rawLengths,finalSeed,outputSeed);
            try (Engine engine = Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
                var compiled = engine.compile(roots,List.of(finalSeed,outputSeed),List.of(initial,sequence,weight));
                try (var prepared = engine.prepare(compiled)) {
                    var supplied = ordered(compiled,pool);
                    try (var first = engine.run(prepared,supplied)) {
                        var a = materializeValues(first); assertLengthLinear(a,new long[]{5,3,0},reverse);
                        var full = new ArrayList<>(supplied); int control = supplied.indexOf(rawLengths); full.set(control,lengthValue(type,5,5,5));
                        try (var second = engine.runValues(prepared,full)) {
                            assertLengthLinear(materializeValues(second),new long[]{5,5,5},reverse);
                            assertLengthLinear(materializeValues(first),new long[]{5,3,0},reverse);
                        }
                        var invalid = new ArrayList<>(supplied); invalid.set(control,lengthValue(type,5,3,6));
                        assertThrows(IllegalArgumentException.class,() -> engine.runValues(prepared,invalid));
                        try (var retried = engine.run(prepared,supplied)) { assertLengthLinear(materializeValues(retried),new long[]{5,3,0},reverse); }
                        try (var workers = Executors.newFixedThreadPool(2)) {
                            var one = workers.submit(() -> { try (var r = engine.runValues(prepared,supplied)) { return materializeValues(r); } });
                            var two = workers.submit(() -> { try (var r = engine.runValues(prepared,full)) { return materializeValues(r); } });
                            assertLengthLinear(one.get(),new long[]{5,3,0},reverse); assertLengthLinear(two.get(),new long[]{5,5,5},reverse);
                        }
                    }
                }
            }
        }
    }

    /**
     * Computes an independent linear active-prefix forward and reverse oracle.
     * @param actual forward final/emission followed by initial/sequence/weight gradients
     * @param lengths admitted per-row prefix lengths
     * @param reverse original forward traversal direction
     */
    private static void assertLengthLinear(List<double[]> actual,long[] lengths,boolean reverse) {
        assertLengthLinear(actual,lengths,reverse,5);
    }
    private static void assertLengthLinear(List<double[]> actual,long[] lengths,boolean reverse,int count) {
        int batch = lengths.length, frame = batch*2;
        double[] state = new double[frame], history = new double[count*frame], dh = new double[frame], dx = new double[count*frame], dw = new double[2];
        Arrays.fill(state,-0d); Arrays.fill(dh,1);
        for (int step = 0; step < count; step++) {
            int time = reverse ? count-1-step : step;
            for (int b = 0; b < batch; b++) if (time < lengths[b]) for (int j = 0; j < 2; j++) history[time*frame+b*2+j] = (state[b*2+j] += .125);
        }
        for (int step = 0; step < count; step++) {
            int time = reverse ? step : count-1-step;
            for (int b = 0; b < batch; b++) if (time < lengths[b]) for (int j = 0; j < 2; j++) {
                dh[b*2+j] += .25; dx[time*frame+b*2+j] = dh[b*2+j]*.25; dw[j] += dh[b*2+j]*.5;
            }
        }
        assertEquals(5,actual.size()); assertArrayEquals(state,actual.get(0)); assertArrayEquals(history,actual.get(1));
        assertArrayEquals(dh,actual.get(2)); assertArrayEquals(dx,actual.get(3)); assertArrayEquals(dw,actual.get(4));
    }

    /**
     * Creates an independently owned floating source for an actual INT64-producing CAST.
     * @param type FLOAT32 or FLOAT64 producer source type
     * @param lengths exact stored control values, including deliberate invalid producer inputs
     * @return canonical non-gradient floating vector backed by a fresh heap array
     */
    private static Tensor lengthValue(DataType type,long... lengths) {
        Shape shape = Shape.of(lengths.length);
        MemorySegment storage;
        if (type == DataType.FLOAT64) { double[] data = new double[lengths.length]; for (int i = 0; i < data.length; i++) data[i] = lengths[i]; storage = MemorySegment.ofArray(data); }
        else { float[] data = new float[lengths.length]; for (int i = 0; i < data.length; i++) data[i] = lengths[i]; storage = MemorySegment.ofArray(data); }
        return TensorFactory.create(new TensorDescriptor(type,shape,Optional.of(LayoutDescriptor.contiguous(shape)),false),
                Optional.empty(),Optional.of(new MemorySegmentStorage(type,lengths.length,storage)));
    }
    /**
     * Real source regressions: Conv1d's squeeze is the forward and reverse primal read view;
     * singleton CONCAT's actual derivative supplies a dense view emission seed. Both execute
     * Compiler reverse kernels and compare every connected numerical gradient with ordinary
     * unrolling using the same input/parameter leaves, not newly initialized lookalikes.
     */
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void denseConv1dReadAndSingletonConcatReverseSeedMatchOrdinaryRnnUnrolling(DataType type) {
        for (boolean convolution : List.of(true,false)) for (boolean reverse : List.of(false,true)) {
            Tensor input = variedValue(type,true,Shape.of(3,2,5),.125);
            Tensor kernel = variedValue(type,true,Shape.of(2,2,2),.0625);
            Tensor prefixBias = variedValue(type,true,Shape.of(2),.03125);
            Tensor sequence = convolution ? input.conv1d(kernel,prefixBias,
                    io.github.pho001.synaptik.model.operation.convolution.Conv1dAttrs.defaults())
                    : variedValue(type,true,Shape.of(3,2,4),.125);
            Tensor weight = variedValue(type,true,Shape.of(3,4),.0625);
            Tensor recurrent = variedValue(type,true,Shape.of(3,3),.03125);
            Tensor bias = variedValue(type,true,Shape.of(3),.015625);
            Tensor initial = value(type,false,Shape.of(2,3),0);
            Tensor h = formal(type,false,Shape.of(2,3)), x = formal(type,true,Shape.of(2,4));
            Tensor w = formal(type,true,Shape.of(3,4)), u = formal(type,true,Shape.of(3,3)), b = formal(type,true,Shape.of(3));
            Tensor next = x.linear(w,b).add(h.linear(u)).tanh();
            var body = new ScanBody(List.of(h),List.of(x),List.of(w,u,b),List.of(next),List.of(next));
            var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(weight,recurrent,bias),body,reverse);
            Tensor emitted = scan.outputs().getFirst();
            var roots = convolution ? List.of(scan.finalCarry().getFirst(),emitted) : List.of(Tensor.concat(0,emitted));
            if (!convolution) assertTrue(roots.getFirst().descriptor().layout().isEmpty(),"real singleton CONCAT, not a fabricated seed descriptor");
            var targets = convolution ? List.of(weight,recurrent,bias,kernel,prefixBias,input) : List.of(weight,recurrent,bias,sequence);
            var seeds = roots.stream().map(r -> variedValue(type,false,r.descriptor().shape(),.25)).toList();
            var pool = new ArrayList<>(List.of(initial,weight,recurrent,bias)); pool.addAll(seeds);
            if (convolution) pool.addAll(List.of(input,kernel,prefixBias)); else pool.add(sequence);
            Tensor state = initial; Tensor[] history = new Tensor[3];
            for (int step = 0; step < 3; step++) {
                int t = reverse ? 2-step : step;
                state = sequence.select(0,t).linear(weight,bias).add(state.linear(recurrent)).tanh(); history[t] = state;
            }
            var referenceRoots = convolution ? List.of(state,Tensor.stack(0,history)) : List.of(Tensor.stack(0,history));
            try (Engine engine = Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
                var reference = engine.compile(referenceRoots,seeds,targets); List<double[]> expected;
                try (var p = engine.prepare(reference); var r = engine.run(p,ordered(reference,pool))) { expected = materializeValues(r); }
                var compiled = engine.compile(roots,seeds,targets);
                try (var p = engine.prepare(compiled); var r = engine.run(p,ordered(compiled,pool))) {
                    var actual = materializeValues(r); assertEquals(roots.size()+targets.size(),actual.size());
                    assertEquals(expected.size(),actual.size());
                    for (int port = 0; port < actual.size(); port++) assertArrayEquals(expected.get(port),actual.get(port),
                            type == DataType.FLOAT64 ? 3e-13 : 5e-6,"actual RNN AD conv="+convolution+" reverse="+reverse+" port="+port);
                    for (int port = roots.size(); port < actual.size(); port++)
                        assertTrue(Arrays.stream(actual.get(port)).anyMatch(v -> Math.abs(v) > 1e-8),"connected target gradient must be exercised, port="+port);
                }
                // Forward-only demand has no saved histories; verify that distinct lowering too.
                var forward = engine.compile(roots);
                try (var p = engine.prepare(forward); var r = engine.run(p,ordered(forward,pool))) {
                    var actual = materializeValues(r);
                    for (int port = 0; port < roots.size(); port++) assertArrayEquals(expected.get(port),actual.get(port),
                            type == DataType.FLOAT64 ? 3e-13 : 5e-6,"forward-only RNN port="+port);
                }
            }
        }
    }

    /**
     * Creates deterministic asymmetric values so transposes, time order and gradients are visible.
     * @param type represented floating dtype
     * @param grad gradient eligibility
     * @param shape static logical domain
     * @param base finite center value
     * @return new owned test leaf with canonical storage and independent identity
     */
    private static Tensor variedValue(DataType type, boolean grad, Shape shape, double base) {
        int count = Math.toIntExact(shape.knownElementCount().orElseThrow());
        double[] data = new double[count]; for (int i = 0; i < count; i++) data[i] = base+(i%11-5)*.0078125;
        MemorySegment segment;
        if (type == DataType.FLOAT64) segment = MemorySegment.ofArray(data);
        else { float[] floats = new float[count]; for (int i = 0; i < count; i++) floats[i] = (float)data[i]; segment = MemorySegment.ofArray(floats); }
        return TensorFactory.create(new TensorDescriptor(type,shape,Optional.of(LayoutDescriptor.contiguous(shape)),grad),
                Optional.empty(),Optional.of(new MemorySegmentStorage(type,count,segment)));
    }

    /**
     * Empty row domains still run actual scan AD: the reverse K-by-zero contraction produces
     * a nonempty exact-positive-zero weight gradient. NaN weights must never be read merely
     * because their invariant descriptor is nonempty; public row reductions also publish zero.
     */
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void emptyRowMatmulActualReversePublishesNonemptyTypedZeroParameterGradient(DataType type) {
        int k = 2, n = 3;
        for (boolean reverse : List.of(false,true)) for (int time : List.of(1,3)) {
            Tensor h = formal(type,true,Shape.of(0,n)), x = formal(type,true,Shape.of(0,k)), w = formal(type,true,Shape.of(k,n));
            Tensor product = x.matmul(w);
            var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(h.add(product)),List.of(product.sum(0,false)));
            Tensor initial = value(type,true,Shape.of(0,n),.125f), sequence = value(type,true,Shape.of(time,0,k),.25f);
            Tensor weight = value(type,true,Shape.of(k,n),Float.NaN);
            var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(weight),body,reverse);
            var roots = List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
            var seeds = roots.stream().map(r -> value(type,false,r.descriptor().shape(),.5f)).toList();
            var pool = new ArrayList<>(List.of(initial,sequence,weight)); pool.addAll(seeds);
            try (Engine engine = Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
                var compiled = engine.compile(roots,seeds,List.of(initial,sequence,weight));
                try (var prepared = engine.prepare(compiled); var result = engine.run(prepared,ordered(compiled,pool))) {
                    var actual = materializeValues(result); assertEquals(5,actual.size());
                    assertArrayEquals(new double[0],actual.get(0));
                    assertArrayEquals(new double[0],actual.get(2)); assertArrayEquals(new double[0],actual.get(3));
                    assertEquals(time*n,actual.get(1).length);
                    for (double value : actual.get(1)) assertEquals(0,Double.doubleToRawLongBits(value),"empty-row SUM exact typed positive zero");
                    var gradient = result.publications().get(4).descriptor();
                    assertEquals(type,gradient.dataType()); assertEquals(Shape.of(k,n),gradient.shape());
                    assertEquals(k*n,actual.get(4).length);
                    for (double value : actual.get(4)) assertEquals(0,Double.doubleToRawLongBits(value),
                            "actual empty-contraction parameter sum: T="+time+" reverse="+reverse);
                }
            }
        }
    }

    /**
     * A zero-sized first carry cannot skip the whole step or its reverse. The second carry
     * advances by a nonempty parameter reduction, and every original time contributes its
     * finite final seed to each weight gradient despite the independent empty MATMUL branch.
     */
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void emptyMatrixCarryDoesNotSkipNonemptyCarryOrActualParameterAccumulation(DataType type) {
        int k = 2, n = 3; double seedValue = .5f;
        for (boolean reverse : List.of(false,true)) for (int time : List.of(1,3)) {
            Tensor h = formal(type,true,Shape.of(0,n)), s = formal(type,true,Shape.of(n));
            Tensor x = formal(type,true,Shape.of(0,k)), w = formal(type,true,Shape.of(k,n));
            var body = new ScanBody(List.of(h,s),List.of(x),List.of(w),
                    List.of(h.add(x.matmul(w)),s.add(w.sum(0,false))),List.of());
            Tensor initial = value(type,true,Shape.of(0,n),.125f), initialS = value(type,true,Shape.of(n),.125f);
            Tensor sequence = value(type,true,Shape.of(time,0,k),.25f), weight = value(type,true,Shape.of(k,n),.25f);
            var scan = Scan.apply(List.of(initial,initialS),List.of(sequence),List.of(weight),body,reverse);
            var seeds = scan.finalCarry().stream().map(r -> value(type,false,r.descriptor().shape(),seedValue)).toList();
            var pool = new ArrayList<>(List.of(initial,initialS,sequence,weight)); pool.addAll(seeds);
            try (Engine engine = Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
                var compiled = engine.compile(scan.finalCarry(),seeds,List.of(initial,initialS,sequence,weight));
                try (var prepared = engine.prepare(compiled); var result = engine.run(prepared,ordered(compiled,pool))) {
                    var actual = materializeValues(result); assertEquals(6,actual.size());
                    assertArrayEquals(new double[0],actual.get(0)); assertArrayEquals(new double[0],actual.get(2));
                    assertArrayEquals(new double[0],actual.get(4));
                    double[] expectedS = new double[n], expectedInitialGradient = new double[n], expectedWeightGradient = new double[k*n];
                    Arrays.fill(expectedS,.125f+time*k*.25f); Arrays.fill(expectedInitialGradient,seedValue);
                    Arrays.fill(expectedWeightGradient,time*seedValue);
                    assertArrayEquals(expectedS,actual.get(1),"nonempty carry advances with empty first carry");
                    assertArrayEquals(expectedInitialGradient,actual.get(3));
                    var gradient = result.publications().get(5).descriptor();
                    assertEquals(type,gradient.dataType()); assertEquals(Shape.of(k,n),gradient.shape());
                    assertArrayEquals(expectedWeightGradient,actual.get(5),"one parameter contribution per original time: T="+time+" reverse="+reverse);
                }
            }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void nonSquareMatmulAndActualReverseMatchUnrolledAndEmptyTimeThroughEngine(DataType type) {
        for (boolean reverse : List.of(false,true)) for (int time : List.of(0,1,3)) {
            Tensor h = formal(type,true,Shape.of(2,3)), x = formal(type,true,Shape.of(2,2));
            Tensor w = formal(type,true,Shape.of(2,3)), b = formal(type,true,Shape.of(3));
            Tensor next = h.add(x.matmul(w)).add(b).tanh();
            var body = new ScanBody(List.of(h),List.of(x),List.of(w,b),List.of(next),List.of(next.sum(0,false)));
            Tensor initial = value(type,true,Shape.of(2,3),.2f), sequence = value(type,true,Shape.of(time,2,2),.125f);
            Tensor weight = value(type,true,Shape.of(2,3),.25f), bias = value(type,true,Shape.of(3),.03125f);
            var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(weight,bias),body,reverse);
            var roots = List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
            var seeds = roots.stream().map(r -> value(type,false,r.descriptor().shape(),.5f)).toList();
            var pool = new ArrayList<>(List.of(initial,sequence,weight,bias)); pool.addAll(seeds);
            try (Engine engine = Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
                var compiled = compileScanGradients(engine,roots,seeds,List.of(initial,sequence,weight,bias),time == 0); List<double[]> actual;
                try (var p = engine.prepare(compiled); var r = engine.run(p,ordered(compiled,pool))) { actual = materializeValues(r); }
                if (time == 0) {
                    assertArrayEquals(new double[] {.2f,.2f,.2f,.2f,.2f,.2f},actual.get(0));
                    assertArrayEquals(new double[] {.5f,.5f,.5f,.5f,.5f,.5f},actual.get(2));
                    assertArrayEquals(new double[6],actual.get(4)); assertArrayEquals(new double[3],actual.get(5));
                    continue;
                }
                Tensor state = initial; Tensor[] emissions = new Tensor[time];
                for (int step = 0; step < time; step++) { int t = reverse ? time-1-step : step;
                    state = state.add(sequence.select(0,t).matmul(weight)).add(bias).tanh(); emissions[t] = state.sum(0,false); }
                var reference = engine.compile(List.of(state,Tensor.stack(0,emissions)),seeds,List.of(initial,sequence,weight,bias));
                try (var p = engine.prepare(reference); var r = engine.run(p,ordered(reference,pool))) {
                    var expected = materializeValues(r); assertEquals(expected.size(),actual.size());
                    for (int i = 0; i < expected.size(); i++) assertArrayEquals(expected.get(i),actual.get(i), type == DataType.FLOAT64 ? 2e-13 : 3e-6f,"public Engine MATMUL AD port="+i);
                }
            }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void broadcastParameterAndReductionReverseMatchUnrolledThroughPublicEngine(DataType type) {
        for (boolean reverse : List.of(false,true)) for (int t : List.of(0,1,3)) for (int batch : List.of(0,2)) {
            Tensor h = formal(type,false,Shape.of(batch,3)), x = formal(type,true,Shape.of(batch,3)), w = formal(type,true,Shape.of(3)), b = formal(type,true,Shape.of());
            Tensor next = h.mul(w).add(x).add(b).tanh();
            var body = new ScanBody(List.of(h),List.of(x),List.of(w,b),List.of(next),List.of(next.sum(0,false)));
            Tensor initial = value(type,false,Shape.of(batch,3),.2f), sequence = value(type,true,Shape.of(t,batch,3),.125f);
            Tensor weight = value(type,true,Shape.of(3),batch == 0 ? Float.NaN : .75f), bias = value(type,true,Shape.of(),.03125f);
            var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(weight,bias),body,reverse);
            var roots = List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
            var seeds = roots.stream().map(r -> value(type,false,r.descriptor().shape(),.5f)).toList();
            var pool = new ArrayList<>(List.of(initial,sequence,weight,bias)); pool.addAll(seeds);
            try (Engine engine = Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
                var compiled = compileScanGradients(engine,roots,seeds,List.of(sequence,weight,bias),t == 0); List<double[]> actual;
                try (var p = engine.prepare(compiled); var r = engine.run(p,ordered(compiled,pool))) { actual = materializeValues(r); }
                if (t == 0 || batch == 0) {
                    assertArrayEquals(new double[3],actual.get(3)); assertArrayEquals(new double[1],actual.get(4));
                    continue;
                }
                Tensor state = initial; Tensor[] emissions = new Tensor[t];
                for (int step = 0; step < t; step++) { int time = reverse ? t-1-step : step;
                    state = state.mul(weight).add(sequence.select(0,time)).add(bias).tanh(); emissions[time] = state.sum(0,false); }
                var unrolled = engine.compile(List.of(state,Tensor.stack(0,emissions)),seeds,List.of(sequence,weight,bias));
                try (var p = engine.prepare(unrolled); var r = engine.run(p,ordered(unrolled,pool))) {
                    var expected = materializeValues(r); assertEquals(expected.size(),actual.size());
                    for (int i = 0; i < expected.size(); i++) assertArrayEquals(expected.get(i),actual.get(i), type == DataType.FLOAT64 ? 2e-13 : 3e-6f,"public Engine broadcast AD port="+i);
                }
            }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void shapedPreparationKeepsOpenResultFreshRunsRetryAndConcurrentUseIsolated(DataType type) throws Exception {
        Tensor h = formal(type,false,Shape.of(2,3)), x = formal(type,false,Shape.of(2,3)), w = formal(type,false,Shape.of(3));
        var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(h.add(x.mul(w))),List.of());
        Tensor initial = value(type,false,Shape.of(2,3),.2f), sequence = value(type,false,Shape.of(3,2,3),.1f), weight = value(type,false,Shape.of(3),.5f);
        var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(weight),body,false);
        try (Engine engine = Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
            var compiled = engine.compile(scan.finalCarry());
            try (var p = engine.prepare(compiled); var first = engine.run(p,ordered(compiled,List.of(initial,sequence,weight)));
                    var executor = Executors.newFixedThreadPool(2)) {
                var preserved = materializeValues(first).getFirst();
                var a = compiled.inputs().stream().map(i -> value(type,false,i.descriptor().shape(),.3f)).toList();
                var b = compiled.inputs().stream().map(i -> value(type,false,i.descriptor().shape(),-.4f)).toList();
                var one = executor.submit(() -> { try (var r = engine.runValues(p,a)) { return materializeValues(r).getFirst(); } });
                var two = executor.submit(() -> { try (var r = engine.runValues(p,b)) { return materializeValues(r).getFirst(); } });
                assertFalse(Arrays.equals(one.get(10,TimeUnit.SECONDS),two.get(10,TimeUnit.SECONDS)));
                assertArrayEquals(preserved,materializeValues(first).getFirst());
                assertThrows(IllegalArgumentException.class,() -> engine.runValues(p,List.of(initial)));
                try (var retry = engine.runValues(p,a)) { assertArrayEquals(one.get(),materializeValues(retry).getFirst()); }
            }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void reusedTemplateTiedActualRolesDisconnectedInvariantAndCrossOccurrenceSumsMatchUnrolled(DataType type) {
        Tensor h = formal(type,true,Shape.of(2)), x = formal(type,true,Shape.of(2)), w = formal(type,true,Shape.of(2)), unused = formal(type,true,Shape.of(2));
        var body = new ScanBody(List.of(h),List.of(x),List.of(w,unused),List.of(h.mul(w).add(x)),List.of());
        Tensor initial = value(type,true,Shape.of(2),.25f), sequence = value(type,true,Shape.of(3,2),.125f), disconnected = value(type,true,Shape.of(2),.7f);
        var a = Scan.apply(List.of(initial),List.of(sequence),List.of(initial,disconnected),body,false);
        var b = Scan.apply(List.of(initial),List.of(sequence),List.of(initial,disconnected),body,true);
        var root = a.finalCarry().getFirst().add(b.finalCarry().getFirst()); Tensor seed = value(type,false,Shape.of(2),1f);
        var targets = List.of(initial,sequence,disconnected); var pool = List.of(initial,sequence,disconnected,seed);
        try (Engine engine = Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
            var compiled = compileScanGradients(engine,List.of(root),List.of(seed),targets,true); List<double[]> actual;
            try (var p = engine.prepare(compiled); var r = engine.run(p,ordered(compiled,pool))) { actual = materializeValues(r); }
            Tensor forward = initial, reverse = initial;
            for (int t = 0; t < 3; t++) { forward = forward.mul(initial).add(sequence.select(0,t));
                reverse = reverse.mul(initial).add(sequence.select(0,2-t)); }
            var reference = engine.compile(List.of(forward.add(reverse)),List.of(seed),List.of(initial,sequence));
            try (var p = engine.prepare(reference); var r = engine.run(p,ordered(reference,pool))) {
                var expected = materializeValues(r);
                assertEquals(4,actual.size()); assertEquals(3,expected.size());
                // Both publications order the forward result, initial gradient, then sequence gradient.
                for (int i = 0; i < expected.size(); i++) assertArrayEquals(expected.get(i),actual.get(i), type == DataType.FLOAT64 ? 2e-13 : 1e-6f);
            }
            assertArrayEquals(new double[2],actual.get(3), "actual scan disconnected-invariant gradient");
        }
    }
    @Test void actualBackwardMatchesUnrolledTensorAutogradAndRepresentedJavaSites() {
        for (boolean reverse : List.of(false, true)) for (int t : List.of(0, 1, 3, 257))
            for (int n : List.of(0, 2)) for (int seedClass = 0; seedClass < 3; seedClass++) {
                try (Engine engine = Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
                    Tensor h = formal(false, Shape.of(n)), x = formal(true, Shape.of(n)), w = formal(true, Shape.of(n));
                    Tensor next = h.mul(w).add(x).tanh();
                    var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(next), List.of(next));
                    Tensor initial = value(false, Shape.of(n), .2f), sequence = value(true, Shape.of(t, n), .125f), weight = value(true, Shape.of(n), .75f);
                    var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(weight), body, reverse);
                    List<Tensor> roots = seedClass == 0 ? scan.finalCarry() : seedClass == 1 ? scan.outputs()
                            : List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst());
                    var seeds = roots.stream().map(r -> value(false, r.descriptor().shape(), .5f)).toList();
                    var pool = new ArrayList<>(List.of(initial, sequence, weight)); pool.addAll(seeds);
                    var compiled = compileScanGradients(engine,roots,seeds,List.of(sequence,weight),t == 0);
                    List<float[]> actual;
                    try (var prepared = engine.prepare(compiled); var result = engine.run(prepared, ordered(compiled, pool))) {
                        actual = materialize(result);
                        assertEquals(roots.size() + 2, actual.size());
                    }
                    float[] state = new float[n]; Arrays.fill(state, .2f);
                    float[] old = new float[t * n], history = new float[t * n];
                    for (int k = 0; k < t; k++) { int time = reverse ? t - 1 - k : k;
                        for (int i = 0; i < n; i++) { int p = time * n + i; old[p] = state[i];
                            float product = state[i] * .75f; float sum = product + .125f;
                            history[p] = state[i] = (float) Math.tanh(sum); }
                    }
                    float[] dc = new float[n], dw = new float[n], dx = new float[t * n];
                    if (seedClass != 1) Arrays.fill(dc, .5f);
                    for (int k = 0; k < t; k++) { int time = reverse ? k : t - 1 - k;
                        for (int i = 0; i < n; i++) { int p = time * n + i;
                            float incoming = dc[i] + (seedClass == 0 ? 0f : .5f);
                            float square = history[p] * history[p]; float factor = 1f - square;
                            float d = incoming * factor; dx[p] = d; dc[i] = d * .75f;
                            float contribution = d * old[p]; dw[i] = dw[i] + contribution; }
                    }
                    assertArrayEquals(seedClass == 1 ? history : state, actual.getFirst());
                    if (seedClass == 2) assertArrayEquals(history, actual.get(1));
                    assertArrayEquals(dx, actual.get(roots.size())); assertArrayEquals(dw, actual.get(roots.size() + 1));
                    if (t > 0 && t <= 3 && n > 0) {
                        Tensor unrolledState = initial; var emissions = new Tensor[t];
                        for (int k = 0; k < t; k++) { int time = reverse ? t - 1 - k : k;
                            unrolledState = unrolledState.mul(weight).add(sequence.select(0, time)).tanh();
                            emissions[time] = unrolledState; }
                        Tensor stacked = Tensor.stack(0, emissions);
                        List<Tensor> unrolledRoots = seedClass == 0 ? List.of(unrolledState) : seedClass == 1
                                ? List.of(stacked) : List.of(unrolledState, stacked);
                        var unrolled = engine.compile(unrolledRoots, seeds, List.of(sequence, weight));
                        try (var prepared = engine.prepare(unrolled); var result = engine.run(prepared, ordered(unrolled, pool))) {
                            var expected = materialize(result);
                            for (int p = 0; p < expected.size(); p++) assertArrayEquals(expected.get(p), actual.get(p),
                                    2e-6f, "unrolled Tensor AD, port=" + p);
                        }
                    }
                }
            }
    }

    @Test void rankSixteenStepStackedRankSeventeenRunsForwardAndBackwardOnCpuOnly() {
        long[] dimensions = new long[16]; Arrays.fill(dimensions, 1); dimensions[15] = 2;
        Shape s = Shape.of(dimensions); long[] stacked = new long[17]; Arrays.fill(stacked, 1); stacked[0] = 3; stacked[16] = 2;
        Tensor h = formal(true, s), x = formal(true, s);
        Tensor next = h.add(x).tanh(); var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(next), List.of(next));
        Tensor initial = value(true, s, .25f), sequence = value(true, Shape.of(stacked), .125f), seed = value(false, s, 1f);
        for (boolean reverse : List.of(false, true)) try (Engine engine = Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
            var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(), body, reverse);
            var compiled = engine.compile(scan.finalCarry(), List.of(seed), List.of(initial, sequence));
            try (var prepared = engine.prepare(compiled); var result = engine.run(prepared, ordered(compiled, List.of(initial, sequence, seed)))) {
                assertEquals(3, result.resultCount());
                assertEquals(17, result.publications().get(2).descriptor().shape().rank());
                var actual = materialize(result);
                float[] history = new float[3]; float state = .25f;
                for (int step = 0; step < 3; step++) {
                    state = (float)Math.tanh(state + .125f); history[step] = state;
                }
                float carryGradient = 1f; float[] sequenceGradient = new float[6];
                for (int step = 2; step >= 0; step--) {
                    carryGradient *= 1f - history[step] * history[step];
                    int time = reverse ? 2 - step : step;
                    sequenceGradient[time * 2] = sequenceGradient[time * 2 + 1] = carryGradient;
                }
                assertArrayEquals(new float[]{state,state},actual.get(0),2e-6f);
                assertArrayEquals(new float[]{carryGradient,carryGradient},actual.get(1),2e-6f);
                assertArrayEquals(sequenceGradient,actual.get(2),2e-6f);
            }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void temporalOldCarryAndItsAutomaticReverseMatchUnrolledInBothDirections(DataType type) {
        for (boolean reverse : List.of(false, true)) try (Engine engine = Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
            Tensor h = formal(type,true, Shape.of(1)), old = formal(type,true, Shape.of(1)), x = formal(type,true, Shape.of(1));
            var body = new ScanBody(List.of(h, old), List.of(x), List.of(), List.of(h.add(x), h), List.of());
            Tensor initial = value(type,true, Shape.of(1), 1), second = value(type,true, Shape.of(1), 100), sequence = value(type,true, Shape.of(3,1), 2);
            Tensor seed0 = value(type,false, Shape.of(1), .5f), seed1 = value(type,false, Shape.of(1), .25f);
            var pool = List.of(initial, second, sequence, seed0, seed1);
            var scan = Scan.apply(List.of(initial, second), List.of(sequence), List.of(), body, reverse);
            var targets = List.of(initial, second, sequence); var seeds = List.of(seed0, seed1);
            var compiled = compileScanGradients(engine,scan.finalCarry(),seeds,targets,true);
            List<double[]> actual;
            try (var p = engine.prepare(compiled); var r = engine.run(p, ordered(compiled, pool))) { actual = materializeValues(r); }
            Tensor state = initial, previous = second;
            for (int k = 0; k < 3; k++) { Tensor prior = state;
                state = state.add(sequence.select(0, reverse ? 2 - k : k)); previous = prior; }
            var reference = engine.compile(List.of(state, previous), seeds, List.of(initial, sequence));
            try (var p = engine.prepare(reference); var r = engine.run(p, ordered(reference, pool))) {
                var expected = materializeValues(r);
                assertEquals(5, actual.size()); assertEquals(4, expected.size());
                // Skip the actual scan's second-initial gradient when joining reference publications.
                int[] actualIndices = {0, 1, 2, 4};
                for (int i = 0; i < expected.size(); i++) assertArrayEquals(expected.get(i), actual.get(actualIndices[i]));
            }
            assertArrayEquals(new double[1], actual.get(3), "actual scan unused second-initial gradient");
            assertArrayEquals(new double[] {7}, actual.get(0)); assertArrayEquals(new double[] {5}, actual.get(1));
        }
    }

    @Test void reusablePreparationKeepsOpenResultAndConcurrentFreshRunsIsolated() throws Exception {
        Tensor h = formal(false, Shape.of(2)), x = formal(false, Shape.of(2));
        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x).tanh()), List.of());
        Tensor initial = value(false, Shape.of(2), .2f), sequence = value(false, Shape.of(3,2), .1f);
        var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(), body, false);
        try (Engine engine = Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
            var compiled = engine.compile(scan.finalCarry());
            try (var p = engine.prepare(compiled); var first = engine.run(p, ordered(compiled, List.of(initial, sequence)));
                    var executor = Executors.newFixedThreadPool(2)) {
                var preserved = materialize(first).getFirst();
                var a = ordered(compiled, List.of(initial, sequence)).stream().map(v -> value(false, v.descriptor().shape(), .3f)).toList();
                var b = ordered(compiled, List.of(initial, sequence)).stream().map(v -> value(false, v.descriptor().shape(), -.4f)).toList();
                var one = executor.submit(() -> { try (var r = engine.runValues(p, a)) { return materialize(r).getFirst(); } });
                var two = executor.submit(() -> { try (var r = engine.runValues(p, b)) { return materialize(r).getFirst(); } });
                assertFalse(Arrays.equals(one.get(10, TimeUnit.SECONDS), two.get(10, TimeUnit.SECONDS)));
                assertArrayEquals(preserved, materialize(first).getFirst());
                assertThrows(IllegalArgumentException.class, () -> engine.runValues(p, List.of(initial)));
                try (var retry = engine.runValues(p, a)) { assertArrayEquals(one.get(), materialize(retry).getFirst()); }
            }
        }
    }

    /** Uses strict ordinary admission unless the finite source has a requested disconnected role. */
    static CompiledGraph compileScanGradients(Engine engine,List<Tensor> roots,List<Tensor> seeds,
            List<Tensor> targets,boolean hasDisconnectedTarget) {
        if (!hasDisconnectedTarget) return engine.compile(roots,seeds,targets);
        var stage = new FunctionalGradientRequest.Stage(roots.stream().map(root ->
                (FunctionalGradientRequest.OutputReference)new FunctionalGradientRequest.ForwardTensorReference(root)).toList(),
                seeds.stream().map(Optional::of).toList(),targets,false,FunctionalGradientRequest.DisconnectedPolicy.ZERO);
        var advanced = EngineMixedOwnerTestAccess.advancedEngine(engine).compile(
                CompileMode.FORWARD_AND_BACKWARD,roots,
                Optional.of(new FunctionalGradientRequest(List.of(stage))),GraphOptimizationConfig.disabled(),
                BackendIntent.unconstrained(),PartitionScoringConfig.neutral());
        return EngineMixedOwnerTestAccess.ordinaryGraph(engine,advanced);
    }

    private static Tensor formal(DataType type, boolean grad, Shape s) {
        return TensorFactory.create(new TensorDescriptor(type,s,Optional.of(LayoutDescriptor.contiguous(s)),grad));
    }

    private static Tensor value(DataType type, boolean grad, Shape s, double fill) {
        int count = Math.toIntExact(s.knownElementCount().orElseThrow());
        MemorySegment storage;
        if (type == DataType.FLOAT64) { double[] data = new double[count]; Arrays.fill(data,fill); storage = MemorySegment.ofArray(data); }
        else { float[] data = new float[count]; Arrays.fill(data,(float)fill); storage = MemorySegment.ofArray(data); }
        return TensorFactory.create(new TensorDescriptor(type,s,Optional.of(LayoutDescriptor.contiguous(s)),grad),
                Optional.empty(),Optional.of(new MemorySegmentStorage(type,count,storage)));
    }

    private static List<double[]> materializeValues(RunResult result) {
        return result.publications().stream().map(p -> {
            var bytes = result.materialize(p,1000000).bytes();
            var type = p.descriptor().dataType(); double[] values = new double[bytes.remaining()/type.byteWidth()];
            for (int i = 0; i < values.length; i++) values[i] = type == DataType.FLOAT64 ? bytes.getDouble() : bytes.getFloat();
            return values;
        }).toList();
    }

    private static Tensor formal(boolean grad, Shape s) {
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, s, Optional.of(LayoutDescriptor.contiguous(s)), grad));
    }

    private static Tensor value(boolean grad, Shape s, float fill) {
        float[] values = new float[Math.toIntExact(s.knownElementCount().orElseThrow())]; Arrays.fill(values, fill);
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, s, Optional.of(LayoutDescriptor.contiguous(s)), grad),
                Optional.empty(), Optional.of(new MemorySegmentStorage(DataType.FLOAT32, values.length, MemorySegment.ofArray(values))));
    }

    private static List<Tensor> ordered(CompiledGraph compiled, List<Tensor> pool) {
        return compiled.inputs().stream().map(i -> pool.stream().filter(t -> t.id().equals(i.tensorId())).findFirst().orElseThrow()).toList();
    }

    private static List<float[]> materialize(RunResult result) {
        return result.publications().stream().map(p -> {
            var bytes = result.materialize(p, 1000000).bytes(); float[] values = new float[bytes.remaining() / 4];
            for (int i = 0; i < values.length; i++) values[i] = bytes.getFloat(); return values;
        }).toList();
    }
}
