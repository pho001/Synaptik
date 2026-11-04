package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.CpuGeneralScanCapabilityTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.FunctionalGradientRequest;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import java.lang.foreign.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Numerical consumers of typed histories, controlled matrices and genuine empty local graphs, using assigned slots. */
public class CpuTypedEmptyScanConsumerTest {
    private static final List<DataType> STORAGE_TYPES = List.of(DataType.values());

    @Test void carryFreeZeroNodeBodyPublishesEveryWidthAndResetsPaddingOnTheSameBoundInvocation() {
        for (boolean named : List.of(false, true)) for (boolean lengthMode : List.of(false, true))
            for (int time : new int[] {0, 1, 3}) for (boolean reverse : List.of(false, true)) {
                // The named recipe is bound to distinct B values, not compiled at a sample B.
                for (int staticBatch : named ? new int[] {3} : new int[] {0, 1, 3}) {
                    Dimension rows = named ? new DynamicDimension("typedRows") : new StaticDimension(staticBatch);
                    Shape row = Shape.ofDimensions(rows, new StaticDimension(2));
                    Shape stack = Shape.ofDimensions(new StaticDimension(time), rows, new StaticDimension(2));
                    var formals = STORAGE_TYPES.stream().map(t -> value(t, false, row)).toList();
                    var sequences = STORAGE_TYPES.stream().map(t -> value(t, false, stack)).toList();
                    Tensor lengths = value(DataType.INT64, false, Shape.ofDimensions(rows));
                    var body = new ScanBody(List.of(), formals, List.of(), List.of(), formals);
                    var scan = lengthMode
                            ? Scan.apply(List.of(), sequences, List.of(), lengths, 0,
                                    Collections.nCopies(sequences.size(), OptionalInt.of(0)), body, reverse)
                            : Scan.apply(List.of(), sequences, List.of(), body, reverse);
                    var artifact = compile(scan.outputs(), List.of(), List.of(), List.of());
                    for (int batch : named ? new int[] {0, 1, 3} : new int[] {staticBatch})
                        try (var consumer = new Consumer(artifact, named ? OptionalLong.of(batch) : OptionalLong.empty())) {
                            long[] ls = lengths(time, batch);
                            for (int pass = 0; pass < 2; pass++) {
                                for (int p = 0; p < sequences.size(); p++) {
                                    DataType type = STORAGE_TYPES.get(p);
                                    long[] bits = new long[time * batch * 2];
                                    for (int i = 0; i < bits.length; i++) bits[i] = pattern(type, i + pass * 3);
                                    consumer.raw(sequences.get(p), bits);
                                }
                                if (lengthMode) consumer.raw(lengths, ls);
                                if (pass == 0) consumer.bind();
                                consumer.bound.execute();
                                for (int p = 0; p < sequences.size(); p++) {
                                    long[] expected = new long[time * batch * 2];
                                    for (int t = 0; t < time; t++) for (int b = 0; b < batch; b++)
                                        for (int j = 0; j < 2; j++) {
                                            int index = (t * batch + b) * 2 + j;
                                            if (!lengthMode || t < ls[b]) expected[index] = pattern(STORAGE_TYPES.get(p), index + pass * 3);
                                        }
                                    assertArrayEquals(expected, consumer.forwardRaw(p),
                                            "typed passthrough " + STORAGE_TYPES.get(p) + " pass=" + pass + " reverse=" + reverse);
                                }
                                // Mutating borrowed lengths must invalidate old activity and old padding.
                                for (int b = 0; b < batch; b++) ls[b] = pass == 0 ? time - ls[b] : ls[b];
                            }
                            if (lengthMode && batch > 0) {
                                var before = new ArrayList<long[]>();
                                for (int p = 0; p < sequences.size(); p++) before.add(consumer.forwardRaw(p));
                                ls[batch - 1] = time + 1L;
                                consumer.raw(lengths, ls);
                                assertThrows(IllegalArgumentException.class, consumer.bound::execute);
                                for (int p = 0; p < sequences.size(); p++) assertArrayEquals(before.get(p), consumer.forwardRaw(p));
                                ls[batch - 1] = 0;
                                consumer.raw(lengths, ls);
                                consumer.bound.execute();
                                for (int p = 0; p < sequences.size(); p++) for (int t = 0; t < time; t++)
                                    for (int j = 0; j < 2; j++) assertEquals(0L,
                                            consumer.forwardRaw(p)[(t * batch + batch - 1) * 2 + j], "retry resets typed padding");
                            }
                        }
                }
            }
    }

    @Test void inferredStaticArithmeticExecutesForwardAndActualSelectedBackward() {
        int time = 3, batch = 2, features = 3;
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64))
            for (boolean lengthMode : List.of(false,true)) for (boolean reverse : List.of(false,true)) {
                Shape row = Shape.of(batch,features), stack = Shape.of(time,batch,features);
                Tensor h = value(type,true,row), x = value(type,true,row), next = h.add(x).relu();
                var body = new ScanBody(List.of(h),List.of(x),List.of(),List.of(next),List.of(next));
                Tensor initial = value(type,true,row), sequence = value(type,true,stack);
                Tensor lengths = value(DataType.INT64,false,Shape.of(batch));
                Tensor finalSeed = value(type,false,row), emissionSeed = value(type,false,stack);
                var scan = lengthMode ? Scan.apply(List.of(initial),List.of(sequence),List.of(),lengths,1,
                        List.of(OptionalInt.of(0)),body,reverse)
                        : Scan.apply(List.of(initial),List.of(sequence),List.of(),body,reverse);
                var roots = List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
                var artifact = compile(roots,roots,List.of(finalSeed,emissionSeed),List.of(initial,sequence));
                double[] initialValues = {.2,-.3,.5,-.4,.6,.1};
                double[] xs = {.4,.2,-.1,.3,-.2,.5,-.1,.3,.2,.4,.1,-.2,.2,-.1,.3,-.2,.4,.2};
                double[] fs = new double[batch*features], es = new double[xs.length];
                for (int i = 0; i < initialValues.length; i++) initialValues[i] = represented(type,initialValues[i]);
                for (int i = 0; i < xs.length; i++) xs[i] = represented(type,xs[i]);
                for (int i = 0; i < fs.length; i++) fs[i] = .25+i*.0625;
                for (int i = 0; i < es.length; i++) es[i] = .03125+(i%5)*.015625;
                long[] ls = {time,1};
                double[] state = initialValues.clone(), history = new double[xs.length];
                boolean[] positive = new boolean[xs.length];
                for (int step = 0; step < time; step++) {
                    int t = reverse ? time-1-step : step;
                    for (int b = 0; b < batch; b++) if (!lengthMode || t < ls[b])
                        for (int f = 0; f < features; f++) {
                            int cell = (t*batch+b)*features+f, carry = b*features+f;
                            double sum = represented(type,state[carry]+xs[cell]);
                            positive[cell] = sum > 0;
                            state[carry] = Math.max(0,sum); history[cell] = state[carry];
                        }
                }
                double[] dh = fs.clone(), dx = new double[xs.length];
                for (int step = 0; step < time; step++) {
                    int t = reverse ? step : time-1-step;
                    for (int b = 0; b < batch; b++) if (!lengthMode || t < ls[b])
                        for (int f = 0; f < features; f++) {
                            int cell = (t*batch+b)*features+f, carry = b*features+f;
                            double combined = represented(type,dh[carry]+es[cell]);
                            dh[carry] = positive[cell] ? combined : 0; dx[cell] = dh[carry];
                        }
                }
                try (var consumer = new Consumer(artifact,OptionalLong.empty())) {
                    consumer.numbers(initial,initialValues); consumer.numbers(sequence,xs);
                    consumer.numbers(finalSeed,fs); consumer.numbers(emissionSeed,es);
                    if (lengthMode) consumer.raw(lengths,ls);
                    consumer.bind(); consumer.execute();
                    assertArrayEquals(state,consumer.forward(0),0);
                    assertArrayEquals(history,consumer.forward(1),0);
                    assertArrayEquals(dh,consumer.gradient(0),0);
                    assertArrayEquals(dx,consumer.gradient(1),0);
                }
            }
    }

    @Test void inferredComputedViewsPreserveValuesAndSelectedSequenceGradients() {
        int time = 3, batch = 2, features = 3;
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64))
            for (boolean lengthMode : List.of(false,true)) for (boolean reverse : List.of(false,true)) {
                Tensor x = value(type,true,Shape.of(batch,features));
                Tensor output = x.neg().permute(1,0).select(0,1);
                var body = new ScanBody(List.of(),List.of(x),List.of(),List.of(),List.of(output));
                Tensor sequence = value(type,true,Shape.of(time,batch,features));
                Tensor seed = value(type,false,Shape.of(time,batch));
                Tensor lengths = value(DataType.INT64,false,Shape.of(batch));
                var scan = lengthMode ? Scan.apply(List.of(),List.of(sequence),List.of(),lengths,0,
                        List.of(OptionalInt.of(0)),body,reverse)
                        : Scan.apply(List.of(),List.of(sequence),List.of(),body,reverse);
                var artifact = compile(scan.outputs(),scan.outputs(),List.of(seed),List.of(sequence));
                double[] xs = new double[time*batch*features], seeds = new double[time*batch];
                for (int i = 0; i < xs.length; i++) xs[i] = (i-5)*.125;
                for (int i = 0; i < seeds.length; i++) seeds[i] = (i+1)*.0625;
                long[] ls = {time,1};
                double[] expected = new double[seeds.length], dx = new double[xs.length];
                for (int t = 0; t < time; t++) for (int b = 0; b < batch; b++)
                    if (!lengthMode || t < ls[b]) {
                        int selected = (t*batch+b)*features+1;
                        expected[t*batch+b] = -xs[selected]; dx[selected] = -seeds[t*batch+b];
                    }
                try (var consumer = new Consumer(artifact,OptionalLong.empty())) {
                    consumer.numbers(sequence,xs); consumer.numbers(seed,seeds);
                    if (lengthMode) consumer.raw(lengths,ls);
                    consumer.bind(); consumer.execute();
                    assertArrayEquals(expected,consumer.forward(0),0);
                    assertArrayEquals(dx,consumer.gradient(0),0);
                }
            }
    }

    /**
     * Rounds an independent reference site to its declared represented arithmetic width.
     * @param type FLOAT32 or FLOAT64 reference type
     * @param value exact host expression result before the declared rounding boundary
     * @return represented value without changing an operation's arithmetic order
     */
    private static double represented(DataType type,double value) {
        return type == DataType.FLOAT32 ? (double)(float)value : value;
    }

    @Test void namedPointwiseChainsRetainVirtualValuesThroughOrdinaryFusion() {
        Dimension rows = new DynamicDimension("fusedRows");
        Shape step = Shape.ofDimensions(rows, new StaticDimension(2));
        Tensor x = value(DataType.FLOAT64, false, step);
        Tensor sequence = value(DataType.FLOAT64, false,
                Shape.ofDimensions(new StaticDimension(2), rows, new StaticDimension(2)));
        var body = new ScanBody(List.of(), List.of(x), List.of(), List.of(),
                List.of(x.neg().relu(), x.relu().neg()));
        var scan = Scan.apply(List.of(), List.of(sequence), List.of(), body, false);
        var artifact = compile(scan.outputs(), List.of(), List.of(), List.of());
        try (var consumer = new Consumer(artifact, OptionalLong.of(2))) {
            consumer.numbers(sequence, new double[]{-3, 2, -4, 5, -6, 7, -8, 9});
            consumer.bind(); consumer.bound.execute();
            assertArrayEquals(new double[]{3, 0, 4, 0, 6, 0, 8, 0}, consumer.forward(0), 0);
            assertArrayEquals(new double[]{-0.0, -2, -0.0, -5, -0.0, -7, -0.0, -9},
                    consumer.forward(1), 0);
        }
    }

    @Test void repeatedAffineDestinationsRetainTheirOriginalActiveRow() {
        for (DataType type : List.of(DataType.FLOAT64, DataType.INT64))
            for (boolean named : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
                Dimension rows = named ? new DynamicDimension("expandedRows") : new StaticDimension(2);
                Tensor x = value(type, false, Shape.ofDimensions(rows, new StaticDimension(1)));
                Tensor sequence = value(type, false,
                        Shape.ofDimensions(new StaticDimension(1), rows, new StaticDimension(1)));
                Tensor lengths = value(DataType.INT64, false, Shape.ofDimensions(rows));
                Tensor expanded = x.expand(Shape.ofDimensions(rows, new StaticDimension(2)));
                var body = new ScanBody(List.of(), List.of(x), List.of(), List.of(), List.of(expanded));
                var scan = Scan.apply(List.of(), List.of(sequence), List.of(), lengths, 0,
                        List.of(OptionalInt.of(0)), body, reverse);
                var artifact = compile(scan.outputs(), List.of(), List.of(), List.of());
                try (var consumer = new Consumer(artifact, named ? OptionalLong.of(2) : OptionalLong.empty())) {
                    long active = type == DataType.FLOAT64 ? Double.doubleToRawLongBits(41.25) : Long.MAX_VALUE - 7;
                    long poison = type == DataType.FLOAT64 ? Double.doubleToRawLongBits(Double.NaN) : Long.MIN_VALUE;
                    consumer.raw(sequence, new long[]{poison, active});
                    consumer.raw(lengths, new long[]{0, 1});
                    consumer.bind(); consumer.bound.execute();
                    assertArrayEquals(new long[]{0, 0, active, active}, consumer.forwardRaw(0));
                }
            }
    }

    @Test void sharedRepeatedAffineDestinationsPreserveRepresentedValuesWithoutRowControl() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean reverse : List.of(false, true)) {
                Tensor x = value(type, false, Shape.of(2, 1));
                Tensor sequence = value(type, false, Shape.of(1, 2, 1));
                var body = new ScanBody(List.of(), List.of(x), List.of(), List.of(),
                        List.of(x.expand(Shape.of(2, 2))));
                var scan = Scan.apply(List.of(), List.of(sequence), List.of(), body, reverse);
                var artifact = compile(scan.outputs(), List.of(), List.of(), List.of());
                try (var consumer = new Consumer(artifact, OptionalLong.empty())) {
                    long negativeZero = pattern(type, 1), nan = pattern(type, 3);
                    consumer.raw(sequence, new long[]{negativeZero, nan});
                    consumer.bind(); consumer.bound.execute();
                    assertArrayEquals(new long[]{negativeZero, negativeZero, nan, nan}, consumer.forwardRaw(0));
                }
            }
    }

    @Test void absorbedMatmulTransposeKeepsRowActivityInEffectiveOperandCoordinates() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean reverse : List.of(false, true)) {
                Tensor x = value(type, false, Shape.of(2, 2));
                Tensor a = value(type, false, Shape.of(1, 2));
                Tensor sequence = value(type, false, Shape.of(1, 2, 2));
                Tensor weight = value(type, false, Shape.of(1, 2));
                Tensor lengths = value(DataType.INT64, false, Shape.of(2));
                Tensor emission = a.matmul(x.transpose()).transpose();
                var body = new ScanBody(List.of(), List.of(x), List.of(a), List.of(), List.of(emission));
                var scan = Scan.apply(List.of(), List.of(sequence), List.of(weight), lengths, 0,
                        List.of(OptionalInt.of(0)), body, reverse);
                var artifact = compile(scan.outputs(), List.of(), List.of(), List.of());
                try (var consumer = new Consumer(artifact, OptionalLong.empty())) {
                    consumer.numbers(sequence, new double[]{3, 5, Double.NaN, Double.NaN});
                    consumer.numbers(weight, new double[]{1, 2});
                    consumer.raw(lengths, new long[]{1, 0});
                    consumer.bind(); consumer.bound.execute();
                    long active = type == DataType.FLOAT64 ? Double.doubleToRawLongBits(13)
                            : Integer.toUnsignedLong(Float.floatToRawIntBits(13));
                    assertArrayEquals(new long[]{active, 0}, consumer.forwardRaw(0),
                            "full active feature contraction and positive-zero inactive row, " + type + " reverse=" + reverse);
                }
            }
    }

    @Test void controlledStaticThirtyTwoCubeMatmulExecutesBeyondTheStandaloneScalarTilingThreshold() {
        int time = 2, width = 32;
        for (boolean reverse : List.of(false, true)) {
            Tensor x = value(DataType.FLOAT32, false, Shape.of(width, width));
            Tensor w = value(DataType.FLOAT32, false, Shape.of(width, width));
            Tensor sequence = value(DataType.FLOAT32, false, Shape.of(time, width, width));
            Tensor weight = value(DataType.FLOAT32, false, Shape.of(width, width));
            var body = new ScanBody(List.of(), List.of(x), List.of(w), List.of(), List.of(x.matmul(w)));
            var scan = Scan.apply(List.of(), List.of(sequence), List.of(weight), body, reverse);
            var artifact = compile(scan.outputs(), List.of(), List.of(), List.of());
            try (var consumer = new Consumer(artifact, OptionalLong.empty())) {
                for (int pass = 0; pass < 2; pass++) {
                    double[] inputs = new double[time * width * width], weights = new double[width * width];
                    for (int i = 0; i < inputs.length; i++) inputs[i] = ((i * 7 + pass * 3) % 29 - 14) * .125;
                    for (int i = 0; i < weights.length; i++) weights[i] = ((i * 11 + pass) % 17 - 8) * .25;
                    consumer.numbers(sequence, inputs); consumer.numbers(weight, weights);
                    if (pass == 0) consumer.bind();
                    consumer.bound.execute();
                    double[] expected = new double[inputs.length];
                    for (int t = 0; t < time; t++) for (int row = 0; row < width; row++)
                        for (int column = 0; column < width; column++) {
                            float sum = 0;
                            for (int k = 0; k < width; k++)
                                sum += (float) inputs[(t * width + row) * width + k]
                                        * (float) weights[k * width + column];
                            expected[(t * width + row) * width + column] = sum;
                        }
                    assertArrayEquals(expected, consumer.forward(0), "complete 32x32x32 product, pass=" + pass);
                }
            }
        }
    }

    @Test void symbolicMixedTypeMatmulBindsEveryBatchWithoutLosingTypedScalarFacts() {
        int time = 2;
        Dimension rows = new DynamicDimension("matrixRows");
        Shape row = Shape.ofDimensions(rows, new StaticDimension(2));
        Shape stack = Shape.ofDimensions(new StaticDimension(time), rows, new StaticDimension(2));
        for (boolean lengthMode : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
            Tensor x = value(DataType.FLOAT64, false, row), w = value(DataType.FLOAT32, false, Shape.of(2, 2));
            Tensor sequence = value(DataType.FLOAT64, false, stack), weight = value(DataType.FLOAT32, false, Shape.of(2, 2));
            Tensor lengths = value(DataType.INT64, false, Shape.ofDimensions(rows));
            var body = new ScanBody(List.of(), List.of(x), List.of(w), List.of(), List.of(x.matmul(w)));
            var scan = lengthMode
                    ? Scan.apply(List.of(), List.of(sequence), List.of(weight), lengths, 0,
                            List.of(OptionalInt.of(0)), body, reverse)
                    : Scan.apply(List.of(), List.of(sequence), List.of(weight), body, reverse);
            var artifact = compile(scan.outputs(), List.of(), List.of(), List.of());
            for (int batch : new int[] {0, 1, 3})
                try (var consumer = new Consumer(artifact, OptionalLong.of(batch))) {
                    long[] ls = lengths(time, batch);
                    for (int pass = 0; pass < 2; pass++) {
                        double[] weights = {.5 + pass, -2, 1.25, .25 - pass};
                        double[] inputs = new double[time * batch * 2], expected = new double[inputs.length];
                        for (int t = 0; t < time; t++) for (int b = 0; b < batch; b++) {
                            int p = (t * batch + b) * 2;
                            boolean active = !lengthMode || t < ls[b];
                            inputs[p] = active ? (p + 1 + pass) * .125 : Double.NaN;
                            inputs[p + 1] = active ? -(p + 2 + pass) * .25 : Double.NaN;
                            if (active) for (int column = 0; column < 2; column++)
                                expected[p + column] = inputs[p] * weights[column]
                                        + inputs[p + 1] * weights[2 + column];
                        }
                        consumer.numbers(sequence, inputs); consumer.numbers(weight, weights);
                        if (lengthMode) consumer.raw(lengths, ls);
                        if (pass == 0) consumer.bind();
                        consumer.bound.execute();
                        assertArrayEquals(expected, consumer.forward(0),
                                "FLOAT64[B,2] x FLOAT32[2,2], B=" + batch + " pass=" + pass);
                        for (int b = 0; b < batch; b++) ls[b] = time - ls[b];
                    }
                }
        }
    }

    @Test void noRowLengthBodyUsesOnlySharedCarryAndNeverInventsARowTensor() {
        for (boolean named : List.of(false, true)) for (int batch : new int[] {0, 1, 3})
            for (int time : new int[] {0, 1, 3}) for (boolean reverse : List.of(false, true)) {
                Dimension rows = named ? new DynamicDimension("lengthOnlyRows") : new StaticDimension(batch);
                Tensor h = value(DataType.FLOAT64, false, Shape.of(2));
                Tensor x = value(DataType.FLOAT64, false, Shape.of(2));
                Tensor initial = value(DataType.FLOAT64, false, Shape.of(2));
                Tensor sequence = value(DataType.FLOAT64, false, Shape.of(time, 2));
                Tensor lengths = value(DataType.INT64, false, Shape.ofDimensions(rows));
                var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x)), List.of());
                var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(), lengths, 0,
                        List.of(OptionalInt.empty()), body, reverse);
                var artifact = compile(scan.finalCarry(), List.of(), List.of(), List.of());
                try (var consumer = new Consumer(artifact, named ? OptionalLong.of(batch) : OptionalLong.empty())) {
                    consumer.numbers(initial, new double[] {7, -11});
                    double[] input = new double[time * 2];
                    for (int t = 0; t < time; t++) { input[2 * t] = 1L << t; input[2 * t + 1] = -(1L << (t + 3)); }
                    consumer.numbers(sequence, input);
                    long[] ls = lengths(time, batch);
                    consumer.raw(lengths, ls);
                    consumer.bind();
                    consumer.bound.execute();
                    double[] expected = {7, -11};
                    for (int step = 0; step < time; step++) {
                        int t = reverse ? time - 1 - step : step;
                        boolean active = false;
                        for (long l : ls) active |= t < l;
                        if (active) { expected[0] += input[2 * t]; expected[1] += input[2 * t + 1]; }
                    }
                    assertArrayEquals(expected, consumer.forward(0));
                    Arrays.fill(ls, 0);
                    consumer.raw(lengths, ls);
                    consumer.numbers(initial, new double[] {-0d, Double.longBitsToDouble(0x7ff8123456789abcL)});
                    consumer.numbers(sequence, filled(input.length, Double.NaN));
                    consumer.bound.execute();
                    assertArrayEquals(new long[] {Long.MIN_VALUE, 0x7ff8123456789abcL}, consumer.forwardRaw(0));
                }
            }
    }

    @Test void carryFreeSelectedReverseAndIdentityCarryMatchOrdinaryAdAtEmptyAndSingletonBoundaries() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (boolean named : List.of(false, true))
            for (int batch : new int[] {0, 1}) for (int time : new int[] {0, 1})
                for (boolean reverse : List.of(false, true)) for (boolean carry : List.of(false, true)) {
                    Dimension rows = named ? new DynamicDimension("identityRows") : new StaticDimension(batch);
                    Shape row = Shape.ofDimensions(rows, new StaticDimension(2));
                    Shape stack = Shape.ofDimensions(new StaticDimension(time), rows, new StaticDimension(2));
                    Tensor h = value(type, true, row), x = value(type, true, row);
                    Tensor initial = value(type, true, row), sequence = value(type, true, stack);
                    Tensor lengths = value(DataType.INT64, false, Shape.ofDimensions(rows));
                    var body = new ScanBody(carry ? List.of(h) : List.of(), List.of(x), List.of(),
                            carry ? List.of(h) : List.of(), List.of(x));
                    var scan = Scan.apply(carry ? List.of(initial) : List.of(), List.of(sequence), List.of(), lengths,
                            carry ? 1 : 0, List.of(OptionalInt.of(0)), body, reverse);
                    var roots = new ArrayList<Tensor>(scan.finalCarry()); roots.addAll(scan.outputs());
                    var seeds = roots.stream().map(t -> value(type, false, t.descriptor().shape())).toList();
                    var targets = carry ? List.of(initial, sequence) : List.of(sequence);
                    var artifact = compile(roots, roots, seeds, targets);
                    try (var consumer = new Consumer(artifact, named ? OptionalLong.of(batch) : OptionalLong.empty())) {
                        if (carry) consumer.numbers(initial, filled(batch * 2, -2));
                        consumer.numbers(sequence, filled(time * batch * 2, 3));
                        consumer.raw(lengths, filledLong(batch, time));
                        for (var seed : seeds) consumer.numbers(seed, filled(seed == seeds.getLast() ? time * batch * 2 : batch * 2, .75));
                        consumer.bind(); consumer.bound.execute();
                        // This independent ordinary graph asks the public AD engine for identity VJPs;
                        // no derivative of the Scan implementation is reproduced here.
                        Tensor ordinaryInitial = value(type, true, Shape.of(batch, 2));
                        Tensor ordinarySequence = value(type, true, Shape.of(time, batch, 2));
                        var ordinaryRoots = new ArrayList<Tensor>();
                        if (carry) ordinaryRoots.add(ordinaryInitial.contiguous());
                        ordinaryRoots.add(ordinarySequence.contiguous());
                        var ordinarySeeds = ordinaryRoots.stream().map(t -> value(type, false, t.descriptor().shape())).toList();
                        var ordinaryTargets = carry ? List.of(ordinaryInitial, ordinarySequence) : List.of(ordinarySequence);
                        var ordinary = compile(ordinaryRoots, ordinaryRoots, ordinarySeeds, ordinaryTargets);
                        try (var reference = new Consumer(ordinary, OptionalLong.empty())) {
                            if (carry) reference.numbers(ordinaryInitial, filled(batch * 2, -2));
                            reference.numbers(ordinarySequence, filled(time * batch * 2, 3));
                            for (int p = 0; p < ordinarySeeds.size(); p++) reference.numbers(ordinarySeeds.get(p),
                                    filled(p == ordinarySeeds.size() - 1 ? time * batch * 2 : batch * 2, .75));
                            reference.bind(); reference.bound.execute();
                            for (int p = 0; p < roots.size(); p++) assertArrayEquals(reference.forward(p), consumer.forward(p));
                            for (int p = 0; p < targets.size(); p++) assertArrayEquals(reference.gradient(p), consumer.gradient(p));
                        }
                    }
                }
    }

    @Test void typedPrimalHistoriesDriveSelectedReverseComparedWithIndependentActivePrefixOrdinaryAd() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (boolean named : List.of(false, true))
            for (boolean reverse : List.of(false, true)) for (boolean lengthMode : List.of(false, true)) {
                int time = 3, batch = 3, width = 2;
                Dimension rows = named ? new DynamicDimension("mixedRows") : new StaticDimension(batch);
                Shape row = Shape.ofDimensions(rows, new StaticDimension(width));
                Shape stack = Shape.ofDimensions(new StaticDimension(time), rows, new StaticDimension(width));
                Tensor h = value(type, true, row), c = value(DataType.INT64, false, row);
                Tensor x = value(type, true, row), index = value(DataType.INT32, false, row), mask = value(DataType.BOOL, false, row);
                Tensor w = value(type, true, Shape.of(width)), table = value(type, true, Shape.of(4));
                Tensor next = mixed(h, c, x, index, mask, w, table);
                var body = new ScanBody(List.of(h, c), List.of(x, index, mask), List.of(w, table),
                        List.of(next, index.cast(DataType.INT64)), List.of(next, mask, index));
                Tensor initial = value(type, true, row), initialC = value(DataType.INT64, false, row);
                Tensor sequence = value(type, true, stack), indices = value(DataType.INT32, false, stack), masks = value(DataType.BOOL, false, stack);
                Tensor weight = value(type, true, Shape.of(width)), lookup = value(type, true, Shape.of(4));
                Tensor lengths = value(DataType.INT64, false, Shape.ofDimensions(rows));
                var scan = lengthMode
                        ? Scan.apply(List.of(initial, initialC), List.of(sequence, indices, masks), List.of(weight, lookup),
                                lengths, 2, Collections.nCopies(3, OptionalInt.of(0)), body, reverse)
                        : Scan.apply(List.of(initial, initialC), List.of(sequence, indices, masks), List.of(weight, lookup), body, reverse);
                var roots = new ArrayList<Tensor>(scan.finalCarry()); roots.addAll(scan.outputs());
                var selected = List.of(roots.get(0), roots.get(2));
                var seeds = List.of(value(type, false, row), value(type, false, stack));
                var targets = List.of(initial, sequence, weight, lookup);
                var artifact = compile(roots, selected, seeds, targets);
                long[] ls = lengthMode ? new long[] {3, 1, 0} : new long[] {3, 3, 3};
                double[] initialValues = {.25, -.5, .75, 1, -2, 3};
                long[] initialInts = {2, 3, 5, 7, 11, 13};
                double[] xs = new double[time * batch * width];
                long[] ix = new long[xs.length], ms = new long[xs.length];
                for (int t = 0; t < time; t++) for (int b = 0; b < batch; b++) for (int j = 0; j < width; j++) {
                    int p = (t * batch + b) * width + j;
                    xs[p] = t < ls[b] ? (p + 1) * .125 : Double.NaN;
                    ix[p] = t < ls[b] ? (p + 1) % 4 : Integer.MAX_VALUE;
                    ms[p] = t < ls[b] ? p % 2 : 127; // inactive BOOL/index values must not be validated or loaded
                }
                try (var consumer = new Consumer(artifact, named ? OptionalLong.of(batch) : OptionalLong.empty())) {
                    consumer.numbers(initial, initialValues); consumer.raw(initialC, initialInts);
                    consumer.numbers(sequence, xs); consumer.raw(indices, ix); consumer.raw(masks, ms);
                    consumer.numbers(weight, new double[] {.5, -.25}); consumer.numbers(lookup, new double[] {1, 2, 4, 8});
                    consumer.raw(lengths, ls); consumer.numbers(seeds.get(0), filled(batch * width, .5));
                    double[] emissionSeeds = filled(xs.length, Double.NaN);
                    for (int t = 0; t < time; t++) for (int b = 0; b < batch; b++) if (t < ls[b])
                        Arrays.fill(emissionSeeds, (t * batch + b) * width, (t * batch + b + 1) * width, .25);
                    consumer.numbers(seeds.get(1), emissionSeeds);
                    consumer.bind(); consumer.bound.execute();
                    double[] expectedFinal = new double[batch * width], expectedHistory = new double[xs.length];
                    double[] expectedInitialGradient = new double[batch * width], expectedSequenceGradient = new double[xs.length];
                    double[] expectedWeightGradient = new double[width], expectedLookupGradient = new double[4];
                    long[] expectedCarry = initialInts.clone(), expectedMasks = new long[xs.length], expectedIndices = new long[xs.length];
                    for (int b = 0; b < batch; b++) {
                        int active = Math.toIntExact(ls[b]);
                        Tensor rh = value(type, true, Shape.of(width)), rc = value(DataType.INT64, false, Shape.of(width));
                        Tensor rx = value(type, true, Shape.of(active, width)), ri = value(DataType.INT32, false, Shape.of(active, width));
                        Tensor rm = value(DataType.BOOL, false, Shape.of(active, width));
                        Tensor rw = value(type, true, Shape.of(width)), rt = value(type, true, Shape.of(4));
                        Tensor state = rh, discrete = rc; Tensor[] emissions = new Tensor[active];
                        for (int step = 0; step < active; step++) {
                            int t = reverse ? active - 1 - step : step;
                            state = mixed(state, discrete, rx.select(0, t), ri.select(0, t), rm.select(0, t), rw, rt);
                            discrete = ri.select(0, t).cast(DataType.INT64); emissions[t] = state;
                        }
                        var rr = new ArrayList<Tensor>(); rr.add(state.contiguous());
                        if (active > 0) rr.add(Tensor.stack(0, emissions));
                        var rs = rr.stream().map(t -> value(type, false, t.descriptor().shape())).toList();
                        // An empty ordinary prefix has only rh in its forward inventory.
                        // ZERO does not authorize requesting rx/rw/rt that are not in that graph.
                        var referenceTargets = active == 0 ? List.of(rh) : List.of(rh, rx, rw, rt);
                        if (active == 0) assertThrows(IllegalArgumentException.class,
                                () -> compile(rr, rr, rs, List.of(rh, rx, rw, rt)));
                        var referenceArtifact = compile(rr, rr, rs, referenceTargets);
                        try (var reference = new Consumer(referenceArtifact, OptionalLong.empty())) {
                            reference.numbers(rh, Arrays.copyOfRange(initialValues, b * width, (b + 1) * width));
                            reference.raw(rc, Arrays.copyOfRange(initialInts, b * width, (b + 1) * width));
                            reference.numbers(rx, rowPrefix(xs, batch, b, active));
                            reference.raw(ri, rowPrefix(ix, batch, b, active)); reference.raw(rm, rowPrefix(ms, batch, b, active));
                            reference.numbers(rw, new double[] {.5, -.25}); reference.numbers(rt, new double[] {1, 2, 4, 8});
                            reference.numbers(rs.getFirst(), filled(width, .5));
                            if (active > 0) reference.numbers(rs.getLast(), filled(active * width, .25));
                            reference.bind(); reference.bound.execute();
                            System.arraycopy(reference.forward(0), 0, expectedFinal, b * width, width);
                            System.arraycopy(reference.gradient(0), 0, expectedInitialGradient, b * width, width);
                            for (int t = 0; t < active; t++) {
                                System.arraycopy(reference.forward(1), t * width, expectedHistory, (t * batch + b) * width, width);
                                System.arraycopy(reference.gradient(1), t * width, expectedSequenceGradient, (t * batch + b) * width, width);
                                for (int j = 0; j < width; j++) {
                                    int p = (t * batch + b) * width + j;
                                    expectedMasks[p] = ms[p]; expectedIndices[p] = ix[p];
                                }
                            }
                            if (active > 0) {
                                for (int j = 0; j < width; j++) expectedWeightGradient[j] += reference.gradient(2)[j];
                                for (int j = 0; j < 4; j++) expectedLookupGradient[j] += reference.gradient(3)[j];
                            }
                        }
                        if (active > 0) for (int j = 0; j < width; j++)
                            expectedCarry[b * width + j] = ix[((reverse ? 0 : active - 1) * batch + b) * width + j];
                    }
                    double tolerance = type == DataType.FLOAT32 ? 1e-5 : 1e-13;
                    assertArrayEquals(expectedFinal, consumer.forward(0), tolerance);
                    assertArrayEquals(expectedCarry, consumer.forwardRaw(1));
                    assertArrayEquals(expectedHistory, consumer.forward(2), tolerance);
                    assertArrayEquals(expectedMasks, consumer.forwardRaw(3)); assertArrayEquals(expectedIndices, consumer.forwardRaw(4));
                    assertArrayEquals(expectedInitialGradient, consumer.gradient(0), tolerance);
                    assertArrayEquals(expectedSequenceGradient, consumer.gradient(1), tolerance);
                    assertArrayEquals(expectedWeightGradient, consumer.gradient(2), tolerance);
                    assertArrayEquals(expectedLookupGradient, consumer.gradient(3), tolerance);
                }
            }
    }

    @Test void invariantOnlySelectedReverseHasNoRowCarryAndResetsItsSharedAccumulatorAfterEmptyRerun() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (boolean named : List.of(false, true))
            for (int batch : new int[] {0, 1, 3}) for (int time : new int[] {0, 1, 3})
                for (boolean reverse : List.of(false, true)) {
                    Dimension rows = named ? new DynamicDimension("invariantOnlyRows") : new StaticDimension(batch);
                    Shape row = Shape.ofDimensions(rows, new StaticDimension(2));
                    Shape stack = Shape.ofDimensions(new StaticDimension(time), rows, new StaticDimension(2));
                    Tensor x = value(type, false, row), w = value(type, true, Shape.of(2));
                    Tensor sequence = value(type, false, stack), weight = value(type, true, Shape.of(2));
                    Tensor lengths = value(DataType.INT64, false, Shape.ofDimensions(rows));
                    var scan = Scan.apply(List.of(), List.of(sequence), List.of(weight), lengths, 0,
                            List.of(OptionalInt.of(0)), new ScanBody(List.of(), List.of(x), List.of(w), List.of(), List.of(x.mul(w))), reverse);
                    Tensor seed = value(type, false, stack);
                    var artifact = compile(scan.outputs(), scan.outputs(), List.of(seed), List.of(weight));
                    long[] ls = lengths(time, batch);
                    double[] xs = filled(time * batch * 2, Double.NaN), ss = filled(xs.length, Double.NaN);
                    double[] expected = new double[xs.length];
                    for (int t = 0; t < time; t++) for (int b = 0; b < batch; b++) if (t < ls[b])
                        for (int j = 0; j < 2; j++) {
                            int p = (t * batch + b) * 2 + j;
                            xs[p] = p + 1; ss[p] = (p % 3 + 1) * .25;
                            expected[p] = xs[p] * (j == 0 ? .5 : -.25);
                        }
                    // Each ordinary output is one active represented time slice, not a Scan
                    // or a manually differentiated recurrence. Exact binary fractions make
                    // its independent root-accumulation order immaterial to this witness.
                    var ordinaryRoots = new ArrayList<Tensor>();
                    var ordinarySeeds = new ArrayList<Tensor>();
                    var ordinaryInputs = new ArrayList<Tensor>();
                    var ordinaryData = new ArrayList<double[]>();
                    var ordinarySeedData = new ArrayList<double[]>();
                    Tensor ordinaryWeight = value(type, true, Shape.of(2));
                    for (int step = 0; step < Math.max(1, time); step++) {
                        int t = reverse ? time - 1 - step : step;
                        int activeRows = 0;
                        for (long length : ls) if (t >= 0 && t < length) activeRows++;
                        Tensor input = value(type, false, Shape.of(activeRows, 2));
                        Tensor output = input.mul(ordinaryWeight);
                        ordinaryInputs.add(input); ordinaryRoots.add(output);
                        ordinarySeeds.add(value(type, false, output.descriptor().shape()));
                        double[] data = new double[activeRows * 2], seedData = new double[data.length];
                        int position = 0;
                        for (int b = 0; b < batch; b++) if (t >= 0 && t < ls[b])
                            for (int j = 0; j < 2; j++) {
                                data[position] = xs[(t * batch + b) * 2 + j];
                                seedData[position++] = ss[(t * batch + b) * 2 + j];
                            }
                        ordinaryData.add(data); ordinarySeedData.add(seedData);
                    }
                    var ordinary = compile(ordinaryRoots, ordinaryRoots, ordinarySeeds, List.of(ordinaryWeight));
                    try (var consumer = new Consumer(artifact, named ? OptionalLong.of(batch) : OptionalLong.empty());
                            var reference = new Consumer(ordinary, OptionalLong.empty())) {
                        consumer.numbers(sequence, xs); consumer.numbers(weight, new double[] {.5, -.25});
                        consumer.numbers(seed, ss); consumer.raw(lengths, ls);
                        reference.numbers(ordinaryWeight, new double[] {.5, -.25});
                        for (int p = 0; p < ordinaryInputs.size(); p++) {
                            reference.numbers(ordinaryInputs.get(p), ordinaryData.get(p));
                            reference.numbers(ordinarySeeds.get(p), ordinarySeedData.get(p));
                        }
                        reference.bind(); reference.bound.execute();
                        consumer.bind(); consumer.bound.execute();
                        assertArrayEquals(expected, consumer.forward(0));
                        assertArrayEquals(reference.gradient(0), consumer.gradient(0));
                        // The same bound reverse must not reuse a completed shared contribution.
                        consumer.raw(lengths, new long[batch]);
                        consumer.numbers(sequence, filled(xs.length, Double.NaN));
                        consumer.numbers(weight, new double[] {Double.NaN, Double.NaN});
                        consumer.bound.execute();
                        assertArrayEquals(new double[xs.length], consumer.forward(0));
                        assertArrayEquals(new long[] {0, 0}, consumer.logical(
                                artifact.publication().gradientBindings().getFirst().valueId()), "absent contributions retain positive-zero initialization");
                    }
                }
    }

    @Test void typedPassthroughRejectsMisalignmentOverlapReadOnlyAndClosedBorrowedStorage() {
        Tensor x = value(DataType.INT64, false, Shape.of(1, 2));
        Tensor sequence = value(DataType.INT64, false, Shape.of(1, 1, 2));
        var scan = Scan.apply(List.of(), List.of(sequence), List.of(),
                new ScanBody(List.of(), List.of(x), List.of(), List.of(), List.of(x)), false);
        var artifact = compile(scan.outputs(), List.of(), List.of(), List.of());
        ValueId input = artifact.constants().bindableInputBindings().stream().filter(b -> b.tensorId().equals(sequence.id())).findFirst().orElseThrow().valueId();
        ValueId output = artifact.publication().forwardBindings().getFirst().valueId();
        try (var arena = Arena.ofShared()) {
            var storage = arena.allocate(64, 8);
            for (var overrides : List.of(Map.of(input, storage.asSlice(1, 16)),
                    Map.of(input, storage.asSlice(0, 16), output, storage.asSlice(0, 16)),
                    Map.of(output, storage.asSlice(0, 16).asReadOnly())))
                try (var consumer = new Consumer(artifact, OptionalLong.empty(), overrides)) {
                    assertThrows(RuntimeException.class, consumer::bind);
                }
        }
        var external = Arena.ofShared();
        try (var consumer = new Consumer(artifact, OptionalLong.empty(), Map.of(input, external.allocate(16, 8)))) {
            consumer.raw(sequence, new long[] {Long.MIN_VALUE, Long.MAX_VALUE});
            consumer.bind(); consumer.bound.execute();
            assertArrayEquals(new long[] {Long.MIN_VALUE, Long.MAX_VALUE}, consumer.forwardRaw(0));
            external.close();
            assertThrows(RuntimeException.class, consumer.bound::execute);
        } finally {
            if (external.scope().isAlive()) external.close();
        }
    }

    private static Tensor mixed(Tensor h, Tensor c, Tensor x, Tensor index, Tensor mask, Tensor w, Tensor table) {
        return h.mul(w).add(Tensor.where(mask, x, x.neg())).add(table.gather(index, 0)).add(c.cast(h.descriptor().dataType()));
    }

    private static Tensor value(DataType type, boolean gradient, Shape shape) {
        return TensorFactory.create(new TensorDescriptor(type, shape,
                shape.isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(), gradient));
    }

    private static CompileArtifacts compile(List<Tensor> roots, List<Tensor> selected, List<Tensor> seeds, List<Tensor> targets) {
        Optional<FunctionalGradientRequest> request = selected.isEmpty() ? Optional.empty()
                : Optional.of(new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(selected.stream()
                        .map(t -> (FunctionalGradientRequest.OutputReference) new FunctionalGradientRequest.ForwardTensorReference(t)).toList(),
                        seeds.stream().map(Optional::of).toList(), targets, false, FunctionalGradientRequest.DisconnectedPolicy.ZERO))));
        return CpuGeneralScanCapabilityTest.compile(roots, request, false, new CpuCapabilityProvider());
    }

    private static long[] lengths(int time, int batch) {
        long[] result = new long[batch];
        for (int b = 0; b < batch; b++) result[b] = b == 0 ? time : b == 1 ? Math.min(1, time) : 0;
        return result;
    }

    private static double[] filled(int count, double value) { double[] result = new double[count]; Arrays.fill(result, value); return result; }
    private static long[] filledLong(int count, long value) { long[] result = new long[count]; Arrays.fill(result, value); return result; }
    private static double[] rowPrefix(double[] values, int batch, int row, int time) {
        double[] result = new double[time * 2];
        for (int t = 0; t < time; t++) System.arraycopy(values, (t * batch + row) * 2, result, t * 2, 2);
        return result;
    }
    private static long[] rowPrefix(long[] values, int batch, int row, int time) {
        long[] result = new long[time * 2];
        for (int t = 0; t < time; t++) System.arraycopy(values, (t * batch + row) * 2, result, t * 2, 2);
        return result;
    }

    private static long pattern(DataType type, int index) {
        return switch (type) {
            case BOOL -> index & 1;
            case INT32 -> new long[] {0, 0x80000000L, 0x7fffffffL, 0xfedcba98L}[index % 4];
            case INT64 -> new long[] {0, Long.MIN_VALUE, Long.MAX_VALUE, 0x123456789abcdef0L}[index % 4];
            case FLOAT16, BFLOAT16 -> new long[] {0, 0x8000, 1, 0x7e35, 0xfe47}[index % 5];
            case FLOAT8_E4M3, FLOAT8_E5M2 -> new long[] {0, 0x80, 1, 0x7f, 0xff, 0x7d, 0xfd}[index % 7];
            case FLOAT32 -> new long[] {0, 0x80000000L, 1, 0x7fc12345L, 0xffc54321L}[index % 5];
            case FLOAT64 -> new long[] {0, Long.MIN_VALUE, 1, 0x7ff8123456789abcL, 0xfff8543210abcdefL}[index % 5];
        };
    }

    /** Owns complete real graph preparation, source publications and borrowed input writers. */
    public static final class Consumer implements AutoCloseable {
        private final CompileArtifacts artifact;
        private final CpuRecurrentScanExecutionTest.Fixture fixture;
        private final long batch;
        private final Set<ValueId> produced = new HashSet<>();
        private io.github.pho001.synaptik.runtime.execution.BoundInvocation bound;

        public Consumer(CompileArtifacts artifact, OptionalLong batch) { this(artifact, batch, Map.of()); }
        public Consumer(CompileArtifacts artifact, OptionalLong batch, Map<ValueId, MemorySegment> overrides) {
            this(artifact,batch,overrides,CpuPartitionAnalysisInputs.DEFAULT);
        }
        public Consumer(CompileArtifacts artifact, OptionalLong batch, Map<ValueId, MemorySegment> overrides,
                CpuPartitionAnalysisInputs inputs) {
            this(artifact, batch, overrides, inputs, new CpuPartitionPreparer().analyze(
                    GraphPreparation.project(artifact, artifact.partitions().getFirst(), inputs)));
        }

        /**
         * Finalizes the actual analyzed plan through complete graph preparation, preserving
         * source-only caller/constant roles instead of inventing partition requirements.
         * @param artifact original compiled graph and publication/constant bindings
         * @param batch actual named batch binding, or empty for static geometry
         * @param overrides borrowed caller storage keyed by actual graph value IDs
         * @param inputs exact immutable inputs used for the supplied actual analysis
         * @param analysis actual CPU analysis of the graph's projected Prepare context
         */
        public Consumer(CompileArtifacts artifact, OptionalLong batch, Map<ValueId, MemorySegment> overrides,
                CpuPartitionAnalysisInputs inputs, BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis) {
            this.artifact = artifact; this.batch = batch.orElse(0);
            artifact.graph().nodes().forEach(node -> produced.addAll(node.outputs()));
            fixture = new CpuRecurrentScanExecutionTest.Fixture(
                    artifact, inputs, analysis, 1, true, true, overrides, batch);
        }
        public void bind() { bound = fixture.executable.bind(fixture.state); }
        public void execute() { bound.execute(); }
        public MemorySegment inputStorage(Tensor tensor) {
            return segment(artifact.constants().bindableInputBindings().stream().filter(b -> b.tensorId().equals(tensor.id()))
                    .findFirst().orElseThrow().valueId());
        }
        public MemorySegment forwardStorage(int position) { return segment(artifact.publication().forwardBindings().get(position).valueId()); }
        public MemorySegment gradientStorage(int position) { return segment(artifact.publication().gradientBindings().get(position).valueId()); }
        public MemorySegment valueStorage(ValueId id) { return segment(id); }

        /**
         * Poisons only genuinely produced publications. Source-only caller aliases and initialized
         * constants are inputs, not writable recomputed outputs; mutating them would change the
         * test's inputs rather than witness a missed write.
         * @param poison byte pattern applied to every actual produced publication
         */
        public void poisonProducedPublications(byte poison) {
            for (var binding : artifact.publication().forwardBindings())
                if (produced.contains(binding.valueId())) segment(binding.valueId()).fill(poison);
            for (var binding : artifact.publication().gradientBindings())
                if (produced.contains(binding.valueId())) segment(binding.valueId()).fill(poison);
        }
        /**
         * Initializes actual declared input storage, ignoring only omitted/unread unpublished
         * source inputs. A source-only publication is never replaced by fallback storage.
         * @param tensor exact original input Tensor
         * @param bits represented cells copied at the Tensor's actual declared width
         * @throws IndexOutOfBoundsException if represented cells exceed assigned input capacity
         * @throws IllegalStateException if actual borrowed storage is inaccessible or closed
         */
        public void raw(Tensor tensor, long[] bits) {
            // An ordinary reference may omit the input entirely; a Scan can retain an unused
            // source actual without declaring a physical read. Neither case invents storage.
            var binding = artifact.constants().bindableInputBindings().stream().filter(b -> b.tensorId().equals(tensor.id())).findFirst();
            if (binding.isEmpty()) return;
            ValueId id = binding.orElseThrow().valueId();
            if (!fixture.storage.containsKey(id)
                    && artifact.publication().forwardBindings().stream().noneMatch(b -> b.valueId().equals(id))
                    && artifact.publication().gradientBindings().stream().noneMatch(b -> b.valueId().equals(id))) return;
            // A published source still requires authentic assigned storage; never hide it.
            var segment = segment(id);
            for (int i = 0; i < bits.length; i++) put(tensor.descriptor().dataType(), segment, i, bits[i]);
        }
        public void numbers(Tensor tensor, double[] values) {
            long[] bits = new long[values.length];
            for (int i = 0; i < values.length; i++) bits[i] = tensor.descriptor().dataType() == DataType.FLOAT64
                    ? Double.doubleToRawLongBits(values[i]) : Integer.toUnsignedLong(Float.floatToRawIntBits((float) values[i]));
            raw(tensor, bits);
        }
        private TensorDescriptor descriptor(ValueId id) {
            return artifact.graph().values().stream().filter(v -> v.id().equals(id)).findFirst().orElseThrow().descriptor();
        }
        private MemorySegment segment(ValueId id) { return fixture.segment(Math.toIntExact(id.value())); }
        public long[] forwardRaw(int position) { return logical(artifact.publication().forwardBindings().get(position).valueId()); }
        public double[] forward(int position) { var id = artifact.publication().forwardBindings().get(position).valueId(); return numbers(id); }
        public double[] gradient(int position) { return numbers(artifact.publication().gradientBindings().get(position).valueId()); }
        private double[] numbers(ValueId id) {
            long[] bits = logical(id); double[] result = new double[bits.length];
            for (int i = 0; i < bits.length; i++) result[i] = descriptor(id).dataType() == DataType.FLOAT64
                    ? Double.longBitsToDouble(bits[i]) : Float.intBitsToFloat((int) bits[i]);
            return result;
        }
        private long[] logical(ValueId id) {
            var descriptor = descriptor(id);
            long[] shape = descriptor.shape().dimensions().stream().mapToLong(d -> d instanceof StaticDimension s ? s.size() : batch).toArray();
            long count = 1; for (long extent : shape) count *= extent;
            long[] strides = descriptor.layout().map(LayoutDescriptor::strides).orElseGet(() -> {
                long[] result = new long[shape.length]; long stride = 1;
                for (int i = shape.length - 1; i >= 0; i--) { result[i] = stride; stride *= shape[i]; }
                return result;
            });
            long[] result = new long[Math.toIntExact(count)]; var segment = segment(id);
            for (int i = 0; i < result.length; i++) {
                long address = descriptor.layout().map(LayoutDescriptor::storageOffset).orElse(0L), remaining = i;
                for (int axis = shape.length - 1; axis >= 0; axis--) { address += (remaining % shape[axis]) * strides[axis]; remaining /= shape[axis]; }
                result[i] = get(descriptor.dataType(), segment, address);
            }
            return result;
        }
        @Override public void close() { fixture.close(); }
    }

    private static void put(DataType type, MemorySegment segment, long index, long bits) {
        switch (type.byteWidth()) {
            case 1 -> segment.setAtIndex(ValueLayout.JAVA_BYTE, index, (byte) bits);
            case 2 -> segment.setAtIndex(ValueLayout.JAVA_SHORT, index, (short) bits);
            case 4 -> segment.setAtIndex(ValueLayout.JAVA_INT, index, (int) bits);
            case 8 -> segment.setAtIndex(ValueLayout.JAVA_LONG, index, bits);
            default -> throw new AssertionError(type);
        }
    }
    private static long get(DataType type, MemorySegment segment, long index) {
        return switch (type.byteWidth()) {
            case 1 -> Byte.toUnsignedLong(segment.getAtIndex(ValueLayout.JAVA_BYTE, index));
            case 2 -> Short.toUnsignedLong(segment.getAtIndex(ValueLayout.JAVA_SHORT, index));
            case 4 -> Integer.toUnsignedLong(segment.getAtIndex(ValueLayout.JAVA_INT, index));
            case 8 -> segment.getAtIndex(ValueLayout.JAVA_LONG, index);
            default -> throw new AssertionError(type);
        };
    }
}
