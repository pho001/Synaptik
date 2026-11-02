package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.CpuGeneralScanCapabilityTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.FunctionalGradientRequest;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Numerical consumers of Compiler-selected Scan derivatives through assigned CPU executables. */
public class CpuSelectedScanActivityConsumerTest {
    @Test void absentSqrtCotangentIsNotAnExplicitZeroEvenWithAnUnselectedPublishedFinalCarry() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean reverse : List.of(false, true)) for (boolean length : List.of(false, true))
                for (boolean explicitZero : List.of(false, true)) {
                    Tensor h = leaf(type, true, 2, 1), clock = leaf(DataType.BOOL, false, 2, 1);
                    Tensor w = leaf(type, true, 1), initial = leaf(type, true, 2, 1);
                    Tensor sequence = leaf(DataType.BOOL, false, 2, 2, 1), weight = leaf(type, true, 1);
                    Tensor lengths = leaf(DataType.INT64, false, 2);
                    var body = new ScanBody(List.of(h), List.of(clock), List.of(w),
                            List.of(h.sqrt().mul(w)), List.of(h));
                    var scan = bind(initial, List.of(sequence), List.of(weight), lengths, body, reverse, length);
                    var roots = List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst());
                    Tensor fs = leaf(type, false, 2, 1), es = leaf(type, false, 2, 2, 1);
                    var selected = explicitZero ? roots : List.of(roots.get(1));
                    var seeds = explicitZero ? List.of(fs, es) : List.of(es);
                    var artifact = compile(roots, selected, seeds, List.of(initial, weight), false);
                    var feeds = feeds(initial, new double[2], sequence, new double[4], weight, new double[]{1.5},
                            lengths, new double[]{2, 1}, fs, new double[2], es, new double[]{1, 1, 1, Double.NaN});
                    if (!length) feeds.put(es, new double[]{1, 1, 1, 1});
                    var result = execute(artifact, feeds);
                    if (explicitZero) {
                        assertTrue(Double.isNaN(result.get(2)[0]), "a present zero must execute sqrt's singular VJP");
                        assertTrue(Double.isNaN(result.get(2)[1]));
                        assertTrue(Double.isNaN(result.get(3)[0]));
                    } else {
                        assertArrayEquals(length ? new double[]{Double.POSITIVE_INFINITY, 1}
                                : new double[]{Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY}, result.get(2));
                        rawZero(0d, result.get(3)[0]);
                    }
                }
    }

    @Test void optionalMergePreservesEitherSoleSignedZeroInsteadOfAddingASyntheticZero() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean optimized : List.of(false, true)) for (int selectedPort : new int[]{0, 1}) {
                Tensor h = leaf(type, true, 1, 1), x = leaf(type, false, 1, 1);
                Tensor initial = leaf(type, true, 1, 1), sequence = leaf(type, false, 1, 1, 1);
                var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(),
                        new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.neg()), List.of(h)), false);
                var roots = List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst());
                Tensor seed = leaf(type, false, selectedPort == 0 ? new long[]{1, 1} : new long[]{1, 1, 1});
                // Final-only NEG produces -0 from +0; emission-only has the raw -0 seed.
                double seedValue = selectedPort == 0 ? 0d : -0d;
                var result = execute(compile(roots, List.of(roots.get(selectedPort)), List.of(seed),
                        List.of(initial), optimized), feeds(initial, new double[]{2}, sequence, new double[]{0}, seed, new double[]{seedValue}));
                rawZero(-0d, result.get(2)[0]);
            }
    }

    @Test void eachCompletedParameterContributionHasItsOwnOrderedAddIncludingTheInitialPositiveZero() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean reverse : List.of(false, true)) for (int time : new int[]{1, 3}) {
                Tensor h = leaf(type, true, 1, 1), x = leaf(type, true, 1, 1), w = leaf(type, true, 1);
                Tensor initial = leaf(type, true, 1, 1), sequence = leaf(type, true, time, 1, 1), weight = leaf(type, true, 1);
                var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(h), List.of(x.mul(w)));
                var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(weight), body, reverse);
                Tensor seed = leaf(type, false, time, 1, 1);
                double large = type == DataType.FLOAT64 ? 0x1p54 : 0x1p25;
                double[] data = time == 1 ? new double[]{-0d} : reverse
                        ? new double[]{large, -large, 1} : new double[]{1, -large, large};
                double[] ones = new double[time]; Arrays.fill(ones, 1);
                var input = feeds(initial, new double[]{0}, sequence, data, weight, new double[]{1}, seed, ones);
                var result = execute(compile(scan.outputs(), scan.outputs(), List.of(seed), List.of(weight), false), input);
                double expected = 0d;
                // Each contribution comes from actual ordinary autograd, not a handwritten VJP.
                for (int step = 0; step < time; step++) {
                    int t = reverse ? step : time - 1 - step;
                    Tensor ordinary = sequence.select(0, t).mul(weight), ordinarySeed = seed.select(0, t);
                    double contribution = execute(compile(List.of(ordinary), List.of(ordinary), List.of(ordinarySeed),
                            List.of(weight), false), input).get(1)[0];
                    expected = type == DataType.FLOAT64 ? expected + contribution : (float)(expected + contribution);
                }
                assertEquals(time == 1 ? 0d : 1d, expected);
                assertEquals(Double.doubleToRawLongBits(expected), Double.doubleToRawLongBits(result.get(1)[0]));
            }
    }

    @Test void anAbsentPrimalOnlyCrossEntropyVjpDoesNotRunItsSoftmaxValidatorButAPresentZeroDoes() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (boolean reverse : List.of(false, true)) {
            Tensor h = leaf(type, true, 2, 2), target = leaf(DataType.INT64, false, 2);
            Tensor next = h.categoricalCrossEntropyWithLogits(target, 1, LossReduction.NONE)
                    .expandDims(1).expand(Shape.of(2, 2));
            Tensor initial = leaf(type, true, 2, 2), targets = leaf(DataType.INT64, false, 2, 2);
            Tensor lengths = leaf(DataType.INT64, false, 2), fs = leaf(type, false, 2, 2), es = leaf(type, false, 2, 2, 2);
            var scan = bind(initial, List.of(targets), List.of(), lengths,
                    new ScanBody(List.of(h), List.of(target), List.of(), List.of(next), List.of(h)), reverse, true);
            var roots = List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst());
            double max = type == DataType.FLOAT64 ? Double.MAX_VALUE : Float.MAX_VALUE;
            var input = feeds(initial, new double[]{-max, max, -max, max}, targets, new double[4], lengths,
                    new double[]{2, 1}, fs, new double[4], es, new double[]{1, 1, 1, 1, 1, 1, Double.NaN, Double.NaN});
            var absent = execute(compile(roots, List.of(roots.get(1)), List.of(es), List.of(initial), false), input);
            assertArrayEquals(new double[]{-1, 3, 1, 1}, absent.get(2));
            var present = compile(roots, roots, List.of(fs, es), List.of(initial), false);
            try (var run = new Run(present, input)) {
                assertThrows(IllegalArgumentException.class, run.bound::execute);
                // The same bound invocation retries after replacing logits; its validator cursors reset.
                run.put(initial, new double[]{-.5, .5, -.5, .5});
                run.bound.execute();
                for (double value : run.publications().get(2)) assertTrue(Double.isFinite(value));
            }
        }
    }

    private enum Guard { SOFTMAX, LOSS_TARGET, BOOL, GATHER, GATHER_ND, SCATTER_BOUNDS, SCATTER_DUPLICATE }

    @Test void aBadLastActiveRowRejectsBeforeEarlierRowsPublishAndTheSameInvocationRetries() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (Guard guard : Guard.values()) {
            Tensor h = leaf(type, true, 3, 3), x = leaf(type, true, 3, 3);
            DataType auxiliaryType = guard == Guard.BOOL ? DataType.BOOL : DataType.INT64;
            long[] auxiliaryShape = guard == Guard.LOSS_TARGET ? new long[]{3}
                    : guard == Guard.GATHER_ND ? new long[]{3, 3, 1} : new long[]{3, 3};
            Tensor aux = leaf(auxiliaryType, false, auxiliaryShape);
            Tensor guarded = guarded(guard, x, aux), next = h.add(guarded);
            Tensor initial = leaf(type, true, 3, 3), sequence = leaf(type, true, 2, 3, 3);
            Tensor auxiliary = leaf(auxiliaryType, false, guard == Guard.LOSS_TARGET ? new long[]{2, 3}
                    : guard == Guard.GATHER_ND ? new long[]{2, 3, 3, 1} : new long[]{2, 3, 3});
            List<Tensor> formals = guard == Guard.SOFTMAX ? List.of(x) : List.of(x, aux);
            List<Tensor> actuals = guard == Guard.SOFTMAX ? List.of(sequence) : List.of(sequence, auxiliary);
            var body = new ScanBody(List.of(h), formals, List.of(), List.of(next), List.of(guarded));
            Tensor lengths = leaf(DataType.INT64, false, 3);
            var scan = bind(initial, actuals, List.of(), lengths, body, false, true);
            var roots = List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst());
            Tensor fs = leaf(type, false, 3, 3), es = leaf(type, false, 2, 3, 3);
            var artifact = compile(roots, roots, List.of(fs, es), List.of(initial, sequence), false);
            double[] validData = {.1, .2, .3, -.2, .4, .1, Double.NaN, Double.NaN, Double.NaN,
                    .3, -.1, .2, Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN};
            double[] auxiliaryData = new double[guard == Guard.LOSS_TARGET ? 6 : 18];
            for (int i = 0; i < auxiliaryData.length; i++) auxiliaryData[i] = guard == Guard.BOOL ? i % 2
                    : guard == Guard.LOSS_TARGET ? i % 3 : i % 3;
            // Bad inactive rows/steps are deliberately retained during every success and retry.
            if (guard != Guard.SOFTMAX) {
                poisonAuxiliary(guard, auxiliaryData, 2, 99);
                poisonAuxiliary(guard, auxiliaryData, 4, 99);
                poisonAuxiliary(guard, auxiliaryData, 5, 99);
            }
            double[] finalSeeds = new double[9], emissionSeeds = new double[18];
            for (int i = 0; i < finalSeeds.length; i++) finalSeeds[i] = .25 + i * .03125;
            Arrays.fill(emissionSeeds, Double.NaN);
            for (int row : new int[]{0, 1, 3}) for (int j = 0; j < 3; j++) emissionSeeds[row * 3 + j] = .125 + j * .0625;
            var input = feeds(initial, new double[9], sequence, validData, auxiliary, auxiliaryData,
                    lengths, new double[]{2, 1, 0}, fs, finalSeeds, es, emissionSeeds);
            try (var run = new Run(artifact, input)) {
                // Row 1 is the last eligible row at t=0. Row 0 must not publish first.
                double[] invalidData = validData.clone(), invalidAuxiliary = auxiliaryData.clone();
                if (guard == Guard.SOFTMAX) invalidData[5] = Double.POSITIVE_INFINITY;
                else poisonAuxiliary(guard, invalidAuxiliary, 1, guard == Guard.SCATTER_DUPLICATE ? 0 : 99);
                run.put(sequence, invalidData);
                if (guard != Guard.SOFTMAX) run.put(auxiliary, invalidAuxiliary);
                ValueId emission = artifact.publication().forwardBindings().get(1).valueId();
                MemorySegment output = run.fixture.storage.get(emission);
                output.fill((byte)0x5a);
                byte[] before = output.asSlice(0, 6L * type.byteWidth()).toArray(ValueLayout.JAVA_BYTE);
                Class<? extends RuntimeException> expectedFailure = switch (guard) {
                    case LOSS_TARGET, GATHER, GATHER_ND, SCATTER_BOUNDS -> IndexOutOfBoundsException.class;
                    default -> IllegalArgumentException.class;
                };
                assertThrows(expectedFailure, run.bound::execute, guard.toString());
                assertArrayEquals(before, output.asSlice(0, 6L * type.byteWidth()).toArray(ValueLayout.JAVA_BYTE),
                        "the complete active domain validates before even row 0 writes: " + guard);
                run.put(sequence, validData);
                if (guard != Guard.SOFTMAX) run.put(auxiliary, auxiliaryData);
                run.bound.execute();
                var first = run.publications();
                run.bound.execute();
                compare(first, run.publications(), 0, guard + " reusable successful retry");
                // Independently unroll only the ordinary active prefixes, including real ordinary AD.
                var referenceInput = referenceFeeds(input);
                var referenceRoots = guardedReference(guard, initial, sequence, auxiliary, referenceInput);
                var expected = execute(compile(referenceRoots, referenceRoots, List.of(fs, es), List.of(initial, sequence), false), referenceInput);
                compare(expected, first, type == DataType.FLOAT64 ? 3e-13 : 2e-5, guard.toString());
            }
        }
    }

    private enum Family { NORMALIZED, ATTENTION, WINDOW, INDEX }

    @Test void ordinaryNormalizedAttentionOverlappingWindowsAndIndexConsumersMatchActivePrefixAutograd() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (Family family : Family.values())
            for (boolean reverse : List.of(false, true)) {
                Tensor h = leaf(type, true, 3, 2, 3), x = leaf(type, true, 3, 2, 3), w = leaf(type, true, 3);
                Tensor aux = leaf(family == Family.ATTENTION ? DataType.BOOL : DataType.INT64, false,
                        family == Family.ATTENTION ? new long[]{3, 2, 2} : new long[]{3, 2, 3});
                Tensor next = transformed(family, h.add(x).mul(w), aux);
                var formals = family == Family.ATTENTION || family == Family.INDEX ? List.of(x, aux) : List.of(x);
                Tensor initial = leaf(type, true, 3, 2, 3), sequence = leaf(type, true, 3, 3, 2, 3), weight = leaf(type, true, 3);
                Tensor auxiliary = leaf(aux.descriptor().dataType(), false,
                        family == Family.ATTENTION ? new long[]{3, 3, 2, 2} : new long[]{3, 3, 2, 3});
                var actuals = formals.size() == 2 ? List.of(sequence, auxiliary) : List.of(sequence);
                Tensor lengths = leaf(DataType.INT64, false, 3);
                var scan = bind(initial, actuals, List.of(weight), lengths,
                        new ScanBody(List.of(h), formals, List.of(w), List.of(next), List.of(next)), reverse, true);
                var roots = List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst());
                Tensor fs = leaf(type, false, 3, 2, 3), es = leaf(type, false, 3, 3, 2, 3);
                double[] data = new double[54], init = new double[18], finalSeeds = new double[18], seeds = new double[54];
                Arrays.fill(data, Double.NaN); Arrays.fill(seeds, Double.NaN);
                for (int i = 0; i < init.length; i++) { init[i] = (i % 7 - 3) * .03125; finalSeeds[i] = (i % 5 + 1) * .125; }
                long[] valid = {3, 1, 0};
                for (int t = 0; t < 3; t++) for (int b = 0; b < 3; b++) if (t < valid[b]) for (int j = 0; j < 6; j++) {
                    int index = (t * 3 + b) * 6 + j;
                    data[index] = ((index * 7) % 17 - 8) * .015625;
                    seeds[index] = (j % 4 - 1) * .125;
                }
                double[] auxiliaries = new double[family == Family.ATTENTION ? 36 : 54];
                for (int t = 0; t < 3; t++) for (int b = 0; b < 3; b++) {
                    int cells = family == Family.ATTENTION ? 4 : 6;
                    for (int j = 0; j < cells; j++) auxiliaries[(t * 3 + b) * cells + j] = t >= valid[b] ? 99
                            : family == Family.ATTENTION ? (j == 1 ? 0 : 1) : new int[]{2, 0, 2, 1, 1, 0}[j];
                }
                var input = feeds(initial, init, sequence, data, weight, new double[]{.75, 1.125, .875},
                        auxiliary, auxiliaries, lengths, new double[]{3, 1, 0}, fs, finalSeeds, es, seeds);
                var reference = familyReference(family, initial, sequence, weight, auxiliary, reverse, input);
                var targets = List.of(initial, sequence, weight);
                var expected = execute(compile(reference, reference, List.of(fs, es), targets, false), referenceFeeds(input));
                var actual = execute(compile(roots, roots, List.of(fs, es), targets, false), input);
                compare(expected, actual, type == DataType.FLOAT64 ? 2e-11 : 4e-4, family + " reverse=" + reverse);
                for (int t = 0; t < 3; t++) for (int b = 0; b < 3; b++) if (t >= valid[b]) for (int j = 0; j < 6; j++) {
                    int index = (t * 3 + b) * 6 + j;
                    rawZero(0d, actual.get(1)[index]);
                    rawZero(0d, actual.get(3)[index]);
                }
            }
    }

    @Test void conv3dPool3dLossExecutesActualInputAndWeightCotangents() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean maximum : List.of(false, true)) {
                Tensor input = leaf(type, true, 1,1,4,4,4), weight = leaf(type, true, 1,1,2,2,2);
                Tensor convolution = input.conv3d(weight,
                        io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs.defaults());
                Tensor pooled = maximum
                        ? convolution.maxPool3d(new io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs(
                                2,2,2,1,1,1,0,0,0,1,1,1,false))
                        : convolution.averagePool3d(new io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs(
                                2,2,2,1,1,1,0,0,0,1,1,1,false));
                Tensor target = leaf(type, false, 1,1,2,2,2), seed = leaf(type, false);
                Tensor loss = pooled.meanSquaredError(target, LossReduction.MEAN);
                double[] inputs = new double[64], weights = new double[8];
                Arrays.fill(inputs, 1); Arrays.fill(weights, .125);
                var actual = execute(compile(List.of(loss), List.of(loss), List.of(seed),
                        List.of(input,weight), false),
                        feeds(input,inputs,weight,weights,target,new double[8],seed,new double[]{1}));
                assertArrayEquals(new double[]{1},actual.getFirst());
                // Ordinary overlap multiplicities: convolution composes two adjacent windows.
                int[] counts = maximum ? new int[]{1,2,1,0} : new int[]{1,3,3,1};
                double scale = maximum ? .03125 : .00390625;
                double[] expectedInput = new double[64], expectedWeight = new double[8];
                for (int d = 0; d < 4; d++) for (int h = 0; h < 4; h++) for (int w = 0; w < 4; w++)
                    expectedInput[(d * 4 + h) * 4 + w] = counts[d] * counts[h] * counts[w] * scale;
                Arrays.fill(expectedWeight, 2);
                assertArrayEquals(expectedInput,actual.get(1));
                assertArrayEquals(expectedWeight,actual.get(2));
            }
    }

    @Test void ordinaryPool3dSelectedBackwardAccumulatesOverlapsAndPreservesExactWinnerEligibility() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            Tensor input = leaf(type, true, 1,1,3,3,3);
            Tensor average = input.averagePool3d(new io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs(
                    2,2,2,1,1,1,0,0,0,1,1,1,false));
            Tensor seed = leaf(type, false, 1,1,2,2,2);
            double[] ones = new double[27], seeds = new double[8]; Arrays.fill(ones, 1); Arrays.fill(seeds, 1);
            var averaged = execute(compile(List.of(average), List.of(average), List.of(seed),
                    List.of(input), false), feeds(input, ones, seed, seeds));
            assertArrayEquals(seeds, averaged.getFirst());
            double[] expected = new double[27];
            for (int d = 0; d < 3; d++) for (int h = 0; h < 3; h++) for (int w = 0; w < 3; w++)
                expected[(d * 3 + h) * 3 + w] = (d == 1 ? 2 : 1) * (h == 1 ? 2 : 1) * (w == 1 ? 2 : 1) / 8d;
            assertArrayEquals(expected, averaged.get(1));

            Tensor special = leaf(type, true, 1,1,1,1,6);
            Tensor max = special.maxPool3d(new io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs(
                    1,1,3,1,1,1,0,0,3,1,1,1,false));
            Tensor maxSeed = leaf(type, false, 1,1,1,1,10);
            var selected = execute(compile(List.of(max), List.of(max), List.of(maxSeed), List.of(special), false),
                    feeds(special, new double[]{Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, -0d, 0d, Double.NaN, Double.NaN},
                            maxSeed, new double[]{512,1,2,4,8,16,32,64,128,1024}));
            double[] forward = selected.getFirst();
            assertArrayEquals(new double[]{Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY,
                    Double.NEGATIVE_INFINITY, -0d, 0d, Double.NaN, Double.NaN, Double.NaN,
                    Double.NaN, Double.NEGATIVE_INFINITY}, forward);
            rawZero(-0d, forward[3]); rawZero(0d, forward[4]);
            assertArrayEquals(new double[]{3,0,4,8,112,128}, selected.get(1));
        }
    }

    @Test void lengthAwarePool3dBackwardIgnoresPoisonedInactiveRowsAndSeeds() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean maximum : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
                Tensor h = leaf(type, true, 3,1,3,3,3), x = leaf(type, true, 3,1,3,3,3);
                Tensor pooled = pool3dSameShape(h.add(x), maximum);
                Tensor initial = leaf(type, true, 3,1,3,3,3), sequence = leaf(type, true, 2,3,1,3,3,3);
                Tensor lengths = leaf(DataType.INT64, false, 3);
                var scan = bind(initial, List.of(sequence), List.of(), lengths,
                        new ScanBody(List.of(h), List.of(x), List.of(), List.of(pooled), List.of(pooled)), reverse, true);
                var roots = List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst());
                Tensor fs = leaf(type, false, 3,1,3,3,3), es = leaf(type, false, 2,3,1,3,3,3);
                double[] init = new double[81], data = new double[162], seeds = new double[162], finalSeeds = new double[81];
                Arrays.fill(data, Double.NaN); Arrays.fill(seeds, Double.NaN); Arrays.fill(finalSeeds, .125);
                int[] valid = {2,1,0};
                for (int i = 0; i < init.length; i++) init[i] = (i % 13 - 6) * .0625;
                for (int t = 0; t < 2; t++) for (int row = 0; row < 3; row++) if (t < valid[row])
                    for (int j = 0; j < 27; j++) {
                        int i = (t * 3 + row) * 27 + j;
                        data[i] = (i % 17 - 8) * .03125; seeds[i] = (j % 5 + 1) * .125;
                    }
                Tensor zero = leaf(type, false, 1,1,3,3,3);
                var input = feeds(initial, init, sequence, data, lengths, new double[]{2,1,0},
                        fs, finalSeeds, es, seeds, zero, new double[27]);
                Tensor[] finals = new Tensor[3]; Tensor[][] history = new Tensor[2][3];
                for (int row = 0; row < 3; row++) {
                    Tensor state = initial.select(0,row).expandDims(0);
                    for (int t = 0; t < 2; t++) history[t][row] = zero;
                    for (int step = 0; step < valid[row]; step++) {
                        int t = reverse ? valid[row] - 1 - step : step;
                        state = pool3dSameShape(state.add(sequence.select(0,t).select(0,row).expandDims(0)), maximum);
                        history[t][row] = state;
                    }
                    finals[row] = state;
                }
                var reference = List.of(Tensor.concat(0,finals), Tensor.stack(0,
                        Tensor.concat(0,history[0]), Tensor.concat(0,history[1])));
                var targets = List.of(initial,sequence);
                var expected = execute(compile(reference,reference,List.of(fs,es),targets,false),referenceFeeds(input));
                var actual = execute(compile(roots,roots,List.of(fs,es),targets,false),input);
                compare(expected,actual,type == DataType.FLOAT64 ? 2e-13 : 3e-6,
                        "Pool3d maximum=" + maximum + " reverse=" + reverse);
                for (int t = 0; t < 2; t++) for (int row = 0; row < 3; row++) if (t >= valid[row])
                    for (int j = 0; j < 27; j++) {
                        rawZero(0d,actual.get(1)[(t * 3 + row) * 27 + j]);
                        rawZero(0d,actual.get(3)[(t * 3 + row) * 27 + j]);
                    }
            }
    }

    /**
     * Builds overlapping dilated, padded ceiling-grid pooling without changing the state shape.
     * @param input floating NCDHW state
     * @param maximum whether to select the exact maximum rather than the fixed-divisor mean
     * @return the same-shaped ordinary Pool3d expression used by Scan and its unrolled reference
     */
    private static Tensor pool3dSameShape(Tensor input, boolean maximum) {
        return maximum ? input.maxPool3d(new io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs(
                2,2,2,1,1,1,1,1,1,2,2,2,true))
                : input.averagePool3d(new io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs(
                        2,2,2,1,1,1,1,1,1,2,2,2,true));
    }

    private static Tensor guarded(Guard guard, Tensor x, Tensor aux) {
        return switch (guard) {
            case SOFTMAX -> x.softmax(1);
            case LOSS_TARGET -> x.categoricalCrossEntropyWithLogits(aux, 1, LossReduction.NONE)
                    .expandDims(1).expand(x.descriptor().shape());
            case BOOL -> Tensor.where(aux, x, x.neg());
            case GATHER -> x.gatherElements(aux, 1);
            case GATHER_ND -> x.gatherNd(aux, 1);
            case SCATTER_BOUNDS, SCATTER_DUPLICATE -> x.scatterElements(aux, x, 1);
        };
    }

    private static void poisonAuxiliary(Guard guard, double[] data, int row, double value) {
        if (guard == Guard.LOSS_TARGET) data[row] = value;
        else if (guard == Guard.SCATTER_DUPLICATE) {
            data[row * 3] = value; data[row * 3 + 1] = value;
        } else data[row * 3 + 2] = value;
    }

    /**
     * Unrolls ordinary active prefixes with an explicit caller-fed padding value.
     * @param guard actual ordinary guarded family
     * @param initial original initial-state leaf
     * @param sequence original sequence leaf
     * @param auxiliary original typed auxiliary leaf
     * @param feeds mutable reference-only feeds receiving the genuine padding input
     * @return ordered final-state and original-time-aligned emission expressions
     */
    private static List<Tensor> guardedReference(Guard guard, Tensor initial, Tensor sequence,
            Tensor auxiliary, Map<Tensor, double[]> feeds) {
        Tensor zero = leaf(initial.descriptor().dataType(), false, 1, 3);
        feeds.put(zero, new double[3]);
        Tensor[] finals = new Tensor[3]; Tensor[][] history = new Tensor[2][3];
        for (int row = 0; row < 3; row++) {
            Tensor state = initial.select(0, row).expandDims(0);
            for (int t = 0; t < 2; t++) {
                if (t < new int[]{2, 1, 0}[row]) {
                    Tensor x = sequence.select(0, t).select(0, row).expandDims(0);
                    Tensor aux = auxiliary.select(0, t).select(0, row).expandDims(0);
                    history[t][row] = guarded(guard, x, aux);
                    state = state.add(history[t][row]);
                } else history[t][row] = zero;
            }
            finals[row] = state;
        }
        return List.of(Tensor.concat(0, finals), Tensor.stack(0, Tensor.concat(0, history[0]), Tensor.concat(0, history[1])));
    }

    private static Tensor transformed(Family family, Tensor input, Tensor auxiliary) {
        return switch (family) {
            case NORMALIZED -> input.layerNorm(Shape.of(3), input.descriptor().dataType() == DataType.FLOAT64
                    ? ScalarValue.float64(.03125) : ScalarValue.float32(.03125f));
            case ATTENTION -> input.scaledDotProductAttentionWithWeights(input, input, auxiliary).output();
            case WINDOW -> input.unfold(2, 2, 1).foldAxis(2, 3, 1);
            case INDEX -> input.gatherElements(auxiliary, 2);
        };
    }

    private static List<Tensor> familyReference(Family family, Tensor initial, Tensor sequence, Tensor weight,
            Tensor auxiliary, boolean reverse, Map<Tensor, double[]> feeds) {
        Tensor zero = leaf(initial.descriptor().dataType(), false, 1, 2, 3); feeds.put(zero, new double[6]);
        Tensor[] finals = new Tensor[3]; Tensor[][] history = new Tensor[3][3];
        for (int row = 0; row < 3; row++) {
            int count = new int[]{3, 1, 0}[row];
            Tensor state = initial.select(0, row).expandDims(0);
            for (int t = 0; t < 3; t++) history[t][row] = zero;
            for (int step = 0; step < count; step++) {
                int t = reverse ? count - 1 - step : step;
                Tensor x = sequence.select(0, t).select(0, row).expandDims(0);
                Tensor aux = auxiliary.select(0, t).select(0, row).expandDims(0);
                state = transformed(family, state.add(x).mul(weight), aux);
                history[t][row] = state;
            }
            finals[row] = state;
        }
        return List.of(Tensor.concat(0, finals), Tensor.stack(0, Tensor.concat(0, history[0]),
                Tensor.concat(0, history[1]), Tensor.concat(0, history[2])));
    }

    private static ScanResult bind(Tensor initial, List<Tensor> sequences, List<Tensor> invariants,
            Tensor lengths, ScanBody body, boolean reverse, boolean length) {
        return length ? Scan.apply(List.of(initial), sequences, invariants, lengths, 1,
                sequences.stream().map(ignored -> OptionalInt.of(0)).toList(), body, reverse)
                : Scan.apply(List.of(initial), sequences, invariants, body, reverse);
    }

    private static Tensor leaf(DataType type, boolean gradient, long... dimensions) {
        Shape shape = Shape.of(dimensions);
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), gradient));
    }

    private static CompileArtifacts compile(List<Tensor> roots, List<Tensor> selected, List<Tensor> seeds,
            List<Tensor> targets, boolean optimized) {
        var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(selected.stream()
                .map(t -> (FunctionalGradientRequest.OutputReference)new FunctionalGradientRequest.ForwardTensorReference(t)).toList(),
                seeds.stream().map(Optional::of).toList(), targets, false, FunctionalGradientRequest.DisconnectedPolicy.ZERO)));
        return CpuGeneralScanCapabilityTest.compile(roots, Optional.of(request), optimized, new CpuCapabilityProvider());
    }

    private static Map<Tensor, double[]> feeds(Object... pairs) {
        var result = new IdentityHashMap<Tensor, double[]>();
        for (int i = 0; i < pairs.length; i += 2) result.put((Tensor)pairs[i], (double[])pairs[i + 1]);
        return result;
    }

    /**
     * Gives the ordinary reference canonical unused backing cells. Only the Scan under test
     * receives poison: ordinary input materialization is not a length-aware consumer.
     * @param input active values plus deliberately poisoned inactive storage and seeds
     * @return independent reference feeds retaining every active value unchanged
     */
    private static Map<Tensor, double[]> referenceFeeds(Map<Tensor, double[]> input) {
        var result = new IdentityHashMap<Tensor, double[]>();
        input.forEach((tensor, values) -> {
            double[] clean = values.clone();
            for (int i = 0; i < clean.length; i++) {
                if (!Double.isFinite(clean[i])
                        || tensor.descriptor().dataType() == DataType.BOOL && clean[i] != 0 && clean[i] != 1
                        || tensor.descriptor().dataType() == DataType.INT64 && clean[i] == 99) clean[i] = 0;
            }
            result.put(tensor, clean);
        });
        return result;
    }

    private static List<double[]> execute(CompileArtifacts artifact, Map<Tensor, double[]> feeds) {
        try (var run = new Run(artifact, feeds)) { run.bound.execute(); return run.publications(); }
    }

    private static void rawZero(double expected, double actual) {
        assertEquals(Double.doubleToRawLongBits(expected), Double.doubleToRawLongBits(actual));
    }

    @Test void inactiveSavedHistoryAndEmissionSeedsAreNeverConsumedByTheActualSelectedBackward() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (boolean reverse : List.of(false, true)) {
            Tensor h = leaf(type, true, 3, 2), x = leaf(type, true, 3, 2);
            Tensor next = h.add(x).tanh();
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(next), List.of(next));
            Tensor initial = leaf(type, true, 3, 2), sequence = leaf(type, true, 3, 3, 2);
            Tensor lengths = leaf(DataType.INT64, false, 3), fs = leaf(type, false, 3, 2), es = leaf(type, false, 3, 3, 2);
            var scan = bind(initial, List.of(sequence), List.of(), lengths, body, reverse, true);
            var selected = List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst());
            var roots = new ArrayList<>(selected); roots.addAll(scan.residuals());
            double[] data = new double[18], seeds = new double[18]; Arrays.fill(data, Double.NaN); Arrays.fill(seeds, Double.NaN);
            int[] valid = {3, 1, 0};
            for (int t = 0; t < 3; t++) for (int row = 0; row < 3; row++) if (t < valid[row]) for (int j = 0; j < 2; j++) {
                int cell = (t * 3 + row) * 2 + j; data[cell] = (cell + 1) * .015625; seeds[cell] = .25 + cell * .03125;
            }
            Tensor zero = leaf(type, false, 1, 2);
            var input = feeds(initial, new double[]{.1, -.2, .3, -.1, -.5, .25}, sequence, data,
                    lengths, new double[]{3, 1, 0}, fs, new double[]{.5, .25, .75, -.5, 1, 2}, es, seeds, zero, new double[2]);
            Tensor[] finalRows = new Tensor[3]; Tensor[][] history = new Tensor[3][3];
            for (int row = 0; row < 3; row++) {
                Tensor state = initial.select(0, row).expandDims(0);
                for (int t = 0; t < 3; t++) history[t][row] = zero;
                for (int step = 0; step < valid[row]; step++) {
                    int t = reverse ? valid[row] - 1 - step : step;
                    state = state.add(sequence.select(0, t).select(0, row).expandDims(0)).tanh();
                    history[t][row] = state;
                }
                finalRows[row] = state;
            }
            var ordinary = List.of(Tensor.concat(0, finalRows), Tensor.stack(0, Tensor.concat(0, history[0]),
                    Tensor.concat(0, history[1]), Tensor.concat(0, history[2])));
            var targets = List.of(initial, sequence);
            var expected = execute(compile(ordinary, ordinary, List.of(fs, es), targets, false), referenceFeeds(input));
            var artifact = compile(roots, selected, List.of(fs, es), targets, false);
            try (var run = new Run(artifact, input)) {
                var children = ((CpuPreparedPartitionExecutable)run.fixture.executable).children().stream()
                        .map(child -> child.bind(run.fixture.state)).toList();
                int reverseUnit = -1;
                for (int i = 0; i < run.units.size(); i++)
                    if (run.units.get(i).runtimeFacts().generalScan().isPresent()) reverseUnit = i;
                for (int i = 0; i < reverseUnit; i++) children.get(i).execute();
                // Mutate actual forward-produced saved publications between the real forward
                // and reverse executable boundaries, rather than recomputing a substitute VJP.
                for (int port = 2; port < roots.size(); port++) {
                    MemorySegment saved = run.fixture.storage.get(artifact.publication().forwardBindings().get(port).valueId());
                    for (int t = 0; t < 3; t++) for (int row = 0; row < 3; row++) if (t >= valid[row])
                        for (int j = 0; j < 2; j++) Run.set(saved, type, (t * 3L + row) * 2 + j,
                                j == 0 ? Double.NaN : Double.NEGATIVE_INFINITY);
                }
                for (int i = reverseUnit; i < children.size(); i++) children.get(i).execute();
                var actual = run.publications();
                compare(expected, List.of(actual.get(0), actual.get(1), actual.get(roots.size()), actual.get(roots.size() + 1)),
                        type == DataType.FLOAT64 ? 3e-13 : 3e-6, "inactive saved primals reverse=" + reverse);
                for (int t = 0; t < 3; t++) for (int row = 0; row < 3; row++) if (t >= valid[row])
                    for (int j = 0; j < 2; j++) rawZero(0d, actual.get(roots.size() + 1)[(t * 3 + row) * 2 + j]);
            }
        }
    }

    private static void compare(List<double[]> expected, List<double[]> actual, double tolerance, String label) {
        assertEquals(expected.size(), actual.size());
        for (int p = 0; p < expected.size(); p++) assertArrayEquals(expected.get(p), actual.get(p), tolerance, label + " publication=" + p);
    }

    /** Owns the existing real assigned-slot fixture; no numerical execution is implemented here. */
    private static final class Run implements AutoCloseable {
        private final CompileArtifacts artifact;
        private final CpuRecurrentScanExecutionTest.Fixture fixture;
        private final List<io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionUnitPlan> units;
        private final io.github.pho001.synaptik.runtime.execution.BoundInvocation bound;

        private Run(CompileArtifacts artifact, Map<Tensor, double[]> feeds) {
            this.artifact = artifact;
            var context = GraphPreparation.project(artifact, artifact.partitions().getFirst(), CpuPartitionAnalysisInputs.DEFAULT);
            var analysis = new CpuPartitionPreparer().analyze(context);
            units = analysis.plan().units();
            fixture = new CpuRecurrentScanExecutionTest.Fixture(
                    artifact, CpuPartitionAnalysisInputs.DEFAULT, analysis, 1, false, false, Map.of());
            for (var input : artifact.constants().bindableInputBindings()) {
                Tensor tensor = feeds.keySet().stream().filter(t -> t.id().equals(input.tensorId())).findFirst().orElseThrow();
                put(tensor, feeds.get(tensor));
            }
            bound = fixture.executable.bind(fixture.state);
        }

        private void put(Tensor tensor, double[] data) {
            var binding = artifact.constants().bindableInputBindings().stream()
                    .filter(b -> b.tensorId().equals(tensor.id())).findFirst();
            if (binding.isEmpty()) return; // A supplied unselected seed need not be captured.
            MemorySegment storage = fixture.storage.get(binding.orElseThrow().valueId());
            for (int i = 0; i < data.length; i++) set(storage, tensor.descriptor().dataType(), i, data[i]);
        }

        private List<double[]> publications() {
            var values = new ArrayList<ValueId>();
            artifact.publication().forwardBindings().forEach(b -> values.add(b.valueId()));
            artifact.publication().gradientBindings().forEach(b -> values.add(b.valueId()));
            return values.stream().map(id -> {
                var descriptor = artifact.graph().values().stream().filter(v -> v.id().equals(id)).findFirst().orElseThrow().descriptor();
                var layout = descriptor.layout().orElseGet(() -> LayoutDescriptor.contiguous(descriptor.shape()));
                long[] shape = descriptor.shape().toLongArray(), strides = layout.strides();
                double[] result = new double[Math.toIntExact(descriptor.shape().knownElementCount().orElseThrow())];
                MemorySegment storage = fixture.storage.get(id);
                for (int i = 0; i < result.length; i++) {
                    long address = layout.storageOffset(), remainder = i;
                    for (int axis = shape.length - 1; axis >= 0; axis--) { address += (remainder % shape[axis]) * strides[axis]; remainder /= shape[axis]; }
                    result[i] = descriptor.dataType() == DataType.FLOAT64 ? storage.getAtIndex(ValueLayout.JAVA_DOUBLE, address)
                            : storage.getAtIndex(ValueLayout.JAVA_FLOAT, address);
                }
                return result;
            }).toList();
        }

        private static void set(MemorySegment storage, DataType type, long index, double value) {
            switch (type) {
                case FLOAT64 -> storage.setAtIndex(ValueLayout.JAVA_DOUBLE, index, value);
                case FLOAT32 -> storage.setAtIndex(ValueLayout.JAVA_FLOAT, index, (float)value);
                case INT64 -> storage.setAtIndex(ValueLayout.JAVA_LONG, index, (long)value);
                case INT32 -> storage.setAtIndex(ValueLayout.JAVA_INT, index, (int)value);
                case BOOL -> storage.setAtIndex(ValueLayout.JAVA_BYTE, index, (byte)value);
                default -> throw new AssertionError("unexpected low carrier in F32/F64 consumer");
            }
        }

        @Override public void close() { fixture.close(); }
    }
}
