package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.CpuGeneralScanCapabilityTest;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.FunctionalGradientRequest;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Numerical counterexamples for temporal validation, real control views and versioned staging. */
class CpuGeneralScanTemporalRegressionTest {
    @Test void sharedSumForwardAndSelectedBackwardRemainNumericalWithALivePreparedForwardOwner() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean lengthMode : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
                Tensor clock = value(DataType.BOOL, false, Shape.of(2));
                Tensor matrix = value(type, true, Shape.of(2, 3));
                Tensor clocks = value(DataType.BOOL, false, Shape.of(3, 2));
                Tensor weights = value(type, true, Shape.of(2, 3));
                Tensor lengths = value(DataType.INT64, false, Shape.of(2));
                Tensor emission = clock.cast(type).reshape(2, 1).add(matrix.sum(1));
                var body = new ScanBody(List.of(), List.of(clock), List.of(matrix), List.of(),
                        List.of(emission));
                var scan = lengthMode
                        ? Scan.apply(List.of(), List.of(clocks), List.of(weights), lengths, 0,
                                List.of(OptionalInt.of(0)), body, reverse)
                        : Scan.apply(List.of(), List.of(clocks), List.of(weights), body, reverse);
                var roots = scan.outputs();
                var forward = compile(roots, Optional.empty());
                Tensor seed = value(type, false, Shape.of(3, 2, 2));
                var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
                        List.of(new FunctionalGradientRequest.ForwardTensorReference(roots.getFirst())),
                        List.of(Optional.of(seed)), List.of(weights), false,
                        FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
                try (var primal = new Consumer(forward, OptionalLong.empty())) {
                    primal.numbers(clocks, 0, 1, 1, 0, 0, 1);
                    primal.numbers(weights, 1, 2, 4, -2, .5, 1.5);
                    if (lengthMode) primal.numbers(lengths, 2, 0);
                    primal.bind(); primal.bound.execute();
                    assertArrayEquals(lengthMode ? new double[] {7, 0, 0, 0, 8, 1, 0, 0, 0, 0, 0, 0}
                            : new double[] {7, 0, 8, 1, 8, 1, 7, 0, 7, 0, 8, 1}, primal.forward(0));

                    // Keep the first prepared artifact live while independently lowering the
                    // same primal inside actual AD. No cache internals or IR assertions substitute
                    // for constructing and executing that second consumer.
                    var differentiated = compile(roots, Optional.of(request));
                    try (var backward = new Consumer(differentiated, OptionalLong.empty())) {
                        for (int pass = 0; pass < 2; pass++) {
                            double[] input = pass == 0 ? new double[] {1, 2, 4, -2, .5, 1.5}
                                    : new double[] {-.5, 1, 3, 4, -2, .25};
                            double[] clockValues = pass == 0 ? new double[] {0, 1, 1, 0, 0, 1}
                                    : new double[] {1, 0, 0, 1, 1, 0};
                            double[] ls = pass == 0 ? new double[] {2, 0} : new double[] {3, 1};
                            double[] seeds = pass == 0 ? new double[] {.5, 2, 1, -.5, -1, .25, 2, 1, 3, -2, -.5, 4}
                                    : new double[] {.25, -.5, .5, -1, 1, -2, 2, -4, 4, -8, 8, -16};
                            if (lengthMode) for (int t = 0; t < 3; t++) for (int b = 0; b < 2; b++)
                                if (t >= ls[b]) Arrays.fill(seeds, (t * 2 + b) * 2, (t * 2 + b + 1) * 2, Double.NaN);
                            double[] expected = pass == 0
                                    ? lengthMode ? new double[] {7, 0, 0, 0, 8, 1, 0, 0, 0, 0, 0, 0}
                                            : new double[] {7, 0, 8, 1, 8, 1, 7, 0, 7, 0, 8, 1}
                                    : lengthMode ? new double[] {4.5, 3.25, 3.5, 2.25, 3.5, 2.25, 0, 0, 4.5, 3.25, 0, 0}
                                            : new double[] {4.5, 3.25, 3.5, 2.25, 3.5, 2.25, 4.5, 3.25, 4.5, 3.25, 3.5, 2.25};
                            double[] gradient = pass == 0
                                    ? lengthMode ? new double[] {-.5, -.5, -.5, 2.25, 2.25, 2.25}
                                            : new double[] {5, 5, 5, 4.75, 4.75, 4.75}
                                    : lengthMode ? new double[] {5.75, 5.75, 5.75, -11.5, -11.5, -11.5}
                                            : new double[] {15.75, 15.75, 15.75, -31.5, -31.5, -31.5};
                            primal.numbers(clocks, clockValues); backward.numbers(clocks, clockValues);
                            primal.numbers(weights, input); backward.numbers(weights, input);
                            backward.numbers(seed, seeds);
                            if (lengthMode) {
                                primal.numbers(lengths, ls);
                                backward.numbers(lengths, ls);
                            }
                            if (pass == 0) backward.bind();
                            primal.bound.execute(); backward.bound.execute();
                            assertArrayEquals(expected, primal.forward(0));
                            assertArrayEquals(expected, backward.forward(0));
                            assertArrayEquals(gradient, backward.gradient(0));
                        }
                    }
                }
            }
    }

    @Test void aBadLastSoftmaxRowLeavesTheProducingEmissionUntouchedUntilCompleteValidationAndRetry() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean reverse : List.of(false, true)) {
                Tensor x = value(type, false, Shape.of(2, 3));
                Tensor sequence = value(type, false, Shape.of(1, 2, 3));
                Tensor lengths = value(DataType.INT64, false, Shape.of(2));
                var body = new ScanBody(List.of(), List.of(x), List.of(), List.of(), List.of(x.softmax(1)));
                var scan = Scan.apply(List.of(), List.of(sequence), List.of(), lengths, 0,
                        List.of(OptionalInt.of(0)), body, reverse);
                var artifact = compile(scan.outputs(), Optional.empty());
                try (var run = new Consumer(artifact, OptionalLong.empty())) {
                    run.numbers(sequence, 0, 0, 0, 0, 0, Double.POSITIVE_INFINITY);
                    run.numbers(lengths, 1, 1);
                    run.bind();
                    run.outputStorage(0).fill((byte) 0x5a);
                    byte[] before = run.outputStorage(0).toArray(ValueLayout.JAVA_BYTE);
                    assertThrows(IllegalArgumentException.class, run.bound::execute);
                    assertArrayEquals(before, run.outputStorage(0).toArray(ValueLayout.JAVA_BYTE));
                    run.numbers(sequence, 0, 0, 0, 0, 0, 0);
                    run.bound.execute();
                    double third = type == DataType.FLOAT32 ? (float) (1d / 3d) : 1d / 3d;
                    assertArrayEquals(new double[] {third, third, third, third, third, third}, run.forward(0));
                    run.numbers(lengths, 1, 0);
                    run.numbers(sequence, 0, 0, 0, Double.NaN, Double.NaN, Double.NaN);
                    run.bound.execute();
                    assertArrayEquals(new double[] {third, third, third, 0, 0, 0}, run.forward(0));
                    for (int i = 3; i < 6; i++) assertEquals(0L, Double.doubleToRawLongBits(run.forward(0)[i]));
                }
            }
    }

    @Test void logicalLengthsUseOffsetOneStrideTwoAndIgnorePoisonedBackingHoles() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean reverse : List.of(false, true)) {
                Tensor x = value(type, false, Shape.of(2, 1));
                Tensor sequence = value(type, false, Shape.of(1, 2, 1));
                Tensor lengths = stridedLengths();
                var scan = Scan.apply(List.of(), List.of(sequence), List.of(), lengths, 0,
                        List.of(OptionalInt.of(0)), new ScanBody(List.of(), List.of(x), List.of(), List.of(), List.of(x)), reverse);
                var artifact = compile(scan.outputs(), Optional.empty());
                try (var run = new Consumer(artifact, OptionalLong.empty())) {
                    run.numbers(sequence, 7, Double.NaN);
                    run.numbers(lengths, 0, 1, 1, 0); // Logical [1,0], not backing prefix [0,1].
                    run.bind();
                    run.bound.execute();
                    assertArrayEquals(new double[] {7, 0}, run.forward(0));
                    run.numbers(lengths, -99, 1, Long.MAX_VALUE, 0);
                    run.bound.execute();
                    assertArrayEquals(new double[] {7, 0}, run.forward(0));
                    run.numbers(lengths, 0, 0, 0, 1);
                    run.numbers(sequence, Double.NaN, -11);
                    run.bound.execute();
                    assertArrayEquals(new double[] {0, -11}, run.forward(0));
                    for (long bad : new long[] {-1, 2, Long.MAX_VALUE}) {
                        run.numbers(lengths, 0, 0, 0, bad);
                        run.outputStorage(0).fill((byte) 0x5a);
                        byte[] before = run.outputStorage(0).toArray(ValueLayout.JAVA_BYTE);
                        assertThrows(IllegalArgumentException.class, run.bound::execute);
                        assertArrayEquals(before, run.outputStorage(0).toArray(ValueLayout.JAVA_BYTE));
                    }
                    run.numbers(lengths, -99, 1, -99, 0);
                    run.numbers(sequence, -3, Double.NaN);
                    run.bound.execute();
                    assertArrayEquals(new double[] {-3, 0}, run.forward(0));
                }
            }
    }

    @Test void zeroTimeStillValidatesTheStridedLogicalLengthsBeforeCarryOrWorkspaceWrites() {
        Tensor h = value(DataType.FLOAT64, false, Shape.of(2, 1));
        Tensor x = value(DataType.FLOAT64, false, Shape.of(2, 1));
        Tensor initial = value(DataType.FLOAT64, false, Shape.of(2, 1));
        Tensor sequence = value(DataType.FLOAT64, false, Shape.of(0, 2, 1));
        Tensor lengths = stridedLengths();
        var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(), lengths, 1,
                List.of(OptionalInt.of(0)), new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x)), List.of()), true);
        var artifact = compile(scan.finalCarry(), Optional.empty());
        try (var run = new Consumer(artifact, OptionalLong.empty())) {
            run.numbers(initial, -0d, 17);
            run.numbers(lengths, 0, 0, 0, 1);
            run.bind();
            var workspace = ((CpuContiguousWorkspace) run.fixture.state.workspaceRepresentation(0)).writableSegment();
            workspace.fill((byte) 0x3c);
            run.outputStorage(0).fill((byte) 0x5a);
            byte[] beforeOutput = run.outputStorage(0).toArray(ValueLayout.JAVA_BYTE);
            byte[] beforeWorkspace = workspace.toArray(ValueLayout.JAVA_BYTE);
            assertThrows(IllegalArgumentException.class, run.bound::execute);
            assertArrayEquals(beforeOutput, run.outputStorage(0).toArray(ValueLayout.JAVA_BYTE));
            assertArrayEquals(beforeWorkspace, workspace.toArray(ValueLayout.JAVA_BYTE));
            run.numbers(lengths, -99, 0, -99, 0);
            run.bound.execute();
            assertArrayEquals(new double[] {-0d, 17}, run.forward(0));
            assertEquals(Long.MIN_VALUE, Double.doubleToRawLongBits(run.forward(0)[0]));
        }
    }

    @Test void sharedInvariantStagingRemainsLazyAndResetsAcrossChangedActivityAndReentry() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (int rowAxis : new int[] {0, 1}) for (boolean reverse : List.of(false, true)) {
                Shape row = rowAxis == 0 ? Shape.of(2, 3) : Shape.of(3, 2);
                Shape shared = rowAxis == 0 ? Shape.of(3) : Shape.of(3, 1);
                Tensor x = value(type, false, row), condition = value(DataType.BOOL, false, shared);
                Tensor sequence = value(type, false, rowAxis == 0 ? Shape.of(3, 2, 3) : Shape.of(3, 3, 2));
                Tensor invariant = value(DataType.BOOL, false, shared);
                Tensor lengths = value(DataType.INT64, false, Shape.of(2));
                Tensor result = Tensor.where(condition, x, x.neg());
                if (rowAxis == 1) result = result.transpose();
                var body = new ScanBody(List.of(), List.of(x), List.of(condition), List.of(), List.of(result));
                var scan = Scan.apply(List.of(), List.of(sequence), List.of(invariant), lengths, 0,
                        List.of(OptionalInt.of(rowAxis)), body, reverse);
                var artifact = compile(scan.outputs(), Optional.empty());
                try (var run = new Consumer(artifact, OptionalLong.empty())) {
                    // A malformed BOOL would reject if any body value were evaluated while absent.
                    run.numbers(invariant, 99, 99, 99);
                    run.numbers(lengths, 0, 0);
                    double[] poison = new double[18]; Arrays.fill(poison, Double.NaN);
                    run.numbers(sequence, poison);
                    run.bind(); run.bound.execute();
                    assertArrayEquals(new double[18], run.forward(0));
                    for (int pass = 0; pass < 2; pass++) {
                        double[] input = new double[18], expected = new double[18];
                        long[] ls = pass == 0 ? new long[] {1, 3} : new long[] {3, 1};
                        double[] flags = pass == 0 ? new double[] {1, 0, 1} : new double[] {0, 1, 0};
                        for (int t = 0; t < 3; t++) for (int b = 0; b < 2; b++) for (int j = 0; j < 3; j++) {
                            int source = t * 6 + (rowAxis == 0 ? b * 3 + j : j * 2 + b);
                            int target = (t * 2 + b) * 3 + j;
                            input[source] = t < ls[b] ? source + 1 + pass : Double.NaN;
                            if (t < ls[b]) expected[target] = flags[j] == 1 ? input[source] : -input[source];
                        }
                        run.numbers(sequence, input); run.numbers(invariant, flags);
                        run.numbers(lengths, ls[0], ls[1]); run.bound.execute();
                        assertArrayEquals(expected, run.forward(0));
                    }
                }
            }
    }

    @Test void sharedSequencesRemainOriginalTimeVersionsInsteadOfBecomingInvocationInvariants() {
        for (boolean reverse : List.of(false, true)) {
            Tensor x = value(DataType.FLOAT64, false, Shape.of(2, 2));
            Tensor temporal = value(DataType.FLOAT64, false, Shape.of(2));
            Tensor w = value(DataType.FLOAT64, false, Shape.of(2));
            Tensor sequence = value(DataType.FLOAT64, false, Shape.of(3, 2, 2));
            Tensor changing = value(DataType.FLOAT64, false, Shape.of(3, 2));
            Tensor weight = value(DataType.FLOAT64, false, Shape.of(2));
            Tensor lengths = value(DataType.INT64, false, Shape.of(2));
            var body = new ScanBody(List.of(), List.of(x, temporal), List.of(w), List.of(), List.of(x.mul(temporal).mul(w)));
            var scan = Scan.apply(List.of(), List.of(sequence, changing), List.of(weight), lengths, 0,
                    List.of(OptionalInt.of(0), OptionalInt.empty()), body, reverse);
            var artifact = compile(scan.outputs(), Optional.empty());
            try (var run = new Consumer(artifact, OptionalLong.empty())) {
                run.numbers(sequence, 1, 1, 1, 1, 1, 1, Double.NaN, Double.NaN, 1, 1, Double.NaN, Double.NaN);
                run.numbers(changing, 1, 2, 3, 4, 5, 6);
                run.numbers(weight, 2, -3); run.numbers(lengths, 3, 1);
                run.bind(); run.bound.execute();
                assertArrayEquals(new double[] {2, -6, 2, -6, 6, -12, 0, 0, 10, -18, 0, 0}, run.forward(0));
                run.numbers(sequence, Double.NaN, Double.NaN, 1, 1, Double.NaN, Double.NaN, 1, 1,
                        Double.NaN, Double.NaN, Double.NaN, Double.NaN);
                run.numbers(changing, 7, 8, 9, 10, Double.NaN, Double.NaN);
                run.numbers(weight, -1, 4); run.numbers(lengths, 0, 2);
                run.bound.execute();
                assertArrayEquals(new double[] {0, 0, -7, 32, 0, 0, -9, 40, 0, 0, 0, 0}, run.forward(0));
            }
        }
    }

    @Test void sharedScopeCachesFollowChangingReverseRootBanksAndRetainEmptyPresence() {
        for (boolean lengthMode : List.of(false, true)) {
            Dimension rows = new DynamicDimension("temporalRoots");
            Shape row = Shape.ofDimensions(rows, new StaticDimension(1));
            Tensor h = value(DataType.FLOAT64, true, row), w = value(DataType.FLOAT64, true, Shape.of(1));
            Tensor clock = value(DataType.BOOL, false, row);
            Tensor initial = value(DataType.FLOAT64, true, row), weight = value(DataType.FLOAT64, true, Shape.of(1));
            Tensor sequence = value(DataType.BOOL, false, Shape.ofDimensions(new StaticDimension(3), rows, new StaticDimension(1)));
            Tensor lengths = value(DataType.INT64, false, Shape.ofDimensions(rows));
            var body = new ScanBody(List.of(h), List.of(clock), List.of(w), List.of(h.mul(w)), List.of(h));
            var scan = lengthMode ? Scan.apply(List.of(initial), List.of(sequence), List.of(weight), lengths, 1,
                    List.of(OptionalInt.of(0)), body, false) : Scan.apply(List.of(initial), List.of(sequence), List.of(weight), body, false);
            Tensor seed = value(DataType.FLOAT64, false, scan.outputs().getFirst().descriptor().shape());
            var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
                    List.of(new FunctionalGradientRequest.ForwardTensorReference(scan.outputs().getFirst())),
                    List.of(Optional.of(seed)), List.of(initial, weight), false, FunctionalGradientRequest.DisconnectedPolicy.ZERO)));
            var artifact = compile(scan.outputs(), Optional.of(request));
            for (int batch : new int[] {0, 1, 2}) try (var run = new Consumer(artifact, OptionalLong.of(batch))) {
                double[] init = new double[batch], clocks = new double[3 * batch], seeds = new double[3 * batch];
                double[] ls = new double[batch], expectedInitial = new double[batch];
                double expectedWeight = 0;
                for (int b = 0; b < batch; b++) {
                    init[b] = 3 + b;
                    int length = lengthMode && b == 1 ? 1 : 3;
                    ls[b] = length;
                    expectedInitial[b] = length == 1 ? 1 : 7;
                    if (length == 3) expectedWeight += 5 * init[b];
                    for (int t = 0; t < 3; t++) seeds[t * batch + b] = t < length ? 1 : Double.NaN;
                }
                run.numbers(initial, init); run.numbers(weight, 2); run.numbers(sequence, clocks);
                if (lengthMode) run.numbers(lengths, ls);
                run.numbers(seed, seeds); run.bind(); run.bound.execute();
                assertArrayEquals(expectedInitial, run.gradient(0));
                assertArrayEquals(new double[] {expectedWeight}, run.gradient(1));
                run.bound.execute();
                assertArrayEquals(expectedInitial, run.gradient(0));
                assertArrayEquals(new double[] {expectedWeight}, run.gradient(1));
            }
        }
    }

    @Test void symbolicCategoricalValidationUsesBoundAxisAndTypedIgnoreBeforeBounds() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (DataType labelType : List.of(DataType.INT32, DataType.INT64))
                for (boolean lengthMode : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
                    Dimension rows = new DynamicDimension("categoricalRows");
                    Shape logitsShape = Shape.ofDimensions(rows, new StaticDimension(3));
                    Shape targetsShape = Shape.ofDimensions(rows);
                    Tensor x = value(type, false, logitsShape), target = value(labelType, false, targetsShape);
                    Tensor sequence = value(type, false, Shape.ofDimensions(new StaticDimension(2), rows, new StaticDimension(3)));
                    Tensor targets = value(labelType, false, Shape.ofDimensions(new StaticDimension(2), rows));
                    Tensor lengths = value(DataType.INT64, false, targetsShape);
                    ScalarValue ignore = labelType == DataType.INT32 ? ScalarValue.int32(-7) : ScalarValue.int64(-7);
                    var loss = x.categoricalCrossEntropyWithLogits(target, 1, LossReduction.NONE, ignore);
                    var body = new ScanBody(List.of(), List.of(x, target), List.of(), List.of(), List.of(loss));
                    var scan = lengthMode ? Scan.apply(List.of(), List.of(sequence, targets), List.of(), lengths, 0,
                            List.of(OptionalInt.of(0), OptionalInt.of(0)), body, reverse)
                            : Scan.apply(List.of(), List.of(sequence, targets), List.of(), body, reverse);
                    var artifact = compile(scan.outputs(), Optional.empty());
                    for (int batch : new int[] {0, 1, 3}) try (var run = new Consumer(artifact, OptionalLong.of(batch))) {
                        double[] logits = new double[2 * batch * 3], labels = new double[2 * batch];
                        double[] ls = new double[batch], expected = new double[2 * batch];
                        for (int b = 0; b < batch; b++) ls[b] = b == 0 ? 2 : b == 1 ? 1 : 0;
                        for (int t = 0; t < 2; t++) for (int b = 0; b < batch; b++) {
                            int sample = t * batch + b;
                            boolean active = !lengthMode || t < ls[b];
                            labels[sample] = active ? b == 1 ? -7 : b % 3 : 99;
                            double sum = 0;
                            for (int j = 0; j < 3; j++) {
                                double number = (sample + j - 2) * .25;
                                logits[3 * sample + j] = active && labels[sample] != -7 ? number : Double.NaN;
                                sum += StrictMath.exp(number);
                            }
                            if (active && labels[sample] != -7)
                                expected[sample] = StrictMath.log(sum) - logits[3 * sample + (int) labels[sample]];
                        }
                        run.numbers(sequence, logits); run.numbers(targets, labels);
                        if (lengthMode) run.numbers(lengths, ls);
                        run.bind(); run.bound.execute();
                        assertArrayEquals(expected, run.forward(0), type == DataType.FLOAT32 ? 3e-7 : 1e-14);
                        if (batch > 0) {
                            int bad = (reverse ? 1 : 0) * batch; // Row zero is active at both times.
                            double[] invalid = labels.clone(); invalid[bad] = 3;
                            run.numbers(targets, invalid);
                            run.outputStorage(0).fill((byte) 0x5a);
                            byte[] before = run.outputStorage(0).toArray(ValueLayout.JAVA_BYTE);
                            assertThrows(IndexOutOfBoundsException.class, run.bound::execute);
                            assertArrayEquals(before, run.outputStorage(0).toArray(ValueLayout.JAVA_BYTE));
                            run.numbers(targets, labels); run.bound.execute();
                            assertArrayEquals(expected, run.forward(0), type == DataType.FLOAT32 ? 3e-7 : 1e-14);
                        }
                    }
                }
    }

    private static Tensor stridedLengths() {
        Shape shape = Shape.of(2);
        return TensorFactory.create(new TensorDescriptor(DataType.INT64, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {2}, 1, true)), false));
    }

    private static Tensor value(DataType type, boolean gradient, Shape shape) {
        return TensorFactory.create(new TensorDescriptor(type, shape,
                shape.isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(), gradient));
    }

    private static CompileArtifacts compile(List<Tensor> roots, Optional<FunctionalGradientRequest> request) {
        return CpuGeneralScanCapabilityTest.compile(roots, request, false, new CpuCapabilityProvider());
    }

    /** Uses the real assigned CPU executable and caller-owned borrowed buffers, not a temporal oracle. */
    private static final class Consumer implements AutoCloseable {
        private final CompileArtifacts artifact;
        private final CpuRecurrentScanExecutionTest.Fixture fixture;
        private BoundInvocation bound;

        private Consumer(CompileArtifacts artifact, OptionalLong batch) {
            this.artifact = artifact;
            var context = GraphPreparation.project(artifact, artifact.partitions().getFirst(), CpuPartitionAnalysisInputs.DEFAULT);
            var analysis = new CpuPartitionPreparer().analyze(context);
            fixture = new CpuRecurrentScanExecutionTest.Fixture(
                    artifact, CpuPartitionAnalysisInputs.DEFAULT, analysis, 1, false, false, Map.of(), batch);
        }

        private void bind() { bound = fixture.executable.bind(fixture.state); }

        /** Writes physical backing intentionally, including holes of the real lengths view. */
        private void numbers(Tensor tensor, double... numbers) {
            var binding = artifact.constants().bindableInputBindings().stream()
                    .filter(b -> b.tensorId().equals(tensor.id())).findFirst();
            if (binding.isEmpty()) return;
            var storage = fixture.storage.get(binding.orElseThrow().valueId());
            for (int i = 0; i < numbers.length; i++) set(storage, tensor.descriptor().dataType(), i, numbers[i]);
        }

        private MemorySegment outputStorage(int port) {
            return fixture.storage.get(artifact.publication().forwardBindings().get(port).valueId());
        }

        private double[] forward(int port) { return read(artifact.publication().forwardBindings().get(port).valueId()); }
        private double[] gradient(int port) { return read(artifact.publication().gradientBindings().get(port).valueId()); }

        private double[] read(ValueId id) {
            var descriptor = artifact.graph().values().stream().filter(v -> v.id().equals(id)).findFirst().orElseThrow().descriptor();
            long batch = fixture.state.geometry().batchSize().orElse(0);
            long[] shape = descriptor.shape().dimensions().stream().mapToLong(d -> d.staticSize().orElse(batch)).toArray();
            long count = 1;
            for (long extent : shape) count *= extent;
            var layout = descriptor.layout().orElseGet(() -> LayoutDescriptor.contiguous(Shape.of(shape)));
            long[] strides = layout.strides();
            double[] result = new double[Math.toIntExact(count)];
            var storage = fixture.storage.get(id);
            for (int i = 0; i < result.length; i++) {
                long address = layout.storageOffset(), remainder = i;
                for (int a = shape.length - 1; a >= 0; a--) { address += remainder % shape[a] * strides[a]; remainder /= shape[a]; }
                result[i] = descriptor.dataType() == DataType.FLOAT64 ? storage.getAtIndex(ValueLayout.JAVA_DOUBLE, address)
                        : storage.getAtIndex(ValueLayout.JAVA_FLOAT, address);
            }
            return result;
        }

        private static void set(MemorySegment storage, DataType type, long index, double number) {
            switch (type) {
                case FLOAT64 -> storage.setAtIndex(ValueLayout.JAVA_DOUBLE, index, number);
                case FLOAT32 -> storage.setAtIndex(ValueLayout.JAVA_FLOAT, index, (float) number);
                case INT64 -> storage.setAtIndex(ValueLayout.JAVA_LONG, index, (long) number);
                case INT32 -> storage.setAtIndex(ValueLayout.JAVA_INT, index, (int) number);
                case BOOL -> storage.setAtIndex(ValueLayout.JAVA_BYTE, index, (byte) number);
                default -> throw new AssertionError("unexpected carrier");
            }
        }

        @Override public void close() { fixture.close(); }
    }
}
