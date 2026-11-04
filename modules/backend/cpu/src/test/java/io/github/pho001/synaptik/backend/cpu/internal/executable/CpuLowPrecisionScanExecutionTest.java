package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.*;
import static io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRecurrentScanLoweringTest.scalarContext;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.CpuGeneralScanCapabilityTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.compiler.FunctionalGradientRequest;
import io.github.pho001.synaptik.model.datatype.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.recurrent.*;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Consumer-visible low storage, active domains, state rounding and actual Compiler reverse scans. */
class CpuLowPrecisionScanExecutionTest {
    private static final List<DataType> LOW = List.of(DataType.FLOAT16, DataType.BFLOAT16,
            DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2);

    @Test void ordinaryAndLengthScansDifferentiateRowValuesAndSharedParametersAtActualBatch() {
        for (var type : LOW) for (boolean symbolic : List.of(false, true))
            for (boolean lengthAware : List.of(false, true)) for (boolean reverse : List.of(false, true))
                for (int time : List.of(0, 2)) {
                    Dimension batchDimension = symbolic ? new DynamicDimension("B") : new StaticDimension(3);
                    Shape row = Shape.ofDimensions(batchDimension, new StaticDimension(1));
                    Shape stack = Shape.ofDimensions(new StaticDimension(time), batchDimension, new StaticDimension(1));
                    Tensor h = tensor(type, row, true), x = tensor(type, row, true);
                    Tensor b = tensor(type, Shape.of(1), true), initial = tensor(type, row, true);
                    Tensor sequence = tensor(type, stack, true), bias = tensor(type, Shape.of(1), true);
                    Tensor lengths = tensor(DataType.INT64, Shape.ofDimensions(batchDimension), false);
                    Tensor next = h.add(x).add(b);
                    var body = new ScanBody(List.of(h), List.of(x), List.of(b), List.of(next), List.of(next));
                    var scan = lengthAware ? Scan.apply(List.of(initial), List.of(sequence), List.of(bias),
                            lengths, 1, List.of(OptionalInt.of(0)), body, reverse)
                            : Scan.apply(List.of(initial), List.of(sequence), List.of(bias), body, reverse);
                    Tensor seed = tensor(type, row, false);
                    var roots = scan.finalCarry();
                    var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
                            List.of(new FunctionalGradientRequest.ForwardTensorReference(roots.getFirst())),
                            List.of(Optional.of(seed)), List.of(initial, sequence, bias), false,
                            FunctionalGradientRequest.DisconnectedPolicy.ZERO)));
                    var artifacts = CpuGeneralScanCapabilityTest.compile(roots, Optional.of(request), false,
                            new CpuCapabilityProvider());
                    var context = io.github.pho001.synaptik.prepare.GraphPreparation.project(artifacts,
                            artifacts.partitions().getFirst(), CpuPartitionAnalysisInputs.DEFAULT);
                    var analysis = new CpuPartitionPreparer().analyze(context);
                    Fixture owner = null;
                    try {
                        for (int batch : symbolic ? List.of(0, 1, 3) : List.of(3)) {
                            var actualBatch = symbolic ? OptionalLong.of(batch) : OptionalLong.empty();
                            var fixture = new Fixture(artifacts, CpuPartitionAnalysisInputs.DEFAULT, analysis,
                                    2, true, true, Map.of(), actualBatch, owner);
                            if (owner == null) owner = fixture;
                            try {
                                long[] counts = new long[batch];
                                for (int rowIndex = 0; rowIndex < batch; rowIndex++)
                                    counts[rowIndex] = lengthAware ? Math.min(time, rowIndex) : time;
                                for (var binding : artifacts.constants().bindableInputBindings()) {
                                    var segment = fixture.storage.get(binding.valueId());
                                    if (binding.tensorId().equals(lengths.id())) {
                                        segment.copyFrom(MemorySegment.ofArray(counts));
                                    } else {
                                        double value = binding.tensorId().equals(seed.id()) ? 1
                                                : binding.tensorId().equals(bias.id()) ? .25
                                                : binding.tensorId().equals(sequence.id()) ? Double.NaN : 0;
                                        for (long i = 0; i < segment.byteSize() / type.byteWidth(); i++) write(type, segment, i, value);
                                        if (binding.tensorId().equals(sequence.id()))
                                            for (int t = 0; t < time; t++) for (int rowIndex = 0; rowIndex < batch; rowIndex++)
                                                if (t < counts[rowIndex]) write(type, segment, (long) t * batch + rowIndex, .25);
                                    }
                                }
                                var bound = fixture.executable.bind(fixture.state);
                                for (int repeat = 0; repeat < 2; repeat++) {
                                    bound.execute();
                                    var forward = fixture.storage.get(artifacts.publication().forwardBindings().getFirst().valueId());
                                    for (int rowIndex = 0; rowIndex < batch; rowIndex++)
                                        assertEquals(counts[rowIndex] * .5, number(type, forward, rowIndex));
                                    var gradients = artifacts.publication().gradientBindings();
                                    var dh = fixture.storage.get(gradients.get(0).valueId());
                                    var dx = fixture.storage.get(gradients.get(1).valueId());
                                    var db = fixture.storage.get(gradients.get(2).valueId());
                                    for (int rowIndex = 0; rowIndex < batch; rowIndex++) assertEquals(1, number(type, dh, rowIndex));
                                    for (int t = 0; t < time; t++) for (int rowIndex = 0; rowIndex < batch; rowIndex++)
                                        assertEquals(t < counts[rowIndex] ? 1 : 0, number(type, dx, (long) t * batch + rowIndex));
                                    assertEquals(Arrays.stream(counts).sum(), number(type, db, 0));
                                }
                            } finally { if (fixture != owner) fixture.close(); }
                        }
                    } finally { if (owner != null) owner.close(); }
                }
    }

    @Test void fixedFamiliesKeepFloatWorkNarrowedStateAndInactiveRawBitsOnAffineReads() {
        for (var type : LOW) for (var kind : RecurrentScanKind.values())
            for (var direction : RecurrentDirection.values()) for (int carrier : List.of(0, 1, 2)) {
                var analysis = new CpuPartitionPreparer().analyze(scalarContext(kind, false,
                        direction, 2, 2, 1, 1, false, type, 2));
                var facts = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
                try (var fixture = new Fixture(analysis, carrier, true, true, Map.of())) {
                    scatterLengths(facts, fixture, 2, 0);
                    scatter(facts, fixture, 0, new double[] {.25, Double.NaN, .25, Double.NaN});
                    scatter(facts, fixture, 2, new double[] {0, -0d});
                    boolean cell = kind == RecurrentScanKind.LSTM;
                    if (cell) scatter(facts, fixture, 3, new double[] {.25, -0d});
                    int weight = cell ? 4 : 3;
                    double[] ones = new double[facts.geometry().gates()]; Arrays.fill(ones, 1);
                    scatter(facts, fixture, weight, ones);
                    scatter(facts, fixture, weight + 1, new double[ones.length]);
                    var bound = fixture.executable.bind(fixture.state);
                    for (int repeat = 0; repeat < 2; repeat++) {
                        bound.execute();
                        float hidden = 0, state = .25f;
                        for (int step = 0; step < 2; step++) {
                            float gate = 1f / (1f + (float) StrictMath.exp(-.25f));
                            float candidate = (float) StrictMath.tanh(.25f);
                            if (kind == RecurrentScanKind.RNN_TANH) hidden = rounded(type, candidate);
                            else if (kind == RecurrentScanKind.GRU_RESET_AFTER)
                                hidden = rounded(type, candidate + gate * (hidden - candidate));
                            else {
                                state = rounded(type, gate * state + gate * candidate);
                                hidden = rounded(type, gate * (float) StrictMath.tanh(state));
                            }
                            int time = direction == RecurrentDirection.REVERSE ? 1 - step : step;
                            assertEquals(hidden, number(type, fixture.segment(facts.readCount()), time * 2L));
                            assertEquals(0, number(type, fixture.segment(facts.readCount()), time * 2L + 1));
                        }
                        assertEquals(hidden, number(type, fixture.segment(facts.readCount() + 1), 0));
                        assertEquals(Double.doubleToRawLongBits(-0d), Double.doubleToRawLongBits(
                                number(type, fixture.segment(facts.readCount() + 1), 1)));
                        if (cell) {
                            assertEquals(state, number(type, fixture.segment(facts.readCount() + 2), 0));
                            assertEquals(Double.doubleToRawLongBits(-0d), Double.doubleToRawLongBits(
                                    number(type, fixture.segment(facts.readCount() + 2), 1)));
                        }
                    }
                }
            }
    }

    @Test void singletonLowRecurrentProjectionsDoNotInsertPositiveZero() {
        for (var type : LOW) {
            var analysis = new CpuPartitionPreparer().analyze(scalarContext(RecurrentScanKind.RNN_TANH,
                    false, RecurrentDirection.FORWARD, 1, 1, 1, 1, false, type, 2));
            var facts = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
            try (var fixture = new Fixture(analysis, 2, true, true, Map.of())) {
                scatterLengths(facts, fixture, 1);
                scatter(facts, fixture, 0, new double[] {-0d}); scatter(facts, fixture, 2, new double[] {-0d});
                scatter(facts, fixture, 3, new double[] {1}); scatter(facts, fixture, 4, new double[] {1});
                fixture.executable.bind(fixture.state).execute();
                assertEquals(Double.doubleToRawLongBits(-0d), Double.doubleToRawLongBits(
                        number(type, fixture.segment(facts.readCount()), 0)));
            }
        }
    }

    @Test void fixedFamiliesUseActualCompilerReverseBodiesAtTheirFloatWorkingSites() {
        for (var type : LOW) for (var kind : RecurrentScanKind.values())
            for (var direction : RecurrentDirection.values()) {
                int gates = kind == RecurrentScanKind.RNN_TANH ? 1
                        : kind == RecurrentScanKind.GRU_RESET_AFTER ? 3 : 4;
                Tensor x = tensor(type, Shape.of(2, 2, 1), true);
                Tensor lengths = tensor(DataType.INT64, Shape.of(2), false);
                Tensor h = tensor(type, Shape.of(2, 1), true), c = tensor(type, Shape.of(2, 1), true);
                Tensor w = tensor(type, Shape.of(gates, 1), true), u = tensor(type, Shape.of(gates, 1), true);
                Tensor seed = tensor(type, Shape.of(2, 1), false);
                Tensor root;
                if (kind == RecurrentScanKind.RNN_TANH) root = RecurrentScan.rnn(x, lengths, h, w, u, direction).finalHidden();
                else if (kind == RecurrentScanKind.GRU_RESET_AFTER) root = RecurrentScan.gru(x, lengths, h, w, u, direction).finalHidden();
                else root = RecurrentScan.lstm(x, lengths, h, c, w, u, direction).finalHidden();
                var targets = kind == RecurrentScanKind.LSTM ? List.of(x, h, c) : List.of(x, h);
                var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
                        List.of(new FunctionalGradientRequest.ForwardTensorReference(root)),
                        List.of(Optional.of(seed)), targets, false, FunctionalGradientRequest.DisconnectedPolicy.ZERO)));
                var artifacts = CpuGeneralScanCapabilityTest.compile(List.of(root), Optional.of(request), false,
                        new CpuCapabilityProvider());
                var context = io.github.pho001.synaptik.prepare.GraphPreparation.project(artifacts,
                        artifacts.partitions().getFirst(), CpuPartitionAnalysisInputs.DEFAULT);
                var analysis = new CpuPartitionPreparer().analyze(context);
                try (var fixture = new Fixture(artifacts, CpuPartitionAnalysisInputs.DEFAULT, analysis,
                        2, true, true, Map.of())) {
                    for (var binding : artifacts.constants().bindableInputBindings()) {
                        var segment = fixture.storage.get(binding.valueId());
                        if (binding.tensorId().equals(lengths.id())) segment.copyFrom(MemorySegment.ofArray(new long[] {2, 0}));
                        else {
                            double value = binding.tensorId().equals(w.id()) || binding.tensorId().equals(seed.id()) ? 1 : 0;
                            for (long i = 0; i < segment.byteSize() / type.byteWidth(); i++) write(type, segment, i, value);
                            if (binding.tensorId().equals(x.id())) {
                                write(type, segment, 1, Double.NaN); write(type, segment, 3, Double.NaN);
                            }
                        }
                    }
                    fixture.executable.bind(fixture.state).execute();
                    var gradients = artifacts.publication().gradientBindings();
                    var dx = fixture.storage.get(gradients.get(0).valueId());
                    double first = kind == RecurrentScanKind.RNN_TANH ? 0
                            : kind == RecurrentScanKind.GRU_RESET_AFTER ? .25 : .125;
                    double last = kind == RecurrentScanKind.RNN_TANH ? 1
                            : kind == RecurrentScanKind.GRU_RESET_AFTER ? .5 : .25;
                    assertEquals(direction == RecurrentDirection.FORWARD ? first : last, number(type, dx, 0));
                    assertEquals(direction == RecurrentDirection.FORWARD ? last : first, number(type, dx, 2));
                    assertEquals(0, number(type, dx, 1)); assertEquals(0, number(type, dx, 3));
                    var dh = fixture.storage.get(gradients.get(1).valueId());
                    assertEquals(kind == RecurrentScanKind.GRU_RESET_AFTER ? .25 : 0, number(type, dh, 0));
                    assertEquals(1, number(type, dh, 1));
                    if (kind == RecurrentScanKind.LSTM) {
                        var dc = fixture.storage.get(gradients.get(2).valueId());
                        assertEquals(.125, number(type, dc, 0)); assertEquals(0, number(type, dc, 1));
                    }
                }
            }
    }

    @Test void tanhReverseConsumesRepresentedSavedOutputsRatherThanReplayingFloatValues() {
        for (var type : LOW) {
            Tensor h = tensor(type, Shape.of(1, 1), true), x = tensor(type, Shape.of(1, 1), false);
            Tensor initial = tensor(type, Shape.of(1, 1), true), sequence = tensor(type, Shape.of(2, 1, 1), false);
            Tensor next = h.add(x).tanh(), seed = tensor(type, Shape.of(1, 1), false);
            var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(),
                    new ScanBody(List.of(h), List.of(x), List.of(), List.of(next), List.of(next)), false);
            var roots = List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst());
            var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
                    List.of(new FunctionalGradientRequest.ForwardTensorReference(roots.getFirst())),
                    List.of(Optional.of(seed)), List.of(initial), false, FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
            var artifacts = CpuGeneralScanCapabilityTest.compile(roots, Optional.of(request), false, new CpuCapabilityProvider());
            var context = io.github.pho001.synaptik.prepare.GraphPreparation.project(artifacts,
                    artifacts.partitions().getFirst(), CpuPartitionAnalysisInputs.DEFAULT);
            var analysis = new CpuPartitionPreparer().analyze(context);
            try (var fixture = new Fixture(artifacts, CpuPartitionAnalysisInputs.DEFAULT, analysis,
                    2, true, true, Map.of())) {
                for (var binding : artifacts.constants().bindableInputBindings()) {
                    var segment = fixture.storage.get(binding.valueId());
                    double value = binding.tensorId().equals(seed.id()) ? 1 : binding.tensorId().equals(initial.id()) ? .25 : 0;
                    for (long i = 0; i < segment.byteSize() / type.byteWidth(); i++) write(type, segment, i, value);
                }
                fixture.executable.bind(fixture.state).execute();
                float first = rounded(type, (float) StrictMath.tanh(.25f));
                float second = rounded(type, (float) StrictMath.tanh(first));
                float firstDerivative = rounded(type, 1 - rounded(type, first * first));
                float secondDerivative = rounded(type, 1 - rounded(type, second * second));
                var outputs = fixture.storage.get(artifacts.publication().forwardBindings().get(1).valueId());
                assertEquals(first, number(type, outputs, 0)); assertEquals(second, number(type, outputs, 1));
                var gradient = fixture.storage.get(artifacts.publication().gradientBindings().getFirst().valueId());
                assertEquals(rounded(type, firstDerivative * secondDerivative), number(type, gradient, 0));
            }
        }
    }

    @Test void lowEmptyDomainsAndInvalidLengthsPreserveOriginalStateBytesBeforeAnyWrite() {
        for (var type : LOW) for (var kind : RecurrentScanKind.values())
            for (int time : List.of(0, 2)) for (int batch : List.of(0, 2)) {
                var analysis = new CpuPartitionPreparer().analyze(scalarContext(kind, false,
                        RecurrentDirection.REVERSE, time, batch, 1, 1, false, type, 0));
                var facts = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
                try (var fixture = new Fixture(analysis, 2, true, true, Map.of())) {
                    scatterLengths(facts, fixture, new long[batch]);
                    int states = kind == RecurrentScanKind.LSTM ? 2 : 1;
                    var originals = new ArrayList<byte[]>();
                    for (int state = 0; state < states; state++) {
                        var source = fixture.segment(facts.rolePositions().get(2 + state));
                        for (int row = 0; row < batch; row++) {
                            if (type.byteWidth() == 1) source.setAtIndex(ValueLayout.JAVA_BYTE, row,
                                    row == 0 ? (byte) 0x80 : (byte) 0xff);
                            else source.setAtIndex(ValueLayout.JAVA_SHORT, row,
                                    row == 0 ? (short) 0x8000 : (short) 0xffff);
                        }
                        originals.add(source.toArray(ValueLayout.JAVA_BYTE));
                    }
                    for (int output = facts.readCount(); output < facts.boundaryValues().size(); output++)
                        fixture.segment(output).fill((byte) 0x5a);
                    var bound = fixture.executable.bind(fixture.state);
                    if (batch > 0) {
                        scatterLengths(facts, fixture, 0, time + 1L);
                        assertThrows(IllegalArgumentException.class, bound::execute);
                        for (int output = facts.readCount(); output < facts.boundaryValues().size(); output++)
                            for (byte value : fixture.segment(output).toArray(ValueLayout.JAVA_BYTE))
                                assertEquals((byte) 0x5a, value);
                        scatterLengths(facts, fixture, new long[batch]);
                    }
                    bound.execute();
                    for (byte value : fixture.segment(facts.readCount()).toArray(ValueLayout.JAVA_BYTE)) assertEquals(0, value);
                    for (int state = 0; state < states; state++)
                        assertArrayEquals(originals.get(state), fixture.segment(facts.readCount() + 1 + state).toArray(ValueLayout.JAVA_BYTE));
                }
            }
    }

    private static Tensor tensor(DataType type, Shape shape, boolean gradient) {
        return TensorFactory.create(new TensorDescriptor(type, shape, shape.isFullyStatic()
                ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(), gradient));
    }

    private static float rounded(DataType type, float value) {
        return switch (type) {
            case FLOAT16 -> Float16Bits.toFloat(Float16Bits.fromFloat(value));
            case BFLOAT16 -> BFloat16Bits.toFloat(BFloat16Bits.fromFloat(value));
            case FLOAT8_E4M3 -> Float8E4M3Bits.toFloat(Float8E4M3Bits.fromFloat(value));
            case FLOAT8_E5M2 -> Float8E5M2Bits.toFloat(Float8E5M2Bits.fromFloat(value));
            default -> throw new AssertionError(type);
        };
    }
}
