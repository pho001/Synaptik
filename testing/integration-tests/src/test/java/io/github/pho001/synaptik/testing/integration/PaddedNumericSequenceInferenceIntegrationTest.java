package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.data.PaddedSequenceBatch;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Model;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Public numeric Data-to-Model binding and explicit single-preparation Engine reuse. */
final class PaddedNumericSequenceInferenceIntegrationTest {
    @ParameterizedTest
    @ValueSource(strings = {"rnn", "gru", "lstm"})
    void freshPaddedBatchesReusePreparedExecutionAndMatchIndependentEquations(String family) {
        for (boolean finalOnly : List.of(false, true)) {
            var builder = Model.builder().input(DataType.FLOAT32, 3, 3, 1).seed(41);
            switch (family) {
                case "rnn" -> builder.rnnScan(1);
                case "gru" -> builder.gruScan(1);
                case "lstm" -> builder.lstmScan(1);
                default -> throw new AssertionError(family);
            }
            if (finalOnly) builder.finalHidden();
            try (Engine engine = Engine.standard(); var model = builder.buildWithLengths()) {
                var parameters = model.stateDictionary().entries().stream()
                        .map(e -> e.value().hostStorage().orElseThrow().segment()
                                .toArray(ValueLayout.JAVA_FLOAT)).toList();
                Tensor prototype = TensorFactory.create(descriptor(DataType.FLOAT32, false, 3, 3, 1));
                Tensor prototypeLengths = batch(prototype, 3, 1, 0).toScanLengthsTensor();
                Tensor output = model.forward(List.of(prototype, prototypeLengths)).getFirst();
                assertTrue(output.descriptor().requiresGrad()); // Floating parameters, not counts.
                assertFalse(prototype.descriptor().requiresGrad());
                assertFalse(prototypeLengths.descriptor().requiresGrad());
                assertTrue(output.descriptor().layout().isEmpty());
                var producer = output.provenance().orElseThrow().producer();
                assertSame(prototype, producer.inputs().get(0));
                assertSame(prototypeLengths, producer.inputs().get(1));
                var compiled = engine.compile(List.of(output));
                // Capture public positional binding once; all remaining scan inputs are fixed leaves.
                List<Tensor> bindings = compiled.inputs().stream().map(input -> producer.inputs().stream()
                        .filter(t -> t.id().equals(input.tensorId())).findFirst().orElseThrow()).toList();
                // Exactly one explicit preparation, outside the changing-values/counts loop.
                try (var prepared = engine.prepare(compiled)) {
                    int pattern = 0;
                    Tensor previousValues = null, previousLengths = null;
                    for (long[] counts : List.of(new long[] {3, 1, 0}, new long[] {0, 0, 0}, new long[] {1, 3, 2})) {
                        float[] x = new float[9];
                        for (int t = 0; t < 3; t++) for (int row = 0; row < 3; row++) {
                            x[t * 3 + row] = t < counts[row]
                                    ? (t * 3 + row + 1 + pattern) * .1f : Float.NaN;
                        }
                        pattern++;
                        Tensor values = TensorFactory.fromFlatArray(prototype.descriptor(), Optional.empty(), x);
                        var batch = batch(values, counts);
                        Tensor lengths = batch.toScanLengthsTensor();
                        assertNotSame(previousValues, values);
                        assertNotSame(previousLengths, lengths);
                        previousValues = values;
                        previousLengths = lengths;
                        List<Tensor> fresh = bindings.stream().map(t -> t == prototype ? values
                                : t == prototypeLengths ? lengths : t).toList();
                        float[] expected = oracle(family, x, counts, parameters).get(finalOnly ? 1 : 0);
                        try (var run = engine.runValues(prepared, fresh)) {
                            HostTensorValue actual = run.materialize(run.publications().getFirst(), 1024);
                            check(expected, actual, counts, finalOnly);
                        }
                        check(expected, model.compute(batch.values(), batch.toScanLengthsTensor()), counts, finalOnly);
                        assertArrayEquals(x, values.hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_FLOAT));
                        assertArrayEquals(counts, batch.validLengths().toLongArray());
                        assertFalse(prepared.isClosed());
                    }
                }
                // Export is not execution admission and never silently fixes a descriptor.
                Tensor lengths = batch(prototype, 3, 1, 0).toScanLengthsTensor();
                for (TensorDescriptor incompatible : List.of(
                        descriptor(DataType.FLOAT64, false, 3, 3, 1),
                        descriptor(DataType.FLOAT32, true, 3, 3, 1),
                        new TensorDescriptor(DataType.FLOAT32, Shape.of(3, 3, 1),
                                Optional.of(LayoutDescriptor.of(Shape.of(3, 3, 1), new long[] {1, 3, 1}, 0, true)), false))) {
                    Tensor values = TensorFactory.allocate(incompatible);
                    assertArrayEquals(new long[] {3, 1, 0}, batch(values, 3, 1, 0).toScanLengthsTensor()
                            .hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_LONG));
                    assertThrows(IllegalArgumentException.class, () -> model.compute(values, lengths));
                    assertSame(incompatible, values.descriptor());
                }
            }
        }
    }

    /**
     * Declares existing numeric values without conversion or storage ownership.
     * @param values borrowed time-major tensor
     * @param counts non-negative valid-prefix counts in original row order
     * @return fresh metadata retaining the exact values
     */
    private static PaddedSequenceBatch batch(Tensor values, long... counts) {
        return PaddedSequenceBatch.builder(values).rightPadded().timeMajor().validLengths(counts).build();
    }

    /**
     * Makes an exact canonical input descriptor for the public factory.
     * @param type logical data type
     * @param grad gradient eligibility
     * @param axes static extents
     * @return non-null canonical descriptor
     */
    private static TensorDescriptor descriptor(DataType type, boolean grad, long... axes) {
        Shape shape = Shape.of(axes);
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), grad);
    }

    /**
     * Checks finite values with arithmetic tolerance and padding/empty-state bits exactly.
     * @param expected scalar equation result
     * @param actual detached public result
     * @param counts original ordered valid counts
     * @param finalOnly whether output omits the time axis
     */
    private static void check(float[] expected, HostTensorValue actual, long[] counts, boolean finalOnly) {
        assertEquals(DataType.FLOAT32, actual.dataType());
        assertEquals(finalOnly ? Shape.of(3, 1) : Shape.of(3, 3, 1), actual.shape());
        float[] observed = new float[expected.length];
        for (int i = 0; i < observed.length; i++) {
            observed[i] = actual.bytes().getFloat(i * Float.BYTES);
            if (finalOnly ? counts[i] == 0 : i / 3 >= counts[i % 3]) {
                assertEquals(0, Float.floatToRawIntBits(observed[i]), "positive-zero padding/initial state");
            }
        }
        assertArrayEquals(expected, observed, 2e-6f);
    }

    /**
     * Evaluates width-one scalar transitions independently, never reading a padded input.
     * @param family rnn, gru or lstm
     * @param x fresh time-major scalar input, with NaN outside valid prefixes
     * @param counts valid counts per original row
     * @param parameters ordered input weight, recurrent weight and bias arrays
     * @return dense hidden values followed by final hidden values, with zero initial states
     */
    private static List<float[]> oracle(String family, float[] x, long[] counts, List<float[]> parameters) {
        float[] dense = new float[9], hidden = new float[3], cell = new float[3];
        float[] w = parameters.get(0), u = parameters.get(1), bias = parameters.get(2);
        for (int row = 0; row < 3; row++) for (int t = 0; t < counts[row]; t++) {
            float[] p = new float[w.length], q = new float[u.length];
            for (int gate = 0; gate < p.length; gate++) {
                p[gate] = x[t * 3 + row] * w[gate] + bias[gate];
                q[gate] = hidden[row] * u[gate];
            }
            switch (family) {
                case "rnn" -> hidden[row] = (float) Math.tanh(p[0] + q[0]);
                case "gru" -> {
                    float reset = sigmoid(p[0] + q[0]), update = sigmoid(p[1] + q[1]);
                    float candidate = (float) Math.tanh(p[2] + reset * q[2]);
                    hidden[row] = candidate + update * (hidden[row] - candidate);
                }
                case "lstm" -> {
                    cell[row] = sigmoid(p[1] + q[1]) * cell[row]
                            + sigmoid(p[0] + q[0]) * (float) Math.tanh(p[2] + q[2]);
                    hidden[row] = sigmoid(p[3] + q[3]) * (float) Math.tanh(cell[row]);
                }
                default -> throw new AssertionError(family);
            }
            dense[t * 3 + row] = hidden[row];
        }
        return List.of(dense, hidden);
    }

    /**
     * Evaluates the independent scalar sigmoid equation.
     * @param x finite gate preactivation
     * @return sigmoid rounded to FLOAT32
     */
    private static float sigmoid(float x) {
        return (float) (1 / (1 + Math.exp(-x)));
    }
}
