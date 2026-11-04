package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFile;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.RecurrentScan;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Model;
import io.github.pho001.synaptik.nn.module.ModelBuilder;
import io.github.pho001.synaptik.training.Sgd;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Public build/save/close/load/compute workflow, qualified against independent scalar equations. */
final class RuntimeLengthScanModelIntegrationTest {
    @TempDir Path directory;

    @ParameterizedTest @ValueSource(strings = {"rnn", "gru", "lstm", "rnn/rnn", "gru/gru", "lstm/lstm", "rnn/gru/lstm", "gru/rnn", "lstm/rnn"})
    void freshOrderedLengthsDriveDenseFinalAndLastOutputsBeforeAndAfterColdLoad(String chain) throws Exception {
        for (String terminal : List.of("dense", "final", "last")) {
            String[] kinds = chain.split("/");
            ModelBuilder builder = Model.builder().input(DataType.FLOAT32, 3, 3, 1).seed(41);
            for (String kind : kinds) append(builder, kind);
            if (terminal.equals("final")) builder.finalHidden();
            if (terminal.equals("last")) builder.lastTimeStep();
            Path file = directory.resolve("scan"), resave = directory.resolve("resaved");
            List<List<float[]>> parameters = new ArrayList<>();
            try (var model = builder.buildWithLengths()) {
                for (int step = 0; step < kinds.length; step++) {
                    String prefix = "layers." + step + ".sequence.cell.";
                    parameters.add(model.stateDictionary().entries().stream().filter(e -> e.path().startsWith(prefix))
                            .map(e -> e.value().hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_FLOAT)).toList());
                }
                model.save(file); // No forward/compute is needed; close original before cold load.
            }
            HostTensorValue retained;
            try (Engine borrowed = Engine.standard(); var loaded = Model.loadGraph(file).withEngine(borrowed)) {
                for (long[] length : List.of(new long[]{3, 1, 0}, new long[]{0, 0, 0}, new long[]{1, 3, 2})) {
                    float[] x = {.2f, .3f, .4f, .5f, .6f, .7f, .8f, .9f, 1};
                    for (int t = 0; t < 3; t++) for (int b = 0; b < 3; b++) if (t >= length[b]) x[t * 3 + b] = Float.NaN;
                    float[] dense = x, finalHidden = null;
                    for (int step = 0; step < kinds.length; step++) {
                        var expected = oracle(kinds[step], dense, length, new float[3], new float[3], parameters.get(step), false);
                        dense = expected.get(0); finalHidden = expected.get(1);
                    }
                    float[] expected = terminal.equals("final") ? finalHidden
                            : terminal.equals("last") ? Arrays.copyOfRange(dense, 6, 9) : dense;
                    Tensor values = floats(false, new long[]{3, 3, 1}, x), lengths = lengths(length);
                    retained = loaded.compute(values, lengths);
                    assertArrayEquals(expected, floats(retained), 2e-6f);
                    assertArrayEquals(floats(retained), floats(loaded.compute(List.of(values, lengths)).getFirst()));
                    assertEquals(terminal.equals("dense") ? Shape.of(3, 3, 1) : Shape.of(3, 1), retained.shape());
                }
                Tensor zero = floats(false, new long[]{3, 3, 1}, new float[9]);
                assertThrows(IllegalArgumentException.class, () -> loaded.compute(zero, lengths(3, 4, 0)));
                retained = loaded.compute(zero, lengths(3, 1, 0)); // Same path remains usable after invalid lengths.
                loaded.save(resave);
                assertArrayEquals(Files.readAllBytes(file), Files.readAllBytes(resave));
                loaded.close(); assertFalse(borrowed.isClosed());
            }
            assertTrue(Float.isFinite(retained.bytes().getFloat()));
        }
    }

    @ParameterizedTest @ValueSource(strings = {"rnn", "gru", "lstm"})
    void advancedExplicitStatesAndFinalStateOnlyRootsKeepCanonicalPortsAndAliases(String family) throws Exception {
        int gates = family.equals("rnn") ? 1 : family.equals("gru") ? 3 : 4;
        Tensor x = TensorFactory.create(descriptor(DataType.FLOAT32, false, 3, 3, 1));
        Tensor l = TensorFactory.create(descriptor(DataType.INT64, false, 3));
        Tensor h = TensorFactory.create(descriptor(DataType.FLOAT32, false, 3, 1));
        Tensor c = TensorFactory.create(h.descriptor());
        float[] w = new float[gates], u = new float[gates], bias = new float[gates];
        for (int i = 0; i < gates; i++) { w[i] = .2f * (i + 1); u[i] = -.1f; bias[i] = .03f * (i + 1); }
        Tensor weight = floats(true, new long[]{gates, 1}, w), hidden = floats(true, new long[]{gates, 1}, u);
        Tensor b = floats(true, new long[]{gates}, bias);
        Tensor root = switch (family) {
            case "rnn" -> RecurrentScan.rnn(x, l, h, weight, hidden, b, RecurrentDirection.REVERSE).finalHidden();
            case "gru" -> RecurrentScan.gru(x, l, h, weight, hidden, b, RecurrentDirection.REVERSE).finalHidden();
            case "lstm" -> RecurrentScan.lstm(x, l, h, c, weight, hidden, b, RecurrentDirection.REVERSE).finalHidden();
            default -> throw new AssertionError(family);
        };
        List<Tensor> inputs = family.equals("lstm") ? List.of(x, l, h, c) : List.of(x, l, h);
        Path file = directory.resolve("explicit");
        try (var model = Model.graph(inputs, List.of(root), List.of(
                new GraphCheckpoint.Leaf("weight", GraphCheckpoint.Role.PARAMETER, weight),
                new GraphCheckpoint.Leaf("hidden", GraphCheckpoint.Role.PARAMETER, hidden),
                new GraphCheckpoint.Leaf("bias", GraphCheckpoint.Role.PARAMETER, b)))) { model.save(file); }
        assertEquals("graph-model.v3", ArtifactFile.read(file,
                new ArtifactLimits(1048576, 1048576, 100, 64, 1048576, 1048576)).kind());
        try (var model = Model.loadGraph(file)) {
            float[] data = {.2f, .3f, Float.NaN, .4f, Float.NaN, Float.NaN, .6f, Float.NaN, Float.NaN};
            float[] state = {-.2f, .1f, -.0f};
            Tensor values = floats(false, new long[]{3, 3, 1}, data), lengths = lengths(3, 1, 0);
            Tensor shared = floats(false, new long[]{3, 1}, state);
            var fresh = family.equals("lstm") ? List.of(values, lengths, shared, shared) : List.of(values, lengths, shared);
            var output = model.forward(fresh).getFirst();
            var producer = output.provenance().orElseThrow().producer();
            assertSame(output, producer.output(1));
            assertEquals(family.equals("lstm") ? 3 : 2, producer.outputCount());
            assertTrue(producer.scanBody().isEmpty());
            if (family.equals("lstm")) assertSame(producer.inputs().get(2), producer.inputs().get(3));
            float[] expected = oracle(family, data, new long[]{3, 1, 0}, state, state, List.of(w, u, bias), true).get(1);
            assertArrayEquals(expected, floats(model.compute(fresh).getFirst()), 2e-6f);
            assertEquals(Float.floatToRawIntBits(-.0f), Float.floatToRawIntBits(floats(model.compute(fresh).getFirst())[2]));
            Path resaved = directory.resolve("fixed-resaved"); model.save(resaved);
            assertArrayEquals(Files.readAllBytes(file), Files.readAllBytes(resaved));
        }
    }

    @Test void ordinaryTwoInputModelsUseDeclaredPositionsNotLengthRoles() throws Exception {
        Tensor a = TensorFactory.create(descriptor(DataType.FLOAT32, false, 1));
        Tensor b = TensorFactory.create(a.descriptor());
        Path file = directory.resolve("ordinary");
        try (var model = Model.graph(List.of(b, a), List.of(a.add(a).add(b)), List.of())) { model.save(file); }
        try (var model = Model.loadGraph(file)) {
            assertEquals(11, model.compute(floats(false, new long[]{1}, 5), floats(false, new long[]{1}, 3)).bytes().getFloat());
        }
    }

    @ParameterizedTest @ValueSource(strings = {"rnn", "gru", "lstm"})
    void cpuOwnedFixedScansUpdateAgainstIndependentFiniteDifferenceAndTrainAfterLoad(String family) throws Exception {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16, DataType.FLOAT16)) {
            try (Engine engine = Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
                if (type == DataType.FLOAT32 || type == DataType.FLOAT64) fixedTraining(engine, family, type, true);
                else assertBackendTrainingRejected(engine, family, type);
            }
        }
    }

    @ParameterizedTest @ValueSource(strings = {"rnn", "gru", "lstm"})
    void metalOnlyOwnedFixedScansTrainOriginalCarriersWithExplicitF32ObjectivesAndRejectF64(String family) throws Exception {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        Assumptions.assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is required for real Metal integration");
        Path library = Path.of(configured).toAbsolutePath().normalize();
        assertTrue(Files.isRegularFile(library), "configured Metal library must exist");
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16, DataType.FLOAT64)) {
            try (Engine.Builder builder = Engine.builder()) {
                builder.takeOwnership(MetalBackendIntegration.open(new MetalBackendConfiguration(library)));
                try (Engine engine = builder.build()) {
                    if (type == DataType.FLOAT64) assertBackendTrainingRejected(engine, family, type);
                    else fixedTraining(engine, family, type, false);
                }
            }
        }
    }

    private static void assertBackendTrainingRejected(Engine engine, String family, DataType type) {
        var builder = Model.builder().input(type, 2, 1, 1).seed(31);
        append(builder, family);
        try (var model = builder.finalHidden().buildWithLengths().withEngine(engine)) {
            var parameters = model.stateDictionary().entries().stream().map(e -> e.value()).toList();
            var before = parameters.stream().map(RuntimeLengthScanModelIntegrationTest::numbers).toList();
            assertThrows(IllegalStateException.class, () -> model.configureTraining(List.of(), (p, t) ->
                    p.getFirst().cast(type == DataType.FLOAT64 ? DataType.FLOAT64 : DataType.FLOAT32).sum(), new Sgd(.125)));
            assertThrows(IllegalStateException.class, () -> model.train((List<Tensor>) null, null));
            for (int p = 0; p < parameters.size(); p++) {
                assertEquals(type, parameters.get(p).descriptor().dataType());
                assertArrayEquals(before.get(p), numbers(parameters.get(p)));
            }
        }
    }

    private void fixedTraining(Engine engine, String family, DataType type, boolean numericalOracle) throws Exception {
        int gates = family.equals("rnn") ? 1 : family.equals("gru") ? 3 : 4;
        Tensor x = TensorFactory.create(descriptor(type, false, 2, 2, 1));
        Tensor l = TensorFactory.create(descriptor(DataType.INT64, false, 2));
        Tensor h = TensorFactory.create(descriptor(type, false, 2, 1));
        Tensor c = TensorFactory.create(h.descriptor());
        double[] w = new double[gates], u = new double[gates], b = new double[gates];
        for (int i = 0; i < gates; i++) { w[i] = .2 * (i + 1); u[i] = -.125; b[i] = .03125 * (i + 1); }
        Tensor weight = value(type, true, new long[]{gates, 1}, w);
        Tensor recurrent = value(type, true, new long[]{gates, 1}, u);
        Tensor bias = value(type, true, new long[]{gates}, b);
        List<Tensor> roots = switch (family) {
            case "rnn" -> {
                var scan = RecurrentScan.rnn(x, l, h, weight, recurrent, bias, RecurrentDirection.FORWARD);
                yield List.of(scan.outputs(), scan.finalHidden());
            }
            case "gru" -> {
                var scan = RecurrentScan.gru(x, l, h, weight, recurrent, bias, RecurrentDirection.FORWARD);
                yield List.of(scan.outputs(), scan.finalHidden());
            }
            case "lstm" -> {
                var scan = RecurrentScan.lstm(x, l, h, c, weight, recurrent, bias, RecurrentDirection.FORWARD);
                yield List.of(scan.outputs(), scan.finalHidden(), scan.finalCell());
            }
            default -> throw new AssertionError(family);
        };
        List<Tensor> inputs = family.equals("lstm") ? List.of(x, l, h, c) : List.of(x, l, h);
        Path before = directory.resolve(family + "-" + type + "-before"), after = directory.resolve(family + "-" + type + "-after");
        try (var model = Model.graph(inputs, roots, List.of(
                new GraphCheckpoint.Leaf("weight", GraphCheckpoint.Role.PARAMETER, weight),
                new GraphCheckpoint.Leaf("hidden", GraphCheckpoint.Role.PARAMETER, recurrent),
                new GraphCheckpoint.Leaf("bias", GraphCheckpoint.Role.PARAMETER, bias))).withEngine(engine)) {
            model.save(before);
            exerciseTraining(model, engine, family, type, numericalOracle);
            model.save(after);
        }
        for (Path saved : List.of(before, after)) {
            try (var loaded = Model.loadGraph(saved).withEngine(engine)) {
                exerciseTraining(loaded, engine, family, type, numericalOracle);
            }
        }
        assertFalse(engine.isClosed());
    }

    private static Tensor objective(List<Tensor> predictions, List<Tensor> targets, DataType objectiveType) {
        Tensor total = null;
        for (int i = 0; i < predictions.size(); i++) {
            Tensor p = predictions.get(i);
            Tensor error = (p.descriptor().dataType() == objectiveType ? p : p.cast(objectiveType)).sub(targets.get(i));
            Tensor term = error.mul(error).sum();
            total = total == null ? term : total.add(term);
        }
        return total;
    }

    private static void exerciseTraining(Model<List<Tensor>, List<Tensor>> model, Engine engine, String family,
            DataType type, boolean numericalOracle) throws Exception {
        var parameters = model.stateDictionary().entries().stream().map(e -> e.value()).toList();
        DataType objectiveType = type == DataType.FLOAT64 ? DataType.FLOAT64 : DataType.FLOAT32;
        Tensor zeroInput = value(type, false, new long[]{2, 2, 1}, 0, 0, 0, 0);
        Tensor initial = value(type, false, new long[]{2, 1}, .125, -.25);
        Tensor initialCell = value(type, false, new long[]{2, 1}, .25, -.125);
        List<Tensor> initialBatch = family.equals("lstm") ? List.of(zeroInput, lengths(2, 0), initial, initialCell)
                : List.of(zeroInput, lengths(2, 0), initial);
        var prototypeOutputs = model.forward(initialBatch);
        assertTrue(prototypeOutputs.getFirst().provenance().orElseThrow().producer().scanBody().isEmpty());
        var targets = prototypeOutputs.stream().map(p -> value(objectiveType, false,
                p.descriptor().shape().toLongArray(),
                new double[Math.toIntExact(p.descriptor().shape().knownElementCount().orElseThrow())])).toList();
        var stores = parameters.stream().map(p -> p.hostStorage().orElseThrow()).toList();
        var callbacks = new java.util.concurrent.atomic.AtomicInteger();
        model.configureTraining(targets, (p, t) -> {
            callbacks.incrementAndGet();
            return objective(p, t, objectiveType);
        }, new Sgd(.125));
        for (int i = 0; i < parameters.size(); i++) {
            assertSame(parameters.get(i), model.stateDictionary().entries().get(i).value());
            assertNotSame(stores.get(i), parameters.get(i).hostStorage().orElseThrow());
            assertEquals(type, parameters.get(i).descriptor().dataType());
        }
        double[][] master = parameters.stream().map(RuntimeLengthScanModelIntegrationTest::numbers).toArray(double[][]::new);
        int execution = 0;
        for (long[] valid : List.of(new long[]{2, 0}, new long[]{1, 2}, new long[]{0, 0}, new long[]{2, 1})) {
            Tensor freshX = value(type, false, new long[]{2, 2, 1}, .25 + execution * .125, -.125, -.25, .375);
            Tensor freshH = value(type, false, new long[]{2, 1}, .125, -.25);
            Tensor freshC = value(type, false, new long[]{2, 1}, .25, -.125);
            List<Tensor> batch = family.equals("lstm") ? List.of(freshX, lengths(valid), freshH, freshC)
                    : List.of(freshX, lengths(valid), freshH);
            double[][] old = parameters.stream().map(RuntimeLengthScanModelIntegrationTest::numbers).toArray(double[][]::new);
            var outputs = model.compute(batch);
            for (var output : outputs) assertEquals(type, output.dataType());
            var expected = engine.backward(objective(model.forward(batch), targets, objectiveType), parameters, 10000);
            if (numericalOracle) {
                double[] x = numbers(freshX), h = numbers(freshH), c = numbers(freshC);
                var independent = fixedOracle(family, x, valid, h, c, old);
                for (int port = 0; port < outputs.size(); port++)
                    assertArrayEquals(independent.get(port), numbers(outputs.get(port)), type == DataType.FLOAT64 ? 1e-12 : 2e-6);
                for (int parameter = 0; parameter < old.length; parameter++) {
                    double[] gradient = numbers(expected.gradients().get(parameter));
                    for (int lane = 0; lane < old[parameter].length; lane++) {
                        double center = old[parameter][lane], epsilon = 1e-5;
                        old[parameter][lane] = center + epsilon;
                        double plus = squared(fixedOracle(family, x, valid, h, c, old));
                        old[parameter][lane] = center - epsilon;
                        double minus = squared(fixedOracle(family, x, valid, h, c, old));
                        old[parameter][lane] = center;
                        assertEquals((plus - minus) / (2 * epsilon), gradient[lane], type == DataType.FLOAT64 ? 2e-9 : 3e-6);
                    }
                }
            }
            var step = model.train(batch, targets);
            assertEquals(++execution, step.executionNumber());
            assertEquals(execution, step.optimizerStep().orElseThrow());
            assertEquals(objectiveType, step.objective().dataType());
            assertArrayEquals(numbers(expected.objective()), numbers(step.objective()), type == DataType.FLOAT64 ? 1e-12 : 2e-6);
            for (int p = 0; p < parameters.size(); p++) {
                double[] gradient = numbers(expected.gradients().get(p));
                double[] updated = numbers(parameters.get(p));
                for (int lane = 0; lane < updated.length; lane++) {
                    master[p][lane] = type == DataType.FLOAT64 ? master[p][lane] - .125 * gradient[lane]
                            : (float) ((float) master[p][lane] - .125f * (float) gradient[lane]);
                    assertEquals(represent(type, master[p][lane]), updated[lane], type == DataType.FLOAT64 ? 1e-12 : 2e-6);
                }
            }
            if (valid[0] != 0 || valid[1] != 0) {
                assertTrue(java.util.stream.IntStream.range(0, parameters.size())
                        .anyMatch(p -> !Arrays.equals(old[p], numbers(parameters.get(p)))));
            }
        }
        assertEquals(1, callbacks.get());
        var beforeInvalid = parameters.stream().map(RuntimeLengthScanModelIntegrationTest::numbers).toList();
        List<Tensor> invalid = family.equals("lstm") ? List.of(zeroInput, lengths(3, 0), initial, initialCell)
                : List.of(zeroInput, lengths(3, 0), initial);
        assertThrows(IllegalArgumentException.class, () -> model.train(invalid, targets));
        for (int p = 0; p < parameters.size(); p++) assertArrayEquals(beforeInvalid.get(p), numbers(parameters.get(p)));
    }

    /** Independent width-one equations, shared by the forward and finite-difference objective oracles. */
    private static List<double[]> fixedOracle(String family, double[] x, long[] lengths, double[] initial,
            double[] initialCell, double[][] parameters) {
        double[] y = new double[4], h = initial.clone(), c = initialCell.clone();
        for (int row = 0; row < 2; row++) for (int t = 0; t < lengths[row]; t++) {
            double[] p = new double[parameters[0].length], q = new double[p.length];
            for (int gate = 0; gate < p.length; gate++) {
                p[gate] = x[t * 2 + row] * parameters[0][gate] + parameters[2][gate];
                q[gate] = h[row] * parameters[1][gate];
            }
            if (family.equals("rnn")) h[row] = Math.tanh(p[0] + q[0]);
            else if (family.equals("gru")) {
                double r = logistic(p[0] + q[0]), z = logistic(p[1] + q[1]);
                double n = Math.tanh(p[2] + r * q[2]); h[row] = n + z * (h[row] - n);
            } else {
                c[row] = logistic(p[1] + q[1]) * c[row] + logistic(p[0] + q[0]) * Math.tanh(p[2] + q[2]);
                h[row] = logistic(p[3] + q[3]) * Math.tanh(c[row]);
            }
            y[t * 2 + row] = h[row];
        }
        return family.equals("lstm") ? List.of(y, h, c) : List.of(y, h);
    }

    private static double logistic(double x) { return 1 / (1 + Math.exp(-x)); }

    private static double squared(List<double[]> outputs) {
        double sum = 0;
        for (double[] output : outputs) for (double lane : output) sum += lane * lane;
        return sum;
    }

    private static double represent(DataType type, double value) {
        return switch (type) {
            case FLOAT64 -> value;
            case FLOAT32 -> (float) value;
            case BFLOAT16 -> BFloat16Bits.toFloat(BFloat16Bits.fromFloat((float) value));
            case FLOAT16 -> Float16Bits.toFloat(Float16Bits.fromFloat((float) value));
            default -> throw new AssertionError(type);
        };
    }

    private static Tensor value(DataType type, boolean grad, long[] axes, double... values) {
        MemorySegment memory;
        if (type == DataType.FLOAT64) memory = MemorySegment.ofArray(values.clone());
        else if (type == DataType.FLOAT32) {
            float[] lanes = new float[values.length];
            for (int i = 0; i < lanes.length; i++) lanes[i] = (float) values[i];
            memory = MemorySegment.ofArray(lanes);
        } else {
            short[] lanes = new short[values.length];
            for (int i = 0; i < lanes.length; i++) lanes[i] = type == DataType.BFLOAT16
                    ? BFloat16Bits.fromFloat((float) values[i]) : Float16Bits.fromFloat((float) values[i]);
            memory = MemorySegment.ofArray(lanes);
        }
        return TensorFactory.create(descriptor(type, grad, axes), Optional.empty(),
                Optional.of(new MemorySegmentStorage(type, values.length, memory)));
    }

    private static double[] numbers(Tensor tensor) {
        var bytes = java.nio.ByteBuffer.wrap(tensor.hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_BYTE))
                .order(java.nio.ByteOrder.nativeOrder());
        return numbers(tensor.descriptor().dataType(), bytes);
    }

    private static double[] numbers(HostTensorValue value) { return numbers(value.dataType(), value.bytes()); }

    private static double[] numbers(DataType type, java.nio.ByteBuffer bytes) {
        double[] result = new double[bytes.remaining() / type.byteWidth()];
        for (int i = 0; i < result.length; i++) result[i] = switch (type) {
            case FLOAT64 -> bytes.getDouble();
            case FLOAT32 -> bytes.getFloat();
            case BFLOAT16 -> BFloat16Bits.toFloat(bytes.getShort());
            case FLOAT16 -> Float16Bits.toFloat(bytes.getShort());
            default -> throw new AssertionError(type);
        };
        return result;
    }


    private static void append(ModelBuilder builder, String family) {
        switch (family) { case "rnn" -> builder.rnnScan(1); case "gru" -> builder.gruScan(1); case "lstm" -> builder.lstmScan(1); default -> throw new AssertionError(); }
    }

    private static TensorDescriptor descriptor(DataType type, boolean grad, long... axes) {
        Shape shape = Shape.of(axes);
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), grad);
    }

    private static Tensor floats(boolean grad, long[] axes, float... values) {
        return TensorFactory.fromFlatArray(descriptor(DataType.FLOAT32, grad, axes), Optional.empty(), values);
    }

    private static Tensor lengths(long... values) {
        return TensorFactory.fromFlatArray(descriptor(DataType.INT64, false, values.length), Optional.empty(), values);
    }

    private static float[] floats(HostTensorValue value) {
        float[] values = new float[value.bytes().remaining() / 4];
        for (int i = 0; i < values.length; i++) values[i] = value.bytes().getFloat(i * 4);
        return values;
    }

    /** Independent scalar equations for width-one packed gates; skips suffixes before reading x. */
    private static List<float[]> oracle(String kind, float[] x, long[] lengths, float[] initialH,
            float[] initialC, List<float[]> parameters, boolean reverse) {
        float[] dense = new float[9], hidden = initialH.clone(), cell = initialC.clone();
        float[] w = parameters.get(0), u = parameters.get(1), bias = parameters.get(2);
        for (int b = 0; b < 3; b++) for (int step = 0; step < lengths[b]; step++) {
            int t = reverse ? (int) lengths[b] - 1 - step : step;
            float[] p = new float[w.length], q = new float[u.length];
            for (int gate = 0; gate < p.length; gate++) {
                p[gate] = x[t * 3 + b] * w[gate] + bias[gate]; q[gate] = hidden[b] * u[gate];
            }
            switch (kind) {
                case "rnn" -> hidden[b] = tanh(p[0] + q[0]);
                case "gru" -> {
                    float reset = sigmoid(p[0] + q[0]), update = sigmoid(p[1] + q[1]);
                    float candidate = tanh(p[2] + reset * q[2]);
                    hidden[b] = candidate + update * (hidden[b] - candidate);
                }
                case "lstm" -> {
                    cell[b] = sigmoid(p[1] + q[1]) * cell[b] + sigmoid(p[0] + q[0]) * tanh(p[2] + q[2]);
                    hidden[b] = sigmoid(p[3] + q[3]) * tanh(cell[b]);
                }
                default -> throw new AssertionError(kind);
            }
            dense[t * 3 + b] = hidden[b];
        }
        return List.of(dense, hidden);
    }

    private static float sigmoid(float x) { return (float) (1 / (1 + Math.exp(-x))); }
    private static float tanh(float x) { return (float) Math.tanh(x); }
}
