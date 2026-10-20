package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Model;
import io.github.pho001.synaptik.training.Sgd;
import java.lang.foreign.ValueLayout;
import io.github.pho001.synaptik.nn.module.ModelBuilder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import java.util.List;
import java.net.URI;
import java.net.URLClassLoader;
import java.util.regex.Pattern;
import javax.tools.DiagnosticCollector;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.tools.ToolProvider;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Public fluent construction shares existing execution/training owners without a checkpoint gate. */
final class FluentModelBuilderIntegrationTest {
    @TempDir Path directory;

    private static Tensor values(boolean gradient, long[] axes, float... lanes) {
        Shape shape = Shape.of(axes);
        return TensorFactory.fromFlatArray(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), gradient), Optional.empty(), lanes);
    }

    private static Tensor objective(Tensor prediction, Tensor target) {
        Tensor error = prediction.sub(target);
        return error.mul(error).sum();
    }

    @Test void documentedFluentWorkflowTrainsPrivateWeightsAndReloadsWithBorrowedEngine() throws Exception {
        Tensor callerWeight = values(true, new long[]{1, 1}, 2);
        Tensor batch = values(false, new long[]{1, 1}, 3), target = values(false, new long[]{1, 1}, 1);
        Path file = directory.resolve("trained.graph");
        HostTensorValue retained;
        try (Engine borrowed = Engine.standard()) {
            try (var model = Model.builder().input(DataType.FLOAT32, 1, 1).linear(callerWeight)
                    .build().withEngine(borrowed)) {
                model.save(file);
                retained = model.compute(batch);
                assertEquals(6, retained.bytes().getFloat());
                // Independent Tensor expression and Engine backward, not builder/forward as oracle.
                Tensor directWeight = values(true, new long[]{1, 1}, 2);
                var direct = borrowed.backward(objective(batch.linear(directWeight), target),
                        java.util.List.of(directWeight), 16);
                assertEquals(25, direct.objective().bytes().getFloat());
                assertEquals(30, direct.gradients().getFirst().bytes().getFloat());
                model.configureTraining(target.descriptor(), FluentModelBuilderIntegrationTest::objective, new Sgd(.1));
                assertEquals(25, model.train(batch, target).objective().bytes().getFloat());
                assertEquals(-2, model.compute(values(false, new long[]{1, 1}, 2)).bytes().getFloat(), 1e-6);
                model.save(file);
            }
            assertFalse(borrowed.isClosed());
            try (var loaded = Model.load(file).withEngine(borrowed)) {
                assertEquals(-2, loaded.compute(values(false, new long[]{1, 1}, 2)).bytes().getFloat(), 1e-6);
            }
        }
        assertEquals(6, retained.bytes().getFloat());
        assertEquals(2, callerWeight.hostStorage().orElseThrow().segment()
                .get(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED, 0));
    }

    /**
     * Runs the supported CNN and stacked-LSTM examples only after their original Models close.
     * @throws Exception if cold persistence or supported execution fails
     */
    @Test void cnnAndRecurrentPublicExamplesSaveBeforeComputeCloseAndReload() throws Exception {
        Path cnnFile = directory.resolve("cnn"), recurrentFile = directory.resolve("recurrent");
        try (var cnn = Model.builder().input(DataType.FLOAT32, 1, 1, 4, 4).seed(42)
                .conv2d(2, 2, 2).relu().maxPool2d(2, 2).flatten().linear(1).build();
                var recurrent = Model.builder().input(DataType.FLOAT32, 2, 1, 1).seed(42)
                        .lstm(1).lstm(1).lastTimeStep().linear(1).build()) {
            cnn.save(cnnFile); recurrent.save(recurrentFile);
        }
        try (var cnn = Model.load(cnnFile); var recurrent = Model.load(recurrentFile)) {
            Tensor image = values(false, new long[]{1, 1, 4, 4}, 1, 2, 3, 4, 5, 6, 7, 8,
                    9, 10, 11, 12, 13, 14, 15, 16);
            Tensor series = values(false, new long[]{2, 1, 1}, .1f, -.2f);
            assertEquals(Shape.of(1, 1), cnn.compute(image).shape());
            var first = recurrent.compute(series);
            assertEquals(Shape.of(1, 1), first.shape());
            assertEquals(first.bytes().getFloat(), recurrent.compute(series).bytes().getFloat());
            recurrent.configureTraining(values(false, new long[]{1, 1}, .2f).descriptor(),
                    FluentModelBuilderIntegrationTest::objective, new Sgd(.01));
            var step = recurrent.train(series, values(false, new long[]{1, 1}, .2f));
            assertTrue(Float.isFinite(step.objective().bytes().getFloat()));
            assertTrue(Float.isFinite(recurrent.compute(series).bytes().getFloat()));
        }
        try (var files = Files.list(directory)) { assertEquals(2, files.count()); }
    }

    /**
     * Checks all three convolution dimensions against scalar arithmetic and supported SGD updates.
     * Conv3d persistence remains inference-only and its training request must reject.
     * @throws Exception if supported cold persistence or execution fails
     */
    @Test void allSmallConvolutionsColdRoundtripAndSupportedAdjointsMatchScalarSgdOracle() throws Exception {
        for (int dimension = 1; dimension <= 3; dimension++) {
            long[] shape = new long[dimension + 2]; java.util.Arrays.fill(shape, 3); shape[0] = 1; shape[1] = 1;
            var builder = Model.builder().input(DataType.FLOAT32, shape).seed(11);
            if (dimension == 1) builder.conv1d(1, 2);
            else if (dimension == 2) builder.conv2d(1, 2, 2);
            else builder.conv3d(1, 2, 2, 2);
            Path file = directory.resolve("convolution" + dimension);
            double[][] weights;
            try (var original = builder.build()) { weights = weights(original); original.save(file); }
            int dims = dimension;
            try (var loaded = Model.load(file)) {
                int count = (int) Math.pow(3, dimension);
                float[] batch = new float[count]; for (int j = 0; j < count; j++) batch[j] = (j - 2) * .07f;
                Tensor input = values(false, shape, batch);
                double[] expected = convolution(batch, weights, dimension);
                same(expected, loaded.compute(input));
                float[] changed = batch.clone(); for (int j = 0; j < count; j++) changed[j] *= -.5f;
                same(convolution(changed, weights, dimension), loaded.compute(values(false, shape, changed)));
                long[] targetShape = new long[dimension + 2]; java.util.Arrays.fill(targetShape, 2);
                targetShape[0] = 1; targetShape[1] = 1;
                float[] targets = new float[expected.length]; java.util.Arrays.fill(targets, .13f);
                Tensor target = values(false, targetShape, targets);
                if (dimension == 3) {
                    assertThrows(IllegalArgumentException.class, () -> loaded.configureTraining(target.descriptor(),
                            FluentModelBuilderIntegrationTest::objective, new Sgd(.01)));
                    same(expected, loaded.compute(input));
                } else {
                    loaded.configureTraining(target.descriptor(), FluentModelBuilderIntegrationTest::objective, new Sgd(.01));
                    step(loaded, input, target, weights, w -> convolution(batch, w, dims), targets);
                    same(convolution(batch, weights, dimension), loaded.compute(input));
                }
                loaded.save(file);
            }
            try (var trained = Model.load(file)) {
                var actual = weights(trained);
                for (int p = 0; p < weights.length; p++) assertArrayEquals(weights[p], actual[p], 2e-5);
            }
        }
    }

    /**
     * Checks static recurrence variants against scalar equations, fresh gradients and trained reloads.
     * @throws Exception if supported persistence or execution fails
     */
    @Test void allStaticFamiliesAllTimeLastAndTwoLayerStacksColdReloadMatchScalarEquationsAndSgd() throws Exception {
        for (String kind : List.of("rnn", "gru", "lstm")) for (int depth : new int[]{1, 2})
            for (boolean last : List.of(false, true)) {
                ModelBuilder builder = Model.builder().input(DataType.FLOAT32, 2, 1, 1).seed(37);
                for (int layer = 0; layer < depth; layer++) {
                    if (kind.equals("rnn")) builder.rnn(1);
                    else if (kind.equals("gru")) builder.gru(1);
                    else builder.lstm(1);
                }
                if (last) builder.lastTimeStep();
                Path file = directory.resolve("static-recurrent"); double[][] weights;
                try (var original = builder.build()) { weights = weights(original); original.save(file); }
                int layers = depth;
                try (var loaded = Model.load(file)) {
                    Tensor target = last ? values(false, new long[]{1, 1}, .13f)
                            : values(false, new long[]{2, 1, 1}, .13f, -.07f);
                    float[] targets = last ? new float[]{.13f} : new float[]{.13f, -.07f};
                    for (float[] lanes : List.of(new float[]{.2f, -.3f}, new float[]{-.1f, .4f})) {
                        Tensor batch = values(false, new long[]{2, 1, 1}, lanes);
                        same(recurrence(lanes, weights, kind, layers, last), loaded.compute(batch));
                    }
                    loaded.configureTraining(target.descriptor(), FluentModelBuilderIntegrationTest::objective, new Sgd(.01));
                    for (float[] lanes : List.of(new float[]{.2f, -.3f}, new float[]{-.1f, .4f})) {
                        Tensor batch = values(false, new long[]{2, 1, 1}, lanes);
                        step(loaded, batch, target, weights, w -> recurrence(lanes, w, kind, layers, last), targets);
                        same(recurrence(lanes, weights, kind, layers, last), loaded.compute(batch));
                    }
                    loaded.save(file);
                }
                try (var trained = Model.load(file)) {
                    float[] lanes = {.3f, -.2f};
                    same(recurrence(lanes, weights, kind, layers, last),
                            trained.compute(values(false, new long[]{2, 1, 1}, lanes)));
                    var actual = weights(trained);
                    for (int p = 0; p < weights.length; p++) assertArrayEquals(weights[p], actual[p], 2e-5);
                }
            }
    }

    /**
     * Snapshots each registered FLOAT32 parameter for an independent scalar oracle.
     * @param model live quiescent fixture Model with accessible canonical parameter storage
     * @return independent parameter-ordered arrays of exactly represented double values
     */
    private static double[][] weights(Model<?, ?> model) {
        return model.stateDictionary().entries().stream().map(entry -> {
            float[] lanes = entry.value().hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_FLOAT_UNALIGNED);
            double[] result = new double[lanes.length];
            for (int j = 0; j < lanes.length; j++) result[j] = lanes[j];
            return result;
        }).toArray(double[][]::new);
    }

    /**
     * Compares every output lane with an independently calculated scalar result.
     * @param expected row-major finite double reference values
     * @param actual detached FLOAT32 publication, never mutated
     */
    private static void same(double[] expected, HostTensorValue actual) {
        assertEquals(expected.length * 4, actual.bytes().remaining());
        for (int j = 0; j < expected.length; j++) assertEquals(expected[j], actual.bytes().getFloat(j * 4), 2e-5, "lane " + j);
    }

    /**
     * Checks one fresh SGD step using scalar double-precision finite differences, not Compiler adjoints.
     * @param model configured live fixture Model to train once
     * @param input current compatible batch, borrowed for the synchronous call
     * @param target compatible stored FLOAT32 target
     * @param w mutable independent oracle parameters, advanced to expected updated values
     * @param equation pure scalar forward equations over the supplied oracle parameter arrays
     * @param targets exact scalar target lanes used by the independent squared-error objective
     */
    private static void step(Model<Tensor, Tensor> model, Tensor input, Tensor target, double[][] w,
            java.util.function.Function<double[][], double[]> equation, float[] targets) {
        double loss = loss(equation.apply(w), targets);
        double[][] gradient = new double[w.length][];
        for (int p = 0; p < w.length; p++) {
            gradient[p] = new double[w[p].length];
            for (int j = 0; j < w[p].length; j++) {
                double old = w[p][j], epsilon = 1e-5;
                w[p][j] = old + epsilon; double plus = loss(equation.apply(w), targets);
                w[p][j] = old - epsilon; double minus = loss(equation.apply(w), targets);
                w[p][j] = old; gradient[p][j] = (plus - minus) / (2 * epsilon);
            }
        }
        assertEquals(loss, model.train(input, target).objective().bytes().getFloat(), 2e-5);
        var actual = weights(model);
        boolean changed = false;
        for (int p = 0; p < w.length; p++) for (int j = 0; j < w[p].length; j++) {
            changed |= Math.abs(gradient[p][j]) > 1e-7;
            w[p][j] = (float) (w[p][j] - .01 * gradient[p][j]);
            assertEquals(w[p][j], actual[p][j], 2e-5, "parameter " + p + " lane " + j);
        }
        assertTrue(changed);
    }

    /**
     * Evaluates scalar sum-of-squared-error without Tensor or Compiler operations.
     * @param predictions finite row-major reference outputs
     * @param targets same-length represented target lanes
     * @return scalar summed squared error
     */
    private static double loss(double[] predictions, float[] targets) {
        double result = 0;
        for (int j = 0; j < predictions.length; j++) result += Math.pow(predictions[j] - targets[j], 2);
        return result;
    }

    /**
     * Evaluates the fixture's one-channel unit-stride cross-correlation directly in scalar Java.
     * @param x flattened spatial input with extent three on every axis
     * @param w independent parameters: extent-two kernel followed by scalar bias
     * @param dimensions spatial rank one, two or three
     * @return row-major output with extent two on every spatial axis
     */
    static double[] convolution(float[] x, double[][] w, int dimensions) {
        int outputs = 1 << dimensions;
        double[] result = new double[outputs];
        for (int i = 0; i < outputs; i++) {
            result[i] = w[1][0];
            for (int k = 0; k < outputs; k++) {
                int index = 0;
                for (int axis = dimensions - 1; axis >= 0; axis--)
                    index = index * 3 + ((i >> axis) & 1) + ((k >> axis) & 1);
                result[i] += x[index] * w[0][k];
            }
        }
        return result;
    }

    /**
     * Supplies the independent scalar gate equation for bounded finite fixture arguments.
     * @param x finite gate preactivation
     * @return logistic gate value
     */
    private static double sigmoid(double x) { return 1 / (1 + StrictMath.exp(-x)); }

    /**
     * Evaluates two time steps with explicit zero states and the fixed cell equations in scalar Java.
     * @param lanes two one-feature, one-row input time steps
     * @param w independent input-weight/hidden-weight/bias arrays for each layer in order
     * @param kind rnn, gru or lstm fixture selector
     * @param depth one or two stacked one-hidden-unit layers
     * @param last whether to retain only the final time step
     * @return independent all-time or final-step hidden values
     */
    private static double[] recurrence(float[] lanes, double[][] w, String kind, int depth, boolean last) {
        double[] sequence = {lanes[0], lanes[1]};
        for (int layer = 0; layer < depth; layer++) {
            double h = 0, c = 0; double[] next = new double[2];
            double[] inputW = w[3 * layer], hiddenW = w[3 * layer + 1], bias = w[3 * layer + 2];
            for (int t = 0; t < 2; t++) {
                double[] px = new double[bias.length], ph = new double[bias.length];
                for (int g = 0; g < bias.length; g++) { px[g] = sequence[t] * inputW[g] + bias[g]; ph[g] = h * hiddenW[g]; }
                if (kind.equals("rnn")) h = StrictMath.tanh(px[0] + ph[0]);
                else if (kind.equals("gru")) {
                    double r = sigmoid(px[0] + ph[0]), z = sigmoid(px[1] + ph[1]);
                    double n = StrictMath.tanh(px[2] + r * ph[2]); h = n + z * (h - n);
                } else {
                    double i = sigmoid(px[0] + ph[0]), f = sigmoid(px[1] + ph[1]);
                    c = f * c + i * StrictMath.tanh(px[2] + ph[2]);
                    h = sigmoid(px[3] + ph[3]) * StrictMath.tanh(c);
                }
                next[t] = h;
            }
            sequence = next;
        }
        return last ? new double[]{sequence[1]} : sequence;
    }

    @Test void publishedFluentExamplesCompileAndTrainingExampleRuns() throws Exception {
        Path root = Path.of("").toAbsolutePath();
        while (root != null && !Files.exists(root.resolve("settings.gradle.kts"))) root = root.getParent();
        assertNotNull(root, "repository documentation must be available");
        var compiler = ToolProvider.getSystemJavaCompiler();
        assertNotNull(compiler, "documentation examples require a JDK");
        for (String[] example : new String[][] {
                {"docs/user-guide/running-models.md", "FluentModelExample"},
                {"docs/api/public-api.md", "FluentBuilderShapesExample"}}) {
            String markdown = Files.readString(root.resolve(example[0]));
            var blocks = Pattern.compile("(?s)```java\\n(.*?)\\n```").matcher(markdown);
            String selected = null;
            while (blocks.find()) {
                if (blocks.group(1).contains("public final class " + example[1] + " {")) {
                    assertNull(selected, "exactly one complete example class");
                    selected = blocks.group(1);
                }
            }
            assertNotNull(selected, example[1]);
            String code = selected;
            var unit = new SimpleJavaFileObject(URI.create("string:///" + example[1] + ".java"),
                    JavaFileObject.Kind.SOURCE) {
                @Override public CharSequence getCharContent(boolean ignoreEncodingErrors) { return code; }
            };
            var diagnostics = new DiagnosticCollector<JavaFileObject>();
            Path output = Files.createDirectory(directory.resolve(example[1]));
            try (var files = compiler.getStandardFileManager(diagnostics, null, null)) {
                boolean success = compiler.getTask(null, files, diagnostics,
                        List.of("-proc:none", "-classpath", System.getProperty("java.class.path"),
                                "-d", output.toString()), null, List.of(unit)).call();
                assertTrue(success, () -> diagnostics.getDiagnostics().toString());
            }
            if (example[1].equals("FluentModelExample")) {
                Path artifact = directory.resolve("documentation.graph");
                try (var loader = new URLClassLoader(new java.net.URL[]{output.toUri().toURL()},
                        getClass().getClassLoader())) {
                    loader.loadClass(example[1]).getMethod("main", String[].class)
                            .invoke(null, (Object) new String[]{artifact.toString()});
                }
                try (var loaded = Model.load(artifact)) {
                    assertEquals(-2, loaded.compute(values(false, new long[]{1, 1}, 2)).bytes().getFloat(), 1e-6);
                }
            }
        }
    }
}
