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

    @Test void cnnAndRecurrentPublicExamplesComputeWithoutSerializationSupport() throws Exception {
        try (var cnn = Model.builder().input(DataType.FLOAT32, 1, 1, 4, 4).seed(42)
                .conv2d(2, 2, 2).relu().maxPool2d(2, 2).flatten().linear(1).build();
                var recurrent = Model.builder().input(DataType.FLOAT32, 2, 1, 1).seed(42)
                        .lstm(1).lstm(1).lastTimeStep().linear(1).build()) {
            Tensor image = values(false, new long[]{1, 1, 4, 4}, 1, 2, 3, 4, 5, 6, 7, 8,
                    9, 10, 11, 12, 13, 14, 15, 16);
            Tensor series = values(false, new long[]{2, 1, 1}, .1f, -.2f);
            for (var model : java.util.List.of(cnn, recurrent)) {
                Path missing = directory.resolve("unsupported");
                assertThrows(IllegalArgumentException.class, () -> model.save(missing));
                assertFalse(Files.exists(missing));
            }
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
        try (var files = Files.list(directory)) { assertEquals(0, files.count()); }
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
