package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.io.File;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Scalar consumers must not inherit the test worker's optional Vector API module. */
final class PlainJvmCpuIntegrationTest {
    @TempDir Path directory;

    @Test void defaultCpuComputesAndDifferentiatesWithoutTheIncubatorModule() throws Exception {
        var classpath = new LinkedHashSet<String>();
        classpath.add(System.getProperty("java.class.path"));
        for (ClassLoader loader = getClass().getClassLoader(); loader != null; loader = loader.getParent()) {
            if (loader instanceof URLClassLoader urls) {
                for (var url : urls.getURLs()) classpath.add(Path.of(url.toURI()).toString());
            }
        }
        Path output = directory.resolve("plain-jvm.log");
        var builder = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "-Xmx256m", "-cp", String.join(File.pathSeparator, classpath), getClass().getName())
                .redirectErrorStream(true).redirectOutput(output.toFile());
        // Inherited launcher options would hide the missing-module consumer boundary.
        builder.environment().remove("JDK_JAVA_OPTIONS");
        builder.environment().remove("JAVA_TOOL_OPTIONS");
        builder.environment().remove("_JAVA_OPTIONS");
        var process = builder.start();
        boolean finished = process.waitFor(90, TimeUnit.SECONDS);
        if (!finished) {
            process.destroyForcibly();
            process.waitFor();
        }
        String evidence = Files.readString(output);
        assertTrue(finished, () -> "Plain JVM consumer timed out:\n" + evidence);
        assertEquals(0, process.exitValue(), () -> "Plain JVM consumer failed:\n" + evidence);
    }

    public static void main(String[] args) {
        assertFalse(ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent());
        try (Engine engine = Engine.standard()) {
            for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
                Tensor input = matrix(type, false, 1, 2, -1, .5);
                Tensor weights = matrix(type, true, .5, -.25, .75, .125);
                Tensor prediction = input.matmul(weights).relu();
                double tolerance = type == DataType.FLOAT64 ? 1e-12 : 1e-6;
                assertArrayEquals(new double[] {2, 0, 0, .3125}, numbers(engine.compute(prediction)), tolerance);
                var result = engine.backward(prediction.mul(prediction).sum(), List.of(weights), 1024);
                // Independent scalar matrix/chain-rule expectations; no alternate Engine oracle.
                assertArrayEquals(new double[] {4.09765625}, numbers(result.objective()), tolerance);
                assertArrayEquals(new double[] {4, -.625, 8, .3125},
                        numbers(result.gradients().getFirst()), tolerance);
                System.out.println(type + " plain JVM forward and gradients qualified");
            }
        }
    }

    private static Tensor matrix(DataType type, boolean gradient, double... values) {
        Shape shape = Shape.of(2, 2);
        var descriptor = new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), gradient);
        if (type == DataType.FLOAT64) return TensorFactory.fromFlatArray(descriptor, Optional.empty(), values);
        float[] floats = new float[values.length];
        for (int i = 0; i < floats.length; i++) floats[i] = (float) values[i];
        return TensorFactory.fromFlatArray(descriptor, Optional.empty(), floats);
    }

    private static double[] numbers(HostTensorValue value) {
        var bytes = value.bytes();
        double[] values = new double[bytes.remaining() / value.dataType().byteWidth()];
        for (int i = 0; i < values.length; i++) {
            values[i] = value.dataType() == DataType.FLOAT64 ? bytes.getDouble() : bytes.getFloat();
        }
        return values;
    }
}
