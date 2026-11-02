package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.PreparedExecution;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.io.File;
import java.lang.ref.Reference;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Retained public metadata must not keep fresh CPU invocation payloads reachable after close. */
final class ClosedPublicationHeapIntegrationTest {
    private static final int ELEMENTS = 1 << 18;
    private static final int RUNS = 128;
    @TempDir Path directory;

    @Test
    void retainedClosedFloat32PublicationsStayWithinBoundedChildHeap() throws Exception {
        runChild(DataType.FLOAT32);
    }

    @Test
    void retainedClosedFloat64PublicationsStayWithinBoundedChildHeap() throws Exception {
        runChild(DataType.FLOAT64);
    }

    private void runChild(DataType type) throws Exception {
        var classpath = new LinkedHashSet<String>();
        classpath.add(System.getProperty("java.class.path"));
        for (ClassLoader loader = getClass().getClassLoader(); loader != null; loader = loader.getParent()) {
            if (loader instanceof URLClassLoader urls) {
                for (var url : urls.getURLs()) classpath.add(Path.of(url.toURI()).toString());
            }
        }
        Path output = directory.resolve("closed-publications-" + type + ".log");
        var builder = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "-Xmx96m", "-cp", String.join(File.pathSeparator, classpath), getClass().getName(), type.name())
                .redirectErrorStream(true).redirectOutput(output.toFile());
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
        assertTrue(finished, () -> type + " retained-metadata child timed out:\n" + evidence);
        assertEquals(0, process.exitValue(), () -> type + " retained-metadata child failed:\n" + evidence);
        assertTrue(evidence.contains(type + " retained_closed_publications=128 numeric_and_metadata=PASS"),
                () -> evidence);
    }

    /**
     * Runs fresh public CPU values in an isolated heap while retaining only closed publications.
     * This checks heap-payload reachability under pressure, not native allocation or a GC deadline.
     *
     * @param args one FLOAT32 or FLOAT64 data-type name
     */
    public static void main(String[] args) {
        DataType type = DataType.valueOf(args[0]);
        assertTrue(type == DataType.FLOAT32 || type == DataType.FLOAT64);
        var metadata = new ArrayList<RunResult.Publication>();
        TensorDescriptor descriptor = descriptor(type);
        Tensor prototype = TensorFactory.create(descriptor);
        Tensor output = prototype.neg();
        try (Engine engine = Engine.standard();
                PreparedExecution prepared = engine.prepare(engine.compile(List.of(output)))) {
            for (int index = 0; index < RUNS; index++) {
                metadata.add(runAndClose(engine, prepared, descriptor));
            }
        }
        for (RunResult.Publication publication : metadata) {
            assertTrue(publication.isClosed());
            assertEquals(0, publication.index());
            assertEquals(output.id(), publication.tensorId());
            assertEquals(descriptor, publication.descriptor());
            assertEquals(RunResult.Role.FORWARD, publication.role());
            assertTrue(publication.derivativeOrder().isEmpty());
            assertTrue(publication.targetIndex().isEmpty());
        }
        System.out.println(type + " retained_closed_publications=" + metadata.size() + " numeric_and_metadata=PASS");
        Reference.reachabilityFence(metadata);
    }

    private static RunResult.Publication runAndClose(
            Engine engine, PreparedExecution prepared, TensorDescriptor descriptor) {
        Tensor input;
        if (descriptor.dataType() == DataType.FLOAT32) {
            float[] values = new float[ELEMENTS];
            for (int index = 0; index < values.length; index++) values[index] = (index & 1) == 0 ? 1 : -2;
            input = TensorFactory.fromFlatArray(descriptor, Optional.empty(), values);
        } else {
            double[] values = new double[ELEMENTS];
            for (int index = 0; index < values.length; index++) values[index] = (index & 1) == 0 ? 1 : -2;
            input = TensorFactory.fromFlatArray(descriptor, Optional.empty(), values);
        }
        try (RunResult result = engine.runValues(prepared, List.of(input))) {
            assertEquals(1, result.resultCount());
            RunResult.Publication publication = result.publications().getFirst();
            var bytes = result.materialize(publication, ELEMENTS * (long) descriptor.dataType().byteWidth()).bytes();
            for (int index = 0; index < ELEMENTS; index++) {
                double actual = descriptor.dataType() == DataType.FLOAT32 ? bytes.getFloat() : bytes.getDouble();
                if (actual != ((index & 1) == 0 ? -1 : 2)) {
                    throw new AssertionError("CPU negation changed at element " + index + ": " + actual);
                }
            }
            assertEquals(0, bytes.remaining());
            return publication;
        }
    }

    private static TensorDescriptor descriptor(DataType type) {
        Shape shape = Shape.of(ELEMENTS);
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }
}
