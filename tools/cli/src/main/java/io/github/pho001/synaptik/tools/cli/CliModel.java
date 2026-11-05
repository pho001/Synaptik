package io.github.pho001.synaptik.tools.cli;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Model;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.Optional;

/** Public source-descriptor admission and one-read, private-state Model acquisition. */
final class CliModel {
    static final ArtifactLimits LIMITS = new ArtifactLimits(
            268435456L, 4194304, 16384, 64, 67108864, 268435456L);

    private CliModel() { }

    /**
     * Loads once; the ordinary heap Source has no close protocol and no Engine dependency.
     * @param path graph checkpoint, not a legacy flexible or optimizer checkpoint
     * @return exact decoded source reused for inspection and private acquisition
     * @throws IOException if decoding or file admission fails
     */
    static GraphCheckpoint.Source load(Path path) throws IOException {
        return GraphCheckpoint.load(path, LIMITS);
    }

    /** Retains complete input descriptors, including layout presence and requiresGrad. */
    record Signature(TensorDescriptor input, TensorDescriptor prediction, TensorDescriptor target,
                     int batch, int features, int outputs) { }

    static Signature signature(GraphCheckpoint.Source source) {
        if (source.inputs().size() != 1 || source.outputs().size() != 1) {
            throw new IllegalArgumentException("CLI requires exactly one declared input and one output");
        }
        return signature(source.inputs().getFirst().descriptor(), source.outputs().getFirst().descriptor());
    }

    static Signature signature(TensorDescriptor input, TensorDescriptor prediction) {
        long[] in = matrix(input, "input");
        long[] out = matrix(prediction, "output");
        if (in[0] != out[0]) throw new IllegalArgumentException("input/output static batch sizes differ");
        validateInputLayout(input);
        return new Signature(input, prediction, f32(false, out),
                Math.toIntExact(in[0]), Math.toIntExact(in[1]), Math.toIntExact(out[1]));
    }

    private static long[] matrix(TensorDescriptor descriptor, String name) {
        if (descriptor.dataType() != DataType.FLOAT32 || descriptor.shape().rank() != 2
                || !descriptor.shape().isFullyStatic()) {
            throw new IllegalArgumentException(name + " must have a static rank-two FLOAT32 descriptor");
        }
        long[] axes = descriptor.shape().toLongArray();
        for (long axis : axes) {
            if (axis <= 0 || axis > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(name + " dimensions must be in 1.." + Integer.MAX_VALUE);
            }
        }
        Math.multiplyExact(Math.multiplyExact(axes[0], axes[1]), 4L);
        return axes;
    }

    private static void validateInputLayout(TensorDescriptor descriptor) {
        if (descriptor.layout().isPresent()
                && !descriptor.layout().orElseThrow().equals(LayoutDescriptor.contiguous(descriptor.shape()))) {
            throw new IllegalArgumentException("input layout must be absent or canonical contiguous, offset-zero and non-view");
        }
    }

    static TensorDescriptor f32(boolean gradient, long... axes) {
        Shape shape = Shape.of(axes);
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), gradient);
    }

    /**
     * Acquires private state from the same decoded source, using the documented cold copy.
     * @param source decoded graph whose descriptors were already admitted
     * @param engine borrowed caller-owned Engine, closed after the returned acquisition
     * @return owned Model/signature pair; caller must close the acquisition
     */
    static Acquired acquire(GraphCheckpoint.Source source, Engine engine) {
        Signature signature = signature(source);
        return new Acquired(Model.graph(source.inputs(), source.outputs(), source.leaves(), engine), signature);
    }

    /**
     * Builds a privately owned biased Linear graph and derives its actual output descriptor by cold forward.
     * @param batch positive static example count
     * @param features positive per-example input width
     * @param outputs positive regression target width
     * @param seed deterministic initializer seed
     * @param engine borrowed live caller-owned Engine, closed after the returned acquisition
     * @return owned Model/signature pair; caller must close it before the borrowed Engine
     */
    static Acquired create(int batch, int features, int outputs, long seed, Engine engine) {
        var model = Model.builder().input(DataType.FLOAT32, batch, features).seed(seed)
                .linear(outputs, true).build();
        try {
            model.withEngine(engine);
            TensorDescriptor input = f32(false, batch, features);
            Signature signature = signature(input, model.forward(TensorFactory.create(input)).descriptor());
            return new Acquired(model, signature);
        } catch (RuntimeException | Error failure) {
            try {
                model.close();
            } catch (RuntimeException | Error cleanup) {
                if (cleanup != failure) failure.addSuppressed(cleanup);
            }
            throw failure;
        }
    }

    /** One Model owner paired with cold public boundary metadata, not private execution state. */
    record Acquired(Model<?, ?> model, Signature signature) implements AutoCloseable {
        @Override
        public void close() { model.close(); }
    }

    /**
     * Prints raw public source metadata, with no signature narrowing or execution.
     * @param source decoded source including all public leaves
     * @param output borrowed destination, never closed
     */
    static void inspect(GraphCheckpoint.Source source, PrintStream output) {
        for (int i = 0; i < source.inputs().size(); i++) {
            output.println("input[" + i + "] " + source.inputs().get(i).descriptor());
        }
        for (int i = 0; i < source.outputs().size(); i++) {
            output.println("output[" + i + "] " + source.outputs().get(i).descriptor());
        }
        for (var leaf : source.leaves()) {
            var storage = leaf.tensor().hostStorage();
            output.println("leaf " + leaf.name() + " role=" + leaf.role()
                    + " identity=" + leaf.tensor().id() + " " + leaf.tensor().descriptor()
                    + " storedBytes=" + (storage.isPresent() ? storage.orElseThrow().byteSize() : "absent"));
        }
        output.println("Graph and current values only; no optimizer state or dataset cursor.");
    }
}
