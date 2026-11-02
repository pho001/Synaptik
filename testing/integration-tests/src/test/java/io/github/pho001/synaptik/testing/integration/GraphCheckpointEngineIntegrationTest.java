package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.Scan;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.convolution.*;
import io.github.pho001.synaptik.nn.layers.Linear;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.api.io.TempDir;

/** Cold graph load joins the existing explicit Engine path, not a new Model or execution facade. */
final class GraphCheckpointEngineIntegrationTest {
    @TempDir Path directory;

    /**
     * Cold body copy outlives native state and both direct bindings and rebind use public Engine.
     * This bounded execution check promises neither low-type execution nor owned NN acquisition.
     * @param type independently supported CPU FLOAT32 or FLOAT64 scan dtype
     */
    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void generalScanCopyAndRebindMatchDirectEngineAfterNativeStateCloses(DataType type) {
        try (Engine engine = Engine.standard()) {
            for (int time : List.of(0, 3)) for (boolean reverse : List.of(false, true)) {
                GraphCheckpoint.Source copied;
                try (Arena arena = Arena.ofConfined()) {
                    Tensor sequence = TensorFactory.create(scanDescriptor(type, Shape.of(time, 2)));
                    Tensor initial = TensorFactory.create(scanDescriptor(type, Shape.of(2)));
                    Tensor heapWeight = scanValue(type, Shape.of(2), .5, -.25);
                    MemorySegment nativeBits = arena.allocate(2L * type.byteWidth(), type.byteWidth());
                    nativeBits.copyFrom(heapWeight.hostStorage().orElseThrow().segment());
                    Tensor weight = TensorFactory.create(heapWeight.descriptor(), Optional.empty(), Optional.of(
                            new MemorySegmentStorage(type, 2, nativeBits)));
                    var roots = scanRoots(initial, sequence, weight, reverse);
                    copied = GraphCheckpoint.copy(new GraphCheckpoint.Source(List.of(sequence, initial), roots, List.of(
                            new GraphCheckpoint.Leaf("weight", GraphCheckpoint.Role.PARAMETER, weight),
                            new GraphCheckpoint.Leaf("tied", GraphCheckpoint.Role.PARAMETER, weight))));
                }
                Tensor ownedWeight = copied.leaves().getFirst().tensor();
                assertSame(ownedWeight, copied.leaves().getLast().tensor());
                assertFalse(ownedWeight.hostStorage().orElseThrow().segment().isNative());
                assertTrue(copied.inputs().stream().allMatch(t -> t.hostStorage().isEmpty()));
                var compiledCopy = engine.compile(copied.outputs());
                try (var preparedCopy = engine.prepare(compiledCopy)) {
                    for (double sign : List.of(1.0, -1.0)) {
                        double[] values = new double[time * 2];
                        for (int i = 0; i < values.length; i++) values[i] = sign * (i - 2) * .125;
                        Tensor sequence = scanValue(type, Shape.of(time, 2), values);
                        Tensor initial = scanValue(type, Shape.of(2), -0.0, .125);
                        Tensor weight = scanValue(type, Shape.of(2), .5, -.25);
                        var directRoots = scanRoots(initial, sequence, weight, reverse);
                        var direct = engine.compile(directRoots);
                        List<HostTensorValue> expected;
                        var pool = List.of(initial, sequence, weight);
                        var bindings = direct.inputs().stream().map(i -> pool.stream()
                                .filter(t -> t.id().equals(i.tensorId())).findFirst().orElseThrow()).toList();
                        try (var p = engine.prepare(direct); var r = engine.run(p, bindings)) {
                            expected = r.publications().stream().map(publication -> r.materialize(publication, 1024)).toList();
                        }
                        // Compile the copied ordinary source once and substitute fresh values by its
                        // public compiled input order, not by a hidden Checkpoint execution path.
                        var copyBindings = compiledCopy.inputs().stream().map(i -> {
                            if (i.tensorId().equals(copied.inputs().get(0).id())) return sequence;
                            if (i.tensorId().equals(copied.inputs().get(1).id())) return initial;
                            assertEquals(ownedWeight.id(), i.tensorId()); return ownedWeight;
                        }).toList();
                        try (var r = engine.runValues(preparedCopy, copyBindings)) {
                            assertEquals(expected.size(), r.publications().size());
                            for (int i = 0; i < expected.size(); i++)
                                assertEquals(expected.get(i).bytes(), r.materialize(r.publications().get(i), 1024).bytes());
                        }
                        var rebound = GraphCheckpoint.rebind(copied, List.of(sequence, initial));
                        for (int i = 0; i < rebound.size(); i++)
                            assertEquals(expected.get(i).bytes(), engine.compute(rebound.get(i)).bytes());
                        if (time == 0) {
                            if (type == DataType.FLOAT32) assertEquals(0x80000000, expected.getFirst().bytes().getInt(0));
                            else assertEquals(Long.MIN_VALUE, expected.getFirst().bytes().getLong(0));
                            assertEquals(0, expected.get(1).bytes().remaining());
                            assertEquals(0, expected.get(2).bytes().remaining());
                        }
                    }
                }
            }
        }
    }

    /**
     * Composes an independent ordinary scan whose retained carry history is publicly requested.
     * @param initial canonical gradient-eligible two-lane initial carry
     * @param sequence matching time-major gradient-eligible sequence
     * @param weight matching two-lane invariant, borrowed without mutation
     * @param reverse original-time traversal flag
     * @return final carry, emission and pre-step carry residual in explicit root order
     */
    private static List<Tensor> scanRoots(Tensor initial, Tensor sequence, Tensor weight, boolean reverse) {
        Tensor h = TensorFactory.create(initial.descriptor());
        Tensor x = TensorFactory.create(initial.descriptor());
        Tensor w = TensorFactory.create(weight.descriptor());
        Tensor next = h.add(x.mul(w)).tanh();
        var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(next), List.of(next));
        var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(weight), body, reverse);
        int previousCarry = body.retainedValues().indexOf(h);
        return List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst(), scan.residuals().get(previousCarry));
    }

    /**
     * Supplies canonical metadata without implying support for any other dtype or layout.
     * @param type FLOAT32 or FLOAT64 logical dtype
     * @param shape fully static shape
     * @return exact canonical gradient-eligible descriptor
     */
    private static TensorDescriptor scanDescriptor(DataType type, Shape shape) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), true);
    }

    /**
     * Creates fresh heap values for the bounded typed Engine regression.
     * @param type FLOAT32 or FLOAT64
     * @param shape static shape matching the supplied lane count
     * @param values finite represented lanes, including signed zero, never mutated
     * @return fresh exact-descriptor stored Tensor
     */
    private static Tensor scanValue(DataType type, Shape shape, double... values) {
        if (type == DataType.FLOAT64)
            return TensorFactory.fromFlatArray(scanDescriptor(type, shape), Optional.empty(), values.clone());
        float[] lanes = new float[values.length];
        for (int i = 0; i < lanes.length; i++) lanes[i] = (float) values[i];
        return TensorFactory.fromFlatArray(scanDescriptor(type, shape), Optional.empty(), lanes);
    }

    @Test
    void linearLoadOutlivesSourceAndMatchesForwardAndOrdinaryAutogradForTwoFreshInputs() throws Exception {
        ArtifactLimits limits = new ArtifactLimits(100_000, 10_000, 10, 32, 10_000, 90_000);
        Path file = directory.resolve("linear.graph");
        try (Arena arena = Arena.ofConfined()) {
            Tensor weight = nativeValue(arena, Shape.of(2, 2), 1, -2, 3, 1);
            Tensor bias = nativeValue(arena, Shape.of(2), 0.5f, -0.5f);
            Tensor prototype = TensorFactory.create(descriptor(Shape.of(1, 2)));
            Tensor output = new Linear(weight, bias).forward(prototype);
            GraphCheckpoint.save(file, new GraphCheckpoint.Source(List.of(prototype), List.of(output), List.of(
                    new GraphCheckpoint.Leaf("weight", GraphCheckpoint.Role.PARAMETER, weight),
                    new GraphCheckpoint.Leaf("bias", GraphCheckpoint.Role.PARAMETER, bias))), limits);
        }
        var loaded = GraphCheckpoint.load(file, limits);
        Files.delete(file);
        assertTrue(loaded.inputs().getFirst().hostStorage().isEmpty());
        for (var leaf : loaded.leaves()) assertFalse(leaf.tensor().hostStorage().orElseThrow().segment().isNative());
        try (Engine engine = Engine.standard()) {
            for (float[] batch : List.of(new float[] {1, 2}, new float[] {-3, 4})) {
                Tensor input = heapValue(Shape.of(1, 2), batch);
                Tensor directWeight = heapValue(Shape.of(2, 2), 1, -2, 3, 1);
                Tensor directBias = heapValue(Shape.of(2), 0.5f, -0.5f);
                Tensor direct = new Linear(directWeight, directBias).forward(input);
                Tensor rebound = GraphCheckpoint.rebind(loaded, List.of(input)).getFirst();
                assertEquals(engine.compute(direct).bytes(), engine.compute(rebound).bytes());
                var expected = engine.backward(direct.sum(), List.of(input, directWeight, directBias), 1000);
                var actual = engine.backward(rebound.sum(), List.of(input,
                        loaded.leaves().get(0).tensor(), loaded.leaves().get(1).tensor()), 1000);
                assertEquals(expected.objective().bytes(), actual.objective().bytes());
                for (int i = 0; i < 3; i++) assertEquals(expected.gradients().get(i).bytes(), actual.gradients().get(i).bytes());
                assertEquals(batch[0] - 2 * batch[1] + 0.5f, engine.compute(rebound).bytes().getFloat(0));
            }
            GraphCheckpoint.save(file, loaded, limits);
            var second = GraphCheckpoint.load(file, limits);
            assertNotSame(loaded.leaves().getFirst().tensor(), second.leaves().getFirst().tensor());
        }
    }

    @ParameterizedTest
    @EnumSource(value = UnaryElementwiseKind.class, names = {"RELU", "TANH", "SIGMOID"})
    void restoredActivationMatchesIndependentOrdinaryForwardAndBackward(UnaryElementwiseKind kind) throws Exception {
        ArtifactLimits limits = new ArtifactLimits(100_000, 10_000, 10, 32, 10_000, 90_000);
        Path file = directory.resolve("activation.graph");
        try (Arena arena = Arena.ofConfined()) {
            Tensor weight = nativeValue(arena, Shape.of(2, 2), 0.5f, -0.25f, 0.25f, 0.5f);
            Tensor prototype = TensorFactory.create(descriptor(Shape.of(1, 2)));
            Tensor output = activate(new Linear(weight).forward(prototype), kind);
            GraphCheckpoint.save(file, new GraphCheckpoint.Source(List.of(prototype), List.of(output),
                    List.of(new GraphCheckpoint.Leaf("weight", GraphCheckpoint.Role.PARAMETER, weight))), limits);
        }
        var loaded = GraphCheckpoint.load(file, limits);
        try (Engine engine = Engine.standard()) {
            for (float[] batch : List.of(new float[] {1, 2}, new float[] {-0.5f, 1})) {
                Tensor input = heapValue(Shape.of(1, 2), batch);
                Tensor weight = heapValue(Shape.of(2, 2), 0.5f, -0.25f, 0.25f, 0.5f);
                Tensor direct = activate(new Linear(weight).forward(input), kind);
                Tensor rebuilt = GraphCheckpoint.rebind(loaded, List.of(input)).getFirst();
                assertEquals(kind, rebuilt.provenance().orElseThrow().operation().kind());
                var forward = engine.compute(direct).bytes();
                var actualForward = engine.compute(rebuilt).bytes();
                for (int lane = 0; lane < 2; lane++)
                    assertEquals(forward.getFloat(lane * 4), actualForward.getFloat(lane * 4), 1e-5);
                var expected = engine.backward(direct.sum(), List.of(input, weight), 1000);
                var actual = engine.backward(rebuilt.sum(), List.of(input, loaded.leaves().getFirst().tensor()), 1000);
                assertEquals(expected.objective().bytes().getFloat(), actual.objective().bytes().getFloat(), 1e-5);
                for (int p = 0; p < 2; p++) {
                    var a = actual.gradients().get(p).bytes();
                    var e = expected.gradients().get(p).bytes();
                    for (int lane = 0; lane < e.remaining() / 4; lane++)
                        assertEquals(e.getFloat(lane * 4), a.getFloat(lane * 4), 1e-5);
                }
            }
        }
    }

    /**
     * Checks raw CNN persistence after native source closure, then ordinary forward/adjoint ownership.
     * @throws Exception if supported persistence, I/O or execution fails
     */
    @Test void cnnRawSourceClosesNativeStateBeforeLoadAndRebindsFreshBatchesForForwardAndAdjoints() throws Exception {
        var limits = new ArtifactLimits(100_000, 10_000, 10, 32, 10_000, 90_000);
        for (int dimension : new int[]{1, 2, 3}) {
            Path file = directory.resolve("cnn" + dimension);
            long[] axes = new long[dimension + 2], kernelAxes = new long[dimension + 2];
            java.util.Arrays.fill(axes, 3); java.util.Arrays.fill(kernelAxes, 2);
            axes[0] = axes[1] = kernelAxes[0] = kernelAxes[1] = 1;
            float[] weights = new float[1 << dimension];
            for (int j = 0; j < weights.length; j++) weights[j] = (j + 1) * .1f;
            try (Arena arena = Arena.ofConfined()) {
                Tensor x = TensorFactory.create(descriptor(Shape.of(axes)));
                Tensor w = nativeValue(arena, Shape.of(kernelAxes), weights);
                Tensor b = nativeValue(arena, Shape.of(1), .2f);
                Tensor y = convolution(x, w, b, dimension);
                GraphCheckpoint.save(file, new GraphCheckpoint.Source(List.of(x), List.of(y), List.of(
                        new GraphCheckpoint.Leaf("weight", GraphCheckpoint.Role.PARAMETER, w),
                        new GraphCheckpoint.Leaf("bias", GraphCheckpoint.Role.PARAMETER, b))), limits);
            }
            var loaded = GraphCheckpoint.load(file, limits);
            assertTrue(loaded.inputs().getFirst().hostStorage().isEmpty());
            try (Engine engine = Engine.standard()) {
                for (float sign : new float[]{1, -1}) {
                    float[] lanes = new float[(int) Math.pow(3, dimension)];
                    for (int j = 0; j < lanes.length; j++) lanes[j] = sign * (j - 2) * .07f;
                    Tensor x = heapValue(Shape.of(axes), lanes);
                    Tensor output = GraphCheckpoint.rebind(loaded, List.of(x)).getFirst();
                    double[] expected = FluentModelBuilderIntegrationTest.convolution(lanes,
                            new double[][]{toDouble(weights), new double[]{.2f}}, dimension);
                    var actual = engine.compute(output).bytes();
                    for (int j = 0; j < expected.length; j++) assertEquals(expected[j], actual.getFloat(j * 4), 2e-5);
                    Tensor w = heapValue(Shape.of(kernelAxes), weights), b = heapValue(Shape.of(1), .2f);
                    var direct = engine.backward(convolution(x, w, b, dimension).sum(), List.of(w, b), 1000);
                    var restored = engine.backward(output.sum(), loaded.leaves().stream().map(GraphCheckpoint.Leaf::tensor).toList(), 1000);
                    for (int p = 0; p < 2; p++) assertEquals(direct.gradients().get(p).bytes(), restored.gradients().get(p).bytes());
                }
            }
        }
    }

    /**
     * Builds an independent ordinary default convolution, without cold source reconstruction.
     * @param input compatible canonical floating spatial input
     * @param weight compatible gradient-eligible kernel
     * @param bias matching output-channel bias
     * @param dimension spatial rank one, two or three
     * @return fresh ordinary convolution expression, including the standard Conv1d composition
     */
    private static Tensor convolution(Tensor input, Tensor weight, Tensor bias, int dimension) {
        return dimension == 1 ? input.conv1d(weight, bias, Conv1dAttrs.defaults())
                : dimension == 2 ? input.conv2d(weight, bias, Conv2dAttrs.defaults())
                : input.conv3d(weight, bias, Conv3dAttrs.defaults());
    }

    /**
     * Widens represented test parameters exactly for the shared scalar oracle.
     * @param values finite FLOAT32 parameter lanes, never mutated
     * @return independent same-length double array
     */
    private static double[] toDouble(float[] values) {
        double[] result = new double[values.length];
        for (int j = 0; j < values.length; j++) result[j] = values[j];
        return result;
    }

    /**
     * @param input ordinary floating source expression
     * @param kind selected activation, independent of cold reconstruction
     * @return ordinary Model activation expression
     */
    private static Tensor activate(Tensor input, UnaryElementwiseKind kind) {
        return switch (kind) {
            case RELU -> input.relu(); case TANH -> input.tanh(); case SIGMOID -> input.sigmoid();
            default -> throw new AssertionError(kind);
        };
    }

    /**
     * @param shape static logical geometry
     * @return exact gradient-eligible canonical FLOAT32 descriptor
     */
    private static TensorDescriptor descriptor(Shape shape) {
        return new TensorDescriptor(DataType.FLOAT32, shape, Optional.of(LayoutDescriptor.contiguous(shape)), true);
    }

    /**
     * @param shape static geometry matching values
     * @param values logical row-major FLOAT32 lanes
     * @return ordinary mutable heap-backed test input/state
     */
    private static Tensor heapValue(Shape shape, float... values) {
        return TensorFactory.create(descriptor(shape), Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.FLOAT32, values.length, MemorySegment.ofArray(values))));
    }

    /**
     * @param arena caller-owned source lifetime, closed before load
     * @param shape static geometry matching values
     * @param values logical row-major FLOAT32 lanes
     * @return borrowed native test state
     */
    private static Tensor nativeValue(Arena arena, Shape shape, float... values) {
        MemorySegment storage = arena.allocate(values.length * 4L, 4);
        storage.copyFrom(MemorySegment.ofArray(values));
        return TensorFactory.create(descriptor(shape), Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.FLOAT32, values.length, storage)));
    }
}
